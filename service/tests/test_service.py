import asyncio
import time

import httpx
import pytest

from service.app import create_app
from service.cache import TTLCache
from service.detector import Verdict, detect
from service.pipeline import ConflictAwareRAG, Route


class FakeLLM:
    """Closed-book prompts get `closed`, RAG prompts get `context`; each call sleeps `delay_s`."""

    def __init__(self, closed="Chuck Russell", context="Chuck Russell", delay_s=0.0):
        self.closed, self.context, self.delay_s = closed, context, delay_s
        self.calls = 0

    async def generate(self, prompt: str) -> str:
        self.calls += 1
        await asyncio.sleep(self.delay_s)
        return self.context if "Context:" in prompt else self.closed


def make_rag(**kw):
    return ConflictAwareRAG(FakeLLM(**kw), TTLCache(max_size=100, ttl_s=60))


# --- detector ---------------------------------------------------------------

@pytest.mark.parametrize("closed, ctx, verdict", [
    ("Chuck Russell", "Chuck Russell.", Verdict.CONSISTENT),
    ("Russell", "Chuck Russell", Verdict.CONSISTENT),
    ("Chuck Russell", "Colin Nutley.", Verdict.CONFLICT),
    ("I don't know", "Colin Nutley.", Verdict.NO_PRIOR),
    ("Chuck Russell", "I don't know.", Verdict.CONTEXT_REFUSED),
])
def test_detect(closed, ctx, verdict):
    assert detect(closed, ctx).verdict is verdict


# --- pipeline -----------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("closed, ctx, route, answer", [
    ("Chuck Russell", "Chuck Russell", Route.ANSWERED, "Chuck Russell"),
    ("I don't know", "Colin Nutley", Route.ANSWERED, "Colin Nutley"),
    ("Chuck Russell", "Colin Nutley", Route.FLAGGED, "Colin Nutley"),
    ("Chuck Russell", "I don't know", Route.ABSTAINED, None),
])
async def test_routing(closed, ctx, route, answer):
    result = await make_rag(closed=closed, context=ctx).answer("Who directed The Mask?", ["..."])
    assert result.route is route
    assert result.answer == answer


@pytest.mark.asyncio
async def test_closed_book_and_context_run_concurrently():
    rag = make_rag(delay_s=0.2)
    start = time.perf_counter()
    await rag.answer("Who directed The Mask?", ["..."])
    assert time.perf_counter() - start < 0.35   # sequential would be >= 0.4


@pytest.mark.asyncio
async def test_closed_book_answer_is_cached():
    rag = make_rag()
    first = await rag.answer("Who directed The Mask?", ["..."])
    second = await rag.answer("who directed the mask", ["..."])
    assert (first.cache_hit, second.cache_hit) == (False, True)
    assert rag.llm.calls == 3   # 2 context calls + 1 closed-book call


def test_cache_expiry_and_eviction():
    now = [0.0]
    cache = TTLCache(max_size=2, ttl_s=10, clock=lambda: now[0])
    cache.set("a", "1"); cache.set("b", "2"); cache.set("c", "3")
    assert cache.get("a") is None and cache.get("c") == "3"
    now[0] = 11
    assert cache.get("c") is None


# --- API ------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_api_flags_conflict():
    app = create_app(llm=FakeLLM(closed="Chuck Russell", context="Colin Nutley"))
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/v1/answer", json={
                "question": "Who directed The Mask?",
                "contexts": ["The Mask was directed by Colin Nutley."],
            })
            plain = await client.post("/v1/answer", json={
                "question": "Who directed The Mask?", "contexts": ["..."], "mode": "plain"})
            bad = await client.post("/v1/answer", json={"question": "x", "contexts": []})
    body = resp.json()
    assert resp.status_code == 200
    assert body["route"] == "flagged"
    assert body["detection"]["verdict"] == "conflict"
    assert body["closed_book_answer"] == "Chuck Russell"
    assert plain.json()["route"] == "answered" and plain.json()["detection"] is None
    assert bad.status_code == 422
