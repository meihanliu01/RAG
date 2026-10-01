import asyncio
import math
import time

import httpx
import pytest

from service.app import create_app
from service.cache import TTLCache
from service.detector import Verdict, detect, log_p_yes
from service.pipeline import ConflictAwareRAG, Route


class FakeLLM:
    """Closed-book prompts get `closed`, RAG prompts get `context`, verify prompts
    get P(Yes) = `p_yes`; each call sleeps `delay_s`."""

    def __init__(self, closed="Chuck Russell", context="Chuck Russell", p_yes=0.5, delay_s=0.0):
        self.closed, self.context, self.p_yes, self.delay_s = closed, context, p_yes, delay_s
        self.calls = 0

    async def generate(self, prompt: str) -> str:
        self.calls += 1
        await asyncio.sleep(self.delay_s)
        return self.context if "Context:" in prompt else self.closed

    async def next_token_logprobs(self, prompt: str, top_k: int = 10) -> dict[str, float]:
        self.calls += 1
        await asyncio.sleep(self.delay_s)
        return {"Yes": math.log(self.p_yes), "No": math.log(1 - self.p_yes)}


def make_rag(detector="agreement", **kw):
    return ConflictAwareRAG(FakeLLM(**kw), TTLCache(max_size=100, ttl_s=60),
                            detector=detector, verify_threshold=math.log(1e-4))


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


def test_log_p_yes_sums_variants_and_renormalizes():
    top = {"Yes": math.log(0.1), " yes": math.log(0.1), "No": math.log(0.6), "Maybe": math.log(0.2)}
    assert log_p_yes(top) == pytest.approx(math.log(0.25))
    assert log_p_yes({"Hmm": -1.0}) == pytest.approx(math.log(0.5))


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
@pytest.mark.parametrize("ctx, p_yes, route, answer", [
    ("Colin Nutley", 1e-6, Route.FLAGGED, "Colin Nutley"),
    ("Chuck Russell", 0.3, Route.ANSWERED, "Chuck Russell"),
    ("I don't know", 0.9, Route.ABSTAINED, None),
])
async def test_verify_routing(ctx, p_yes, route, answer):
    rag = make_rag(detector="verify", context=ctx, p_yes=p_yes)
    result = await rag.answer("Who directed The Mask?", ["..."])
    assert result.route is route
    assert result.answer == answer
    assert rag.llm.calls == (1 if route is Route.ABSTAINED else 2)   # no verify call on refusal


@pytest.mark.asyncio
async def test_verify_result_is_cached():
    rag = make_rag(detector="verify", context="Colin Nutley", p_yes=1e-6)
    first = await rag.answer("Who directed The Mask?", ["..."])
    second = await rag.answer("Who directed the Mask", ["..."])
    assert (first.cache_hit, second.cache_hit) == (False, True)
    assert rag.llm.calls == 3


def test_unknown_detector_rejected():
    with pytest.raises(ValueError):
        make_rag(detector="nope")


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
    app = create_app(llm=FakeLLM(closed="Chuck Russell", context="Colin Nutley"), detector="agreement")
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
