"""
Conflict-aware answering: the closed-book answer and the context-grounded answer
are generated concurrently, compared by the detector, and routed.

    consistent / no_prior  -> answered
    conflict               -> answered, flagged, both answers returned
    context_refused        -> abstained
"""
import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Awaitable, Optional, TypeVar

from .cache import TTLCache
from .detector import Detection, Verdict, detect, normalize
from .llm import LLMClient
from .prompts import closed_book_prompt, rag_prompt


class Route(str, Enum):
    ANSWERED = "answered"
    FLAGGED = "flagged"
    ABSTAINED = "abstained"


ROUTES = {
    Verdict.CONSISTENT: Route.ANSWERED,
    Verdict.NO_PRIOR: Route.ANSWERED,
    Verdict.CONFLICT: Route.FLAGGED,
    Verdict.CONTEXT_REFUSED: Route.ABSTAINED,
}


@dataclass
class AnswerResult:
    answer: Optional[str]
    route: Route
    context_answer: str
    closed_book_answer: Optional[str] = None
    detection: Optional[Detection] = None
    cache_hit: bool = False
    timings_ms: dict[str, float] = field(default_factory=dict)


T = TypeVar("T")


async def _timed(coro: Awaitable[T]) -> tuple[T, float]:
    start = time.perf_counter()
    out = await coro
    return out, (time.perf_counter() - start) * 1000


class ConflictAwareRAG:
    def __init__(self, llm: LLMClient, cache: TTLCache):
        self.llm = llm
        self.cache = cache

    async def closed_book(self, question: str) -> tuple[str, bool]:
        key = normalize(question)
        cached = self.cache.get(key)
        if cached is not None:
            return cached, True
        answer = await self.llm.generate(closed_book_prompt(question))
        self.cache.set(key, answer)
        return answer, False

    async def answer_plain(self, question: str, contexts: list[str]) -> AnswerResult:
        """Standard RAG, no conflict check (baseline for latency comparisons)."""
        start = time.perf_counter()
        ctx_answer, ctx_ms = await _timed(self.llm.generate(rag_prompt(question, contexts)))
        return AnswerResult(
            answer=ctx_answer, route=Route.ANSWERED, context_answer=ctx_answer,
            timings_ms={"context": ctx_ms, "total": (time.perf_counter() - start) * 1000},
        )

    async def answer(self, question: str, contexts: list[str]) -> AnswerResult:
        start = time.perf_counter()
        (closed, cb_ms), (ctx_answer, ctx_ms) = await asyncio.gather(
            _timed(self.closed_book(question)),
            _timed(self.llm.generate(rag_prompt(question, contexts))),
        )
        closed_answer, cache_hit = closed
        detection = detect(closed_answer, ctx_answer)
        route = ROUTES[detection.verdict]
        return AnswerResult(
            answer=None if route is Route.ABSTAINED else ctx_answer,
            route=route,
            context_answer=ctx_answer,
            closed_book_answer=closed_answer,
            detection=detection,
            cache_hit=cache_hit,
            timings_ms={"closed_book": cb_ms, "context": ctx_ms,
                        "total": (time.perf_counter() - start) * 1000},
        )
