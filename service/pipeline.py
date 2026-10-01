"""
Conflict-aware answering. Two detectors (service/detector.py):

agreement  closed-book and context-grounded answers are generated concurrently
           and compared.
verify     the context-grounded answer is generated, then the model scores it
           with a one-token Yes/No check.

Routing:
    consistent / no_prior / verified -> answered
    conflict / rejected              -> flagged (answer returned with a warning)
    context_refused                  -> abstained
"""
import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Awaitable, Optional, TypeVar

from .cache import TTLCache
from .detector import Detection, Verdict, detect, detect_verified, is_refusal, log_p_yes, normalize
from .llm import LLMClient
from .prompts import closed_book_prompt, rag_prompt, verify_prompt


class Route(str, Enum):
    ANSWERED = "answered"
    FLAGGED = "flagged"
    ABSTAINED = "abstained"


ROUTES = {
    Verdict.CONSISTENT: Route.ANSWERED,
    Verdict.NO_PRIOR: Route.ANSWERED,
    Verdict.VERIFIED: Route.ANSWERED,
    Verdict.CONFLICT: Route.FLAGGED,
    Verdict.REJECTED: Route.FLAGGED,
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
    def __init__(self, llm: LLMClient, cache: TTLCache, detector: str = "verify",
                 verify_threshold: float = -9.245):
        if detector not in ("verify", "agreement"):
            raise ValueError(f"unknown detector {detector!r}")
        self.llm = llm
        self.cache = cache
        self.detector = detector
        self.verify_threshold = verify_threshold

    async def closed_book(self, question: str) -> tuple[str, bool]:
        key = "cb:" + normalize(question)
        cached = self.cache.get(key)
        if cached is not None:
            return cached, True
        answer = await self.llm.generate(closed_book_prompt(question))
        self.cache.set(key, answer)
        return answer, False

    async def verify(self, question: str, answer: str) -> tuple[float, bool]:
        key = f"vf:{normalize(question)}|{normalize(answer)}"
        cached = self.cache.get(key)
        if cached is not None:
            return cached, True
        score = log_p_yes(await self.llm.next_token_logprobs(verify_prompt(question, answer)))
        self.cache.set(key, score)
        return score, False

    async def answer_plain(self, question: str, contexts: list[str]) -> AnswerResult:
        """Standard RAG, no conflict check (baseline for latency comparisons)."""
        start = time.perf_counter()
        ctx_answer, ctx_ms = await _timed(self.llm.generate(rag_prompt(question, contexts)))
        return AnswerResult(
            answer=ctx_answer, route=Route.ANSWERED, context_answer=ctx_answer,
            timings_ms={"context": ctx_ms, "total": (time.perf_counter() - start) * 1000},
        )

    async def answer(self, question: str, contexts: list[str]) -> AnswerResult:
        if self.detector == "agreement":
            return await self._answer_agreement(question, contexts)
        return await self._answer_verify(question, contexts)

    async def _answer_verify(self, question: str, contexts: list[str]) -> AnswerResult:
        start = time.perf_counter()
        ctx_answer, ctx_ms = await _timed(self.llm.generate(rag_prompt(question, contexts)))
        timings = {"context": ctx_ms}
        score, cache_hit = None, False
        if not is_refusal(ctx_answer):
            (score, cache_hit), timings["verify"] = await _timed(self.verify(question, ctx_answer))
        detection = detect_verified(ctx_answer, score, self.verify_threshold)
        route = ROUTES[detection.verdict]
        timings["total"] = (time.perf_counter() - start) * 1000
        return AnswerResult(
            answer=None if route is Route.ABSTAINED else ctx_answer,
            route=route, context_answer=ctx_answer, detection=detection,
            cache_hit=cache_hit, timings_ms=timings,
        )

    async def _answer_agreement(self, question: str, contexts: list[str]) -> AnswerResult:
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
