"""
FastAPI entry point.

    uvicorn service.app:app --port 8000
    curl -X POST localhost:8000/v1/answer -H 'Content-Type: application/json' \
         -d '{"question": "Who directed The Mask?", "contexts": ["The Mask was directed by Chuck Russell."]}'
"""
from contextlib import asynccontextmanager
from typing import Literal, Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .cache import TTLCache
from .config import settings
from .llm import LLMClient, OllamaClient
from .pipeline import ConflictAwareRAG, Route


class AnswerRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    contexts: list[str] = Field(min_length=1, max_length=20)
    mode: Literal["guarded", "plain"] = "guarded"


class DetectionOut(BaseModel):
    verdict: str
    overlap: float = 0.0
    log_p_yes: Optional[float] = None


class AnswerResponse(BaseModel):
    answer: Optional[str]
    route: Route
    context_answer: str
    closed_book_answer: Optional[str] = None
    detection: Optional[DetectionOut] = None
    cache_hit: bool = False
    timings_ms: dict[str, float]


def create_app(llm: Optional[LLMClient] = None, detector: Optional[str] = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        client = llm or OllamaClient(settings.ollama_url, settings.model,
                                     settings.llm_timeout_s, settings.max_tokens)
        app.state.rag = ConflictAwareRAG(
            client, TTLCache(settings.cache_size, settings.cache_ttl_s),
            detector=detector or settings.detector, verify_threshold=settings.verify_log_p_yes_threshold,
        )
        yield
        if llm is None:
            await client.aclose()

    app = FastAPI(title="Conflict-aware RAG", lifespan=lifespan)

    @app.get("/healthz")
    async def healthz():
        cache = app.state.rag.cache
        return {"status": "ok", "detector": app.state.rag.detector, "cache_entries": len(cache),
                "cache_hits": cache.hits, "cache_misses": cache.misses}

    @app.post("/v1/answer", response_model=AnswerResponse)
    async def answer(req: AnswerRequest):
        rag: ConflictAwareRAG = app.state.rag
        try:
            if req.mode == "plain":
                result = await rag.answer_plain(req.question, req.contexts)
            else:
                result = await rag.answer(req.question, req.contexts)
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"LLM backend error: {exc}") from exc
        det = result.detection
        detection = (DetectionOut(verdict=det.verdict.value, overlap=det.overlap, log_p_yes=det.log_p_yes)
                     if det else None)
        return AnswerResponse(
            answer=result.answer, route=result.route, context_answer=result.context_answer,
            closed_book_answer=result.closed_book_answer, detection=detection,
            cache_hit=result.cache_hit, timings_ms=result.timings_ms,
        )

    return app


app = create_app()
