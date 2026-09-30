from typing import Protocol

import httpx


class LLMClient(Protocol):
    async def generate(self, prompt: str) -> str: ...


class OllamaClient:
    """Async client for Ollama's /api/generate, deterministic decoding."""

    def __init__(self, base_url: str, model: str, timeout_s: float, max_tokens: int):
        self.model = model
        self._max_tokens = max_tokens
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout_s)

    async def generate(self, prompt: str) -> str:
        resp = await self._client.post("/api/generate", json={
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": self._max_tokens},
        })
        resp.raise_for_status()
        return resp.json().get("response", "").strip()

    async def aclose(self) -> None:
        await self._client.aclose()
