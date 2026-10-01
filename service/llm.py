from typing import Protocol

import httpx


class LLMClient(Protocol):
    async def generate(self, prompt: str) -> str: ...

    async def next_token_logprobs(self, prompt: str, top_k: int = 10) -> dict[str, float]:
        """Log-probabilities of the top-k candidates for the first generated token."""
        ...


class OllamaClient:
    """Async client for Ollama's /api/generate, deterministic decoding."""

    def __init__(self, base_url: str, model: str, timeout_s: float, max_tokens: int):
        self.model = model
        self._max_tokens = max_tokens
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout_s)

    async def _post(self, prompt: str, num_predict: int, **extra) -> dict:
        resp = await self._client.post("/api/generate", json={
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": num_predict},
            **extra,
        })
        resp.raise_for_status()
        return resp.json()

    async def generate(self, prompt: str) -> str:
        return (await self._post(prompt, self._max_tokens)).get("response", "").strip()

    async def next_token_logprobs(self, prompt: str, top_k: int = 10) -> dict[str, float]:
        body = await self._post(prompt, 1, logprobs=True, top_logprobs=top_k)
        first = (body.get("logprobs") or [{}])[0]
        return {t["token"]: t["logprob"] for t in first.get("top_logprobs", [])}

    async def aclose(self) -> None:
        await self._client.aclose()
