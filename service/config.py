import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    model: str = os.getenv("CONFLICT_MODEL", "llama3:8b")
    llm_timeout_s: float = float(os.getenv("LLM_TIMEOUT_S", "60"))
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "48"))
    cache_size: int = int(os.getenv("CLOSED_BOOK_CACHE_SIZE", "10000"))
    cache_ttl_s: float = float(os.getenv("CLOSED_BOOK_CACHE_TTL_S", "3600"))


settings = Settings()
