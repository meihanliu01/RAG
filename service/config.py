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
    # "verify" (default) or "agreement" (step-1 baseline)
    detector: str = os.getenv("DETECTOR", "verify")
    # Flag when log P(Yes) <= threshold. -9.245 = 10% false-alarm budget on the
    # dev split for llama3:8b (scripts/detectors/eval_detectors.py); re-tune per model.
    verify_log_p_yes_threshold: float = float(os.getenv("VERIFY_LOG_P_YES_THRESHOLD", "-9.245"))


settings = Settings()
