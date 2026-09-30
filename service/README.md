# Conflict-aware RAG service

The v2 evaluation showed that Llama-3-8B and Qwen2.5-7B adopt plausible false
context 86–92% of the time and almost never correct it, and that prompt wording
alone cannot fix this. This service moves the check out of the prompt and into
the system: every question is answered twice, **concurrently**, and the two
answers are compared before anything is returned.

```
             ┌─ closed-book answer (cached per question) ─┐
question ────┤                                             ├─> detector ─> route
             └─ context-grounded answer ───────────────────┘

consistent / model has no prior  -> answered
answers disagree                 -> flagged   (context answer + closed-book answer returned)
model refuses from context       -> abstained
```

## Run

```bash
pip install -r service/requirements.txt
uvicorn service.app:app --port 8000          # needs Ollama with llama3:8b (CONFLICT_MODEL to change)

curl -X POST localhost:8000/v1/answer -H 'Content-Type: application/json' -d '{
  "question": "Who was the director of The Mask?",
  "contexts": ["The Mask was directed by Colin Nutley."]
}'
# -> {"answer": "Colin Nutley.", "route": "flagged", "closed_book_answer": "Chuck Russell", ...}
```

`"mode": "plain"` skips the check (standard RAG) for comparison. `GET /healthz`
reports cache size and hit counts.

## Test and benchmark

```bash
pytest service/tests -q                                   # no Ollama needed (fake LLM)
python service/bench/run_benchmark.py --per-tier 100      # against a running server
```

The benchmark sends each PopQA question with false context (plain and guarded)
and with true context (guarded), and reports how often a false answer is served
unflagged, the detector's catch / false-alarm rates, and p50 / p95 latency
against plain RAG.

## Layout

| File | Role |
|---|---|
| `app.py` | FastAPI app, request validation, 502 on LLM backend errors |
| `pipeline.py` | concurrent closed-book + RAG calls, routing, per-stage timings |
| `detector.py` | baseline detector (answer agreement); the interface later detectors plug into |
| `cache.py` | bounded LRU + TTL cache for closed-book answers |
| `llm.py` | async Ollama client behind an `LLMClient` protocol |
| `config.py` | settings from environment variables |
