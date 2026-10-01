# Conflict-aware RAG service

The v2 evaluation showed that Llama-3-8B and Qwen2.5-7B adopt plausible false
context 86–92% of the time and almost never correct it, and that prompt wording
alone cannot fix this. This service moves the check out of the prompt and into
the system: the context-grounded answer is checked before it is returned.

Two detectors (`DETECTOR` env var):

```
verify (default)
question + context ─> RAG answer ─> "is <answer> correct? Yes/No" (1 token, logprobs) ─> route
                                     flag if log P(Yes) <= dev-tuned threshold

agreement (step-1 baseline)
             ┌─ closed-book answer (cached per question) ─┐
question ────┤          (asyncio.gather)                   ├─> compare ─> route
             └─ context-grounded answer ───────────────────┘

accepted / consistent / no prior -> answered
rejected / answers disagree      -> flagged   (answer returned with a warning)
model refuses from context       -> abstained
```

## Results

End-to-end benchmark, Llama-3-8B, the same 300 PopQA test questions (100 per
popularity tier), each sent with a false and with a true context. Thresholds were
tuned on a separate 600-question dev split.

| | Plain RAG | Agreement (step 1) | **Verify (step 2)** |
|---|---|---|---|
| False answers served without a flag | 87.0% | 57.3% | **49.0%** |
| Catch rate (of adopted false answers) | – | 34.1% | **43.7%** |
| False alarms on true context | – | 21.0% | **10.7%** |
| p50 / p95 latency, true context | 504 / 746 ms | 616 / 919 ms | 620 / 950 ms |
| p50 overhead vs plain | – | +112 ms | +102 ms |

Verify catches more with half the false alarms. Its advantage comes from cases
where the model cannot recall the answer closed-book (60% of the agreement
detector's misses): recognising that an answer is wrong does not require
recalling the right one. The threshold trades catch rate against false alarms;
on the test split a 20% false-alarm budget catches 61.7%.

Offline detector comparison (`scripts/detectors/`): answer agreement, agreement
gated on closed-book confidence (logprobs), verification, their union, and a
logistic regression over all signals. Verification alone matched the logistic
regression (43.7% vs 42.5% catch at ~11% false alarms), so the service uses the
single signal.

Latency: Apple M4, `OLLAMA_NUM_PARALLEL=1`, one request at a time. Overhead is
measured on true-context requests; the false-context guarded request runs right
after the plain one with the same prompt and benefits from Ollama's prompt cache.

## Run

```bash
pip install -r service/requirements.txt
uvicorn service.app:app --port 8000          # needs Ollama with llama3:8b (CONFLICT_MODEL to change)

curl -X POST localhost:8000/v1/answer -H 'Content-Type: application/json' -d '{
  "question": "Who was the director of The Mask?",
  "contexts": ["The Mask was directed by Colin Nutley."]
}'
# -> {"answer": "Colin Nutley.", "route": "flagged",
#     "detection": {"verdict": "rejected", "log_p_yes": -11.07}, ...}
```

`"mode": "plain"` skips the check (standard RAG) for comparison. `GET /healthz`
reports cache size and hit counts.

## Test and benchmark

```bash
pytest service/tests -q                                   # no Ollama needed (fake LLM)
python service/bench/run_benchmark.py --per-tier 100      # against a running server

# collect signals and re-tune the verify threshold (per model)
python scripts/detectors/collect_signals.py --model llama3:8b
python scripts/detectors/eval_detectors.py --model llama3:8b
```

The benchmark sends each PopQA question with false context (plain and guarded)
and with true context (guarded), and reports how often a false answer is served
unflagged, the detector's catch / false-alarm rates, and p50 / p95 latency
against plain RAG.

## Layout

| File | Role |
|---|---|
| `app.py` | FastAPI app, request validation, 502 on LLM backend errors |
| `pipeline.py` | verify / agreement flows, routing, per-stage timings |
| `detector.py` | verify and agreement detectors, `Verdict` / `Detection` types |
| `cache.py` | bounded LRU + TTL cache for closed-book answers and verify scores |
| `llm.py` | async Ollama client behind an `LLMClient` protocol |
| `config.py` | settings from environment variables |
