"""
run_conflict_eval.py
====================
v2 inference: for every sample in data/popqa_conflicts_v2.json, query the model
in three conditions and label each output with matching_v2.classify:

    probe       no context             -> does the model know the answer?
    conflict    false context          -> Persistence / Adherence / ...
    support     true context (control) -> does it accept correct evidence?

Plus an instruction ablation (neutral / strict_context / strict_parametric) on
the conflict condition for a fixed subset of every tier.

Resumable: results are checkpointed to results/v2/<model>.json and finished
(sample, condition) pairs are skipped on re-run.

Usage:
    rag_env/bin/python scripts/inference/run_conflict_eval.py --model llama3:8b
    rag_env/bin/python scripts/inference/run_conflict_eval.py --model qwen2.5:7b --limit 30
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "analysis"))
from matching_v2 import classify  # noqa: E402

DATA_FILE = os.path.join(PROJECT_ROOT, "data", "popqa_conflicts_v2.json")
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "v2")
OLLAMA_URL = "http://localhost:11434/api/generate"

PROBE_PROMPT = (
    "Instruction: Answer the following question in 1-3 words based on your internal knowledge. "
    "If the answer is unknown, strictly respond with 'I don't know'.\n\n"
    "Question: {q}\nAnswer:"
)
RAG_PROMPT = "{instruction}\n\nContext: {context}\n\nQuestion: {q}\nAnswer:"
# Same wording as the v1 ablation (ablation_instruction_strength.py)
INSTRUCTIONS = {
    "neutral": (
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer based on the context, just say you don't know. "
        "Keep the answer as short as possible."
    ),
    "strict_context": (
        "CRITICAL INSTRUCTION: You must ONLY use the provided context to answer the question. "
        "Ignore any prior knowledge you may have. The context is the sole source of truth. "
        "If the answer is in the context, extract it exactly. Keep the answer as short as possible."
    ),
    "strict_parametric": (
        "Use the following context as a reference, but prioritize factual accuracy above all else. "
        "If the context contains information that contradicts well-known facts, correct it in your response. "
        "Keep the answer as short as possible."
    ),
}
ABLATION_PER_TIER = 100


def ollama(model, prompt, retries=3):
    payload = {"model": model, "prompt": prompt, "stream": False,
               "options": {"temperature": 0.0, "num_predict": 48}}
    for attempt in range(retries):
        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=120)
            r.raise_for_status()
            return r.json().get("response", "").strip()
        except requests.RequestException as exc:
            if attempt == retries - 1:
                return f"Error: {exc}"
            time.sleep(2 ** attempt)


def build_jobs(samples):
    """(sample_id, condition, prompt) for every query this run needs."""
    jobs = []
    ablation_ids = set()
    for tier in ('High', 'Medium', 'Low'):
        tier_ids = [s['id'] for s in samples if s['saliency'] == tier]
        ablation_ids.update(tier_ids[:ABLATION_PER_TIER])
    for s in samples:
        jobs.append((s['id'], 'probe', PROBE_PROMPT.format(q=s['q'])))
        jobs.append((s['id'], 'conflict', RAG_PROMPT.format(
            instruction=INSTRUCTIONS['neutral'], context=s['conflicting_context'], q=s['q'])))
        jobs.append((s['id'], 'support', RAG_PROMPT.format(
            instruction=INSTRUCTIONS['neutral'], context=s['supporting_context'], q=s['q'])))
        if s['id'] in ablation_ids:
            for variant in ('strict_context', 'strict_parametric'):
                jobs.append((s['id'], f'conflict_{variant}', RAG_PROMPT.format(
                    instruction=INSTRUCTIONS[variant], context=s['conflicting_context'], q=s['q'])))
    return jobs


def label(sample, condition, pred):
    if condition == 'probe':
        # Known = names the true answer; the fake is irrelevant without context
        return 'Known' if classify(pred, sample['gt'], '') == 'Persistence' else 'Unknown'
    return classify(pred, sample['gt'], sample['fake'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--limit', type=int, default=None, help='samples per tier (smoke test)')
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()

    with open(DATA_FILE, encoding='utf-8') as f:
        samples = json.load(f)['details']
    if args.limit:
        samples = [s for t in ('High', 'Medium', 'Low')
                   for s in [x for x in samples if x['saliency'] == t][:args.limit]]
    by_id = {s['id']: s for s in samples}

    os.makedirs(OUT_DIR, exist_ok=True)
    out_file = os.path.join(OUT_DIR, f"{args.model.replace(':', '_')}.json")
    results = {}
    if os.path.exists(out_file):
        with open(out_file, encoding='utf-8') as f:
            results = json.load(f)['results']

    def done(sid, cond):
        pred = results.get(sid, {}).get(cond, {}).get('pred')
        return pred is not None and not pred.startswith('Error')

    jobs = [j for j in build_jobs(samples) if not done(j[0], j[1])]
    print(f"{args.model}: {len(jobs)} queries to run ({len(samples)} samples)")

    def save():
        tmp = out_file + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump({'model': args.model, 'results': results}, f, indent=1, ensure_ascii=False)
        os.replace(tmp, out_file)

    start = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(ollama, args.model, prompt): (sid, cond) for sid, cond, prompt in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            sid, cond = futures[fut]
            pred = fut.result()
            results.setdefault(sid, {})[cond] = {'pred': pred, 'label': label(by_id[sid], cond, pred)}
            if i % 200 == 0 or i == len(jobs):
                save()
                rate = i / (time.time() - start)
                print(f"  {i}/{len(jobs)}  {rate:.1f} q/s  eta {(len(jobs) - i) / rate / 60:.0f} min", flush=True)
    save()
    print(f"Saved -> {out_file}")


if __name__ == "__main__":
    main()
