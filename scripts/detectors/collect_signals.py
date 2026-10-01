"""
collect_signals.py
==================
Collect the extra model signals the step-2 detectors need, on a fixed dev/test
split of the v2 PopQA set. Context answers are reused from results/v2/<model>.json
(same prompt, greedy decoding), so only these calls are new:

    closed_book   closed-book answer with token logprobs  -> confidence
    verify_<cond> "is <context answer> correct?" 1 token  -> P(Yes)
                  for cond in {conflict, support}

Split: test = last 100 questions per tier (the same 300 the service benchmark
uses), dev = the 200 before them per tier.

Resumable; writes results/detectors/<model>_signals.json.

Usage:
    rag_env/bin/python scripts/detectors/collect_signals.py --model llama3:8b
"""
import argparse
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_FILE = os.path.join(PROJECT_ROOT, "data", "popqa_conflicts_v2.json")
V2_DIR = os.path.join(PROJECT_ROOT, "results", "v2")
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "detectors")
OLLAMA_URL = "http://localhost:11434/api/generate"

TEST_PER_TIER, DEV_PER_TIER = 100, 200

CLOSED_BOOK = (
    "Instruction: Answer the following question in 1-3 words based on your internal knowledge. "
    "If the answer is unknown, strictly respond with 'I don't know'.\n\n"
    "Question: {q}\nAnswer:"
)
VERIFY = (
    "Question: {q}\nProposed answer: {a}\n"
    "Based on your own knowledge, is the proposed answer correct? Reply with Yes or No.\nReply:"
)


def split_ids(samples):
    split = {}
    for tier in ("High", "Medium", "Low"):
        ids = [s["id"] for s in samples if s["saliency"] == tier]
        for i in ids[-TEST_PER_TIER:]:
            split[i] = "test"
        for i in ids[-(TEST_PER_TIER + DEV_PER_TIER):-TEST_PER_TIER]:
            split[i] = "dev"
    return split


def ollama(model, prompt, num_predict, top_logprobs=0):
    payload = {"model": model, "prompt": prompt, "stream": False, "logprobs": True,
               "options": {"temperature": 0.0, "num_predict": num_predict}}
    if top_logprobs:
        payload["top_logprobs"] = top_logprobs
    for attempt in range(3):
        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=120)
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def closed_book_signal(model, q):
    out = ollama(model, CLOSED_BOOK.format(q=q), num_predict=16)
    lps = [t["logprob"] for t in out.get("logprobs") or []]
    # Confidence over the answer tokens only (stop at the first newline)
    answer_lps = []
    for t, lp in zip(out.get("logprobs") or [], lps):
        if "\n" in t["token"] and answer_lps:
            break
        answer_lps.append(lp)
    return {
        "pred": out.get("response", "").strip(),
        "first_token_p": math.exp(answer_lps[0]) if answer_lps else 0.0,
        "mean_logprob": sum(answer_lps) / len(answer_lps) if answer_lps else -99.0,
        "min_logprob": min(answer_lps) if answer_lps else -99.0,
    }


def verify_signal(model, q, answer):
    out = ollama(model, VERIFY.format(q=q, a=answer), num_predict=1, top_logprobs=10)
    top = (out.get("logprobs") or [{}])[0].get("top_logprobs", [])
    p_yes = sum(math.exp(t["logprob"]) for t in top if t["token"].strip().lower().startswith("yes"))
    p_no = sum(math.exp(t["logprob"]) for t in top if t["token"].strip().lower().startswith("no"))
    return {"reply": out.get("response", "").strip(), "p_yes": p_yes, "p_no": p_no,
            "p_yes_norm": p_yes / (p_yes + p_no) if p_yes + p_no > 0 else 0.5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    with open(DATA_FILE, encoding="utf-8") as f:
        samples = json.load(f)["details"]
    by_id = {s["id"]: s for s in samples}
    split = split_ids(samples)
    tag = args.model.replace(":", "_")
    with open(os.path.join(V2_DIR, f"{tag}.json"), encoding="utf-8") as f:
        v2 = json.load(f)["results"]

    os.makedirs(OUT_DIR, exist_ok=True)
    out_file = os.path.join(OUT_DIR, f"{tag}_signals.json")
    signals = {}
    if os.path.exists(out_file):
        with open(out_file, encoding="utf-8") as f:
            signals = json.load(f)["signals"]

    jobs = []
    for sid, part in split.items():
        rec = signals.setdefault(sid, {"split": part})
        q = by_id[sid]["q"]
        if "closed_book" not in rec:
            jobs.append((sid, "closed_book", lambda q=q: closed_book_signal(args.model, q)))
        for cond in ("conflict", "support"):
            key = f"verify_{cond}"
            if key not in rec:
                ans = v2[sid][cond]["pred"]
                jobs.append((sid, key, lambda q=q, a=ans: verify_signal(args.model, q, a)))
    print(f"{args.model}: {len(jobs)} calls for {len(split)} questions", flush=True)

    def save():
        tmp = out_file + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"model": args.model, "signals": signals}, f, indent=1, ensure_ascii=False)
        os.replace(tmp, out_file)

    start = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fn): (sid, key) for sid, key, fn in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            sid, key = futures[fut]
            signals[sid][key] = fut.result()
            if i % 200 == 0 or i == len(jobs):
                save()
                print(f"  {i}/{len(jobs)}  {i / (time.time() - start):.1f} calls/s", flush=True)
    save()
    print(f"Saved -> {out_file}")


if __name__ == "__main__":
    main()
