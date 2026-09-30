"""
End-to-end benchmark against a running service (uvicorn service.app:app).

For each sampled PopQA question it sends, in this order:
    1. plain   + false context   -> baseline RAG answer and latency
    2. guarded + false context   -> detection on conflicts (closed-book cache cold)
    3. guarded + true context    -> false alarms (closed-book cache warm)

Reports:
    - unflagged false answers served: plain RAG vs guarded
    - detector catch rate / false-alarm rate
    - p50 / p95 latency per request type

Usage:
    rag_env/bin/python service/bench/run_benchmark.py --per-tier 100 --url http://localhost:8000
"""
import argparse
import asyncio
import json
import os
import statistics
import sys
import time

import httpx

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "analysis"))
from matching_v2 import classify  # noqa: E402

DATA_FILE = os.path.join(PROJECT_ROOT, "data", "popqa_conflicts_v2.json")
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "service")


def pctl(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))] if values else float("nan")


async def post(client, question, context, mode):
    start = time.perf_counter()
    resp = await client.post("/v1/answer", json={"question": question, "contexts": [context], "mode": mode})
    resp.raise_for_status()
    body = resp.json()
    body["client_ms"] = (time.perf_counter() - start) * 1000
    return body


async def run(args):
    with open(DATA_FILE, encoding="utf-8") as f:
        data = json.load(f)["details"]
    samples = [s for tier in ("High", "Medium", "Low")
               for s in [x for x in data if x["saliency"] == tier][-args.per_tier:]]

    sem = asyncio.Semaphore(args.concurrency)
    records = []

    async with httpx.AsyncClient(base_url=args.url, timeout=300) as client:
        async def one(s):
            async with sem:
                plain = await post(client, s["q"], s["conflicting_context"], "plain")
                conflict = await post(client, s["q"], s["conflicting_context"], "guarded")
                support = await post(client, s["q"], s["supporting_context"], "guarded")
            records.append({"id": s["id"], "saliency": s["saliency"], "gt": s["gt"], "fake": s["fake"],
                            "plain": plain, "conflict": conflict, "support": support})
            if len(records) % 25 == 0:
                print(f"  {len(records)}/{len(samples)}", flush=True)

        start = time.perf_counter()
        await asyncio.gather(*(one(s) for s in samples))
        wall_s = time.perf_counter() - start

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"benchmark_{args.tag}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "wall_s": wall_s, "records": records}, f, indent=1, ensure_ascii=False)
    report(records, wall_s)
    print(f"\nSaved -> {out}")


def report(records, wall_s):
    n = len(records)

    def adopted(r, key):
        return classify(r[key]["context_answer"], r["gt"], r["fake"]) == "Adherence"

    plain_bad = sum(adopted(r, "plain") for r in records)
    adopted_conflicts = [r for r in records if adopted(r, "conflict")]
    caught = sum(r["conflict"]["route"] != "answered" for r in adopted_conflicts)
    guarded_bad = len(adopted_conflicts) - caught
    false_alarm = sum(r["support"]["route"] != "answered" for r in records)

    print(f"\n=== Safety (N={n} questions with false context) ===")
    print(f"  plain RAG serves the false answer         {100 * plain_bad / n:5.1f}%")
    print(f"  guarded serves it unflagged               {100 * guarded_bad / n:5.1f}%")
    print(f"  detector catch rate (of adopted answers)  {100 * caught / max(1, len(adopted_conflicts)):5.1f}%")
    print(f"  false alarms on true context              {100 * false_alarm / n:5.1f}%")

    print("\n=== Latency (ms, client-side) ===")
    rows = [("plain RAG", [r["plain"]["client_ms"] for r in records]),
            ("guarded, cache cold", [r["conflict"]["client_ms"] for r in records]),
            ("guarded, cache warm", [r["support"]["client_ms"] for r in records])]
    for name, vals in rows:
        print(f"  {name:<22} p50 {pctl(vals, 0.5):7.0f}   p95 {pctl(vals, 0.95):7.0f}   mean {statistics.mean(vals):7.0f}")
    base = statistics.median(rows[0][1])
    for name, vals in rows[1:]:
        print(f"  overhead vs plain, {name.split(', ')[1]:<10} p50 {statistics.median(vals) - base:+7.0f} ms")
    print(f"  cache hit rate on warm requests          {100 * sum(r['support']['cache_hit'] for r in records) / n:5.1f}%")
    print(f"  throughput {3 * n / wall_s:.2f} req/s over {wall_s:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--per-tier", type=int, default=100)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--tag", default="baseline")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
