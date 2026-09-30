"""
analyze_v2.py
=============
Tables for the v2 (PopQA) experiment, read from results/v2/<model>.json.

    1. Conflict behavior by saliency tier (neutral prompt), with 95% Wilson CIs
    2. Control: accuracy with SUPPORTING (true) context
    3. Conflict behavior split by zero-context probe (known vs unknown)
    4. Instruction ablation (100 samples per tier)

Usage:
    rag_env/bin/python scripts/analysis/analyze_v2.py
"""
import json
import math
import os
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_FILE = os.path.join(PROJECT_ROOT, "data", "popqa_conflicts_v2.json")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "v2")
MODELS = [("Llama-3-8B", "llama3_8b.json"), ("Qwen2.5-7B", "qwen2.5_7b.json")]
TIERS = ['High', 'Medium', 'Low']
VARIANTS = [('neutral', 'conflict'), ('strict_context', 'conflict_strict_context'),
            ('strict_parametric', 'conflict_strict_parametric')]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (100 * (centre - half), 100 * (centre + half))


def dist(rows, cond):
    labels = [r[cond]['label'] for r in rows if cond in r]
    n = len(labels)
    c = Counter(labels)
    return n, {k: c[k] for k in ('Persistence', 'Adherence', 'Mixed', 'Uncertain/Other')}


def fmt(k, n):
    return f"{100 * k / n:5.1f}%" if n else "   n/a"


def report(name, results, meta):
    rows_by_tier = {t: [results[i] for i in results if meta[i]['saliency'] == t] for t in TIERS}
    all_rows = [r for t in TIERS for r in rows_by_tier[t]]

    print(f"\n{'=' * 78}\n{name}  (N={len(all_rows)})\n{'=' * 78}")

    print("\n[1] False context, neutral prompt")
    print(f"{'Tier':<8}{'N':>5}  {'Persist':>7}  {'Adhere':>7} {'95% CI':>15}  {'Mixed':>6}  {'Uncert':>6}")
    for t in TIERS + ['All']:
        rows = all_rows if t == 'All' else rows_by_tier[t]
        n, c = dist(rows, 'conflict')
        lo, hi = wilson(c['Adherence'], n)
        print(f"{t:<8}{n:>5}  {fmt(c['Persistence'], n):>7}  {fmt(c['Adherence'], n):>7} "
              f"[{lo:5.1f}, {hi:5.1f}]  {fmt(c['Mixed'], n):>6}  {fmt(c['Uncertain/Other'], n):>6}")

    print("\n[2] Control: true context (accuracy = answers the true value)  |  zero-context probe accuracy")
    for t in TIERS + ['All']:
        rows = all_rows if t == 'All' else rows_by_tier[t]
        n, c = dist(rows, 'support')
        known = sum(r['probe']['label'] == 'Known' for r in rows if 'probe' in r)
        print(f"{t:<8}{n:>5}  support acc {fmt(c['Persistence'], n)}   probe acc {fmt(known, n)}")

    print("\n[3] False context, split by whether the model knows the answer without context")
    for t in TIERS + ['All']:
        rows = all_rows if t == 'All' else rows_by_tier[t]
        for known in ('Known', 'Unknown'):
            sub = [r for r in rows if r.get('probe', {}).get('label') == known]
            n, c = dist(sub, 'conflict')
            if n:
                print(f"{t:<8}{known:<8} N={n:<5} Persist {fmt(c['Persistence'], n)}  "
                      f"Adhere {fmt(c['Adherence'], n)}  Mixed {fmt(c['Mixed'], n)}")

    print("\n[4] Instruction ablation (same 100 samples per tier)")
    print(f"{'Tier':<8}" + ''.join(f"{v:>22}" for v, _ in VARIANTS) + "   (Adhere / Persist)")
    for t in TIERS + ['All']:
        rows = all_rows if t == 'All' else rows_by_tier[t]
        rows = [r for r in rows if 'conflict_strict_context' in r]
        line = f"{t:<8}"
        for _, cond in VARIANTS:
            n, c = dist(rows, cond)
            line += f"{fmt(c['Adherence'], n):>14} / {fmt(c['Persistence'], n)}"
        print(line + f"   N={len(rows)}")


def main():
    with open(DATA_FILE, encoding='utf-8') as f:
        meta = {s['id']: s for s in json.load(f)['details']}
    for name, fname in MODELS:
        path = os.path.join(RESULTS_DIR, fname)
        if not os.path.exists(path):
            print(f"\n{name}: {path} not found, skipping")
            continue
        with open(path, encoding='utf-8') as f:
            results = json.load(f)['results']
        report(name, results, meta)


if __name__ == "__main__":
    main()
