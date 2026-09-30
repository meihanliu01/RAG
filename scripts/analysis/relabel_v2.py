"""
relabel_v2.py
=============
Re-label existing model outputs with the fixed matcher (matching_v2) -- no new
inference. Originals are left untouched; v2 files are written alongside them.

Outputs:
    results/rag_final_labeled_v2.json     (Llama-3-8B)
    results/qwen_final_labeled_v2.json    (Qwen2.5-7B)
    results/ablation_labeled_v2.json
    results/manual_check_sample.csv       (stratified sample for human annotation)

Usage:
    rag_env/bin/python scripts/analysis/relabel_v2.py
"""
import csv
import json
import os
import random
from collections import Counter, defaultdict

from matching_v2 import classify, perturbation_type

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
TIERS = ['High', 'Medium', 'Low']
LABELS = ['Persistence', 'Adherence', 'Mixed', 'Uncertain/Other']


def load(name):
    with open(os.path.join(RESULTS_DIR, name), encoding='utf-8') as f:
        raw = json.load(f)
    return raw.get('details', raw) if isinstance(raw, dict) else raw


def save(name, samples):
    with open(os.path.join(RESULTS_DIR, name), 'w', encoding='utf-8') as f:
        json.dump({"details": samples}, f, indent=2, ensure_ascii=False)


def relabel(samples):
    for s in samples:
        s['label_v1'] = s.get('label')
        s['label'] = classify(s.get('pred', ''), s.get('gt') or s.get('gold'), s.get('fake', ''))
    return samples


def pct(n, d):
    return 100 * n / d if d else 0.0


def print_comparison(model, samples):
    print(f"\n=== {model}: v1 -> v2 by saliency tier ===")
    print(f"{'Tier':<8}{'N':>5} | {'v1 P':>6}{'v1 A':>6}{'v1 U':>6} | "
          f"{'v2 P':>6}{'v2 A':>6}{'v2 Mix':>7}{'v2 U':>6} | {'changed':>8}")
    for tier in TIERS + ['All']:
        sub = samples if tier == 'All' else [s for s in samples if s['saliency'] == tier]
        n = len(sub)
        v1 = Counter(s['label_v1'] for s in sub)
        v2 = Counter(s['label'] for s in sub)
        changed = sum(s['label'] != s['label_v1'] for s in sub)
        print(f"{tier:<8}{n:>5} | {pct(v1['Persistence'], n):>5.1f}%{pct(v1['Adherence'], n):>5.1f}%"
              f"{pct(v1['Uncertain/Other'], n):>5.1f}% | {pct(v2['Persistence'], n):>5.1f}%"
              f"{pct(v2['Adherence'], n):>5.1f}%{pct(v2['Mixed'], n):>6.1f}%"
              f"{pct(v2['Uncertain/Other'], n):>5.1f}% | {changed:>8}")

    transitions = Counter((s['label_v1'], s['label']) for s in samples if s['label'] != s['label_v1'])
    print("  Label transitions (v1 -> v2):")
    for (a, b), c in transitions.most_common():
        print(f"    {a:<16} -> {b:<16} {c:>5}")


def print_probe_interaction(model, samples):
    print(f"\n=== {model}: v2 persistence by tier x zero-context probe ===")
    for tier in TIERS:
        for known in (True, False):
            sub = [s for s in samples if s['saliency'] == tier and bool(s.get('is_known_by_model')) == known]
            if not sub:
                continue
            c = Counter(s['label'] for s in sub)
            n = len(sub)
            print(f"  {tier:<7}{'known' if known else 'unknown':<9} N={n:<5}"
                  f"P={pct(c['Persistence'], n):5.1f}%  A={pct(c['Adherence'], n):5.1f}%  "
                  f"Mix={pct(c['Mixed'], n):4.1f}%  U={pct(c['Uncertain/Other'], n):5.1f}%")


def print_by_perturbation(model, samples):
    print(f"\n=== {model}: v2 labels by perturbation type (High + Medium tiers) ===")
    sub = [s for s in samples if s['saliency'] in ('High', 'Medium')]
    by = defaultdict(list)
    for s in sub:
        by[perturbation_type(s['fake'])].append(s)
    for kind in ['alt_prefix', 'year', 'number']:
        g = by.get(kind, [])
        n = len(g)
        if not n:
            continue
        c = Counter(s['label'] for s in g)
        print(f"  {kind:<11} N={n:<5}P={pct(c['Persistence'], n):5.1f}%  A={pct(c['Adherence'], n):5.1f}%  "
              f"Mix={pct(c['Mixed'], n):4.1f}%  U={pct(c['Uncertain/Other'], n):5.1f}%")


def ablation_table(rows):
    print("\n=== Instruction ablation (Llama-3-8B, High tier): v1 -> v2 ===")
    by = defaultdict(list)
    for r in rows:
        by[r['variant']].append(r)
    for variant in ['neutral', 'strict_context', 'strict_parametric']:
        sub = by.get(variant, [])
        n = len(sub)
        if not n:
            continue
        c = Counter(r['label'] for r in sub)
        print(f"  {variant:<18} N={n}  P={pct(c['Persistence'], n):5.1f}%  A={pct(c['Adherence'], n):5.1f}%  "
              f"Mix={pct(c['Mixed'], n):4.1f}%  U={pct(c['Uncertain/Other'], n):5.1f}%")


def export_manual_sample(llama, per_tier=40, seed=0):
    """Stratified sample for human annotation; half drawn from label changes."""
    rng = random.Random(seed)
    rows = []
    for tier in TIERS:
        sub = [s for s in llama if s['saliency'] == tier]
        changed = [s for s in sub if s['label'] != s['label_v1']]
        same = [s for s in sub if s['label'] == s['label_v1']]
        pick = rng.sample(changed, min(per_tier // 2, len(changed)))
        pick += rng.sample(same, min(per_tier - len(pick), len(same)))
        rows += pick
    path = os.path.join(RESULTS_DIR, "manual_check_sample.csv")
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['id', 'saliency', 'question', 'gold', 'fake', 'model_output',
                    'label_v1', 'label_v2', 'human_label'])
        for i, s in enumerate(rows):
            w.writerow([i, s['saliency'], s['q'], ' | '.join(s['gt']), s['fake'],
                        s['pred'], s['label_v1'], s['label'], ''])
    print(f"\nManual-check sample ({len(rows)} rows) -> {path}")


def main():
    llama = relabel(load("rag_final_labeled_augmented.json"))
    qwen = relabel(load("qwen_final_labeled.json"))
    ablation = relabel(load("ablation_results.json"))

    save("rag_final_labeled_v2.json", llama)
    save("qwen_final_labeled_v2.json", qwen)
    save("ablation_labeled_v2.json", ablation)

    for model, samples in [("Llama-3-8B", llama), ("Qwen2.5-7B", qwen)]:
        print_comparison(model, samples)
    for model, samples in [("Llama-3-8B", llama), ("Qwen2.5-7B", qwen)]:
        print_by_perturbation(model, samples)
    print_probe_interaction("Llama-3-8B", llama)
    ablation_table(ablation)
    export_manual_sample(llama)


if __name__ == "__main__":
    main()
