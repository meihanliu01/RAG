"""
generate_all_tables.py
======================
一键生成论文中所有 Table (2-6) 的数据。
确保所有表格数据来自同一数据源 (rag_final_labeled_augmented.json)，
彻底消除因多个脚本使用不同字段/文件导致的数据不一致。

Usage:
    python scripts/analysis/generate_all_tables.py
"""
import json
import os
import numpy as np
from collections import defaultdict, Counter

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# ============================================================
# 数据加载
# ============================================================
def load_labeled_data(filename="rag_final_labeled_augmented.json"):
    filepath = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(filepath):
        print(f"❌ Error: {filepath} not found.")
        print("   请先运行: python scripts/analysis/behavioral_classifier.py")
        return None
    
    with open(filepath, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    
    samples = raw.get('details', raw) if isinstance(raw, dict) else raw
    print(f"✅ Loaded {len(samples)} samples from {filename}")
    return samples


def pct(count, total, decimals=1):
    return round(count / total * 100, decimals) if total > 0 else 0.0


def ci_95(p_hat, n):
    """Wilson score interval for 95% CI"""
    if n == 0:
        return (0.0, 0.0)
    z = 1.96
    margin = z * np.sqrt((p_hat * (1 - p_hat)) / n)
    return (round((p_hat - margin) * 100, 1), round((p_hat + margin) * 100, 1))


# ============================================================
# Table 2: Baseline Behavior Distribution
# ============================================================
def table_2_baseline(samples):
    print("\n" + "="*80)
    print("TABLE 2: Baseline Model Behavior Distribution")
    print("="*80)
    
    total = len(samples)
    label_counts = Counter(s.get('label', 'Uncertain/Other') for s in samples)
    
    print(f"\n{'Category':<20} {'Count':>6} {'Percentage':>12} {'95% CI':>20} {'Note':<25}")
    print("-" * 85)
    
    for label in ['Uncertain/Other', 'Adherence', 'Persistence']:
        count = label_counts.get(label, 0)
        p = pct(count, total)
        lo, hi = ci_95(count / total, total)
        note = {
            'Uncertain/Other': 'Maximum Entropy',
            'Adherence': 'Cognitive Surrender',
            'Persistence': 'Parametric Sovereignty'
        }.get(label, '')
        print(f"{label:<20} {count:>6} {p:>10.1f}% [{lo:.1f}, {hi:.1f}]{'':<3} {note}")
    
    print(f"\nTotal: N={total}")
    return label_counts


# ============================================================
# Table 3: Saliency Breakdown with Resilience
# ============================================================
def table_3_saliency(samples):
    print("\n" + "="*80)
    print("TABLE 3: Knowledge Sovereignty - Base Accuracy vs. Behavioral Resilience")
    print("="*80)
    
    stats = defaultdict(lambda: defaultdict(int))
    totals = defaultdict(int)
    known_counts = defaultdict(int)
    resilient_counts = defaultdict(int)
    
    for entry in samples:
        saliency = entry.get('saliency', 'Medium')
        label = entry.get('label', 'Uncertain/Other')
        is_known = entry.get('is_known_by_model', False)
        
        stats[saliency][label] += 1
        totals[saliency] += 1
        
        if is_known:
            known_counts[saliency] += 1
            if label == "Persistence":
                resilient_counts[saliency] += 1
    
    print(f"\n{'Tier':<8} {'Base Acc':>10} {'Resilience':>12} {'Persist':>10} {'Adhere':>10} {'Uncert':>10} {'N':>6}")
    print("-" * 70)
    
    for tier in ['High', 'Medium', 'Low']:
        n = totals[tier]
        if n == 0:
            print(f"{tier:<8} {'N/A':>10} {'N/A':>12} {'N/A':>10} {'N/A':>10} {'N/A':>10} {0:>6}")
            continue
        
        base_acc = pct(known_counts[tier], n)
        resilience = pct(resilient_counts[tier], known_counts[tier]) if known_counts[tier] > 0 else 0.0
        p = pct(stats[tier].get('Persistence', 0), n)
        a = pct(stats[tier].get('Adherence', 0), n)
        u = pct(stats[tier].get('Uncertain/Other', 0), n)
        
        print(f"{tier:<8} {base_acc:>9.1f}% {resilience:>10.1f}% {p:>9.1f}% {a:>9.1f}% {u:>9.1f}% {n:>6}")
    
    print(f"\nTotal: N={sum(totals.values())}")


# ============================================================
# Table 4: Ablation Results (from ablation_results.json)
# ============================================================
def table_4_ablation():
    print("\n" + "="*80)
    print("TABLE 4: Model Behavior Across Instruction Variants")
    print("="*80)
    
    ablation_file = os.path.join(RESULTS_DIR, "ablation_results.json")
    if not os.path.exists(ablation_file):
        print("⚠️ ablation_results.json not found. Skipping Table 4.")
        print("   请先运行: python scripts/inference/ablation_instruction_strength.py")
        return
    
    with open(ablation_file, 'r') as f:
        data = json.load(f)
    
    import re, string
    def normalize(s):
        s = str(s).lower()
        s = re.sub(r'\b(a|an|the)\b', ' ', s)
        s = ''.join(ch for ch in s if ch not in set(string.punctuation))
        return ' '.join(s.split())
    
    def is_soft_match(pred, target):
        if not pred or not target: return False
        p, t = normalize(pred), normalize(target)
        if t in p or p in t: return True
        pt, tt = set(p.split()), set(t.split())
        if not tt: return False
        return len(pt & tt) / len(tt) >= 0.5
    
    variants = defaultdict(lambda: Counter())
    variant_totals = defaultdict(int)
    
    for entry in data:
        variant = entry.get('variant', 'unknown')
        pred = entry.get('pred', '')
        gold = entry.get('gold', '')
        fake = entry.get('fake', '')
        
        # 使用与 behavioral_classifier 一致的分类逻辑
        pred_lower = pred.lower()
        if any(msg in pred_lower for msg in ["don't know", "dont know", "not mentioned", "no information"]) or not pred.strip():
            behavior = 'Uncertain'
        elif is_soft_match(pred, gold):
            behavior = 'Persistence'
        elif is_soft_match(pred, fake):
            behavior = 'Adherence'
        else:
            behavior = 'Uncertain'
        
        variants[variant][behavior] += 1
        variant_totals[variant] += 1
    
    print(f"\n{'Variant':<22} {'Persistence':>14} {'Adherence':>14} {'Uncertain':>14} {'N':>6}")
    print("-" * 72)
    
    for variant in ['neutral', 'strict_context', 'strict_parametric']:
        total = variant_totals[variant]
        if total == 0:
            continue
        p = pct(variants[variant]['Persistence'], total)
        a = pct(variants[variant]['Adherence'], total)
        u = pct(variants[variant]['Uncertain'], total)
        print(f"{variant:<22} {p:>13.1f}% {a:>13.1f}% {u:>13.1f}% {total:>6}")


# ============================================================
# Table 5: Interaction Between Saliency and Parametric Knowledge
# ============================================================
def table_5_interaction(samples):
    print("\n" + "="*80)
    print("TABLE 5: Interaction Between Saliency and Parametric Knowledge")
    print("="*80)
    
    stats = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    
    for item in samples:
        saliency = item.get('saliency', 'Unknown')
        known = "Correct" if item.get('is_known_by_model', False) else "Incorrect"
        label = item.get('label', 'Uncertain/Other')
        
        if label not in ['Persistence', 'Adherence', 'Uncertain/Other']:
            label = 'Uncertain/Other'
        
        stats[saliency][known]['N'] += 1
        stats[saliency][known][label] += 1
    
    print(f"\n{'Saliency':<10} {'Probe':<12} {'Persist':>10} {'Adhere':>10} {'Uncert':>12} {'N':>6}")
    print("-" * 65)
    
    for sal in ['High', 'Medium', 'Low']:
        for known in ['Correct', 'Incorrect']:
            total = stats[sal][known]['N']
            if total == 0:
                print(f"{sal:<10} {known:<12} {'N/A':>10} {'N/A':>10} {'N/A':>12} {0:>6}")
                continue
            p = pct(stats[sal][known]['Persistence'], total)
            a = pct(stats[sal][known]['Adherence'], total)
            u = pct(stats[sal][known]['Uncertain/Other'], total)
            print(f"{sal:<10} {known:<12} {p:>9.1f}% {a:>9.1f}% {u:>11.1f}% {total:>6}")
        print("-" * 65)


# ============================================================
# Table 6: Behavior Conditioned on Parametric Knowledge
# ============================================================
def table_6_knowledge_conditioning(samples):
    print("\n" + "="*80)
    print("TABLE 6: Model Behavior Conditioned on Parametric Knowledge")
    print("="*80)
    
    groups = {'Probe-Correct': [], 'Probe-Incorrect': []}
    
    for item in samples:
        key = 'Probe-Correct' if item.get('is_known_by_model', False) else 'Probe-Incorrect'
        groups[key].append(item)
    
    print(f"\n{'Base Knowledge':<18} {'Persist':>12} {'Adhere':>12} {'Uncert':>14} {'N':>8}")
    print("-" * 68)
    
    for group_name in ['Probe-Correct', 'Probe-Incorrect']:
        items = groups[group_name]
        total = len(items)
        if total == 0:
            continue
        labels = Counter(i.get('label', 'Uncertain/Other') for i in items)
        p = pct(labels.get('Persistence', 0), total)
        a = pct(labels.get('Adherence', 0), total)
        u = pct(labels.get('Uncertain/Other', 0), total)
        print(f"{group_name:<18} {p:>11.1f}% {a:>11.1f}% {u:>13.1f}% {total:>8}")


# ============================================================
# Consistency Check
# ============================================================
def consistency_check(samples):
    print("\n" + "="*80)
    print("CONSISTENCY CHECK")
    print("="*80)
    
    total = len(samples)
    
    # Check label field coverage
    has_label = sum(1 for s in samples if 'label' in s)
    has_final_label = sum(1 for s in samples if 'final_label' in s)
    has_pred_base = sum(1 for s in samples if 'pred_base' in s)
    has_is_known = sum(1 for s in samples if 'is_known_by_model' in s)
    
    print(f"\n  Total samples: {total}")
    print(f"  Has 'label':              {has_label}/{total}")
    print(f"  Has 'final_label':        {has_final_label}/{total}")
    print(f"  Has 'pred_base':          {has_pred_base}/{total}")
    print(f"  Has 'is_known_by_model':  {has_is_known}/{total}")
    
    # Check label vs final_label agreement
    if has_label > 0 and has_final_label > 0:
        agree = sum(1 for s in samples if s.get('label') == s.get('final_label'))
        print(f"\n  ⚠️ label == final_label agreement: {agree}/{total} ({pct(agree, total)}%)")
        if agree < total:
            print("  → These fields differ! All table generation uses 'label' (from classifier).")
    
    # Saliency distribution
    sal_counts = Counter(s.get('saliency') for s in samples)
    print(f"\n  Saliency distribution: {dict(sal_counts)}")
    
    # Warning for small sample sizes
    for tier, count in sal_counts.items():
        if count < 100:
            print(f"  ⚠️ Warning: {tier} tier has only N={count} samples (recommend >= 100)")
    
    print()


# ============================================================
# Table 7: Cross-Model Comparison (Llama-3 vs Qwen2.5)
# ============================================================
def table_7_cross_model(llama_samples):
    print("\n" + "="*80)
    print("TABLE 7: Cross-Model Comparison (Llama-3-8B vs Qwen2.5-7B)")
    print("="*80)
    
    qwen_file = os.path.join(RESULTS_DIR, "qwen_final_labeled.json")
    if not os.path.exists(qwen_file):
        print("⚠️ qwen_final_labeled.json not found. Skipping Table 7.")
        print("   请先运行: python scripts/inference/qwen_full_pipeline.py")
        return
    
    with open(qwen_file, 'r', encoding='utf-8') as f:
        qwen_raw = json.load(f)
    qwen_samples = qwen_raw.get('details', qwen_raw) if isinstance(qwen_raw, dict) else qwen_raw
    
    def compute_tier_stats(samples):
        stats = {}
        for tier in ['High', 'Medium', 'Low']:
            subset = [s for s in samples if s.get('saliency') == tier]
            n = len(subset)
            if n == 0:
                stats[tier] = {'p': 0, 'a': 0, 'u': 0, 'n': 0}
                continue
            labels = Counter(s.get('label', 'Uncertain/Other') for s in subset)
            stats[tier] = {
                'p': round(labels.get('Persistence', 0) / n * 100, 1),
                'a': round(labels.get('Adherence', 0) / n * 100, 1),
                'u': round(labels.get('Uncertain/Other', 0) / n * 100, 1),
                'n': n
            }
        return stats
    
    llama_stats = compute_tier_stats(llama_samples)
    qwen_stats = compute_tier_stats(qwen_samples)
    
    print(f"\n{'Tier':<8} | {'--- Llama-3-8B ---':^36} | {'--- Qwen2.5-7B ---':^36}")
    print(f"{'':8} | {'Persist':>8} {'Adhere':>8} {'Uncert':>8} {'N':>6} | {'Persist':>8} {'Adhere':>8} {'Uncert':>8} {'N':>6}")
    print("-" * 90)
    
    for tier in ['High', 'Medium', 'Low']:
        ls = llama_stats[tier]
        qs = qwen_stats[tier]
        print(f"{tier:<8} | {ls['p']:>7.1f}% {ls['a']:>7.1f}% {ls['u']:>7.1f}% {ls['n']:>5} | {qs['p']:>7.1f}% {qs['a']:>7.1f}% {qs['u']:>7.1f}% {qs['n']:>5}")
    
    # Overall
    print("-" * 90)
    for model_name, samples in [("Llama-3", llama_samples), ("Qwen2.5", qwen_samples)]:
        n = len(samples)
        labels = Counter(s.get('label', 'Uncertain/Other') for s in samples)
        p = round(labels.get('Persistence', 0) / n * 100, 1) if n > 0 else 0
        a = round(labels.get('Adherence', 0) / n * 100, 1) if n > 0 else 0
        u = round(labels.get('Uncertain/Other', 0) / n * 100, 1) if n > 0 else 0
        print(f"  {model_name} Overall: Persist={p}%, Adhere={a}%, Uncert={u}% (N={n})")


# ============================================================
# Main
# ============================================================
def main():
    samples = load_labeled_data()
    if samples is None:
        return
    
    consistency_check(samples)
    table_2_baseline(samples)
    table_3_saliency(samples)
    table_4_ablation()
    table_5_interaction(samples)
    table_6_knowledge_conditioning(samples)
    table_7_cross_model(samples)
    
    print("\n" + "="*80)
    print("✅ All tables generated successfully.")
    print("   Llama data: results/rag_final_labeled_augmented.json")
    print("   Qwen data:  results/qwen_final_labeled.json")
    print("="*80)

if __name__ == "__main__":
    main()
