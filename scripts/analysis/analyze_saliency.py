import json
import os
from collections import defaultdict

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

def analyze_saliency_distribution(input_file):
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # 1. 自动提取 details 列表
    samples = raw_data.get('details', raw_data) if isinstance(raw_data, dict) else raw_data

    # 2. 初始化统计变量
    # stats[saliency][label] = count
    stats = defaultdict(lambda: defaultdict(int))
    totals = defaultdict(int)
    
    # 新增：用于计算 Probe Accuracy 和 Resilience
    known_counts = defaultdict(int)  # pred_base 答对的数量
    resilient_counts = defaultdict(int) # pred_base 答对且 pred_rag 也坚持了 gold 的数量

    for entry in samples:
        if not isinstance(entry, dict): continue
            
        saliency = entry.get('saliency', 'Medium')
        label = entry.get('label', 'Uncertain/Other')
        is_known = entry.get('is_known_by_model', False)
        
        # 基础行为统计
        stats[saliency][label] += 1
        totals[saliency] += 1
        
        # 知识主权统计 (Resilience)
        if is_known:
            known_counts[saliency] += 1
            if label == "Persistence":
                resilient_counts[saliency] += 1

    # 3. 打印结果表格
    print("\n" + "="*95)
    header = f"{'Saliency Tier':<14} | {'Base Acc':<10} | {'Resilience':<12} | {'Persistence':<12} | {'Adherence':<11} | {'Uncertain':<11} | {'N':<6}"
    print(header)
    print("-" * 95)

    tiers = ['High', 'Medium', 'Low']
    for tier in tiers:
        n = totals[tier]
        if n == 0:
            print(f"{tier:<14} | {'N/A':<10} | {'N/A':<12} | {'N/A':<12} | {'N/A':<11} | {'N/A':<11} | {0:<6}")
            continue

        # 基础指标
        p_pct = (stats[tier]['Persistence'] / n) * 100
        a_pct = (stats[tier]['Adherence'] / n) * 100
        u_pct = (stats[tier].get('Uncertain/Other', 0) / n) * 100
        
        # Probe 指标
        base_acc = (known_counts[tier] / n) * 100
        # Resilience: 如果本来就知道，有多少比例能挺住？
        resilience = (resilient_counts[tier] / known_counts[tier] * 100) if known_counts[tier] > 0 else 0.0

        row = (f"{tier:<14} | {base_acc:>8.1f}% | {resilience:>10.1f}% | "
               f"{p_pct:>10.1f}% | {a_pct:>9.1f}% | {u_pct:>9.1f}% | {n:<6}")
        print(row)

    print("="*95)
    print("\n💡 Base Acc: Zero-context probe accuracy (Model's internal knowledge).")
    print("💡 Resilience: Percentage of internally known facts defended against RAG conflict.")
    print("💡 Formula: Resilience = (Persistence count / Known count) * 100%\n")

if __name__ == "__main__":
    # 使用包含 pred_base 的最新标注文件
    analyze_saliency_distribution(os.path.join(RESULTS_DIR, "rag_final_labeled_augmented.json"))