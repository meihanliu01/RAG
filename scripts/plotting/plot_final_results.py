import matplotlib.pyplot as plt
import numpy as np
import os
import json
import seaborn as sns
from collections import defaultdict

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")

# 设置学术绘图风格
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def load_saliency_stats(result_file):
    """从标注结果文件中自动计算各 Saliency Tier 的行为分布"""
    with open(result_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    samples = data.get('details', data) if isinstance(data, dict) else data
    
    stats = defaultdict(lambda: defaultdict(int))
    totals = defaultdict(int)
    
    for entry in samples:
        if not isinstance(entry, dict):
            continue
        saliency = entry.get('saliency', 'Medium')
        label = entry.get('label', 'Uncertain/Other')
        stats[saliency][label] += 1
        totals[saliency] += 1
    
    categories = ['High', 'Medium', 'Low']
    persistence = []
    adherence = []
    uncertain = []
    sample_sizes = []
    
    for tier in categories:
        n = totals[tier]
        sample_sizes.append(n)
        if n > 0:
            persistence.append(round(stats[tier].get('Persistence', 0) / n * 100, 1))
            adherence.append(round(stats[tier].get('Adherence', 0) / n * 100, 1))
            uncertain.append(round(stats[tier].get('Uncertain/Other', 0) / n * 100, 1))
        else:
            persistence.append(0.0)
            adherence.append(0.0)
            uncertain.append(0.0)
    
    return categories, persistence, adherence, uncertain, sample_sizes


def plot_saliency_behavior():
    # --- 自动从结果文件读取数据 ---
    result_file = os.path.join(RESULTS_DIR, "rag_final_labeled_augmented.json")
    if not os.path.exists(result_file):
        print(f"❌ Error: {result_file} not found. Run the classifier first.")
        return
    
    categories, persistence, adherence, uncertain, sample_sizes = load_saliency_stats(result_file)
    
    print(">>> Data loaded from results:")
    for i, cat in enumerate(categories):
        print(f"  {cat} (N={sample_sizes[i]}): Persist={persistence[i]}%, Adhere={adherence[i]}%, Uncert={uncertain[i]}%")

    x = np.arange(len(categories))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)

    # 绘制柱状图
    rects1 = ax.bar(x - width, persistence, width, label='Persistence (Internal Memory)', 
                    color='#E24A33', edgecolor='black', linewidth=0.8, alpha=0.85)
    rects2 = ax.bar(x, adherence, width, label='Adherence (Context)', 
                    color='#348ABD', edgecolor='black', linewidth=0.8, alpha=0.85)
    rects3 = ax.bar(x + width, uncertain, width, label='Uncertain / Other', 
                    color='#988ED5', edgecolor='black', linewidth=0.8, alpha=0.85)

    # 标签与标题 (含样本量)
    x_labels = [f"{cat}\n(N={n})" for cat, n in zip(categories, sample_sizes)]
    ax.set_ylabel('Percentage of Samples (%)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title('Model Behavior under Knowledge Conflict\n(Varying Entity Saliency Tiers)', 
                 fontsize=14, pad=20, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 100)
    
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.xaxis.grid(False)
    ax.legend(loc='upper right', frameon=True, fontsize=10, shadow=True)

    # 数值标注
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0.5:  # 只标注 > 0.5% 的柱子
                ax.annotate(f'{height}%',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 5),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=10, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    # 趋势辅助虚线
    ax.plot(x - width, persistence, color='#E24A33', marker='o', markersize=4, 
            linestyle=':', linewidth=1.5, alpha=0.6)
    ax.plot(x, adherence, color='#348ABD', marker='s', markersize=4, 
            linestyle=':', linewidth=1.5, alpha=0.6)

    plt.tight_layout()
    
    output_filename = os.path.join(FIGURES_DIR, 'saliency_behavior_analysis_final.png')
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"✅ Success! Plot saved as {output_filename}")
    plt.show()

if __name__ == "__main__":
    plot_saliency_behavior()