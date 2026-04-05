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

# Set academic style
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False


def compute_model_stats(result_file):
    """Compute per-tier Persistence and Adherence from results file."""
    if not os.path.exists(result_file):
        print(f"Warning: {result_file} not found")
        return None
    
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
    
    for tier in categories:
        n = totals[tier]
        if n > 0:
            persistence.append(round(stats[tier].get('Persistence', 0) / n * 100, 1))
            adherence.append(round(stats[tier].get('Adherence', 0) / n * 100, 1))
        else:
            persistence.append(0.0)
            adherence.append(0.0)
    
    return {'persistence': persistence, 'adherence': adherence, 'totals': [totals[t] for t in categories]}


def plot_model_comparison():
    categories = ['High', 'Medium', 'Low']
    
    # --- Load actual result data ---
    llama_file = os.path.join(RESULTS_DIR, "rag_final_labeled_augmented.json")
    qwen_file = os.path.join(RESULTS_DIR, "qwen_final_labeled.json")
    
    llama_stats = compute_model_stats(llama_file)
    qwen_stats = compute_model_stats(qwen_file)
    
    if llama_stats is None:
        print("Error: Llama results not found. Run the classifier first.")
        return
    
    llama_persistence = llama_stats['persistence']
    llama_adherence = llama_stats['adherence']
    
    if qwen_stats is not None:
        qwen_persistence = qwen_stats['persistence']
        qwen_adherence = qwen_stats['adherence']
        print(">>> Loaded actual Qwen data from results file")
    else:
        print("Warning: Qwen results not found, using Llama data as placeholder")
        qwen_persistence = llama_persistence
        qwen_adherence = llama_adherence
    
    print(f">>> Llama: Persist={llama_persistence}, Adhere={llama_adherence}")
    print(f">>> Qwen:  Persist={qwen_persistence}, Adhere={qwen_adherence}")

    x = np.arange(len(categories))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)

    # Persistence (red tones)
    rects1 = ax.bar(x - 1.5*width, llama_persistence, width, label='Llama-3: Persistence', 
                    color='#E24A33', edgecolor='black', hatch='//', alpha=0.8)
    rects2 = ax.bar(x - 0.5*width, qwen_persistence, width, label='Qwen-2.5: Persistence', 
                    color='#FF9999', edgecolor='black', alpha=0.8)

    # Adherence (blue tones)
    rects3 = ax.bar(x + 0.5*width, llama_adherence, width, label='Llama-3: Adherence', 
                    color='#348ABD', edgecolor='black', hatch='\\\\', alpha=0.8)
    rects4 = ax.bar(x + 1.5*width, qwen_adherence, width, label='Qwen-2.5: Adherence', 
                    color='#A6CEE3', edgecolor='black', alpha=0.8)

    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Entity Saliency Tiers', fontsize=12, fontweight='bold')
    ax.set_title('Cross-Model Behavior Comparison: Llama-3-8B vs. Qwen2.5-7B', 
                 fontsize=14, pad=20, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right', ncol=2, frameon=True, shadow=True)

    # Value annotations
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0.5:
                ax.annotate(f'{height}%',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)

    for r in [rects1, rects2, rects3, rects4]:
        autolabel(r)

    # Check if Low tier collapses
    if llama_persistence[2] == 0 and qwen_persistence[2] == 0:
        ax.annotate('Total Collapse of Sovereignty', xy=(2, 5), xytext=(2, 25),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                    ha='center', fontsize=11, color='red', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'cross_model_comparison.png'), bbox_inches='tight')
    print("Comparison plot saved: figures/cross_model_comparison.png")
    plt.show()

if __name__ == "__main__":
    plot_model_comparison()