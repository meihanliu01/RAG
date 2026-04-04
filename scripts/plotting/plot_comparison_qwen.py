import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 设置学术风格
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

def plot_model_comparison():
    categories = ['High', 'Medium', 'Low']
    
    # --- 填入你的 Llama-3-8B 数据 ---
    llama_persistence = [64.1, 64.1, 0.0]
    llama_adherence = [18.8, 13.4, 0.0]
    
    # --- 填入你的 Qwen2.5-7B 数据 ---
    # 根据你刚才跑出的结果，Qwen 的数据几乎完全一致
    qwen_persistence = [64.1, 64.1, 0.0]
    qwen_adherence = [18.8, 13.4, 0.0]

    x = np.arange(len(categories))
    width = 0.2  # 每个柱子的宽度

    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)

    # 绘制 Persistence (红色系)
    rects1 = ax.bar(x - 1.5*width, llama_persistence, width, label='Llama-3: Persistence', 
                    color='#E24A33', edgecolor='black', hatch='//', alpha=0.8)
    rects2 = ax.bar(x - 0.5*width, qwen_persistence, width, label='Qwen-2.5: Persistence', 
                    color='#FF9999', edgecolor='black', alpha=0.8)

    # 绘制 Adherence (蓝色系)
    rects3 = ax.bar(x + 0.5*width, llama_adherence, width, label='Llama-3: Adherence', 
                    color='#348ABD', edgecolor='black', hatch='\\\\', alpha=0.8)
    rects4 = ax.bar(x + 1.5*width, qwen_adherence, width, label='Qwen-2.5: Adherence', 
                    color='#A6CEE3', edgecolor='black', alpha=0.8)

    # 装饰
    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Entity Saliency Tiers', fontsize=12, fontweight='bold')
    ax.set_title('Cross-Model Behavior Comparison: Llama-3-8B vs. Qwen2.5-7B', 
                 fontsize=14, pad=20, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 100)
    
    # 放置图例
    ax.legend(loc='upper right', ncol=2, frameon=True, shadow=True)

    # 数值标注
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{height}%',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)

    for r in [rects1, rects2, rects3, rects4]:
        autolabel(r)

    # 强调 Low Saliency 的一致性
    ax.annotate('Total Collapse of Sovereignty', xy=(2, 5), xytext=(2, 25),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                ha='center', fontsize=11, color='red', fontweight='bold')

    plt.tight_layout()
    plt.savefig('cross_model_comparison.png', bbox_inches='tight')
    print("🚀 对比图已生成：cross_model_comparison.png")
    plt.show()

if __name__ == "__main__":
    plot_model_comparison()