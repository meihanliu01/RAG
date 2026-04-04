import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 设置学术绘图风格
sns.set_theme(style="whitegrid")
# 如果你的电脑没有 Arial，matplotlib 会自动回退到默认字体
plt.rcParams['font.sans-serif'] = ['Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def plot_saliency_behavior():
    # --- 数据填入区 ---
    # 类别标签
    categories = ['High', 'Medium', 'Low']
    
    # 对应的三个指标百分比 (%)
    persistence = [41.0, 40.6, 0.0]
    adherence = [18.8, 8.5, 34.0]
    uncertain = [40.2, 50.9, 66.0]

    x = np.arange(len(categories))  # 标签位置
    width = 0.25  # 柱状图宽度

    # 创建画布
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)

    # --- 绘制柱状图 (修复了颜色代码错误) ---
    # E24A33: 砖红色 (内部记忆)
    # 348ABD: 钢蓝色 (外部文档)
    # 988ED5: 紫灰色 (不确定性)
    rects1 = ax.bar(x - width, persistence, width, label='Persistence (Internal Memory)', 
                    color='#E24A33', edgecolor='black', linewidth=0.8, alpha=0.85)
    rects2 = ax.bar(x, adherence, width, label='Adherence (Context)', 
                    color='#348ABD', edgecolor='black', linewidth=0.8, alpha=0.85)
    rects3 = ax.bar(x + width, uncertain, width, label='Uncertain / Other', 
                    color='#988ED5', edgecolor='black', linewidth=0.8, alpha=0.85)

    # --- 装饰与标签 ---
    ax.set_ylabel('Percentage of Samples (%)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title('Model Behavior under Knowledge Conflict\n(Varying Entity Saliency Tiers)', 
                 fontsize=14, pad=20, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 100)  # 百分比上限
    
    # 设置网格线（仅 y 轴）
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.xaxis.grid(False)

    # 图例设置
    ax.legend(loc='upper right', frameon=True, fontsize=10, shadow=True)

    # --- 数值标注函数 ---
    def autolabel(rects):
        """在每个柱子上方添加百分比数值"""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 5),  # 5点垂直偏移
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    # --- 绘制趋势辅助虚线 ---
    # 重点展示 Persistence 的下降趋势
    ax.plot(x - width, persistence, color='#E24A33', marker='o', markersize=4, 
            linestyle=':', linewidth=1.5, alpha=0.6)
    # 重点展示 Adherence 的回升趋势
    ax.plot(x, adherence, color='#348ABD', marker='s', markersize=4, 
            linestyle=':', linewidth=1.5, alpha=0.6)

    # 布局优化
    plt.tight_layout()
    
    # --- 保存并显示 ---
    output_filename = 'saliency_behavior_analysis_final.png'
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"✅ Success! Plot saved as {output_filename}")
    plt.show()

if __name__ == "__main__":
    plot_saliency_behavior()