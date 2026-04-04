import numpy as np
from scipy import stats

# 1. 计算 95% 置信区间
def get_ci(p, n):
    margin = 1.96 * np.sqrt((p * (1 - p)) / n)
    return (p - margin) * 100, (p + margin) * 100

print(f"Uncertain 95% CI: {get_ci(0.747, 1000)}")

# 2. 卡方检验 (Saliency vs Behavior)
# 填入实际个数 [Uncertain, Adherence, Persistence]
obs = np.array([
    [68, 23, 22],  # High (N=113)
    [673, 81, 124], # Medium (N=878)
    [6, 2, 1]      # Low (N=9)
])
chi2, p, dof, ex = stats.chi2_contingency(obs)
print(f"Saliency p-value: {p:.4f}")

# 3. 稳定性计算
persist_rates = [88.0, 88.0, 90.0]
print(f"Mean: {np.mean(persist_rates):.2f}%")
print(f"Std (Sigma): {np.std(persist_rates):.2f}%")