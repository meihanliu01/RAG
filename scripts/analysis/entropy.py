import matplotlib.pyplot as plt
import numpy as np

# 模拟数据
x = np.arange(5)
probs_normal = [0.95, 0.02, 0.01, 0.01, 0.01]  # 无冲突：聚焦
probs_conflict = [0.42, 0.38, 0.10, 0.05, 0.05] # 有冲突：平滑

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.bar(x, probs_normal, color='skyblue')
ax1.set_title("Low Entropy (No Conflict)\nFocus on '1856'")
ax1.set_ylim(0, 1)

ax2.bar(x, probs_conflict, color='salmon')
ax2.set_title("High Entropy (Knowledge Conflict)\nCompetition: '1856' vs '2026'")
ax2.set_ylim(0, 1)

plt.show()