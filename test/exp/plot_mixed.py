import matplotlib.pyplot as plt
import numpy as np

processes = [1, 4, 8]

# Median PROF total time
final = [1.974, 1.129, 1.232]
no_cap = [1.938, 1.127, 1.514]

x = np.arange(len(processes))
width = 0.35

plt.figure(figsize=(7, 5))

bars1 = plt.bar(
    x - width / 2,
    final,
    width,
    label="Final (max 4 active ranks)"
)

bars2 = plt.bar(
    x + width / 2,
    no_cap,
    width,
    label="No cap"
)

plt.xlabel("Number of MPI Processes")
plt.ylabel("Execution Time (s)")
plt.title("Mixed-Core Performance")
plt.xticks(x, processes)
plt.legend()

# 顯示每根柱子的數值
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.025,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

plt.ylim(0, 2.2)
plt.tight_layout()

plt.savefig(
    "test/exp/results/mixed_core_performance.png",
    dpi=300
)
