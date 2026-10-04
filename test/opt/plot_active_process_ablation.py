import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------
# (a) MIN_ELEMENTS_PER_RANK = 4096
# --------------------------------------------------

sizes = ["100", "1,000", "10,000"]

with_selection = [4.69, 4.70, 7.30]
without_selection = [6.06, 6.22, 7.27]

x1 = np.arange(len(sizes))
width = 0.35

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax1 = axes[0]

bars1 = ax1.bar(
    x1 - width / 2,
    with_selection,
    width,
    label="With Selection"
)

bars2 = ax1.bar(
    x1 + width / 2,
    without_selection,
    width,
    label="Without Selection"
)

ax1.set_xlabel("Number of Elements")
ax1.set_ylabel("Execution Time (ms)")
ax1.set_title("(a) Minimum Elements per Active Rank")
ax1.set_xticks(x1)
ax1.set_xticklabels(sizes)
ax1.legend()

for bars in [bars1, bars2]:
    for bar in bars:
        value = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.2f}",
            ha="center",
            va="bottom"
        )


# --------------------------------------------------
# (b) MAX_ACTIVE_RANKS = 4
# --------------------------------------------------

processes = ["1", "4", "8"]

final = [1.974, 1.129, 1.232]
no_cap = [1.938, 1.127, 1.514]

x2 = np.arange(len(processes))

ax2 = axes[1]

bars3 = ax2.bar(
    x2 - width / 2,
    final,
    width,
    label="Final (Max 4 Active)"
)

bars4 = ax2.bar(
    x2 + width / 2,
    no_cap,
    width,
    label="No Cap"
)

ax2.set_xlabel("Number of MPI Processes")
ax2.set_ylabel("Execution Time (s)")
ax2.set_title("(b) Maximum Active-Rank Cap")
ax2.set_xticks(x2)
ax2.set_xticklabels(processes)
ax2.legend()

for bars in [bars3, bars4]:
    for bar in bars:
        value = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.3f}",
            ha="center",
            va="bottom"
        )


fig.suptitle("Active Process Selection Ablation")

plt.tight_layout()

plt.savefig(
    "test/opt/results/active_process_ablation.png",
    dpi=300,
    bbox_inches="tight"
)

print("Saved: test/opt/results/active_process_ablation.png")
