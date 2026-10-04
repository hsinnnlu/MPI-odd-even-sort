import matplotlib.pyplot as plt
import numpy as np

# Controlled strong-scaling experiment
# All configurations use shared NFS.
#
# p1, p2, p4: one node
# p8: two nodes
#
# Each value is the median PROF total time over five trials.

processes = np.array([1, 2, 4, 8])

times = np.array([
    4.204430,   # p1
    3.036093,   # p2
    2.070070,   # p4
    3.843641,   # p8
])

speedup = times[0] / times
ideal_speedup = processes


# ============================================================
# Figure 1: Execution Time
# ============================================================

plt.figure(figsize=(7, 5))

plt.plot(
    processes,
    times,
    marker="o",
    linewidth=2
)

for x, y in zip(processes, times):
    plt.text(
        x,
        y + 0.08,
        f"{y:.3f}",
        ha="center",
        va="bottom"
    )

plt.xticks(processes)

plt.xlabel("Number of MPI Processes")
plt.ylabel("Execution Time (s)")
plt.title("Strong Scaling on Big Cores")

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "test/exp/results/strong_scaling_big.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# Figure 2: Speedup
# ============================================================

plt.figure(figsize=(7, 5))

plt.plot(
    processes,
    speedup,
    marker="o",
    linewidth=2,
    label="Measured Speedup"
)

plt.plot(
    processes,
    ideal_speedup,
    linestyle="--",
    linewidth=1.5,
    label="Ideal Speedup"
)

for x, y in zip(processes, speedup):
    plt.text(
        x,
        y + 0.15,
        f"{y:.2f}x",
        ha="center",
        va="bottom"
    )

plt.xticks(processes)

plt.xlabel("Number of MPI Processes")
plt.ylabel("Speedup")
plt.title("Strong Scaling Speedup on Big Cores")

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "test/exp/results/strong_scaling_speedup.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("Saved:")
print("  test/exp/results/strong_scaling_big.png")
print("  test/exp/results/strong_scaling_speedup.png")
