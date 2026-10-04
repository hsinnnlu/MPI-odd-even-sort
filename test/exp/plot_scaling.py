import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Controlled strong-scaling experiment
# All configurations use shared NFS.
#
# p1, p2, p4 : 1 node
# p8         : 2 nodes
#
# Each value is the median over five trials.
# ============================================================

processes = np.array([1, 2, 4, 8])
labels = ["1", "2", "4", "8"]

# Median total PROF time
total = np.array([
    4.204430,
    3.036093,
    2.070070,
    3.843641,
])

# Time-profile components
#
# p1/p2/p4:
# mean across active sorting ranks for each trial,
# then median across 5 trials.
#
# p8:
# only ranks 0-3 are active sorting ranks in the final version,
# so the component values use those active ranks.

io_time = np.array([
    2.247809,
    1.800734,
    0.975388,
    2.738277,
])

comm_time = np.array([
    0.000000,
    0.080862,
    0.266825,
    0.265309,
])

sync_time = np.array([
    0.000003,
    0.000464,
    0.014630,
    0.019702,
])

compute_time = np.array([
    1.956618,
    1.154036,
    0.817273,
    0.826274,
])


# ============================================================
# Figure 1: Strong Scaling Time Profile
# ============================================================

x = np.arange(len(processes))

plt.figure(figsize=(8, 5))

plt.bar(
    x,
    compute_time,
    label="Computation"
)

plt.bar(
    x,
    comm_time,
    bottom=compute_time,
    label="Communication"
)

plt.bar(
    x,
    sync_time,
    bottom=compute_time + comm_time,
    label="Synchronization"
)

plt.bar(
    x,
    io_time,
    bottom=compute_time + comm_time + sync_time,
    label="I/O"
)

for i, t in enumerate(total):
    plt.text(
        x[i],
        t + 0.08,
        f"{t:.3f}",
        ha="center",
        va="bottom"
    )

plt.xticks(x, labels)

plt.xlabel("Number of MPI Processes")
plt.ylabel("Time (s)")
plt.title("Strong Scaling Time Profile on Big Cores")

plt.legend()
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    "test/exp/results/strong_scaling_big.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# Figure 2: Strong Scaling Speedup
# ============================================================

speedup = total[0] / total
ideal_speedup = processes

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

for x_value, y_value in zip(processes, speedup):
    plt.text(
        x_value,
        y_value + 0.15,
        f"{y_value:.2f}x",
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
