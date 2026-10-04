import matplotlib.pyplot as plt

# Final implementation
# p=1,2,4: single node, node-local /tmp
# p=8: two nodes, shared NFS
labels = ["1", "2", "4", "8\n(2 nodes)"]
total_time = [1.967, 1.250, 1.130, 3.123]

x = range(len(labels))

plt.figure(figsize=(7, 5))

plt.plot(x, total_time, marker="o", linewidth=2)

plt.xlabel("Number of MPI Processes")
plt.ylabel("Execution Time (s)")
plt.title("Scaling Results on Big Cores")

plt.xticks(x, labels)
plt.grid(True, alpha=0.3)

for i, y in enumerate(total_time):
    plt.text(i, y + 0.06, f"{y:.3f} s", ha="center")

plt.tight_layout()

plt.savefig(
    "test/exp/results/strong_scaling_big.png",
    dpi=300
)
