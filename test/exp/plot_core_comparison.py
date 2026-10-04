import matplotlib.pyplot as plt

processes = [1, 2, 4]

# Final implementation
big_time = [1.967, 1.250, 1.130]
little_time = [2.424, 1.524, 1.252]

plt.figure(figsize=(7, 5))

plt.plot(
    processes,
    big_time,
    marker="o",
    linewidth=2,
    label="Big Cores"
)

plt.plot(
    processes,
    little_time,
    marker="o",
    linewidth=2,
    label="Little Cores"
)

plt.xlabel("Number of MPI Processes")
plt.ylabel("Execution Time (s)")
plt.title("Big Cores vs. Little Cores")

plt.xticks(processes)
plt.grid(True, alpha=0.3)
plt.legend()

# Label measured execution times
for x, y in zip(processes, big_time):
    plt.text(x, y - 0.08, f"{y:.3f} s", ha="center")

for x, y in zip(processes, little_time):
    plt.text(x, y + 0.05, f"{y:.3f} s", ha="center")

plt.tight_layout()

plt.savefig(
    "test/exp/results/big_vs_little.png",
    dpi=300
)
