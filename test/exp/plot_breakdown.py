import csv
import matplotlib.pyplot as plt

# Read summarized experiment results
csv_path = "test/exp/results/summary.csv"

rows = []

with open(csv_path, newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        # Only use:
        # final version, big cores, 1 node, p = 1, 2, 4
        if (
            row["version"] == "final"
            and row["partition"] == "big"
            and int(row["nodes"]) == 1
            and int(row["procs"]) in [1, 2, 4]
        ):
            rows.append(row)

# Sort by number of processes
rows.sort(key=lambda r: int(r["procs"]))

processes = [int(r["procs"]) for r in rows]
io = [float(r["io_s"]) for r in rows]
comm = [float(r["comm_s"]) for r in rows]
sync = [float(r["sync_s"]) for r in rows]
compute = [float(r["compute_s"]) for r in rows]

# Bottom positions for stacked bars
bottom_comm = compute

bottom_sync = [
    compute[i] + comm[i]
    for i in range(len(rows))
]

bottom_io = [
    compute[i] + comm[i] + sync[i]
    for i in range(len(rows))
]

# Draw stacked bars
plt.figure(figsize=(7, 5))

plt.bar(processes, compute, label="Computation")
plt.bar(processes, comm, bottom=bottom_comm,
        label="Communication")
plt.bar(processes, sync, bottom=bottom_sync,
        label="Synchronization")
plt.bar(processes, io, bottom=bottom_io,
        label="I/O")

plt.xlabel("Number of MPI Processes")
plt.ylabel("Time (s)")
plt.title("Execution-Time Breakdown on Big Cores")
plt.xticks(processes)

plt.legend()
plt.tight_layout()

# Save figure
plt.savefig(
    "test/exp/results/execution_time_breakdown_big.png",
    dpi=300
)

print("Saved: test/exp/results/execution_time_breakdown_big.png")
