import matplotlib.pyplot as plt

variants = [
    "std::sort",
    "Radix\n8+8+8+8",
    "Radix\n16+16",
    "Radix\n11+11+10",
    "11+11+10\nFused",
]

times = [
    1696.9,
    133.4,
    131.9,
    127.1,
    114.7,
]

plt.figure(figsize=(8, 5))

bars = plt.bar(variants, times)

plt.ylabel("Local Sort Time (ms)")
plt.xlabel("Sorting Method")
plt.title("Local Sorting Optimization Ablation")

for bar, value in zip(bars, times):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.1f}",
        ha="center",
        va="bottom",
    )

plt.tight_layout()
plt.savefig(
    "test/opt/results/radix_ablation.png",
    dpi=300,
)

print("Saved: test/opt/results/radix_ablation.png")
