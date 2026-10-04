import matplotlib.pyplot as plt

variants = [
    "V0\nFull + Full Merge",
    "V1\nPartial Merge",
    "V2\nBoundary Check",
    "V3\nSelective (Final)",
]

times = [
    130.6,
    87.1,
    44.4,
    43.6,
]

plt.figure(figsize=(8, 5))

bars = plt.bar(variants, times)

plt.ylabel("Compare-Split Time (ms)")
plt.xlabel("Compare-Split Variant")
plt.title("Compare-Split Optimization Ablation")

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
    "test/opt/results/compare_split_ablation.png",
    dpi=300,
)

print("Saved: test/opt/results/compare_split_ablation.png")
