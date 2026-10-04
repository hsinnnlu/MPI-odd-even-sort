import matplotlib.pyplot as plt
import numpy as np

# Same-node comparison:
# nova-c05, big cores, 4 MPI processes, case 10, 25 rounds

categories = ["Total", "I/O", "Communication", "Computation"]

# ORI:
# total = max across ranks
# other components = mean across the 4 active ranks
ori = [
    17.390560,
    np.mean([1.600756, 1.599462, 1.599504, 1.599271]),
    np.mean([0.356973, 0.867382, 0.917808, 0.324069]),
    np.mean([13.798653, 14.872881, 14.854795, 14.147991]),
]

# FINAL
final = [
    2.162741,
    np.mean([1.091326, 1.090532, 1.090168, 1.090145]),
    np.mean([0.266049, 0.209898, 0.217928, 0.269916]),
    np.mean([0.774851, 0.861919, 0.854289, 0.773290]),
]

x = np.arange(len(categories))
width = 0.35

plt.figure(figsize=(8, 5))

bars1 = plt.bar(x - width / 2, ori, width, label="Original")
bars2 = plt.bar(x + width / 2, final, width, label="Final")

plt.ylabel("Time (s)")
plt.xlabel("Metric")
plt.title("Original vs. Final Implementation")
plt.xticks(x, categories)
plt.legend()

for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            h,
            f"{h:.2f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

plt.tight_layout()
plt.savefig(
    "test/exp/results/ori_vs_final.png",
    dpi=300
)
