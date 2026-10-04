#!/usr/bin/env python3
"""分析測資的數值分佈（報告 Methodology → Test cases 的依據）。

用法（課程機器）：
  python3 test/exp/analyze_input.py /srv/nova/scratch/coursedata/pp2026/hw1/10.in [輸出目錄]

輸出：
  * 螢幕：筆數、最小/最大值、平均、標準差、負數比例、±0 數量、整數比例、
          以及「和同範圍均勻分佈比較」的卡方值（64 個 bin）
  * <輸出目錄>/input_distribution_case10.png：直方圖（64 個 bin）加上均勻分佈的期望值
  * <輸出目錄>/input_distribution_case10.txt：同樣的統計數字（方便貼進報告）
"""
import os
import sys

import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "/srv/nova/scratch/coursedata/pp2026/hw1/10.in"
out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), "results")
os.makedirs(out_dir, exist_ok=True)
name = os.path.splitext(os.path.basename(path))[0]

data = np.fromfile(path, dtype="<f4")
n = data.size
finite = np.isfinite(data)
x = data[finite].astype(np.float64)

lo, hi = float(x.min()), float(x.max())
bins = 64
counts, edges = np.histogram(x, bins=bins, range=(lo, hi))
expected = n / bins
chi2 = float(((counts - expected) ** 2 / expected).sum())
cv = float(counts.std() / counts.mean())     # 各 bin 筆數的變異係數，均勻分佈時接近 0

stats = [
    ("file", path),
    ("N (elements)", f"{n:,}"),
    ("non-finite values", f"{int((~finite).sum()):,}"),
    ("min", f"{lo:.6g}"),
    ("max", f"{hi:.6g}"),
    ("min / 2^23", f"{lo / 2**23:.4f}"),
    ("max / 2^23", f"{hi / 2**23:.4f}"),
    ("mean", f"{x.mean():.6g}"),
    ("std", f"{x.std():.6g}"),
    ("std of uniform on [min,max]", f"{(hi - lo) / np.sqrt(12):.6g}"),
    ("negative fraction", f"{(x < 0).mean():.4f}"),
    ("+0 / -0 count", f"{int(((data == 0) & ~np.signbit(data)).sum())} / {int(((data == 0) & np.signbit(data)).sum())}"),
    ("integer-valued fraction", f"{(x == np.round(x)).mean():.4f}"),
    ("distinct values (exact)", f"{np.unique(data[finite]).size:,}"),
    (f"histogram: {bins} bins, min/max count", f"{counts.min():,} / {counts.max():,} (expected {expected:,.0f})"),
    ("histogram: coefficient of variation", f"{cv:.4f}"),
    ("chi-square vs uniform (63 dof)", f"{chi2:.1f}"),
]
width = max(len(k) for k, _ in stats)
text = "\n".join(f"{k:<{width}}  {v}" for k, v in stats)
print(text)
with open(os.path.join(out_dir, f"input_distribution_{name}.txt"), "w") as f:
    f.write(text + "\n")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("（沒有 matplotlib，略過畫圖：python3 -m pip install --user matplotlib）")
    sys.exit(0)

fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
centers = (edges[:-1] + edges[1:]) / 2
ax.bar(centers / 2**23, counts / 1e3, width=(edges[1] - edges[0]) / 2**23 * 0.9,
       color="#4C72B0", label="observed count per bin")
ax.axhline(expected / 1e3, color="#C44E52", linestyle="--", linewidth=1.5,
           label=f"uniform expectation ({expected / 1e3:,.0f}k per bin)")
ax.set_title(f"Value distribution of public test case {name} (N = {n:,})")
ax.set_xlabel(r"value / $2^{23}$")
ax.set_ylabel("count per bin (thousands)")
ax.set_ylim(0, max(counts.max(), expected) / 1e3 * 1.15)
ax.legend(loc="lower center")
ax.text(0.01, 0.97, f"{bins} equal-width bins over [min, max]\nCV of bin counts = {cv:.3f}",
        transform=ax.transAxes, va="top", fontsize=8, color="#555555")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
png = os.path.join(out_dir, f"input_distribution_{name}.png")
fig.savefig(png)
print(f"\n已輸出 {png}")
