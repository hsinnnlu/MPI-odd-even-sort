#!/usr/bin/env python3
"""產生隨機測資（binary32，little-endian）。

用法：python3 gen.py N output_file [mode] [seed]
mode：
  uniform  均勻分佈 [-1e6, 1e6]（預設）
  wide     指數範圍很大（1e-38 ~ 1e38），含正負號
  dup      大量重複值（只有 -3..3 七種值）
  sorted   已排序
  reverse  反向排序
  zeros    混合 +0.0 和 -0.0 以及少量其他值
"""
import random
import struct
import sys

n = int(sys.argv[1])
path = sys.argv[2]
mode = sys.argv[3] if len(sys.argv) > 3 else "uniform"
random.seed(int(sys.argv[4]) if len(sys.argv) > 4 else n)

if mode == "uniform":
    values = [random.uniform(-1e6, 1e6) for _ in range(n)]
elif mode == "wide":
    values = [random.choice((-1, 1)) * random.random() * 10.0 ** random.randint(-38, 37) for _ in range(n)]
elif mode == "dup":
    values = [float(random.randint(-3, 3)) for _ in range(n)]
elif mode == "sorted":
    values = sorted(random.uniform(-1e6, 1e6) for _ in range(n))
elif mode == "reverse":
    values = sorted((random.uniform(-1e6, 1e6) for _ in range(n)), reverse=True)
elif mode == "zeros":
    values = [random.choice((0.0, -0.0, 0.0, -0.0, 1.5, -2.5)) for _ in range(n)]
else:
    sys.exit(f"unknown mode: {mode}")

with open(path, "wb") as f:
    f.write(struct.pack(f"<{n}f", *values))
