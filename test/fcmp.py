#!/usr/bin/env python3
"""比對兩個 binary32 檔案。

用法：python3 fcmp.py answer.out my.out
- 先比檔案大小（必須剛好 4N bytes）
- 再逐一比對 float 值；+0.0 和 -0.0 視為相同
- 回傳碼 0 = 相同，1 = 不同（會印出第一個不同的位置）
"""
import struct
import sys

a_path, b_path = sys.argv[1], sys.argv[2]
a = open(a_path, "rb").read()
b = open(b_path, "rb").read()

if len(a) != len(b):
    print(f"size differs: {a_path}={len(a)} bytes, {b_path}={len(b)} bytes")
    sys.exit(1)
if a == b:
    sys.exit(0)

n = len(a) // 4
fa = struct.unpack(f"<{n}f", a)
fb = struct.unpack(f"<{n}f", b)
for i, (x, y) in enumerate(zip(fa, fb)):
    if x != y:   # +0.0 == -0.0 在 Python 裡是 True
        print(f"first mismatch at index {i}: expected {x!r}, got {y!r}")
        sys.exit(1)
sys.exit(0)   # 只差在 +0/-0
