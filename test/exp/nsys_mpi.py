#!/usr/bin/env python3
"""把 nsys_job.sh 產生的 SQLite（每個 rank 一個）整理成表格與 timeline 圖。

用法：
  python3 test/exp/nsys_mpi.py test/exp/results/nsys/<tag> [--zoom START END]
  （START、END 以秒為單位，相對於最早的 MPI 事件；不給就自動放大中間 4% 的時間）

輸出（同一個資料夾）：
  mpi_summary.csv       每個 rank × 每種 MPI 呼叫：次數、總時間、平均時間、傳送量
  timeline_full.png     全部 rank 的 MPI 事件 timeline（空白處 = 不在 MPI 內，也就是計算）
  timeline_zoom.png     放大一小段時間，看得到每個 phase 的 Sendrecv
  msg_size_hist.png     MPI_Sendrecv 的 message size 分佈（若 SQLite 有 size 欄位）

不同版本的 nsys 表格名稱略有不同，這裡會找所有名稱以 MPI 開頭、且有 start/end 欄位的表格。
"""
import csv
import glob
import os
import re
import sqlite3
import sys
from collections import defaultdict

folder = sys.argv[1] if len(sys.argv) > 1 else "."
zoom = None
if "--zoom" in sys.argv:
    i = sys.argv.index("--zoom")
    zoom = (float(sys.argv[i + 1]), float(sys.argv[i + 2]))


def category(name):
    n = name.lower()
    if "file" in n:
        return "I/O"
    if any(k in n for k in ("sendrecv", "send", "recv", "isend", "irecv", "wait")):
        return "Communication"
    return "Synchronization / other MPI"


COLOURS = {"I/O": "#8C8C8C", "Communication": "#DD8452", "Synchronization / other MPI": "#55A868"}


def read_rank(path):
    con = sqlite3.connect(path)
    cur = con.cursor()
    tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    strings = {}
    if "StringIds" in tables:
        strings = dict(cur.execute("SELECT id, value FROM StringIds"))
    events = []
    for t in tables:
        if not t.upper().startswith("MPI"):
            continue
        cols = [r[1] for r in cur.execute(f"PRAGMA table_info({t})")]
        if "start" not in cols or "end" not in cols:
            continue
        name_col = next((c for c in ("textId", "nameId", "name") if c in cols), None)
        size_col = next((c for c in ("size", "messageSize", "bytes", "sendSize") if c in cols), None)
        sel = ["start", "end", name_col or "NULL", size_col or "NULL"]
        for start, end, nm, size in cur.execute(f"SELECT {', '.join(sel)} FROM {t}"):
            if start is None or end is None:
                continue
            label = strings.get(nm, nm) if name_col else t
            events.append((str(label), int(start), int(end), size))
    con.close()
    return events


files = sorted(glob.glob(os.path.join(folder, "rank*.sqlite")),
               key=lambda p: int(re.search(r"rank(\d+)", p).group(1)))
if not files:
    sys.exit(f"在 {folder} 找不到 rank*.sqlite（請確認 nsys export 有成功，見 info.txt）")
ranks = {int(re.search(r"rank(\d+)", f).group(1)): read_rank(f) for f in files}
if not any(ranks.values()):
    sys.exit("SQLite 裡找不到 MPI 事件：請確認 nsys profile 有加 --trace=mpi，或用 Nsight GUI 打開 .nsys-rep 檢查")

t0 = min(e[1] for ev in ranks.values() for e in ev)
t_end = max(e[2] for ev in ranks.values() for e in ev)

# ---- 表格 ----
rows = []
for r, ev in sorted(ranks.items()):
    agg = defaultdict(lambda: [0, 0, 0])
    for name, s, e, size in ev:
        a = agg[name]
        a[0] += 1
        a[1] += e - s
        a[2] += size or 0
    for name, (cnt, ns, by) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
        rows.append({"rank": r, "mpi_call": name, "category": category(name), "count": cnt,
                     "total_ms": round(ns / 1e6, 3), "mean_us": round(ns / cnt / 1e3, 2),
                     "bytes": by})
with open(os.path.join(folder, "mpi_summary.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

span = (t_end - t0) / 1e9
print(f"ranks={len(ranks)}  MPI 事件範圍 {span:.3f} s")
print(f"{'rank':>4} {'category':<28} {'time (s)':>9} {'share':>7}")
for r, ev in sorted(ranks.items()):
    per = defaultdict(int)
    for name, s, e, _ in ev:
        per[category(name)] += e - s
    for c, ns in sorted(per.items()):
        print(f"{r:>4} {c:<28} {ns / 1e9:9.3f} {ns / 1e9 / span:7.1%}")
print(f"已寫入 {os.path.join(folder, 'mpi_summary.csv')}")

# ---- 圖 ----
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
except ImportError:
    sys.exit("（沒有 matplotlib，略過畫圖：python3 -m pip install --user matplotlib）")

info = ""
info_path = os.path.join(folder, "info.txt")
if os.path.exists(info_path):
    first = open(info_path).readline().strip("# \n")
    info = first


def timeline(window, fname, title):
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(ranks) + 1.8), dpi=150)
    for r, ev in sorted(ranks.items()):
        bars = defaultdict(list)
        for name, s, e, _ in ev:
            a, b = (s - t0) / 1e9, (e - t0) / 1e9
            if window and (b < window[0] or a > window[1]):
                continue
            bars[category(name)].append((a, max(b - a, 1e-6)))
        for c, segs in bars.items():
            ax.broken_barh(segs, (r - 0.35, 0.7), facecolors=COLOURS[c])
    ax.set_yticks(sorted(ranks))
    ax.set_yticklabels([f"rank {r}" for r in sorted(ranks)])
    ax.invert_yaxis()
    if window:
        ax.set_xlim(*window)
    ax.set_xlabel("time since first MPI event (s)")
    ax.set_title(title, fontsize=9)
    handles = [Patch(color=v, label=k) for k, v in COLOURS.items()] + \
              [Patch(facecolor="white", edgecolor="#999999", label="outside MPI (local computation)")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4, fontsize=8, frameon=False)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(folder, fname))
    plt.close(fig)
    print(f"已輸出 {os.path.join(folder, fname)}")


timeline(None, "timeline_full.png", f"MPI timeline per rank (Nsight Systems)\n{info}")
if zoom is None:
    mid = span / 2
    zoom = (mid, mid + 0.04 * span)
timeline(zoom, "timeline_zoom.png",
         f"MPI timeline, zoom {zoom[0]:.3f}–{zoom[1]:.3f} s\n{info}")

sizes = [size for ev in ranks.values() for name, s, e, size in ev
         if size and "sendrecv" in name.lower()]
if sizes:
    fig, ax = plt.subplots(figsize=(7, 3.8), dpi=150)
    big = [s for s in sizes if s > 64]
    ax.hist([s / 1e6 for s in big], bins=30, color="#DD8452")
    ax.set_xlabel("MPI_Sendrecv message size (MB)")
    ax.set_ylabel("number of calls")
    ax.set_title(f"Sendrecv message sizes > 64 B ({len(big)} of {len(sizes)} calls)\n{info}", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(folder, "msg_size_hist.png"))
    print(f"已輸出 {os.path.join(folder, 'msg_size_hist.png')}")
else:
    print("（SQLite 沒有 message size 欄位，略過 msg_size_hist.png）")
