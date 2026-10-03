#!/usr/bin/env python3
"""把 test/exp/results/*.csv 整理成報告 5.3 需要的表格。

用法：python3 test/exp/summarize.py [results 目錄]

計算方式（報告的 Performance metrics 要寫清楚）：
  * 每次執行（trial）：
      total   = 所有 rank 的 total 最大值（各 rank 在 MPI_Init 後與 MPI_Finalize 前都有 Barrier，所以幾乎相同）
      io / comm / sync / compute = 「有參與排序的 rank」的平均值，四項相加 = total，可直接畫堆疊圖
                                    （有參與 = 有呼叫 MPI_Sendrecv，或只有 1 個 process）
      imbalance = 有參與的 rank 中 compute 最大值 / 最小值
  * 每個設定：取 TRIALS 次的中位數
  * speedup = 同版本、同 partition、1 個 process 的 total 中位數 / 這個設定的 total 中位數
  * wall = srun 整體時間（含 MPI 啟動），只當參考
輸出：螢幕上的表格 + results/summary.csv
"""
import csv
import glob
import os
import statistics
import sys
from collections import defaultdict

res_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "results")
prof = defaultdict(list)                   # (ver, part, nodes, p, trial) -> [rank rows]
wall = defaultdict(list)                   # (ver, part, nodes, p) -> [(wall, ok)]

for path in glob.glob(os.path.join(res_dir, "*.csv")):
    if path.endswith("summary.csv"):
        continue
    for line in open(path):
        f = line.strip().split(",")
        if not f or f[0] not in ("PROF", "WALL"):
            continue
        ver, part, nodes, p, trial = f[1].split("/")
        key = (ver, part, int(nodes[1:]), int(p[1:]))
        if f[0] == "PROF":
            prof[key + (int(trial[1:]),)].append({
                "rank": int(f[2]), "host": f[3], "cpu": int(f[4]),
                "total": float(f[5]), "io": float(f[6]), "comm": float(f[7]),
                "sync": float(f[8]), "compute": float(f[9]), "calls": float(f[10]), "MB": float(f[11])})
        else:
            wall[key].append((float(f[2]), f[3]))

per_cfg = defaultdict(lambda: defaultdict(list))
mapping = {}
for (ver, part, nodes, p, trial), ranks in prof.items():
    busy = [r for r in ranks if r["calls"] > 0] or ranks
    cfg = (ver, part, nodes, p)
    d = per_cfg[cfg]
    d["total"].append(max(r["total"] for r in ranks))
    for k in ("io", "comm", "sync", "compute"):
        d[k].append(statistics.mean(r[k] for r in busy))
    mins = min(r["compute"] for r in busy)
    d["imbalance"].append(max(r["compute"] for r in busy) / mins if mins > 0 else float("nan"))
    d["busy"].append(len(busy))
    d["MB"].append(sum(r["MB"] for r in ranks))
    if trial == 1:
        mapping[cfg] = " ".join(f"r{r['rank']}:{r['host'].split('.')[0]}/cpu{r['cpu']}" for r in sorted(ranks, key=lambda r: r["rank"]))

med = {cfg: {k: statistics.median(v) for k, v in d.items()} for cfg, d in per_cfg.items()}
order = {"ori": 0, "v4": 1, "final": 2, "final_nocap": 3}
cfgs = sorted(med, key=lambda c: (order.get(c[0], 9), c[1], c[2], c[3]))

def base_total(ver, part):
    for c in cfgs:
        if c[0] == ver and c[1] == part and c[2] == 1 and c[3] == 1:
            return med[c]["total"]
    return None

rows = []
print(f"{'version':<12}{'part':<8}{'N':>2}{'p':>3}{'busy':>5} | {'total':>7}{'io':>7}{'comm':>7}{'sync':>7}{'compute':>8} | "
      f"{'speedup':>7}{'imbal':>6}{'MB sent':>9}{'wall':>7}{'ok':>6}{'n':>3}")
for c in cfgs:
    m = med[c]
    b = base_total(c[0], c[1]) or (base_total(c[0], "big") if c[2] > 1 else None)
    sp = b / m["total"] if b else float("nan")
    walls = wall.get(c, [])
    wmed = statistics.median(w for w, _ in walls) if walls else float("nan")
    ok = f"{sum(1 for _, s in walls if s == 'OK')}/{len(walls)}"
    print(f"{c[0]:<12}{c[1]:<8}{c[2]:>2}{c[3]:>3}{int(m['busy']):>5} | {m['total']:7.3f}{m['io']:7.3f}{m['comm']:7.3f}"
          f"{m['sync']:7.3f}{m['compute']:8.3f} | {sp:7.2f}{m['imbalance']:6.2f}{m['MB']:9.0f}{wmed:7.2f}{ok:>6}{len(per_cfg[c]['total']):>3}")
    rows.append({"version": c[0], "partition": c[1], "nodes": c[2], "procs": c[3], "busy_ranks": int(m["busy"]),
                 "total_s": round(m["total"], 4), "io_s": round(m["io"], 4), "comm_s": round(m["comm"], 4),
                 "sync_s": round(m["sync"], 4), "compute_s": round(m["compute"], 4), "speedup": round(sp, 3),
                 "compute_imbalance": round(m["imbalance"], 3), "sendrecv_MB": round(m["MB"]),
                 "wall_s": round(wmed, 3), "correct": ok, "trials": len(per_cfg[c]["total"])})

print("\n各設定的 rank → 節點/CPU 對應（第 1 次執行）：")
for c in cfgs:
    if c in mapping:
        print(f"  {c[0]:<12}{c[1]:<8}N{c[2]} p{c[3]}: {mapping[c]}")

out = os.path.join(res_dir, "summary.csv")
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["empty"])
    w.writeheader()
    w.writerows(rows)
print(f"\n已寫入 {out}（時間單位：秒，皆為中位數）")
