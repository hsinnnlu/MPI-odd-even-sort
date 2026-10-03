#!/usr/bin/env python3
"""把 test/exp/results/*.csv 整理成報告 5.3 需要的表格。

用法：python3 test/exp/summarize.py [results 目錄]

計算方式（報告的 Performance metrics 要寫清楚）：
  * 每次執行（trial）：
      total   = 所有 rank 的 total 最大值（各 rank 在 MPI_Init 後與 MPI_Finalize 前都有 Barrier，所以幾乎相同）
      io / comm / sync / compute = 「有參與排序的 rank」的平均值，四項相加 = total，可直接畫堆疊圖
                                    （有參與 = 有呼叫 MPI_Sendrecv，或只有 1 個 process）
      read / write = io 中的讀檔與寫檔（含 set_size）時間，同樣是平均值
      bw      = 每個參與 rank 的 Sendrecv 送出量 / 它的 comm 時間，再取平均（MB/s）。
                comm 含等待對方的時間，所以這是「有效頻寬」的下限
      imbal   = 有參與的 rank 中 compute 最大值 / 最小值
  * 每個設定：取 TRIALS 次的中位數
  * speedup = 同版本、同 partition、同測資、1 個 process 的 total 中位數 / 這個設定的 total 中位數
              （2 節點的 big 以單節點 big 的 1 個 process 為基準）
  * wall = srun 整體時間（含 MPI 啟動與程式結束），只當參考
輸出：螢幕上的表格 + results/summary.csv
"""
import csv
import glob
import os
import statistics
import sys
from collections import defaultdict

res_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "results")
prof = defaultdict(list)                   # (case, ver, part, nodes, p, trial) -> [rank rows]
wall = defaultdict(list)                   # (case, ver, part, nodes, p) -> [(wall, ok)]
storage = {}                               # case/part/nodes -> 儲存位置說明


def parse_tag(tag):
    f = tag.split("/")
    case = f[5][1:] if len(f) > 5 else "10"
    return case, f[0], f[1], int(f[2][1:]), int(f[3][1:]), int(f[4][1:])


for path in glob.glob(os.path.join(res_dir, "*.csv")):
    if path.endswith("summary.csv"):
        continue
    for line in open(path):
        f = line.strip().split(",")
        if line.startswith("# storage:"):
            storage[os.path.basename(path)] = line.strip()[2:]
        if not f or f[0] not in ("PROF", "WALL"):
            continue
        case, ver, part, nodes, p, trial = parse_tag(f[1])
        if f[0] == "PROF":
            prof[(case, ver, part, nodes, p, trial)].append({
                "rank": int(f[2]), "host": f[3], "cpu": int(f[4]),
                "total": float(f[5]), "io": float(f[6]), "comm": float(f[7]),
                "sync": float(f[8]), "compute": float(f[9]), "calls": float(f[10]), "MB": float(f[11]),
                "read": float(f[12]) if len(f) > 12 else float("nan"),
                "write": float(f[13]) if len(f) > 13 else float("nan")})
        else:
            wall[(case, ver, part, nodes, p)].append((float(f[2]), f[3]))

per_cfg = defaultdict(lambda: defaultdict(list))
mapping = {}
for (case, ver, part, nodes, p, trial), ranks in prof.items():
    busy = [r for r in ranks if r["calls"] > 0] or ranks
    cfg = (case, ver, part, nodes, p)
    d = per_cfg[cfg]
    d["total"].append(max(r["total"] for r in ranks))
    for k in ("io", "read", "write", "comm", "sync", "compute"):
        d[k].append(statistics.mean(r[k] for r in busy))
    bws = [r["MB"] / r["comm"] for r in busy if r["comm"] > 0 and r["MB"] > 0]
    d["bw"].append(statistics.mean(bws) if bws else float("nan"))
    mins = min(r["compute"] for r in busy)
    d["imbalance"].append(max(r["compute"] for r in busy) / mins if mins > 0 else float("nan"))
    d["busy"].append(len(busy))
    d["MB"].append(sum(r["MB"] for r in ranks))
    if trial == 1:
        mapping[cfg] = " ".join(f"r{r['rank']}:{r['host'].split('.')[0]}/cpu{r['cpu']}"
                                for r in sorted(ranks, key=lambda r: r["rank"]))

med = {cfg: {k: statistics.median(v) for k, v in d.items()} for cfg, d in per_cfg.items()}
order = {"ori": 0, "v4": 1, "final": 2, "final_nocap": 3}
part_order = {"big": 0, "little": 1, "mixed": 2}
cfgs = sorted(med, key=lambda c: (c[0], order.get(c[1], 9), part_order.get(c[2], 9), c[3], c[4]))


def base_total(case, ver, part):
    # final_nocap 在 1 個 process 時和 final 完全相同（只差 rank 上限），缺資料時借用 final 的基準
    for v in (ver, "final") if ver == "final_nocap" else (ver,):
        c = (case, v, part, 1, 1)
        if c in med:
            return med[c]["total"]
    return None


rows = []
hdr = (f"{'case':<5}{'version':<12}{'part':<7}{'N':>2}{'p':>3}{'busy':>5} | {'total':>7}{'io':>7}{'(read':>7}{'write)':>7}"
       f"{'comm':>7}{'sync':>7}{'compute':>8} | {'speedup':>7}{'imbal':>6}{'MB sent':>8}{'bw MB/s':>8}{'wall':>7}{'ok':>6}{'n':>3}")
print(hdr)
print("-" * len(hdr))
for c in cfgs:
    m = med[c]
    b = base_total(c[0], c[1], c[2])
    sp = b / m["total"] if b else float("nan")
    walls = wall.get(c, [])
    wmed = statistics.median(w for w, _ in walls) if walls else float("nan")
    ok = f"{sum(1 for _, s in walls if s == 'OK')}/{len(walls)}"
    print(f"{c[0]:<5}{c[1]:<12}{c[2]:<7}{c[3]:>2}{c[4]:>3}{int(m['busy']):>5} | {m['total']:7.3f}{m['io']:7.3f}"
          f"{m['read']:7.3f}{m['write']:7.3f}{m['comm']:7.3f}{m['sync']:7.3f}{m['compute']:8.3f} | "
          f"{sp:7.2f}{m['imbalance']:6.2f}{m['MB']:8.0f}{m['bw']:8.0f}{wmed:7.2f}{ok:>6}{len(per_cfg[c]['total']):>3}")
    rows.append({"case": c[0], "version": c[1], "partition": c[2], "nodes": c[3], "procs": c[4],
                 "busy_ranks": int(m["busy"]), "total_s": round(m["total"], 4), "io_s": round(m["io"], 4),
                 "io_read_s": round(m["read"], 4), "io_write_s": round(m["write"], 4),
                 "comm_s": round(m["comm"], 4), "sync_s": round(m["sync"], 4), "compute_s": round(m["compute"], 4),
                 "speedup": round(sp, 3), "compute_imbalance": round(m["imbalance"], 3),
                 "sendrecv_MB": round(m["MB"]), "comm_bw_MBps": round(m["bw"]) if m["bw"] == m["bw"] else "",
                 "wall_s": round(wmed, 3), "correct": ok, "trials": len(per_cfg[c]["total"])})

print("\n儲存位置：")
for name, s in sorted(storage.items()):
    print(f"  {name}: {s}")
print("\n各設定的 rank → 節點/CPU 對應（第 1 次執行）：")
for c in cfgs:
    if c in mapping:
        print(f"  case{c[0]} {c[1]:<12}{c[2]:<7}N{c[3]} p{c[4]}: {mapping[c]}")

out = os.path.join(res_dir, "summary.csv")
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["empty"])
    w.writeheader()
    w.writerows(rows)
print(f"\n已寫入 {out}（時間單位：秒，皆為中位數）")
