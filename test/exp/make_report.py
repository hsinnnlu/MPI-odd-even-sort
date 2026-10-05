#!/usr/bin/env python3
"""從實驗結果一次產生報告需要的所有圖與 LaTeX 表格（不寫死任何數字）。

用法（repo 根目錄）：
  python3 test/exp/summarize.py                 # 先產生 test/exp/results/summary.csv
  python3 test/exp/make_report.py [--case 10] [--scaling-storage local|nfs|auto] [--out test/exp/report]

讀取：
  test/exp/results/summary.csv     job.sh / job_scaling.sh 的結果（由 summarize.py 整理，皆為 5 次中位數）
  test/opt/results/opt_*.txt       run_opt.sh 的三個優化佐證實驗（取最新的一個檔案）

輸出（--out 目錄，預設 test/exp/report/）：
  fig_breakdown_big.png      final、big、1/2/4 process 的時間組成堆疊圖（I/O / Comm / Sync / Compute）
  fig_scaling.png            strong scaling：總時間與 speedup（含 compute-only speedup、理想線）
  fig_big_vs_little.png      big 與 little 的總時間比較
  fig_scaling_big_little.png 單節點 strong scaling：big 與 little 的 speedup 與各項時間
  fig_mixed.png              mixed 1/4/8 的時間組成（big + little 混合節點）
  fig_ori_vs_final.png       最初版本 vs final（對數座標，標出加速倍數）
  fig_compression.png        2 節點：跨節點壓縮開 / 關（final vs final_nocomp）的時間組成與傳送量
  fig_opt_radix.png          Optimization 1：round 0 的 local sort 各做法時間
  fig_opt_radix24.png        Optimization 1：hash 之後（24-bit key）的 hash + local sort 各做法時間
  fig_opt_split.png          Optimization 2：compare-split 四種做法的時間與傳送量
  tables.tex                 以上全部的 LaTeX 表格（booktabs；\\usepackage{booktabs}）
  （某項資料不存在時跳過那張圖/表並印出原因）

儲存位置：summary.csv 的 storage 欄位區分 local（node-local /tmp）與 nfs。
  * breakdown、big vs little、mixed、ori vs final 只用 local（規定的正式實驗）
  * scaling 依 --scaling-storage：
      nfs    （預設，報告採用）1/2/4/8 全部用 nfs（submit_all.sh 的 bignfs + big2n）
      local  1/2/4 用 local；8 process（2 節點）只能用 nfs，圖上加註
      auto   nfs 有 1/2/4/8 就用 nfs，否則用 local
"""
import argparse
from collections import defaultdict
import csv
import glob
import os
import re
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--case", default="10")
ap.add_argument("--summary", default="test/exp/results/summary.csv")
ap.add_argument("--opt", default=None, help="run_opt.sh 的結果檔（預設取 test/opt/results/opt_*.txt 最新的）")
ap.add_argument("--scaling-storage", default="nfs", choices=["auto", "local", "nfs"])
ap.add_argument("--out", default="test/exp/report")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("需要 matplotlib：python3 -m pip install --user matplotlib")

C_IO, C_COMM, C_SYNC, C_COMP = "#8C8C8C", "#DD8452", "#55A868", "#4C72B0"
tex = ["% 由 test/exp/make_report.py 自動產生；需要 \\usepackage{booktabs}", ""]


def save(fig, name, top=1.0):
    path = os.path.join(args.out, name)
    # 有 footnote（fig.text）時，留出底部空間，避免和 x 軸標籤重疊；top < 1 時上方留給整張圖共用的圖例
    fig.tight_layout(rect=(0, 0.08 if fig.texts else 0, 1, top))
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  圖：{path}")


def table(caption, label, header, rows, align=None):
    align = align or "l" + "r" * (len(header) - 1)
    tex.extend([
        "\\begin{table}[htbp]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{align}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"])
    tex.extend(" & ".join(str(x) for x in r) + " \\\\" for r in rows)
    tex.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    print(f"  表：{label}")


COMPONENTS = (("compute_s", "Computation", C_COMP), ("comm_s", "Communication", C_COMM),
              ("sync_s", "Synchronization", C_SYNC), ("io_s", "I/O", C_IO))


# 圖下方的共同註解：定義衍生指標，讓每張圖單獨看也看得懂
PROFILE_NOTE = ("Median of 5 trials. Total = max over ranks; I/O, Comm. (MPI_Sendrecv), Sync. (MPI_Allreduce) = mean over "
                "sorting ranks; Computation = Total - I/O - Comm. - Sync.")
SPEEDUP_NOTE = "Speedup(p) = T(1) / T(p); compute-only speedup uses computation time only. Median of 5 trials."


def footnote(fig, text):
    fig.text(0.5, 0.005, text, ha="center", va="bottom", fontsize=6.5, color="#555555", wrap=True)


def stacked_profile(ax, rows, xlabels, err=True):
    """rows = [row]：畫 Computation / Communication / Synchronization / I/O 堆疊圖，
    上方標 total（中位數），誤差線 = 5 次中 total 的最小值～最大值。"""
    xs = list(range(len(rows)))
    bottom = [0.0] * len(rows)
    for key, lab, col in (("compute_s", "Computation", C_COMP), ("comm_s", "Communication", C_COMM),
                          ("sync_s", "Synchronization", C_SYNC), ("io_s", "I/O", C_IO)):
        vals = [r[key] for r in rows]
        ax.bar(xs, vals, bottom=bottom, color=col, label=lab, width=0.6)
        bottom = [b + v for b, v in zip(bottom, vals)]
    tops = [r["total_s"] for r in rows]
    if err and all("total_min_s" in r for r in rows):
        lo = [r["total_s"] - r["total_min_s"] for r in rows]
        hi = [r["total_max_s"] - r["total_s"] for r in rows]
        ax.errorbar(xs, tops, yerr=[lo, hi], fmt="none", ecolor="black", capsize=4, linewidth=1,
                    label="total: min-max of 5 trials")
        tops = [r["total_max_s"] for r in rows]
    for x, r, t, h in zip(xs, rows, tops, bottom):
        ax.text(x, max(t, h), f"{r['total_s']:.2f} s", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs, xlabels)
    ax.set_ylabel("time (s)")
    ax.set_ylim(0, max(max(bottom), max(tops)) * 1.4)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7, loc="upper left", ncol=2)


def skip(what, why):
    print(f"  略過 {what}：{why}")


# =============================================================================
# summary.csv
# =============================================================================
R = {}
if os.path.exists(args.summary):
    for r in csv.DictReader(open(args.summary)):
        if r.get("case", args.case) != args.case:
            continue
        key = (r.get("storage", "local"), r["version"], r["partition"], int(r["nodes"]), int(r["procs"]))
        R[key] = {k: (float(v) if k != "cpus" and re.fullmatch(r"-?[\d.]+(e-?\d+)?|nan", v or "x") else v)
                  for k, v in r.items()}
else:
    print(f"找不到 {args.summary}（先跑 python3 test/exp/summarize.py），只處理 opt 結果")


def get(store, ver, part, nodes, p):
    return R.get((store, ver, part, nodes, p))


def series(store, ver, part, nodes, procs):
    """回傳 [(p, row)]，只包含有資料的 p。"""
    return [(p, get(store, ver, part, nodes, p)) for p in procs if get(store, ver, part, nodes, p)]


def f3(x):
    return f"{x:.3f}"


print(f"[summary] case {args.case}")

# ---- 1. 時間組成（final / big / local / 1 節點） ----
s = series("local", "final", "big", 1, [1, 2, 4])
if s:
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    stacked_profile(ax, [r for _, r in s], [str(p) for p, _ in s])
    ax.set_xlabel("number of MPI processes (big partition, 1 node)")
    ax.set_title(f"Time profile, final version\n(case {args.case}, N = 23,987,513, 25 rounds, node-local /tmp)",
                 fontsize=10)
    footnote(fig, PROFILE_NOTE)
    save(fig, "fig_breakdown_big.png")
    table(f"Time profile of the final version on big cores (case {args.case}, node-local /tmp, median of 5 trials, "
          "seconds). Range = min--max of the total time over the 5 trials.",
          "tab:breakdown",
          ["Processes", "Total", "Range", "I/O (read / write)", "Comm.", "Sync.", "Compute", "Speedup"],
          [[p, f3(r["total_s"]), f"{r['total_min_s']:.2f}--{r['total_max_s']:.2f}",
            f"{r['io_s']:.3f} ({r['io_read_s']:.3f} / {r['io_write_s']:.3f})",
            f3(r["comm_s"]), f3(r["sync_s"]), f3(r["compute_s"]), f"{r['speedup']:.2f}"] for p, r in s])
else:
    skip("時間組成", "沒有 local/final/big/N1 的結果（跑 submit_all.sh 的 big job）")

# ---- 2. Strong scaling ----
nfs_s = series("nfs", "final", "big", 1, [1, 2, 4]) + series("nfs", "final", "big", 2, [8])
loc_s = series("local", "final", "big", 1, [1, 2, 4])
mode = args.scaling_storage
if mode == "auto":
    mode = "nfs" if len(nfs_s) == 4 else "local"
if mode == "nfs":
    sc = [(p, r, "nfs") for p, r in nfs_s]
else:
    sc = [(p, r, "local") for p, r in loc_s] + [(8, r, "nfs") for _, r in series("nfs", "final", "big", 2, [8])]
if len(sc) >= 2 and sc[0][0] == 1:
    base, base_c = sc[0][1]["total_s"], sc[0][1]["compute_s"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.6))
    ps = [p for p, _, _ in sc]
    stacked_profile(a1, [r for _, r, _ in sc],
                    [f"{p}\n({int(r['nodes'])} node{'s' if r['nodes'] > 1 else ''})" for p, r, _ in sc])
    a1.set_xlabel("number of MPI processes (big partition)")
    a1.set_title("Time profile")
    a2.plot(ps, ps, "k--", alpha=0.5, label="ideal")
    a2.plot(ps, [base / r["total_s"] for _, r, _ in sc], "o-", color=C_COMP, label="total speedup")
    a2.plot(ps, [base_c / r["compute_s"] for _, r, _ in sc], "s--", color="#8172B3", label="compute-only speedup")
    for p, r, st in sc:
        if st != sc[0][2]:
            a2.annotate("2 nodes, NFS", (p, base / r["total_s"]), textcoords="offset points",
                        xytext=(-40, 25), fontsize=8, arrowprops=dict(arrowstyle="->", color="#C44E52"), color="#C44E52")
    a2.set_xscale("log", base=2)
    a2.set_xticks(ps, [str(p) for p in ps])
    a2.set_xlabel("number of MPI processes")
    a2.set_ylabel("speedup over 1 process")
    a2.set_title("Strong scaling")
    a2.grid(alpha=0.3)
    a2.legend(fontsize=8)
    footnote(fig, PROFILE_NOTE + " " + SPEEDUP_NOTE.replace(" Median of 5 trials.", ""))
    note = "all runs on shared NFS" if mode == "nfs" else "1-4 procs: node-local /tmp; 8 procs: 2 nodes, shared NFS"
    fig.suptitle(f"Strong scaling, final version, big partition (case {args.case}, 25 rounds; {note})", fontsize=10)
    save(fig, "fig_scaling.png")
    table(f"Strong scaling of the final version on big cores (case {args.case}; {note}). "
          "Speedup is relative to 1 process; compute-only speedup excludes I/O, communication and synchronization.",
          "tab:scaling",
          ["Processes", "Nodes", "Storage", "Total (s)", "Range (s)", "I/O (s)", "Comm. (s)", "Compute (s)",
           "Speedup", "Compute speedup"],
          [[p, int(r["nodes"]), "/tmp" if st == "local" else "NFS", f3(r["total_s"]),
            f"{r['total_min_s']:.2f}--{r['total_max_s']:.2f}", f3(r["io_s"]), f3(r["comm_s"]), f3(r["compute_s"]),
            f"{base / r['total_s']:.2f}", f"{base_c / r['compute_s']:.2f}"]
           for p, r, st in sc])
else:
    skip("strong scaling", f"storage={mode} 沒有包含 1 process 的 final/big 結果")

# ---- 3. big vs little ----
b, l = dict(series("local", "final", "big", 1, [1, 2, 4])), dict(series("local", "final", "little", 1, [1, 2, 4]))
ps = sorted(set(b) & set(l))
if ps:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    w = 0.38
    xs = list(range(len(ps)))
    for off, d, part, hatch in ((-w / 2, b, "big", None), (w / 2, l, "little", "//")):
        bottom = [0.0] * len(ps)
        for key, lab, col in COMPONENTS:
            vals = [d[p][key] for p in ps]
            ax.bar([x + off for x in xs], vals, w, bottom=bottom, color=col, hatch=hatch, edgecolor="white",
                   linewidth=0.5, label=lab if part == "big" else None)
            bottom = [bb + v for bb, v in zip(bottom, vals)]
        ax.errorbar([x + off for x in xs], [d[p]["total_s"] for p in ps],
                    yerr=[[d[p]["total_s"] - d[p]["total_min_s"] for p in ps],
                          [d[p]["total_max_s"] - d[p]["total_s"] for p in ps]],
                    fmt="none", ecolor="black", capsize=3, linewidth=1)
        for x, p, h in zip(xs, ps, bottom):
            ax.text(x + off, max(h, d[p]["total_max_s"]), f"{part}\n{d[p]['total_s']:.2f} s", ha="center",
                    va="bottom", fontsize=7)
    ax.set_xticks(xs, [str(p) for p in ps])
    ax.set_xlabel("number of MPI processes (1 node; left bar = big partition, right hatched bar = little partition)")
    ax.set_ylabel("time (s)")
    ax.set_title(f"Big vs little partition, final version\n(case {args.case}, 25 rounds, node-local /tmp)", fontsize=10)
    ax.set_ylim(0, max(max(d[p]["total_max_s"] for p in ps) for d in (b, l)) * 1.4)
    ax.legend(fontsize=7, ncol=4, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    footnote(fig, PROFILE_NOTE + " Error bar = min-max total of 5 trials.")
    save(fig, "fig_big_vs_little.png")
    table(f"Big versus little partition, final version (case {args.case}, node-local /tmp, median of 5 trials, "
          "seconds).", "tab:big-little",
          ["Partition", "Processes", "Total", "Compute", "Comm.", "Sync.", "I/O", "Little / big (total)"],
          [[part, p, f3(d[p]["total_s"]), f3(d[p]["compute_s"]), f3(d[p]["comm_s"]), f3(d[p]["sync_s"]),
            f3(d[p]["io_s"]), f"{l[p]['total_s'] / b[p]['total_s']:.2f}" if part == "little" else "--"]
           for p in ps for part, d in (("big", b), ("little", l))], "lrrrrrrr")
else:
    skip("big vs little", "缺少 local 的 big 或 little 結果")

# ---- 3b. 單節點 strong scaling：big vs little ----
if ps and 1 in ps:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
    styles = (("big", b, "-", "o"), ("little", l, "--", "s"))
    a1.plot(ps, ps, color="black", linestyle=":", alpha=0.6, label="ideal")
    for part, d, ls, mk in styles:
        a1.plot(ps, [d[1]["total_s"] / d[p]["total_s"] for p in ps], ls, marker=mk, color=C_COMP,
                label=f"{part}: total speedup")
        a1.plot(ps, [d[1]["compute_s"] / d[p]["compute_s"] for p in ps], ls, marker=mk, color="#8172B3",
                label=f"{part}: compute-only speedup")
    a1.set_xscale("log", base=2)
    a1.set_xticks(ps, [str(p) for p in ps])
    a1.set_xlabel("number of MPI processes (1 node)")
    a1.set_ylabel("speedup over 1 process of the same partition")
    a1.set_title("Speedup")
    a1.grid(alpha=0.3)
    a1.legend(fontsize=7, loc="upper left")
    for part, d, ls, mk in styles:
        for key, lab, col in COMPONENTS:
            a2.plot(ps, [d[p][key] for p in ps], ls, marker=mk, color=col,
                    label=f"{part}: {lab}")
    a2.set_xscale("log", base=2)
    a2.set_xticks(ps, [str(p) for p in ps])
    a2.set_xlabel("number of MPI processes (1 node)")
    a2.set_ylabel("time (s)")
    a2.set_title("Time profile per component")
    a2.grid(alpha=0.3)
    a2.legend(fontsize=6.5, ncol=2, loc="upper left")
    a2.set_ylim(0, max(d[p]["compute_s"] for _, d, _, _ in styles for p in ps) * 1.45)
    fig.suptitle(f"Single-node strong scaling, big vs little partition, final version "
                 f"(case {args.case}, 25 rounds, node-local /tmp)", fontsize=10)
    footnote(fig, PROFILE_NOTE + " Speedup(p) = T(1) / T(p) within the same partition (solid = big, dashed = little).")
    save(fig, "fig_scaling_big_little.png")
    table(f"Single-node strong scaling on the big and little partitions (final version, case {args.case}, node-local "
          "/tmp, median of 5 trials). Efficiency = speedup / processes.", "tab:scaling-big-little",
          ["Partition", "Processes", "Total (s)", "Compute (s)", "Comm. (s)", "Sync. (s)", "I/O (s)", "Speedup",
           "Efficiency", "Compute speedup"],
          [[part, p, f3(d[p]["total_s"]), f3(d[p]["compute_s"]), f3(d[p]["comm_s"]), f3(d[p]["sync_s"]),
            f3(d[p]["io_s"]), f"{d[1]['total_s'] / d[p]['total_s']:.2f}",
            f"{d[1]['total_s'] / d[p]['total_s'] / p * 100:.0f}\\%", f"{d[1]['compute_s'] / d[p]['compute_s']:.2f}"]
           for part, d, _, _ in styles for p in ps], "lrrrrrrrrr")
else:
    skip("single-node scaling big vs little", "缺少 big 或 little 的 1 process 結果")

# ---- 4. mixed ----
mx = series("local", "final", "mixed", 1, [1, 4, 8])


def cpu_set(part):
    """big / little 實驗中用到的 CPU 編號，用來判斷 mixed 節點上每個 rank 是 big 還是 little core。"""
    out = set()
    for (st, v, pt, nodes, p), r in R.items():
        if pt == part and nodes == 1 and isinstance(r.get("cpus"), str):
            out |= {int(c) for c in r["cpus"].split()}
    return out


def composition(r):
    big_cpus, little_cpus = cpu_set("big"), cpu_set("little")
    cpus = [int(c) for c in str(r.get("cpus", "")).split()] if isinstance(r.get("cpus"), str) else []
    if not cpus or not big_cpus or not little_cpus:
        return ""
    nb = sum(c in big_cpus for c in cpus)
    nl = sum(c in little_cpus for c in cpus)
    return f"{nb} big + {nl} little"


if mx:
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    stacked_profile(ax, [r for _, r in mx], [f"{p}\n({composition(r)})" if composition(r) else str(p) for p, r in mx])
    ax.set_xlabel("number of MPI processes (mixed partition, 1 node with 4 big + 4 little cores)")
    ax.set_title(f"Heterogeneous whole-node run, final version\n(case {args.case}, 25 rounds, node-local /tmp)",
                 fontsize=10)
    footnote(fig, PROFILE_NOTE + " Core types per rank are taken from the CPU ids used in the big / little runs.")
    save(fig, "fig_mixed.png")
    table(f"Mixed partition, final version (case {args.case}, node-local /tmp, median of 5 trials, seconds). "
          "Imbalance = max / min computation time among ranks. This series is not part of the homogeneous "
          "big-core speedup curve because its core composition changes with the process count.", "tab:mixed",
          ["Processes", "Cores used", "Total", "Range", "I/O", "Comm.", "Sync.", "Compute", "Imbalance"],
          [[p, composition(r) or "--", f3(r["total_s"]), f"{r['total_min_s']:.2f}--{r['total_max_s']:.2f}",
            f3(r["io_s"]), f3(r["comm_s"]), f3(r["sync_s"]), f3(r["compute_s"]),
            f"{r['compute_imbalance']:.2f}"] for p, r in mx])
else:
    skip("mixed", "沒有 local 的 mixed 結果")

# ---- 5. ori vs final ----
pairs = []
for part in ("big", "little"):
    for p in (1, 2, 4):
        o, f = get("local", "ori", part, 1, p), get("local", "final", part, 1, p)
        if o and f:
            pairs.append((part, p, o, f))
if pairs:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    labels = [f"{part}\np = {p}" for part, p, _, _ in pairs]
    for ax, idx, name in ((axes[0], 2, "Original version"), (axes[1], 3, "Final version")):
        stacked_profile(ax, [pr[idx] for pr in pairs], labels)
        ax.set_title(name)
        ax.set_xlabel("partition and number of MPI processes (1 node)")
    for x, (_, _, o, f) in enumerate(pairs):
        axes[1].text(x, f["total_s"] * 0.5, f"{o['total_s'] / f['total_s']:.1f}x\nfaster", ha="center",
                     va="center", fontsize=7, color="white")
    fig.suptitle(f"Original vs final version (case {args.case}, 25 rounds, node-local /tmp; note different y scales)",
                 fontsize=10)
    footnote(fig, PROFILE_NOTE + " Original: std::sort + full-block exchange + full merge. "
                  "Label in the right panel = T(original) / T(final).")
    save(fig, "fig_ori_vs_final.png")
    table(f"Original versus final version (case {args.case}, node-local /tmp, median of 5 trials, seconds).",
          "tab:ori-final",
          ["Partition", "Processes", "Version", "Total", "Compute", "Comm.", "Sync.", "I/O", "Sendrecv calls / rank",
           "Sent (MB)", "Speedup"],
          [[part, p, name, f3(r["total_s"]), f3(r["compute_s"]), f3(r["comm_s"]), f3(r["sync_s"]), f3(r["io_s"]),
            f"{r.get('sendrecv_calls_per_rank', 0):.0f}", f"{r['sendrecv_MB']:.0f}",
            "1.0" if name == "original" else f"{o['total_s'] / f['total_s']:.1f}$\\times$"]
           for part, p, o, f in pairs for name, r in (("original", o), ("final", f))], "lrlrrrrrrrr")
else:
    skip("ori vs final", "缺少 local 的 ori 或 final 結果")

# ---- 6. 通訊量與有效頻寬 ----
rows = []
for (st, v, part, nodes, p), r in sorted(R.items(), key=lambda kv: (kv[0][1], kv[0][2], kv[0][3], kv[0][4])):
    if p > 1 and v in ("ori", "final", "final_nocomp") and r.get("sendrecv_MB"):
        bw = r.get("comm_bw_MBps")
        rows.append([v.replace("_", "\\_"), part, nodes, p, "/tmp" if st == "local" else "NFS",
                     f"{r['sendrecv_MB']:.0f}", f3(r["comm_s"]), f"{bw:.0f}" if isinstance(bw, float) else "--"])
if rows:
    table(f"Communication volume (sum of MPI\\_Sendrecv payload over all ranks) and effective bandwidth "
          f"(case {args.case}). Communication time includes waiting for the partner, so the bandwidth is a lower "
          "bound.", "tab:comm",
          ["Version", "Partition", "Nodes", "Processes", "Storage", "Sent (MB)", "Comm. (s)", "Eff. BW (MB/s)"],
          rows, "lllrlrrr")

# ---- 7. 跨節點壓縮（2 節點：final vs final_nocomp） ----
comp = []
for (st, v, part, nodes, p), r in sorted(R.items(), key=lambda kv: (kv[0][2], kv[0][4])):
    if v == "final" and nodes > 1 and (st, "final_nocomp", part, nodes, p) in R:
        comp.append((part, nodes, p, R[(st, "final_nocomp", part, nodes, p)], r))
if comp:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    labels, xs = [], []
    for i, (part, nodes, p, off, on) in enumerate(comp):
        for j, (r, name) in enumerate(((off, "off"), (on, "on"))):
            x = i * 3 + j
            xs.append(x)
            labels.append(f"{part} {p}p\ncompress {name}")
            bottom = 0.0
            for key, lab, col in (("compute_s", "Computation", C_COMP), ("comm_s", "Communication", C_COMM),
                                  ("sync_s", "Synchronization", C_SYNC), ("io_s", "I/O", C_IO)):
                a1.bar(x, r[key], bottom=bottom, color=col, label=lab if x == 0 else None, width=0.8)
                bottom += r[key]
            a1.text(x, bottom, f"{r['total_s']:.2f}", ha="center", va="bottom", fontsize=8)
            a2.bar(x, r["node_edge_MB"], color="#8C8C8C" if name == "off" else C_COMP, width=0.8)
            a2.text(x, r["node_edge_MB"], f"{r['node_edge_MB']:.0f}", ha="center", va="bottom", fontsize=8)
    for ax, yl, t in ((a1, "time (s)", "Time profile"),
                      (a2, "MB sent by the two ranks at the node boundary", "Data sent by the node-boundary ranks")):
        ax.set_xticks(xs, labels, fontsize=7)
        ax.set_ylabel(yl)
        ax.set_title(t)
        ax.grid(axis="y", alpha=0.3)
    a1.legend(fontsize=7, loc="upper right")
    a1.set_ylim(0, max(max(off["total_s"], on["total_s"]) for *_, off, on in comp) * 1.35)
    fig.suptitle(f"Cross-node compression on two nodes (case {args.case}, shared NFS)", fontsize=10)
    save(fig, "fig_compression.png")
    table(f"Effect of compressing the data exchanged between ranks on different nodes (two nodes, case {args.case}, "
          "shared NFS, median of 5 trials). Only the pair of ranks at the node boundary is compressed; "
          "boundary sent = data sent by these two ranks (to both of their neighbours).", "tab:compression",
          ["Partition", "Processes", "Compression", "Total (s)", "Comm. (s)", "Sync. (s)", "Boundary sent (MB)",
           "All ranks sent (MB)", "Speedup"],
          [row for part, nodes, p, off, on in comp for row in (
              [part, p, "off", f3(off["total_s"]), f3(off["comm_s"]), f3(off["sync_s"]),
               f"{off['node_edge_MB']:.0f}", f"{off['sendrecv_MB']:.0f}", "1.00"],
              [part, p, "on", f3(on["total_s"]), f3(on["comm_s"]), f3(on["sync_s"]),
               f"{on['node_edge_MB']:.0f}", f"{on['sendrecv_MB']:.0f}", f"{off['total_s'] / on['total_s']:.2f}"])],
          "lrlrrrrrr")
else:
    skip("跨節點壓縮", "沒有 2 節點的 final 與 final_nocomp 結果（submit_all.sh 的 big2n、mixed2n）")

# ---- 8. 每個 rank 的時間組成：誰在等誰、負載是否平衡 ----
pr_path = os.path.join(os.path.dirname(args.summary), "per_rank.csv")
PR = defaultdict(list)
if os.path.exists(pr_path):
    for r in csv.DictReader(open(pr_path)):
        if r["case"] != args.case:
            continue
        key = (r["storage"], r["version"], r["partition"], int(r["nodes"]), int(r["procs"]))
        PR[key].append({k: (float(v) if k.endswith("_s") or k in ("sendrecv_MB", "sendrecv_calls") else v)
                        for k, v in r.items()})
cases_pr = [(("local", "final", "big", 1, 4), "big, 4 processes, 1 node (/tmp)"),
            (("local", "final", "mixed", 1, 8), "mixed, 8 processes, 1 node (/tmp)"),
            (("nfs", "final", "big", 2, 8), "big, 8 processes, 2 nodes (NFS)"),
            (("nfs", "final", "mixed", 2, 16), "mixed, 16 processes, 2 nodes (NFS)")]
cases_pr = [(k, t) for k, t in cases_pr if k in PR]
if cases_pr:
    big_cpus, little_cpus = cpu_set("big"), cpu_set("little")
    fig, axes = plt.subplots(len(cases_pr), 1, figsize=(10, 2.6 * len(cases_pr) + 0.8), squeeze=False)
    rows_tex = []
    for ax, (key, title) in zip(axes[:, 0], cases_pr):
        rs = sorted(PR[key], key=lambda r: int(r["rank"]))
        xs = list(range(len(rs)))
        bottom = [0.0] * len(rs)
        for k, lab, col in COMPONENTS:
            vals = [r[k] for r in rs]
            ax.bar(xs, vals, bottom=bottom, color=col, label=lab, width=0.7)
            bottom = [bb + v for bb, v in zip(bottom, vals)]

        def core(r):
            c = int(r["cpu"])
            return "B" if c in big_cpus and key[2] == "mixed" else ("L" if c in little_cpus and key[2] == "mixed" else "")
        hosts = []
        for r in rs:
            if r["host"] not in hosts:
                hosts.append(r["host"])
        ax.set_xticks(xs, [f"{int(r['rank'])}{core(r)}\nn{hosts.index(r['host'])}" for r in rs], fontsize=7)
        ax.set_ylabel("time (s)")
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        comp = [r["compute_s"] for r in rs]
        wait = [r["comm_s"] + r["sync_s"] for r in rs]
        rows_tex.append([title.replace("/tmp", "\\texttt{/tmp}"), f"{min(comp):.3f}", f"{max(comp):.3f}",
                         f"{max(comp) / min(comp):.2f}", f"{min(wait):.3f}", f"{max(wait):.3f}",
                         f"{min(r['sendrecv_MB'] for r in rs):.0f}--{max(r['sendrecv_MB'] for r in rs):.0f}"])
    h_, l_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(h_, l_, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.965))
    axes[-1, 0].set_xlabel("rank (B = big core, L = little core on mixed nodes; n0 / n1 = node)")
    fig.suptitle(f"Per-rank time profile, final version (case {args.case}, 25 rounds)", fontsize=10, y=0.995)
    footnote(fig, "Median of 5 trials per rank. Communication = MPI_Sendrecv incl. waiting for the partner; "
                  "Synchronization = MPI_Allreduce incl. waiting for the slowest rank; "
                  "Computation = Total - I/O - Comm. - Sync. of that rank.")
    save(fig, "fig_per_rank.png", top=0.94)
    table(f"Load balance across ranks (final version, case {args.case}, median of 5 trials). Waiting = communication "
          "+ synchronization time of a rank; a rank with less computation spends more time waiting.", "tab:per-rank",
          ["Configuration", "Min compute (s)", "Max compute (s)", "Max / min", "Min waiting (s)",
           "Max waiting (s)", "Sent per rank (MB)"], rows_tex, "lrrrrrr")
else:
    skip("per-rank", f"找不到 {pr_path}（先跑 summarize.py）")

# ---- 9. 哪些 MPI 操作最貴（Nsight Systems） ----
nsys_dirs = sorted(glob.glob(os.path.join(os.path.dirname(args.summary), "nsys", "*", "mpi_summary.csv")))
ops = {}
for path in nsys_dirs:
    ver = os.path.basename(os.path.dirname(path)).split("_")[0]
    agg = defaultdict(lambda: [0.0, 0.0, 0.0])
    nranks = set()
    for r in csv.DictReader(open(path)):
        nranks.add(r["rank"])
        a = agg[r["mpi_call"]]
        a[0] += float(r["total_ms"])
        a[1] += float(r["count"])
        a[2] += float(r["bytes"] or 0)
    n = max(len(nranks), 1)
    ops[ver] = {k: (v[0] / n, v[1] / n, v[2] / n / 1e6) for k, v in agg.items()}   # 每個 rank 的平均
if ops:
    vers = [v for v in ("ori", "final") if v in ops]
    calls = sorted({c for v in vers for c in ops[v]}, key=lambda c: -max(ops[v].get(c, (0,))[0] for v in vers))
    calls = [c for c in calls if max(ops[v].get(c, (0,))[0] for v in vers) >= 0.5]   # 去掉 < 0.5 ms 的
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(calls) + 1.6))
    h = 0.8 / len(vers)
    for j, v in enumerate(vers):
        ys = [i + (j - (len(vers) - 1) / 2) * h for i in range(len(calls))]
        vals = [ops[v].get(c, (0, 0, 0))[0] for c in calls]
        cols = [C_IO if "File" in c else C_COMM if "Sendrecv" in c else C_SYNC for c in calls]
        ax.barh(ys, vals, h, color=cols, alpha=1.0 if v == "final" else 0.45,
                hatch=None if v == "final" else "//", edgecolor="white")
        for y, val, c in zip(ys, vals, calls):
            cnt = ops[v].get(c, (0, 0, 0))[1]
            ax.text(val, y, f" {val:.0f} ms, {cnt:.0f} calls ({'original' if v == 'ori' else 'final'})",
                    va="center", fontsize=6.5)
    ax.set_yticks(range(len(calls)), calls, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("time per rank inside the MPI call (ms, mean over 4 ranks)")
    ax.set_xlim(right=max(ops[v].get(c, (0,))[0] for v in vers for c in calls) * 1.7)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C_COMM, label="Communication"), Patch(color=C_SYNC, label="Synchronization / other"),
                       Patch(color=C_IO, label="I/O"),
                       Patch(facecolor="#BBBBBB", hatch="//", edgecolor="white", label="original version (hatched)")],
              fontsize=7, loc="lower right")
    ax.set_title(f"Cost of individual MPI operations (Nsight Systems, big, 4 processes, case {args.case})", fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    footnote(fig, "Traced with nsys --trace=mpi; tracing adds overhead, so the values are used to compare operations, "
                  "not as the reported run time. MPI_Init / MPI_Finalize include start-up and shut-down.")
    save(fig, "fig_mpi_ops.png")
    table("Time spent in each MPI operation per rank (Nsight Systems, \\texttt{big}, 4 processes, mean over ranks).",
          "tab:mpi-ops", ["MPI operation"] + [f"{'Original' if v == 'ori' else 'Final'} {x}" for v in vers
                                               for x in ("time (ms)", "calls", "MB")],
          [[c.replace("_", "\\_")] + [x for v in vers for x in (
              f"{ops[v].get(c, (0, 0, 0))[0]:.1f}", f"{ops[v].get(c, (0, 0, 0))[1]:.0f}",
              f"{ops[v].get(c, (0, 0, 0))[2]:.0f}")] for c in calls])
else:
    skip("MPI operations", "找不到 test/exp/results/nsys/*/mpi_summary.csv（先跑 nsys_mpi.py）")

# ---- 10. 總覽：每個設定的各項時間、MPI 呼叫數、通訊量、speedup ----
ov = []
for (st, v, part, nodes, p), r in sorted(R.items(), key=lambda kv: (kv[0][1] != "final", kv[0][0], kv[0][2],
                                                                     kv[0][3], kv[0][4])):
    if v not in ("final", "ori", "final_nocomp"):
        continue
    t = r["total_s"]
    ov.append([v.replace("_", "\\_"), part, nodes, p, "/tmp" if st == "local" else "NFS", f3(t),
               f"{r['compute_s'] / t * 100:.0f}\\%", f"{r['comm_s'] / t * 100:.0f}\\%",
               f"{r['sync_s'] / t * 100:.0f}\\%", f"{r['io_s'] / t * 100:.0f}\\%",
               f"{r.get('sendrecv_calls_per_rank', 0):.0f}", f"{r['sendrecv_MB']:.0f}",
               f"{r['compute_imbalance']:.2f}"])
if ov:
    table(f"Overview of all configurations (case {args.case}, median of 5 trials). Percentages are shares of the total "
          "time; Sendrecv calls are per sorting rank; sent volume is the sum over all ranks; imbalance = max / min "
          "computation time among sorting ranks.", "tab:overview",
          ["Version", "Partition", "Nodes", "Procs", "Storage", "Total (s)", "Comp.", "Comm.", "Sync.", "I/O",
           "Calls", "Sent (MB)", "Imbal."], ov, "llrrlrrrrrrrr")

# =============================================================================
# test/opt/results/opt_*.txt
# =============================================================================
opt_path = args.opt or max(glob.glob("test/opt/results/opt_*.txt"), key=os.path.getmtime, default=None)
print(f"[opt] {opt_path or '找不到 test/opt/results/opt_*.txt（sbatch test/opt/run_opt.sh）'}")
if opt_path:
    text = open(opt_path).read()
    sections = re.split(r"^===== \[(\d)\].*$", text, flags=re.M)
    sec = {sections[i]: sections[i + 1] for i in range(1, len(sections) - 1, 2)}

    # [1] radix（round 0，float key）與 [4] radix（round 1 之後，含 hash 的 24-bit key）：
    #     每一行「名稱 ... 數字 ms」，取行中第一個浮點數當中位數時間
    def radix_section(key, fname, label, title, caption):
        rows = []
        for line in sec.get(key, "").splitlines():
            m = re.match(r"^\s*(\S.*?)\s{2,}([\d.]+)", line)
            if m and not line.lstrip().startswith(("#", "input", "variant", "method", "N=")):
                rows.append((m.group(1).strip(), float(m.group(2))))
        if not rows:
            skip(label, f"{opt_path} 的 [{key}] 區段解析不到資料")
            return
        fig, ax = plt.subplots(figsize=(7.5, 3.6))
        names = [n for n, _ in rows][::-1]
        vals = [v for _, v in rows][::-1]
        ax.barh(names, vals, color=[C_COMP if "final" in n.lower() else "#8C8C8C" for n in names])
        for i, v in enumerate(vals):
            ax.text(v, i, f" {v:.1f} ms", va="center", fontsize=8)
        ax.set_xscale("log")
        ax.set_xlabel("time (ms, log scale, median)")
        ax.set_title(title, fontsize=10)
        ax.set_xlim(right=max(vals) * 4)
        save(fig, fname)
        base = rows[0][1]
        table(caption, label, ["Method", "Time (ms)", "Speedup vs.\\ first row"],
              [[n.replace("_", "\\_"), f"{v:.1f}", f"{base / v:.1f}$\\times$"] for n, v in rows])

    radix_section("1", "fig_opt_radix.png", "tab:opt-radix",
                  "Optimization 1 (round 0): local sort of the input floats\n(1 process, big core; computation only - no MPI, no I/O)",
                  "Local sort of the original input values in round 0 (case 10, 1 process on a big core, median).")
    radix_section("4", "fig_opt_radix24.png", "tab:opt-radix24",
                  "Optimization 1 (rounds 1-24): hash + local sort of 24-bit keys\n(1 process, big core; computation only - no MPI, no I/O)",
                  "Hash plus local sort in the rounds after hashing (24-bit keys, case 10, 1 process on a big core, "
                  "median). The final version computes the radix histogram inside the hash loop.")

    # [2] compare-split：區塊以「input=... ranks=... 」開頭，接著四列 variant。
    #     新版 bench_split 每列有 total / comm / sync / comp（ms）；舊版只有 total。
    split_rows = []
    kind, ranks = None, 0
    for line in sec.get("2", "").splitlines():
        h = re.match(r"^input=(\S+) N=\d+ ranks=(\d+)", line)
        if h:
            inp, ranks = h.group(1), int(h.group(2))
            kind = "nearly sorted" if "nearly" in inp else ("random (gen)" if "gen:random" in inp else "case 10")
            continue
        m = re.match(r"^(V\d.*?)\s{2,}([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s+(\d+)\s+(\S+)",
                     line)
        if m and kind:
            split_rows.append(dict(kind=kind, p=ranks, v=m.group(1).strip(), total=float(m.group(2)),
                                   comm=float(m.group(3)), sync=float(m.group(4)), comp=float(m.group(5)),
                                   mb=float(m.group(6)), ex=int(m.group(7)), sk=int(m.group(8)), ok=m.group(9)))
            continue
        m = re.match(r"^(V\d.*?)\s{2,}([\d.]+)\s+([\d.]+)\s+(\d+)\s+(\d+)\s+(\S+)", line)
        if m and kind:
            split_rows.append(dict(kind=kind, p=ranks, v=m.group(1).strip(), total=float(m.group(2)), comm=None,
                                   sync=None, comp=None, mb=float(m.group(3)), ex=int(m.group(4)),
                                   sk=int(m.group(5)), ok=m.group(6)))
    if split_rows:
        split_rows.sort(key=lambda r: (r["kind"], r["p"], r["v"]))
        has_split = all(r["comm"] is not None for r in split_rows)
        groups = sorted({(r["kind"], r["p"]) for r in split_rows})
        variants = sorted({r["v"] for r in split_rows})
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8))
        w = 0.8 / len(variants)
        short = {v: v.split()[0] for v in variants}
        for j, v in enumerate(variants):
            xs = [i + (j - (len(variants) - 1) / 2) * w for i in range(len(groups))]
            rows = [next((r for r in split_rows if (r["kind"], r["p"]) == g and r["v"] == v), None) for g in groups]
            if has_split:
                bottom = [0.0] * len(xs)
                for key, lab, col in (("comp", "Computation (merge)", C_COMP), ("comm", "Communication", C_COMM),
                                      ("sync", "Synchronization", C_SYNC)):
                    vals = [r[key] if r else 0 for r in rows]
                    a1.bar(xs, vals, w, bottom=bottom, color=col, edgecolor="white", linewidth=0.5,
                           label=lab if j == 0 else None)
                    bottom = [bb + vv for bb, vv in zip(bottom, vals)]
            else:
                a1.bar(xs, [r["total"] if r else 0 for r in rows], w, color="#8C8C8C")
            for x, r in zip(xs, rows):
                if r:
                    a1.text(x, r["total"], short[v], ha="center", va="bottom", fontsize=6)
            a2.bar(xs, [r["mb"] if r else 0 for r in rows], w, color=C_COMM, alpha=0.4 + 0.2 * j)
            for x, r in zip(xs, rows):
                if r:
                    a2.text(x, r["mb"], f"{short[v]}\n{r['mb']:.0f}" if r["mb"] >= 10 else f"{short[v]}\n{r['mb']:.1f}",
                            ha="center", va="bottom", fontsize=6)
        for ax, yl, t in ((a1, "time (ms, median of 5)", "Time of the odd-even phases"),
                          (a2, "MPI_Sendrecv payload, all ranks (MB)", "Communication volume")):
            ax.set_xticks(range(len(groups)), [f"{k}\np = {p}" for k, p in groups])
            ax.set_xlabel("input and number of MPI processes (big, 1 node)")
            ax.set_ylabel(yl)
            ax.set_title(t)
            ax.grid(axis="y", alpha=0.3)
            ax.set_ylim(top=ax.get_ylim()[1] * 1.15)
        if has_split:
            a1.legend(fontsize=7, loc="upper right")
        fig.suptitle("Optimization 2: compare-split variants (one complete odd-even sort from locally sorted blocks; "
                     "no I/O)", fontsize=10)
        footnote(fig, "V0 = full-block exchange + full merge (original); V1 = full exchange + merge only the kept half; "
                      "V2 = V1 + boundary check; V3 = V2 + send only elements that may move (final). "
                      "Computation = total - Comm. (MPI_Sendrecv) - Sync. (MPI_Allreduce), mean over ranks.")
        save(fig, "fig_opt_split.png")
        hdr = ["Input", "Procs", "Variant", "Total (ms)"] + (["Comp. (ms)", "Comm. (ms)", "Sync. (ms)"] if has_split else []) + \
              ["Sent (MB)", "Exchanges", "Skipped", "Result"]
        table("Compare-split variants: time of one complete odd-even sort (median of 5 trials) split into computation, "
              "communication and synchronization, total MPI\\_Sendrecv payload, and number of exchanges performed "
              "or skipped by the boundary check.", "tab:opt-split", hdr,
              [[r["kind"], r["p"], r["v"], f"{r['total']:.1f}"] +
               ([f"{r['comp']:.1f}", f"{r['comm']:.1f}", f"{r['sync']:.1f}"] if has_split else []) +
               [f"{r['mb']:.1f}", r["ex"], r["sk"], r["ok"]] for r in split_rows],
              "lrl" + "r" * (len(hdr) - 4) + "l")
        if not has_split:
            print("  注意：opt 結果是舊版 bench_split（沒有 comm / sync 分開），請重跑 sbatch test/opt/run_opt.sh")
    else:
        skip("compare-split", f"{opt_path} 的 [2] 區段解析不到資料")

    # [3] 每 rank 至少 4096 筆
    act = []
    for line in sec.get("3", "").splitlines():
        m = re.match(r"^(\d+)\s+(\d+ vs \d+)\s+([\d.]+)\s+([\d.]+)\s+(\S+)", line)
        if m:
            act.append((int(m.group(1)), m.group(2), float(m.group(3)), float(m.group(4)), m.group(5)))
    if act:
        # 只輸出表格：這不是報告的三個優化之一，而且只有總時間，不畫圖
        table("Effect of limiting active ranks for small inputs (4 processes on one big node, 25 rounds, median).",
              "tab:opt-active",
              ["N", "Active ranks", "Final (ms)", "All ranks (ms)", "Reduction", "Same output"],
              [[f"{n:,}", a.split()[0], f"{f:.2f}", f"{nm:.2f}", f"{(1 - f / nm) * 100:.1f}\\%", same]
               for n, a, f, nm, same in act])
    else:
        skip("active process", f"{opt_path} 的 [3] 區段解析不到資料")

out = os.path.join(args.out, "tables.tex")
with open(out, "w") as f:
    f.write("\n".join(tex))
print(f"LaTeX 表格：{out}")
