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
  fig_mixed.png              mixed 1/4/8：final（只用 rank 0 節點上的 process）vs final_nocap（全部 process）
  fig_ori_vs_final.png       最初版本 vs final（對數座標，標出加速倍數）
  fig_opt_radix.png          Optimization 1：local sort 各做法時間
  fig_opt_split.png          Optimization 2：compare-split 四種做法的時間與傳送量
  fig_opt_active.png         Optimization 3：小 N 時每 rank 至少 4096 筆的效果
  tables.tex                 以上全部的 LaTeX 表格（booktabs；\\usepackage{booktabs}）
  （某項資料不存在時跳過那張圖/表並印出原因）

儲存位置：summary.csv 的 storage 欄位區分 local（node-local /tmp）與 nfs。
  * breakdown、big vs little、mixed、ori vs final 只用 local（規定的正式實驗）
  * scaling 依 --scaling-storage：
      local  （預設，報告採用）1/2/4 用 local；8 process（2 節點）只能用 nfs，圖上加註
      nfs    全部用 nfs（job_scaling.sh 的「同一儲存位置」設計）
      auto   nfs 有 1/2/4/8 就用 nfs，否則用 local
"""
import argparse
import csv
import glob
import os
import re
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--case", default="10")
ap.add_argument("--summary", default="test/exp/results/summary.csv")
ap.add_argument("--opt", default=None, help="run_opt.sh 的結果檔（預設取 test/opt/results/opt_*.txt 最新的）")
ap.add_argument("--scaling-storage", default="local", choices=["auto", "local", "nfs"])
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


def save(fig, name):
    path = os.path.join(args.out, name)
    fig.tight_layout()
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
        R[key] = {k: (float(v) if re.fullmatch(r"-?[\d.]+(e-?\d+)?|nan", v or "x") else v) for k, v in r.items()}
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
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    xs = [str(p) for p, _ in s]
    bottom = [0.0] * len(s)
    for key, lab, col in (("compute_s", "Computation", C_COMP), ("comm_s", "Communication", C_COMM),
                          ("sync_s", "Synchronization", C_SYNC), ("io_s", "I/O", C_IO)):
        vals = [r[key] for _, r in s]
        ax.bar(xs, vals, bottom=bottom, color=col, label=lab, width=0.6)
        bottom = [b + v for b, v in zip(bottom, vals)]
    for i, (_, r) in enumerate(s):
        ax.text(i, bottom[i], f"{r['total_s']:.2f} s", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("number of MPI processes (big cores, 1 node)")
    ax.set_ylabel("time (s)")
    ax.set_title(f"Time profile of the final version (case {args.case}, node-local /tmp)")
    ax.set_ylim(0, max(bottom) * 1.15)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    save(fig, "fig_breakdown_big.png")
    table(f"Time profile of the final version on big cores (case {args.case}, median of 5 trials, seconds).",
          "tab:breakdown",
          ["Processes", "Total", "I/O (read / write)", "Comm.", "Sync.", "Compute", "Speedup"],
          [[p, f3(r["total_s"]), f"{r['io_s']:.3f} ({r['io_read_s']:.3f} / {r['io_write_s']:.3f})",
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
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    ps = [p for p, _, _ in sc]
    a1.plot(ps, [r["total_s"] for _, r, _ in sc], "o-", color=C_COMP, label="total")
    a1.plot(ps, [r["compute_s"] for _, r, _ in sc], "s--", color="#8172B3", label="compute only")
    a1.plot(ps, [r["io_s"] for _, r, _ in sc], "^:", color=C_IO, label="I/O")
    a1.set_xscale("log", base=2)
    a1.set_xticks(ps, [str(p) for p in ps])
    a1.set_xlabel("number of MPI processes")
    a1.set_ylabel("time (s)")
    a1.set_title("Execution time")
    a1.grid(alpha=0.3)
    a1.legend(fontsize=8)
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
    note = "all runs on shared NFS" if mode == "nfs" else "1-4 procs: node-local /tmp; 8 procs: 2 nodes, shared NFS"
    fig.suptitle(f"Strong scaling, final version, big cores (case {args.case}; {note})", fontsize=10)
    save(fig, "fig_scaling.png")
    table(f"Strong scaling of the final version on big cores (case {args.case}; {note}). "
          "Speedup is relative to 1 process; compute-only speedup excludes I/O, communication and synchronization.",
          "tab:scaling",
          ["Processes", "Nodes", "Active ranks", "Storage", "Total (s)", "I/O (s)", "Compute (s)", "Speedup",
           "Compute speedup"],
          [[p, int(r["nodes"]), int(r["busy_ranks"]), "/tmp" if st == "local" else "NFS", f3(r["total_s"]),
            f3(r["io_s"]), f3(r["compute_s"]), f"{base / r['total_s']:.2f}", f"{base_c / r['compute_s']:.2f}"]
           for p, r, st in sc])
else:
    skip("strong scaling", f"storage={mode} 沒有包含 1 process 的 final/big 結果")

# ---- 3. big vs little ----
b, l = dict(series("local", "final", "big", 1, [1, 2, 4])), dict(series("local", "final", "little", 1, [1, 2, 4]))
ps = sorted(set(b) & set(l))
if ps:
    fig, ax = plt.subplots(figsize=(6.4, 4))
    w = 0.38
    xs = range(len(ps))
    for off, d, lab, col in ((-w / 2, b, "big", C_COMP), (w / 2, l, "little", "#DD8452")):
        ax.bar([x + off for x in xs], [d[p]["compute_s"] for p in ps], w, color=col, label=f"{lab}: compute")
        ax.bar([x + off for x in xs], [d[p]["total_s"] - d[p]["compute_s"] for p in ps], w,
               bottom=[d[p]["compute_s"] for p in ps], color=col, alpha=0.35, label=f"{lab}: I/O + comm + sync")
        for x, p in zip(xs, ps):
            ax.text(x + off, d[p]["total_s"], f"{d[p]['total_s']:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(list(xs), [str(p) for p in ps])
    ax.set_xlabel("number of MPI processes (1 node)")
    ax.set_ylabel("time (s)")
    ax.set_title(f"Big vs little cores, final version (case {args.case})")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    save(fig, "fig_big_vs_little.png")
    table(f"Big versus little cores (final version, case {args.case}, node-local /tmp, seconds).",
          "tab:big-little",
          ["Processes", "Big total", "Big compute", "Little total", "Little compute", "Little / big (total)",
           "Little / big (compute)"],
          [[p, f3(b[p]["total_s"]), f3(b[p]["compute_s"]), f3(l[p]["total_s"]), f3(l[p]["compute_s"]),
            f"{l[p]['total_s'] / b[p]['total_s']:.2f}", f"{l[p]['compute_s'] / b[p]['compute_s']:.2f}"] for p in ps])
else:
    skip("big vs little", "缺少 local 的 big 或 little 結果")

# ---- 4. mixed ----
mx = {v: dict(series("local", v, "mixed", 1, [1, 4, 8])) for v in ("final", "final_nocap")}
ps = sorted(set(mx["final"]) | set(mx["final_nocap"]))
if ps:
    fig, ax = plt.subplots(figsize=(6.4, 4))
    w = 0.38
    for off, v, lab, col in ((-w / 2, "final", "final (active ranks on rank 0's node, max 4)", C_COMP),
                             (w / 2, "final_nocap", "final_nocap (all processes sort)", "#C44E52")):
        xs = [i for i, p in enumerate(ps) if p in mx[v]]
        ys = [mx[v][ps[i]]["total_s"] for i in xs]
        ax.bar([x + off for x in xs], ys, w, color=col, label=lab)
        for x, y in zip(xs, ys):
            ax.text(x + off, y, f"{y:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(ps)), [str(p) for p in ps])
    ax.set_xlabel("number of MPI processes (mixed partition)")
    ax.set_ylabel("total time (s)")
    ax.set_title(f"Mixed big/little node (case {args.case})")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    save(fig, "fig_mixed.png")
    rows = []
    for p in ps:
        for v in ("final", "final_nocap"):
            r = mx[v].get(p)
            if r:
                rows.append([p, v.replace("_", "\\_"), int(r["busy_ranks"]), f3(r["total_s"]), f3(r["io_s"]),
                             f3(r["comm_s"]), f3(r["sync_s"]), f3(r["compute_s"]), f"{r['compute_imbalance']:.2f}"])
    table(f"Mixed partition (case {args.case}, node-local /tmp, seconds). Imbalance = max / min compute time "
          "among active ranks.", "tab:mixed",
          ["Processes", "Version", "Active", "Total", "I/O", "Comm.", "Sync.", "Compute", "Imbalance"], rows)
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
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    w = 0.38
    xs = range(len(pairs))
    ax.bar([x - w / 2 for x in xs], [o["total_s"] for _, _, o, _ in pairs], w, color="#8C8C8C", label="original")
    ax.bar([x + w / 2 for x in xs], [f["total_s"] for _, _, _, f in pairs], w, color=C_COMP, label="final")
    for x, (_, _, o, f) in zip(xs, pairs):
        ax.text(x, o["total_s"] * 1.1, f"{o['total_s'] / f['total_s']:.1f}x", ha="center", fontsize=8)
    ax.set_yscale("log")
    ax.set_xticks(list(xs), [f"{part}\np={p}" for part, p, _, _ in pairs])
    ax.set_ylabel("total time (s, log scale)")
    ax.set_title(f"Original vs final version (case {args.case}, node-local /tmp)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3, which="both")
    save(fig, "fig_ori_vs_final.png")
    table(f"Original versus final version (case {args.case}, node-local /tmp, seconds).", "tab:ori-final",
          ["Partition", "Processes", "Original total", "Original compute", "Final total", "Final compute",
           "Speedup"],
          [[part, p, f3(o["total_s"]), f3(o["compute_s"]), f3(f["total_s"]), f3(f["compute_s"]),
            f"{o['total_s'] / f['total_s']:.1f}$\\times$"] for part, p, o, f in pairs])
else:
    skip("ori vs final", "缺少 local 的 ori 或 final 結果")

# ---- 6. 通訊量與有效頻寬 ----
rows = []
for (st, v, part, nodes, p), r in sorted(R.items(), key=lambda kv: (kv[0][1], kv[0][2], kv[0][3], kv[0][4])):
    if p > 1 and v in ("ori", "final", "final_nocap") and r.get("sendrecv_MB"):
        bw = r.get("comm_bw_MBps")
        rows.append([v.replace("_", "\\_"), part, nodes, p, "/tmp" if st == "local" else "NFS",
                     f"{r['sendrecv_MB']:.0f}", f3(r["comm_s"]), f"{bw:.0f}" if isinstance(bw, float) else "--"])
if rows:
    table(f"Communication volume (sum of MPI\\_Sendrecv payload over all ranks) and effective bandwidth "
          f"(case {args.case}). Communication time includes waiting for the partner, so the bandwidth is a lower "
          "bound.", "tab:comm",
          ["Version", "Partition", "Nodes", "Processes", "Storage", "Sent (MB)", "Comm. (s)", "Eff. BW (MB/s)"],
          rows, "lllrlrrr")

# =============================================================================
# test/opt/results/opt_*.txt
# =============================================================================
opt_path = args.opt or max(glob.glob("test/opt/results/opt_*.txt"), key=os.path.getmtime, default=None)
print(f"[opt] {opt_path or '找不到 test/opt/results/opt_*.txt（sbatch test/opt/run_opt.sh）'}")
if opt_path:
    text = open(opt_path).read()
    sections = re.split(r"^===== \[(\d)\].*$", text, flags=re.M)
    sec = {sections[i]: sections[i + 1] for i in range(1, len(sections) - 1, 2)}

    # [1] radix：每一行「名稱 ... 數字 ms」，取行中第一個浮點數當中位數時間
    radix = []
    for line in sec.get("1", "").splitlines():
        m = re.match(r"^\s*(\S.*?)\s{2,}([\d.]+)", line)
        if m and not line.lstrip().startswith(("#", "input", "variant", "method")):
            radix.append((m.group(1).strip(), float(m.group(2))))
    if radix:
        fig, ax = plt.subplots(figsize=(7, 3.6))
        names = [n for n, _ in radix][::-1]
        vals = [v for _, v in radix][::-1]
        ax.barh(names, vals, color=[C_COMP if "fused" in n.lower() else "#8C8C8C" for n in names])
        for i, v in enumerate(vals):
            ax.text(v, i, f" {v:.1f} ms", va="center", fontsize=8)
        ax.set_xscale("log")
        ax.set_xlabel("time (ms, log scale, median)")
        ax.set_title("Optimization 1: local sort of one rank's block (1 process, big core)")
        ax.set_xlim(right=max(vals) * 4)
        save(fig, "fig_opt_radix.png")
        base = radix[0][1]
        table("Local sort methods (case 10 values, 1 process on a big core, median).", "tab:opt-radix",
              ["Method", "Time (ms)", "Speedup vs.\\ first row"],
              [[n.replace("_", "\\_").replace("+", "+"), f"{v:.1f}", f"{base / v:.1f}$\\times$"] for n, v in radix])
    else:
        skip("radix", f"{opt_path} 的 [1] 區段解析不到資料")

    # [2] compare-split：區塊以「input=... ranks=... 」開頭，接著四列 variant
    split_rows = []
    kind, ranks = None, 0
    for line in sec.get("2", "").splitlines():
        h = re.match(r"^input=(\S+) N=\d+ ranks=(\d+)", line)
        if h:
            inp, ranks = h.group(1), int(h.group(2))
            kind = "nearly sorted" if "nearly" in inp else ("random (gen)" if "gen:random" in inp else "case 10")
            continue
        m = re.match(r"^(V\d.*?)\s{2,}([\d.]+)\s+([\d.]+)\s+(\d+)\s+(\d+)\s+(\S+)", line)
        if m and kind:
            split_rows.append((kind, ranks, m.group(1).strip(), float(m.group(2)), float(m.group(3)),
                               int(m.group(4)), int(m.group(5)), m.group(6)))
    if split_rows:
        split_rows.sort(key=lambda r: (r[0], r[1], r[2]))
        groups = sorted({(k, p) for k, p, *_ in split_rows})
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
        variants = sorted({r[2] for r in split_rows})
        cols = ["#8C8C8C", "#CCB974", "#DD8452", C_COMP]
        w = 0.8 / len(variants)
        for j, v in enumerate(variants):
            for ax, idx in ((a1, 3), (a2, 4)):
                ys = [next((r[idx] for r in split_rows if (r[0], r[1]) == g and r[2] == v), 0) for g in groups]
                xs = [i + (j - (len(variants) - 1) / 2) * w for i in range(len(groups))]
                ax.bar(xs, ys, w, color=cols[j % len(cols)], label=v)
                for x, y in zip(xs, ys):     # 標數字：例如 V3 在幾乎排好的資料只送不到 1 MB，柱子看不到
                    ax.text(x, y, f"{y:.0f}" if y >= 10 else f"{y:.1f}", ha="center", va="bottom", fontsize=6)
        for ax, yl, t in ((a1, "time (ms, median)", "Odd-even phase time"),
                          (a2, "MPI_Sendrecv payload, all ranks (MB)", "Communication volume")):
            ax.set_xticks(range(len(groups)), [f"{k}\np={p}" for k, p in groups])
            ax.set_ylabel(yl)
            ax.set_title(t)
            ax.grid(axis="y", alpha=0.3)
        a1.legend(fontsize=7)
        fig.suptitle("Optimization 2: compare-split variants (starting from locally sorted blocks)", fontsize=10)
        save(fig, "fig_opt_split.png")
        table("Compare-split variants: time of the complete odd-even phase loop, total MPI\\_Sendrecv payload, "
              "and number of exchanges performed or skipped by the boundary check (median of 5 trials).",
              "tab:opt-split",
              ["Input", "Processes", "Variant", "Time (ms)", "Sent (MB)", "Exchanges", "Skipped", "Result"],
              [[k, p, v, f"{t:.1f}", f"{mb:.1f}", ex, sk, ok] for k, p, v, t, mb, ex, sk, ok in split_rows],
              "lrlrrrrl")
    else:
        skip("compare-split", f"{opt_path} 的 [2] 區段解析不到資料")

    # [3] 每 rank 至少 4096 筆
    act = []
    for line in sec.get("3", "").splitlines():
        m = re.match(r"^(\d+)\s+(\d+ vs \d+)\s+([\d.]+)\s+([\d.]+)\s+(\S+)", line)
        if m:
            act.append((int(m.group(1)), m.group(2), float(m.group(3)), float(m.group(4)), m.group(5)))
    if act:
        fig, ax = plt.subplots(figsize=(6.4, 4))
        w = 0.38
        xs = range(len(act))
        ax.bar([x - w / 2 for x in xs], [a[3] for a in act], w, color="#8C8C8C", label="all 4 ranks sort")
        ax.bar([x + w / 2 for x in xs], [a[2] for a in act], w, color=C_COMP,
               label="final: >= 4096 elements per active rank")
        for x, a in zip(xs, act):
            ax.text(x + w / 2, a[2], f"{a[1].split()[0]} active", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(list(xs), [f"N={a[0]:,}" for a in act])
        ax.set_ylabel("time from MPI_Init to MPI_Finalize (ms)")
        ax.set_title("Optimization 3: active process selection for small N (4 processes)")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        save(fig, "fig_opt_active.png")
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
