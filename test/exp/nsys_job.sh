#!/bin/bash
#SBATCH -J hw1nsys
#SBATCH -p big
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 10:00
#SBATCH -o test/exp/results/nsys/slurm_%j.txt
# =============================================================================
# Nsight Systems profiling（報告 Methodology：只用來看 timeline 與通訊行為，
# 不拿來算正式的執行時間或 speedup）。
#
# 用法（課程機器，repo 根目錄，需要 hash.h）：
#   mkdir -p test/exp/results/nsys
#   sbatch test/exp/nsys_job.sh                          # big，4 process，final
#   sbatch -p mixed -n 8 --export=ALL,PROCS=8 test/exp/nsys_job.sh
#   sbatch --export=ALL,VERSION=ori test/exp/nsys_job.sh # 最初版本（比較通訊型態）
#   跑完：python3 test/exp/nsys_mpi.py test/exp/results/nsys/<tag>
#
# 產出（test/exp/results/nsys/<tag>/）：
#   rank<r>.nsys-rep     每個 rank 一個，可用 Nsight Systems GUI 打開看 timeline
#   rank<r>.sqlite       nsys export 的 SQLite（給 nsys_mpi.py 畫圖用）
#   stats_rank<r>_*.csv  nsys stats 的 MPI 統計報表
# =============================================================================
set -u
REPO=${SLURM_SUBMIT_DIR:-$PWD}
cd "$REPO"
CASES=${CASES:-/srv/nova/scratch/coursedata/pp2026/hw1}
CASE_ID=${CASE_ID:-10}
VERSION=${VERSION:-final}             # final 或 ori
PROCS=${PROCS:-4}
PART=${SLURM_JOB_PARTITION:-big}
NSYS_MODULE=${NSYS_MODULE:-profiler/nsight-systems/2026.1.3}
TAG=${VERSION}_${PART}_p${PROCS}_${SLURM_JOB_ID:-local}
OUT=$REPO/test/exp/results/nsys/$TAG
mkdir -p "$OUT" test/build/nsys

if ! type module >/dev/null 2>&1; then
    for f in /etc/profile.d/lmod.sh /etc/profile.d/modules.sh /usr/share/lmod/lmod/init/bash; do
        [ -f "$f" ] && source "$f" && break
    done
fi
module load compiler/gcc/13 mpi/openmpi/5.0.10 "$NSYS_MODULE" || {
    echo "無法載入 $NSYS_MODULE；可用的版本："; module avail nsight 2>&1; exit 1; }
echo "nsys: $(command -v nsys) ($(nsys --version 2>/dev/null))"

# ---- 編譯：不加計時外掛，量的是繳交程式本身 ----
case $VERSION in
    final) SRC=hw1.cc ;;
    ori)   SRC=test/exp/baseline_ori.cc ;;
    *)     echo "未知的 VERSION=$VERSION"; exit 1 ;;
esac
BIN=test/build/nsys/$VERSION
mpicxx -O3 -g -I. -o "$BIN" "$SRC" || exit 1

read -r N _n _p _part ROUNDS < <(python3 test/parse_case.py "$CASES/$CASE_ID.txt")

# ---- 輸入放到 node-local /tmp ----
WORK=/tmp/hw1nsys_${SLURM_JOB_ID:-local}
srun -N1 -n1 -c1 bash -c "mkdir -p $WORK && cp $CASES/$CASE_ID.in $WORK/in" || exit 1
trap 'srun -N1 -n1 -c1 rm -rf "$WORK"' EXIT

{
    echo "# job=${SLURM_JOB_ID:-local} partition=$PART procs=$PROCS version=$VERSION case=$CASE_ID N=$N rounds=$ROUNDS"
    echo "# modules: ${LOADEDMODULES:-?}"
    echo "# storage: node-local $WORK"
} > "$OUT/info.txt"

# ---- profile：每個 rank 各自一個報告檔（%q{SLURM_PROCID} = rank 編號） ----
srun -N1 -n"$PROCS" -c1 nsys profile \
    --trace=mpi,osrt --mpi-impl=openmpi \
    --sample=none --cpuctxsw=none \
    --force-overwrite=true \
    -o "$OUT/rank%q{SLURM_PROCID}" \
    "$REPO/$BIN" "$N" "$WORK/in" "$WORK/out" "$ROUNDS"
echo "exit code: $?" >> "$OUT/info.txt"

if srun -N1 -n1 -c1 cmp -s "$WORK/out" "$CASES/$CASE_ID.out"; then
    echo "output: OK" >> "$OUT/info.txt"
else
    echo "output: WRONG" >> "$OUT/info.txt"
fi

# ---- 統計報表與 SQLite ----
for rep in "$OUT"/rank*.nsys-rep; do
    r=$(basename "$rep" .nsys-rep)
    nsys export --type sqlite --force-overwrite true -o "$OUT/$r.sqlite" "$rep" > /dev/null 2>&1 \
        || echo "nsys export 失敗：$rep" >> "$OUT/info.txt"
    # 報表名稱依 nsys 版本不同；可用 nsys stats --help-reports 查詢
    for report in mpi_event_sum osrt_sum; do
        nsys stats --report "$report" --format csv --force-export=true \
            -o "$OUT/stats_${r}" "$rep" > /dev/null 2>&1 \
            || echo "nsys stats --report $report 失敗（$r），可能是版本不支援" >> "$OUT/info.txt"
    done
done
nsys stats --help-reports > "$OUT/available_reports.txt" 2>&1
cat "$OUT/info.txt"
echo "下一步：python3 test/exp/nsys_mpi.py $OUT"
