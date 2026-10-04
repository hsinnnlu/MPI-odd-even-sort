#!/bin/bash
#SBATCH -J hw1opt
#SBATCH -p big
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 10:00
#SBATCH -o test/opt/results/slurm_%j.txt
# =============================================================================
# 報告「三個優化」的佐證實驗（一個 job 跑完，約 3 分鐘）。
#
# 用法（課程機器，repo 根目錄，需要 hash.h；先 module load）：
#   mkdir -p test/opt/results && sbatch test/opt/run_opt.sh
#   結果：test/opt/results/opt_<job id>.txt
#
#   [1] Radix sort（round 0）：std::sort、8+8+8+8、16+16、11+11+10（分開算 histogram / 一次算好）
#   [4] Radix sort（round 1 之後，含 hash）：std::sort、12+12、8+8+8（另外算 histogram / 在 hash 裡順便算）
#   [2] Compare-split：V0 整塊+全部 merge、V1 整塊+部分 merge、V2 +邊界檢查、V3 繳交版（只送可能移動的）
#       輸入：課程測資 10（隨機）與 gen:nearly（幾乎排好），2 與 4 個 process
#   [3] 每個 rank 至少 4096 筆：final vs 拿掉這條規則的 final_nomin，小 N、4 個 process
# 輸入檔依規定複製到 node-local /tmp，job 結束刪除。
# =============================================================================
set -u
REPO=${SLURM_SUBMIT_DIR:-$PWD}
cd "$REPO"
CASES=${CASES:-/srv/nova/scratch/coursedata/pp2026/hw1}
TRIALS=${TRIALS:-5}
OUT=test/opt/results/opt_${SLURM_JOB_ID:-local}.txt
mkdir -p test/opt/results test/build/opt
B=test/build/opt

if ! type module >/dev/null 2>&1; then
    for f in /etc/profile.d/lmod.sh /etc/profile.d/modules.sh /usr/share/lmod/lmod/init/bash; do
        [ -f "$f" ] && source "$f" && break
    done
fi
type module >/dev/null 2>&1 && module load compiler/gcc/13 mpi/openmpi/5.0.10 >/dev/null 2>&1

# ---- 編譯 ----
mpicxx -O3 -I. -o $B/bench_radix test/opt/bench_radix.cc || exit 1
mpicxx -O3 -I. -o $B/bench_split test/opt/bench_split.cc || exit 1
cp hw1.cc $B/final.cc
sed 's/constexpr long long MIN_ELEMENTS_PER_RANK = 4096;/constexpr long long MIN_ELEMENTS_PER_RANK = 1;/' hw1.cc > $B/final_nomin.cc
grep -q 'MIN_ELEMENTS_PER_RANK = 1;' $B/final_nomin.cc || { echo "產生 final_nomin 失敗"; exit 1; }
for v in final final_nomin; do mpicxx -O3 -I. -include test/exp/prof_wrap.h -o $B/$v $B/$v.cc || exit 1; done

# ---- 輸入放到 node-local /tmp ----
WORK=/tmp/hw1opt_${SLURM_JOB_ID:-local}
srun -N1 -n1 -c1 bash -c "mkdir -p $WORK && cp $CASES/10.in $WORK/in10" || exit 1
trap 'srun -N1 -n1 -c1 rm -rf "$WORK"' EXIT
N10=${N10:-23987513}

{
echo "# job=${SLURM_JOB_ID:-local} node=${SLURM_JOB_NODELIST:-?} partition=big modules=${LOADEDMODULES:-?}"
echo "# $(mpicxx --version | head -1) / $(mpirun --version 2>/dev/null | head -1)"
echo "# storage: node-local $WORK; trials=$TRIALS"
echo
echo "===== [1] Local sort（1 process，big core，測資 10 的 float）====="
srun -N1 -n1 -c1 $B/bench_radix $WORK/in10 $N10 $TRIALS
echo
echo "===== [2] Compare-split（round 0 的完整 odd-even 排序）====="
for p in 2 4; do
    srun -N1 -n$p -c1 $B/bench_split $WORK/in10 $N10 $TRIALS
    echo
    srun -N1 -n$p -c1 $B/bench_split gen:nearly $N10 $TRIALS
    echo
done
echo "===== [3] 每個 rank 至少 4096 筆（4 process，25 輪；時間 = MPI_Init 後到 Finalize 前，不含 srun 啟動）====="
printf "%-8s %-8s %14s %14s %8s\n" N active "final(ms)" "nomin(ms)" same
for n in 100 1000 10000; do
    srun -N1 -n1 -c1 python3 test/gen.py $n $WORK/small_$n uniform
    declare -A med
    for v in final final_nomin; do
        ts=()
        for ((t = 1; t <= TRIALS; ++t)); do
            line=$(EXP_TAG=x srun -N1 -n4 -c1 $B/$v $n $WORK/small_$n $WORK/out_${v}_$n 25 2>&1 >/dev/null | grep '^PROF' | head -1)
            ts+=("$(echo "$line" | cut -d, -f6)")
        done
        med[$v]=$(printf '%s\n' "${ts[@]}" | sort -n | sed -n "$(( (TRIALS + 1) / 2 ))p")
    done
    same=$(srun -N1 -n1 -c1 cmp -s $WORK/out_final_$n $WORK/out_final_nomin_$n && echo yes || echo NO)
    active=$(( (n + 4095) / 4096 )); [ $active -gt 4 ] && active=4
    python3 -c "print(f'{$n:<8} {\"$active vs 4\":<8} {${med[final]}*1000:14.2f} {${med[final_nomin]}*1000:14.2f} {\"$same\":>8}')"
done
echo
echo "===== [4] Local sort，round 1 之後（hash 輸出的 24-bit key，含 hash 時間；1 process，big core）====="
srun -N1 -n1 -c1 $B/bench_radix $WORK/in10 $N10 $TRIALS hash
} > "$OUT" 2>&1
echo "結果：$OUT"
