#!/bin/bash
# =============================================================================
# 報告 5.3 的實驗：編譯各版本（含計時外掛），送出所有實驗 job。
#
# 用法（課程機器，repo 根目錄，需要 hash.h）：
#   bash test/exp/submit_all.sh            # 全部送出
#   TRIALS=3 bash test/exp/submit_all.sh   # 先少跑幾次試水溫
#   ONLY="big mixed" bash test/exp/submit_all.sh
#   LIMIT=5:00 bash test/exp/submit_all.sh   # 每個 job 的時間上限（預設 10:00），不能超過課程 QOS 的限制
#   EXCLUSIVE=1 bash test/exp/submit_all.sh   # 整台節點獨占，避免別人的 job 在同一台搶記憶體頻寬（排隊可能較久）
#   CASE_ID=08 ONLY="big little" bash test/exp/submit_all.sh   # 額外：用 8M 筆的測資再跑一次（不同問題大小）
# 全部跑完後：python3 test/exp/summarize.py
#
# 版本：
#   final        最終繳交的 hw1.cc
#   final_nocomp 最終版，但拿掉跨節點壓縮（只用在 2 節點的實驗，量壓縮的效果）
#   ori          最初版本（std::sort + 整塊交換 + std::merge），test/exp/baseline_ori.cc
#
# 實驗（測資 10，N = 23,987,513，25 輪；每個設定 TRIALS 次）：
#   big     -p big    -N1  procs 1 2 4   版本 final ori   ← strong scaling（同質）+ 前後比較
#   little  -p little -N1  procs 1 2 4   版本 final ori   ← big vs little
#   mixed   -p mixed  -N1  procs 8 4 1   版本 final   ← 整台節點、異質核心（1 = mixed 自己的 speedup 基準）
#   bignfs  -p big    -N1  procs 1 2 4   版本 final   ← strong scaling 用：和 big2n 一樣放 NFS
#   big2n   -p big    -N2  procs 8       版本 final final_nocomp   ← big 8 個 process（每節點只有 4 個 big core，必須 2 節點）
#   mixed2n -p mixed  -N2  procs 16      版本 final final_nocomp   ← 2 節點 16 process（同測資 36–40 的設定）
#   （2 節點時兩台節點的 process 都參與排序；final vs final_nocomp 量跨節點壓縮的效果）
# =============================================================================
set -eu
cd "$(dirname "$0")/../.."
REPO=$PWD
[ -f hash.h ] || { echo "找不到 hash.h，請先放到 repo 根目錄"; exit 1; }

if ! type module >/dev/null 2>&1; then
    for f in /etc/profile.d/lmod.sh /etc/profile.d/modules.sh /usr/share/lmod/lmod/init/bash; do
        [ -f "$f" ] && source "$f" && break
    done
fi
module purge
module load compiler/gcc/13 mpi/openmpi/5.0.10

# ---- 編譯 ----------------------------------------------------------------------
B=test/build/exp
mkdir -p "$B" test/exp/results
WRAP="-include test/exp/prof_wrap.h"

cp hw1.cc "$B/final.cc"
sed 's/const bool compress = L.node\[partner\] != L.node\[rank\];/const bool compress = false;/' hw1.cc > "$B/final_nocomp.cc"
grep -q 'const bool compress = false;' "$B/final_nocomp.cc" || { echo "產生 final_nocomp 失敗：hw1.cc 的寫法和預期不同"; exit 1; }
cp test/exp/baseline_ori.cc "$B/ori.cc"

for v in final final_nocomp ori; do
    mpicxx -O3 -I. $WRAP -o "$B/$v" "$B/$v.cc"
done
echo "已編譯：final final_nocomp ori（$B/）"

# ---- 送出 job -------------------------------------------------------------------
export REPO TRIALS=${TRIALS:-5} CASE_ID=${CASE_ID:-10}
SUFFIX=$([ "$CASE_ID" = 10 ] || echo "_case$CASE_ID")
ONLY=${ONLY:-"big little mixed bignfs big2n mixed2n"}
# 每個 job 的時間上限。課程的 QOS 有「單一 job 最長執行時間」的限制，
# 超過的 job 會一直卡在 PD（QOSMaxWallDurationPerJobLimit）。
# 查限制：sacctmgr -P show qos format=Name,MaxWall ；或 scontrol show partition big | grep MaxTime
LIMIT=${LIMIT:-10:00}
submit() {   # submit <名字> <sbatch 參數> <procs> <versions> [local|nfs]
    local name=$1 opts=$2 procs=$3 versions=$4 storage=${5:-local}
    [[ " $ONLY " == *" ${name%%_*} "* ]] || return 0
    EXP_NAME=$name$SUFFIX PROCS=$procs VERSIONS=$versions STORAGE_MODE=$storage \
        sbatch --parsable -J "hw1exp-$name" $opts ${EXCLUSIVE:+--exclusive} -t "$LIMIT" -o "test/exp/results/slurm_${name}_%j.txt" test/exp/job.sh \
        | sed "s/^/送出 $name（procs: $procs；版本: $versions），job id = /"
}
# 拆成小 job，每個都能在 LIMIT 內跑完。最初版本（ori）很慢，每種 process 數各自一個 job。
submit big           "-p big -N1 -n4"    "1 2 4" "final"
submit little        "-p little -N1 -n4" "1 2 4" "final"
for p in 1 2 4; do
    submit big_ori_p$p    "-p big -N1 -n4"    "$p" "ori"
    submit little_ori_p$p "-p little -N1 -n4" "$p" "ori"
done
submit mixed         "-p mixed -N1 -n8"  "8 4 1" "final"
submit bignfs        "-p big -N1 -n4"    "1 2 4" "final" nfs
submit big2n         "-p big -N2 -n8"    "8"     "final final_nocomp"
submit mixed2n       "-p mixed -N2 -n16" "16"    "final final_nocomp"
echo
echo "用 squeue -u \$USER 查看進度；全部結束後執行：python3 test/exp/summarize.py"
