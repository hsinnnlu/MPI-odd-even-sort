#!/bin/bash
# =============================================================================
# 報告 5.3 的實驗：編譯各版本（含計時外掛），送出所有實驗 job。
#
# 用法（課程機器，repo 根目錄，需要 hash.h）：
#   bash test/exp/submit_all.sh            # 全部送出
#   TRIALS=3 bash test/exp/submit_all.sh   # 先少跑幾次試水溫
#   ONLY="big mixed" bash test/exp/submit_all.sh
#   EXCLUSIVE=1 bash test/exp/submit_all.sh   # 整台節點獨占，避免別人的 job 在同一台搶記憶體頻寬（排隊可能較久）
#   CASE_ID=08 ONLY="big little" bash test/exp/submit_all.sh   # 額外：用 8M 筆的測資再跑一次（不同問題大小）
# 全部跑完後：python3 test/exp/summarize.py
#
# 版本：
#   final        最終繳交的 hw1.cc
#   final_nocap  最終版，但拿掉「只用第一台節點」與「最多 4 個 rank」（額外實驗用）
#   v4           我自己的 v4（commit 68eda46）
#   ori          最初版本（std::sort + 整塊交換 + std::merge），test/exp/baseline_ori.cc
#
# 實驗（測資 10，N = 23,987,513，25 輪；每個設定 TRIALS 次）：
#   big     -p big    -N1  procs 1 2 4   版本 final v4 ori   ← strong scaling（同質）+ 前後比較
#   little  -p little -N1  procs 1 2 4   版本 final v4 ori   ← big vs little
#   mixed   -p mixed  -N1  procs 8 4 1   版本 final final_nocap v4   ← 整台節點、異質核心（1 = mixed 自己的 speedup 基準）
#   big2n   -p big    -N2  procs 8       版本 final final_nocap v4   ← big 8 個 process（每節點只有 4 個 big core，必須 2 節點）
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
sed -e 's/constexpr int MAX_ACTIVE_RANKS = 4;/constexpr int MAX_ACTIVE_RANKS = 0;/' \
    -e 's/long long active = count_ranks_on_first_node(rank, size);/long long active = size;/' \
    hw1.cc > "$B/final_nocap.cc"
grep -q 'MAX_ACTIVE_RANKS = 0;' "$B/final_nocap.cc" && grep -q 'long long active = size;' "$B/final_nocap.cc" \
    || { echo "產生 final_nocap 失敗：hw1.cc 的寫法和預期不同"; exit 1; }
git show 68eda46:hw1.cc > "$B/v4.cc"
cp test/exp/baseline_ori.cc "$B/ori.cc"

for v in final final_nocap v4 ori; do
    mpicxx -O3 -I. $WRAP -o "$B/$v" "$B/$v.cc"
done
echo "已編譯：final final_nocap v4 ori（$B/）"

# ---- 送出 job -------------------------------------------------------------------
export REPO TRIALS=${TRIALS:-5} CASE_ID=${CASE_ID:-10}
SUFFIX=$([ "$CASE_ID" = 10 ] || echo "_case$CASE_ID")
ONLY=${ONLY:-"big little mixed big2n"}
submit() {   # submit <名字> <sbatch 參數> <procs> <versions> <時間上限>
    local name=$1 opts=$2 procs=$3 versions=$4 limit=$5
    [[ " $ONLY " == *" $name "* ]] || return 0
    EXP_NAME=$name$SUFFIX PROCS=$procs VERSIONS=$versions \
        sbatch --parsable -J "hw1exp-$name" $opts ${EXCLUSIVE:+--exclusive} -t "$limit" -o "test/exp/results/slurm_${name}_%j.txt" test/exp/job.sh \
        | sed "s/^/送出 $name，job id = /"
}
submit big    "-p big -N1 -n4"    "1 2 4" "final v4 ori"            "40:00"
submit little "-p little -N1 -n4" "1 2 4" "final v4 ori"            "50:00"
submit mixed  "-p mixed -N1 -n8"  "8 4 1" "final final_nocap v4"   "30:00"
submit big2n  "-p big -N2 -n8"    "8"     "final final_nocap v4"    "20:00"
echo
echo "用 squeue -u \$USER 查看進度；全部結束後執行：python3 test/exp/summarize.py"
