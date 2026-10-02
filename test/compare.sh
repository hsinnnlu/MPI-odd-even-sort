#!/bin/bash
# =============================================================================
# 兩個版本的 hw1.cc 正面對決：在同樣的測資上「交替」執行，比較中位數時間。
#
# 為什麼要交替：judge 機器的負載會變動，同一份程式隔一段時間再跑可能差 30%。
# A、B、A、B…… 輪流跑，兩邊遇到的負載比較接近，比較結果才可信。
#
# 用法（在課程機器上，repo 根目錄）：
#   bash test/compare.sh                       # 目前的 hw1.cc vs 原本的 v4
#   bash test/compare.sh 10 20 30              # 只比較指定測資
#   A=<git commit> B=<git commit> bash test/compare.sh   # 任選兩個 commit
#   REPEAT=7 bash test/compare.sh
# =============================================================================
set -u
cd "$(dirname "$0")/.."

CASES=${CASES:-/srv/nova/scratch/coursedata/pp2026/hw1}
A=${A:-68eda46}       # 預設：原本的 v4
B=${B:-WORKTREE}      # 預設：目前工作目錄的 hw1.cc
REPEAT=${REPEAT:-5}
BUILD=$PWD/test/build
OUT=$PWD/test/out     # 必須是所有節點都看得到的位置（不能用 /tmp）
mkdir -p "$BUILD" "$OUT"
[ -f hash.h ] || { echo "找不到 hash.h，請先放到 repo 根目錄"; exit 1; }

build() {   # build <commit|WORKTREE> <輸出名稱>
    if [ "$1" = WORKTREE ]; then cp hw1.cc "$BUILD/$2.cc"
    else git show "$1:hw1.cc" > "$BUILD/$2.cc" || exit 1; fi
    mpicxx -O3 -I. -o "$BUILD/$2" "$BUILD/$2.cc" || exit 1
}
build "$A" verA
build "$B" verB
echo "A = $A    B = $B    （每筆各 $REPEAT 次，交替執行）"

if [ $# -gt 0 ]; then ids=("$@"); else
    ids=(); for f in "$CASES"/*.txt; do ids+=("$(basename "$f" .txt)"); done
fi

run_once() {   # run_once <binary>  → 印出秒數；輸出錯誤時印 ERR
    local start; start=$(date +%s.%N)
    srun -p "$part" -N"$nodes" -n"$procs" "$1" "$n" "$CASES/$id.in" "$OUT/cmp.out" "$rounds" \
        > "$OUT/cmp.log" 2>&1 || { echo ERR; return; }
    python3 test/fcmp.py "$CASES/$id.out" "$OUT/cmp.out" > /dev/null || { echo ERR; return; }
    python3 -c "print($(date +%s.%N) - $start)"
}
median() { python3 -c "import sys,statistics as s; v=[float(x) for x in sys.argv[1:] if x!='ERR']; print(f'{s.median(v):.2f}' if v else 'ERR')" "$@"; }

printf "%-5s %-9s %-13s %-7s %-7s %s\n" case N part/N/procs A B "B-A"
sumA=0; sumB=0
for id in "${ids[@]}"; do
    read -r n nodes procs part rounds < <(python3 test/parse_case.py "$CASES/$id.txt") || continue
    ta=(); tb=()
    for ((t = 1; t <= REPEAT; ++t)); do
        ta+=("$(run_once "$BUILD/verA")")
        tb+=("$(run_once "$BUILD/verB")")
    done
    ma=$(median "${ta[@]}"); mb=$(median "${tb[@]}")
    diff=$(python3 -c "print(f'{$mb - $ma:+.2f}')" 2>/dev/null || echo "?")
    printf "%-5s %-9s %-13s %-7s %-7s %s\n" "$id" "$n" "$part/$nodes/$procs" "$ma" "$mb" "$diff"
    sumA=$(python3 -c "print($sumA + ${ma/ERR/0})"); sumB=$(python3 -c "print($sumB + ${mb/ERR/0})")
done
python3 -c "print(f'總計  A = {$sumA:.2f}   B = {$sumB:.2f}   B-A = {$sumB - $sumA:+.2f}')"
echo "（ERR = 執行失敗或輸出錯誤；B-A 為負代表 B 比較快）"
