#!/bin/bash
# =============================================================================
# 比較「最多幾個 rank 參與排序」（hw1.cc 的 MAX_ACTIVE_RANKS）哪個最快。
#
# 不會修改 hw1.cc：把 hw1.cc 複製到 test/build/，只改複製檔裡的常數再編譯。
#
# 用法（在課程機器上，repo 根目錄）：
#   bash test/sweep_active.sh                  # 預設測資、上限 1 2 4 8、每組 3 次
#   CAPS="1 2 3 4" bash test/sweep_active.sh 10 30
#   REPEAT=5 bash test/sweep_active.sh
#
# 每筆測資照 NN.txt 的 partition / 節點數 / process 數執行；
# 上限只決定其中幾個 rank 分到資料。表格裡是每組的中位數時間（秒）。
# =============================================================================
set -u
cd "$(dirname "$0")/.."

CASES=${CASES:-/srv/nova/scratch/coursedata/pp2026/hw1}
CAPS=${CAPS:-"1 2 4 8"}
REPEAT=${REPEAT:-3}
BUILD=$PWD/test/build
OUT=$PWD/test/out          # 必須是所有節點都看得到的位置（不能用 /tmp）
mkdir -p "$BUILD" "$OUT"

if [ $# -gt 0 ]; then ids=("$@"); else ids=(08 10 18 20 28 30 33 34 35); fi
[ -f hash.h ] || { echo "找不到 hash.h，請先放到 repo 根目錄"; exit 1; }

# ---- 編譯各個版本 -------------------------------------------------------------
for cap in $CAPS; do
    sed "s/constexpr int MAX_ACTIVE_RANKS = [0-9]*;/constexpr int MAX_ACTIVE_RANKS = $cap;/" hw1.cc > "$BUILD/hw1_cap$cap.cc"
    grep -q "MAX_ACTIVE_RANKS = $cap;" "$BUILD/hw1_cap$cap.cc" || { echo "hw1.cc 裡找不到 MAX_ACTIVE_RANKS"; exit 1; }
    mpicxx -O3 -I. -o "$BUILD/hw1_cap$cap" "$BUILD/hw1_cap$cap.cc" || exit 1
done
echo "已編譯：$(for c in $CAPS; do printf 'cap%s ' "$c"; done)"

# ---- 執行 -----------------------------------------------------------------------
printf "%-5s %-9s %-14s" case N "part/N/procs"
for cap in $CAPS; do printf "%-10s" "cap=$cap"; done; echo

for id in "${ids[@]}"; do
    read -r n nodes procs part rounds < <(python3 test/parse_case.py "$CASES/$id.txt") || continue
    printf "%-5s %-9s %-14s" "$id" "$n" "$part/$nodes/$procs"
    for cap in $CAPS; do
        times=(); ok=1
        for ((t = 1; t <= REPEAT; ++t)); do
            start=$(date +%s.%N)
            srun -p "$part" -N"$nodes" -n"$procs" "$BUILD/hw1_cap$cap" "$n" "$CASES/$id.in" "$OUT/sweep.out" "$rounds" \
                > "$OUT/sweep.log" 2>&1 || ok=0
            times+=("$(python3 -c "print($(date +%s.%N) - $start)")")
            python3 test/fcmp.py "$CASES/$id.out" "$OUT/sweep.out" > /dev/null || ok=0
        done
        med=$(printf '%s\n' "${times[@]}" | python3 -c "import sys,statistics as s; print(f'{s.median(float(x) for x in sys.stdin):.2f}')")
        [ $ok = 1 ] && printf "%-10s" "$med" || printf "%-10s" "${med}✗"
    done
    echo
done
echo "（✗ = 輸出錯誤或執行失敗；時間包含 srun 啟動，約 0.7 秒）"
