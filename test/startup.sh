#!/bin/bash
# =============================================================================
# 比較不同 MPI 實作（Open MPI / Intel MPI）下 hw1 的固定開銷與整體時間。
#
# 為什麼：N=1 的測資幾乎沒有計算，時間 = srun + MPI 啟動 + 檔案開關。
# 我們 N=1 約 0.75~0.80 s，第一名約 0.50~0.57 s；40 筆都會付這個成本。
# 課程 Lab1 的 judge 預設用 Intel MPI，所以先量兩種 MPI 的差別。
#
# 用法（課程機器，repo 根目錄，需要 hash.h）：
#   bash test/startup.sh               # 預設測資 01 10 30
#   REPEAT=7 bash test/startup.sh 01 21 10
# =============================================================================
set -u
cd "$(dirname "$0")/.."
CASES=${CASES:-/srv/nova/scratch/coursedata/pp2026/hw1}
REPEAT=${REPEAT:-5}
OUT=$PWD/test/out
mkdir -p "$OUT" test/build
[ -f hash.h ] || { echo "找不到 hash.h"; exit 1; }
if [ $# -gt 0 ]; then ids=("$@"); else ids=(01 10 30); fi

if ! type module >/dev/null 2>&1; then
    for f in /etc/profile.d/lmod.sh /etc/profile.d/modules.sh /usr/share/lmod/lmod/init/bash; do
        [ -f "$f" ] && source "$f" && break
    done
fi

median() { python3 -c "import sys,statistics as s; v=[float(x) for x in sys.argv[1:] if x!='ERR']; print(f'{s.median(v):.3f}' if v else 'ERR')" "$@"; }

for spec in "openmpi:mpi/openmpi/5.0.10" "intel:mpi/intel"; do
    name=${spec%%:*}; mod=${spec#*:}
    module purge >/dev/null 2>&1
    if ! module load compiler/gcc/13 "$mod" >/dev/null 2>&1; then echo "== $name：無法載入 $mod，略過"; continue; fi
    bin=test/build/hw1_$name
    mpicxx -O3 -I. -o "$bin" hw1.cc 2> "$OUT/build_$name.log" || { echo "== $name：編譯失敗（見 $OUT/build_$name.log）"; continue; }
    echo "== $name（$(command -v mpicxx)）"
    for id in "${ids[@]}"; do
        read -r n nodes procs part rounds < <(python3 test/parse_case.py "$CASES/$id.txt") || continue
        ts=()
        for ((t = 1; t <= REPEAT; ++t)); do
            s=$(date +%s.%N)
            if srun -p "$part" -N"$nodes" -n"$procs" "$bin" "$n" "$CASES/$id.in" "$OUT/startup.out" "$rounds" > "$OUT/startup_$name.log" 2>&1 \
               && python3 test/fcmp.py "$CASES/$id.out" "$OUT/startup.out" > /dev/null; then
                ts+=("$(python3 -c "print($(date +%s.%N) - $s)")")
            else
                ts+=(ERR)
            fi
        done
        printf "   case %s  N=%-9s %-6s %s/%s procs   median %s s   （各次：%s）\n" "$id" "$n" "$part" "$nodes" "$procs" "$(median "${ts[@]}")" "${ts[*]}"
    done
done
echo "（ERR = 執行失敗或輸出錯誤；Intel MPI 若失敗，log 在 test/out/startup_intel.log）"
