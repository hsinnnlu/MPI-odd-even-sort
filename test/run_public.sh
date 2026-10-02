#!/bin/bash
# 跑課程提供的公開測資，檢查正確性並記錄時間。
#
# 用法（在 repo 根目錄、已 make 出 ./hw1 之後）：
#   bash test/run_public.sh              # 跑全部
#   bash test/run_public.sh 01 05 17     # 只跑指定編號
#
# 可用環境變數調整：
#   CASES=測資目錄（預設 /srv/nova/scratch/coursedata/pp2026/hw1）
#   PARTITION=big|little|mixed  強制指定 partition（預設用 NN.txt 裡的設定）
#   LAUNCH=mpirun               沒有 srun 時（例如自己電腦）改用 mpirun
set -u
cd "$(dirname "$0")/.."

CASES=${CASES:-/srv/nova/scratch/coursedata/pp2026/hw1}
OUT_DIR=${OUT_DIR:-/tmp/${USER:-$(id -un)}-hw1-test}
LAUNCH=${LAUNCH:-srun}
mkdir -p "$OUT_DIR"

if [ ! -x ./hw1 ]; then echo "找不到 ./hw1，請先 make"; exit 1; fi

if [ $# -gt 0 ]; then ids=("$@"); else
    ids=(); for f in "$CASES"/*.txt; do ids+=("$(basename "$f" .txt)"); done
fi

pass=0; fail=0
printf "%-6s %-10s %-6s %-6s %-8s %-7s %-9s %s\n" case N nodes procs part rounds time result
for id in "${ids[@]}"; do
    read -r n nodes procs part rounds < <(python3 test/parse_case.py "$CASES/$id.txt") || { echo "$id: 無法解析 $CASES/$id.txt"; fail=$((fail+1)); continue; }
    part=${PARTITION:-$part}
    out="$OUT_DIR/$id.out"
    # 先寫入一些垃圾，順便檢查程式有沒有把舊檔案截斷
    head -c 4096 /dev/urandom > "$out"

    if [ "$LAUNCH" = srun ]; then
        cmd=(srun -p "$part" -N"$nodes" -n"$procs" ./hw1 "$n" "$CASES/$id.in" "$out" "$rounds")
    else
        cmd=(mpirun -np "$procs" ./hw1 "$n" "$CASES/$id.in" "$out" "$rounds")
    fi

    start=$(date +%s.%N)
    "${cmd[@]}" > "$OUT_DIR/$id.log" 2>&1
    rc=$?
    elapsed=$(echo "$(date +%s.%N) - $start" | bc)

    if [ $rc -ne 0 ]; then
        result="FAIL (exit $rc, 見 $OUT_DIR/$id.log)"; fail=$((fail+1))
    elif msg=$(python3 test/fcmp.py "$CASES/$id.out" "$out"); then
        result="OK"; pass=$((pass+1))
    else
        result="WRONG: $msg"; fail=$((fail+1))
    fi
    printf "%-6s %-10s %-6s %-6s %-8s %-7s %-9.3f %s\n" "$id" "$n" "$nodes" "$procs" "$part" "$rounds" "$elapsed" "$result"
done
echo "通過 $pass / $((pass+fail))"
[ $fail -eq 0 ]
