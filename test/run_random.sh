#!/bin/bash
# 自己產生隨機測資，和單機參考答案 (test/ref) 比對。
# 用來測公開測資沒涵蓋到的情況：很小的 N、process 比資料多、重複值、+0/-0 ……
#
# 用法（在 repo 根目錄）：
#   bash test/run_random.sh
# 可用環境變數：
#   LAUNCH="srun -p big -N1"   每次執行前面加的指令（預設 srun -p mixed -N1）
#                               自己電腦上用 LAUNCH="mpirun --oversubscribe"
#   PROCS="1 2 3 4 5 8"        要測的 process 數
#   SIZES="1 2 3 7 100 ..."    要測的 N
set -u
cd "$(dirname "$0")/.."

LAUNCH=${LAUNCH:-"srun -p mixed -N1"}
PROCS=${PROCS:-"1 2 3 4 5 8"}
SIZES=${SIZES:-"1 2 3 7 100 4097 9000 50000 300001 2000003"}
MODES=${MODES:-"uniform wide dup sorted reverse zeros"}
ROUNDS=${ROUNDS:-"1 2 25"}
# 不能用 /tmp：srun 在 compute node 執行，看不到 login node 的 /tmp（見 run_public.sh）
WORK=${WORK:-$PWD/test/work}
mkdir -p "$WORK"

if [ ! -x ./hw1 ] || [ ! -x test/ref ]; then echo "請先 make（需要 ./hw1 和 test/ref）"; exit 1; fi

pass=0; fail=0
for mode in $MODES; do
  for n in $SIZES; do
    python3 test/gen.py "$n" "$WORK/in" "$mode"
    for r in $ROUNDS; do
      ./test/ref "$n" "$WORK/in" "$WORK/ans" "$r"
      for p in $PROCS; do
        head -c 4096 /dev/urandom > "$WORK/out"     # 檢查舊資料有被截斷
        if $LAUNCH -n "$p" ./hw1 "$n" "$WORK/in" "$WORK/out" "$r" > "$WORK/log" 2>&1 \
           && msg=$(python3 test/fcmp.py "$WORK/ans" "$WORK/out"); then
          pass=$((pass+1))
        else
          fail=$((fail+1))
          echo "FAIL mode=$mode N=$n rounds=$r procs=$p ${msg:-}"; head -5 "$WORK/log"
        fi
        msg=""
      done
    done
  done
  echo "mode=$mode 完成（目前 通過 $pass，失敗 $fail）"
done
echo "總計：通過 $pass / $((pass+fail))"
rm -rf "$WORK"
[ $fail -eq 0 ]
