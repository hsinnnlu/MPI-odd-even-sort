#!/bin/bash
# =============================================================================
# 一個實驗 job（由 test/exp/submit_all.sh 用 sbatch 送出，不要直接執行）。
#
# 依作業 5.3.1 的 Storage 規定：
#   1. 把輸入檔複製到每個節點的 /tmp/hw1exp_<job id>/（node-local）
#   2. 輸出也寫在那裡
#   3. job 結束時（不論成功失敗）刪除整個目錄
#
# 由環境變數控制（submit_all.sh 會設定）：
#   EXP_NAME   這個 job 的名字（結果檔名用）
#   VERSIONS   要比較的版本（test/build/exp/ 底下的執行檔名）
#   PROCS      要跑的 process 數，例如 "1 2 4"
#   TRIALS     每個設定的重複次數（預設 5）
#   CASE_ID    使用的公開測資編號（預設 10，N = 23,987,513）
#
# 每次執行寫兩種紀錄到 test/exp/results/<EXP_NAME>_<job id>.csv：
#   PROF,...  每個 rank 的 total/io/comm/sync/compute（由 prof_wrap.h 印出）
#   WALL,...  srun 整體時間（含啟動）與正確性
# =============================================================================
set -u
REPO=${REPO:?}
cd "$REPO"
CASES=${CASES:-/srv/nova/scratch/coursedata/pp2026/hw1}
TRIALS=${TRIALS:-5}
CASE_ID=${CASE_ID:-10}
PART=${SLURM_JOB_PARTITION:-unknown}
NODES=${SLURM_JOB_NUM_NODES:-1}
BIN=$REPO/test/build/exp
RESULTS=$REPO/test/exp/results
mkdir -p "$RESULTS"
CSV=$RESULTS/${EXP_NAME}_${SLURM_JOB_ID}.csv
LOG=$RESULTS/${EXP_NAME}_${SLURM_JOB_ID}.log

read -r N _nodes _procs _part ROUNDS < <(python3 test/parse_case.py "$CASES/$CASE_ID.txt")
WORK=/tmp/hw1exp_${SLURM_JOB_ID}

# ---- 1. 每個節點都放一份輸入（2 節點的 job 兩台都要有） ----
srun -N"$NODES" --ntasks-per-node=1 bash -c "mkdir -p $WORK && cp $CASES/$CASE_ID.in $WORK/in" || { echo "staging 失敗" >&2; exit 1; }
cleanup() { srun -N"$NODES" --ntasks-per-node=1 rm -rf "$WORK"; }
trap cleanup EXIT

{
    echo "# job=$SLURM_JOB_ID partition=$PART nodes=$NODES nodelist=$SLURM_JOB_NODELIST"
    echo "# case=$CASE_ID N=$N rounds=$ROUNDS trials=$TRIALS versions=[$VERSIONS] procs=[$PROCS]"
    echo "# modules: $(module -t list 2>&1 | tr '\n' ' ')"
    echo "# storage: input and output in node-local $WORK (deleted at job end)"
} > "$CSV"

# ---- 2. 執行：每個 trial 裡把所有版本與 process 數輪流跑一遍，讓各設定遇到的負載接近 ----
for ((t = 1; t <= TRIALS; ++t)); do
    for ver in $VERSIONS; do
        for p in $PROCS; do
            tag="$ver/$PART/N$NODES/p$p/t$t"
            out=$WORK/out_$ver
            start=$(date +%s.%N)
            EXP_TAG=$tag srun -N"$NODES" -n"$p" "$BIN/$ver" "$N" "$WORK/in" "$out" "$ROUNDS" \
                2> >(tee -a "$LOG" | grep '^PROF' >> "$CSV") > /dev/null
            rc=$?
            wall=$(python3 -c "print(f'{$(date +%s.%N) - $start:.4f}')")
            # 只有 rank 0 所在節點（第一台）有完整輸出；用 srun 在第一台上比對
            if [ $rc -eq 0 ] && srun -N1 -n1 -w "$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)" \
                    cmp -s "$out" "$CASES/$CASE_ID.out"; then ok=OK; else ok=WRONG; fi
            echo "WALL,$tag,$wall,$ok" >> "$CSV"
            echo "$tag  wall=${wall}s  $ok"
        done
    done
done
sleep 1   # 讓 tee 把最後幾行寫完
