#!/bin/bash
# =============================================================================
# Strong Scaling Experiment
#
# Purpose:
#   Run controlled strong-scaling experiments using the SAME storage policy.
#
# Storage:
#   All configurations use shared NFS.
#
# Intended configurations:
#   p1 / p2 / p4 : 1 node
#   p8           : 2 nodes
#
# Environment variables:
#   REPO       repository path
#   EXP_NAME   result file prefix
#   VERSIONS   executable versions, normally "final"
#   PROCS      process counts
#   TRIALS     number of trials, default 5
#   CASE_ID    public test case, default 10
#
# Output:
#   test/exp/results/<EXP_NAME>_<job id>.csv
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


# -----------------------------------------------------------------------------
# Read test-case information
# -----------------------------------------------------------------------------

read -r N _nodes _procs _part ROUNDS < <(
    python3 test/parse_case.py "$CASES/$CASE_ID.txt"
)


# -----------------------------------------------------------------------------
# Shared NFS storage
#
# IMPORTANT:
# Strong scaling must use the same storage policy for p1/p2/p4/p8.
# Therefore even single-node runs use this shared NFS directory.
# -----------------------------------------------------------------------------

WORK=$REPO/test/exp/scaling_work_${SLURM_JOB_ID}
STORAGE="shared NFS $WORK"

mkdir -p "$WORK"

cp "$CASES/$CASE_ID.in" "$WORK/in" || {
    echo "Input staging failed" >&2
    exit 1
}


# -----------------------------------------------------------------------------
# Cleanup
# -----------------------------------------------------------------------------

cleanup() {
    rm -rf "$WORK"
}

trap cleanup EXIT


# -----------------------------------------------------------------------------
# Multi-node Open MPI network configuration
#
# Some interfaces may have identical addresses on different nodes.
# Exclude these interfaces to avoid Open MPI connecting through them.
# -----------------------------------------------------------------------------

if [ "$NODES" -gt 1 ] && [ -z "${OMPI_MCA_btl_tcp_if_exclude:-}" ]; then

    bad=$(
        srun -N"$NODES" \
             --ntasks-per-node=1 \
             -c 1 \
             -l ip -4 -o addr show 2>/dev/null |
        awk '
            $3 != "lo" {
                split($5, a, "/");
                seen[a[1]]++;
                name[a[1]] = $3
            }

            END {
                for (ip in seen)
                    if (seen[ip] > 1)
                        print name[ip]
            }
        ' |
        sort -u |
        tr '\n' ','
    )

    export OMPI_MCA_btl_tcp_if_exclude="lo${bad:+,${bad%,}}"
fi


# -----------------------------------------------------------------------------
# Write experiment metadata
# -----------------------------------------------------------------------------

{
    echo "# job=$SLURM_JOB_ID partition=$PART nodes=$NODES nodelist=$SLURM_JOB_NODELIST"

    echo "# case=$CASE_ID N=$N rounds=$ROUNDS trials=$TRIALS versions=[$VERSIONS] procs=[$PROCS]"

    echo "# modules: ${LOADEDMODULES:-not recorded}"

    echo "# mpicxx: $(mpicxx --version 2>/dev/null | head -1) / $(mpirun --version 2>/dev/null | head -1)"

    echo "# storage: input and output in $STORAGE (deleted at job end)"

    echo "# OMPI_MCA_btl_tcp_if_exclude=${OMPI_MCA_btl_tcp_if_exclude:-not set}"

} > "$CSV"


# -----------------------------------------------------------------------------
# Run experiments
#
# Each MPI process receives one CPU (-c 1).
#
# PROF lines:
#   Produced by prof_wrap.h
#
# WALL lines:
#   External timing around srun
# -----------------------------------------------------------------------------

for ((t = 1; t <= TRIALS; ++t)); do

    for ver in $VERSIONS; do

        for p in $PROCS; do

            tag="$ver/$PART/N$NODES/p$p/t$t/c$CASE_ID"

            out=$WORK/out_${ver}_p${p}_t${t}

            start=$(date +%s.%N)

            EXP_TAG=$tag \
            srun \
                -N"$NODES" \
                -n"$p" \
                -c 1 \
                "$BIN/$ver" \
                "$N" \
                "$WORK/in" \
                "$out" \
                "$ROUNDS" \
                2> >(
                    tee -a "$LOG" |
                    grep '^PROF' >> "$CSV"
                ) \
                > /dev/null

            rc=$?

            end=$(date +%s.%N)

            wall=$(
                python3 -c "print(f'{$end - $start:.4f}')"
            )


            # -----------------------------------------------------------------
            # Correctness check
            #
            # The output is stored on shared NFS, so rank 0's node can compare
            # it directly with the reference output.
            # -----------------------------------------------------------------

            if [ "$rc" -eq 0 ] && \
               cmp -s "$out" "$CASES/$CASE_ID.out"; then

                ok=OK

            else

                ok=WRONG

            fi


            echo "WALL,$tag,$wall,$ok" >> "$CSV"

            echo "$tag  wall=${wall}s  $ok"

        done

    done

done


# Give process-substitution / tee time to flush the final PROF lines.
sleep 1
