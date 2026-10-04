#!/bin/bash
# =============================================================================
# Submit controlled strong-scaling experiments.
#
# p1 / p2 / p4 : 1 big-core node
# p8           : 2 big-core nodes
#
# All configurations use the same shared NFS storage policy through
# job_scaling.sh.
# =============================================================================

set -e

REPO=$(pwd)

mkdir -p test/exp/results

# -----------------------------------------------------------------------------
# 1 node: p1, p2, p4
# -----------------------------------------------------------------------------

echo "Submitting single-node strong-scaling job..."

sbatch \
    -p big \
    -N 1 \
    -n 4 \
    -c 1 \
    -t 00:10:00 \
    --export=ALL,REPO="$REPO",EXP_NAME=strong_scaling_1n,VERSIONS=final,PROCS="1 2 4",TRIALS=5,CASE_ID=10 \
    test/exp/job_scaling.sh


# -----------------------------------------------------------------------------
# 2 nodes: p8
# -----------------------------------------------------------------------------

echo "Submitting two-node strong-scaling job..."

sbatch \
    -p big \
    -N 2 \
    -n 8 \
    --ntasks-per-node=4 \
    -c 1 \
    -t 00:10:00 \
    --export=ALL,REPO="$REPO",EXP_NAME=strong_scaling_2n,VERSIONS=final,PROCS=8,TRIALS=5,CASE_ID=10 \
    test/exp/job_scaling.sh

echo "Done."
