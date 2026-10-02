#!/bin/bash
# 用 Open MPI 的兩種 MPI-IO 實作（OMPIO = 預設、ROMIO）各跑一次 test/bench_io。
# 用法（課程機器，repo 根目錄）：bash test/io_matrix.sh [N] [input]
set -u
cd "$(dirname "$0")/.."
N=${1:-23987513}
IN=${2:-/srv/nova/scratch/coursedata/pp2026/hw1/10.in}
mkdir -p test/out

if ! type module >/dev/null 2>&1; then
    for f in /etc/profile.d/lmod.sh /etc/profile.d/modules.sh /usr/share/lmod/lmod/init/bash; do
        [ -f "$f" ] && source "$f" && break
    done
fi
module purge >/dev/null 2>&1
module load compiler/gcc/13 mpi/openmpi/5.0.10 || exit 1
mpicxx -O3 -I. -o test/bench_io test/bench_io.cc || exit 1

romio=$(ompi_info 2>/dev/null | grep -o 'MCA io: romio[0-9]*' | awk '{print $3}' | head -1)
echo "輸入檔所在的檔案系統：$(df -T "$IN" | tail -1 | awk '{print $2, $1}')"
echo "輸出目錄所在的檔案系統：$(df -T test/out | tail -1 | awk '{print $2, $1}')"
echo
echo "===== OMPIO（Open MPI 預設）====="
srun -p big -N1 -n4 ./test/bench_io "$N" "$IN" test/out
if [ -n "$romio" ]; then
    echo
    echo "===== ROMIO（OMPI_MCA_io=$romio）====="
    OMPI_MCA_io=$romio srun -p big -N1 -n4 ./test/bench_io "$N" "$IN" test/out
else
    echo "（ompi_info 裡找不到 romio 元件，略過）"
fi
rm -f test/out/bench_io.out
