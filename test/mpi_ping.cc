// 最小的 MPI 連線測試：和 hw1 無關，只檢查「跨節點的 MPI 通訊」本身能不能動。
//
// 每個 rank 和左右鄰居交換一次訊息（和 odd-even sort 的通訊模式一樣），
// 再做一次 Allreduce。全部成功時，rank 0 會印出 "mpi_ping OK"。
//
// 編譯：mpicxx -O2 -o test/mpi_ping test/mpi_ping.cc
// 執行：srun -p big -N2 -n8 ./test/mpi_ping
#include <mpi.h>

#include <cstdio>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1, len = 0;
    char host[MPI_MAX_PROCESSOR_NAME];
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Get_processor_name(host, &len);
    std::printf("rank %d on %s\n", rank, host);

    // 和右邊、左邊的鄰居各交換一次（不存在的鄰居用 MPI_PROC_NULL，呼叫會直接略過）
    const int right = rank + 1 < size ? rank + 1 : MPI_PROC_NULL;
    const int left = rank - 1 >= 0 ? rank - 1 : MPI_PROC_NULL;
    int from_left = -1, from_right = -1;
    MPI_Sendrecv(&rank, 1, MPI_INT, right, 0, &from_left, 1, MPI_INT, left, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(&rank, 1, MPI_INT, left, 1, &from_right, 1, MPI_INT, right, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    int sum = 0;
    MPI_Allreduce(&rank, &sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0)
        std::printf("mpi_ping OK: %d ranks, sum of ranks = %d (expected %d)\n",
                    size, sum, size * (size - 1) / 2);
    MPI_Finalize();
    return 0;
}
