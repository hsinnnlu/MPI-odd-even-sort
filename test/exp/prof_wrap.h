// =============================================================================
// 實驗用的計時外掛：不修改任何一版 hw1.cc，編譯時用 -include 強制載入。
//
//   mpicxx -O3 -I. -include test/exp/prof_wrap.h -o hw1_prof hw1.cc
//
// 做法：先載入 mpi.h，再用 #define 把程式裡的 MPI 呼叫換成下面的計時版本
// （計時版本內部呼叫 PMPI_* 做真正的工作）。程式之後再 #include <mpi.h>
// 會被 include guard 擋掉，所以只有「呼叫」被替換。
//
// 分類（每個 rank 各自累計）：
//   io    MPI_File_*（open、read、set_size、write、close）
//   comm  MPI_Sendrecv（odd-even 的鄰居交換；含等待對方的時間）
//   sync  Allreduce、Barrier、Bcast、Allgather、Exscan、Reduce（終止判斷與初始化的 collective）
//   total MPI_Init 結束（先做一次 Barrier）到 MPI_Finalize 開始（先做一次 Barrier）
//   compute = total - io - comm - sync（local sort、merge、hash、bucket 重組……）
//
// MPI_Finalize 時把每個 rank 的數字收集到 rank 0，印出 CSV（stderr）：
//   PROF,tag,rank,host,cpu,total,io,comm,sync,compute,sendrecv_calls,sendrecv_MB
// tag 來自環境變數 EXP_TAG（由實驗腳本設定，例如 final/big/N1/p4/t3）。
// =============================================================================
#pragma once
#include <mpi.h>
#include <sched.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace prof {
inline double t_start = 0, t_io = 0, t_comm = 0, t_sync = 0;
inline long long sendrecv_calls = 0;
inline double sendrecv_bytes = 0;
inline bool finalized = false;

struct Scope {
    double& acc;
    double s;
    explicit Scope(double& a) : acc(a), s(PMPI_Wtime()) {}
    ~Scope() { acc += PMPI_Wtime() - s; }
};
}  // namespace prof

// ---- 生命週期 -----------------------------------------------------------------
inline int prof_Init(int* argc, char*** argv) {
    const int rc = PMPI_Init(argc, argv);
    PMPI_Barrier(MPI_COMM_WORLD);
    prof::t_start = PMPI_Wtime();
    return rc;
}

inline int prof_Finalize() {
    if (prof::finalized) return PMPI_Finalize();
    prof::finalized = true;
    PMPI_Barrier(MPI_COMM_WORLD);
    const double total = PMPI_Wtime() - prof::t_start;

    int rank = 0, size = 1;
    PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
    PMPI_Comm_size(MPI_COMM_WORLD, &size);
    double mine[7] = {total, prof::t_io, prof::t_comm, prof::t_sync,
                      total - prof::t_io - prof::t_comm - prof::t_sync,
                      static_cast<double>(prof::sendrecv_calls), prof::sendrecv_bytes / 1e6};
    char host[MPI_MAX_PROCESSOR_NAME] = {};
    int len = 0;
    PMPI_Get_processor_name(host, &len);
    const int cpu = sched_getcpu();

    std::vector<double> all(rank == 0 ? 7 * size : 0);
    std::vector<char> hosts(rank == 0 ? static_cast<size_t>(MPI_MAX_PROCESSOR_NAME) * size : 0);
    std::vector<int> cpus(rank == 0 ? size : 0);
    PMPI_Gather(mine, 7, MPI_DOUBLE, all.data(), 7, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    PMPI_Gather(host, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, hosts.data(), MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 0, MPI_COMM_WORLD);
    PMPI_Gather(&cpu, 1, MPI_INT, cpus.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        const char* tag = std::getenv("EXP_TAG");
        for (int r = 0; r < size; ++r) {
            const double* v = all.data() + 7 * r;
            std::fprintf(stderr, "PROF,%s,%d,%s,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.0f,%.1f\n",
                         tag ? tag : "untagged", r, hosts.data() + static_cast<size_t>(MPI_MAX_PROCESSOR_NAME) * r,
                         cpus[r], v[0], v[1], v[2], v[3], v[4], v[5], v[6]);
        }
        std::fflush(stderr);
    }
    return PMPI_Finalize();
}

// ---- MPI-IO -------------------------------------------------------------------
inline int prof_File_open(MPI_Comm c, const char* f, int m, MPI_Info i, MPI_File* fh) {
    prof::Scope s(prof::t_io); return PMPI_File_open(c, f, m, i, fh);
}
inline int prof_File_close(MPI_File* fh) { prof::Scope s(prof::t_io); return PMPI_File_close(fh); }
inline int prof_File_set_size(MPI_File fh, MPI_Offset n) { prof::Scope s(prof::t_io); return PMPI_File_set_size(fh, n); }
inline int prof_File_read_at_all(MPI_File fh, MPI_Offset o, void* b, int n, MPI_Datatype t, MPI_Status* st) {
    prof::Scope s(prof::t_io); return PMPI_File_read_at_all(fh, o, b, n, t, st);
}
inline int prof_File_read_at(MPI_File fh, MPI_Offset o, void* b, int n, MPI_Datatype t, MPI_Status* st) {
    prof::Scope s(prof::t_io); return PMPI_File_read_at(fh, o, b, n, t, st);
}
inline int prof_File_write_at_all(MPI_File fh, MPI_Offset o, const void* b, int n, MPI_Datatype t, MPI_Status* st) {
    prof::Scope s(prof::t_io); return PMPI_File_write_at_all(fh, o, b, n, t, st);
}
inline int prof_File_write_at(MPI_File fh, MPI_Offset o, const void* b, int n, MPI_Datatype t, MPI_Status* st) {
    prof::Scope s(prof::t_io); return PMPI_File_write_at(fh, o, b, n, t, st);
}

// ---- 點對點通訊 -----------------------------------------------------------------
inline int prof_Sendrecv(const void* sb, int sc, MPI_Datatype st, int dst, int stag,
                         void* rb, int rc, MPI_Datatype rt, int src, int rtag, MPI_Comm c, MPI_Status* s) {
    prof::Scope sc_(prof::t_comm);
    int sz = 0;
    PMPI_Type_size(st, &sz);
    ++prof::sendrecv_calls;
    prof::sendrecv_bytes += static_cast<double>(sc) * sz;
    return PMPI_Sendrecv(sb, sc, st, dst, stag, rb, rc, rt, src, rtag, c, s);
}

// ---- collective（同步／終止判斷／初始化）--------------------------------------
inline int prof_Allreduce(const void* s, void* r, int n, MPI_Datatype t, MPI_Op o, MPI_Comm c) {
    prof::Scope sc(prof::t_sync); return PMPI_Allreduce(s, r, n, t, o, c);
}
inline int prof_Barrier(MPI_Comm c) { prof::Scope sc(prof::t_sync); return PMPI_Barrier(c); }
inline int prof_Bcast(void* b, int n, MPI_Datatype t, int root, MPI_Comm c) {
    prof::Scope sc(prof::t_sync); return PMPI_Bcast(b, n, t, root, c);
}
inline int prof_Allgather(const void* s, int sn, MPI_Datatype st, void* r, int rn, MPI_Datatype rt, MPI_Comm c) {
    prof::Scope sc(prof::t_sync); return PMPI_Allgather(s, sn, st, r, rn, rt, c);
}
inline int prof_Exscan(const void* s, void* r, int n, MPI_Datatype t, MPI_Op o, MPI_Comm c) {
    prof::Scope sc(prof::t_sync); return PMPI_Exscan(s, r, n, t, o, c);
}
inline int prof_Reduce(const void* s, void* r, int n, MPI_Datatype t, MPI_Op o, int root, MPI_Comm c) {
    prof::Scope sc(prof::t_sync); return PMPI_Reduce(s, r, n, t, o, root, c);
}

#define MPI_Init prof_Init
#define MPI_Finalize prof_Finalize
#define MPI_File_open prof_File_open
#define MPI_File_close prof_File_close
#define MPI_File_set_size prof_File_set_size
#define MPI_File_read_at_all prof_File_read_at_all
#define MPI_File_read_at prof_File_read_at
#define MPI_File_write_at_all prof_File_write_at_all
#define MPI_File_write_at prof_File_write_at
#define MPI_Sendrecv prof_Sendrecv
#define MPI_Allreduce prof_Allreduce
#define MPI_Barrier prof_Barrier
#define MPI_Bcast prof_Bcast
#define MPI_Allgather prof_Allgather
#define MPI_Exscan prof_Exscan
#define MPI_Reduce prof_Reduce
