// 比較幾種讀寫方式在課程機器上的速度（只做分析用，不會交出去）。
//
// 資料分配和 hw1.cc 一樣（#include hw1.cc 的 make_layout）。每種方式跑 TRIALS 次，
// 前後都有 MPI_Barrier，印出最慢 rank 的時間。
//
//   POSIX pread / pwrite   基準線：NFS 本身最快能多快（作業規定要用 MPI-IO，這只是對照）
//   read_at_all            目前 hw1.cc 的讀法（collective）
//   read_at                independent
//   set_size + write_at_all    目前 hw1.cc 的寫法
//   set_size + write_at        independent
//   rank 0 寫全部              先 MPI_Gatherv 到 rank 0，再由 rank 0 一次寫完（MPI_COMM_SELF 開檔）
//
// 編譯：mpicxx -O3 -I. -o test/bench_io test/bench_io.cc
// 執行：srun -p big -N1 -n4 ./test/bench_io 23987513 /srv/nova/scratch/coursedata/pp2026/hw1/10.in test/out
//       （test/io_matrix.sh 會用 OMPIO 與 ROMIO 各跑一次）
#define main hw1_main
#include "../hw1.cc"
#undef main

#include <fcntl.h>
#include <unistd.h>

#include <string>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (argc < 4) {
        if (rank == 0) std::fprintf(stderr, "usage: %s N input out_dir [trials=3]\n", argv[0]);
        MPI_Finalize();
        return 1;
    }
    const int n = std::atoi(argv[1]);
    const char* input = argv[2];
    const std::string out = std::string(argv[3]) + "/bench_io.out";
    const int trials = argc > 4 ? std::atoi(argv[4]) : 3;

    const Layout L = make_layout(rank, size, n);
    std::vector<Key> buf(L.local_n);
    const MPI_Offset offset = static_cast<MPI_Offset>(L.global_start) * sizeof(float);
    const size_t bytes = static_cast<size_t>(L.local_n) * sizeof(float);

    auto timed = [&](const char* name, auto&& work) {
        double best = 1e30, worst = 0;
        for (int t = 0; t < trials; ++t) {
            MPI_Barrier(MPI_COMM_WORLD);
            const double s = MPI_Wtime();
            work();
            MPI_Barrier(MPI_COMM_WORLD);
            const double e = MPI_Wtime() - s;
            best = std::min(best, e);
            worst = std::max(worst, e);
        }
        if (rank == 0) std::printf("  %-28s 最快 %7.0f ms   最慢 %7.0f ms\n", name, best * 1e3, worst * 1e3);
    };

    if (rank == 0) std::printf("N=%d (%.0f MB) ranks=%d active=%d\n讀檔：\n", n, n * 4.0 / 1e6, size, L.active);
    timed("POSIX pread（基準）", [&] {
        const int fd = open(input, O_RDONLY);
        size_t done = 0;
        while (done < bytes) {
            const ssize_t r = pread(fd, reinterpret_cast<char*>(buf.data()) + done, bytes - done, offset + done);
            if (r <= 0) break;
            done += static_cast<size_t>(r);
        }
        close(fd);
    });
    timed("read_at_all（目前）", [&] {
        MPI_File fh;
        check_io(MPI_File_open(MPI_COMM_WORLD, input, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh), "open", input);
        check_io(MPI_File_read_at_all(fh, offset, buf.data(), L.local_n, MPI_UINT32_T, MPI_STATUS_IGNORE), "read", input);
        MPI_File_close(&fh);
    });
    timed("read_at", [&] {
        MPI_File fh;
        check_io(MPI_File_open(MPI_COMM_WORLD, input, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh), "open", input);
        if (L.local_n > 0)
            check_io(MPI_File_read_at(fh, offset, buf.data(), L.local_n, MPI_UINT32_T, MPI_STATUS_IGNORE), "read", input);
        MPI_File_close(&fh);
    });

    if (rank == 0) std::printf("寫檔：\n");
    timed("POSIX pwrite（基準）", [&] {
        if (rank == 0) { const int fd = open(out.c_str(), O_WRONLY | O_CREAT, 0644); if (ftruncate(fd, static_cast<off_t>(n) * 4) != 0) {} close(fd); }
        MPI_Barrier(MPI_COMM_WORLD);
        const int fd = open(out.c_str(), O_WRONLY);
        size_t done = 0;
        while (done < bytes) {
            const ssize_t w = pwrite(fd, reinterpret_cast<const char*>(buf.data()) + done, bytes - done, offset + done);
            if (w <= 0) break;
            done += static_cast<size_t>(w);
        }
        close(fd);   // NFS 在 close 時會把資料送到 server
    });
    timed("set_size + write_at_all（目前）", [&] {
        MPI_File fh;
        check_io(MPI_File_open(MPI_COMM_WORLD, out.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh), "open", out.c_str());
        check_io(MPI_File_set_size(fh, static_cast<MPI_Offset>(n) * sizeof(float)), "set_size", out.c_str());
        check_io(MPI_File_write_at_all(fh, offset, buf.data(), L.local_n, MPI_UINT32_T, MPI_STATUS_IGNORE), "write", out.c_str());
        MPI_File_close(&fh);
    });
    timed("set_size + write_at", [&] {
        MPI_File fh;
        check_io(MPI_File_open(MPI_COMM_WORLD, out.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh), "open", out.c_str());
        check_io(MPI_File_set_size(fh, static_cast<MPI_Offset>(n) * sizeof(float)), "set_size", out.c_str());
        if (L.local_n > 0)
            check_io(MPI_File_write_at(fh, offset, buf.data(), L.local_n, MPI_UINT32_T, MPI_STATUS_IGNORE), "write", out.c_str());
        MPI_File_close(&fh);
    });
    std::vector<Key> all(rank == 0 ? static_cast<size_t>(n) : 0);
    std::vector<int> displs(size);
    for (int r = 0, sum = 0; r < size; ++r) { displs[r] = sum; sum += L.count[r]; }
    timed("rank 0 寫全部（Gatherv）", [&] {
        MPI_Gatherv(buf.data(), L.local_n, MPI_UINT32_T, all.data(), L.count.data(), displs.data(), MPI_UINT32_T, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            MPI_File fh;
            check_io(MPI_File_open(MPI_COMM_SELF, out.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh), "open", out.c_str());
            check_io(MPI_File_set_size(fh, static_cast<MPI_Offset>(n) * sizeof(float)), "set_size", out.c_str());
            check_io(MPI_File_write_at(fh, 0, all.data(), n, MPI_UINT32_T, MPI_STATUS_IGNORE), "write", out.c_str());
            MPI_File_close(&fh);
        }
    });
    MPI_Finalize();
    return 0;
}
