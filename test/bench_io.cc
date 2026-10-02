// 比較幾種 MPI-IO 寫法的讀寫時間（只在課程機器上做分析用，不會交出去）。
//
// 資料分配和 hw1.cc 一樣（#include hw1.cc 的 make_layout），每種寫法跑 TRIALS 次，
// 每次前後都有 MPI_Barrier，印出最慢 rank 的時間。
//
// 讀：
//   read_at_all   collective（目前 hw1.cc 的寫法）
//   read_at       independent，每個 rank 自己讀自己那段
// 寫：
//   set_size + write_at_all   （目前 hw1.cc 的寫法）
//   set_size + write_at       independent 寫
//   delete + write_at         先由 rank 0 用 MPI_File_delete 刪掉舊檔，再建立新檔，
//                             不需要 set_size 也不會有殘留資料
//
// 編譯：mpicxx -O3 -I. -o test/bench_io test/bench_io.cc
// 執行：srun -p big -N1 -n4 ./test/bench_io 23987513 /srv/nova/scratch/coursedata/pp2026/hw1/10.in test/out
#define main hw1_main
#include "../hw1.cc"
#undef main

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

    auto timed = [&](const char* name, auto&& work) {
        for (int t = 0; t < trials; ++t) {
            MPI_Barrier(MPI_COMM_WORLD);
            const double s = MPI_Wtime();
            work();
            MPI_Barrier(MPI_COMM_WORLD);
            const double e = MPI_Wtime() - s;
            if (rank == 0) std::printf("  %-28s trial %d: %8.1f ms\n", name, t + 1, e * 1e3);
        }
    };

    if (rank == 0) std::printf("N=%d ranks=%d active=%d\n讀檔：\n", n, size, L.active);
    timed("read_at_all (目前)", [&] {
        MPI_File fh;
        check_io(MPI_File_open(MPI_COMM_WORLD, input, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh), "open", input);
        check_io(MPI_File_read_at_all(fh, offset, buf.data(), L.local_n, MPI_UINT32_T, MPI_STATUS_IGNORE), "read", input);
        MPI_File_close(&fh);
    });
    timed("read_at (independent)", [&] {
        MPI_File fh;
        check_io(MPI_File_open(MPI_COMM_WORLD, input, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh), "open", input);
        if (L.local_n > 0)
            check_io(MPI_File_read_at(fh, offset, buf.data(), L.local_n, MPI_UINT32_T, MPI_STATUS_IGNORE), "read", input);
        MPI_File_close(&fh);
    });

    if (rank == 0) std::printf("寫檔：\n");
    timed("set_size + write_at_all (目前)", [&] {
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
    timed("delete + write_at", [&] {
        if (rank == 0) MPI_File_delete(out.c_str(), MPI_INFO_NULL);   // 檔案不存在時會回傳錯誤，忽略即可
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_File fh;
        check_io(MPI_File_open(MPI_COMM_WORLD, out.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh), "open", out.c_str());
        if (L.local_n > 0)
            check_io(MPI_File_write_at(fh, offset, buf.data(), L.local_n, MPI_UINT32_T, MPI_STATUS_IGNORE), "write", out.c_str());
        MPI_File_close(&fh);
    });
    MPI_Finalize();
    return 0;
}
