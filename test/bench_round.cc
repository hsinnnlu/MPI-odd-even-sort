// 量測 hw1.cc 每一輪各階段花多少時間（只在課程機器上做分析用，不會交出去）。
//
// 直接 #include hw1.cc，用同樣的函式，但不讀檔：每個 rank 自己產生隨機的 24-bit key
// （和 hash 之後的資料分佈相同），然後跑幾輪「hash → scatter → odd-even → bucket 內排序」，
// 分別計時。每個階段前後都有 MPI_Barrier，所以時間是「最慢的 rank」的時間。
//
// 編譯：mpicxx -O3 -I. -o test/bench_round test/bench_round.cc
// 執行：srun -p big -N1 -n4 ./test/bench_round 23987513 5
//        參數：N（預設 23987513）、量測輪數（預設 5）
#define main hw1_main
#include "../hw1.cc"
#undef main

#include <random>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const int n = argc > 1 ? std::atoi(argv[1]) : 23987513;
    const int rounds = argc > 2 ? std::atoi(argv[2]) : 5;

    // 和 hw1.cc 的 main 相同的設定
    const Layout L = make_layout(rank, size, n);
    Workspace ws;
    ws.keys.resize(L.local_n);
    ws.tmp.resize(L.local_n);
    ws.recv.resize(std::max(L.max_count, 1));
    ws.hist.resize(NUM_BUCKETS);
    ws.partner_hist.resize(NUM_BUCKETS);
    ws.start.resize(NUM_BUCKETS + 1);
    ws.low_count.resize(std::max(NUM_BUCKETS + 1, 1 << LOW_BITS));
    ws.scratch.resize(LARGE_BUCKET);

    std::mt19937 rng(12345 + rank);
    for (Key& k : ws.keys) k = rng() & 0xffffff;

    // 先跑一次 round 0 的路徑（32-bit radix + 已排序的 compare-split）
    double t_round0_sort = 0, t_round0_phases = 0;
    {
        std::vector<Key> as_float(ws.keys);
        for (Key& k : ws.keys) k = float_bits_to_key(static_cast<uint32_t>(rng()) & 0xbfffffffu);
        MPI_Barrier(MPI_COMM_WORLD);
        double t = MPI_Wtime();
        radix_sort_32(ws.keys, ws.tmp);
        MPI_Barrier(MPI_COMM_WORLD);
        t_round0_sort = MPI_Wtime() - t;
        t = MPI_Wtime();
        run_odd_even_phases(L, [&](int partner) { return compare_split_sorted(ws, rank, partner); });
        MPI_Barrier(MPI_COMM_WORLD);
        t_round0_phases = MPI_Wtime() - t;
        ws.keys.swap(as_float);
    }

    // 第一次 hash 前要先算好 histogram（hw1.cc 裡是上一輪 hash 順便算的）
    hash_and_count<false>(ws, L.global_start, 1);

    enum { HASH, SCATTER, PHASES, BUCKET_SORT, STAGES };
    const char* names[STAGES] = {"hash+count", "scatter", "odd-even phases", "bucket sort"};
    double total[STAGES] = {};

    for (int r = 0; r < rounds; ++r) {
        auto timed = [&](int stage, auto&& work) {
            MPI_Barrier(MPI_COMM_WORLD);
            const double t = MPI_Wtime();
            work();
            MPI_Barrier(MPI_COMM_WORLD);
            total[stage] += MPI_Wtime() - t;
        };
        timed(SCATTER, [&] { scatter_into_buckets(ws); });
        timed(PHASES, [&] {
            run_odd_even_phases(L, [&](int partner) { return compare_split_buckets(ws, L, partner); });
        });
        timed(BUCKET_SORT, [&] { sort_inside_buckets(ws); });
        timed(HASH, [&] { hash_and_count<false>(ws, L.global_start, static_cast<uint32_t>(r + 2)); });
    }

    if (rank == 0) {
        std::printf("N=%d ranks=%d active=%d local_n=%d  (每輪平均, ms)\n", n, size, L.active, L.local_n);
        double sum = 0;
        for (int s = 0; s < STAGES; ++s) {
            std::printf("  %-16s %8.2f\n", names[s], total[s] / rounds * 1e3);
            sum += total[s] / rounds;
        }
        std::printf("  %-16s %8.2f  （x24 輪 = %.2f s）\n", "一輪合計", sum * 1e3, sum * 24);
        std::printf("  round 0：radix_sort_32 %.2f ms，phases %.2f ms\n", t_round0_sort * 1e3, t_round0_phases * 1e3);
    }
    MPI_Finalize();
    return 0;
}
