// =============================================================================
// Optimization 2 的佐證：比較 compare-split 的四種做法（MPI，多個 process）。
//
// 直接 #include 繳交的 hw1.cc：讀檔、資料分配（make_layout）、local radix sort、
// odd-even 主迴圈與終止判斷（run_odd_even_phases）都和繳交版相同，只換 compare-split：
//
//   V0 full + full merge     兩邊交換整塊資料，std::merge 全部合併，再取自己要的一半（最初版的做法）
//   V1 full + partial merge  交換整塊資料，但只 merge 出需要保留的 local_n 個
//   V2 + boundary check      先交換邊界值，已有序就跳過；否則同 V1
//   V3 selective (final)     邊界檢查 + 只送「可能需要移動」的元素 + partial merge（hw1.cc 的 compare_split_sorted）
//
// 量測：一次完整的 odd-even 排序（從各 rank 已排好的 block 開始，直到全域有序）的時間、
// 所有 rank 的 MPI_Sendrecv 總傳送量、實際交換資料與被邊界檢查跳過的次數。
// 每種跑 TRIALS 次取中位數，並確認四種做法的最終結果完全相同。
//
// 編譯：mpicxx -O3 -I. -o test/opt/bench_split test/opt/bench_split.cc
// 執行：srun -p big -N1 -n4 -c1 ./test/opt/bench_split /tmp/.../10.in 23987513 5
//       輸入也可以用產生的資料（不讀檔）：
//         gen:random  每個值是均勻亂數（和課程測資類似，相鄰 block 的值域幾乎完全重疊）
//         gen:nearly  第 i 筆約等於 i，加上 ±0.1%·N 的雜訊（幾乎排好，相鄰 block 只在邊界重疊）
// =============================================================================
#include <mpi.h>

// 計算 MPI_Sendrecv 的傳送量（hw1.cc 裡的呼叫也會經過這裡）
static double g_sent_bytes = 0;
static inline int count_Sendrecv(const void* sb, int sc, MPI_Datatype st, int dst, int stag, void* rb, int rc,
                                  MPI_Datatype rt, int src, int rtag, MPI_Comm c, MPI_Status* s) {
    int sz = 0;
    MPI_Type_size(st, &sz);
    g_sent_bytes += static_cast<double>(sc) * sz;
    return PMPI_Sendrecv(sb, sc, st, dst, stag, rb, rc, rt, src, rtag, c, s);
}
#define MPI_Sendrecv count_Sendrecv

#include <random>
#include <string>

#define main hw1_main
#include "../../hw1.cc"
#undef main

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (argc < 3) {
        if (rank == 0) std::fprintf(stderr, "usage: %s input N [trials=5]\n", argv[0]);
        MPI_Finalize();
        return 1;
    }
    const int n = std::atoi(argv[2]);
    const int trials = argc > 3 ? std::atoi(argv[3]) : 5;

    // ---- 和 hw1.cc 相同的讀檔與 round 0 的 local sort ----
    const Layout L = make_layout(rank, size, n);
    Workspace ws;
    ws.keys.resize(L.local_n);
    ws.tmp.resize(L.local_n);
    ws.recv.resize(std::max(L.max_count, 1));
    const std::string source = argv[1];
    if (source.rfind("gen:", 0) == 0) {
        std::mt19937 rng(1234 + rank);
        const double noise = std::max(1.0, 0.001 * n);
        std::uniform_real_distribution<double> uni(-1.0, 1.0);
        for (int i = 0; i < L.local_n; ++i) {
            const double g = L.global_start + i;
            const float v = static_cast<float>(source == "gen:nearly" ? g + noise * uni(rng) : 1e6 * uni(rng));
            uint32_t b;
            std::memcpy(&b, &v, 4);
            ws.keys[i] = float_bits_to_key(b);
        }
    } else {
        MPI_File fh;
        check_io(MPI_File_open(MPI_COMM_WORLD, argv[1], MPI_MODE_RDONLY, MPI_INFO_NULL, &fh), "open", argv[1]);
        check_io(MPI_File_read_at_all(fh, static_cast<MPI_Offset>(L.global_start) * 4, ws.keys.data(), L.local_n,
                                      MPI_UINT32_T, MPI_STATUS_IGNORE), "read", argv[1]);
        MPI_File_close(&fh);
        for (Key& k : ws.keys) k = float_bits_to_key(k);
    }
    radix_sort_32(ws.keys, ws.tmp);
    const std::vector<Key> sorted_block = ws.keys;   // 每次試驗都從這裡開始

    // ---- 四種 compare-split ----
    long long exchanged = 0, skipped = 0;
    std::vector<Key> merged;   // V0 用：兩塊完整合併的結果
    auto full_exchange = [&](int partner) {
        const int pn = L.count[partner];
        MPI_Sendrecv(ws.keys.data(), L.local_n, MPI_UINT32_T, partner, TAG_DATA,
                     ws.recv.data(), pn, MPI_UINT32_T, partner, TAG_DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        return pn;
    };
    auto v0 = [&](int partner) {             // 整塊交換 + std::merge 全部
        const int pn = full_exchange(partner), m = L.local_n;
        merged.resize(static_cast<size_t>(m) + pn);
        std::merge(ws.keys.begin(), ws.keys.end(), ws.recv.begin(), ws.recv.begin() + pn, merged.begin());
        const Key* keep = rank < partner ? merged.data() : merged.data() + pn;
        const bool changed = !std::equal(keep, keep + m, ws.keys.data());
        std::copy(keep, keep + m, ws.keys.data());
        ++exchanged;
        return changed;
    };
    auto v1_core = [&](int partner) {         // 整塊交換 + 只 merge 需要的一半
        const int pn = full_exchange(partner), m = L.local_n;
        const bool lower = rank < partner;
        const bool changed = lower ? ws.keys.back() > ws.recv[0] : ws.recv[pn - 1] > ws.keys.front();
        if (lower) merge_smallest(ws.keys.data(), m, ws.recv.data(), pn, ws.tmp.data(), m);
        else       merge_largest (ws.keys.data(), m, ws.recv.data(), pn, ws.tmp.data(), m);
        ws.keys.swap(ws.tmp);
        ++exchanged;
        return changed;
    };
    auto v2 = [&](int partner) {              // 先檢查邊界，已有序就跳過
        const bool lower = rank < partner;
        const Key mine = lower ? ws.keys.back() : ws.keys.front();
        const Key theirs = exchange_one(mine, partner);
        if ((lower ? mine : theirs) <= (lower ? theirs : mine)) { ++skipped; return false; }
        return v1_core(partner);
    };
    auto v3 = [&](int partner) {              // 繳交版
        const bool changed = compare_split_sorted(ws, rank, partner);
        if (changed) ++exchanged; else ++skipped;
        return changed;
    };

    struct Result { double ms; double mb; long long ex, sk; bool ok; };
    std::vector<Key> reference;
    auto run = [&](const char* name, auto&& split) {
        std::vector<double> times;
        double mb = 0;
        long long ex = 0, sk = 0;
        bool ok = true;
        for (int t = 0; t < trials; ++t) {
            ws.keys = sorted_block;
            exchanged = skipped = 0;
            g_sent_bytes = 0;
            MPI_Barrier(MPI_COMM_WORLD);
            const double s = MPI_Wtime();
            run_odd_even_phases(L, split);
            MPI_Barrier(MPI_COMM_WORLD);
            times.push_back(MPI_Wtime() - s);
            double sent = g_sent_bytes;
            long long counts[2] = {exchanged, skipped}, sums[2];
            MPI_Reduce(&sent, &mb, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(counts, sums, 2, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
            ex = sums[0] / 2;   // 每次交換由兩個 rank 各算一次
            sk = sums[1] / 2;
            if (reference.empty()) reference = ws.keys;
            int same = ws.keys == reference, all_same = 0;
            MPI_Allreduce(&same, &all_same, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
            ok = ok && all_same;
        }
        std::sort(times.begin(), times.end());
        if (rank == 0)
            std::printf("%-26s %10.1f %12.1f %10lld %9lld   %s\n", name, times[times.size() / 2] * 1e3,
                        mb / 1e6, ex, sk, ok ? "OK" : "WRONG");
    };

    if (rank == 0) {
        std::printf("input=%s N=%d ranks=%d active=%d local_n≈%d trials=%d（時間為中位數；從各 rank 已排好的 block 開始）\n",
                    argv[1], n, size, L.active, L.local_n, trials);
        std::printf("%-26s %10s %12s %10s %9s   %s\n", "variant", "ms", "sent MB", "exchanges", "skipped", "same result");
    }
    run("V3 selective (final)", v3);   // 先跑繳交版，當作正確結果的參考
    run("V0 full + full merge", v0);
    run("V1 full + partial merge", v1_core);
    run("V2 + boundary check", v2);
    MPI_Finalize();
    return 0;
}
