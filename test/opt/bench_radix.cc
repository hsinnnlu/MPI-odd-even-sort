// =============================================================================
// Optimization 1 的佐證：比較 local sort 的幾種做法（單一 process）。
//
// 直接 #include 繳交的 hw1.cc，所以「final」用的就是繳交版的 radix_sort_32。
// 輸入：課程測資的 float（例如 10.in，N = 23,987,513），先轉成 sortable uint32 key。
//
//   std::sort            對 float 做 comparison sort（O(n log n)）
//   radix 8+8+8+8        4 個 pass，每 pass 256 個 bucket
//   radix 16+16          2 個 pass，每 pass 65536 個 bucket
//   radix 11+11+10 (分開) 3 個 pass，每個 pass 前重新掃一次資料算 histogram
//   radix 11+11+10 (final) 3 個 pass，第一次掃描就算好三個 pass 的 histogram（hw1.cc）
//
// 每種跑 TRIALS 次，印出中位數與每個元素的平均時間，並確認結果都和 std::sort 相同。
//
// 編譯：mpicxx -O3 -I. -o test/opt/bench_radix test/opt/bench_radix.cc
// 執行：srun -p big -N1 -n1 -c1 ./test/opt/bench_radix /tmp/.../10.in 23987513 5
// =============================================================================
#define main hw1_main
#include "../../hw1.cc"
#undef main

#include <chrono>
#include <string>

static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// 一般化的 LSD radix sort。widths = 每個 pass 的 bit 數（由低位到高位）。
// fused = true：一次掃描算好所有 pass 的 histogram；false：每個 pass 前重新掃描。
static void lsd_radix(std::vector<Key>& keys, std::vector<Key>& tmp, const std::vector<int>& widths, bool fused) {
    const int passes = static_cast<int>(widths.size());
    std::vector<int> shift(passes);
    for (int p = 0, s = 0; p < passes; ++p) { shift[p] = s; s += widths[p]; }
    std::vector<std::vector<uint32_t>> pos(passes);
    for (int p = 0; p < passes; ++p) pos[p].assign(size_t(1) << widths[p], 0);

    auto count_pass = [&](int p) {
        const Key mask = (Key(1) << widths[p]) - 1;
        for (Key k : keys) ++pos[p][(k >> shift[p]) & mask];
    };
    auto prefix = [&](int p) {
        uint32_t sum = 0;
        for (auto& c : pos[p]) { const uint32_t v = c; c = sum; sum += v; }
    };
    if (fused) {
        for (Key k : keys)
            for (int p = 0; p < passes; ++p) ++pos[p][(k >> shift[p]) & ((Key(1) << widths[p]) - 1)];
        for (int p = 0; p < passes; ++p) prefix(p);
    }
    for (int p = 0; p < passes; ++p) {
        if (!fused) { count_pass(p); prefix(p); }
        const Key mask = (Key(1) << widths[p]) - 1;
        uint32_t* at = pos[p].data();
        for (Key k : keys) tmp[at[(k >> shift[p]) & mask]++] = k;
        keys.swap(tmp);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s input N [trials=5]\n", argv[0]);
        MPI_Finalize();
        return 1;
    }
    const int n = std::atoi(argv[2]);
    const int trials = argc > 3 ? std::atoi(argv[3]) : 5;

    // 讀入原始 float bit pattern（只是讀資料，用一般 fread 即可；不計時）
    std::vector<Key> input(n);
    FILE* f = std::fopen(argv[1], "rb");
    if (!f || std::fread(input.data(), 4, n, f) != static_cast<size_t>(n)) { std::fprintf(stderr, "cannot read input\n"); return 1; }
    std::fclose(f);

    // 參考答案：std::sort 的結果轉成 key
    std::vector<float> ref(n);
    std::memcpy(ref.data(), input.data(), size_t(n) * 4);
    std::sort(ref.begin(), ref.end());
    std::vector<Key> ref_keys(n);
    for (int i = 0; i < n; ++i) { uint32_t b; std::memcpy(&b, &ref[i], 4); ref_keys[i] = float_bits_to_key(b); }

    struct Variant { std::string name; int kind; std::vector<int> widths; bool fused; };
    const std::vector<Variant> variants = {
        {"std::sort (float)", 0, {}, false},
        {"radix 8+8+8+8", 1, {8, 8, 8, 8}, true},
        {"radix 16+16", 1, {16, 16}, true},
        {"radix 11+11+10 per-pass hist", 1, {11, 11, 10}, false},
        {"radix 11+11+10 fused (final)", 2, {}, true},
    };

    std::printf("N=%d trials=%d（時間為中位數）\n", n, trials);
    std::printf("%-32s %10s %10s %8s %s\n", "variant", "ms", "ns/elem", "vs sort", "correct");
    double sort_ms = 0;
    std::vector<Key> keys(n), tmp(n);
    std::vector<float> fl(n);
    for (const auto& v : variants) {
        std::vector<double> t;
        bool ok = true;
        for (int r = 0; r < trials; ++r) {
            if (v.kind == 0) {
                std::memcpy(fl.data(), input.data(), size_t(n) * 4);
                const double s = now();
                std::sort(fl.begin(), fl.end());
                t.push_back(now() - s);
                for (int i = 0; i < n; ++i) { uint32_t b; std::memcpy(&b, &fl[i], 4); keys[i] = float_bits_to_key(b); }
            } else {
                for (int i = 0; i < n; ++i) keys[i] = float_bits_to_key(input[i]);   // 轉換不計時（各版本相同）
                const double s = now();
                if (v.kind == 1) lsd_radix(keys, tmp, v.widths, v.fused);
                else radix_sort_32(keys, tmp);
                t.push_back(now() - s);
            }
            ok = ok && keys == ref_keys;
        }
        std::sort(t.begin(), t.end());
        const double ms = t[t.size() / 2] * 1e3;
        if (v.kind == 0) sort_ms = ms;
        std::printf("%-32s %10.1f %10.2f %7.1fx %s\n", v.name.c_str(), ms, ms * 1e6 / n, sort_ms / ms, ok ? "OK" : "WRONG");
    }
    MPI_Finalize();
    return 0;
}
