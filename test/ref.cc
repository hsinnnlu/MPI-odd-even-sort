// 單機版參考答案：std::sort + hash，和作業規格的定義一字不差。
// 用來產生「正確輸出」，再和 hw1 的輸出比對。
//
// 編譯：g++ -O2 -I<hash.h 所在目錄> -o ref ref.cc
// 執行：./ref N input output [hash_rounds=25]
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "hash.h"

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "Usage: %s N input output [hash_rounds=25]\n", argv[0]);
        return 1;
    }
    const long long n = std::atoll(argv[1]);
    const int rounds = argc >= 5 ? std::atoi(argv[4]) : 25;

    std::vector<float> data(n);
    FILE* in = std::fopen(argv[2], "rb");
    if (!in || std::fread(data.data(), sizeof(float), n, in) != size_t(n)) {
        std::fprintf(stderr, "cannot read %lld floats from %s\n", n, argv[2]);
        return 1;
    }
    std::fclose(in);

    for (int round = 0; round < rounds; ++round) {
        std::sort(data.begin(), data.end());
        if (round + 1 < rounds)
            for (long long i = 0; i < n; ++i)
                data[i] = hw1_hash(data[i], uint32_t(i), uint32_t(round + 1));
    }

    FILE* out = std::fopen(argv[3], "wb");
    std::fwrite(data.data(), sizeof(float), n, out);
    std::fclose(out);
    return 0;
}
