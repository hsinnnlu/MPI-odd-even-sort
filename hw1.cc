// =============================================================================
// CS542200 HW1: Odd-Even Sort (MPI) with multi-round hashing
//
// 整體流程（每個 rank 都執行同一份程式）：
//
//   1. 讀檔：所有 rank 平均分配資料，rank r 讀取連續的一段（block distribution）。
//   2. 把每個 float 轉成「可排序的 uint32 key」：
//        無號整數的大小順序 = 原本浮點數的大小順序。
//      之後所有排序、比較、交換都直接用整數 key 完成。
//   3. 每一輪：
//        a. local LSD radix sort
//             round 0：32-bit key，11 + 11 + 10 bit 三個 pass
//             之後  ：hash 輸出只有 24 bit，12 + 12 bit 兩個 pass
//        b. odd-even phases，每次 compare-split 只交換「可能需要移動」的元素
//   4. 每輪結束（最後一輪除外）都做 hash，最後寫檔。
//
// 主要演算法優化：
//   (A) radix sort 取代 comparison sort；histogram 一次算好（round 0 在排序前一次讀完，
//       之後的 round 在 hash 的同一個迴圈裡順便統計），每個 pass 只需要 scatter。
//   (B) selective compare-split：先交換邊界值，已有序就跳過；
//       否則只送出可能移動的元素，並且只 merge 出自己要留下的那一半。
//   (C) 用整數 key 取代 float：radix、比較都直接作用在整數上。
//   (D) 跨節點壓縮：相鄰兩個 rank 在不同節點時，送出的已排序資料改送
//       「和前一個數的差值」並用變長編碼（varint），網路傳輸量約剩 1/4～1/2。
//   (E) 終止條件：理論上 p 個 phase 內一定排好，所以前 p 個 phase
//       完全不做 Allreduce，之後才每個 cycle 檢查一次。
// =============================================================================
#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "hash.h"

using Key = uint32_t;

// MPI tag：不同用途的訊息用不同 tag，避免互相配對錯誤
enum Tag { TAG_EDGE = 1, TAG_DATA = 2 };

// =============================================================================
// Key 編碼
// =============================================================================

// ---- Round 0：任意 float ↔ 32-bit key ---------------------------------------
// IEEE-754 的正數直接比 bit pattern 就是正確順序；負數的順序剛好相反。
//   正數：把 sign bit 設成 1         → 排在所有負數後面
//   負數：所有 bit 取反（~bits）      → 絕對值越大，key 越小
static inline Key float_bits_to_key(uint32_t bits) {
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}
static inline uint32_t key_to_float_bits(Key key) {
    return (key & 0x80000000u) ? (key & 0x7fffffffu) : ~key;
}

static inline float bits_to_float(uint32_t bits) {
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

// ---- Round 1..R-1：hash 輸出 ↔ 24-bit key ----------------------------------
// hw1_hash 的輸出一定是 [-2^23, 2^23) 之間的整數，加上 2^23 之後
// 就變成 [0, 2^24) 的非負整數，大小順序不變。
constexpr int32_t HASH_OFFSET = 1 << 23;

static inline Key hash_output_to_key(float value) {
    return static_cast<Key>(static_cast<int32_t>(value) + HASH_OFFSET);
}
static inline float hash_key_to_float(Key key) {
    return static_cast<float>(static_cast<int32_t>(key) - HASH_OFFSET);
}

// ---- 24-bit key 的 radix sort：12 + 12 bit 兩個 pass ----------------------
constexpr int DIGIT_BITS_24 = 12;
constexpr int RADIX_24 = 1 << DIGIT_BITS_24;     // 每個 pass 4096 個桶
constexpr Key DIGIT_MASK_24 = RADIX_24 - 1;

// =============================================================================
// 資料分配
// =============================================================================

// 資料太少時，讓很多 rank 一起做反而只會增加通訊次數。
// 所以每個參與排序的 rank 至少要分到這麼多元素；資料夠多時（N >= 4096 × size）
// 所有 rank（包含其他節點上的 rank）都會分到資料、參與排序。
constexpr long long MIN_ELEMENTS_PER_RANK = 4096;

struct Layout {
    int rank = 0;
    int size = 1;
    int active = 1;           // 實際分到資料的 rank 數（rank 0..active-1）
    std::vector<int> count;   // count[r] = rank r 擁有的元素數
    int local_n = 0;          // = count[rank]
    int global_start = 0;     // 本 rank 第一個元素在全域排序結果中的 index
    int max_count = 0;        // 所有 rank 中最大的 count（決定接收 buffer 大小）
    std::vector<int> node;    // node[r] = rank r 所在節點的編號（同一台節點的 rank 編號相同）
};

// 查出每個 rank 在哪一台節點：交換主機名稱，名稱相同就是同一台。
// 只在初始化時做一次（規定允許初始化使用 collective）。
static std::vector<int> find_nodes(int size) {
    char name[MPI_MAX_PROCESSOR_NAME] = {};
    int len = 0;
    MPI_Get_processor_name(name, &len);
    std::vector<char> all(static_cast<size_t>(size) * MPI_MAX_PROCESSOR_NAME);
    MPI_Allgather(name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, all.data(), MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                  MPI_COMM_WORLD);

    // 節點編號 = 第一個和自己同名的 rank 的編號
    std::vector<int> node(size);
    for (int r = 0; r < size; ++r) {
        node[r] = r;
        for (int q = 0; q < r; ++q) {
            if (std::strcmp(&all[static_cast<size_t>(q) * MPI_MAX_PROCESSOR_NAME],
                            &all[static_cast<size_t>(r) * MPI_MAX_PROCESSOR_NAME]) == 0) {
                node[r] = node[q];
                break;
            }
        }
    }
    return node;
}

static Layout make_layout(int rank, int size, int n) {
    Layout L;
    L.rank = rank;
    L.size = size;

    // 參與排序的 rank 數 = min(size, ceil(n / MIN_ELEMENTS_PER_RANK))，至少 1 個
    const long long by_size = (n + MIN_ELEMENTS_PER_RANK - 1) / MIN_ELEMENTS_PER_RANK;
    L.active = static_cast<int>(std::max(1LL, std::min<long long>(size, by_size)));

    // 前 rem 個 rank 多拿一個，其餘拿 base 個；rank >= active 拿 0 個
    const int base = n / L.active;
    const int rem = n % L.active;
    L.count.assign(size, 0);
    for (int r = 0; r < L.active; ++r) L.count[r] = base + (r < rem ? 1 : 0);

    L.local_n = L.count[rank];
    L.max_count = base + (rem > 0 ? 1 : 0);

    // global_start = 前面所有 rank 的元素數總和
    // （等同作業說明中的 MPI_Exscan；每個 rank 的元素數全程不變，所以只算一次）
    L.global_start = 0;
    for (int r = 0; r < rank; ++r) L.global_start += L.count[r];

    L.node = find_nodes(size);
    return L;
}

// MPI-IO 的錯誤預設是「回傳錯誤碼」而不是中止程式（MPI_ERRORS_RETURN），
// 不檢查的話，檔案打不開時程式會什麼都沒寫就正常結束。
// 所以每個 MPI_File_* 呼叫都要檢查，失敗就印出原因並中止。
static void check_io(int rc, const char* what, const char* path) {
    if (rc == MPI_SUCCESS) return;
    char reason[MPI_MAX_ERROR_STRING];
    int len = 0;
    MPI_Error_string(rc, reason, &len);
    std::fprintf(stderr, "hw1: %s '%s' failed: %s\n", what, path, reason);
    MPI_Abort(MPI_COMM_WORLD, 1);
}

// 和鄰居交換一個 key
static Key exchange_one(Key mine, int partner) {
    Key theirs = 0;
    MPI_Sendrecv(&mine, 1, MPI_UINT32_T, partner, TAG_EDGE,
                 &theirs, 1, MPI_UINT32_T, partner, TAG_EDGE,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    return theirs;
}

// =============================================================================
// 所有 rank 共用的工作空間（整個程式只配置一次，不在迴圈內 allocate）
// =============================================================================
struct Workspace {
    std::vector<Key> keys;        // 本 rank 的資料
    std::vector<Key> tmp;         // 和 keys 一樣大，用來做 out-of-place 的重排
    std::vector<Key> recv;        // 接收鄰居資料
    std::vector<uint32_t> hist;   // 24-bit radix sort 兩個 pass 的 histogram（hash 時順便算好）
    std::vector<uint8_t> send_bytes;  // 跨節點壓縮用（只有鄰居在別台節點的 rank 才配置）
    std::vector<uint8_t> recv_bytes;
};

// =============================================================================
// Odd-even phase 的主迴圈（每一輪都一樣）
//
// compare_split(partner) 負責和 partner 做一次 compare-split，
// 有資料移動就回傳 true。
//
// 終止條件：
//   * block 版 odd-even sort 在各 block 已經可比較的情況下，active 個 phase
//     內就會排好。所以前 active 個 phase 不做任何全域同步。
//   * 之後每做完一個 cycle（even + odd）做一次 Allreduce；
//     如果整個 cycle 所有 rank 都沒有移動資料，代表每一對相鄰 block 都已有序，
//     整體就排好了。這個檢查保證正確性，不依賴上面的理論界限。
// =============================================================================
template <class CompareSplit>
static void run_odd_even_phases(const Layout& L, CompareSplit&& compare_split) {
    if (L.active <= 1) return;   // 只有一個 rank 有資料：local sort 就是全部

    for (int phase = 0;;) {
        int changed = 0;

        // 一個 cycle = even phase + odd phase
        for (int half = 0; half < 2; ++half, ++phase) {
            // even phase 配對 (0,1),(2,3)...；odd phase 配對 (1,2),(3,4)...
            const bool pair_with_right = ((phase + L.rank) % 2 == 0);
            const int partner = pair_with_right ? L.rank + 1 : L.rank - 1;

            if (L.rank < L.active && partner >= 0 && partner < L.active && compare_split(partner))
                changed = 1;
        }

        if (phase < L.active) continue;   // 還在理論界限內，先不檢查

        int any_changed = 0;
        MPI_Allreduce(&changed, &any_changed, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        if (!any_changed) break;
    }
}

// =============================================================================
// Local sort：LSD radix sort
// =============================================================================

// Round 0：任意 32-bit key。，32 bit 拆成 11 + 11 + 10 bit 三個 pass。
// 一次讀取就算出三個 pass 的 histogram，之後每個 pass 只需要 scatter。
static void radix_sort_32(std::vector<Key>& keys, std::vector<Key>& tmp) {
    const size_t n = keys.size();
    if (n <= 1) return;

    constexpr int PASSES = 3;
    constexpr int SHIFT[PASSES] = {0, 11, 22};
    constexpr Key MASK[PASSES] = {0x7ff, 0x7ff, 0x3ff};
    constexpr int RADIX = 2048;

    std::vector<uint32_t> offset(PASSES * RADIX, 0);
    for (Key k : keys)
        for (int p = 0; p < PASSES; ++p) ++offset[p * RADIX + ((k >> SHIFT[p]) & MASK[p])];

    // histogram → 每個 digit 的起始位置（exclusive prefix sum）
    for (int p = 0; p < PASSES; ++p) {
        uint32_t sum = 0;
        for (int d = 0; d < RADIX; ++d) {
            const uint32_t c = offset[p * RADIX + d];
            offset[p * RADIX + d] = sum;
            sum += c;
        }
    }

    for (int p = 0; p < PASSES; ++p) {
        uint32_t* pos = offset.data() + p * RADIX;
        for (Key k : keys) tmp[pos[(k >> SHIFT[p]) & MASK[p]]++] = k;
        keys.swap(tmp);   // 每個 pass 結束後，結果都放回 keys
    }
}

// Round 1 之後：hash 輸出的 24-bit key，12 + 12 bit 兩個 pass。
// histogram 已經在 hash_and_count 裡算好：hist[0..4095] 是低 12 bit，hist[4096..8191] 是高 12 bit。
static void radix_sort_24(Workspace& ws) {
    if (ws.keys.size() <= 1) return;
    uint32_t* pos_low = ws.hist.data();
    uint32_t* pos_high = ws.hist.data() + RADIX_24;
    uint32_t sum_low = 0, sum_high = 0;
    for (int d = 0; d < RADIX_24; ++d) {
        const uint32_t c_low = pos_low[d], c_high = pos_high[d];
        pos_low[d] = sum_low;   sum_low += c_low;
        pos_high[d] = sum_high; sum_high += c_high;
    }
    for (Key k : ws.keys) ws.tmp[pos_low[k & DIGIT_MASK_24]++] = k;                    // 依低 12 bit
    for (Key k : ws.tmp)  ws.keys[pos_high[(k >> DIGIT_BITS_24) & DIGIT_MASK_24]++] = k;  // 依高 12 bit
}

// 合併兩個已排序陣列 a、b，輸出最小的 count 個到 out。
// 迴圈內用條件選擇取代 if/else，避免資料隨機時 branch misprediction。
static void merge_smallest(const Key* a, int na, const Key* b, int nb, Key* out, int count) {
    int i = 0, j = 0, k = 0;
    while (k < count && i < na && j < nb) {
        const Key x = a[i], y = b[j];
        const bool take_a = (x <= y);
        out[k++] = take_a ? x : y;
        i += take_a;
        j += !take_a;
    }
    // 其中一邊用完了，剩下的直接從另一邊複製
    if (k < count && i < na) { const int c = std::min(count - k, na - i); std::memcpy(out + k, a + i, c * sizeof(Key)); k += c; }
    if (k < count && j < nb) { const int c = std::min(count - k, nb - j); std::memcpy(out + k, b + j, c * sizeof(Key)); k += c; }
}

// 合併兩個已排序陣列 a、b，輸出最大的 count 個到 out（由後往前填）。
static void merge_largest(const Key* a, int na, const Key* b, int nb, Key* out, int count) {
    int i = na - 1, j = nb - 1, k = count - 1;
    while (k >= 0 && i >= 0 && j >= 0) {
        const Key x = a[i], y = b[j];
        const bool take_a = (x >= y);
        out[k--] = take_a ? x : y;
        i -= take_a;
        j -= !take_a;
    }
    if (k >= 0 && i >= 0) { const int c = std::min(k + 1, i + 1); std::memcpy(out + k + 1 - c, a + i + 1 - c, c * sizeof(Key)); k -= c; }
    if (k >= 0 && j >= 0) { const int c = std::min(k + 1, j + 1); std::memcpy(out + k + 1 - c, b + j + 1 - c, c * sizeof(Key)); k -= c; }
}

// =============================================================================
// 跨節點壓縮：已排序的 key 改送「差值」，並用 varint 編碼
//
// 例：8000000, 8000007, 8000019 → 8000000, 7, 12
// varint：每個 byte 放 7 bit，最高位元 = 1 表示後面還有 byte。
//   差值 < 128 → 1 byte，< 16384 → 2 byte，最多 5 byte（32-bit）。
// hash 之後的 key 只有 24 bit，相鄰差值通常不到 128，所以大多只要 1 byte。
// 解碼時把差值一個一個加回去，得到的資料和原本完全相同。
// =============================================================================
constexpr int MAX_VARINT_BYTES = 5;

static int encode_sorted(const Key* in, int n, uint8_t* out) {
    uint8_t* p = out;
    Key prev = 0;
    for (int i = 0; i < n; ++i) {
        uint32_t d = in[i] - prev;   // 已排序，所以差值一定 >= 0
        prev = in[i];
        while (d >= 0x80) {
            *p++ = static_cast<uint8_t>(d | 0x80);
            d >>= 7;
        }
        *p++ = static_cast<uint8_t>(d);
    }
    return static_cast<int>(p - out);
}

// 回傳解出的 key 數量
static int decode_sorted(const uint8_t* in, int num_bytes, Key* out) {
    const uint8_t* p = in;
    const uint8_t* end = in + num_bytes;
    Key prev = 0;
    int n = 0;
    while (p < end) {
        uint32_t d = 0;
        int shift = 0;
        uint8_t b;
        do {
            b = *p++;
            d |= static_cast<uint32_t>(b & 0x7f) << shift;
            shift += 7;
        } while (b & 0x80);
        prev += d;
        out[n++] = prev;
    }
    return n;
}

// 和 partner 交換一段已排序的 key，回傳收到的數量。
// 跨節點（compress = true）時壓縮後再送；同一台節點走共享記憶體，直接送比較快。
static int exchange_sorted(Workspace& ws, const Key* send_ptr, int send_n, int partner, bool compress) {
    MPI_Status status;
    int recv_n = 0;
    if (!compress) {
        // 對方送多少，事先不知道：用最大可能的長度接收，再用 MPI_Get_count 讀出實際數量
        MPI_Sendrecv(send_ptr, send_n, MPI_UINT32_T, partner, TAG_DATA,
                     ws.recv.data(), static_cast<int>(ws.recv.size()), MPI_UINT32_T, partner, TAG_DATA,
                     MPI_COMM_WORLD, &status);
        MPI_Get_count(&status, MPI_UINT32_T, &recv_n);
        return recv_n;
    }
    const int send_bytes = encode_sorted(send_ptr, send_n, ws.send_bytes.data());
    MPI_Sendrecv(ws.send_bytes.data(), send_bytes, MPI_BYTE, partner, TAG_DATA,
                 ws.recv_bytes.data(), static_cast<int>(ws.recv_bytes.size()), MPI_BYTE, partner, TAG_DATA,
                 MPI_COMM_WORLD, &status);
    int recv_bytes = 0;
    MPI_Get_count(&status, MPI_BYTE, &recv_bytes);
    return decode_sorted(ws.recv_bytes.data(), recv_bytes, ws.recv.data());
}

// 兩個「已排序」block 之間的 compare-split。
//   lower rank（編號較小）留下聯集中最小的 local_n 個；
//   upper rank 留下最大的 local_n 個。
static bool compare_split_sorted(Workspace& ws, int rank, int partner, bool compress) {
    std::vector<Key>& mine = ws.keys;
    const int n = static_cast<int>(mine.size());
    const bool is_lower = rank < partner;

    // Step 1：交換邊界值。lower 的最大值 <= upper 的最小值 → 已有序，不用交換。
    const Key my_edge = is_lower ? mine.back() : mine.front();
    const Key partner_edge = exchange_one(my_edge, partner);
    const Key lower_max = is_lower ? my_edge : partner_edge;
    const Key upper_min = is_lower ? partner_edge : my_edge;
    if (lower_max <= upper_min) return false;

    // Step 2：只送出「可能需要移動」的元素。
    //   lower：<= upper_min 的元素一定會留下，只送出 > upper_min 的尾端
    //   upper：>= lower_max 的元素一定會留下，只送出 < lower_max 的前端
    const Key* send_ptr;
    int send_n;
    if (is_lower) {
        const Key* first = std::upper_bound(mine.data(), mine.data() + n, upper_min);
        send_ptr = first;
        send_n = static_cast<int>(mine.data() + n - first);
    } else {
        const Key* last = std::lower_bound(mine.data(), mine.data() + n, lower_max);
        send_ptr = mine.data();
        send_n = static_cast<int>(last - mine.data());
    }

    const int recv_n = exchange_sorted(ws, send_ptr, send_n, partner, compress);

    // Step 3：只合併需要的那一半，結果寫到 tmp 再交換指標（不 copy 回來）
    if (is_lower) merge_smallest(mine.data(), n, ws.recv.data(), recv_n, ws.tmp.data(), n);
    else          merge_largest (mine.data(), n, ws.recv.data(), recv_n, ws.tmp.data(), n);
    mine.swap(ws.tmp);
    return true;
}

// =============================================================================
// Hash（同時統計下一輪 radix sort 需要的 histogram）
// =============================================================================
//
// 分成小段處理：每段先做 hash（這個迴圈沒有相依性，編譯器可以向量化），
// 再對同一段統計 histogram（這段資料還在 cache 裡，幾乎不用再讀記憶體）。
// 如果把 ++hist[...] 寫在 hash 的同一個迴圈裡，整個迴圈就無法向量化。
//
// target_clones：編譯器同時產生 AVX2 版和一般版，執行時依 CPU 自動挑選。
// hw1_hash 裡有 32-bit 整數乘法，沒有 AVX2 時很難向量化。
#if defined(__GNUC__) && !defined(__clang__) && defined(__x86_64__)
#define VECTORIZE_CLONES __attribute__((target_clones("avx2", "default")))
#else
#define VECTORIZE_CLONES
#endif
template <bool KEYS_ARE_FLOAT>
VECTORIZE_CLONES static void hash_and_count(Workspace& ws, uint32_t global_start, uint32_t hash_round) {
    constexpr size_t CHUNK = 2048;
    std::fill(ws.hist.begin(), ws.hist.end(), 0u);
    Key* keys = ws.keys.data();
    const size_t n = ws.keys.size();

    for (size_t begin = 0; begin < n; begin += CHUNK) {
        const size_t end = std::min(n, begin + CHUNK);

        for (size_t i = begin; i < end; ++i) {
            // 先把 key 還原成原本的 float 值，再交給 hash.h 的 hw1_hash
            const float value = KEYS_ARE_FLOAT ? bits_to_float(key_to_float_bits(keys[i]))
                                               : hash_key_to_float(keys[i]);
            const float hashed = hw1_hash(value, global_start + static_cast<uint32_t>(i), hash_round);
            keys[i] = hash_output_to_key(hashed);
        }
        for (size_t i = begin; i < end; ++i) {
            ++ws.hist[keys[i] & DIGIT_MASK_24];
            ++ws.hist[RADIX_24 + ((keys[i] >> DIGIT_BITS_24) & DIGIT_MASK_24)];
        }
    }
}

// =============================================================================
// main
// =============================================================================
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 4) {
        if (rank == 0) std::fprintf(stderr, "Usage: %s N input output [hash_rounds=25]\n", argv[0]);
        MPI_Finalize();
        return 1;
    }
    const int n = std::atoi(argv[1]);
    const char* input_file = argv[2];
    const char* output_file = argv[3];
    const int rounds = (argc >= 5) ? std::atoi(argv[4]) : 25;

    const Layout L = make_layout(rank, size, n);

    Workspace ws;
    ws.keys.resize(L.local_n);
    ws.tmp.resize(L.local_n);
    ws.recv.resize(std::max(L.max_count, 1));
    ws.hist.resize(2 * RADIX_24);

    // 左右鄰居是否在別台節點：是的話，和它交換資料時要壓縮
    const bool remote_left = rank > 0 && L.node[rank - 1] != L.node[rank];
    const bool remote_right = rank + 1 < size && L.node[rank + 1] != L.node[rank];
    if (remote_left || remote_right) {
        const size_t bytes = static_cast<size_t>(std::max(L.max_count, 1)) * MAX_VARINT_BYTES;
        ws.send_bytes.resize(bytes);
        ws.recv_bytes.resize(bytes);
    }

    // ---- 讀檔：直接以 32-bit 整數讀入 float 的 bit pattern ----------------------
    {
        MPI_File fh;
        check_io(MPI_File_open(MPI_COMM_WORLD, input_file, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh),
                 "open input", input_file);
        check_io(MPI_File_read_at_all(fh, static_cast<MPI_Offset>(L.global_start) * sizeof(float),
                                      ws.keys.data(), L.local_n, MPI_UINT32_T, MPI_STATUS_IGNORE),
                 "read input", input_file);
        MPI_File_close(&fh);
    }
    for (Key& k : ws.keys) k = float_bits_to_key(k);

    // ---- 多輪排序 ---------------------------------------------------------------
    bool keys_are_float = true;   // round 0 是 32-bit float key，之後都是 24-bit hash key

    for (int round = 0; round < rounds; ++round) {
        if (keys_are_float) radix_sort_32(ws.keys, ws.tmp);
        else                radix_sort_24(ws);   // histogram 已在上一輪結尾的 hash 中算好
        run_odd_even_phases(L, [&](int partner) {
            const bool compress = L.node[partner] != L.node[rank];
            return compare_split_sorted(ws, rank, partner, compress);
        });

        // 這一輪全域已排好；除了最後一輪，都要 hash（hash round 從 1 開始）
        if (round + 1 < rounds) {
            if (keys_are_float) hash_and_count<true>(ws, L.global_start, round + 1);
            else                hash_and_count<false>(ws, L.global_start, round + 1);
            keys_are_float = false;
        }
    }

    // ---- 寫檔：key 轉回 float 的 bit pattern ------------------------------------
    for (Key& k : ws.keys) {
        if (keys_are_float) {
            k = key_to_float_bits(k);
        } else {
            const float value = hash_key_to_float(k);
            std::memcpy(&k, &value, sizeof(k));
        }
    }
    {
        MPI_File fh;
        check_io(MPI_File_open(MPI_COMM_WORLD, output_file, MPI_MODE_CREATE | MPI_MODE_WRONLY,
                               MPI_INFO_NULL, &fh),
                 "open output", output_file);
        // 檔案原本若比較大，先截斷，避免殘留舊資料
        check_io(MPI_File_set_size(fh, static_cast<MPI_Offset>(n) * sizeof(float)),
                 "truncate output", output_file);
        check_io(MPI_File_write_at_all(fh, static_cast<MPI_Offset>(L.global_start) * sizeof(float),
                                       ws.keys.data(), L.local_n, MPI_UINT32_T, MPI_STATUS_IGNORE),
                 "write output", output_file);
        MPI_File_close(&fh);
    }

    MPI_Finalize();
    return 0;
}
