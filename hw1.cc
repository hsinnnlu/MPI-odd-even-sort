// =============================================================================
// CS542200 HW1: Odd-Even Sort (MPI) with multi-round hashing
//
// 整體流程（每個 rank 都執行同一份程式）：
//
//   1. 讀檔：rank r 讀取連續的一段資料（block distribution）。
//   2. 把每個 float 轉成「可排序的 uint32 key」：
//        無號整數的大小順序 = 原本浮點數的大小順序。
//      之後所有排序、比較、交換都直接用整數 key 完成。
//   3. Round 0（輸入是任意 float）：
//        a. local LSD radix sort
//        b. odd-even phases，每次 compare-split 只交換「可能需要移動」的元素
//   4. Round 1 .. R-1（資料是上一輪的 hash 輸出，只是 24-bit 整數）：
//        a. hash 的同一個迴圈裡順便統計 bucket 數量（bucket = key 的高 12 bit）
//        b. 只做「一次」bucket 分配，bucket 內部先不排序
//        c. odd-even phases：以 bucket 為單位做 compare-split。
//           只需要 memcpy 整個 bucket，不需要逐元素比較的 merge。
//        d. 所有 phase 結束後，才在每個 bucket 內用 counting sort 排好。
//   5. 每輪結束（最後一輪除外）都做 hash，最後寫檔。
//
// 主要演算法優化：
//   (A) 延遲排序（lazy local sort）：rounds 1..R-1 不先做完整 local sort，
//       而是維持「依 bucket 分組」的狀態做 compare-split。compare-split 的結果
//       （lower rank 拿到聯集裡最小的 m 個）和傳統作法完全相同，
//       只是把 bucket 內的排序延到最後才做一次。
//   (B) hash 和下一輪的 histogram 合併成一個迴圈，少讀一次整塊資料。
//   (C) 用整數 key 取代 float：radix、比較、counting sort 都直接作用在整數上。
//   (D) 終止條件：理論上 p 個 phase 內一定排好，所以前 p 個 phase
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
enum Tag { TAG_EDGE = 1, TAG_HIST = 2, TAG_DATA = 3 };

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

// ---- 24-bit key 拆成 bucket（高 12 bit）和 bucket 內的值（低 12 bit） -------
constexpr int KEY_BITS = 24;
constexpr int HIGH_BITS = 12;                    // bucket 編號用的 bit 數
constexpr int LOW_BITS = KEY_BITS - HIGH_BITS;   // bucket 內部的 bit 數
constexpr int NUM_BUCKETS = 1 << HIGH_BITS;      // 4096 個 bucket
constexpr Key LOW_MASK = (1u << LOW_BITS) - 1;

static inline int bucket_of(Key key) { return static_cast<int>(key >> LOW_BITS); }

// =============================================================================
// 資料分配
// =============================================================================

// 資料太少時，讓很多 rank 一起做反而只會增加通訊次數。
// 所以每個參與排序的 rank 至少要分到這麼多元素。
constexpr long long MIN_ELEMENTS_PER_RANK = 4096;

// 最多幾個 rank 參與排序（0 = 不限制）。
// odd-even sort 每一輪搬移的資料量和 rank 數成正比，而同一台機器上的
// rank 共用記憶體頻寬，所以 rank 不是越多越快。
// 在課程機器上量測（test/sweep_active.sh，每組 5 次取中位數）：
// 上限 1/2/4/8 中，4 在 big、little、mixed 都是最快或接近最快。
constexpr int MAX_ACTIVE_RANKS = 4;

// 計算「從 rank 0 開始、連續和 rank 0 在同一台節點上」的 rank 有幾個。
//
// 原因：odd-even sort 的資料只能一個鄰居一個鄰居地傳，hash 之後資料是隨機的，
// 每一輪大約有一半的資料要穿過兩台節點之間的網路，而網路比共享記憶體慢很多。
// 所以只讓第一台節點上的 rank 參與排序，其他節點的 rank 不分配資料
// （仍然參與 MPI-IO 和終止判斷，這兩者規則允許使用 collective）。
// 只取「連續」的 rank，才能保證參與排序的 rank 彼此都是相鄰的。
static int count_ranks_on_first_node(int rank, int size) {
    char my_host[MPI_MAX_PROCESSOR_NAME] = {};
    char root_host[MPI_MAX_PROCESSOR_NAME] = {};
    int len = 0;
    MPI_Get_processor_name(my_host, &len);
    if (rank == 0) std::memcpy(root_host, my_host, sizeof(root_host));
    MPI_Bcast(root_host, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 0, MPI_COMM_WORLD);

    const int same_node = std::strcmp(my_host, root_host) == 0 ? 1 : 0;
    std::vector<int> all_same(size);
    MPI_Allgather(&same_node, 1, MPI_INT, all_same.data(), 1, MPI_INT, MPI_COMM_WORLD);

    int count = 0;
    while (count < size && all_same[count]) ++count;
    return count;
}

struct Layout {
    int rank = 0;
    int size = 1;
    int active = 1;           // 實際分到資料的 rank 數（rank 0..active-1）
    std::vector<int> count;   // count[r] = rank r 擁有的元素數
    int local_n = 0;          // = count[rank]
    int global_start = 0;     // 本 rank 第一個元素在全域排序結果中的 index
    int max_count = 0;        // 所有 rank 中最大的 count（決定接收 buffer 大小）
};

static Layout make_layout(int rank, int size, int n) {
    Layout L;
    L.rank = rank;
    L.size = size;

    // 參與排序的 rank 數 = 下面三個限制中最小的一個（至少 1 個）
    long long active = count_ranks_on_first_node(rank, size);                      // 只用第一台節點
    active = std::min(active, (n + MIN_ELEMENTS_PER_RANK - 1) / MIN_ELEMENTS_PER_RANK);  // 資料太少就少用幾個
    if (MAX_ACTIVE_RANKS > 0) active = std::min<long long>(active, MAX_ACTIVE_RANKS);
    L.active = static_cast<int>(std::max(1LL, active));

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
    std::vector<Key> keys;            // 本 rank 的資料
    std::vector<Key> tmp;             // 和 keys 一樣大，用來做 out-of-place 的重排
    std::vector<Key> recv;            // 接收鄰居資料
    std::vector<Key> boundary;        // 邊界 bucket 的聯集（bucket 模式用）
    std::vector<uint32_t> hist;       // hist[b]   = 本 rank 在 bucket b 的元素數
    std::vector<uint32_t> start;      // start[b]  = bucket b 在 keys 中的起點（共 NUM_BUCKETS+1 個）
    std::vector<uint32_t> partner_hist;
    std::vector<uint32_t> low_count;  // bucket 內 counting sort 用（也借給 compare-split 當暫存）
};

// =============================================================================
// Odd-even phase 的主迴圈（round 0 和 bucket 模式共用）
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
// Round 0：任意 32-bit key
// =============================================================================

// LSD radix sort，32 bit 拆成 11 + 11 + 10 bit 三個 pass。
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

// 兩個「已排序」block 之間的 compare-split。
//   lower rank（編號較小）留下聯集中最小的 local_n 個；
//   upper rank 留下最大的 local_n 個。
static bool compare_split_sorted(Workspace& ws, int rank, int partner) {
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

    // 對方送多少，事先不知道：用最大可能的長度接收，再用 MPI_Get_count 讀出實際數量
    MPI_Status status;
    MPI_Sendrecv(send_ptr, send_n, MPI_UINT32_T, partner, TAG_DATA,
                 ws.recv.data(), static_cast<int>(ws.recv.size()), MPI_UINT32_T, partner, TAG_DATA,
                 MPI_COMM_WORLD, &status);
    int recv_n = 0;
    MPI_Get_count(&status, MPI_UINT32_T, &recv_n);

    // Step 3：只合併需要的那一半，結果寫到 tmp 再交換指標（不 copy 回來）
    if (is_lower) merge_smallest(mine.data(), n, ws.recv.data(), recv_n, ws.tmp.data(), n);
    else          merge_largest (mine.data(), n, ws.recv.data(), recv_n, ws.tmp.data(), n);
    mine.swap(ws.tmp);
    return true;
}

// =============================================================================
// Rounds 1..R-1：bucket 模式
//
// 狀態：keys 依 bucket 編號（key 的高 12 bit）由小到大排列，
//       同一個 bucket 內部的順序「不重要」。
//       hist[b] / start[b] 記錄每個 bucket 的大小與位置。
// =============================================================================

// 由 hist 算出每個 bucket 的起點
static void compute_bucket_start(Workspace& ws) {
    uint32_t sum = 0;
    for (int b = 0; b < NUM_BUCKETS; ++b) {
        ws.start[b] = sum;
        sum += ws.hist[b];
    }
    ws.start[NUM_BUCKETS] = sum;
}

// 把 keys 依 bucket 分組（只做一次 scatter）。hist 必須已經算好。
static void scatter_into_buckets(Workspace& ws) {
    compute_bucket_start(ws);
    std::vector<uint32_t> pos(ws.start.begin(), ws.start.end() - 1);   // 每個 bucket 的寫入位置
    for (Key k : ws.keys) ws.tmp[pos[bucket_of(k)]++] = k;
    ws.keys.swap(ws.tmp);
}

// 本 rank 的最小值 / 最大值：只需要掃描第一個 / 最後一個非空的 bucket
static Key block_min(const Workspace& ws) {
    int b = 0;
    while (ws.hist[b] == 0) ++b;
    return *std::min_element(ws.keys.begin() + ws.start[b], ws.keys.begin() + ws.start[b + 1]);
}
static Key block_max(const Workspace& ws) {
    int b = NUM_BUCKETS - 1;
    while (ws.hist[b] == 0) --b;
    return *std::max_element(ws.keys.begin() + ws.start[b], ws.keys.begin() + ws.start[b + 1]);
}

// 兩個「bucket 分組」block 之間的 compare-split。
//
// 目標和傳統 compare-split 一樣：lower 留下聯集中最小的 n_lower 個。
// 做法：
//   1. 交換 histogram 後，兩邊都能算出「分界 bucket」s：
//        bucket < s 的元素全部屬於 lower，bucket > s 的全部屬於 upper，
//        只有 bucket s 需要依照實際值再切一刀。
//   2. lower 把 bucket >= s 的元素送給 upper；upper 把 bucket <= s 的元素送給 lower。
//      因為 keys 依 bucket 排好，這兩段在記憶體中都是連續的。
//   3. bucket s 的聯集很小，用 nth_element 找出最小的 take 個給 lower。
//   4. 重新組合：每個 bucket 只要把「自己的」和「收到的」兩段 memcpy 接起來。
//      不需要逐元素比較的 merge。
static bool compare_split_buckets(Workspace& ws, const Layout& L, int partner) {
    const bool is_lower = L.rank < partner;

    // ---- Step 1：邊界檢查 ---------------------------------------------------
    const Key my_edge = is_lower ? block_max(ws) : block_min(ws);
    const Key partner_edge = exchange_one(my_edge, partner);
    const Key lower_max = is_lower ? my_edge : partner_edge;
    const Key upper_min = is_lower ? partner_edge : my_edge;
    if (lower_max <= upper_min) return false;

    // ---- Step 2：交換 histogram（4096 個整數，和資料量相比很小） --------------
    MPI_Sendrecv(ws.hist.data(), NUM_BUCKETS, MPI_UINT32_T, partner, TAG_HIST,
                 ws.partner_hist.data(), NUM_BUCKETS, MPI_UINT32_T, partner, TAG_HIST,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    const uint32_t* hist_lower = is_lower ? ws.hist.data() : ws.partner_hist.data();
    const uint32_t* hist_upper = is_lower ? ws.partner_hist.data() : ws.hist.data();
    const uint32_t n_lower = static_cast<uint32_t>(L.count[std::min(L.rank, partner)]);

    // ---- Step 3：找分界 bucket s ---------------------------------------------
    // below = 兩邊在 bucket < s 的元素總數（全部歸 lower）
    // lower 還要從 bucket s 的聯集中拿 take 個最小的
    // 兩個 rank 用同樣的 histogram 計算，所以會得到同樣的 s 和 take
    int s = 0;
    uint32_t below = 0;
    while (below + hist_lower[s] + hist_upper[s] < n_lower) {
        below += hist_lower[s] + hist_upper[s];
        ++s;
    }
    const uint32_t take = n_lower - below;   // 1 <= take <= bucket s 的聯集大小

    // ---- Step 4：交換資料 -----------------------------------------------------
    //   lower 送出自己 bucket s..4095（keys 的尾段），收到對方 bucket 0..s
    //   upper 送出自己 bucket 0..s   （keys 的頭段），收到對方 bucket s..4095
    // 因為 keys 依 bucket 排列，送出的部分在記憶體中是連續的一段。
    const int recv_first = is_lower ? 0 : s;                  // 收到的 bucket 範圍
    const int recv_last = is_lower ? s : NUM_BUCKETS - 1;     // [recv_first, recv_last]
    const uint32_t send_begin = is_lower ? ws.start[s] : 0;
    const uint32_t send_end = is_lower ? ws.start[NUM_BUCKETS] : ws.start[s + 1];

    // 收到的資料也依 bucket 排列；用對方的 histogram 算出
    // recv_pos[b] = 對方 bucket b 在 recv 中的起點，總和就是要收的數量
    std::vector<uint32_t>& recv_pos = ws.low_count;   // 借用暫存陣列（大小 >= NUM_BUCKETS）
    uint32_t recv_n = 0;
    for (int b = recv_first; b <= recv_last; ++b) {
        recv_pos[b] = recv_n;
        recv_n += ws.partner_hist[b];
    }

    MPI_Sendrecv(ws.keys.data() + send_begin, static_cast<int>(send_end - send_begin), MPI_UINT32_T,
                 partner, TAG_DATA,
                 ws.recv.data(), static_cast<int>(recv_n), MPI_UINT32_T,
                 partner, TAG_DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    auto recv_bucket = [&](int b) { return ws.recv.data() + recv_pos[b]; };
    auto my_bucket = [&](int b) { return ws.keys.data() + ws.start[b]; };

    // ---- Step 5：切分界 bucket s ----------------------------------------------
    // 把兩邊的 bucket s 接成一個陣列，nth_element 之後前 take 個就是最小的 take 個。
    // 兩邊的聯集是同一個 multiset，而值相同的 key 本來就無法區分，
    // 所以兩個 rank 各自切出來的結果一定互補。
    ws.boundary.clear();
    ws.boundary.insert(ws.boundary.end(), my_bucket(s), my_bucket(s) + ws.hist[s]);
    ws.boundary.insert(ws.boundary.end(), recv_bucket(s), recv_bucket(s) + ws.partner_hist[s]);
    if (take < ws.boundary.size())
        std::nth_element(ws.boundary.begin(), ws.boundary.begin() + take, ws.boundary.end());

    // ---- Step 6：重新組合成新的 bucket 排列（寫到 tmp） ------------------------
    Key* out = ws.tmp.data();
    auto append = [&](const Key* src, uint32_t count) {
        std::memcpy(out, src, count * sizeof(Key));
        out += count;
    };

    if (is_lower) {
        // bucket < s：自己的 + 收到的；bucket s：聯集中最小的 take 個；bucket > s：清空
        for (int b = 0; b < s; ++b) {
            append(my_bucket(b), ws.hist[b]);
            append(recv_bucket(b), ws.partner_hist[b]);
            ws.hist[b] += ws.partner_hist[b];
        }
        append(ws.boundary.data(), take);
        ws.hist[s] = take;
        for (int b = s + 1; b < NUM_BUCKETS; ++b) ws.hist[b] = 0;
    } else {
        // bucket < s：清空；bucket s：聯集中剩下的；bucket > s：收到的 + 自己的
        for (int b = 0; b < s; ++b) ws.hist[b] = 0;
        const uint32_t rest = static_cast<uint32_t>(ws.boundary.size()) - take;
        append(ws.boundary.data() + take, rest);
        // 這裡 hist[b]（b > s）還是舊值，剛好就是自己 bucket b 的大小；複製完才更新
        for (int b = s + 1; b < NUM_BUCKETS; ++b) {
            append(recv_bucket(b), ws.partner_hist[b]);
            append(my_bucket(b), ws.hist[b]);
        }
        ws.hist[s] = rest;
        for (int b = s + 1; b < NUM_BUCKETS; ++b) ws.hist[b] += ws.partner_hist[b];
    }

    ws.keys.swap(ws.tmp);
    compute_bucket_start(ws);
    return true;
}

// 所有 phase 結束後，把每個 bucket 內部排好，結果寫到 tmp 再和 keys 交換。
// 同一個 bucket 的 key 高 12 bit 都一樣，只差低 12 bit，所以：
//   1. 數每個低 12 bit 的值出現幾次
//   2. 依序把 (bucket << 12 | low) 寫出 count 次
// 資料本身就是 key，沒有附帶 payload，所以可以直接「重新產生」排好的序列。
// 很小的 bucket 用 std::sort 比較划算（掃 4096 個計數器反而比較慢）。
static void sort_inside_buckets(Workspace& ws) {
    constexpr uint32_t SMALL_BUCKET = 512;
    constexpr uint32_t BURST = 8;   // 一次固定寫 8 個（編譯器會變成一個向量 store）
    std::vector<uint32_t>& cnt = ws.low_count;
    std::fill(cnt.begin(), cnt.end(), 0u);

    Key* const out_end = ws.tmp.data() + ws.tmp.size();

    for (int b = 0; b < NUM_BUCKETS; ++b) {
        const uint32_t n = ws.hist[b];
        const Key* in = ws.keys.data() + ws.start[b];
        Key* out = ws.tmp.data() + ws.start[b];

        if (n < SMALL_BUCKET) {
            std::copy(in, in + n, out);
            std::sort(out, out + n);
            continue;
        }
        for (uint32_t i = 0; i < n; ++i) ++cnt[in[i] & LOW_MASK];

        const Key high = static_cast<Key>(b) << LOW_BITS;
        for (Key low = 0; low <= LOW_MASK; ++low) {
            const uint32_t c = cnt[low];
            const Key value = high | low;
            cnt[low] = 0;   // 順便歸零，給下一個 bucket 用

            if (out + BURST <= out_end) {
                // 不管 c 是多少，都先寫 8 個，再把指標往前移 c 格。
                // 多寫的部分會被下一個值覆蓋（out 只會往後走），
                // 這樣就不會因為 c 大小不一而發生 branch misprediction。
                for (uint32_t i = 0; i < BURST; ++i) out[i] = value;
                for (uint32_t i = BURST; i < c; ++i) out[i] = value;   // c > 8 時才會執行
            } else {
                // 快到整個陣列尾端：不能多寫，老實地寫 c 個
                for (uint32_t i = 0; i < c; ++i) out[i] = value;
            }
            out += c;
        }
    }
    ws.keys.swap(ws.tmp);
}

// =============================================================================
// Hash（同時統計下一輪需要的 bucket histogram）
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
        for (size_t i = begin; i < end; ++i) ++ws.hist[bucket_of(keys[i])];
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
    ws.hist.resize(NUM_BUCKETS);
    ws.partner_hist.resize(NUM_BUCKETS);
    ws.start.resize(NUM_BUCKETS + 1);
    ws.low_count.resize(std::max(NUM_BUCKETS + 1, 1 << LOW_BITS));

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
        if (round == 0) {
            radix_sort_32(ws.keys, ws.tmp);
            run_odd_even_phases(L, [&](int partner) { return compare_split_sorted(ws, rank, partner); });
        } else {
            // 上一輪結尾的 hash 已經算好 hist，這裡只要 scatter 一次
            scatter_into_buckets(ws);
            run_odd_even_phases(L, [&](int partner) { return compare_split_buckets(ws, L, partner); });
            sort_inside_buckets(ws);
        }

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
