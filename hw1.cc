// Candidate v4: same v3 placement, sorting and MPI protocol; independent merge streams.
// Topology-aware active ranks, batched neighbor partition probes,
// branchless sparse merge and shared scratch. No non-neighbor element exchanges.
#ifndef HW1_SORT
#define HW1_SORT 2
#endif
#include <mpi.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>


#include "hash.h"

// Compile-time switches support one-change-at-a-time comparisons.
// SORT=0 original 16/16; SORT=1 previous 11/11/10; SORT=2 24-bit hash keys.
#ifndef HW1_SORT
#define HW1_SORT 0
#endif
#ifndef HW1_MERGE
#define HW1_MERGE 1
#endif
#ifndef HW1_STOP
#define HW1_STOP 1
#endif
// Use a contiguous prefix on rank 0's host to avoid cross-node element traffic.
// All ranks still participate in MPI-IO and termination checks.
#ifndef HW1_FIRST_NODE
#define HW1_FIRST_NODE 1
#endif
#ifndef HW1_MAX_ACTIVE
#define HW1_MAX_ACTIVE 4
#endif

#ifndef HW1_PROBES
#define HW1_PROBES 7
#endif
#ifndef HW1_BUFFERED
#define HW1_BUFFERED 0
#endif
#ifndef HW1_BUFFERED_MIN
#define HW1_BUFFERED_MIN 262144
#endif
#ifndef HW1_BRANCHLESS
#define HW1_BRANCHLESS 1
#endif
static_assert(HW1_PROBES >= 1 && HW1_PROBES <= 31, "Invalid probe count");

static_assert(HW1_SORT >= 0 && HW1_SORT <= 2, "Invalid sort mode");

#ifdef HW1_PROFILE
static double hw1_times[6] = {};
struct Hw1Timer {
    int id; double start;
    explicit Hw1Timer(int i) : id(i), start(MPI_Wtime()) {}
    ~Hw1Timer() { hw1_times[id] += MPI_Wtime() - start; }
};
#define HW1_TIMER(id) Hw1Timer hw1_timer(id)
#else
#define HW1_TIMER(id) ((void)0)
#endif



// -----------------------------------------------------------------------------
// Stable 2-pass LSD radix sort for finite IEEE-754 binary32 floats.
//
// Transform the raw float bits into an unsigned key whose integer ordering
// matches numeric float ordering:
//   negative: ~bits
//   positive: bits ^ 0x80000000
//
// We never overwrite the float with the transformed key.  The original 32-bit
// float payload is copied unchanged to the output buffer, which is important
// because later hw1_hash() must see the original float value/bit pattern.
//
// Each pass uses 16 radix bits => 65536 buckets, so sorting takes two linear
// passes instead of comparison-based O(n log n) sorting.
// -----------------------------------------------------------------------------
static inline uint32_t float_sort_key(float x) {
    uint32_t bits;
    std::memcpy(&bits, &x, sizeof(bits));

    if (bits & 0x80000000u)
        return ~bits;

    return bits ^ 0x80000000u;
}

static void radix_sort_float(
    std::vector<float>& data,
    std::vector<float>& temp,
    std::vector<size_t>& count
) {
    const size_t n = data.size();
    if (n <= 1)
        return;

    constexpr size_t RADIX = 1u << 16;
    constexpr uint32_t MASK = 0xffffu;

    // Pass 0: low 16 bits of transformed key.
    std::fill(count.begin(), count.end(), size_t{0});

    for (size_t i = 0; i < n; ++i) {
        const uint32_t key = float_sort_key(data[i]);
        ++count[key & MASK];
    }

    size_t sum = 0;
    for (size_t b = 0; b < RADIX; ++b) {
        const size_t c = count[b];
        count[b] = sum;
        sum += c;
    }

    for (size_t i = 0; i < n; ++i) {
        const float x = data[i];
        const uint32_t key = float_sort_key(x);
        temp[count[key & MASK]++] = x;
    }

    // Pass 1: high 16 bits of transformed key.
    std::fill(count.begin(), count.end(), size_t{0});

    for (size_t i = 0; i < n; ++i) {
        const uint32_t key = float_sort_key(temp[i]);
        ++count[(key >> 16) & MASK];
    }

    sum = 0;
    for (size_t b = 0; b < RADIX; ++b) {
        const size_t c = count[b];
        count[b] = sum;
        sum += c;
    }

    for (size_t i = 0; i < n; ++i) {
        const float x = temp[i];
        const uint32_t key = float_sort_key(x);
        data[count[(key >> 16) & MASK]++] = x;
    }
}

[[maybe_unused]] static void radix_sort_float_11(std::vector<float>& data,
                             std::vector<float>& temp,
                             std::vector<uint32_t>& count) {
    const size_t n = data.size();
    if (n <= 1) return;
    if (n <= 2048) {
        std::sort(data.begin(), data.end(), [](float a, float b) {
            return float_sort_key(a) < float_sort_key(b);
        });
        return;
    }
    constexpr unsigned B = 2048;
    std::fill(count.begin(), count.end(), 0u);
    for (float x : data) {
        const uint32_t k = float_sort_key(x);
        ++count[k & 2047];
        ++count[B + ((k >> 11) & 2047)];
        ++count[2 * B + (k >> 22)];
    }
    for (unsigned pass = 0; pass < 3; ++pass) {
        uint32_t sum = 0;
        const unsigned buckets = pass == 2 ? 1024 : B;
        for (unsigned b = 0; b < buckets; ++b) {
            const uint32_t c = count[pass * B + b];
            count[pass * B + b] = sum;
            sum += c;
        }
    }
    for (unsigned pass = 0; pass < 3; ++pass) {
        const unsigned shift = 11 * pass;
        const uint32_t mask = pass == 2 ? 1023 : 2047;
        uint32_t* offsets = count.data() + pass * B;
        for (float x : data)
            temp[offsets[(float_sort_key(x) >> shift) & mask]++] = x;
        data.swap(temp);
    }
}

static void merge_sparse(std::vector<float>& data,
                         const std::vector<float>& received, int k, bool lower) {
    const int n = static_cast<int>(data.size());
    if (lower) {
        int i = n - k - 1, j = k - 1, out = n - 1;
        while (i >= 0 && j >= 0) {
#if HW1_BRANCHLESS
            const float own = data[i], incoming = received[j];
            const bool take_own = own > incoming;
            data[out--] = take_own ? own : incoming;
            i -= int(take_own);
            j -= int(!take_own);
#else
            if (data[i] > received[j]) data[out--] = data[i--];
            else data[out--] = received[j--];
#endif
        }
        if (j >= 0)
            std::memcpy(data.data(), received.data(), size_t(j + 1) * sizeof(float));
    } else {
        int i = k, j = 0, out = 0;
        while (i < n && j < k) {
#if HW1_BRANCHLESS
            const float own = data[i], incoming = received[j];
            const bool take_own = own < incoming;
            data[out++] = take_own ? own : incoming;
            i += int(take_own);
            j += int(!take_own);
#else
            if (data[i] < received[j]) data[out++] = data[i++];
            else data[out++] = received[j++];
#endif
        }
        if (j < k)
            std::memcpy(data.data() + out, received.data() + j,
                        size_t(k - j) * sizeof(float));
    }
}


#ifndef HW1_MERGE_LANES
#define HW1_MERGE_LANES 2
#endif
static_assert(HW1_MERGE_LANES == 2 || HW1_MERGE_LANES == 4, "Use 2 or 4 merge streams");
#ifndef HW1_INTERLEAVED_MERGE
#define HW1_INTERLEAVED_MERGE 1
#endif

// Find how many A values belong in the first 'diagonal' merged values.
// Equal values from A precede equal values from B.
static int merge_cut(const float* a, int na, const float* b, int nb, int diagonal) {
    int lo = std::max(0, diagonal - nb), hi = std::min(diagonal, na);
    while(lo <= hi) {
        const int i = lo + (hi - lo) / 2, j = diagonal - i;
        if(i > 0 && j < nb && a[i-1] > b[j]) hi = i - 1;
        else if(j > 0 && i < na && b[j-1] >= a[i]) lo = i + 1;
        else return i;
    }
    std::abort(); // Finite sorted inputs always admit a partition.
}

// Independent output partitions break the single i/j dependency chain.
// No additional threads: one CPU interleaves independent scalar merge streams.
static void merge_interleaved(const float* a, int na, const float* b, int nb,
                              float* output) {
    constexpr int L = HW1_MERGE_LANES;
    const int n = na + nb;
    int ia[L], ib[L], ea[L], eb[L], pos[L];
    int previous_a = 0, previous_b = 0;
    for(int lane = 0; lane < L; ++lane) {
        const int end = int(int64_t(n) * (lane + 1) / L);
        const int cut_a = lane + 1 == L ? na : merge_cut(a, na, b, nb, end);
        const int cut_b = end - cut_a;
        ia[lane] = previous_a; ib[lane] = previous_b;
        ea[lane] = cut_a; eb[lane] = cut_b;
        pos[lane] = previous_a + previous_b;
        previous_a = cut_a; previous_b = cut_b;
    }
    for(;;) {
        int batch = n;
        for(int lane = 0; lane < L; ++lane)
            batch = std::min(batch, std::min(ea[lane]-ia[lane], eb[lane]-ib[lane]));
        if(batch == 0) break;
        // Neither input can be exhausted within these 'batch' steps.
        for(int step = 0; step < batch; ++step) {
#pragma GCC unroll 4
            for(int lane = 0; lane < L; ++lane) {
                const float x = a[ia[lane]], y = b[ib[lane]];
                const bool take_a = x <= y;
                output[pos[lane]++] = take_a ? x : y;
                ia[lane] += int(take_a);
                ib[lane] += int(!take_a);
            }
        }
    }
    for(int lane = 0; lane < L; ++lane) {
        int i = ia[lane], j = ib[lane], out = pos[lane];
        while(i < ea[lane] && j < eb[lane]) {
            const float x = a[i], y = b[j];
            const bool take_a = x <= y;
            output[out++] = take_a ? x : y;
            i += int(take_a); j += int(!take_a);
        }
        if(i < ea[lane]) std::memcpy(output + out, a + i, size_t(ea[lane]-i)*sizeof(float));
        else if(j < eb[lane]) std::memcpy(output + out, b + j, size_t(eb[lane]-j)*sizeof(float));
    }
}

// Valid ONLY after hw1_hash: every value is an exact integer in [-2^23,2^23).
// Map that known domain to a 24-bit unsigned key, keeping original float values.
static inline uint32_t hash_sort_key(float x) {
    return static_cast<uint32_t>(static_cast<int32_t>(x) + 8388608);
}
// Software write combining: amortize random scatter writes over 16 floats.
static void buffered_scatter(const std::vector<float>& src, std::vector<float>& dst,
                             uint32_t* offsets, unsigned shift,
                             std::vector<float>& staging) {
    constexpr unsigned B = 4096, LANES = 16;
    uint8_t used[B] = {};
    for(float x : src) {
        const unsigned bucket = (hash_sort_key(x) >> shift) & (B - 1);
        unsigned count = used[bucket];
        staging[bucket * LANES + count] = x;
        if (++count == LANES) {
            std::memcpy(dst.data() + offsets[bucket], staging.data() + bucket * LANES,
                        LANES * sizeof(float));
            offsets[bucket] += LANES;
            count = 0;
        }
        used[bucket] = static_cast<uint8_t>(count);
    }
    for(unsigned bucket = 0; bucket < B; ++bucket) {
        std::memcpy(dst.data() + offsets[bucket], staging.data() + bucket * LANES,
                    unsigned(used[bucket]) * sizeof(float));
    }
}

static void radix_sort_hash24(std::vector<float>& data,
                              std::vector<float>& temp,
                              std::vector<uint32_t>& counts) {
    if (data.size() <= 1) return;
    constexpr unsigned B = 4096;
    std::fill(counts.begin(), counts.end(), 0u);
    for(float x : data) {
        uint32_t key = hash_sort_key(x);
        ++counts[key & (B - 1)];
        ++counts[B + (key >> 12)];
    }
    for(unsigned pass = 0; pass < 2; ++pass) {
        uint32_t sum = 0;
        for(unsigned b = 0; b < B; ++b) {
            uint32_t c = counts[pass * B + b];
            counts[pass * B + b] = sum;
            sum += c;
        }
    }
    if (HW1_BUFFERED && data.size() >= HW1_BUFFERED_MIN) {
        // Each MPI process has its own scratch. thread_local also supports the test shim.
        static thread_local std::vector<float> staging(B * 16);
        buffered_scatter(data, temp, counts.data(), 0, staging);
        buffered_scatter(temp, data, counts.data() + B, 12, staging);
    } else {
        for(float x : data)
            temp[counts[hash_sort_key(x) & (B - 1)]++] = x;
        for(float x : temp)
            data[counts[B + (hash_sort_key(x) >> 12)]++] = x;
    }
}

static void sort_local(std::vector<float>& data, std::vector<float>& temp,
                       std::vector<size_t>& original_counts,
                       std::vector<uint32_t>& small_counts, int round) {
#if HW1_SORT == 1
    (void) original_counts; (void) round;
    radix_sort_float_11(data, temp, small_counts);
#elif HW1_SORT == 2
    if(round == 0) radix_sort_float(data, temp, original_counts);
    else radix_sort_hash24(data, temp, small_counts);
#else
    (void) small_counts; (void) round;
    radix_sort_float(data, temp, original_counts);
#endif
}

// Usage:
//   ./hw1 N input output [rounds=25]

// -----------------------------------------------------------------------------
// Sparse compare-split for one adjacent rank pair.
//
// The lower rank owns sorted block A of size m.
// The higher rank owns sorted block B of size n.
//
// If A.back() <= B.front(), the pair is already ordered and no data exchange is
// necessary.
//
// Otherwise, find an exchange count k such that:
//   lower rank sends A[m-k .. m-1]
//   higher rank sends B[0 .. k-1]
//
// Both ranks find the same k with a distributed multi-probe search. Each
// iteration exchanges two values at each of up to HW1_PROBES partitions.  Once k is known, only k floats
// are exchanged in each direction.
//
// Returns true iff this pair performed a real compare-split.
// -----------------------------------------------------------------------------
static bool sparse_compare_split(
    int rank,
    int partner,
    const std::vector<int>& rank_sizes,
    std::vector<float>& data,
    std::vector<float>& partner_buffer,
    std::vector<float>& output_buffer,
    int boundary_tag,
    int search_tag,
    int data_tag,
    MPI_Comm comm
) {
    HW1_TIMER(1);
    (void) output_buffer;
    const int local_n = static_cast<int>(data.size());
    const int partner_n = rank_sizes[partner];

    // With the initial contiguous block distribution, zero-sized ranks can
    // only appear at the end.  Both sides know all rank sizes, so both sides
    // take this branch consistently and no communication is attempted.
    if (local_n == 0 || partner_n == 0)
        return false;

    const bool am_lower = (rank < partner);

    // -------------------------------------------------------------------------
    // 1. Boundary check
    // -------------------------------------------------------------------------
    float my_boundary = am_lower ? data[local_n - 1] : data[0];
    float partner_boundary = 0.0f;

    MPI_Sendrecv(
        &my_boundary,
        1,
        MPI_FLOAT,
        partner,
        boundary_tag,
        &partner_boundary,
        1,
        MPI_FLOAT,
        partner,
        boundary_tag,
        comm,
        MPI_STATUS_IGNORE
    );

    float lower_max;
    float higher_min;

    if (am_lower) {
        lower_max = my_boundary;
        higher_min = partner_boundary;
    } else {
        lower_max = partner_boundary;
        higher_min = my_boundary;
    }

    const bool need_exchange = (lower_max > higher_min);

    if (!need_exchange)
        return false;

    // -------------------------------------------------------------------------
    // 2. Find the exact number k of elements that must cross the rank boundary.
    //
    // Let A be the lower-rank block with size m and B be the higher-rank block
    // with size n.  After compare-split, lower rank must still own m elements.
    // If k values are taken from B, exactly k values must leave A.
    //
    // The valid partition satisfies:
    //   A[m-k-1] <= B[k]     (when those indices exist)
    //   B[k-1]   <= A[m-k]   (when those indices exist)
    //
    // Both ranks run the same multi-probe search state. Lower sends A probes
    // and higher sends B probes. Thus both ranks derive
    // the same k and will use identical MPI counts in the final Sendrecv.
    // -------------------------------------------------------------------------
    const int lower_rank = std::min(rank, partner);
    const int higher_rank = std::max(rank, partner);

    const int m = rank_sizes[lower_rank];
    const int n = rank_sizes[higher_rank];

    int lo = 1;  // need_exchange == true guarantees k > 0
    int hi = std::min(m, n);
    int exchange_count = -1;

    const float neg_inf = -std::numeric_limits<float>::infinity();
    const float pos_inf = std::numeric_limits<float>::infinity();

    while (lo <= hi && exchange_count < 0) {
        const int span = hi - lo + 1;
        const int q = std::min(HW1_PROBES, span);
        int pivots[HW1_PROBES];
        float probes[2 * HW1_PROBES], peer[2 * HW1_PROBES];
        for(int j = 0; j < q; ++j) {
            const int k = lo + int(int64_t(j + 1) * span / (q + 1));
            pivots[j] = k;
            if(am_lower) {
                probes[2*j] = m-k-1 >= 0 ? data[m-k-1] : neg_inf;
                probes[2*j+1] = data[m-k];
            } else {
                probes[2*j] = data[k-1];
                probes[2*j+1] = k < n ? data[k] : pos_inf;
            }
        }
        MPI_Sendrecv(probes, 2*q, MPI_FLOAT, partner, search_tag,
                     peer, 2*q, MPI_FLOAT, partner, search_tag, comm, MPI_STATUS_IGNORE);
        for(int j = 0; j < q; ++j) {
            const float a_left = am_lower ? probes[2*j] : peer[2*j];
            const float a_right = am_lower ? probes[2*j+1] : peer[2*j+1];
            const float b_left = am_lower ? peer[2*j] : probes[2*j];
            const float b_right = am_lower ? peer[2*j+1] : probes[2*j+1];
            const int k = pivots[j];
            if(a_left > b_right) lo = k + 1;
            else if(b_left > a_right) { hi = k - 1; break; }
            else { exchange_count = k; break; }
        }
    }

    // For two finite sorted blocks with an inverted boundary, a valid
    // partition must exist.  Abort rather than silently risk mismatched MPI
    // counts if something unexpected happens.
    if (exchange_count <= 0) {
        std::fprintf(
            stderr,
            "Rank %d: failed to find compare-split partition with rank %d\n",
            rank,
            partner
        );
        MPI_Abort(comm, 2);
    }

    const int k = exchange_count;

    // k <= min(local_n, partner_n), so partner_buffer(local_n) is always large
    // enough on the receiving rank.
    const float* send_ptr = nullptr;

    if (am_lower) {
        // Lower rank sends only its largest k values.
        send_ptr = data.data() + (local_n - k);
    } else {
        // Higher rank sends only its smallest k values.
        send_ptr = data.data();
    }

    MPI_Sendrecv(
        send_ptr,
        k,
        MPI_FLOAT,
        partner,
        data_tag,
        partner_buffer.data(),
        k,
        MPI_FLOAT,
        partner,
        data_tag,
        comm,
        MPI_STATUS_IGNORE
    );

#if HW1_MERGE
    if(HW1_INTERLEAVED_MERGE && local_n >= 65536 && std::min(k, local_n-k) >= local_n/16) {
        const float* retained = data.data() + (am_lower ? 0 : k);
        merge_interleaved(retained, local_n-k, partner_buffer.data(), k, output_buffer.data());
        data.swap(output_buffer);
    } else {
        merge_sparse(data, partner_buffer, k, am_lower);
    }
#else
    // -------------------------------------------------------------------------
    // 3. Partial merge using reusable output_buffer.
    //
    // Lower rank:
    //   merge A[0 .. m-k-1] with received B[0 .. k-1]
    //   and produce exactly m values.
    //
    // Higher rank:
    //   merge received A[m-k .. m-1] with B[k .. n-1] from the back
    //   and produce exactly n values.
    // -------------------------------------------------------------------------
    if (am_lower) {
        int i = 0;
        int j = 0;
        int out = 0;

        const int own_end = local_n - k;

        while (out < local_n) {
            if (i < own_end &&
                (j >= k || data[i] <= partner_buffer[j])) {
                output_buffer[out++] = data[i++];
            } else {
                output_buffer[out++] = partner_buffer[j++];
            }
        }
    } else {
        int i = local_n - 1;
        int j = k - 1;
        int out = local_n - 1;

        const int own_begin = k;

        while (out >= 0) {
            if (i >= own_begin &&
                (j < 0 || data[i] >= partner_buffer[j])) {
                output_buffer[out--] = data[i--];
            } else {
                output_buffer[out--] = partner_buffer[j--];
            }
        }
    }

    data.swap(output_buffer);
#endif
    return true;
}

// Runtime dispatch keeps the default executable usable on other x86 CPUs.
// The hash.h implementation itself is unchanged.
#ifndef HW1_AVX2_HASH
#define HW1_AVX2_HASH 1
#endif
#if HW1_AVX2_HASH && defined(__GNUC__) && !defined(__clang__) && defined(__x86_64__)
__attribute__((target_clones("avx2", "default")))
#endif
static void hash_block(float* values, int count, uint32_t start, uint32_t round) {
    for(int i = 0; i < count; ++i)
        values[i] = hw1_hash(values[i], start + uint32_t(i), round);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
#ifdef HW1_PROFILE
    const double hw1_total_start = MPI_Wtime();
#endif

    int rank = 0;
    int size = 1;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 4) {
        if (rank == 0) {
            std::fprintf(
                stderr,
                "Usage: %s N input output [rounds=25]\n",
                argv[0]
            );
        }

        MPI_Finalize();
        return 1;
    }

    const long long n_ll = std::atoll(argv[1]);

    if (n_ll < 1 || n_ll > 536870911LL) {
        if (rank == 0)
            std::fprintf(stderr, "Invalid N\n");

        MPI_Finalize();
        return 1;
    }

    const int n = static_cast<int>(n_ll);

    const char* input_file = argv[2];
    const char* output_file = argv[3];

    int rounds = 25;

    if (argc >= 5)
        rounds = std::atoi(argv[4]);

    if (rounds < 1) {
        if (rank == 0)
            std::fprintf(stderr, "Invalid rounds\n");

        MPI_Finalize();
        return 1;
    }

    // =========================================================================
    // Initial block distribution
    // =========================================================================
    int active_size = size;
    if(HW1_FIRST_NODE) {
        char hostname[MPI_MAX_PROCESSOR_NAME] = {};
        int hostname_length = 0;
        MPI_Get_processor_name(hostname, &hostname_length);
        std::vector<char> hosts(size_t(size) * MPI_MAX_PROCESSOR_NAME);
        MPI_Allgather(hostname, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                      hosts.data(), MPI_MAX_PROCESSOR_NAME, MPI_CHAR, MPI_COMM_WORLD);
        active_size = 1;
        while(active_size < size &&
              std::strcmp(hosts.data(), hosts.data() + size_t(active_size) * MPI_MAX_PROCESSOR_NAME) == 0)
            ++active_size;
    }
    if(HW1_MAX_ACTIVE > 0) active_size = std::min(active_size, HW1_MAX_ACTIVE);
    const int base = n / active_size;
    const int rem = n % active_size;

    const int local_n = rank < active_size ? base + (rank < rem ? 1 : 0) : 0;

    // local_n never changes, so global_start is constant across all rounds.
    const int global_start = rank < active_size
        ? rank * base + std::min(rank, rem) : n;

    std::vector<int> rank_sizes(size);

    for (int r = 0; r < size; ++r) {
        rank_sizes[r] =
            r < active_size ? base + (r < rem ? 1 : 0) : 0;
    }

    std::vector<float> data(local_n);

    // Reusable buffers.  A rank never receives more than local_n values in the
    // sparse exchange because k <= min(local_n, partner_n).
    std::vector<float> partner_buffer(local_n);
    std::vector<float> output_buffer(local_n);

    // Reused by every local radix sort; no per-round allocation.
    std::vector<size_t> radix_count(1u << 16);
    std::vector<uint32_t> small_counts(HW1_SORT == 1 ? 6144 : HW1_SORT == 2 ? 8192 : 0);

    // =========================================================================
    // MPI-IO input
    // =========================================================================
    {
    HW1_TIMER(4);
    MPI_File input_fh;

    MPI_File_open(
        MPI_COMM_WORLD,
        input_file,
        MPI_MODE_RDONLY,
        MPI_INFO_NULL,
        &input_fh
    );

    const MPI_Offset input_offset =
        static_cast<MPI_Offset>(global_start) * sizeof(float);

    MPI_File_read_at_all(
        input_fh,
        input_offset,
        data.data(),
        local_n,
        MPI_FLOAT,
        MPI_STATUS_IGNORE
    );

    MPI_File_close(&input_fh);
    }

    // Keep the same finite-input safety check.
    for (float x : data) {
        if (!std::isfinite(x)) {
            std::fprintf(
                stderr,
                "Rank %d: invalid input value\n",
                rank
            );
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // =========================================================================
    // Multi-round sort + hash
    // =========================================================================
    for (int round = 0; round < rounds; ++round) {

        // Each round starts by sorting the rank's current local block.
        {
            HW1_TIMER(0);
            sort_local(data, output_buffer, radix_count, small_counts, round);
        }

        bool globally_sorted = HW1_STOP && (std::min(n, active_size) <= 1);

        // Check every two cycles. HW1_STOP uses the latest cycle only;
        // its zero-change result already proves all adjacent boundaries are ordered.
        int pending_changed = 0;
        int cycles_since_allreduce = 0;

        while (!globally_sorted) {
            int cycle_changed = 0;

            // =================================================================
            // EVEN phase: (0,1), (2,3), ...
            // =================================================================
            int partner;

            if ((rank & 1) == 0)
                partner = rank + 1;
            else
                partner = rank - 1;

            if (partner >= 0 && partner < size) {
                const bool changed = sparse_compare_split(
                    rank,
                    partner,
                    rank_sizes,
                    data,
                    partner_buffer,
                    output_buffer,
                    10,  // boundary tag
                    11,  // partition-search tag
                    12,  // sparse-data tag
                    MPI_COMM_WORLD
                );

                if (changed)
                    cycle_changed = 1;
            }

            // =================================================================
            // ODD phase: (1,2), (3,4), ...
            // =================================================================
            if ((rank & 1) == 0)
                partner = rank - 1;
            else
                partner = rank + 1;

            if (partner >= 0 && partner < size) {
                const bool changed = sparse_compare_split(
                    rank,
                    partner,
                    rank_sizes,
                    data,
                    partner_buffer,
                    output_buffer,
                    20,  // boundary tag
                    21,  // partition-search tag
                    22,  // sparse-data tag
                    MPI_COMM_WORLD
                );

                if (changed)
                    cycle_changed = 1;
            }

            // Preserve any change observed during the two-cycle window.
            if (cycle_changed)
                pending_changed = 1;

            ++cycles_since_allreduce;

            // =================================================================
            // Global termination detection every 2 complete cycles.
            //
            // This cannot terminate early: global_changed is zero only when no
            // rank changed in either of the two cycles in the current window.
            // It can only do some extra work compared with checking every cycle.
            // =================================================================
            if (cycles_since_allreduce == 2) {
                int global_changed = 0;

                {
                    HW1_TIMER(2);
                MPI_Allreduce(
                    HW1_STOP ? &cycle_changed : &pending_changed,
                    &global_changed,
                    1,
                    MPI_INT,
                    MPI_MAX,
                    MPI_COMM_WORLD
                );

                }
                globally_sorted = (global_changed == 0);

                pending_changed = 0;
                cycles_since_allreduce = 0;
            }
        }

        // global_start is unchanged because every rank always retains exactly
        // local_n elements.  No per-round MPI_Exscan is necessary.

        if (round + 1 < rounds) {
            HW1_TIMER(3);
            hash_block(data.data(), local_n, uint32_t(global_start), uint32_t(round + 1));
        }
    }

    // =========================================================================
    // MPI-IO output
    // =========================================================================
    {
    HW1_TIMER(5);
    MPI_File output_fh;

    MPI_File_open(
        MPI_COMM_WORLD,
        output_file,
        MPI_MODE_CREATE | MPI_MODE_WRONLY,
        MPI_INFO_NULL,
        &output_fh
    );

    MPI_File_set_size(
        output_fh,
        static_cast<MPI_Offset>(n) * sizeof(float)
    );

    const MPI_Offset output_offset =
        static_cast<MPI_Offset>(global_start) * sizeof(float);

    MPI_File_write_at_all(
        output_fh,
        output_offset,
        data.data(),
        local_n,
        MPI_FLOAT,
        MPI_STATUS_IGNORE
    );

    MPI_File_close(&output_fh);
    }
#ifdef HW1_PROFILE
    double local[7], maxima[7];
    for(int i = 0; i < 6; ++i) local[i] = hw1_times[i];
    local[6] = MPI_Wtime() - hw1_total_start;
    MPI_Reduce(local, maxima, 7, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if(rank == 0) std::fprintf(stderr,
        "HW1_PROFILE sort=%d merge=%d stop=%d ranks=%d active=%d N=%d rounds=%d "
        "local_sort=%.9f compare_split=%.9f termination=%.9f hash=%.9f "
        "input=%.9f output=%.9f total=%.9f\n",
        HW1_SORT, HW1_MERGE, HW1_STOP, size, active_size, n, rounds,
        maxima[0], maxima[1], maxima[2], maxima[3], maxima[4], maxima[5], maxima[6]);
#endif

    MPI_Finalize();
    return 0;
}

