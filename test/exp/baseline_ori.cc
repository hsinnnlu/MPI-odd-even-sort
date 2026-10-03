#include <mpi.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <vector>

#include "hash.h"

// Usage:
// ./hw1_profile N input output [rounds=25]

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 4) {
        if (rank == 0) {
            fprintf(stderr,
                    "Usage: %s N input output [rounds=25]\n",
                    argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    long long n_ll = atoll(argv[1]);
    if (n_ll < 1 || n_ll > INT_MAX) {
        if (rank == 0)
            fprintf(stderr, "Invalid N\n");
        MPI_Finalize();
        return 1;
    }

    int n = static_cast<int>(n_ll);
    const char* input_file = argv[2];
    const char* output_file = argv[3];

    int rounds = 25;
    if (argc >= 5)
        rounds = atoi(argv[4]);

    if (rounds < 1) {
        if (rank == 0)
            fprintf(stderr, "Invalid rounds\n");
        MPI_Finalize();
        return 1;
    }

    // ------------------------------------------------------------
    // Profiling counters
    // ------------------------------------------------------------

    double time_io = 0.0;
    double time_local_sort = 0.0;
    double time_sendrecv = 0.0;
    double time_merge = 0.0;
    double time_allreduce = 0.0;

    long long iteration_count = 0;

    // Synchronize before measuring total execution.
    MPI_Barrier(MPI_COMM_WORLD);
    double total_start = MPI_Wtime();

    // ------------------------------------------------------------
    // Initial data distribution
    // ------------------------------------------------------------

    int base = n / size;
    int rem = n % size;

    int local_n = base + (rank < rem ? 1 : 0);

    int global_start =
        rank * base + (rank < rem ? rank : rem);

    std::vector<float> data(local_n);

    // ------------------------------------------------------------
    // MPI-IO: input
    // ------------------------------------------------------------

    double t0 = MPI_Wtime();

    MPI_File input_fh;

    MPI_File_open(
        MPI_COMM_WORLD,
        input_file,
        MPI_MODE_RDONLY,
        MPI_INFO_NULL,
        &input_fh
    );

    MPI_Offset input_offset =
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

    time_io += MPI_Wtime() - t0;

    // Check invalid values.
    for (float x : data) {
        if (!std::isfinite(x)) {
            fprintf(stderr,
                    "Rank %d: invalid input value\n",
                    rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // ============================================================
    // Multi-round sorting
    // ============================================================

    for (int round = 0; round < rounds; ++round) {

        // --------------------------------------------------------
        // Local sort
        // --------------------------------------------------------

        t0 = MPI_Wtime();

        std::sort(data.begin(), data.end());

        time_local_sort += MPI_Wtime() - t0;

        // --------------------------------------------------------
        // Global Odd-Even Sort
        // --------------------------------------------------------

        bool globally_sorted = false;

        while (!globally_sorted) {

            ++iteration_count;

            int local_changed = 0;

            // ====================================================
            // EVEN PHASE
            //
            // (0,1), (2,3), (4,5), ...
            // ====================================================

            int partner;

            if (rank % 2 == 0)
                partner = rank + 1;
            else
                partner = rank - 1;

            if (partner >= 0 && partner < size) {

                int partner_n = 0;

                // -----------------------------------------------
                // Sendrecv #1: exchange block sizes
                // -----------------------------------------------

                t0 = MPI_Wtime();

                MPI_Sendrecv(
                    &local_n,
                    1,
                    MPI_INT,
                    partner,
                    0,

                    &partner_n,
                    1,
                    MPI_INT,
                    partner,
                    0,

                    MPI_COMM_WORLD,
                    MPI_STATUS_IGNORE
                );

                time_sendrecv += MPI_Wtime() - t0;

                std::vector<float> partner_data(partner_n);

                // -----------------------------------------------
                // Sendrecv #2: exchange complete blocks
                // -----------------------------------------------

                t0 = MPI_Wtime();

                MPI_Sendrecv(
                    data.data(),
                    local_n,
                    MPI_FLOAT,
                    partner,
                    1,

                    partner_data.data(),
                    partner_n,
                    MPI_FLOAT,
                    partner,
                    1,

                    MPI_COMM_WORLD,
                    MPI_STATUS_IGNORE
                );

                time_sendrecv += MPI_Wtime() - t0;

                // -----------------------------------------------
                // Merge + compare-split
                // -----------------------------------------------

                t0 = MPI_Wtime();

                std::vector<float> merged;
                merged.reserve(local_n + partner_n);

                std::merge(
                    data.begin(),
                    data.end(),
                    partner_data.begin(),
                    partner_data.end(),
                    std::back_inserter(merged)
                );

                std::vector<float> new_data;

                if (rank < partner) {
                    // Smaller rank keeps smaller local_n elements.
                    new_data.assign(
                        merged.begin(),
                        merged.begin() + local_n
                    );
                } else {
                    // Larger rank keeps larger local_n elements.
                    new_data.assign(
                        merged.end() - local_n,
                        merged.end()
                    );
                }

                if (new_data != data)
                    local_changed = 1;

                data.swap(new_data);

                time_merge += MPI_Wtime() - t0;
            }

            // ====================================================
            // ODD PHASE
            //
            // (1,2), (3,4), (5,6), ...
            // ====================================================

            if (rank % 2 == 0)
                partner = rank - 1;
            else
                partner = rank + 1;

            if (partner >= 0 && partner < size) {

                int partner_n = 0;

                // -----------------------------------------------
                // Sendrecv #1: exchange block sizes
                // -----------------------------------------------

                t0 = MPI_Wtime();

                MPI_Sendrecv(
                    &local_n,
                    1,
                    MPI_INT,
                    partner,
                    2,

                    &partner_n,
                    1,
                    MPI_INT,
                    partner,
                    2,

                    MPI_COMM_WORLD,
                    MPI_STATUS_IGNORE
                );

                time_sendrecv += MPI_Wtime() - t0;

                std::vector<float> partner_data(partner_n);

                // -----------------------------------------------
                // Sendrecv #2: exchange complete blocks
                // -----------------------------------------------

                t0 = MPI_Wtime();

                MPI_Sendrecv(
                    data.data(),
                    local_n,
                    MPI_FLOAT,
                    partner,
                    3,

                    partner_data.data(),
                    partner_n,
                    MPI_FLOAT,
                    partner,
                    3,

                    MPI_COMM_WORLD,
                    MPI_STATUS_IGNORE
                );

                time_sendrecv += MPI_Wtime() - t0;

                // -----------------------------------------------
                // Merge + compare-split
                // -----------------------------------------------

                t0 = MPI_Wtime();

                std::vector<float> merged;
                merged.reserve(local_n + partner_n);

                std::merge(
                    data.begin(),
                    data.end(),
                    partner_data.begin(),
                    partner_data.end(),
                    std::back_inserter(merged)
                );

                std::vector<float> new_data;

                if (rank < partner) {
                    new_data.assign(
                        merged.begin(),
                        merged.begin() + local_n
                    );
                } else {
                    new_data.assign(
                        merged.end() - local_n,
                        merged.end()
                    );
                }

                if (new_data != data)
                    local_changed = 1;

                data.swap(new_data);

                time_merge += MPI_Wtime() - t0;
            }

            // ====================================================
            // Termination detection
            // ====================================================

            int global_changed = 0;

            t0 = MPI_Wtime();

            MPI_Allreduce(
                &local_changed,
                &global_changed,
                1,
                MPI_INT,
                MPI_MAX,
                MPI_COMM_WORLD
            );

            time_allreduce += MPI_Wtime() - t0;

            globally_sorted = (global_changed == 0);
        }

        // --------------------------------------------------------
        // Global position for hash
        // --------------------------------------------------------

        local_n = static_cast<int>(data.size());

        global_start = 0;

        MPI_Exscan(
            &local_n,
            &global_start,
            1,
            MPI_INT,
            MPI_SUM,
            MPI_COMM_WORLD
        );

        if (rank == 0)
            global_start = 0;

        // --------------------------------------------------------
        // Hash except after final sorting round
        // --------------------------------------------------------

        if (round + 1 < rounds) {
            for (int i = 0; i < local_n; ++i) {
                data[i] = hw1_hash(
                    data[i],
                    static_cast<uint32_t>(global_start + i),
                    static_cast<uint32_t>(round + 1)
                );
            }
        }
    }

    // ------------------------------------------------------------
    // MPI-IO: output
    // ------------------------------------------------------------

    t0 = MPI_Wtime();

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

    MPI_Offset output_offset =
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

    time_io += MPI_Wtime() - t0;

    // ------------------------------------------------------------
    // Total time
    // ------------------------------------------------------------

    MPI_Barrier(MPI_COMM_WORLD);

    double total_time = MPI_Wtime() - total_start;

    // ============================================================
    // Reduce profiling statistics
    //
    // Report MAX rank time because parallel execution is limited
    // by the slowest rank.
    // ============================================================

    double max_total = 0.0;
    double max_io = 0.0;
    double max_local_sort = 0.0;
    double max_sendrecv = 0.0;
    double max_merge = 0.0;
    double max_allreduce = 0.0;

    long long max_iterations = 0;

    MPI_Reduce(
        &total_time,
        &max_total,
        1,
        MPI_DOUBLE,
        MPI_MAX,
        0,
        MPI_COMM_WORLD
    );

    MPI_Reduce(
        &time_io,
        &max_io,
        1,
        MPI_DOUBLE,
        MPI_MAX,
        0,
        MPI_COMM_WORLD
    );

    MPI_Reduce(
        &time_local_sort,
        &max_local_sort,
        1,
        MPI_DOUBLE,
        MPI_MAX,
        0,
        MPI_COMM_WORLD
    );

    MPI_Reduce(
        &time_sendrecv,
        &max_sendrecv,
        1,
        MPI_DOUBLE,
        MPI_MAX,
        0,
        MPI_COMM_WORLD
    );

    MPI_Reduce(
        &time_merge,
        &max_merge,
        1,
        MPI_DOUBLE,
        MPI_MAX,
        0,
        MPI_COMM_WORLD
    );

    MPI_Reduce(
        &time_allreduce,
        &max_allreduce,
        1,
        MPI_DOUBLE,
        MPI_MAX,
        0,
        MPI_COMM_WORLD
    );

    MPI_Reduce(
        &iteration_count,
        &max_iterations,
        1,
        MPI_LONG_LONG,
        MPI_MAX,
        0,
        MPI_COMM_WORLD
    );

    // ------------------------------------------------------------
    // Print result
    // ------------------------------------------------------------

    if (rank == 0) {

        double measured =
            max_io +
            max_local_sort +
            max_sendrecv +
            max_merge +
            max_allreduce;

        printf("\n");
        printf("============================================\n");
        printf(" HW1 Baseline Profiling\n");
        printf("============================================\n");

        printf("N               : %d\n", n);
        printf("Processes       : %d\n", size);
        printf("Rounds          : %d\n", rounds);

        printf("--------------------------------------------\n");

        printf("Total           : %.6f s\n", max_total);
        printf("I/O             : %.6f s\n", max_io);
        printf("Local sort      : %.6f s\n", max_local_sort);
        printf("Sendrecv        : %.6f s\n", max_sendrecv);
        printf("Merge           : %.6f s\n", max_merge);
        printf("Allreduce       : %.6f s\n", max_allreduce);

        printf("--------------------------------------------\n");

        printf("Measured sum    : %.6f s\n", measured);
        printf("Other           : %.6f s\n",
               max_total - measured);

        printf("Iterations      : %lld\n",
               max_iterations);

        printf("Iterations/round: %.3f\n",
               static_cast<double>(max_iterations) /
               rounds);

        printf("============================================\n");
    }

    MPI_Finalize();
    return 0;
}
