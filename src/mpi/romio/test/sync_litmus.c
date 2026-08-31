/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

/*
 * Litmus test for the directed/group/release-acquire MPI-IO consistency
 * primitives (P1, P2, P3) introduced in the formal MPI-IO consistency model.
 *
 * Three sub-tests, one per primitive:
 *
 *   T1 -- P1, Directed Sync, AP_pc (producer-consumer):
 *     Process 0 writes N_ITERS sentinel values; MPI_File_sync_to/from
 *     synchronizes with process 1, which reads and verifies each value.
 *     Verifies: 0 violations over N_ITERS iterations.
 *
 *   T2 -- P2, Group Sync, AP_G (subgroup):
 *     Processes {0..k-1} form subgroup G (k = nprocs/2), which builds a
 *     subcommunicator once and reuses it across iterations.  Each writes its
 *     rank to a dedicated file slot, calls MPI_File_sync_group on that
 *     subcommunicator, then reads all peers' slots.  Processes {k..n-1} open
 *     the file but deliberately do NOT participate in the sync, exercising
 *     the subgroup isolation property.
 *     Verifies: 0 violations within G over N_ITERS iterations.
 *
 *   T3 -- P3, Release-Acquire, AP_WR (writer-reader phases):
 *     Processes {0..w-1} are writers (w = nprocs/2); {w..n-1} are readers.
 *     Writers write, call MPI_File_release; readers call MPI_File_acquire,
 *     then read and verify.
 *     Verifies: 0 violations over N_ITERS iterations.
 *
 * Each sub-test also runs with sync-barrier-sync as a sanity baseline.
 *
 * With --check-violations: additionally runs each pattern WITHOUT any sync
 * and reports the observed violation rate.  Violations are
 * filesystem-dependent: reliably observed on distributed filesystems
 * (Lustre, NFS) or with direct_io; may not appear on single-node local
 * filesystems where processes share the OS page cache.
 *
 * Usage:
 *   mpiexec -n <nprocs> sync_litmus -fname <path> [--check-violations]
 *
 * Minimum: 2 processes for T1; 4 processes for T2 and T3.
 */

#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_ITERS     200
#define SENTINEL    0xDEADBEEF
#define T2_SUBCOMM_TAG 100

/* Each "slot" in the file is one int (4 bytes).  We use nprocs slots so
 * each process has its own dedicated region, avoiding unintentional conflicts
 * in tests that do not test conflicting accesses. */
#define SLOT_BYTES  ((MPI_Offset)sizeof(int))

static int mynod, nprocs;
static int check_violations = 0;

/* ------------------------------------------------------------------ */
/* Utilities                                                           */
/* ------------------------------------------------------------------ */

static void handle_error(int errcode, const char *str)
{
    char msg[MPI_MAX_ERROR_STRING];
    int resultlen;
    MPI_Error_string(errcode, msg, &resultlen);
    fprintf(stderr, "rank %d: %s: %s\n", mynod, str, msg);
    MPI_Abort(MPI_COMM_WORLD, 1);
}

#define MPI_CHECK(fn) \
    do { int _e = (fn); if (_e != MPI_SUCCESS) handle_error(_e, #fn); } while(0)

/* Open the test file with direct_io hint (best-effort; ignored if unsupported)
 * to bypass the OS page cache and maximize violation probability. */
static MPI_File open_file(const char *filename, MPI_Comm comm)
{
    MPI_File fh;
    MPI_Info info;
    MPI_CHECK(MPI_Info_create(&info));
    MPI_CHECK(MPI_Info_set(info, "direct_io", "true"));
    MPI_CHECK(MPI_File_open(comm, filename,
                            MPI_MODE_CREATE | MPI_MODE_RDWR, info, &fh));
    MPI_CHECK(MPI_Info_free(&info));
    return fh;
}

/* Write a single int value to the file at slot `slot`. */
static void write_slot(MPI_File fh, int slot, int value)
{
    MPI_Status status;
    MPI_CHECK(MPI_File_write_at(fh, slot * SLOT_BYTES,
                                &value, 1, MPI_INT, &status));
}

/* Read a single int value from slot `slot`.  Returns the value read. */
static int read_slot(MPI_File fh, int slot)
{
    int value;
    MPI_Status status;
    MPI_CHECK(MPI_File_read_at(fh, slot * SLOT_BYTES,
                               &value, 1, MPI_INT, &status));
    return value;
}

/* Zero-initialize all nprocs slots (called before each sub-test). */
static void init_file(const char *filename)
{
    if (mynod == 0) {
        MPI_File fh;
        int zero = 0;
        int i;
        MPI_CHECK(MPI_File_open(MPI_COMM_SELF, filename,
                                MPI_MODE_CREATE | MPI_MODE_RDWR,
                                MPI_INFO_NULL, &fh));
        for (i = 0; i < nprocs; i++)
            write_slot(fh, i, 0);
        MPI_CHECK(MPI_File_sync(fh));
        MPI_CHECK(MPI_File_close(&fh));
        (void)zero;
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

/* Sum up local violation count across all processes. */
static int global_violations(int local_errs)
{
    int total;
    MPI_Allreduce(&local_errs, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    return total;
}

/* ------------------------------------------------------------------ */
/* T1: P1 Directed Sync -- producer-consumer (AP_pc)                  */
/* ------------------------------------------------------------------ */
/*
 * Pattern: process 0 writes N_ITERS sentinel values to slot 0; for each
 * write it synchronizes to process 1 using the supplied sync method.
 * Process 1 reads slot 0 after the sync and checks for the sentinel.
 *
 * sync_mode: 0 = none, 1 = baseline (sync-barrier-sync), 2 = P1
 *
 * Returns: number of violations observed by process 1.
 */
static int t1_run(const char *filename, int sync_mode)
{
    MPI_File fh;
    int iter, errs = 0;
    const int WRITER = 0, READER = 1;

    if (nprocs < 2) return 0;   /* skip: need at least 2 processes */

    init_file(filename);
    fh = open_file(filename, MPI_COMM_WORLD);

    for (iter = 1; iter <= N_ITERS; iter++) {
        int val = (int)(SENTINEL ^ (unsigned)iter);

        /* Writer: write a unique sentinel-based value */
        if (mynod == WRITER)
            write_slot(fh, 0, val);

        /* Forward sync: guarantees reader sees the write (modes 1 and 2).
         * Collective operations involve ALL processes; P1 point-to-point
         * only touches WRITER and READER. */
        switch (sync_mode) {
        case 1: /* sync-barrier-sync: all processes participate */
            MPI_CHECK(MPI_File_sync(fh));
            MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
            MPI_CHECK(MPI_File_sync(fh));
            break;
        case 2: /* P1: directed sync (point-to-point, only WRITER/READER) */
            if (mynod == WRITER)
                MPI_CHECK(MPI_File_sync_to(fh, READER, MPI_COMM_WORLD));
            else if (mynod == READER)
                MPI_CHECK(MPI_File_sync_from(fh, WRITER, MPI_COMM_WORLD));
            break;
        default: /* no sync: reader may observe stale data (violation expected) */
            break;
        }

        /* Reader: verify */
        if (mynod == READER) {
            int got = read_slot(fh, 0);
            if (got != val)
                errs++;
        }

        /* End-of-iteration barrier: prevents the writer from starting the
         * next write before the reader has finished reading the current slot.
         * Also paces iterations for the no-sync violation test. */
        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    }

    MPI_CHECK(MPI_File_close(&fh));
    return global_violations(errs);
}

/* ------------------------------------------------------------------ */
/* T2: P2 Group Sync -- subgroup consistency (AP_G)                   */
/* ------------------------------------------------------------------ */
/*
 * Pattern: G = {0 .. k-1}, k = nprocs/2.  Each process in G writes its
 * rank to its dedicated file slot, syncs within G, then reads all peers'
 * slots and verifies.  Processes outside G open the file but do NOT call
 * any sync, exercising the subgroup isolation property.
 *
 * sync_mode: 0 = none, 1 = baseline (sync-barrier-sync on ALL), 2 = P2
 */
static int t2_run(const char *filename, int sync_mode)
{
    MPI_File fh;
    MPI_Group world_group, g_group;
    MPI_Comm g_comm = MPI_COMM_NULL;
    int k, iter, errs = 0, in_G;
    int i;

    if (nprocs < 4) return 0;

    k = nprocs / 2;
    in_G = (mynod < k);

    /* Build group G from ranks 0..k-1, and a subcommunicator spanning G,
     * ONCE here -- MPI_File_sync_group takes a caller-owned subcommunicator
     * and reuses it across all N_ITERS calls below rather than recreating
     * one per call (recreating it per call was measured to cost far more
     * than the barrier itself; see MPIR_File_sync_group_impl's comment). */
    {
        int *g_ranks = (int *)malloc(k * sizeof(int));
        for (i = 0; i < k; i++) g_ranks[i] = i;
        MPI_Comm_group(MPI_COMM_WORLD, &world_group);
        MPI_Group_incl(world_group, k, g_ranks, &g_group);
        MPI_Group_free(&world_group);
        free(g_ranks);
    }
    if (in_G)
        MPI_CHECK(MPI_Comm_create_group(MPI_COMM_WORLD, g_group, T2_SUBCOMM_TAG, &g_comm));

    init_file(filename);
    fh = open_file(filename, MPI_COMM_WORLD);

    for (iter = 1; iter <= N_ITERS; iter++) {

        /* All processes in G write their rank as value */
        if (in_G)
            write_slot(fh, mynod, iter * 1000 + mynod);

        /* Synchronize */
        switch (sync_mode) {
        case 1: /* baseline: full communicator sync-barrier-sync */
            MPI_CHECK(MPI_File_sync(fh));
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_CHECK(MPI_File_sync(fh));
            break;
        case 2: /* P2: group sync -- only G participates, on the
                 * pre-built, reused subcommunicator */
            if (in_G)
                MPI_CHECK(MPI_File_sync_group(fh, g_comm));
            /* Processes outside G do nothing -- this is the correctness point */
            break;
        default: /* no sync */
            MPI_Barrier(MPI_COMM_WORLD);
            break;
        }

        /* All processes in G verify all slots in G */
        if (in_G) {
            for (i = 0; i < k; i++) {
                int expected = iter * 1000 + i;
                int got = read_slot(fh, i);
                if (got != expected)
                    errs++;
            }
        }

        /* End-of-iteration barrier: prevents in-G processes from starting the
         * next write before all peers have finished reading this iteration. */
        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    }

    if (in_G)
        MPI_Comm_free(&g_comm);
    MPI_Group_free(&g_group);
    MPI_CHECK(MPI_File_close(&fh));
    return global_violations(errs);
}

/* ------------------------------------------------------------------ */
/* T3: P3 Release-Acquire -- writer-reader phases (AP_WR)             */
/* ------------------------------------------------------------------ */
/*
 * Pattern: W = {0..w-1}, R = {w..n-1}, w = nprocs/2.
 * Each writer writes iter*1000+mynod to its slot; release.
 * Each reader: acquire, then reads all writers' slots and verifies.
 *
 * sync_mode: 0 = none, 1 = baseline, 2 = P3
 */
static int t3_run(const char *filename, int sync_mode)
{
    MPI_File fh;
    MPI_Group world_group, w_group, r_group;
    int w, iter, errs = 0, is_writer;
    int i;
    int *w_ranks, *r_ranks;

    if (nprocs < 4) return 0;

    w = nprocs / 2;
    is_writer = (mynod < w);

    w_ranks = (int *)malloc(w * sizeof(int));
    r_ranks = (int *)malloc((nprocs - w) * sizeof(int));
    for (i = 0; i < w; i++)           w_ranks[i] = i;
    for (i = 0; i < nprocs - w; i++)  r_ranks[i] = w + i;

    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    MPI_Group_incl(world_group, w,           w_ranks, &w_group);
    MPI_Group_incl(world_group, nprocs - w,  r_ranks, &r_group);
    MPI_Group_free(&world_group);
    free(w_ranks);
    free(r_ranks);

    init_file(filename);
    fh = open_file(filename, MPI_COMM_WORLD);

    for (iter = 1; iter <= N_ITERS; iter++) {

        /* Writers write */
        if (is_writer)
            write_slot(fh, mynod, iter * 1000 + mynod);

        /* Synchronize */
        switch (sync_mode) {
        case 1:
            MPI_CHECK(MPI_File_sync(fh));
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_CHECK(MPI_File_sync(fh));
            break;
        case 2:
            if (is_writer)
                MPI_CHECK(MPI_File_release(fh, w_group, r_group));
            else
                MPI_CHECK(MPI_File_acquire(fh, w_group, r_group));
            break;
        default:
            MPI_Barrier(MPI_COMM_WORLD);
            break;
        }

        /* Readers verify all writers' slots */
        if (!is_writer) {
            for (i = 0; i < w; i++) {
                int expected = iter * 1000 + i;
                int got = read_slot(fh, i);
                if (got != expected)
                    errs++;
            }
        }

        /* End-of-iteration barrier: prevents writers from starting the next
         * write before readers finish reading, and also unblocks any writer
         * Waitall stalled waiting for readers to post their acquire Irecv. */
        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    }

    MPI_Group_free(&w_group);
    MPI_Group_free(&r_group);
    MPI_CHECK(MPI_File_close(&fh));
    return global_violations(errs);
}

/* ------------------------------------------------------------------ */
/* Main                                                                */
/* ------------------------------------------------------------------ */

int main(int argc, char **argv)
{
    char *filename = NULL;
    int i, total_errs = 0;
    int v;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mynod);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    /* Parse arguments */
    for (i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-fname") && i + 1 < argc) {
            filename = argv[++i];
        } else if (!strcmp(argv[i], "--check-violations")) {
            check_violations = 1;
        }
    }
    if (!filename) {
        if (mynod == 0)
            fprintf(stderr, "Usage: %s -fname <path> [--check-violations]\n",
                    argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (mynod == 0) {
        printf("MPI-IO sync primitive litmus test\n");
        printf("  processes: %d, iterations per sub-test: %d\n",
               nprocs, N_ITERS);
        printf("  check-violations mode: %s\n\n",
               check_violations ? "ON (results are filesystem-dependent)" : "off");
    }

    /* ---- T1: P1 Directed Sync ---- */
    if (nprocs < 2) {
        if (mynod == 0) printf("T1 (P1 Directed Sync): SKIP (need >= 2 procs)\n");
    } else {
        if (mynod == 0) printf("T1: P1 Directed Sync (AP_pc, 2 procs)\n");

        if (check_violations) {
            v = t1_run(filename, 0);
            if (mynod == 0)
                printf("  no-sync:          %4d / %d violations (filesystem-dependent)\n",
                       v, N_ITERS);
        }

        v = t1_run(filename, 1);
        if (mynod == 0)
            printf("  baseline sync:    %4d violations -- %s\n",
                   v, v == 0 ? "PASS" : "FAIL (baseline broken)");
        total_errs += v;

        v = t1_run(filename, 2);
        if (mynod == 0)
            printf("  MPI_File_sync_to/from: %d violations -- %s\n",
                   v, v == 0 ? "PASS" : "FAIL");
        total_errs += v;
    }

    /* ---- T2: P2 Group Sync ---- */
    if (nprocs < 4) {
        if (mynod == 0) printf("\nT2 (P2 Group Sync): SKIP (need >= 4 procs)\n");
    } else {
        if (mynod == 0)
            printf("\nT2: P2 Group Sync (AP_G, subgroup G = first %d of %d procs)\n",
                   nprocs / 2, nprocs);

        if (check_violations) {
            v = t2_run(filename, 0);
            if (mynod == 0)
                printf("  no-sync:          %4d / %d violations (filesystem-dependent)\n",
                       v, N_ITERS * (nprocs / 2) * (nprocs / 2));
        }

        v = t2_run(filename, 1);
        if (mynod == 0)
            printf("  baseline sync:    %4d violations -- %s\n",
                   v, v == 0 ? "PASS" : "FAIL (baseline broken)");
        total_errs += v;

        v = t2_run(filename, 2);
        if (mynod == 0)
            printf("  MPI_File_sync_group: %d violations -- %s\n",
                   v, v == 0 ? "PASS" : "FAIL");
        total_errs += v;
    }

    /* ---- T3: P3 Release-Acquire ---- */
    if (nprocs < 4) {
        if (mynod == 0) printf("\nT3 (P3 Release-Acquire): SKIP (need >= 4 procs)\n");
    } else {
        if (mynod == 0)
            printf("\nT3: P3 Release-Acquire (AP_WR, %d writers / %d readers)\n",
                   nprocs / 2, nprocs - nprocs / 2);

        if (check_violations) {
            v = t3_run(filename, 0);
            if (mynod == 0)
                printf("  no-sync:          %4d / %d violations (filesystem-dependent)\n",
                       v, N_ITERS * (nprocs / 2) * (nprocs - nprocs / 2));
        }

        v = t3_run(filename, 1);
        if (mynod == 0)
            printf("  baseline sync:    %4d violations -- %s\n",
                   v, v == 0 ? "PASS" : "FAIL (baseline broken)");
        total_errs += v;

        v = t3_run(filename, 2);
        if (mynod == 0)
            printf("  MPI_File_release/acquire: %d violations -- %s\n",
                   v, v == 0 ? "PASS" : "FAIL");
        total_errs += v;
    }

    /* ---- Summary ---- */
    if (mynod == 0) {
        printf("\n");
        if (total_errs == 0)
            printf(" No Errors\n");
        else
            printf("Found %d errors\n", total_errs);
    }

    MPI_Finalize();
    return (total_errs > 0);
}
