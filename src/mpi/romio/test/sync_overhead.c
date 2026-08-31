/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 *
 * Sync overhead microbenchmark for MPI-IO directed/group/release-acquire
 * consistency primitives (P1, P2, P3).
 *
 * Measures the wall-clock latency of each synchronization mechanism in
 * isolation, across a sweep of process counts.  Each process writes 1 MiB
 * to ensure data spans multiple Lustre OSTs; this prevents OST lock contention
 * from non-participating processes from inflating the P2 subgroup measurements.
 * Data movement (the write and read calls surrounding the sync) is NOT
 * included in the timed region;
 * we measure only the synchronization overhead -- the quantity that the
 * formal cost theorems (Theorems 6, 8, 10 in the paper) bound analytically.
 *
 * Sync methods timed:
 *   0. sync-barrier-sync (MPI_File_sync + MPI_Barrier + MPI_File_sync)
 *   1. P1: MPI_File_sync_to / MPI_File_sync_from  (2-process pair)
 *   2. P2 full:    MPI_File_sync_group, G = all nprocs
 *   3. P2 quarter: MPI_File_sync_group, G = nprocs/4 (first quarter)
 *   4. P2 half:    MPI_File_sync_group, G = nprocs/2 (first half)
 *      P2's subcommunicators (comm_full/quarter/half) are each built ONCE at
 *      setup via MPI_Comm_create_group and reused for every timed iteration
 *      -- MPI_File_sync_group takes a caller-owned communicator, not a group,
 *      specifically so this setup cost is paid once rather than on every
 *      call (an earlier version built it fresh per call and measured that
 *      cost to dominate the barrier itself).
 *   5. P3: MPI_File_release / MPI_File_acquire, fixed 4 writers + 4 readers
 *          regardless of nprocs (remaining processes are bystanders).
 *          This is the intended use case: a small subset synchronizes while
 *          the rest continue unimpeded.  |W|x|R| = 16 messages total,
 *          flat with N, demonstrating the O(1) cost of Theorem 10 vs the
 *          baseline's O(log N) barrier that blocks all N processes.
 *
 * Output: CSV to stdout.
 *   nprocs, method, mean_us, stddev_us, min_us, max_us
 *
 * Usage:
 *   mpiexec -n <nprocs> sync_overhead -fname <path> [-iters N] [-warmup W]
 *
 * Typical use: run with increasing nprocs (8 32 128 512 1024 4096) and
 * collect the CSV into a table/figure for the paper's evaluation section.
 */

#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define DEFAULT_ITERS  500
#define DEFAULT_WARMUP  50
#define WRITE_SIZE     (1 << 20)  /* 1 MiB per process -- spreads data across Lustre OSTs */

static int mynod, nprocs;

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

/* ------------------------------------------------------------------ */
/* Statistics helpers                                                  */
/* ------------------------------------------------------------------ */

static double mean(const double *v, int n)
{
    double s = 0.0;
    int i;
    for (i = 0; i < n; i++) s += v[i];
    return s / n;
}

static double stddev(const double *v, int n, double m)
{
    double s = 0.0;
    int i;
    for (i = 0; i < n; i++) s += (v[i] - m) * (v[i] - m);
    return sqrt(s / n);
}

static double dmin(const double *v, int n)
{
    double m = v[0];
    int i;
    for (i = 1; i < n; i++) if (v[i] < m) m = v[i];
    return m;
}

static double dmax(const double *v, int n)
{
    double m = v[0];
    int i;
    for (i = 1; i < n; i++) if (v[i] > m) m = v[i];
    return m;
}

/* Report the MAX latency across all processes (the bottleneck is whoever
 * finishes last -- this matches how barrier latency is typically reported). */
static double global_max_latency(double local_t)
{
    double gmax;
    MPI_Reduce(&local_t, &gmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    return gmax;
}

/* ------------------------------------------------------------------ */
/* Write payload to establish dirty_write state before timing sync    */
/* ------------------------------------------------------------------ */

static char *wbuf;

static void do_write(MPI_File fh)
{
    MPI_Status status;
    MPI_Offset off = (MPI_Offset)mynod * WRITE_SIZE;
    MPI_CHECK(MPI_File_write_at(fh, off, wbuf, WRITE_SIZE, MPI_BYTE, &status));
}

/* ------------------------------------------------------------------ */
/* Time one sync method over `iters` iterations.                      */
/* Returns array of per-iteration latencies in microseconds (caller   */
/* frees).  Rank 0 holds the global-max latencies; other ranks hold   */
/* their own (unused by caller).                                      */
/* ------------------------------------------------------------------ */

typedef struct {
    int method;         /* 0..5 */
    MPI_File fh;
    MPI_Comm comm_full, comm_quarter, comm_half;  /* P2: prebuilt, reused */
    MPI_Group w_group, r_group;  /* P3: fixed-size groups */
    int p3_nw, p3_nr;            /* number of writers / readers in P3 */
} SyncCtx;

static double *time_sync(SyncCtx *ctx, int iters)
{
    double *times = (double *)malloc(iters * sizeof(double));
    int it;

    for (it = 0; it < iters; it++) {
        double t0, t1, elapsed;

        /* Write to establish dirty state -- NOT timed */
        do_write(ctx->fh);

        /* Timed region */
        MPI_Barrier(MPI_COMM_WORLD);   /* align start */
        t0 = MPI_Wtime();

        switch (ctx->method) {
        case 0: /* sync-barrier-sync (full communicator) */
            MPI_CHECK(MPI_File_sync(ctx->fh));
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_CHECK(MPI_File_sync(ctx->fh));
            break;

        case 1: /* P1: directed sync between rank 0 (writer) and rank 1 (reader) */
            if (mynod == 0)
                MPI_CHECK(MPI_File_sync_to(ctx->fh, 1, MPI_COMM_WORLD));
            else if (mynod == 1)
                MPI_CHECK(MPI_File_sync_from(ctx->fh, 0, MPI_COMM_WORLD));
            else
                /* Remaining ranks do nothing -- they are not part of the pair */
                (void)0;
            break;

        case 2: /* P2 full: sync_group on all nprocs, on the prebuilt comm */
            MPI_CHECK(MPI_File_sync_group(ctx->fh, ctx->comm_full));
            break;

        case 3: /* P2 quarter: sync_group on nprocs/4, on the prebuilt comm */
            if (mynod < nprocs / 4)
                MPI_CHECK(MPI_File_sync_group(ctx->fh, ctx->comm_quarter));
            break;

        case 4: /* P2 half: sync_group on nprocs/2, on the prebuilt comm */
            if (mynod < nprocs / 2)
                MPI_CHECK(MPI_File_sync_group(ctx->fh, ctx->comm_half));
            break;

        case 5: /* P3: release-acquire, fixed 4w+4r, rest are bystanders.
                 * Only processes actually IN w_group / r_group participate;
                 * the condition must match the group membership exactly. */
            if (mynod < ctx->p3_nw)
                MPI_CHECK(MPI_File_release(ctx->fh, ctx->w_group, ctx->r_group));
            else if (mynod >= nprocs - ctx->p3_nr)
                MPI_CHECK(MPI_File_acquire(ctx->fh, ctx->w_group, ctx->r_group));
            /* else: bystander -- not in either group, do nothing */
            break;
        }

        t1 = MPI_Wtime();

        elapsed = (t1 - t0) * 1e6;   /* microseconds */
        times[it] = global_max_latency(elapsed);
    }

    return times;
}

static void print_stats(const char *label, const double *times, int iters)
{
    if (mynod != 0) return;
    double m  = mean(times, iters);
    double sd = stddev(times, iters, m);
    printf("%d,%s,%.2f,%.2f,%.2f,%.2f\n",
           nprocs, label, m, sd, dmin(times, iters), dmax(times, iters));
    fflush(stdout);
}

/* ------------------------------------------------------------------ */
/* Build groups                                                        */
/* ------------------------------------------------------------------ */

static MPI_Group make_range_group(int first, int last)
{
    MPI_Group world, g;
    int *ranks, i, n = last - first + 1;
    ranks = (int *)malloc(n * sizeof(int));
    for (i = 0; i < n; i++) ranks[i] = first + i;
    MPI_Comm_group(MPI_COMM_WORLD, &world);
    MPI_Group_incl(world, n, ranks, &g);
    MPI_Group_free(&world);
    free(ranks);
    return g;
}

/* Builds a subcommunicator spanning ranks [first, last], for use with
 * MPI_File_sync_group (which takes a caller-owned communicator, not a
 * group -- see its comment). Only ranks in [first, last] actually call
 * MPI_Comm_create_group, as MPI-3.0 requires; other ranks get MPI_COMM_NULL
 * back without participating in any collective here. Called once at setup,
 * not on every sync -- that's the whole point of this API shape. */
static MPI_Comm make_range_comm(int first, int last, int tag)
{
    MPI_Group world, g;
    MPI_Comm comm = MPI_COMM_NULL;
    int *ranks, i, n = last - first + 1;
    ranks = (int *)malloc(n * sizeof(int));
    for (i = 0; i < n; i++) ranks[i] = first + i;
    MPI_Comm_group(MPI_COMM_WORLD, &world);
    MPI_Group_incl(world, n, ranks, &g);
    MPI_Group_free(&world);
    free(ranks);

    if (mynod >= first && mynod <= last)
        MPI_CHECK(MPI_Comm_create_group(MPI_COMM_WORLD, g, tag, &comm));
    MPI_Group_free(&g);
    return comm;
}

/* ------------------------------------------------------------------ */
/* Main                                                                */
/* ------------------------------------------------------------------ */

int main(int argc, char **argv)
{
    char *filename = NULL;
    int iters = DEFAULT_ITERS, warmup = DEFAULT_WARMUP;
    int i;
    MPI_File fh;
    MPI_Info info;
    SyncCtx ctx;
    double *times;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mynod);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    for (i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-fname") && i + 1 < argc)
            filename = argv[++i];
        else if (!strcmp(argv[i], "-iters") && i + 1 < argc)
            iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-warmup") && i + 1 < argc)
            warmup = atoi(argv[++i]);
    }
    if (!filename) {
        if (mynod == 0)
            fprintf(stderr, "Usage: %s -fname <path> [-iters N] [-warmup W]\n",
                    argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (nprocs < 2) {
        if (mynod == 0) fprintf(stderr, "Need at least 2 processes.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Print CSV header */
    if (mynod == 0) {
        printf("# MPI-IO sync overhead microbenchmark\n");
        printf("# nprocs=%d  iters=%d  warmup=%d  write_size=%dB\n",
               nprocs, iters, warmup, WRITE_SIZE);
        printf("nprocs,method,mean_us,stddev_us,min_us,max_us\n");
        fflush(stdout);
    }

    /* Open file */
    MPI_CHECK(MPI_Info_create(&info));
    MPI_CHECK(MPI_Info_set(info, "direct_io", "true"));
    MPI_CHECK(MPI_File_open(MPI_COMM_WORLD, filename,
                            MPI_MODE_CREATE | MPI_MODE_RDWR, info, &fh));
    MPI_CHECK(MPI_Info_free(&info));

    /* Build P2's subcommunicators once, up front -- reused for every timed
     * iteration below (see MPI_File_sync_group's comment for why). */
    memset(&ctx, 0, sizeof(ctx));
    ctx.fh          = fh;
    ctx.comm_full    = make_range_comm(0, nprocs - 1, 200);
    ctx.comm_quarter = (nprocs >= 4) ? make_range_comm(0, nprocs / 4 - 1, 201)
                                     : ctx.comm_full;
    ctx.comm_half    = (nprocs >= 2) ? make_range_comm(0, nprocs / 2 - 1, 202)
                                     : ctx.comm_full;

    /* P3 uses a fixed group size (4 writers, 4 readers) regardless of nprocs.
     * Writers = first p3_nw ranks; readers = last p3_nr ranks.
     * The rest are bystanders and do not participate in any call.
     * This models the realistic use case where only a small fraction of
     * processes need to synchronize, exposing the O(|W|x|R|) flat cost. */
    ctx.p3_nw   = (nprocs >= 8) ? 4 : 1;
    ctx.p3_nr   = (nprocs >= 8) ? 4 : 1;
    ctx.w_group = make_range_group(0, ctx.p3_nw - 1);
    ctx.r_group = make_range_group(nprocs - ctx.p3_nr, nprocs - 1);

    wbuf = (char *)malloc(WRITE_SIZE);
    memset(wbuf, mynod & 0xFF, WRITE_SIZE);

    /* Warm up (results discarded) */
    for (i = 0; i < 6; i++) {
        ctx.method = i;
        times = time_sync(&ctx, warmup);
        free(times);
    }

    /* Timed runs */
    const char *labels[] = {
        "baseline-sync-bar-sync",
        "P1-directed",
        "P2-full",
        "P2-quarter",
        "P2-half",
        "P3-release-acquire"
    };

    for (i = 0; i < 6; i++) {
        /* Skip subgroup methods when nprocs < 4 */
        if (nprocs < 4 && (i == 3 || i == 4 || i == 5)) continue;

        ctx.method = i;
        times = time_sync(&ctx, iters);
        print_stats(labels[i], times, iters);
        free(times);
    }

    /* Cleanup. Check comm_quarter's alias to comm_full BEFORE freeing
     * comm_full -- freeing sets ctx.comm_full to MPI_COMM_NULL by reference,
     * which would make a post-free comparison wrongly see them as distinct
     * and double-free the same handle. Every rank has a real comm_full (it
     * spans everyone), but comm_quarter/comm_half are MPI_COMM_NULL on ranks
     * outside their range, so guard each free accordingly. */
    free(wbuf);
    {
        int quarter_is_alias = (ctx.comm_quarter == ctx.comm_full);
        if (ctx.comm_full != MPI_COMM_NULL)
            MPI_Comm_free(&ctx.comm_full);
        if (!quarter_is_alias && ctx.comm_quarter != MPI_COMM_NULL)
            MPI_Comm_free(&ctx.comm_quarter);
        if (ctx.comm_half != MPI_COMM_NULL)
            MPI_Comm_free(&ctx.comm_half);
    }
    MPI_Group_free(&ctx.w_group);
    MPI_Group_free(&ctx.r_group);
    MPI_CHECK(MPI_File_close(&fh));

    MPI_Finalize();
    return 0;
}
