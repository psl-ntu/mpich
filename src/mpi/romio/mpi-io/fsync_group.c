/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "mpioimpl.h"

/* P2: Group Sync -- MPI_File_sync_group
 *
 * Implements the C2 (group consistency) level for an arbitrary subgroup
 * G ⊆ C of the communicator that opened the file.  All processes in G call
 * this function collectively; no synchronization is required with processes
 * outside G.
 *
 * Semantics (from the formal model):
 *   For all p, p' ∈ G, p ≠ p':
 *     s_grp^p  --sw_G-->  s_grp^{p'}
 * so that every write by any p ∈ G before the call is visible to every read
 * by any p' ∈ G after the call returns.
 *
 * Implementation: the sync-barrier-sync idiom restricted to G.
 *   1. ADIO_LocalFlush  -- flush this process's dirty writes to storage
 *   2. MPI_Barrier(comm)  -- all-pairs synchronization within G
 *   3. ADIO_LocalFlush  -- now that all peers have flushed, pull their data
 *                          into visibility (resets dirty_write; no-op if
 *                          nothing to do)
 *
 * comm is a subcommunicator spanning exactly G, created and owned by the
 * caller (e.g. via MPI_Comm_create_group at setup time) and reused across
 * repeated calls -- NOT created fresh inside this function. Building a new
 * subcommunicator on every call was tried first and measured to cost far
 * more than the barrier itself (MPI_Comm_create_group's own context-ID
 * negotiation dominates), which is why the API takes an already-built
 * communicator instead of a group. This function does not construct, tear
 * down, or validate comm: comm's group must be a subset of fh's
 * communicator's group (PRECONDITION, not checked -- same convention as
 * MPI_Comm_create_group's own "group must be a subset of comm" requirement,
 * and as P1/P3's rank/group arguments, neither of which are validated
 * against fh's communicator either). An earlier version validated this on
 * every call; that check alone cost as much as the rest of the primitive
 * combined, so it was removed to follow MPI's usual convention of trusting
 * the caller for handle-relationship preconditions rather than paying a
 * runtime cost on every call to catch programmer error.
 */

#ifdef HAVE_WEAK_SYMBOLS

#if defined(HAVE_PRAGMA_WEAK)
#pragma weak MPI_File_sync_group = PMPI_File_sync_group
#elif defined(HAVE_PRAGMA_HP_SEC_DEF)
#pragma _HP_SECONDARY_DEF PMPI_File_sync_group MPI_File_sync_group
#elif defined(HAVE_PRAGMA_CRI_DUP)
#pragma _CRI duplicate MPI_File_sync_group as PMPI_File_sync_group
#elif defined(HAVE_WEAK_ATTRIBUTE)
int MPI_File_sync_group(MPI_File fh, MPI_Comm comm)
    __attribute__ ((weak, alias("PMPI_File_sync_group")));
#endif

/* Include mapping from MPI->PMPI */
#define MPIO_BUILD_PROFILING
#include "mpioprof.h"
#endif

/*@
    MPI_File_sync_group - Flush writes to storage and synchronize with all
                          processes in a subcommunicator (group consistency,
                          C2 level)

Input Parameters:
. fh   - file handle (handle)
. comm - subcommunicator of processes that participate in this sync (handle);
         must be a subset of the communicator used to open fh (not verified;
         behavior is undefined if violated, as with MPI_Comm_create_group's
         own group-subset requirement), and should be created once by the
         caller and reused across repeated calls

Notes:
  All processes in comm must call this function collectively. After the
  call returns on all processes in comm, every write issued by any process
  in comm before the call is visible to reads issued by any process in comm
  after the call. Processes not in comm are not involved and are not blocked.
  This function does not create or free comm; the caller owns its lifetime.

.N fortran
@*/
int MPI_File_sync_group(MPI_File fh, MPI_Comm comm)
{
    int error_code;
    ROMIO_THREAD_CS_ENTER();

    error_code = MPIR_File_sync_group_impl(fh, comm);
    if (error_code) {
        goto fn_fail;
    }

  fn_exit:
    ROMIO_THREAD_CS_EXIT();
    return error_code;
  fn_fail:
    error_code = MPIO_Err_return_file(fh, error_code);
    goto fn_exit;
}
