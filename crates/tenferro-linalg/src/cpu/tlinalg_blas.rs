//! Adapter from tenferro's CPU session to the extracted LAPACK/BLAS provider (`tlinalg-blas`).
//!
//! The provider owns batch execution, scratch requirements and vendor calls.
//! Vendor/application configuration owns threading. Tenferro supplies pooled
//! storage through [`super::tlinalg_workspace::TlinalgWorkspace`], not lane policy.
//!
//! Unlike the faer adapter, this one passes no parallelism token, because
//! `tlinalg-blas` has none to take: its batch loop is serial and the vendor call
//! inside it owns its own threading, so a Rayon fan-out around it would fight the
//! vendor's pool. Tenferro places the call on the coordinator thread and gives
//! the provider only pooled scratch. Every vendor call runs through
//! [`vendor_call`], which gives a coordinator that leads a placed team its full
//! CPU set back for the call (#2553).

#![cfg(feature = "blas")]

use tcontract_exec::Exec;
use tenferro_cpu::linalg_interop::BufferPool;
use tlinalg_blas::lu::{lu_factor, lu_factor_solve, lu_solve_prepared};
use tlinalg_blas::{LapackScalar, Op};

use super::tlinalg_error::map_blas_error;
use super::tlinalg_workspace::TlinalgWorkspace;

/// Run one vendor BLAS/LAPACK call with the calling thread's full CPU set.
///
/// threadteam 0.5 pins the thread that builds a `Team` (the engine's leader) to `{S[0]}` for
/// the team's life, and a vendor library creates its own thread team from the calling thread.
/// A call made from that leader therefore confined the whole vendor team to one CPU: with
/// `MKL_NUM_THREADS=4` on a 4-thread engine, MKL's four OpenMP threads shared one CPU (#2553);
/// with the count unset, MKL saw one CPU and ran single-threaded.
/// [`tenferro_cpu::threadteam::TeamExec::external`] widens the leader back to `S` for the call
/// and pins it again after it, on return and on unwind. That widening is decided by the calling
/// thread's own placement, so the team of one of a serial token is enough: no member is
/// reserved, no section is dispatched, and on a thread nobody placed (a serial or one-worker
/// engine, a pool with pinning off, a caller whose mask is smaller than the team) this is a
/// plain call. The vendor call cannot reach the engine's team, which is the rule `external`
/// keeps while the leader is wide.
///
/// Cost: two `sched_setaffinity` calls per vendor call on a placed leader, nothing elsewhere.
/// The thread count stays the vendor's own setting (`MKL_NUM_THREADS`,
/// `OPENBLAS_NUM_THREADS`, ...); this only decides which CPUs that vendor team may run on.
pub(crate) fn vendor_call<R>(call: impl FnOnce() -> R) -> R {
    let mut call = Some(call);
    let mut result = None;
    // A serial token yields its team of one without touching any pool, so this never refuses; if
    // it ever did, the call has not run yet and runs below, unwidened.
    let _ = tenferro_cpu::threadteam::Exec::serial().team(|mut team| {
        team.external(|| result = call.take().map(|call| call()));
    });
    match (result, call) {
        (Some(result), _) => result,
        (None, Some(call)) => call(),
        // INVARIANT: `call` is taken only by the closure that stores its result.
        (None, None) => unreachable!("a taken vendor call stored its result"),
    }
}

/// The session pool, as the provider's scratch contract.
pub(crate) fn workspace(pool: &mut BufferPool) -> TlinalgWorkspace<'_> {
    TlinalgWorkspace::new(pool)
}

/// Refuse a forced batch strategy this lane cannot honor, before any write.
///
/// `tlinalg-blas` takes no `tcontract_exec::Exec` and has no tenferro lanes:
/// the vendor call owns its own threading. `Auto` and `Sequential` are what the
/// lane already does; every other forced route is a typed unsupported request
/// resolved ahead of every empty, zero or single-item shortcut in the linalg
/// entry points.
pub(crate) fn reject_forced_batch_strategy(exec: &mut Exec<'_>) -> tenferro_tensor::Result<()> {
    let strategy = exec.batch_policy().strategy;
    if matches!(
        strategy,
        tenferro_cpu::CpuBatchStrategy::Auto | tenferro_cpu::CpuBatchStrategy::Sequential
    ) {
        return Ok(());
    }
    Err(tenferro_tensor::Error::unsupported(
        "linalg",
        format!(
            "batch strategy {strategy:?} is unavailable on the BLAS linalg lane; it has no \
             tenferro fan-out and the vendor call owns its threading"
        ),
    ))
}

/// Factor `batch` compact `m x n` matrices in place.
pub(crate) fn factor_batch<T: LapackScalar>(
    _ctx: &mut Exec<'_>,
    op: Op,
    m: usize,
    n: usize,
    lu: &mut [T],
    pivots: &mut [i32],
    parity: &mut [T],
) -> tenferro_tensor::Result<()> {
    vendor_call(|| lu_factor(op, m, n, lu, pivots, parity))
        .map_err(|error| map_blas_error(op, error))
}

/// Solve `op(A) X = B` for `batch` compact systems from packed factors.
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_prepared_batch<T: LapackScalar>(
    _ctx: &mut Exec<'_>,
    op: Op,
    n: usize,
    nrhs: usize,
    packed_lu: &[T],
    pivots: &[i32],
    output: &mut [T],
    transpose_a: bool,
    conjugate_a: bool,
) -> tenferro_tensor::Result<()> {
    vendor_call(|| {
        lu_solve_prepared(
            op,
            n,
            nrhs,
            packed_lu,
            pivots,
            output,
            transpose_a,
            conjugate_a,
        )
    })
    .map_err(|error| map_blas_error(op, error))
}

/// Factor and solve `A X = B` for `batch` compact systems, keeping the packed factors.
pub(crate) fn factor_solve_batch<T: LapackScalar>(
    _ctx: &mut Exec<'_>,
    op: Op,
    n: usize,
    nrhs: usize,
    packed_lu: &mut [T],
    pivots: &mut [i32],
    output: &mut [T],
) -> tenferro_tensor::Result<()> {
    vendor_call(|| lu_factor_solve(op, n, nrhs, packed_lu, pivots, output))
        .map_err(|error| map_blas_error(op, error))
}
