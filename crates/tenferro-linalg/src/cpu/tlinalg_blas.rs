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
//! the provider only pooled scratch; the caller-affinity guard has already
//! narrowed that thread's mask.

#![cfg(feature = "blas")]

use tenferro_cpu::linalg_interop::BufferPool;
use tenferro_cpu::CpuExecutionContext;
use tlinalg_blas::lu::{lu_factor, lu_factor_solve, lu_solve_prepared};
use tlinalg_blas::{LapackScalar, Op};

use super::tlinalg_error::map_blas_error;
use super::tlinalg_workspace::TlinalgWorkspace;

/// The session pool, as the provider's scratch contract.
pub(crate) fn workspace(pool: &mut BufferPool) -> TlinalgWorkspace<'_> {
    TlinalgWorkspace::new(pool)
}

/// Factor `batch` compact `m x n` matrices in place.
pub(crate) fn factor_batch<T: LapackScalar>(
    _ctx: &CpuExecutionContext<'_>,
    op: Op,
    m: usize,
    n: usize,
    lu: &mut [T],
    pivots: &mut [i32],
    parity: &mut [T],
) -> tenferro_tensor::Result<()> {
    lu_factor(op, m, n, lu, pivots, parity).map_err(|error| map_blas_error(op, error))
}

/// Solve `op(A) X = B` for `batch` compact systems from packed factors.
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_prepared_batch<T: LapackScalar>(
    _ctx: &CpuExecutionContext<'_>,
    op: Op,
    n: usize,
    nrhs: usize,
    packed_lu: &[T],
    pivots: &[i32],
    output: &mut [T],
    transpose_a: bool,
    conjugate_a: bool,
) -> tenferro_tensor::Result<()> {
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
    .map_err(|error| map_blas_error(op, error))
}

/// Factor and solve `A X = B` for `batch` compact systems, keeping the packed factors.
pub(crate) fn factor_solve_batch<T: LapackScalar>(
    _ctx: &CpuExecutionContext<'_>,
    op: Op,
    n: usize,
    nrhs: usize,
    packed_lu: &mut [T],
    pivots: &mut [i32],
    output: &mut [T],
) -> tenferro_tensor::Result<()> {
    lu_factor_solve(op, n, nrhs, packed_lu, pivots, output)
        .map_err(|error| map_blas_error(op, error))
}
