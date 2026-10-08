//! Adapter from tenferro's CPU session to the extracted `tlinalg` (faer) provider.
//!
//! Tenferro supplies one [`tlinalg::Parallel`] resource. The provider owns batch
//! lane selection, scratch, and numerical execution; this adapter only translates
//! the session's retained pool and budget and maps provider errors.

#![cfg(feature = "native")]

use strided_view::RawStridedRef;
use tenferro_cpu::CpuExecutionContext;
use tlinalg::packed_lu::{factor, factor_solve, solve_prepared, validate_pivots};
use tlinalg::{FaerScalar, Op, Parallel};

use super::tlinalg_error::map_error;

/// Build the one parallelism token accepted by `tlinalg`.
pub(crate) fn parallel_from<'a>(ctx: &CpuExecutionContext<'a>) -> Parallel<'a> {
    match ctx.rayon_pool() {
        Some(pool) => Parallel::Pool {
            pool,
            budget: ctx.thread_budget(),
        },
        None => Parallel::Sequential,
    }
}

/// Factor `batch` compact `m x n` matrices in place.
pub(crate) fn factor_batch<T: FaerScalar>(
    ctx: &CpuExecutionContext<'_>,
    op: Op,
    m: usize,
    n: usize,
    lu: &mut [T],
    pivots: &mut [i32],
    parity: &mut [T],
) -> tenferro_tensor::Result<()> {
    factor(op, m, n, lu, pivots, parity, parallel_from(ctx)).map_err(|error| map_error(op, error))
}

/// Solve `op(A) X = B` for compact systems from packed factors.
// INVARIANT: the argument list mirrors the lower provider's prepared-solve contract.
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_prepared_batch<T: FaerScalar>(
    ctx: &CpuExecutionContext<'_>,
    op: Op,
    n: usize,
    nrhs: usize,
    packed_lu: &[T],
    pivots: &[i32],
    output: &mut [T],
    transpose_a: bool,
    conjugate_a: bool,
) -> tenferro_tensor::Result<()> {
    let batch = pivots.len().checked_div(n).unwrap_or(0);
    if n > 0 && nrhs > 0 {
        validate_pivots(Op::LuSolvePrepared, n, pivots)
            .map_err(|error| map_error(Op::LuSolvePrepared, error))?;
    }
    if n == 0 || nrhs == 0 {
        return Ok(());
    }
    let lu_dims = [n, n, batch];
    let lu_strides = compact_strides3(n, n);
    let pivot_dims = [n, batch];
    let pivot_strides = [1, n as isize];
    let packed_lu = RawStridedRef::new(packed_lu, &lu_dims, &lu_strides, 0)
        .map_err(|error| layout_error(op, error))?;
    let pivots = RawStridedRef::new(pivots, &pivot_dims, &pivot_strides, 0)
        .map_err(|error| layout_error(op, error))?;
    solve_prepared(
        op,
        packed_lu,
        pivots,
        nrhs,
        output,
        transpose_a,
        conjugate_a,
        parallel_from(ctx),
    )
    .map_err(|error| map_error(op, error))
}

/// Factor and solve compact systems, keeping the packed factors.
// INVARIANT: the argument list mirrors the lower provider's fused contract.
#[allow(clippy::too_many_arguments)]
pub(crate) fn factor_solve_batch<T: FaerScalar>(
    ctx: &CpuExecutionContext<'_>,
    op: Op,
    n: usize,
    nrhs: usize,
    packed_lu: &mut [T],
    pivots: &mut [i32],
    output: &mut [T],
) -> tenferro_tensor::Result<()> {
    let par = parallel_from(ctx);
    factor_solve(op, n, nrhs, packed_lu, pivots, output, par).map_err(|error| map_error(op, error))
}

/// Column-major strides for a compact `[rows, cols, batch]` stack.
fn compact_strides3(rows: usize, cols: usize) -> [isize; 3] {
    // INVARIANT: callers pass compact buffers allocated from checked shape products.
    [1, rows as isize, (rows * cols) as isize]
}

fn layout_error(op: Op, error: impl std::fmt::Display) -> tenferro_tensor::Error {
    tenferro_tensor::Error::invalid_argument(op.as_str(), "layout", error.to_string())
}
