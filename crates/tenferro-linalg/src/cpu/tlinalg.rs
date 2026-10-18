//! Adapter from tenferro's CPU session to the extracted `tlinalg` (faer) provider.
//!
//! Tenferro supplies one [`tcontract_exec::Exec`] resource. The provider owns batch
//! lane selection, scratch, and numerical execution; this adapter only translates
//! the session's retained pool and budget and maps provider errors.
//!
//! Every family is driven through the provider's borrowed-workspace pair: `*_workspace` reports the
//! bytes one call needs at the route the same `Exec` resolves, and `*_into` carves all of its
//! per-lane scratch from the [`PooledWorkspace`] region this module takes from the session pool.
//! Nothing here allocates per call once the pool is warm.

#![cfg(feature = "native")]

use core::mem::MaybeUninit;

use strided_view::RawStridedRef;
use tcontract_exec::Exec;
use tenferro_cpu::linalg_interop::{BufferPool, PoolScalar, PooledUninitOutput};
use tenferro_cpu::CpuExecutionContext;
use tlinalg::packed_lu::{
    factor_into, factor_solve_into, factor_solve_workspace, factor_workspace, solve_prepared_into,
    validate_pivots,
};
use tlinalg::{FaerScalar, Op};

use super::tlinalg_error::map_error;

/// Build the one execution token accepted by `tlinalg`.
///
/// The token borrows the retained execution resources of the caller's
/// owner-scoped context, so the provider reuses the domain's pool wrapper and
/// workspace instead of a per-call `Pool::borrow`.
pub(crate) fn exec_from<'a>(ctx: &CpuExecutionContext<'a>) -> tenferro_tensor::Result<Exec<'a>> {
    ctx.contraction_exec()
}

/// Pooled bytes one provider `*_into` call borrows as its complete scratch region.
///
/// The provider carves each lane's chunk and every faer sub-buffer out of these bytes with its own
/// worst-case alignment padding, so the host only has to hand over `bytes` bytes of pooled storage;
/// nothing here is typed. The region comes from the session's retained pool and returns to it when
/// the call ends, which is what keeps the faer route allocation-free at steady state: the query
/// sizes it, so the pool only has to serve the same capacity again on the next call.
pub(crate) struct PooledWorkspace<T: PoolScalar> {
    /// `None` when the query reported no scratch: a zero-length checkout would still take a pooled
    /// buffer out of its bin and return it, which is exactly the churn the pool exists to avoid.
    lease: Option<PooledUninitOutput<T>>,
}

impl<T: PoolScalar> PooledWorkspace<T> {
    /// Take a pooled region of at least `bytes` bytes from `pool`.
    pub(crate) fn new(pool: &BufferPool, bytes: usize) -> tenferro_tensor::Result<Self> {
        if bytes == 0 {
            return Ok(Self { lease: None });
        }
        // INVARIANT: a pooled scalar's size is nonzero, and the region is byte-addressed above.
        let elements = bytes.div_ceil(core::mem::size_of::<T>());
        Ok(Self {
            lease: Some(PooledUninitOutput::<T>::new(pool, [elements])?),
        })
    }

    /// The region as the provider's byte-addressed scratch.
    pub(crate) fn bytes_mut(&mut self) -> &mut [MaybeUninit<u8>] {
        match self.lease.as_mut() {
            Some(lease) => lease.as_uninit_bytes_mut(),
            None => &mut [],
        }
    }
}

/// Factor `batch` compact `m x n` matrices in place.
// INVARIANT: the argument list mirrors the lower provider's factor contract.
#[allow(clippy::too_many_arguments)]
pub(crate) fn factor_batch<T: FaerScalar + PoolScalar>(
    ctx: &CpuExecutionContext<'_>,
    buffers: &BufferPool,
    op: Op,
    m: usize,
    n: usize,
    lu: &mut [T],
    pivots: &mut [i32],
    parity: &mut [T],
) -> tenferro_tensor::Result<()> {
    let exec = exec_from(ctx)?;
    let mut workspace = PooledWorkspace::<T>::new(
        buffers,
        factor_workspace::<T>(m, n, parity.len(), exec).map_err(|error| map_error(op, error))?,
    )?;
    factor_into(op, m, n, lu, pivots, parity, workspace.bytes_mut(), exec)
        .map_err(|error| map_error(op, error))
}

/// Solve `op(A) X = B` for compact systems from packed factors.
// INVARIANT: the argument list mirrors the lower provider's prepared-solve contract.
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_prepared_batch<T: FaerScalar + PoolScalar>(
    ctx: &CpuExecutionContext<'_>,
    buffers: &BufferPool,
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
    // The prepared solve is the two triangular solves alone: it has no scratch to size, so the
    // contract's natural workspace is the empty region.
    let mut workspace = PooledWorkspace::<T>::new(buffers, 0)?;
    solve_prepared_into(
        op,
        packed_lu,
        pivots,
        nrhs,
        output,
        transpose_a,
        conjugate_a,
        workspace.bytes_mut(),
        exec_from(ctx)?,
    )
    .map_err(|error| map_error(op, error))
}

/// Factor and solve compact systems, keeping the packed factors.
// INVARIANT: the argument list mirrors the lower provider's fused contract.
#[allow(clippy::too_many_arguments)]
pub(crate) fn factor_solve_batch<T: FaerScalar + PoolScalar>(
    ctx: &CpuExecutionContext<'_>,
    buffers: &BufferPool,
    op: Op,
    n: usize,
    nrhs: usize,
    packed_lu: &mut [T],
    pivots: &mut [i32],
    output: &mut [T],
) -> tenferro_tensor::Result<()> {
    let exec = exec_from(ctx)?;
    // The provider derives the batch from the packed factors; the query takes it explicitly.
    let batch = packed_lu.len() / n.saturating_mul(n).max(1);
    let mut workspace = PooledWorkspace::<T>::new(
        buffers,
        factor_solve_workspace::<T>(n, nrhs, batch, exec).map_err(|error| map_error(op, error))?,
    )?;
    factor_solve_into(
        op,
        n,
        nrhs,
        packed_lu,
        pivots,
        output,
        workspace.bytes_mut(),
        exec,
    )
    .map_err(|error| map_error(op, error))
}

/// Column-major strides for a compact `[rows, cols, batch]` stack.
fn compact_strides3(rows: usize, cols: usize) -> [isize; 3] {
    // INVARIANT: callers pass compact buffers allocated from checked shape products.
    [1, rows as isize, (rows * cols) as isize]
}

fn layout_error(op: Op, error: impl std::fmt::Display) -> tenferro_tensor::Error {
    tenferro_tensor::Error::invalid_argument(op.as_str(), "layout", error.to_string())
}
