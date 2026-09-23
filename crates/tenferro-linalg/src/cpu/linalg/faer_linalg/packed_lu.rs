//! Batched packed partial-pivot LU kernels for the faer provider.
//!
//! These kernels back `lu_factor`, `lu_solve_prepared`, and the fused
//! `lu_factor_solve`. Each call sizes its scratch once, factors or solves every
//! matrix of the batch in place inside the caller's output buffers, and keeps
//! the LAPACK packed format (unit-lower `L` below the diagonal, `U` on and above
//! it, one-based row-swap pivots) so the factors are interchangeable with the
//! LAPACK provider.

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::prelude::ReborrowMut;
use faer::{Conj, MatMut, MatRef};
use num_complex::{Complex32, Complex64};

use tenferro_cpu::linalg_interop::PoolScalar;
use tenferro_cpu::CpuExecutionContext;

use super::{checked_product, invalid_config, singular_matrix};

/// A tensor scalar with a layout-identical faer entity.
pub(crate) trait FaerPackedLu: Copy + Default + PartialEq + PoolScalar {
    /// The faer scalar sharing this type's memory layout.
    type Entity: faer::traits::ComplexField + Copy + PartialEq + Default;

    /// Reinterpret a slice as the faer entity type.
    fn entity_slice(data: &[Self]) -> &[Self::Entity];
    /// Reinterpret a mutable slice as the faer entity type.
    fn entity_slice_mut(data: &mut [Self]) -> &mut [Self::Entity];
    /// The permutation parity scalar, `+1` or `-1`.
    fn parity(odd: bool) -> Self;
}

macro_rules! impl_real_packed_lu {
    ($scalar:ty) => {
        impl FaerPackedLu for $scalar {
            type Entity = $scalar;

            fn entity_slice(data: &[Self]) -> &[Self::Entity] {
                data
            }

            fn entity_slice_mut(data: &mut [Self]) -> &mut [Self::Entity] {
                data
            }

            fn parity(odd: bool) -> Self {
                if odd {
                    -1.0
                } else {
                    1.0
                }
            }
        }
    };
}

macro_rules! impl_complex_packed_lu {
    ($scalar:ty, $entity:ty, $to_slice:path, $to_slice_mut:path) => {
        impl FaerPackedLu for $scalar {
            type Entity = $entity;

            fn entity_slice(data: &[Self]) -> &[Self::Entity] {
                $to_slice(data)
            }

            fn entity_slice_mut(data: &mut [Self]) -> &mut [Self::Entity] {
                $to_slice_mut(data)
            }

            fn parity(odd: bool) -> Self {
                <$scalar>::new(if odd { -1.0 } else { 1.0 }, 0.0)
            }
        }
    };
}

impl_real_packed_lu!(f32);
impl_real_packed_lu!(f64);
impl_complex_packed_lu!(
    Complex32,
    faer::c32,
    super::complex32_to_faer_slice,
    super::complex32_to_faer_slice_mut
);
impl_complex_packed_lu!(
    Complex64,
    faer::c64,
    super::complex64_to_faer_slice,
    super::complex64_to_faer_slice_mut
);

/// Reusable per-call state for factoring a batch of `m x n` matrices.
struct PackedLuFactorScratch {
    m: usize,
    k: usize,
    perm: Vec<usize>,
    perm_inv: Vec<usize>,
    current: Vec<usize>,
    position: Vec<usize>,
    mem: MemBuffer,
}

impl PackedLuFactorScratch {
    fn new<E: faer::traits::ComplexField>(
        ctx: &CpuExecutionContext<'_>,
        m: usize,
        n: usize,
    ) -> Self {
        Self {
            m,
            k: m.min(n),
            perm: vec![0; m],
            perm_inv: vec![0; m],
            current: vec![0; m],
            position: vec![0; m],
            mem: MemBuffer::new(
                faer::linalg::lu::partial_pivoting::factor::lu_in_place_scratch::<usize, E>(
                    m,
                    n,
                    ctx.faer_parallelism(),
                    Default::default(),
                ),
            ),
        }
    }

    /// Factor one compact column-major matrix in place and write its
    /// one-based swap sequence into `ipiv`. Returns whether the permutation is
    /// odd.
    fn factor<E: faer::traits::ComplexField>(
        &mut self,
        ctx: &CpuExecutionContext<'_>,
        matrix: MatMut<'_, E>,
        ipiv: &mut [i32],
        op: &'static str,
    ) -> tenferro_tensor::Result<bool> {
        let stack = MemStack::new(&mut self.mem);
        let info = faer::linalg::lu::partial_pivoting::factor::lu_in_place(
            matrix,
            &mut self.perm,
            &mut self.perm_inv,
            ctx.faer_parallelism(),
            stack,
            Default::default(),
        )
        .0;

        // faer returns `perm` with `(P A)[i, :] = A[perm[i], :]`. Replay it as
        // the LAPACK swap sequence: at step `i`, swap row `i` with the current
        // position of row `perm[i]`. `current`/`position` track the running
        // permutation and its inverse, so each step is O(1).
        for (idx, (slot, pos)) in self
            .current
            .iter_mut()
            .zip(self.position.iter_mut())
            .enumerate()
        {
            *slot = idx;
            *pos = idx;
        }
        // INVARIANT: `perm` is a permutation of `0..m` (faer contract) and
        // `ipiv.len() == k <= m`, so every index below is in bounds.
        for (step, slot) in ipiv.iter_mut().enumerate().take(self.k) {
            let wanted = self.perm[step];
            let pivot = self.position[wanted];
            if pivot >= self.m {
                return Err(invalid_config(op, "invalid row permutation"));
            }
            let displaced = self.current[step];
            self.current.swap(step, pivot);
            self.position[wanted] = step;
            self.position[displaced] = pivot;
            *slot = i32::try_from(pivot + 1)
                .map_err(|_| invalid_config(op, "pivot index exceeds i32 range"))?;
        }
        Ok(info.transposition_count % 2 == 1)
    }
}

/// Check that every `(per_matrix_len, buffer_len)` pair describes the same
/// `batch_total` matrices, in the order packed LU, pivots, batch buffer.
fn check_batches(
    op: &'static str,
    buffers: [(usize, usize); 3],
    batch_total: usize,
) -> tenferro_tensor::Result<()> {
    for ((per_matrix, len), what) in buffers
        .into_iter()
        .zip(["packed LU", "pivots", "rhs batch"])
    {
        if len != checked_product(op, what, &[per_matrix, batch_total])? {
            return Err(tenferro_tensor::Error::Internal(format!(
                "{op}: packed LU, pivot, and batch buffers describe different batches"
            )));
        }
    }
    Ok(())
}

/// Factor every compact column-major `m x n` matrix of `lu_data` in place.
///
/// `pivot_data` receives `min(m, n)` one-based pivots per matrix and
/// `parity_data` one permutation parity per matrix. Exactly singular matrices
/// are not an error, matching LAPACK `?getrf` with positive `info`.
///
/// # Errors
///
/// Returns `Error::Internal` when the buffers describe different batches and
/// `Error::InvalidArgument` when faer returns an invalid row permutation.
pub(crate) fn lu_factor_batched_in_place<T: FaerPackedLu>(
    ctx: &CpuExecutionContext<'_>,
    op: &'static str,
    m: usize,
    n: usize,
    lu_data: &mut [T],
    pivot_data: &mut [i32],
    parity_data: &mut [T],
) -> tenferro_tensor::Result<()> {
    let k = m.min(n);
    let matrix_len = checked_product(op, "matrix shape", &[m, n])?;
    let batch_total = parity_data.len();
    check_batches(
        op,
        [
            (matrix_len, lu_data.len()),
            (k, pivot_data.len()),
            (1, parity_data.len()),
        ],
        batch_total,
    )?;
    if matrix_len == 0 || batch_total == 0 {
        return Ok(());
    }
    let mut scratch = PackedLuFactorScratch::new::<T::Entity>(ctx, m, n);
    // INVARIANT: lengths were checked above and `matrix_len > 0` implies
    // `k > 0`, so the three chunk iterators yield exactly `batch_total`
    // aligned items. faer owns any threading inside `lu_in_place`, so the
    // batch loop stays serial and reuses one scratch set.
    for ((matrix, ipiv), parity) in lu_data
        .chunks_exact_mut(matrix_len)
        .zip(pivot_data.chunks_exact_mut(k))
        .zip(parity_data.iter_mut())
    {
        let mat = MatMut::from_column_major_slice_mut(T::entity_slice_mut(matrix), m, n);
        let odd = scratch.factor(ctx, mat, ipiv, op)?;
        *parity = T::parity(odd);
    }
    Ok(())
}

/// Apply a one-based LAPACK swap sequence to the rows of a compact
/// column-major `n x nrhs` block, forward (`P b`) or in reverse (`P^T b`).
fn apply_row_swaps<T: Copy>(rhs: &mut [T], n: usize, nrhs: usize, ipiv: &[i32], reverse: bool) {
    let swap = |rhs: &mut [T], step: usize, pivot_one_based: i32| {
        // INVARIANT: callers validated every pivot in `1..=n` first.
        let pivot = pivot_one_based as usize - 1;
        if pivot != step {
            for col in 0..nrhs {
                rhs.swap(step + col * n, pivot + col * n);
            }
        }
    };
    if reverse {
        for (step, &pivot) in ipiv.iter().enumerate().rev() {
            swap(rhs, step, pivot);
        }
    } else {
        for (step, &pivot) in ipiv.iter().enumerate() {
            swap(rhs, step, pivot);
        }
    }
}

fn validate_pivots(op: &'static str, n: usize, ipiv: &[i32]) -> tenferro_tensor::Result<()> {
    for &pivot_one_based in ipiv {
        let in_range = usize::try_from(pivot_one_based)
            .map(|pivot| (1..=n).contains(&pivot))
            .unwrap_or(false);
        if !in_range {
            return Err(tenferro_tensor::Error::invalid_argument(
                op,
                "pivot",
                format!("LU pivot index {pivot_one_based} is outside 1..={n}"),
            ));
        }
    }
    Ok(())
}

/// Solve `op(A) x = b` for one matrix from packed factors, in place.
///
/// With `P A = L U`: `A x = b` is `x = U^-1 L^-1 P b`, and `A^T x = b` is
/// `x = P^T L^-T U^-T b`. Conjugation conjugates `L` and `U` implicitly.
// INVARIANT: the flags mirror the `LuSolvePrepared` op attributes one-to-one.
fn solve_one<T: FaerPackedLu>(
    ctx: &CpuExecutionContext<'_>,
    (n, nrhs): (usize, usize),
    matrix: &[T],
    ipiv: &[i32],
    rhs: &mut [T],
    (transpose_a, conjugate_a): (bool, bool),
) {
    let par = ctx.faer_parallelism();
    let conj = if conjugate_a { Conj::Yes } else { Conj::No };
    let lu = MatRef::from_column_major_slice(T::entity_slice(matrix), n, n);
    if transpose_a {
        {
            let mut x = MatMut::from_column_major_slice_mut(T::entity_slice_mut(rhs), n, nrhs);
            let lu_t = lu.transpose();
            faer::linalg::triangular_solve::solve_lower_triangular_in_place_with_conj(
                lu_t,
                conj,
                x.rb_mut(),
                par,
            );
            faer::linalg::triangular_solve::solve_unit_upper_triangular_in_place_with_conj(
                lu_t, conj, x, par,
            );
        }
        apply_row_swaps(rhs, n, nrhs, ipiv, true);
    } else {
        apply_row_swaps(rhs, n, nrhs, ipiv, false);
        let mut x = MatMut::from_column_major_slice_mut(T::entity_slice_mut(rhs), n, nrhs);
        faer::linalg::triangular_solve::solve_unit_lower_triangular_in_place_with_conj(
            lu,
            conj,
            x.rb_mut(),
            par,
        );
        faer::linalg::triangular_solve::solve_upper_triangular_in_place_with_conj(lu, conj, x, par);
    }
}

/// Solve `op(A) X = B` for every batch from packed partial-pivot factors.
///
/// `output` enters holding the compact column-major RHS batch and leaves
/// holding the solution. The factors must be nonsingular; callers check the
/// `U` diagonal first.
///
/// # Errors
///
/// Returns `Error::InvalidArgument` for a pivot outside `1..=n` and
/// `Error::Internal` when the buffers describe different batches.
// INVARIANT: the flags mirror the `LuSolvePrepared` op attributes one-to-one.
pub(crate) fn lu_solve_prepared_batched_in_place<T: FaerPackedLu>(
    ctx: &CpuExecutionContext<'_>,
    op: &'static str,
    (n, nrhs): (usize, usize),
    packed_lu: &[T],
    pivots: &[i32],
    output: &mut [T],
    (transpose_a, conjugate_a): (bool, bool),
) -> tenferro_tensor::Result<()> {
    let matrix_len = checked_product(op, "matrix", &[n, n])?;
    let rhs_len = checked_product(op, "rhs", &[n, nrhs])?;
    if matrix_len == 0 || rhs_len == 0 {
        return Ok(());
    }
    let batch_total = packed_lu.len() / matrix_len;
    check_batches(
        op,
        [
            (matrix_len, packed_lu.len()),
            (n, pivots.len()),
            (rhs_len, output.len()),
        ],
        batch_total,
    )?;
    validate_pivots(op, n, pivots)?;
    // INVARIANT: lengths were checked above, so the chunk iterators yield
    // exactly `batch_total` aligned nonempty items, and all pivots are in
    // range for `apply_row_swaps`.
    for ((matrix, ipiv), rhs) in packed_lu
        .chunks_exact(matrix_len)
        .zip(pivots.chunks_exact(n))
        .zip(output.chunks_exact_mut(rhs_len))
    {
        solve_one(
            ctx,
            (n, nrhs),
            matrix,
            ipiv,
            rhs,
            (transpose_a, conjugate_a),
        );
    }
    Ok(())
}

/// Factor and solve `A X = B` for every batch, keeping the packed factors.
///
/// `packed_lu` enters holding the compact `A` batch and leaves holding the
/// packed factors; `pivots` receives one-based pivots; `output` enters holding
/// the RHS batch and leaves holding `X`.
///
/// # Errors
///
/// Returns `Error::Extension` with `crate::Error::Singular` when a factor has
/// an exactly zero `U` diagonal and there is a nonempty RHS to solve, and
/// `Error::Internal` when the buffers describe different batches. A
/// zero-column RHS only factors, matching `lu_factor` on singular input.
pub(crate) fn lu_factor_solve_batched_in_place<T: FaerPackedLu>(
    ctx: &CpuExecutionContext<'_>,
    op: &'static str,
    n: usize,
    nrhs: usize,
    packed_lu: &mut [T],
    pivots: &mut [i32],
    output: &mut [T],
) -> tenferro_tensor::Result<()> {
    let matrix_len = checked_product(op, "matrix", &[n, n])?;
    let rhs_len = checked_product(op, "rhs", &[n, nrhs])?;
    if matrix_len == 0 {
        return Ok(());
    }
    let batch_total = packed_lu.len() / matrix_len;
    check_batches(
        op,
        [
            (matrix_len, packed_lu.len()),
            (n, pivots.len()),
            (rhs_len, output.len()),
        ],
        batch_total,
    )?;
    let mut scratch = PackedLuFactorScratch::new::<T::Entity>(ctx, n, n);
    let zero = T::default();
    // INVARIANT: lengths were checked above; a zero-column RHS yields empty
    // RHS blocks, which `chunks_mut` below cannot express with a zero chunk
    // size, so the RHS block is sliced by offset instead. faer owns any
    // threading inside the factorization and triangular solves.
    for (batch, (matrix, ipiv)) in packed_lu
        .chunks_exact_mut(matrix_len)
        .zip(pivots.chunks_exact_mut(n))
        .enumerate()
    {
        {
            let mat = MatMut::from_column_major_slice_mut(T::entity_slice_mut(matrix), n, n);
            scratch.factor(ctx, mat, ipiv, op)?;
        }
        if rhs_len > 0 {
            if (0..n).any(|i| matrix[i + i * n] == zero) {
                return Err(singular_matrix(op));
            }
            let start = batch * rhs_len;
            let rhs = &mut output[start..start + rhs_len];
            solve_one(ctx, (n, nrhs), matrix, ipiv, rhs, (false, false));
        }
    }
    Ok(())
}
