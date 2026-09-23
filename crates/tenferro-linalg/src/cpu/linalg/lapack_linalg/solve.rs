use num_complex::{Complex32, Complex64};

use tenferro_cpu::linalg_interop::{BufferPool, PoolScalar};
use tenferro_tensor::{TypedTensor, TypedTensorView, TypedTensorViewMut};

use super::helpers::{
    check_lapack_info, checked_product, dim_i32, has_zero_dim, matrix_core_and_batch_result,
    pooled_zeroed, square_core_and_batch_result, tensor_from_vec_with_template,
};

#[cfg(test)]
mod tests;

pub(crate) trait LapackSolve: Clone + Copy + PoolScalar {
    fn getrf(m: i32, n: i32, data: &mut [Self], lda: i32, ipiv: &mut [i32], info: &mut i32);
    fn getrs(args: GetrsArgs<'_, Self>);
    /// Conjugate every element in place; a no-op for real scalars.
    fn conj_in_place(_data: &mut [Self]) {}
}

pub(crate) struct GetrsArgs<'a, T> {
    trans: u8,
    n: i32,
    nrhs: i32,
    a: &'a [T],
    lda: i32,
    ipiv: &'a [i32],
    b: &'a mut [T],
    ldb: i32,
    info: &'a mut i32,
}

impl LapackSolve for f64 {
    fn getrf(m: i32, n: i32, data: &mut [Self], lda: i32, ipiv: &mut [i32], info: &mut i32) {
        // SAFETY: callers validate dimensions and provide a mutable
        // column-major `lda x n` matrix, pivot storage, and live `info`.
        unsafe {
            lapack::dgetrf(m, n, data, lda, ipiv, info);
        }
    }

    fn getrs(args: GetrsArgs<'_, Self>) {
        let GetrsArgs {
            trans,
            n,
            nrhs,
            a,
            lda,
            ipiv,
            b,
            ldb,
            info,
        } = args;
        // SAFETY: `a` holds a prior getrf factorization, `ipiv` matches it,
        // `b` is a mutable `ldb x nrhs` RHS buffer, and all dims are validated.
        unsafe {
            lapack::dgetrs(trans, n, nrhs, a, lda, ipiv, b, ldb, info);
        }
    }
}

impl LapackSolve for f32 {
    fn getrf(m: i32, n: i32, data: &mut [Self], lda: i32, ipiv: &mut [i32], info: &mut i32) {
        // SAFETY: callers validate dimensions and provide a mutable
        // column-major `lda x n` matrix, pivot storage, and live `info`.
        unsafe {
            lapack::sgetrf(m, n, data, lda, ipiv, info);
        }
    }

    fn getrs(args: GetrsArgs<'_, Self>) {
        let GetrsArgs {
            trans,
            n,
            nrhs,
            a,
            lda,
            ipiv,
            b,
            ldb,
            info,
        } = args;
        // SAFETY: `a` holds a prior getrf factorization, `ipiv` matches it,
        // `b` is a mutable `ldb x nrhs` RHS buffer, and all dims are validated.
        unsafe {
            lapack::sgetrs(trans, n, nrhs, a, lda, ipiv, b, ldb, info);
        }
    }
}

impl LapackSolve for Complex32 {
    fn conj_in_place(data: &mut [Self]) {
        for value in data {
            *value = value.conj();
        }
    }

    fn getrf(m: i32, n: i32, data: &mut [Self], lda: i32, ipiv: &mut [i32], info: &mut i32) {
        // SAFETY: callers validate dimensions and provide a mutable
        // column-major `lda x n` matrix, pivot storage, and live `info`.
        unsafe {
            lapack::cgetrf(m, n, data, lda, ipiv, info);
        }
    }

    fn getrs(args: GetrsArgs<'_, Self>) {
        let GetrsArgs {
            trans,
            n,
            nrhs,
            a,
            lda,
            ipiv,
            b,
            ldb,
            info,
        } = args;
        // SAFETY: `a` holds a prior getrf factorization, `ipiv` matches it,
        // `b` is a mutable `ldb x nrhs` RHS buffer, and all dims are validated.
        unsafe {
            lapack::cgetrs(trans, n, nrhs, a, lda, ipiv, b, ldb, info);
        }
    }
}

impl LapackSolve for Complex64 {
    fn conj_in_place(data: &mut [Self]) {
        for value in data {
            *value = value.conj();
        }
    }

    fn getrf(m: i32, n: i32, data: &mut [Self], lda: i32, ipiv: &mut [i32], info: &mut i32) {
        // SAFETY: callers validate dimensions and provide a mutable
        // column-major `lda x n` matrix, pivot storage, and live `info`.
        unsafe {
            lapack::zgetrf(m, n, data, lda, ipiv, info);
        }
    }

    fn getrs(args: GetrsArgs<'_, Self>) {
        let GetrsArgs {
            trans,
            n,
            nrhs,
            a,
            lda,
            ipiv,
            b,
            ldb,
            info,
        } = args;
        // SAFETY: `a` holds a prior getrf factorization, `ipiv` matches it,
        // `b` is a mutable `ldb x nrhs` RHS buffer, and all dims are validated.
        unsafe {
            lapack::zgetrs(trans, n, nrhs, a, lda, ipiv, b, ldb, info);
        }
    }
}

fn solve_2d<T: LapackSolve + 'static>(
    buffers: &mut BufferPool,
    a: &TypedTensor<T>,
    b: &TypedTensor<T>,
    transpose_a: bool,
) -> tenferro_tensor::Result<TypedTensor<T>> {
    solve_from_views(buffers, a.as_view(), b.as_view(), transpose_a)
}

pub(crate) fn solve<T: LapackSolve + 'static>(
    buffers: &mut BufferPool,
    a: &TypedTensor<T>,
    b: &TypedTensor<T>,
    transpose_a: bool,
) -> tenferro_tensor::Result<TypedTensor<T>> {
    let (n, a_batch_shape) = square_core_and_batch_result(a, "solve")?;
    let (b_rows, b_cols, b_batch_shape) = matrix_core_and_batch_result(b, "solve")?;
    if b_rows != n {
        return Err(tenferro_tensor::Error::shape_mismatch(
            "solve",
            vec![n],
            vec![b_rows],
        ));
    }
    if a_batch_shape != b_batch_shape {
        return Err(tenferro_tensor::Error::shape_mismatch(
            "solve",
            a_batch_shape.to_vec(),
            b_batch_shape.to_vec(),
        ));
    }
    if has_zero_dim(a.shape()) || has_zero_dim(b.shape()) {
        return tensor_from_vec_with_template(b.shape().to_vec(), Vec::new(), b);
    }
    if a_batch_shape.is_empty() {
        return solve_2d(buffers, a, b, transpose_a);
    }

    let matrix_len = checked_product("solve", "matrix", &[n, n])?;
    let rhs_len = checked_product("solve", "rhs", &[n, b_cols])?;
    let n_i32 = dim_i32(n, "solve")?;
    let nrhs = dim_i32(b_cols, "solve")?;
    let mut lu = buffers.acquire_with_capacity::<T>(matrix_len);
    let mut ipiv = pooled_zeroed::<i32>(buffers, n);
    let mut output = buffers.acquire_with_capacity::<T>(b.n_elements());
    output.extend_from_slice(b.host_data()?);
    // INVARIANT: owned tensors are compact column-major; validated matching batch
    // shapes give equally many nonempty matrix/RHS chunks. LAPACK overwrites only
    // private LU scratch and the final output, preserving both caller inputs.
    // LAPACK owns provider threading; reuse its serial batch loop's scratch rather
    // than adding an independent Rayon pool around provider calls.
    for (matrix, rhs) in a
        .host_data()?
        .chunks_exact(matrix_len)
        .zip(output.chunks_exact_mut(rhs_len))
    {
        lu.clear();
        lu.extend_from_slice(matrix);
        let mut info = 0;
        T::getrf(n_i32, n_i32, &mut lu, n_i32, &mut ipiv, &mut info);
        check_lapack_info("solve", "getrf", info.min(0))?;
        if info > 0 {
            return Err(crate::error::into_tensor_error(
                "solve",
                crate::Error::Singular { op: "solve" },
            ));
        }
        T::getrs(GetrsArgs {
            trans: if transpose_a { b'T' } else { b'N' },
            n: n_i32,
            nrhs,
            a: &lu,
            lda: n_i32,
            ipiv: &ipiv,
            b: rhs,
            ldb: n_i32,
            info: &mut info,
        });
        check_lapack_info("solve", "getrs", info)?;
    }
    tensor_from_vec_with_template(b.shape().to_vec(), output, b)
}

/// Reject pivots that would make LAPACK `?getrs` index outside the matrix.
///
/// `?getrs` applies `ipiv` through `?laswp` without bounds checks, so every
/// stored pivot must be a one-based row index in `1..=n` before the call.
fn validate_lapack_pivots(op: &'static str, n: usize, ipiv: &[i32]) -> tenferro_tensor::Result<()> {
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

/// Solve `op(A) X = B` for every batch from packed `getrf` factors.
///
/// `output` enters holding the compact column-major RHS batch and leaves
/// holding the solution. `op(A)` is `A`, `A^T`, `A^H`, or `conj(A)` from the
/// `(transpose_a, conjugate_a)` flags; the first three map directly to
/// `?getrs` with `trans = N/T/C`, and `conj(A)` uses
/// `conj(A) x = b  <=>  A conj(x) = conj(b)`.
///
/// # Errors
///
/// Returns `Error::InvalidArgument` for a pivot outside `1..=n`, a dimension
/// outside the LAPACK `i32` range, inconsistent buffer lengths, or an illegal
/// LAPACK argument.
pub(crate) fn lu_solve_prepared_batched_in_place<T: LapackSolve>(
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
    if packed_lu.len() != checked_product(op, "packed LU", &[matrix_len, batch_total])?
        || pivots.len() != checked_product(op, "pivots", &[n, batch_total])?
        || output.len() != checked_product(op, "rhs batch", &[rhs_len, batch_total])?
    {
        return Err(tenferro_tensor::Error::Internal(format!(
            "{op}: packed LU, pivot, and rhs buffers describe different batches"
        )));
    }
    validate_lapack_pivots(op, n, pivots)?;
    let n_i32 = dim_i32(n, op)?;
    let nrhs_i32 = dim_i32(nrhs, op)?;
    let (trans, conjugate_rhs) = match (transpose_a, conjugate_a) {
        (false, false) => (b'N', false),
        (true, false) => (b'T', false),
        (true, true) => (b'C', false),
        (false, true) => (b'N', true),
    };
    if conjugate_rhs {
        T::conj_in_place(output);
    }
    // INVARIANT: the buffer lengths were checked above to hold exactly
    // `batch_total` nonempty matrices, pivot vectors, and RHS blocks, and every
    // pivot is in `1..=n`, so each `?getrs` call reads only its own factors and
    // writes only its own RHS block. The serial loop is intentional: the
    // LAPACK provider owns threading inside `?getrs`.
    for ((matrix, ipiv), rhs) in packed_lu
        .chunks_exact(matrix_len)
        .zip(pivots.chunks_exact(n))
        .zip(output.chunks_exact_mut(rhs_len))
    {
        let mut info = 0;
        T::getrs(GetrsArgs {
            trans,
            n: n_i32,
            nrhs: nrhs_i32,
            a: matrix,
            lda: n_i32,
            ipiv,
            b: rhs,
            ldb: n_i32,
            info: &mut info,
        });
        check_lapack_info(op, "getrs", info)?;
    }
    if conjugate_rhs {
        T::conj_in_place(output);
    }
    Ok(())
}

/// Factor and solve `A X = B` for every batch, keeping the packed factors.
///
/// `packed_lu` enters holding the compact column-major `A` batch and leaves
/// holding the packed `getrf` factors; `pivots` receives the one-based
/// pivots; `output` enters holding the RHS batch and leaves holding `X`.
/// This is the fused primal of `lu_factor` followed by `lu_solve_prepared`:
/// one `?getrf` and one `?getrs` per matrix, with no scratch at all because
/// the factors are themselves an output.
///
/// # Errors
///
/// Returns `Error::Extension` with `crate::Error::Singular` when a matrix is
/// exactly singular and there is a nonempty RHS to solve, and `Error::InvalidArgument` for inconsistent buffer
/// lengths, out-of-range dimensions, or an illegal LAPACK argument.
pub(crate) fn lu_factor_solve_batched_in_place<T: LapackSolve>(
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
    if packed_lu.len() != checked_product(op, "packed LU", &[matrix_len, batch_total])?
        || pivots.len() != checked_product(op, "pivots", &[n, batch_total])?
        || output.len() != checked_product(op, "rhs batch", &[rhs_len, batch_total])?
    {
        return Err(tenferro_tensor::Error::Internal(format!(
            "{op}: packed LU, pivot, and rhs buffers describe different batches"
        )));
    }
    let n_i32 = dim_i32(n, op)?;
    let nrhs_i32 = dim_i32(nrhs, op)?;
    // INVARIANT: the buffer lengths were checked above to hold exactly
    // `batch_total` nonempty matrices and pivot vectors plus as many RHS
    // blocks. A zero-column RHS has `rhs_len == 0`, so it is handled by a
    // separate factor-only loop instead of `chunks_exact_mut(0)`. The serial
    // loop is intentional: the LAPACK provider owns threading inside
    // `?getrf`/`?getrs`.
    let factor_one = |matrix: &mut [T],
                      ipiv: &mut [i32],
                      reject_singular: bool|
     -> tenferro_tensor::Result<()> {
        let mut info = 0;
        T::getrf(n_i32, n_i32, matrix, n_i32, ipiv, &mut info);
        check_lapack_info(op, "getrf", info.min(0))?;
        if reject_singular && info > 0 {
            return Err(crate::error::into_tensor_error(
                op,
                crate::Error::Singular { op },
            ));
        }
        Ok(())
    };
    if rhs_len == 0 {
        // Nothing to solve: only factor, matching `lu_factor` on singular input.
        for (matrix, ipiv) in packed_lu
            .chunks_exact_mut(matrix_len)
            .zip(pivots.chunks_exact_mut(n))
        {
            factor_one(matrix, ipiv, false)?;
        }
        return Ok(());
    }
    for ((matrix, ipiv), rhs) in packed_lu
        .chunks_exact_mut(matrix_len)
        .zip(pivots.chunks_exact_mut(n))
        .zip(output.chunks_exact_mut(rhs_len))
    {
        factor_one(matrix, ipiv, true)?;
        let mut info = 0;
        T::getrs(GetrsArgs {
            trans: b'N',
            n: n_i32,
            nrhs: nrhs_i32,
            a: matrix,
            lda: n_i32,
            ipiv,
            b: rhs,
            ldb: n_i32,
            info: &mut info,
        });
        check_lapack_info(op, "getrs", info)?;
    }
    Ok(())
}

/// Solve a single matrix system directly into a positive column-major output
/// view. The destination is copied from the RHS only after factorization has
/// succeeded, preserving the caller's buffer on validation and singularity
/// failures.
pub(crate) fn solve_into<T: LapackSolve + 'static>(
    buffers: &mut BufferPool,
    a: TypedTensorView<'_, T>,
    b: TypedTensorView<'_, T>,
    out: &mut TypedTensorViewMut<'_, T>,
    transpose_a: bool,
) -> tenferro_tensor::Result<()> {
    solve_in_place(buffers, a, b, out, transpose_a, true, "solve_read_into")
}

pub(crate) fn solve_from_views<T: LapackSolve + 'static>(
    buffers: &mut BufferPool,
    a: TypedTensorView<'_, T>,
    b: TypedTensorView<'_, T>,
    transpose_a: bool,
) -> tenferro_tensor::Result<TypedTensor<T>> {
    let mut output = super::super::output_from_rhs_view(buffers, &b, "solve")?;
    let mut out = output.as_view_mut();
    solve_in_place(buffers, a, b, &mut out, transpose_a, false, "solve")?;
    Ok(output)
}

fn solve_in_place<T: LapackSolve + 'static>(
    buffers: &mut BufferPool,
    a: TypedTensorView<'_, T>,
    b: TypedTensorView<'_, T>,
    out: &mut TypedTensorViewMut<'_, T>,
    transpose_a: bool,
    copy_rhs: bool,
    op: &'static str,
) -> tenferro_tensor::Result<()> {
    let n = square_matrix_dim_view(&a, op)?;
    let (b_rows, b_cols) = rhs_matrix_dims_view(&b, op)?;
    if b_rows != n {
        return Err(tenferro_tensor::Error::shape_mismatch(
            op,
            vec![n],
            vec![b_rows],
        ));
    }
    if out.strides().first().copied() != Some(1) {
        return Err(tenferro_tensor::Error::invalid_argument(
            op,
            "out",
            "direct LAPACK solve requires unit row stride",
        ));
    }

    let lu_len = n.checked_mul(n).ok_or_else(|| {
        tenferro_tensor::Error::invalid_argument(op, "a", "matrix size overflows usize")
    })?;
    let mut lu = buffers.acquire_with_capacity::<T>(lu_len);
    if a.is_col_major_contiguous()? {
        lu.extend_from_slice(a.as_slice()?);
    } else {
        for col in 0..n {
            for row in 0..n {
                let value = a.get(&[row, col]).ok_or_else(|| {
                    tenferro_tensor::Error::runtime_state(
                        op,
                        "CPU LAPACK solve input view is not host-addressable",
                    )
                })?;
                lu.push(*value);
            }
        }
    }

    let n_i32 = dim_i32(n, op)?;
    let b_cols_i32 = dim_i32(b_cols, op)?;
    let mut ipiv = pooled_zeroed::<i32>(buffers, n);
    let mut info = 0;
    T::getrf(n_i32, n_i32, &mut lu, n_i32, &mut ipiv, &mut info);
    check_lapack_info(op, "getrf", info.min(0))?;
    if info > 0 {
        return Err(crate::error::into_tensor_error(
            op,
            crate::Error::Singular { op },
        ));
    }

    let ldb = if out.shape().len() == 1 {
        n
    } else {
        out.strides()[1].try_into().map_err(|_| {
            tenferro_tensor::Error::invalid_argument(
                op,
                "out",
                "output leading dimension does not fit LAPACK",
            )
        })?
    };
    let ldb_i32 = dim_i32(ldb, op)?;
    if copy_rhs {
        copy_rhs_view_into(&b, out, n, b_cols, op)?;
    }
    let rhs = output_slice_mut(out, n, b_cols, ldb, op)?;
    let mut info = 0;
    T::getrs(GetrsArgs {
        trans: if transpose_a { b'T' } else { b'N' },
        n: n_i32,
        nrhs: b_cols_i32,
        a: &lu,
        lda: n_i32,
        ipiv: &ipiv,
        b: rhs,
        ldb: ldb_i32,
        info: &mut info,
    });
    check_lapack_info(op, "getrs", info)
}

fn square_matrix_dim_view<T: 'static>(
    view: &TypedTensorView<'_, T>,
    op: &'static str,
) -> tenferro_tensor::Result<usize> {
    let (rows, cols) = matrix_dims_view(view, op)?;
    if rows != cols {
        return Err(tenferro_tensor::Error::shape_mismatch(
            op,
            vec![rows],
            vec![cols],
        ));
    }
    Ok(rows)
}

fn rhs_matrix_dims_view<T: 'static>(
    view: &TypedTensorView<'_, T>,
    op: &'static str,
) -> tenferro_tensor::Result<(usize, usize)> {
    match view.shape() {
        [rows] => Ok((*rows, 1)),
        _ => matrix_dims_view(view, op),
    }
}

fn matrix_dims_view<T: 'static>(
    view: &TypedTensorView<'_, T>,
    op: &'static str,
) -> tenferro_tensor::Result<(usize, usize)> {
    if view.shape().len() != 2 {
        return Err(tenferro_tensor::Error::rank_mismatch(
            op,
            2,
            view.shape().len(),
        ));
    }
    Ok((view.shape()[0], view.shape()[1]))
}

fn copy_rhs_view_into<T: Copy + 'static>(
    src: &TypedTensorView<'_, T>,
    dst: &mut TypedTensorViewMut<'_, T>,
    rows: usize,
    cols: usize,
    op: &'static str,
) -> tenferro_tensor::Result<()> {
    if dst.shape().len() == 1 {
        for row in 0..rows {
            let value = src.get(&[row]).ok_or_else(|| {
                tenferro_tensor::Error::runtime_state(op, "RHS view is not host-addressable")
            })?;
            let target = dst.get_mut(&[row]).ok_or_else(|| {
                tenferro_tensor::Error::runtime_state(op, "output view is not host-addressable")
            })?;
            *target = *value;
        }
    } else {
        for col in 0..cols {
            for row in 0..rows {
                let value = src.get(&[row, col]).ok_or_else(|| {
                    tenferro_tensor::Error::runtime_state(op, "RHS view is not host-addressable")
                })?;
                let target = dst.get_mut(&[row, col]).ok_or_else(|| {
                    tenferro_tensor::Error::runtime_state(op, "output view is not host-addressable")
                })?;
                *target = *value;
            }
        }
    }
    Ok(())
}

fn output_slice_mut<'out, 'view, T: 'static>(
    out: &'out mut TypedTensorViewMut<'view, T>,
    rows: usize,
    cols: usize,
    ldb: usize,
    op: &'static str,
) -> tenferro_tensor::Result<&'out mut [T]> {
    let offset = out.offset();
    if offset < 0 {
        return Err(tenferro_tensor::Error::runtime_state(
            op,
            "output view offset is negative",
        ));
    }
    let span = cols
        .checked_sub(1)
        .and_then(|last_col| last_col.checked_mul(ldb))
        .and_then(|last_offset| last_offset.checked_add(rows))
        .ok_or_else(|| {
            tenferro_tensor::Error::invalid_argument(op, "out", "output view span overflows usize")
        })?;
    let offset = usize::try_from(offset)
        .map_err(|_| tenferro_tensor::Error::runtime_state(op, "output view offset is negative"))?;
    let end = offset.checked_add(span).ok_or_else(|| {
        tenferro_tensor::Error::invalid_argument(
            op,
            "out",
            "output view end offset overflows usize",
        )
    })?;
    let storage = out.host_storage_mut()?;
    storage.get_mut(offset..end).ok_or_else(|| {
        tenferro_tensor::Error::runtime_state(
            op,
            "output view does not contain the requested LAPACK span",
        )
    })
}
