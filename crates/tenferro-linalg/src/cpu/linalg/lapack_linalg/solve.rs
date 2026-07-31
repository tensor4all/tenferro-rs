use num_complex::{Complex32, Complex64};

use tenferro_cpu::linalg_interop::{BufferPool, PoolScalar};
use tenferro_tensor::{TypedTensor, TypedTensorView, TypedTensorViewMut};

use super::helpers::{
    batched_binary_result, check_lapack_info, dim_i32, has_zero_dim, matrix_core_and_batch_result,
    square_core_and_batch_result, tensor_from_vec_with_template,
};

pub(crate) trait LapackSolve: Clone + Copy + PoolScalar {
    fn getrf(m: i32, n: i32, data: &mut [Self], lda: i32, ipiv: &mut [i32], info: &mut i32);
    fn getrs(args: GetrsArgs<'_, Self>);
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
    if has_zero_dim(a.shape()) || has_zero_dim(b.shape()) {
        let (n, a_batch_shape) = square_core_and_batch_result(a, "solve")?;
        let (b_rows, _, b_batch_shape) = matrix_core_and_batch_result(b, "solve")?;
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
        return tensor_from_vec_with_template(b.shape().to_vec(), Vec::new(), b);
    }

    batched_binary_result("solve", buffers, a, b, |buffers, a, b| {
        solve_2d(buffers, a, b, transpose_a)
    })
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

    let n_i32 = dim_i32(n, op)?;
    let b_cols_i32 = dim_i32(b_cols, op)?;
    let mut ipiv = vec![0_i32; n];
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
