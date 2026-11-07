use crate::cpu::linalg::HostView;
use strided_view::{RawStridedMut, RawStridedRef};
use tenferro_cpu::linalg_interop::BufferPool;
use tenferro_tensor::{TypedTensor, TypedTensorView, TypedTensorViewMut};
use tlinalg_blas::Op;

use super::helpers::{
    has_zero_dim, matrix_core_and_batch_result, pooled_output, provider,
    square_core_and_batch_result, tensor_from_vec_with_template, LapackLinalg,
};

#[cfg(test)]
#[path = "solve/tests.rs"]
mod tests;

pub(crate) fn solve<T: LapackLinalg>(
    buffers: &mut BufferPool,
    a: &HostView<'_, T>,
    b: &HostView<'_, T>,
    transpose_a: bool,
) -> tenferro_tensor::Result<TypedTensor<T>> {
    const OP: &str = "solve";
    let (n, a_batch_shape) = square_core_and_batch_result(a, OP)?;
    let (b_rows, _, b_batch_shape) = matrix_core_and_batch_result(b, OP)?;
    if b_rows != n {
        return Err(tenferro_tensor::Error::shape_mismatch(
            OP,
            vec![n],
            vec![b_rows],
        ));
    }
    if a_batch_shape != b_batch_shape {
        return Err(tenferro_tensor::Error::shape_mismatch(
            OP,
            a_batch_shape.to_vec(),
            b_batch_shape.to_vec(),
        ));
    }
    if has_zero_dim(a.shape()) || has_zero_dim(b.shape()) {
        return tensor_from_vec_with_template(b.shape().to_vec(), Vec::new(), b);
    }
    let mut output = pooled_output::<T>(buffers, OP, "rhs", b.shape())?;
    let (a_view, b_view) = (a.clone().into_dynamic(), b.clone().into_dynamic());
    T::solve(
        buffers,
        transpose_a,
        super::super::raw_view(OP, &a_view)?,
        super::super::raw_view(OP, &b_view)?,
        &mut output,
    )
    .map_err(provider(Op::Solve))?;
    tensor_from_vec_with_template(b.shape().to_vec(), output, b)
}

/// Validate a single borrowed system and return `n`.
fn system_dim<T: 'static>(
    a: &TypedTensorView<'_, T>,
    b: &TypedTensorView<'_, T>,
    op: &'static str,
) -> tenferro_tensor::Result<usize> {
    if a.shape().len() != 2 {
        return Err(tenferro_tensor::Error::rank_mismatch(
            op,
            2,
            a.shape().len(),
        ));
    }
    let n = a.shape()[0];
    if a.shape()[1] != n {
        return Err(tenferro_tensor::Error::shape_mismatch(
            op,
            vec![n],
            vec![a.shape()[1]],
        ));
    }
    let b_rows = match b.shape() {
        [rows] => *rows,
        [rows, _] => *rows,
        other => {
            return Err(tenferro_tensor::Error::rank_mismatch(op, 2, other.len()));
        }
    };
    if b_rows != n {
        return Err(tenferro_tensor::Error::shape_mismatch(
            op,
            vec![n],
            vec![b_rows],
        ));
    }
    Ok(n)
}

/// The `(dims, strides)` a rank-1 or rank-2 right-hand side presents as an `n x nrhs` matrix.
fn rhs_layout(shape: &[usize], strides: &[isize], n: usize) -> ([usize; 2], [isize; 2]) {
    match (shape, strides) {
        ([rows], [row_stride]) => ([*rows, 1], [*row_stride, n as isize]),
        _ => ([shape[0], shape[1]], [strides[0], strides[1]]),
    }
}

fn rhs_descriptor<'v, T: 'static>(
    b: &'v TypedTensorView<'_, T>,
    dims: &'v [usize; 2],
    strides: &'v [isize; 2],
    op: &'static str,
) -> tenferro_tensor::Result<RawStridedRef<'v, T>> {
    if strides.iter().any(|&stride| stride < 0) {
        return Err(tenferro_tensor::Error::unsupported(
            op,
            "a negative stride is not a supported linalg input layout",
        ));
    }
    RawStridedRef::new(b.host_storage()?, dims, strides, b.offset())
        .map_err(|error| tenferro_tensor::Error::invalid_argument(op, "b", error.to_string()))
}

/// Report a provider failure under the route the caller used.
fn route_error(op: &'static str) -> impl Fn(tlinalg_blas::Error) -> tenferro_tensor::Error {
    move |error| match error {
        tlinalg_blas::Error::Singular { .. } => {
            crate::error::into_tensor_error(op, crate::Error::Singular { op })
        }
        other => crate::cpu::tlinalg_error::map_blas_error(Op::Solve, other),
    }
}

/// Solve a single matrix system directly into a positive column-major output view. The
/// destination is written by the provider only after factorization has succeeded, preserving the
/// caller's buffer on validation and singularity failures.
pub(crate) fn solve_into<T: LapackLinalg>(
    buffers: &mut BufferPool,
    a: TypedTensorView<'_, T>,
    b: TypedTensorView<'_, T>,
    out: &mut TypedTensorViewMut<'_, T>,
    transpose_a: bool,
) -> tenferro_tensor::Result<()> {
    const OP: &str = "solve_read_into";
    let n = system_dim(&a, &b, OP)?;
    if out.strides().first().copied() != Some(1) {
        return Err(tenferro_tensor::Error::invalid_argument(
            OP,
            "out",
            "direct LAPACK solve requires unit row stride",
        ));
    }
    let a_copy = super::super::provider_readable(buffers, &a, OP)?;
    let b_copy = super::super::provider_readable(buffers, &b, OP)?;
    let a = a_copy.as_ref().map_or(a, |copy| copy.as_view());
    let b = b_copy.as_ref().map_or(b, |copy| copy.as_view());
    let (b_dims, b_strides) = rhs_layout(b.shape(), b.strides(), n);
    let (out_dims, out_strides) = rhs_layout(out.shape(), out.strides(), n);
    let out_offset = out.offset();
    let a_descriptor = super::super::raw_view(OP, &a)?;
    let b_descriptor = rhs_descriptor(&b, &b_dims, &b_strides, OP)?;
    let out = RawStridedMut::new(out.host_storage_mut()?, &out_dims, &out_strides, out_offset)
        .map_err(|error| tenferro_tensor::Error::invalid_argument(OP, "out", error.to_string()))?;
    T::solve_into(buffers, transpose_a, a_descriptor, b_descriptor, out).map_err(route_error(OP))
}

pub(crate) fn solve_from_views<T: LapackLinalg>(
    buffers: &mut BufferPool,
    a: TypedTensorView<'_, T>,
    b: TypedTensorView<'_, T>,
    transpose_a: bool,
) -> tenferro_tensor::Result<TypedTensor<T>> {
    const OP: &str = "solve";
    let n = system_dim(&a, &b, OP)?;
    let a_copy = super::super::provider_readable(buffers, &a, OP)?;
    let b_copy = super::super::provider_readable(buffers, &b, OP)?;
    let a = a_copy.as_ref().map_or(a, |copy| copy.as_view());
    let b = b_copy.as_ref().map_or(b, |copy| copy.as_view());
    let (b_dims, b_strides) = rhs_layout(b.shape(), b.strides(), n);
    let mut output = pooled_output::<T>(buffers, OP, "rhs", &b_dims)?;
    let a_descriptor = super::super::raw_view(OP, &a)?;
    let b_descriptor = rhs_descriptor(&b, &b_dims, &b_strides, OP)?;
    T::solve(
        buffers,
        transpose_a,
        a_descriptor,
        b_descriptor,
        &mut output,
    )
    .map_err(route_error(OP))?;
    let mut tensor = TypedTensor::from_vec_col_major(b.shape().to_vec(), output)?;
    tensor.set_placement(b.placement().clone());
    Ok(tensor)
}
