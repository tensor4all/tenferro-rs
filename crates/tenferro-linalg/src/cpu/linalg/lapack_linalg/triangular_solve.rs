use crate::cpu::linalg::HostView;
use tenferro_cpu::linalg_interop::BufferPool;
use tenferro_tensor::TypedTensor;
use tlinalg_blas::triangular_solve::TriangularSolveOptions;
use tlinalg_blas::Op;

use super::helpers::{
    has_zero_dim, matrix_core_and_batch_result, pooled_output, provider,
    square_core_and_batch_result, tensor_from_vec_with_template, LapackLinalg,
};

pub(crate) fn triangular_solve<T: LapackLinalg>(
    buffers: &mut BufferPool,
    a: &HostView<'_, T>,
    b: &HostView<'_, T>,
    left_side: bool,
    lower: bool,
    transpose_a: bool,
    unit_diagonal: bool,
) -> tenferro_tensor::Result<TypedTensor<T>> {
    const OP: &str = "triangular_solve";
    let (n, a_batch_shape) = square_core_and_batch_result(a, OP)?;
    let (b_rows, b_cols, b_batch_shape) = matrix_core_and_batch_result(b, OP)?;
    let rhs_core_dim = if left_side { b_rows } else { b_cols };
    if rhs_core_dim != n {
        return Err(tenferro_tensor::Error::shape_mismatch(
            OP,
            vec![n],
            vec![rhs_core_dim],
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
    T::triangular_solve(
        buffers,
        TriangularSolveOptions {
            left_side,
            lower,
            transpose_a,
            unit_diagonal,
        },
        super::super::raw_view(OP, &a_view)?,
        super::super::raw_view(OP, &b_view)?,
        &mut output,
    )
    .map_err(provider(Op::TriangularSolve))?;
    tensor_from_vec_with_template(b.shape().to_vec(), output, b)
}
