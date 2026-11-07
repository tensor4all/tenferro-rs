use tenferro_cpu::linalg_interop::BufferPool;
use tenferro_tensor::TypedTensor;
use tlinalg_blas::Op;

use super::helpers::{
    has_zero_dim, matrix_with_batch_shape, pooled_output, provider, square_core_and_batch_result,
    tensor_from_vec_with_template, vector_with_batch_shape, LapackLinalg,
};

/// Eigenvalues and eigenvectors of every Hermitian matrix of a batch.
///
/// The values are returned in the real scalar, as the faer route returns them.
pub(crate) fn eigh<T: LapackLinalg>(
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> tenferro_tensor::Result<(TypedTensor<T::RealScalar>, TypedTensor<T>)> {
    const OP: &str = "eigh";
    let (n, batch_shape) = square_core_and_batch_result(input, OP)?;
    let values_shape = vector_with_batch_shape(n, batch_shape);
    if has_zero_dim(input.shape()) {
        return Ok((
            tensor_from_vec_with_template(values_shape, Vec::new(), input)?,
            tensor_from_vec_with_template(
                matrix_with_batch_shape(n, n, batch_shape),
                Vec::new(),
                input,
            )?,
        ));
    }
    let mut values = pooled_output::<T::RealScalar>(buffers, OP, "values", &values_shape)?;
    let mut vectors = pooled_output::<T>(buffers, OP, "vectors", input.shape())?;
    let view = input.as_view();
    T::eigh(
        buffers,
        Op::Eigh,
        super::super::raw_view(OP, &view)?,
        &mut values,
        Some(&mut vectors),
    )
    .map_err(provider(Op::Eigh))?;
    Ok((
        tensor_from_vec_with_template(values_shape, values, input)?,
        tensor_from_vec_with_template(input.shape().to_vec(), vectors, input)?,
    ))
}

pub(crate) fn eigh_values<T: LapackLinalg>(
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> tenferro_tensor::Result<TypedTensor<T::RealScalar>> {
    const OP: &str = "eigh_values";
    let (n, batch_shape) = square_core_and_batch_result(input, OP)?;
    let values_shape = vector_with_batch_shape(n, batch_shape);
    if has_zero_dim(input.shape()) {
        return tensor_from_vec_with_template(values_shape, Vec::new(), input);
    }
    let mut values = pooled_output::<T::RealScalar>(buffers, OP, "values", &values_shape)?;
    let view = input.as_view();
    T::eigh(
        buffers,
        Op::EighValues,
        super::super::raw_view(OP, &view)?,
        &mut values,
        None,
    )
    .map_err(provider(Op::EighValues))?;
    tensor_from_vec_with_template(values_shape, values, input)
}
