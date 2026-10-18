use strided_view::RawStridedRef;
use tenferro_cpu::linalg_interop::BufferPool;
use tenferro_tensor::TypedTensor;
use tlinalg_blas::Op;

use super::helpers::{
    batch_element_count, checked_product, has_zero_dim, matrix_with_batch_shape, pooled_output,
    provider, square_core_and_batch_result, tensor_from_vec_with_template, LapackLinalg,
};

/// Lower Cholesky factor of one compact `n x n` matrix, for the managed (prepared) route.
pub(crate) fn cholesky_compact_data<T: LapackLinalg>(
    buffers: &mut BufferPool,
    input: &[T],
    n: usize,
) -> tenferro_tensor::Result<TypedTensor<T>> {
    let expected_len = n.checked_mul(n).ok_or_else(|| {
        tenferro_tensor::Error::invalid_argument(
            "cholesky",
            "input storage",
            "matrix element count overflows usize",
        )
    })?;
    if input.len() != expected_len {
        return Err(tenferro_tensor::Error::invalid_argument(
            "cholesky",
            "input storage",
            format!("expected {expected_len} elements, got {}", input.len()),
        ));
    }
    let mut lower = buffers.acquire_with_capacity::<T>(expected_len);
    if n == 0 {
        return TypedTensor::from_vec_col_major([0, 0], lower);
    }
    let dims = [n, n];
    let strides = [1, n as isize];
    let a = RawStridedRef::new(input, &dims, &strides, 0).map_err(|error| {
        tenferro_tensor::Error::invalid_argument("cholesky", "layout", error.to_string())
    })?;
    T::cholesky(buffers, a, &mut lower).map_err(provider(Op::Cholesky))?;
    TypedTensor::from_vec_col_major([n, n], lower)
}

pub(crate) fn cholesky<T: LapackLinalg>(
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> tenferro_tensor::Result<TypedTensor<T>> {
    const OP: &str = "cholesky";
    let (n, batch_shape) = square_core_and_batch_result(input, OP)?;
    let shape = matrix_with_batch_shape(n, n, batch_shape);
    if has_zero_dim(input.shape()) {
        return tensor_from_vec_with_template(shape, Vec::new(), input);
    }
    batch_element_count(OP, batch_shape)?;
    checked_product(OP, "matrix", &shape)?;
    let mut lower = pooled_output::<T>(buffers, OP, "matrix", &shape)?;
    let view = input.as_view();
    T::cholesky(buffers, super::super::raw_view(OP, &view)?, &mut lower)
        .map_err(provider(Op::Cholesky))?;
    tensor_from_vec_with_template(shape, lower, input)
}
