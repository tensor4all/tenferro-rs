use num_complex::{Complex32, Complex64};

use tenferro_cpu::linalg_interop::{BufferPool, PoolScalar};
use tenferro_tensor::TypedTensor;

use super::helpers::{
    batched_single, check_lapack_info, dim_i32, has_zero_dim, lower_triangle_from_lapack,
    matrix_with_batch_shape, square_core_and_batch_result, square_matrix_dim,
    tensor_from_vec_with_template,
};

pub(crate) trait LapackCholesky: Clone + Copy + Default + PoolScalar {
    fn potrf(uplo: u8, n: i32, factor: &mut [Self], lda: i32, info: &mut i32);
}

impl LapackCholesky for f64 {
    fn potrf(uplo: u8, n: i32, factor: &mut [Self], lda: i32, info: &mut i32) {
        // SAFETY: callers pass a mutable column-major `lda x n` factor buffer,
        // validated i32 dimensions, and a live `info` output for this call.
        unsafe {
            lapack::dpotrf(uplo, n, factor, lda, info);
        }
    }
}

impl LapackCholesky for f32 {
    fn potrf(uplo: u8, n: i32, factor: &mut [Self], lda: i32, info: &mut i32) {
        // SAFETY: callers pass a mutable column-major `lda x n` factor buffer,
        // validated i32 dimensions, and a live `info` output for this call.
        unsafe {
            lapack::spotrf(uplo, n, factor, lda, info);
        }
    }
}

impl LapackCholesky for Complex32 {
    fn potrf(uplo: u8, n: i32, factor: &mut [Self], lda: i32, info: &mut i32) {
        // SAFETY: callers pass a mutable column-major `lda x n` factor buffer,
        // validated i32 dimensions, and a live `info` output for this call.
        unsafe {
            lapack::cpotrf(uplo, n, factor, lda, info);
        }
    }
}

impl LapackCholesky for Complex64 {
    fn potrf(uplo: u8, n: i32, factor: &mut [Self], lda: i32, info: &mut i32) {
        // SAFETY: callers pass a mutable column-major `lda x n` factor buffer,
        // validated i32 dimensions, and a live `info` output for this call.
        unsafe {
            lapack::zpotrf(uplo, n, factor, lda, info);
        }
    }
}

fn cholesky_2d<T: LapackCholesky>(
    _buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> tenferro_tensor::Result<TypedTensor<T>> {
    let n = square_matrix_dim(input, "cholesky")?;
    let n_i32 = dim_i32(n, "cholesky")?;
    let mut factor = input.host_data()?.to_vec();
    let mut info = 0;
    T::potrf(b'L', n_i32, &mut factor, n_i32, &mut info);
    if info > 0 {
        return Err(crate::error::into_tensor_error(
            "cholesky",
            crate::Error::NonConvergence { op: "cholesky" },
        ));
    }
    check_lapack_info("cholesky", "dpotrf", info)?;
    tensor_from_vec_with_template(
        vec![n, n],
        lower_triangle_from_lapack(&factor, n, n)?,
        input,
    )
}

pub(crate) fn cholesky<T: LapackCholesky>(
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> tenferro_tensor::Result<TypedTensor<T>> {
    if has_zero_dim(input.shape()) {
        let (n, batch_shape) = square_core_and_batch_result(input, "cholesky")?;
        return tensor_from_vec_with_template(
            matrix_with_batch_shape(n, n, batch_shape),
            Vec::new(),
            input,
        );
    }
    batched_single("cholesky", buffers, input, cholesky_2d)
}
