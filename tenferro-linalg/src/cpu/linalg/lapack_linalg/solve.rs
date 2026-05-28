use num_complex::{Complex32, Complex64};

use tenferro_tensor::buffer_pool::BufferPool;
use tenferro_tensor::TypedTensor;

use super::helpers::{
    batched_binary_result, check_lapack_info, dim_i32, has_zero_dim, matrix_core_and_batch_result,
    matrix_dims, square_core_and_batch_result, square_matrix_dim, tensor_from_vec_with_template,
};

pub(crate) trait LapackSolve: Clone + Copy {
    fn getrf(m: i32, n: i32, data: &mut [Self], lda: i32, ipiv: &mut [i32], info: &mut i32);
    fn getrs(
        trans: u8,
        n: i32,
        nrhs: i32,
        a: &[Self],
        lda: i32,
        ipiv: &[i32],
        b: &mut [Self],
        ldb: i32,
        info: &mut i32,
    );
}

impl LapackSolve for f64 {
    fn getrf(m: i32, n: i32, data: &mut [Self], lda: i32, ipiv: &mut [i32], info: &mut i32) {
        unsafe {
            lapack::dgetrf(m, n, data, lda, ipiv, info);
        }
    }

    fn getrs(
        trans: u8,
        n: i32,
        nrhs: i32,
        a: &[Self],
        lda: i32,
        ipiv: &[i32],
        b: &mut [Self],
        ldb: i32,
        info: &mut i32,
    ) {
        unsafe {
            lapack::dgetrs(trans, n, nrhs, a, lda, ipiv, b, ldb, info);
        }
    }
}

impl LapackSolve for f32 {
    fn getrf(m: i32, n: i32, data: &mut [Self], lda: i32, ipiv: &mut [i32], info: &mut i32) {
        unsafe {
            lapack::sgetrf(m, n, data, lda, ipiv, info);
        }
    }

    fn getrs(
        trans: u8,
        n: i32,
        nrhs: i32,
        a: &[Self],
        lda: i32,
        ipiv: &[i32],
        b: &mut [Self],
        ldb: i32,
        info: &mut i32,
    ) {
        unsafe {
            lapack::sgetrs(trans, n, nrhs, a, lda, ipiv, b, ldb, info);
        }
    }
}

impl LapackSolve for Complex32 {
    fn getrf(m: i32, n: i32, data: &mut [Self], lda: i32, ipiv: &mut [i32], info: &mut i32) {
        unsafe {
            lapack::cgetrf(m, n, data, lda, ipiv, info);
        }
    }

    fn getrs(
        trans: u8,
        n: i32,
        nrhs: i32,
        a: &[Self],
        lda: i32,
        ipiv: &[i32],
        b: &mut [Self],
        ldb: i32,
        info: &mut i32,
    ) {
        unsafe {
            lapack::cgetrs(trans, n, nrhs, a, lda, ipiv, b, ldb, info);
        }
    }
}

impl LapackSolve for Complex64 {
    fn getrf(m: i32, n: i32, data: &mut [Self], lda: i32, ipiv: &mut [i32], info: &mut i32) {
        unsafe {
            lapack::zgetrf(m, n, data, lda, ipiv, info);
        }
    }

    fn getrs(
        trans: u8,
        n: i32,
        nrhs: i32,
        a: &[Self],
        lda: i32,
        ipiv: &[i32],
        b: &mut [Self],
        ldb: i32,
        info: &mut i32,
    ) {
        unsafe {
            lapack::zgetrs(trans, n, nrhs, a, lda, ipiv, b, ldb, info);
        }
    }
}

fn solve_2d<T: LapackSolve>(
    _buffers: &mut BufferPool,
    a: &TypedTensor<T>,
    b: &TypedTensor<T>,
    transpose_a: bool,
) -> tenferro_tensor::Result<TypedTensor<T>> {
    let n = square_matrix_dim(a, "solve")?;
    let (b_rows, b_cols) = matrix_dims(b, "solve")?;
    if b_rows != n {
        return Err(tenferro_tensor::Error::ShapeMismatch {
            op: "solve",
            lhs: vec![n],
            rhs: vec![b_rows],
        });
    }

    let n_i32 = dim_i32(n, "solve")?;
    let b_cols_i32 = dim_i32(b_cols, "solve")?;
    let mut lu = a.host_data().to_vec();
    let mut ipiv = vec![0_i32; n];
    let mut info = 0;
    T::getrf(n_i32, n_i32, &mut lu, n_i32, &mut ipiv, &mut info);
    check_lapack_info("solve", "getrf", info.min(0))?;
    if info > 0 {
        return Err(tenferro_tensor::Error::backend_failure(
            "solve",
            "matrix is singular",
        ));
    }

    let mut rhs = b.host_data().to_vec();
    let mut info = 0;
    T::getrs(
        if transpose_a { b'T' } else { b'N' },
        n_i32,
        b_cols_i32,
        &lu,
        n_i32,
        &ipiv,
        &mut rhs,
        n_i32,
        &mut info,
    );
    check_lapack_info("solve", "getrs", info)?;

    Ok(tensor_from_vec_with_template(vec![n, b_cols], rhs, b))
}

pub(crate) fn solve<T: LapackSolve>(
    buffers: &mut BufferPool,
    a: &TypedTensor<T>,
    b: &TypedTensor<T>,
    transpose_a: bool,
) -> tenferro_tensor::Result<TypedTensor<T>> {
    if has_zero_dim(a.shape()) || has_zero_dim(b.shape()) {
        let (n, a_batch_shape) = square_core_and_batch_result(a, "solve")?;
        let (b_rows, _, b_batch_shape) = matrix_core_and_batch_result(b, "solve")?;
        if b_rows != n {
            return Err(tenferro_tensor::Error::ShapeMismatch {
                op: "solve",
                lhs: vec![n],
                rhs: vec![b_rows],
            });
        }
        if a_batch_shape != b_batch_shape {
            return Err(tenferro_tensor::Error::ShapeMismatch {
                op: "solve",
                lhs: a_batch_shape.to_vec(),
                rhs: b_batch_shape.to_vec(),
            });
        }
        return Ok(tensor_from_vec_with_template(
            b.shape().to_vec(),
            Vec::new(),
            b,
        ));
    }

    batched_binary_result("solve", buffers, a, b, |buffers, a, b| {
        solve_2d(buffers, a, b, transpose_a)
    })
}
