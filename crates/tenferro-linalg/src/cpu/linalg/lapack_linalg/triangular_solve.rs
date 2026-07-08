use cblas_sys::{CBLAS_DIAG, CBLAS_LAYOUT, CBLAS_SIDE, CBLAS_TRANSPOSE, CBLAS_UPLO};
use num_complex::{Complex32, Complex64};
use num_traits::Zero;

use tenferro_cpu::linalg_interop::{BufferPool, PoolScalar};
use tenferro_tensor::TypedTensor;

use super::helpers::{
    batched_binary_result, check_lapack_info, dim_i32, has_zero_dim, matrix_core_and_batch_result,
    matrix_dims, square_core_and_batch_result, square_matrix_dim, tensor_from_vec_with_template,
};

pub(crate) trait LapackTriangularSolve:
    Clone + Copy + PartialEq + PoolScalar + Zero
{
    fn trtrs(
        uplo: u8,
        trans: u8,
        diag: u8,
        n: i32,
        nrhs: i32,
        a: &[Self],
        lda: i32,
        b: &mut [Self],
        ldb: i32,
        info: &mut i32,
    );

    fn trsm(
        side: CBLAS_SIDE,
        uplo: CBLAS_UPLO,
        transa: CBLAS_TRANSPOSE,
        diag: CBLAS_DIAG,
        m: i32,
        n: i32,
        a: &[Self],
        lda: i32,
        b: &mut [Self],
        ldb: i32,
    );
}

impl LapackTriangularSolve for f64 {
    fn trtrs(
        uplo: u8,
        trans: u8,
        diag: u8,
        n: i32,
        nrhs: i32,
        a: &[Self],
        lda: i32,
        b: &mut [Self],
        ldb: i32,
        info: &mut i32,
    ) {
        // SAFETY: callers validate the triangular matrix and RHS shapes,
        // provide column-major `a`/`b` buffers matching `lda`/`ldb`, and live `info`.
        unsafe {
            lapack::dtrtrs(uplo, trans, diag, n, nrhs, a, lda, b, ldb, info);
        }
    }

    fn trsm(
        side: CBLAS_SIDE,
        uplo: CBLAS_UPLO,
        transa: CBLAS_TRANSPOSE,
        diag: CBLAS_DIAG,
        m: i32,
        n: i32,
        a: &[Self],
        lda: i32,
        b: &mut [Self],
        ldb: i32,
    ) {
        // SAFETY: callers validate dimensions and provide compact column-major
        // `a` and writable `b` buffers with matching leading dimensions.
        unsafe {
            cblas_sys::cblas_dtrsm(
                CBLAS_LAYOUT::CblasColMajor,
                side,
                uplo,
                transa,
                diag,
                m,
                n,
                1.0,
                a.as_ptr(),
                lda,
                b.as_mut_ptr(),
                ldb,
            );
        }
    }
}

impl LapackTriangularSolve for f32 {
    fn trtrs(
        uplo: u8,
        trans: u8,
        diag: u8,
        n: i32,
        nrhs: i32,
        a: &[Self],
        lda: i32,
        b: &mut [Self],
        ldb: i32,
        info: &mut i32,
    ) {
        // SAFETY: callers validate the triangular matrix and RHS shapes,
        // provide column-major `a`/`b` buffers matching `lda`/`ldb`, and live `info`.
        unsafe {
            lapack::strtrs(uplo, trans, diag, n, nrhs, a, lda, b, ldb, info);
        }
    }

    fn trsm(
        side: CBLAS_SIDE,
        uplo: CBLAS_UPLO,
        transa: CBLAS_TRANSPOSE,
        diag: CBLAS_DIAG,
        m: i32,
        n: i32,
        a: &[Self],
        lda: i32,
        b: &mut [Self],
        ldb: i32,
    ) {
        // SAFETY: callers validate dimensions and provide compact column-major
        // `a` and writable `b` buffers with matching leading dimensions.
        unsafe {
            cblas_sys::cblas_strsm(
                CBLAS_LAYOUT::CblasColMajor,
                side,
                uplo,
                transa,
                diag,
                m,
                n,
                1.0,
                a.as_ptr(),
                lda,
                b.as_mut_ptr(),
                ldb,
            );
        }
    }
}

impl LapackTriangularSolve for Complex32 {
    fn trtrs(
        uplo: u8,
        trans: u8,
        diag: u8,
        n: i32,
        nrhs: i32,
        a: &[Self],
        lda: i32,
        b: &mut [Self],
        ldb: i32,
        info: &mut i32,
    ) {
        // SAFETY: callers validate the triangular matrix and RHS shapes,
        // provide column-major `a`/`b` buffers matching `lda`/`ldb`, and live `info`.
        unsafe {
            lapack::ctrtrs(uplo, trans, diag, n, nrhs, a, lda, b, ldb, info);
        }
    }

    fn trsm(
        side: CBLAS_SIDE,
        uplo: CBLAS_UPLO,
        transa: CBLAS_TRANSPOSE,
        diag: CBLAS_DIAG,
        m: i32,
        n: i32,
        a: &[Self],
        lda: i32,
        b: &mut [Self],
        ldb: i32,
    ) {
        let alpha = Complex32::new(1.0, 0.0);
        // SAFETY: callers validate dimensions and provide compact column-major
        // `a` and writable `b` buffers with matching leading dimensions.
        unsafe {
            cblas_sys::cblas_ctrsm(
                CBLAS_LAYOUT::CblasColMajor,
                side,
                uplo,
                transa,
                diag,
                m,
                n,
                (&alpha as *const Complex32).cast(),
                a.as_ptr().cast(),
                lda,
                b.as_mut_ptr().cast(),
                ldb,
            );
        }
    }
}

impl LapackTriangularSolve for Complex64 {
    fn trtrs(
        uplo: u8,
        trans: u8,
        diag: u8,
        n: i32,
        nrhs: i32,
        a: &[Self],
        lda: i32,
        b: &mut [Self],
        ldb: i32,
        info: &mut i32,
    ) {
        // SAFETY: callers validate the triangular matrix and RHS shapes,
        // provide column-major `a`/`b` buffers matching `lda`/`ldb`, and live `info`.
        unsafe {
            lapack::ztrtrs(uplo, trans, diag, n, nrhs, a, lda, b, ldb, info);
        }
    }

    fn trsm(
        side: CBLAS_SIDE,
        uplo: CBLAS_UPLO,
        transa: CBLAS_TRANSPOSE,
        diag: CBLAS_DIAG,
        m: i32,
        n: i32,
        a: &[Self],
        lda: i32,
        b: &mut [Self],
        ldb: i32,
    ) {
        let alpha = Complex64::new(1.0, 0.0);
        // SAFETY: callers validate dimensions and provide compact column-major
        // `a` and writable `b` buffers with matching leading dimensions.
        unsafe {
            cblas_sys::cblas_ztrsm(
                CBLAS_LAYOUT::CblasColMajor,
                side,
                uplo,
                transa,
                diag,
                m,
                n,
                (&alpha as *const Complex64).cast(),
                a.as_ptr().cast(),
                lda,
                b.as_mut_ptr().cast(),
                ldb,
            );
        }
    }
}

fn cblas_uplo(lower: bool) -> CBLAS_UPLO {
    if lower {
        CBLAS_UPLO::CblasLower
    } else {
        CBLAS_UPLO::CblasUpper
    }
}

fn cblas_transpose(transpose: bool) -> CBLAS_TRANSPOSE {
    if transpose {
        CBLAS_TRANSPOSE::CblasTrans
    } else {
        CBLAS_TRANSPOSE::CblasNoTrans
    }
}

fn cblas_diag(unit_diagonal: bool) -> CBLAS_DIAG {
    if unit_diagonal {
        CBLAS_DIAG::CblasUnit
    } else {
        CBLAS_DIAG::CblasNonUnit
    }
}

fn validate_non_unit_diagonal<T: LapackTriangularSolve>(
    a: &TypedTensor<T>,
    n: usize,
    unit_diagonal: bool,
) -> tenferro_tensor::Result<()> {
    if unit_diagonal {
        return Ok(());
    }

    let data = a.host_data()?;
    for idx in 0..n {
        if data[idx + idx * n] == T::zero() {
            return Err(tenferro_tensor::Error::backend_failure(
                "triangular_solve",
                "matrix is singular",
            ));
        }
    }
    Ok(())
}

fn solve_left<T: LapackTriangularSolve>(
    a: &TypedTensor<T>,
    b: &TypedTensor<T>,
    lower: bool,
    transpose_a: bool,
    unit_diagonal: bool,
) -> tenferro_tensor::Result<TypedTensor<T>> {
    let n = square_matrix_dim(a, "triangular_solve")?;
    let (b_rows, b_cols) = matrix_dims(b, "triangular_solve")?;
    if b_rows != n {
        return Err(tenferro_tensor::Error::ShapeMismatch {
            op: "triangular_solve",
            lhs: vec![n],
            rhs: vec![b_rows],
        });
    }

    let mut rhs = b.host_data()?.to_vec();
    let mut info = 0;
    T::trtrs(
        if lower { b'L' } else { b'U' },
        if transpose_a { b'T' } else { b'N' },
        if unit_diagonal { b'U' } else { b'N' },
        dim_i32(n, "triangular_solve")?,
        dim_i32(b_cols, "triangular_solve")?,
        a.host_data()?,
        dim_i32(n, "triangular_solve")?,
        &mut rhs,
        dim_i32(n, "triangular_solve")?,
        &mut info,
    );
    check_lapack_info("triangular_solve", "trtrs", info)?;
    tensor_from_vec_with_template(vec![n, b_cols], rhs, b)
}

fn solve_right<T: LapackTriangularSolve>(
    a: &TypedTensor<T>,
    b: &TypedTensor<T>,
    lower: bool,
    transpose_a: bool,
    unit_diagonal: bool,
) -> tenferro_tensor::Result<TypedTensor<T>> {
    let n = square_matrix_dim(a, "triangular_solve")?;
    let (b_rows, b_cols) = matrix_dims(b, "triangular_solve")?;
    if b_cols != n {
        return Err(tenferro_tensor::Error::ShapeMismatch {
            op: "triangular_solve",
            lhs: vec![n],
            rhs: vec![b_cols],
        });
    }

    validate_non_unit_diagonal(a, n, unit_diagonal)?;
    let mut rhs = b.host_data()?.to_vec();
    T::trsm(
        CBLAS_SIDE::CblasRight,
        cblas_uplo(lower),
        cblas_transpose(transpose_a),
        cblas_diag(unit_diagonal),
        dim_i32(b_rows, "triangular_solve")?,
        dim_i32(n, "triangular_solve")?,
        a.host_data()?,
        dim_i32(n, "triangular_solve")?,
        &mut rhs,
        dim_i32(b_rows, "triangular_solve")?,
    );
    tensor_from_vec_with_template(vec![b_rows, n], rhs, b)
}

fn triangular_solve_2d<T: LapackTriangularSolve>(
    _buffers: &mut BufferPool,
    a: &TypedTensor<T>,
    b: &TypedTensor<T>,
    left_side: bool,
    lower: bool,
    transpose_a: bool,
    unit_diagonal: bool,
) -> tenferro_tensor::Result<TypedTensor<T>> {
    if left_side {
        solve_left(a, b, lower, transpose_a, unit_diagonal)
    } else {
        solve_right(a, b, lower, transpose_a, unit_diagonal)
    }
}

pub(crate) fn triangular_solve<T: LapackTriangularSolve>(
    buffers: &mut BufferPool,
    a: &TypedTensor<T>,
    b: &TypedTensor<T>,
    left_side: bool,
    lower: bool,
    transpose_a: bool,
    unit_diagonal: bool,
) -> tenferro_tensor::Result<TypedTensor<T>> {
    if has_zero_dim(a.shape()) || has_zero_dim(b.shape()) {
        let (n, a_batch_shape) = square_core_and_batch_result(a, "triangular_solve")?;
        let (b_rows, b_cols, b_batch_shape) = matrix_core_and_batch_result(b, "triangular_solve")?;
        let rhs_core_dim = if left_side { b_rows } else { b_cols };
        if rhs_core_dim != n {
            return Err(tenferro_tensor::Error::ShapeMismatch {
                op: "triangular_solve",
                lhs: vec![n],
                rhs: vec![rhs_core_dim],
            });
        }
        if a_batch_shape != b_batch_shape {
            return Err(tenferro_tensor::Error::ShapeMismatch {
                op: "triangular_solve",
                lhs: a_batch_shape.to_vec(),
                rhs: b_batch_shape.to_vec(),
            });
        }
        return tensor_from_vec_with_template(b.shape().to_vec(), Vec::new(), b);
    }
    batched_binary_result("triangular_solve", buffers, a, b, |buffers, a, b| {
        triangular_solve_2d(buffers, a, b, left_side, lower, transpose_a, unit_diagonal)
    })
}
