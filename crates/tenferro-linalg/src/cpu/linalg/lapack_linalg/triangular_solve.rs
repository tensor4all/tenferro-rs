use cblas_sys::{CBLAS_DIAG, CBLAS_LAYOUT, CBLAS_SIDE, CBLAS_TRANSPOSE, CBLAS_UPLO};
use num_complex::{Complex32, Complex64};
use num_traits::Zero;

use tenferro_cpu::linalg_interop::{BufferPool, PoolScalar};
use tenferro_tensor::TypedTensor;

use super::helpers::{
    check_lapack_info, checked_product, dim_i32, has_zero_dim, matrix_core_and_batch_result,
    pooled_copy, square_core_and_batch_result, tensor_from_vec_with_template,
};

pub(crate) trait LapackTriangularSolve:
    Clone + Copy + PartialEq + PoolScalar + Zero
{
    fn trtrs(args: TrtrsArgs<'_, Self>);

    fn trsm(args: TrsmArgs<'_, Self>);
}

pub(crate) struct TrtrsArgs<'a, T> {
    uplo: u8,
    trans: u8,
    diag: u8,
    n: i32,
    nrhs: i32,
    a: &'a [T],
    lda: i32,
    b: &'a mut [T],
    ldb: i32,
    info: &'a mut i32,
}

pub(crate) struct TrsmArgs<'a, T> {
    side: CBLAS_SIDE,
    uplo: CBLAS_UPLO,
    transa: CBLAS_TRANSPOSE,
    diag: CBLAS_DIAG,
    m: i32,
    n: i32,
    a: &'a [T],
    lda: i32,
    b: &'a mut [T],
    ldb: i32,
}

impl LapackTriangularSolve for f64 {
    fn trtrs(args: TrtrsArgs<'_, Self>) {
        let TrtrsArgs {
            uplo,
            trans,
            diag,
            n,
            nrhs,
            a,
            lda,
            b,
            ldb,
            info,
        } = args;
        // SAFETY: callers validate the triangular matrix and RHS shapes,
        // provide column-major `a`/`b` buffers matching `lda`/`ldb`, and live `info`.
        unsafe {
            lapack::dtrtrs(uplo, trans, diag, n, nrhs, a, lda, b, ldb, info);
        }
    }

    fn trsm(args: TrsmArgs<'_, Self>) {
        let TrsmArgs {
            side,
            uplo,
            transa,
            diag,
            m,
            n,
            a,
            lda,
            b,
            ldb,
        } = args;
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
    fn trtrs(args: TrtrsArgs<'_, Self>) {
        let TrtrsArgs {
            uplo,
            trans,
            diag,
            n,
            nrhs,
            a,
            lda,
            b,
            ldb,
            info,
        } = args;
        // SAFETY: callers validate the triangular matrix and RHS shapes,
        // provide column-major `a`/`b` buffers matching `lda`/`ldb`, and live `info`.
        unsafe {
            lapack::strtrs(uplo, trans, diag, n, nrhs, a, lda, b, ldb, info);
        }
    }

    fn trsm(args: TrsmArgs<'_, Self>) {
        let TrsmArgs {
            side,
            uplo,
            transa,
            diag,
            m,
            n,
            a,
            lda,
            b,
            ldb,
        } = args;
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
    fn trtrs(args: TrtrsArgs<'_, Self>) {
        let TrtrsArgs {
            uplo,
            trans,
            diag,
            n,
            nrhs,
            a,
            lda,
            b,
            ldb,
            info,
        } = args;
        // SAFETY: callers validate the triangular matrix and RHS shapes,
        // provide column-major `a`/`b` buffers matching `lda`/`ldb`, and live `info`.
        unsafe {
            lapack::ctrtrs(uplo, trans, diag, n, nrhs, a, lda, b, ldb, info);
        }
    }

    fn trsm(args: TrsmArgs<'_, Self>) {
        let TrsmArgs {
            side,
            uplo,
            transa,
            diag,
            m,
            n,
            a,
            lda,
            b,
            ldb,
        } = args;
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
    fn trtrs(args: TrtrsArgs<'_, Self>) {
        let TrtrsArgs {
            uplo,
            trans,
            diag,
            n,
            nrhs,
            a,
            lda,
            b,
            ldb,
            info,
        } = args;
        // SAFETY: callers validate the triangular matrix and RHS shapes,
        // provide column-major `a`/`b` buffers matching `lda`/`ldb`, and live `info`.
        unsafe {
            lapack::ztrtrs(uplo, trans, diag, n, nrhs, a, lda, b, ldb, info);
        }
    }

    fn trsm(args: TrsmArgs<'_, Self>) {
        let TrsmArgs {
            side,
            uplo,
            transa,
            diag,
            m,
            n,
            a,
            lda,
            b,
            ldb,
        } = args;
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

/// Reject an exactly zero diagonal of one compact `n x n` triangular matrix.
fn validate_non_unit_diagonal<T: LapackTriangularSolve>(
    matrix: &[T],
    n: usize,
) -> tenferro_tensor::Result<()> {
    // INVARIANT: callers pass one compact `n x n` chunk, so every diagonal
    // index `idx * (n + 1)` is in bounds.
    if (0..n).any(|idx| matrix[idx + idx * n] == T::zero()) {
        return Err(crate::error::into_tensor_error(
            "triangular_solve",
            crate::Error::Singular {
                op: "triangular_solve",
            },
        ));
    }
    Ok(())
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

    let matrix_len = checked_product(OP, "matrix", &[n, n])?;
    let rhs_len = checked_product(OP, "rhs", &[b_rows, b_cols])?;
    let n_i32 = dim_i32(n, OP)?;
    let rows_i32 = dim_i32(b_rows, OP)?;
    let cols_i32 = dim_i32(b_cols, OP)?;
    let mut output = pooled_copy(buffers, b.host_data()?);
    // INVARIANT: owned tensors are compact column-major, the batch shapes
    // match, and no dimension is zero, so both chunk iterators yield the same
    // number of nonempty matrix and RHS blocks. Each provider call reads its
    // own triangle and overwrites only its own RHS block in the output. The
    // serial loop is intentional: the BLAS/LAPACK provider owns threading.
    for (matrix, rhs) in a
        .host_data()?
        .chunks_exact(matrix_len)
        .zip(output.chunks_exact_mut(rhs_len))
    {
        if left_side {
            let mut info = 0;
            T::trtrs(TrtrsArgs {
                uplo: if lower { b'L' } else { b'U' },
                trans: if transpose_a { b'T' } else { b'N' },
                diag: if unit_diagonal { b'U' } else { b'N' },
                n: n_i32,
                nrhs: cols_i32,
                a: matrix,
                lda: n_i32,
                b: rhs,
                ldb: n_i32,
                info: &mut info,
            });
            check_lapack_info(OP, "trtrs", info)?;
        } else {
            if !unit_diagonal {
                validate_non_unit_diagonal(matrix, n)?;
            }
            T::trsm(TrsmArgs {
                side: CBLAS_SIDE::CblasRight,
                uplo: cblas_uplo(lower),
                transa: cblas_transpose(transpose_a),
                diag: cblas_diag(unit_diagonal),
                m: rows_i32,
                n: n_i32,
                a: matrix,
                lda: n_i32,
                b: rhs,
                ldb: rows_i32,
            });
        }
    }
    tensor_from_vec_with_template(b.shape().to_vec(), output, b)
}
