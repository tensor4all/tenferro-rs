use num_complex::Complex64;

use crate::buffer_pool::BufferPool;
use crate::TypedTensor;

use super::helpers::{
    batched_binary, dim_i32, has_zero_dim, matrix_dims, panic_on_lapack_error, square_matrix_dim,
    tensor_from_vec_with_template, transpose_col_major_data,
};

pub(crate) trait LapackTriangularSolve: Clone + Copy {
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
        unsafe {
            lapack::dtrtrs(uplo, trans, diag, n, nrhs, a, lda, b, ldb, info);
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
        unsafe {
            lapack::ztrtrs(uplo, trans, diag, n, nrhs, a, lda, b, ldb, info);
        }
    }
}

fn solve_left<T: LapackTriangularSolve>(
    a: &TypedTensor<T>,
    b: &TypedTensor<T>,
    lower: bool,
    transpose_a: bool,
    unit_diagonal: bool,
) -> TypedTensor<T> {
    let n = square_matrix_dim(a, "triangular_solve");
    let (b_rows, b_cols) = matrix_dims(b, "triangular_solve");
    assert_eq!(b_rows, n, "triangular_solve: rhs row count mismatch");

    let mut rhs = b.host_data().to_vec();
    let mut info = 0;
    T::trtrs(
        if lower { b'L' } else { b'U' },
        if transpose_a { b'T' } else { b'N' },
        if unit_diagonal { b'U' } else { b'N' },
        dim_i32(n, "triangular_solve"),
        dim_i32(b_cols, "triangular_solve"),
        a.host_data(),
        dim_i32(n, "triangular_solve"),
        &mut rhs,
        dim_i32(n, "triangular_solve"),
        &mut info,
    );
    panic_on_lapack_error("triangular_solve", "dtrtrs", info);
    tensor_from_vec_with_template(vec![n, b_cols], rhs, b)
}

fn solve_right<T: LapackTriangularSolve>(
    a: &TypedTensor<T>,
    b: &TypedTensor<T>,
    lower: bool,
    transpose_a: bool,
    unit_diagonal: bool,
) -> TypedTensor<T> {
    let n = square_matrix_dim(a, "triangular_solve");
    let (b_rows, b_cols) = matrix_dims(b, "triangular_solve");
    assert_eq!(b_cols, n, "triangular_solve: rhs column count mismatch");

    let mut rhs_t = transpose_col_major_data(b.host_data(), b_rows, n);
    let mut info = 0;
    T::trtrs(
        if lower { b'L' } else { b'U' },
        if transpose_a { b'N' } else { b'T' },
        if unit_diagonal { b'U' } else { b'N' },
        dim_i32(n, "triangular_solve"),
        dim_i32(b_rows, "triangular_solve"),
        a.host_data(),
        dim_i32(n, "triangular_solve"),
        &mut rhs_t,
        dim_i32(n, "triangular_solve"),
        &mut info,
    );
    panic_on_lapack_error("triangular_solve", "dtrtrs", info);
    let result = transpose_col_major_data(&rhs_t, n, b_rows);
    tensor_from_vec_with_template(vec![b_rows, n], result, b)
}

fn triangular_solve_2d<T: LapackTriangularSolve>(
    _buffers: &mut BufferPool,
    a: &TypedTensor<T>,
    b: &TypedTensor<T>,
    left_side: bool,
    lower: bool,
    transpose_a: bool,
    unit_diagonal: bool,
) -> TypedTensor<T> {
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
) -> TypedTensor<T> {
    if has_zero_dim(&a.shape) || has_zero_dim(&b.shape) {
        return tensor_from_vec_with_template(b.shape.clone(), Vec::new(), b);
    }
    batched_binary(buffers, a, b, |buffers, a, b| {
        triangular_solve_2d(buffers, a, b, left_side, lower, transpose_a, unit_diagonal)
    })
}
