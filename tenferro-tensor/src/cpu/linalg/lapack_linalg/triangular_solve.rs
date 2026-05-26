use num_complex::{Complex32, Complex64};

use crate::buffer_pool::BufferPool;
use crate::TypedTensor;

use super::helpers::{
    batched_binary_result, check_lapack_info, dim_i32, has_zero_dim, matrix_core_and_batch_result,
    matrix_dims, square_core_and_batch_result, square_matrix_dim, tensor_from_vec_with_template,
    transpose_col_major_data,
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
        unsafe {
            lapack::strtrs(uplo, trans, diag, n, nrhs, a, lda, b, ldb, info);
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
        unsafe {
            lapack::ctrtrs(uplo, trans, diag, n, nrhs, a, lda, b, ldb, info);
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
) -> crate::Result<TypedTensor<T>> {
    let n = square_matrix_dim(a, "triangular_solve")?;
    let (b_rows, b_cols) = matrix_dims(b, "triangular_solve")?;
    if b_rows != n {
        return Err(crate::Error::ShapeMismatch {
            op: "triangular_solve",
            lhs: vec![n],
            rhs: vec![b_rows],
        });
    }

    let mut rhs = b.host_data().to_vec();
    let mut info = 0;
    T::trtrs(
        if lower { b'L' } else { b'U' },
        if transpose_a { b'T' } else { b'N' },
        if unit_diagonal { b'U' } else { b'N' },
        dim_i32(n, "triangular_solve")?,
        dim_i32(b_cols, "triangular_solve")?,
        a.host_data(),
        dim_i32(n, "triangular_solve")?,
        &mut rhs,
        dim_i32(n, "triangular_solve")?,
        &mut info,
    );
    check_lapack_info("triangular_solve", "trtrs", info)?;
    Ok(tensor_from_vec_with_template(vec![n, b_cols], rhs, b))
}

fn solve_right<T: LapackTriangularSolve>(
    a: &TypedTensor<T>,
    b: &TypedTensor<T>,
    lower: bool,
    transpose_a: bool,
    unit_diagonal: bool,
) -> crate::Result<TypedTensor<T>> {
    let n = square_matrix_dim(a, "triangular_solve")?;
    let (b_rows, b_cols) = matrix_dims(b, "triangular_solve")?;
    if b_cols != n {
        return Err(crate::Error::ShapeMismatch {
            op: "triangular_solve",
            lhs: vec![n],
            rhs: vec![b_cols],
        });
    }

    let mut rhs_t = transpose_col_major_data(b.host_data(), b_rows, n);
    let mut info = 0;
    T::trtrs(
        if lower { b'L' } else { b'U' },
        if transpose_a { b'N' } else { b'T' },
        if unit_diagonal { b'U' } else { b'N' },
        dim_i32(n, "triangular_solve")?,
        dim_i32(b_rows, "triangular_solve")?,
        a.host_data(),
        dim_i32(n, "triangular_solve")?,
        &mut rhs_t,
        dim_i32(n, "triangular_solve")?,
        &mut info,
    );
    check_lapack_info("triangular_solve", "trtrs", info)?;
    let result = transpose_col_major_data(&rhs_t, n, b_rows);
    Ok(tensor_from_vec_with_template(vec![b_rows, n], result, b))
}

fn triangular_solve_2d<T: LapackTriangularSolve>(
    _buffers: &mut BufferPool,
    a: &TypedTensor<T>,
    b: &TypedTensor<T>,
    left_side: bool,
    lower: bool,
    transpose_a: bool,
    unit_diagonal: bool,
) -> crate::Result<TypedTensor<T>> {
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
) -> crate::Result<TypedTensor<T>> {
    if has_zero_dim(&a.shape) || has_zero_dim(&b.shape) {
        let (n, a_batch_shape) = square_core_and_batch_result(a, "triangular_solve")?;
        let (b_rows, b_cols, b_batch_shape) = matrix_core_and_batch_result(b, "triangular_solve")?;
        let rhs_core_dim = if left_side { b_rows } else { b_cols };
        if rhs_core_dim != n {
            return Err(crate::Error::ShapeMismatch {
                op: "triangular_solve",
                lhs: vec![n],
                rhs: vec![rhs_core_dim],
            });
        }
        if a_batch_shape != b_batch_shape {
            return Err(crate::Error::ShapeMismatch {
                op: "triangular_solve",
                lhs: a_batch_shape.to_vec(),
                rhs: b_batch_shape.to_vec(),
            });
        }
        return Ok(tensor_from_vec_with_template(
            b.shape.clone(),
            Vec::new(),
            b,
        ));
    }
    batched_binary_result("triangular_solve", buffers, a, b, |buffers, a, b| {
        triangular_solve_2d(buffers, a, b, left_side, lower, transpose_a, unit_diagonal)
    })
}
