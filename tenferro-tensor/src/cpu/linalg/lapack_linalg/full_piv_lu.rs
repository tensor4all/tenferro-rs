use num_complex::Complex64;

use crate::buffer_pool::BufferPool;
use crate::TypedTensor;

use super::helpers::{
    dim_i32, has_zero_dim, leading_upper_triangle_from_lapack, lower_triangle_from_lapack,
    matrix_dims, matrix_with_batch_shape, panic_on_lapack_error, square_matrix_dim,
    tensor_from_vec_with_template, transpose_col_major_data,
};

extern "C" {
    #[link_name = "dgetc2_"]
    fn dgetc2_ffi(
        n: *const i32,
        a: *mut f64,
        lda: *const i32,
        ipiv: *mut i32,
        jpiv: *mut i32,
        info: *mut i32,
    );

    #[link_name = "dgesc2_"]
    fn dgesc2_ffi(
        n: *const i32,
        a: *const f64,
        lda: *const i32,
        rhs: *mut f64,
        ipiv: *const i32,
        jpiv: *const i32,
        scale: *mut f64,
    );

    #[link_name = "zgetc2_"]
    fn zgetc2_ffi(
        n: *const i32,
        a: *mut Complex64,
        lda: *const i32,
        ipiv: *mut i32,
        jpiv: *mut i32,
        info: *mut i32,
    );

    #[link_name = "zgesc2_"]
    fn zgesc2_ffi(
        n: *const i32,
        a: *const Complex64,
        lda: *const i32,
        rhs: *mut Complex64,
        ipiv: *const i32,
        jpiv: *const i32,
        scale: *mut f64,
    );
}

pub(crate) trait LapackFullPivLu: Clone + Copy + Default {
    fn one() -> Self;
    fn negative_one() -> Self;
    fn getc2(
        n: i32,
        data: &mut [Self],
        lda: i32,
        ipiv: &mut [i32],
        jpiv: &mut [i32],
        info: &mut i32,
    );
    fn gesc2(
        n: i32,
        data: &[Self],
        lda: i32,
        rhs: &mut [Self],
        ipiv: &[i32],
        jpiv: &[i32],
        scale: &mut f64,
    );
    fn apply_inverse_scale(rhs: &mut [Self], scale: f64);
}

impl LapackFullPivLu for f64 {
    fn one() -> Self {
        1.0
    }

    fn negative_one() -> Self {
        -1.0
    }

    fn getc2(
        n: i32,
        data: &mut [Self],
        lda: i32,
        ipiv: &mut [i32],
        jpiv: &mut [i32],
        info: &mut i32,
    ) {
        // SAFETY: `data` stores an `lda x n` LAPACK column-major matrix,
        // pivot arrays have length at least `n`, and all pointers are valid
        // for the duration of the FFI call.
        unsafe {
            dgetc2_ffi(
                &n,
                data.as_mut_ptr(),
                &lda,
                ipiv.as_mut_ptr(),
                jpiv.as_mut_ptr(),
                info,
            );
        }
    }

    fn gesc2(
        n: i32,
        data: &[Self],
        lda: i32,
        rhs: &mut [Self],
        ipiv: &[i32],
        jpiv: &[i32],
        scale: &mut f64,
    ) {
        // SAFETY: `data` stores the factorized `lda x n` matrix, `rhs` has
        // length at least `n`, pivot arrays have length at least `n`, and
        // LAPACK only writes through `rhs` and `scale`.
        unsafe {
            dgesc2_ffi(
                &n,
                data.as_ptr(),
                &lda,
                rhs.as_mut_ptr(),
                ipiv.as_ptr(),
                jpiv.as_ptr(),
                scale,
            );
        }
    }

    fn apply_inverse_scale(rhs: &mut [Self], scale: f64) {
        if scale != 1.0 {
            for value in rhs {
                *value /= scale;
            }
        }
    }
}

impl LapackFullPivLu for Complex64 {
    fn one() -> Self {
        Complex64::new(1.0, 0.0)
    }

    fn negative_one() -> Self {
        Complex64::new(-1.0, 0.0)
    }

    fn getc2(
        n: i32,
        data: &mut [Self],
        lda: i32,
        ipiv: &mut [i32],
        jpiv: &mut [i32],
        info: &mut i32,
    ) {
        // SAFETY: `data` stores an `lda x n` LAPACK column-major matrix,
        // pivot arrays have length at least `n`, and all pointers are valid
        // for the duration of the FFI call.
        unsafe {
            zgetc2_ffi(
                &n,
                data.as_mut_ptr(),
                &lda,
                ipiv.as_mut_ptr(),
                jpiv.as_mut_ptr(),
                info,
            );
        }
    }

    fn gesc2(
        n: i32,
        data: &[Self],
        lda: i32,
        rhs: &mut [Self],
        ipiv: &[i32],
        jpiv: &[i32],
        scale: &mut f64,
    ) {
        // SAFETY: `data` stores the factorized `lda x n` matrix, `rhs` has
        // length at least `n`, pivot arrays have length at least `n`, and
        // LAPACK only writes through `rhs` and `scale`.
        unsafe {
            zgesc2_ffi(
                &n,
                data.as_ptr(),
                &lda,
                rhs.as_mut_ptr(),
                ipiv.as_ptr(),
                jpiv.as_ptr(),
                scale,
            );
        }
    }

    fn apply_inverse_scale(rhs: &mut [Self], scale: f64) {
        if scale != 1.0 {
            for value in rhs {
                *value /= scale;
            }
        }
    }
}

fn permutation_from_lapack_pivots(pivots: &[i32], op: &str) -> Vec<usize> {
    let mut permutation: Vec<usize> = (0..pivots.len()).collect();
    for (idx, &pivot_one_based) in pivots.iter().enumerate() {
        let pivot = match usize::try_from(pivot_one_based - 1) {
            Ok(pivot) if pivot < pivots.len() => pivot,
            _ => panic!("{op}: LAPACK getc2 returned invalid pivot index"),
        };
        if pivot != idx {
            permutation.swap(idx, pivot);
        }
    }
    permutation
}

fn permutation_matrix<T: LapackFullPivLu>(permutation: &[usize]) -> Vec<T> {
    let n = permutation.len();
    let mut data = vec![T::default(); n * n];
    for (row, &source) in permutation.iter().enumerate() {
        data[row + source * n] = T::one();
    }
    data
}

fn factor_getc2<T: LapackFullPivLu>(
    op: &'static str,
    data: &mut [T],
    n: usize,
) -> (Vec<i32>, Vec<i32>, i32) {
    let n_i32 = dim_i32(n, op);
    let mut ipiv = vec![0_i32; n];
    let mut jpiv = vec![0_i32; n];
    let mut info = 0;
    T::getc2(n_i32, data, n_i32, &mut ipiv, &mut jpiv, &mut info);
    panic_on_lapack_error(op, "getc2", info.min(0));
    (ipiv, jpiv, info)
}

fn full_piv_lu_2d<T: LapackFullPivLu>(
    _buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> Vec<TypedTensor<T>> {
    let n = square_matrix_dim(input, "full_piv_lu");
    let mut lu = input.host_data().to_vec();
    let (ipiv, jpiv, _info) = factor_getc2("full_piv_lu", &mut lu, n);

    let row_perm = permutation_from_lapack_pivots(&ipiv, "full_piv_lu");
    let col_perm = permutation_from_lapack_pivots(&jpiv, "full_piv_lu");
    let p_data = permutation_matrix::<T>(&row_perm);
    let q_data = permutation_matrix::<T>(&col_perm);
    let mut l_data = lower_triangle_from_lapack(&lu, n, n);
    for index in 0..n {
        l_data[index + index * n] = T::one();
    }
    let u_data = leading_upper_triangle_from_lapack(&lu, n, n, n);
    let row_swap_count = ipiv
        .iter()
        .enumerate()
        .filter(|(idx, row)| **row != (*idx as i32 + 1))
        .count();
    let col_swap_count = jpiv
        .iter()
        .enumerate()
        .filter(|(idx, col)| **col != (*idx as i32 + 1))
        .count();
    let parity = if (row_swap_count + col_swap_count) % 2 == 0 {
        T::one()
    } else {
        T::negative_one()
    };

    vec![
        tensor_from_vec_with_template(vec![n, n], p_data, input),
        tensor_from_vec_with_template(vec![n, n], l_data, input),
        tensor_from_vec_with_template(vec![n, n], u_data, input),
        tensor_from_vec_with_template(vec![n, n], q_data, input),
        tensor_from_vec_with_template(vec![], vec![parity], input),
    ]
}

fn solve_2d<T: LapackFullPivLu>(
    _buffers: &mut BufferPool,
    a: &TypedTensor<T>,
    b: &TypedTensor<T>,
    transpose_a: bool,
) -> crate::Result<TypedTensor<T>> {
    let n = square_matrix_dim(a, "full_piv_lu_solve");
    let (b_rows, b_cols) = matrix_dims(b, "full_piv_lu_solve");
    assert_eq!(b_rows, n, "full_piv_lu_solve: rhs row count mismatch");

    let mut lu = if transpose_a {
        transpose_col_major_data(a.host_data(), n, n)
    } else {
        a.host_data().to_vec()
    };
    let (ipiv, jpiv, info) = factor_getc2("full_piv_lu_solve", &mut lu, n);
    if info > 0 {
        return Err(crate::Error::BackendFailure {
            op: "full_piv_lu_solve",
            message: "matrix is singular".into(),
        });
    }

    let mut rhs = b.host_data().to_vec();
    let n_i32 = dim_i32(n, "full_piv_lu_solve");
    for col in 0..b_cols {
        let start = col * n;
        let end = start + n;
        let mut scale = 1.0;
        T::gesc2(
            n_i32,
            &lu,
            n_i32,
            &mut rhs[start..end],
            &ipiv,
            &jpiv,
            &mut scale,
        );
        T::apply_inverse_scale(&mut rhs[start..end], scale);
    }

    Ok(tensor_from_vec_with_template(vec![n, b_cols], rhs, b))
}

fn batched_binary_result<T, F>(
    buffers: &mut BufferPool,
    a: &TypedTensor<T>,
    b: &TypedTensor<T>,
    op: F,
) -> crate::Result<TypedTensor<T>>
where
    T: Clone,
    F: Fn(&mut BufferPool, &TypedTensor<T>, &TypedTensor<T>) -> crate::Result<TypedTensor<T>>,
{
    let (a_core_shape, a_batch_shape) =
        super::helpers::split_core_and_batch(a, 2, "batched_binary_result");
    let (b_core_shape, b_batch_shape) =
        super::helpers::split_core_and_batch(b, 2, "batched_binary_result");
    assert_eq!(
        a_batch_shape, b_batch_shape,
        "batched_binary_result: batch shape mismatch"
    );

    if a_batch_shape.is_empty() {
        return op(buffers, a, b);
    }

    let a_slice_size: usize = a_core_shape.iter().product();
    let b_slice_size: usize = b_core_shape.iter().product();
    let batch_count: usize = a_batch_shape.iter().product();
    assert!(
        batch_count > 0,
        "batched_binary_result: zero-sized batch dims are unsupported"
    );

    let mut out_core_shape: Option<Vec<usize>> = None;
    let mut out_data: Option<Vec<T>> = None;

    for batch_idx in 0..batch_count {
        let a_start = batch_idx * a_slice_size;
        let a_end = a_start + a_slice_size;
        let b_start = batch_idx * b_slice_size;
        let b_end = b_start + b_slice_size;

        let batch_a = tensor_from_vec_with_template(
            a_core_shape.to_vec(),
            a.host_data()[a_start..a_end].to_vec(),
            a,
        );
        let batch_b = tensor_from_vec_with_template(
            b_core_shape.to_vec(),
            b.host_data()[b_start..b_end].to_vec(),
            b,
        );
        let batch_output = op(buffers, &batch_a, &batch_b)?;

        if let Some(expected_shape) = &out_core_shape {
            assert_eq!(
                batch_output.shape.as_slice(),
                expected_shape.as_slice(),
                "batched_binary_result: output core shape mismatch across batches"
            );
        } else {
            out_data = Some(Vec::with_capacity(batch_output.n_elements() * batch_count));
            out_core_shape = Some(batch_output.shape.clone());
        }

        match &mut out_data {
            Some(data) => data.extend_from_slice(batch_output.host_data()),
            None => panic!("batched_binary_result: missing output buffer"),
        }
    }

    let mut out_shape = match out_core_shape {
        Some(shape) => shape,
        None => panic!("batched_binary_result: missing output shape"),
    };
    out_shape.extend_from_slice(a_batch_shape);
    Ok(tensor_from_vec_with_template(
        out_shape,
        match out_data {
            Some(data) => data,
            None => panic!("batched_binary_result: missing output data"),
        },
        b,
    ))
}

pub(crate) fn full_piv_lu<T: LapackFullPivLu>(
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> Vec<TypedTensor<T>> {
    if has_zero_dim(&input.shape) {
        let n = input.shape[0];
        let batch_shape = &input.shape[2..];
        let parity_elements = if batch_shape.is_empty() {
            1
        } else {
            batch_shape.iter().product()
        };
        return vec![
            tensor_from_vec_with_template(
                matrix_with_batch_shape(n, n, batch_shape),
                Vec::new(),
                input,
            ),
            tensor_from_vec_with_template(
                matrix_with_batch_shape(n, n, batch_shape),
                Vec::new(),
                input,
            ),
            tensor_from_vec_with_template(
                matrix_with_batch_shape(n, n, batch_shape),
                Vec::new(),
                input,
            ),
            tensor_from_vec_with_template(
                matrix_with_batch_shape(n, n, batch_shape),
                Vec::new(),
                input,
            ),
            tensor_from_vec_with_template(
                batch_shape.to_vec(),
                vec![T::one(); parity_elements],
                input,
            ),
        ];
    }
    super::helpers::batched_multi(buffers, input, full_piv_lu_2d)
}

pub(crate) fn full_piv_lu_solve<T: LapackFullPivLu>(
    buffers: &mut BufferPool,
    a: &TypedTensor<T>,
    b: &TypedTensor<T>,
    transpose_a: bool,
) -> crate::Result<TypedTensor<T>> {
    if has_zero_dim(&a.shape) || has_zero_dim(&b.shape) {
        return Ok(tensor_from_vec_with_template(
            b.shape.clone(),
            Vec::new(),
            b,
        ));
    }
    batched_binary_result(buffers, a, b, |buffers, a, b| {
        solve_2d(buffers, a, b, transpose_a)
    })
}
