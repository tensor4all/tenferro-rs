use num_complex::Complex64;

use crate::buffer_pool::BufferPool;
use crate::TypedTensor;

use super::helpers::{
    batched_multi, dim_i32, has_zero_dim, leading_upper_triangle_from_lapack, matrix_dims,
    matrix_with_batch_shape, panic_on_lapack_error, tensor_from_vec_with_template,
};

pub(crate) trait LapackLu: Clone + Copy + Default {
    fn one() -> Self;
    fn negative_one() -> Self;
    fn getrf(m: i32, n: i32, data: &mut [Self], lda: i32, ipiv: &mut [i32], info: &mut i32);
}

impl LapackLu for f64 {
    fn one() -> Self {
        1.0
    }

    fn negative_one() -> Self {
        -1.0
    }

    fn getrf(m: i32, n: i32, data: &mut [Self], lda: i32, ipiv: &mut [i32], info: &mut i32) {
        unsafe {
            lapack::dgetrf(m, n, data, lda, ipiv, info);
        }
    }
}

impl LapackLu for Complex64 {
    fn one() -> Self {
        Complex64::new(1.0, 0.0)
    }

    fn negative_one() -> Self {
        Complex64::new(-1.0, 0.0)
    }

    fn getrf(m: i32, n: i32, data: &mut [Self], lda: i32, ipiv: &mut [i32], info: &mut i32) {
        unsafe {
            lapack::zgetrf(m, n, data, lda, ipiv, info);
        }
    }
}

fn lu_2d<T: LapackLu>(_buffers: &mut BufferPool, input: &TypedTensor<T>) -> Vec<TypedTensor<T>> {
    let (m, n) = matrix_dims(input, "lu");
    let k = m.min(n);
    let m_i32 = dim_i32(m, "lu");
    let n_i32 = dim_i32(n, "lu");
    let mut lu = input.host_data().to_vec();
    let mut ipiv = vec![0_i32; k];
    let mut info = 0;
    T::getrf(m_i32, n_i32, &mut lu, m_i32, &mut ipiv, &mut info);
    panic_on_lapack_error("lu", "dgetrf", info.min(0));

    let mut permutation: Vec<usize> = (0..m).collect();
    let mut swap_count = 0usize;
    for (idx, &pivot_one_based) in ipiv.iter().enumerate() {
        let pivot = match usize::try_from(pivot_one_based - 1) {
            Ok(pivot) => pivot,
            Err(_) => panic!("lu: LAPACK dgetrf returned invalid pivot index"),
        };
        if pivot != idx {
            permutation.swap(idx, pivot);
            swap_count += 1;
        }
    }

    let mut p_data = vec![T::default(); m * m];
    for (row, &source_row) in permutation.iter().enumerate() {
        p_data[row + source_row * m] = T::one();
    }
    let parity = if swap_count % 2 == 0 {
        T::one()
    } else {
        T::negative_one()
    };

    let mut l_data = vec![T::default(); m * k];
    for col in 0..k {
        for row in col..m {
            l_data[row + col * m] = lu[row + col * m];
        }
        l_data[col + col * m] = T::one();
    }
    let u_data = leading_upper_triangle_from_lapack(&lu, m, k, n);

    vec![
        tensor_from_vec_with_template(vec![m, m], p_data, input),
        tensor_from_vec_with_template(vec![m, k], l_data, input),
        tensor_from_vec_with_template(vec![k, n], u_data, input),
        tensor_from_vec_with_template(vec![], vec![parity], input),
    ]
}

pub(crate) fn lu<T: LapackLu>(
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> Vec<TypedTensor<T>> {
    if has_zero_dim(&input.shape) {
        let m = input.shape[0];
        let n = input.shape[1];
        let k = m.min(n);
        let batch_shape = &input.shape[2..];
        let parity_elements = if batch_shape.is_empty() {
            1
        } else {
            batch_shape.iter().product()
        };
        return vec![
            tensor_from_vec_with_template(
                matrix_with_batch_shape(m, m, batch_shape),
                Vec::new(),
                input,
            ),
            tensor_from_vec_with_template(
                matrix_with_batch_shape(m, k, batch_shape),
                Vec::new(),
                input,
            ),
            tensor_from_vec_with_template(
                matrix_with_batch_shape(k, n, batch_shape),
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
    batched_multi(buffers, input, lu_2d)
}
