use num_complex::Complex64;

use crate::buffer_pool::BufferPool;
use crate::TypedTensor;

use super::helpers::{
    batched_multi, dim_i32, has_zero_dim, matrix_dims, matrix_with_batch_shape,
    panic_on_lapack_error, split_core_and_batch, tensor_from_vec_with_template,
    vector_with_batch_shape, work_len,
};

pub(crate) trait LapackSvd: Clone + Copy + Default {
    fn svd_2d(buffers: &mut BufferPool, input: &TypedTensor<Self>) -> Vec<TypedTensor<Self>>;
}

impl LapackSvd for f64 {
    fn svd_2d(_buffers: &mut BufferPool, input: &TypedTensor<Self>) -> Vec<TypedTensor<Self>> {
        let (m, n) = matrix_dims(input, "svd");
        let k = m.min(n);
        let m_i32 = dim_i32(m, "svd");
        let n_i32 = dim_i32(n, "svd");
        let k_i32 = dim_i32(k, "svd");

        let mut a = input.host_data().to_vec();
        let mut s = vec![0.0; k];
        let mut u = vec![0.0; m * k];
        let mut vt = vec![0.0; k * n];
        let mut query = vec![0.0; 1];
        let mut info = 0;
        unsafe {
            lapack::dgesvd(
                b'S', b'S', m_i32, n_i32, &mut a, m_i32, &mut s, &mut u, m_i32, &mut vt, k_i32,
                &mut query, -1, &mut info,
            );
        }
        panic_on_lapack_error("svd", "dgesvd(work query)", info);
        let lwork = work_len(query[0], "svd", "dgesvd");
        let mut work = vec![0.0; lwork as usize];
        unsafe {
            lapack::dgesvd(
                b'S', b'S', m_i32, n_i32, &mut a, m_i32, &mut s, &mut u, m_i32, &mut vt, k_i32,
                &mut work, lwork, &mut info,
            );
        }
        panic_on_lapack_error("svd", "dgesvd", info);

        vec![
            tensor_from_vec_with_template(vec![m, k], u, input),
            tensor_from_vec_with_template(vec![k], s, input),
            tensor_from_vec_with_template(vec![k, n], vt, input),
        ]
    }
}

impl LapackSvd for Complex64 {
    fn svd_2d(_buffers: &mut BufferPool, input: &TypedTensor<Self>) -> Vec<TypedTensor<Self>> {
        let (m, n) = matrix_dims(input, "svd");
        let k = m.min(n);
        let m_i32 = dim_i32(m, "svd");
        let n_i32 = dim_i32(n, "svd");
        let k_i32 = dim_i32(k, "svd");

        let mut a = input.host_data().to_vec();
        let mut s = vec![0.0; k];
        let mut u = vec![Complex64::new(0.0, 0.0); m * k];
        let mut vt = vec![Complex64::new(0.0, 0.0); k * n];
        let mut query = vec![Complex64::new(0.0, 0.0); 1];
        let mut rwork = vec![0.0; 5 * k.max(1)];
        let mut info = 0;
        unsafe {
            lapack::zgesvd(
                b'S', b'S', m_i32, n_i32, &mut a, m_i32, &mut s, &mut u, m_i32, &mut vt, k_i32,
                &mut query, -1, &mut rwork, &mut info,
            );
        }
        panic_on_lapack_error("svd", "zgesvd(work query)", info);
        let lwork = work_len(query[0].re, "svd", "zgesvd");
        let mut work = vec![Complex64::new(0.0, 0.0); lwork as usize];
        unsafe {
            lapack::zgesvd(
                b'S', b'S', m_i32, n_i32, &mut a, m_i32, &mut s, &mut u, m_i32, &mut vt, k_i32,
                &mut work, lwork, &mut rwork, &mut info,
            );
        }
        panic_on_lapack_error("svd", "zgesvd", info);

        vec![
            tensor_from_vec_with_template(vec![m, k], u, input),
            tensor_from_vec_with_template(
                vec![k],
                s.into_iter()
                    .map(|value| Complex64::new(value, 0.0))
                    .collect(),
                input,
            ),
            tensor_from_vec_with_template(vec![k, n], vt, input),
        ]
    }
}

fn svd_2d<T: LapackSvd>(buffers: &mut BufferPool, input: &TypedTensor<T>) -> Vec<TypedTensor<T>> {
    T::svd_2d(buffers, input)
}

pub(crate) fn svd<T: LapackSvd>(
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> Vec<TypedTensor<T>> {
    if has_zero_dim(&input.shape) {
        let (matrix_shape, batch_shape) = split_core_and_batch(input, 2, "svd");
        let m = matrix_shape[0];
        let n = matrix_shape[1];
        let k = m.min(n);
        return vec![
            tensor_from_vec_with_template(
                matrix_with_batch_shape(m, k, batch_shape),
                Vec::new(),
                input,
            ),
            tensor_from_vec_with_template(
                vector_with_batch_shape(k, batch_shape),
                Vec::new(),
                input,
            ),
            tensor_from_vec_with_template(
                matrix_with_batch_shape(k, n, batch_shape),
                Vec::new(),
                input,
            ),
        ];
    }
    batched_multi(buffers, input, svd_2d)
}
