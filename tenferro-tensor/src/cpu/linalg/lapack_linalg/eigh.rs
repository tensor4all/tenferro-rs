use num_complex::Complex64;

use crate::buffer_pool::BufferPool;
use crate::TypedTensor;

use super::helpers::{
    batched_multi, dim_i32, has_zero_dim, matrix_with_batch_shape, panic_on_lapack_error,
    square_matrix_dim, tensor_from_vec_with_template, vector_with_batch_shape, work_len,
};

pub(crate) trait LapackEigh: Clone + Copy + Default {
    fn eigh_2d(buffers: &mut BufferPool, input: &TypedTensor<Self>) -> Vec<TypedTensor<Self>>;
}

impl LapackEigh for f64 {
    fn eigh_2d(_buffers: &mut BufferPool, input: &TypedTensor<Self>) -> Vec<TypedTensor<Self>> {
        let n = square_matrix_dim(input, "eigh");
        let n_i32 = dim_i32(n, "eigh");
        let mut vectors = input.host_data().to_vec();
        let mut values = vec![0.0; n];
        let mut query = vec![0.0; 1];
        let mut info = 0;
        unsafe {
            lapack::dsyev(
                b'V',
                b'L',
                n_i32,
                &mut vectors,
                n_i32,
                &mut values,
                &mut query,
                -1,
                &mut info,
            );
        }
        panic_on_lapack_error("eigh", "dsyev(work query)", info);
        let lwork = work_len(query[0], "eigh", "dsyev");
        let mut work = vec![0.0; lwork as usize];
        unsafe {
            lapack::dsyev(
                b'V',
                b'L',
                n_i32,
                &mut vectors,
                n_i32,
                &mut values,
                &mut work,
                lwork,
                &mut info,
            );
        }
        panic_on_lapack_error("eigh", "dsyev", info);

        vec![
            tensor_from_vec_with_template(vec![n], values, input),
            tensor_from_vec_with_template(vec![n, n], vectors, input),
        ]
    }
}

impl LapackEigh for Complex64 {
    fn eigh_2d(_buffers: &mut BufferPool, input: &TypedTensor<Self>) -> Vec<TypedTensor<Self>> {
        let n = square_matrix_dim(input, "eigh");
        let n_i32 = dim_i32(n, "eigh");
        let mut vectors = input.host_data().to_vec();
        let mut values = vec![0.0; n];
        let mut query = vec![Complex64::new(0.0, 0.0); 1];
        let mut rwork = vec![0.0; (3 * n).saturating_sub(2).max(1)];
        let mut info = 0;
        unsafe {
            lapack::zheev(
                b'V',
                b'L',
                n_i32,
                &mut vectors,
                n_i32,
                &mut values,
                &mut query,
                -1,
                &mut rwork,
                &mut info,
            );
        }
        panic_on_lapack_error("eigh", "zheev(work query)", info);
        let lwork = work_len(query[0].re, "eigh", "zheev");
        let mut work = vec![Complex64::new(0.0, 0.0); lwork as usize];
        unsafe {
            lapack::zheev(
                b'V',
                b'L',
                n_i32,
                &mut vectors,
                n_i32,
                &mut values,
                &mut work,
                lwork,
                &mut rwork,
                &mut info,
            );
        }
        panic_on_lapack_error("eigh", "zheev", info);

        vec![
            tensor_from_vec_with_template(
                vec![n],
                values
                    .into_iter()
                    .map(|value| Complex64::new(value, 0.0))
                    .collect(),
                input,
            ),
            tensor_from_vec_with_template(vec![n, n], vectors, input),
        ]
    }
}

fn eigh_2d<T: LapackEigh>(buffers: &mut BufferPool, input: &TypedTensor<T>) -> Vec<TypedTensor<T>> {
    T::eigh_2d(buffers, input)
}

pub(crate) fn eigh<T: LapackEigh>(
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> Vec<TypedTensor<T>> {
    if has_zero_dim(&input.shape) {
        let n = input.shape[0];
        let batch_shape = &input.shape[2..];
        return vec![
            tensor_from_vec_with_template(
                vector_with_batch_shape(n, batch_shape),
                Vec::new(),
                input,
            ),
            tensor_from_vec_with_template(
                matrix_with_batch_shape(n, n, batch_shape),
                Vec::new(),
                input,
            ),
        ];
    }
    batched_multi(buffers, input, eigh_2d)
}
