use num_complex::{Complex32, Complex64};

use crate::buffer_pool::BufferPool;
use crate::TypedTensor;

use super::helpers::{
    batched_multi, check_lapack_info, dim_i32, has_zero_dim, matrix_dims, matrix_with_batch_shape,
    split_core_and_batch_result, tensor_from_vec_with_template, vector_with_batch_shape, work_len,
};

pub(crate) trait LapackSvd: Clone + Copy + Default {
    fn svd_2d(
        buffers: &mut BufferPool,
        input: &TypedTensor<Self>,
    ) -> crate::Result<Vec<TypedTensor<Self>>>;
}

macro_rules! impl_real_svd {
    ($scalar:ty, $gesvd:path, $routine:literal) => {
        impl LapackSvd for $scalar {
            fn svd_2d(
                _buffers: &mut BufferPool,
                input: &TypedTensor<Self>,
            ) -> crate::Result<Vec<TypedTensor<Self>>> {
                let (m, n) = matrix_dims(input, "svd")?;
                let k = m.min(n);
                let m_i32 = dim_i32(m, "svd")?;
                let n_i32 = dim_i32(n, "svd")?;
                let k_i32 = dim_i32(k, "svd")?;

                let mut a = input.host_data().to_vec();
                let mut s = vec![0.0 as $scalar; k];
                let mut u = vec![0.0 as $scalar; m * k];
                let mut vt = vec![0.0 as $scalar; k * n];
                let mut query = vec![0.0 as $scalar; 1];
                let mut info = 0;
                unsafe {
                    $gesvd(
                        b'S', b'S', m_i32, n_i32, &mut a, m_i32, &mut s, &mut u, m_i32, &mut vt,
                        k_i32, &mut query, -1, &mut info,
                    );
                }
                check_lapack_info("svd", concat!($routine, "(work query)"), info)?;
                let lwork = work_len(query[0] as f64, "svd", $routine)?;
                let mut work = vec![0.0 as $scalar; lwork as usize];
                unsafe {
                    $gesvd(
                        b'S', b'S', m_i32, n_i32, &mut a, m_i32, &mut s, &mut u, m_i32, &mut vt,
                        k_i32, &mut work, lwork, &mut info,
                    );
                }
                check_lapack_info("svd", $routine, info)?;

                Ok(vec![
                    tensor_from_vec_with_template(vec![m, k], u, input),
                    tensor_from_vec_with_template(vec![k], s, input),
                    tensor_from_vec_with_template(vec![k, n], vt, input),
                ])
            }
        }
    };
}

macro_rules! impl_complex_svd {
    ($complex:ty, $real:ty, $gesvd:path, $routine:literal) => {
        impl LapackSvd for $complex {
            fn svd_2d(
                _buffers: &mut BufferPool,
                input: &TypedTensor<Self>,
            ) -> crate::Result<Vec<TypedTensor<Self>>> {
                let (m, n) = matrix_dims(input, "svd")?;
                let k = m.min(n);
                let m_i32 = dim_i32(m, "svd")?;
                let n_i32 = dim_i32(n, "svd")?;
                let k_i32 = dim_i32(k, "svd")?;

                let mut a = input.host_data().to_vec();
                let mut s = vec![0.0 as $real; k];
                let mut u = vec![<$complex>::new(0.0, 0.0); m * k];
                let mut vt = vec![<$complex>::new(0.0, 0.0); k * n];
                let mut query = vec![<$complex>::new(0.0, 0.0); 1];
                let mut rwork = vec![0.0 as $real; 5 * k.max(1)];
                let mut info = 0;
                unsafe {
                    $gesvd(
                        b'S', b'S', m_i32, n_i32, &mut a, m_i32, &mut s, &mut u, m_i32, &mut vt,
                        k_i32, &mut query, -1, &mut rwork, &mut info,
                    );
                }
                check_lapack_info("svd", concat!($routine, "(work query)"), info)?;
                let lwork = work_len(query[0].re as f64, "svd", $routine)?;
                let mut work = vec![<$complex>::new(0.0, 0.0); lwork as usize];
                unsafe {
                    $gesvd(
                        b'S', b'S', m_i32, n_i32, &mut a, m_i32, &mut s, &mut u, m_i32, &mut vt,
                        k_i32, &mut work, lwork, &mut rwork, &mut info,
                    );
                }
                check_lapack_info("svd", $routine, info)?;

                Ok(vec![
                    tensor_from_vec_with_template(vec![m, k], u, input),
                    tensor_from_vec_with_template(
                        vec![k],
                        s.into_iter()
                            .map(|value| <$complex>::new(value, 0.0))
                            .collect(),
                        input,
                    ),
                    tensor_from_vec_with_template(vec![k, n], vt, input),
                ])
            }
        }
    };
}

impl_real_svd!(f32, lapack::sgesvd, "sgesvd");
impl_real_svd!(f64, lapack::dgesvd, "dgesvd");
impl_complex_svd!(Complex32, f32, lapack::cgesvd, "cgesvd");
impl_complex_svd!(Complex64, f64, lapack::zgesvd, "zgesvd");

fn svd_2d<T: LapackSvd>(
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> crate::Result<Vec<TypedTensor<T>>> {
    T::svd_2d(buffers, input)
}

pub(crate) fn svd<T: LapackSvd>(
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> crate::Result<Vec<TypedTensor<T>>> {
    if has_zero_dim(&input.shape) {
        let (matrix_shape, batch_shape) = split_core_and_batch_result(input, 2, "svd")?;
        let m = matrix_shape[0];
        let n = matrix_shape[1];
        let k = m.min(n);
        return Ok(vec![
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
        ]);
    }
    batched_multi("svd", buffers, input, svd_2d)
}
