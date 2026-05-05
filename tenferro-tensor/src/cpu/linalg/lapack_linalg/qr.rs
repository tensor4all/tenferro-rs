use num_complex::{Complex32, Complex64};

use crate::buffer_pool::BufferPool;
use crate::TypedTensor;

use super::helpers::{
    batched_multi, check_lapack_info, dim_i32, has_zero_dim, leading_upper_triangle_from_lapack,
    matrix_dims, matrix_with_batch_shape, split_core_and_batch_result,
    tensor_from_vec_with_template, work_len,
};

pub(crate) trait LapackQr: Clone + Copy + Default {
    fn qr_2d(
        buffers: &mut BufferPool,
        input: &TypedTensor<Self>,
    ) -> crate::Result<Vec<TypedTensor<Self>>>;
}

macro_rules! impl_real_qr {
    ($scalar:ty, $geqrf:path, $orgqr:path, $geqrf_name:literal, $orgqr_name:literal) => {
        impl LapackQr for $scalar {
            fn qr_2d(
                _buffers: &mut BufferPool,
                input: &TypedTensor<Self>,
            ) -> crate::Result<Vec<TypedTensor<Self>>> {
                let (m, n) = matrix_dims(input, "qr")?;
                let k = m.min(n);
                let m_i32 = dim_i32(m, "qr")?;
                let n_i32 = dim_i32(n, "qr")?;
                let k_i32 = dim_i32(k, "qr")?;

                let mut qr = input.host_data().to_vec();
                let mut tau = vec![0.0 as $scalar; k];
                let mut query = vec![0.0 as $scalar; 1];
                let mut info = 0;
                unsafe {
                    $geqrf(
                        m_i32, n_i32, &mut qr, m_i32, &mut tau, &mut query, -1, &mut info,
                    );
                }
                check_lapack_info("qr", concat!($geqrf_name, "(work query)"), info)?;
                let lwork = work_len(query[0] as f64, "qr", $geqrf_name)?;
                let mut work = vec![0.0 as $scalar; lwork as usize];
                unsafe {
                    $geqrf(
                        m_i32, n_i32, &mut qr, m_i32, &mut tau, &mut work, lwork, &mut info,
                    );
                }
                check_lapack_info("qr", $geqrf_name, info)?;

                let r = leading_upper_triangle_from_lapack(&qr, m, k, n);
                let mut q = Vec::with_capacity(m * k);
                for col in 0..k {
                    let start = col * m;
                    q.extend_from_slice(&qr[start..start + m]);
                }

                let mut query = vec![0.0 as $scalar; 1];
                unsafe {
                    $orgqr(
                        m_i32, k_i32, k_i32, &mut q, m_i32, &tau, &mut query, -1, &mut info,
                    );
                }
                check_lapack_info("qr", concat!($orgqr_name, "(work query)"), info)?;
                let lwork = work_len(query[0] as f64, "qr", $orgqr_name)?;
                let mut work = vec![0.0 as $scalar; lwork as usize];
                unsafe {
                    $orgqr(
                        m_i32, k_i32, k_i32, &mut q, m_i32, &tau, &mut work, lwork, &mut info,
                    );
                }
                check_lapack_info("qr", $orgqr_name, info)?;

                Ok(vec![
                    tensor_from_vec_with_template(vec![m, k], q, input),
                    tensor_from_vec_with_template(vec![k, n], r, input),
                ])
            }
        }
    };
}

macro_rules! impl_complex_qr {
    ($complex:ty, $geqrf:path, $ungqr:path, $geqrf_name:literal, $ungqr_name:literal) => {
        impl LapackQr for $complex {
            fn qr_2d(
                _buffers: &mut BufferPool,
                input: &TypedTensor<Self>,
            ) -> crate::Result<Vec<TypedTensor<Self>>> {
                let (m, n) = matrix_dims(input, "qr")?;
                let k = m.min(n);
                let m_i32 = dim_i32(m, "qr")?;
                let n_i32 = dim_i32(n, "qr")?;
                let k_i32 = dim_i32(k, "qr")?;

                let mut qr = input.host_data().to_vec();
                let mut tau = vec![<$complex>::new(0.0, 0.0); k];
                let mut query = vec![<$complex>::new(0.0, 0.0); 1];
                let mut info = 0;
                unsafe {
                    $geqrf(
                        m_i32, n_i32, &mut qr, m_i32, &mut tau, &mut query, -1, &mut info,
                    );
                }
                check_lapack_info("qr", concat!($geqrf_name, "(work query)"), info)?;
                let lwork = work_len(query[0].re as f64, "qr", $geqrf_name)?;
                let mut work = vec![<$complex>::new(0.0, 0.0); lwork as usize];
                unsafe {
                    $geqrf(
                        m_i32, n_i32, &mut qr, m_i32, &mut tau, &mut work, lwork, &mut info,
                    );
                }
                check_lapack_info("qr", $geqrf_name, info)?;

                let r = leading_upper_triangle_from_lapack(&qr, m, k, n);
                let mut q = Vec::with_capacity(m * k);
                for col in 0..k {
                    let start = col * m;
                    q.extend_from_slice(&qr[start..start + m]);
                }

                let mut query = vec![<$complex>::new(0.0, 0.0); 1];
                unsafe {
                    $ungqr(
                        m_i32, k_i32, k_i32, &mut q, m_i32, &tau, &mut query, -1, &mut info,
                    );
                }
                check_lapack_info("qr", concat!($ungqr_name, "(work query)"), info)?;
                let lwork = work_len(query[0].re as f64, "qr", $ungqr_name)?;
                let mut work = vec![<$complex>::new(0.0, 0.0); lwork as usize];
                unsafe {
                    $ungqr(
                        m_i32, k_i32, k_i32, &mut q, m_i32, &tau, &mut work, lwork, &mut info,
                    );
                }
                check_lapack_info("qr", $ungqr_name, info)?;

                Ok(vec![
                    tensor_from_vec_with_template(vec![m, k], q, input),
                    tensor_from_vec_with_template(vec![k, n], r, input),
                ])
            }
        }
    };
}

impl_real_qr!(f32, lapack::sgeqrf, lapack::sorgqr, "sgeqrf", "sorgqr");
impl_real_qr!(f64, lapack::dgeqrf, lapack::dorgqr, "dgeqrf", "dorgqr");
impl_complex_qr!(
    Complex32,
    lapack::cgeqrf,
    lapack::cungqr,
    "cgeqrf",
    "cungqr"
);
impl_complex_qr!(
    Complex64,
    lapack::zgeqrf,
    lapack::zungqr,
    "zgeqrf",
    "zungqr"
);

fn qr_2d<T: LapackQr>(
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> crate::Result<Vec<TypedTensor<T>>> {
    T::qr_2d(buffers, input)
}

pub(crate) fn qr<T: LapackQr>(
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> crate::Result<Vec<TypedTensor<T>>> {
    if has_zero_dim(&input.shape) {
        let (matrix_shape, batch_shape) = split_core_and_batch_result(input, 2, "qr")?;
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
                matrix_with_batch_shape(k, n, batch_shape),
                Vec::new(),
                input,
            ),
        ]);
    }
    batched_multi("qr", buffers, input, qr_2d)
}
