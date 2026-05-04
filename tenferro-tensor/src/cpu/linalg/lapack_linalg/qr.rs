use num_complex::Complex64;

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

impl LapackQr for f64 {
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
        let mut tau = vec![0.0; k];
        let mut query = vec![0.0; 1];
        let mut info = 0;
        unsafe {
            lapack::dgeqrf(
                m_i32, n_i32, &mut qr, m_i32, &mut tau, &mut query, -1, &mut info,
            );
        }
        check_lapack_info("qr", "dgeqrf(work query)", info)?;
        let lwork = work_len(query[0], "qr", "dgeqrf")?;
        let mut work = vec![0.0; lwork as usize];
        unsafe {
            lapack::dgeqrf(
                m_i32, n_i32, &mut qr, m_i32, &mut tau, &mut work, lwork, &mut info,
            );
        }
        check_lapack_info("qr", "dgeqrf", info)?;

        let r = leading_upper_triangle_from_lapack(&qr, m, k, n);
        let mut q = Vec::with_capacity(m * k);
        for col in 0..k {
            let start = col * m;
            q.extend_from_slice(&qr[start..start + m]);
        }

        let mut query = vec![0.0; 1];
        unsafe {
            lapack::dorgqr(
                m_i32, k_i32, k_i32, &mut q, m_i32, &tau, &mut query, -1, &mut info,
            );
        }
        check_lapack_info("qr", "dorgqr(work query)", info)?;
        let lwork = work_len(query[0], "qr", "dorgqr")?;
        let mut work = vec![0.0; lwork as usize];
        unsafe {
            lapack::dorgqr(
                m_i32, k_i32, k_i32, &mut q, m_i32, &tau, &mut work, lwork, &mut info,
            );
        }
        check_lapack_info("qr", "dorgqr", info)?;

        Ok(vec![
            tensor_from_vec_with_template(vec![m, k], q, input),
            tensor_from_vec_with_template(vec![k, n], r, input),
        ])
    }
}

impl LapackQr for Complex64 {
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
        let mut tau = vec![Complex64::new(0.0, 0.0); k];
        let mut query = vec![Complex64::new(0.0, 0.0); 1];
        let mut info = 0;
        unsafe {
            lapack::zgeqrf(
                m_i32, n_i32, &mut qr, m_i32, &mut tau, &mut query, -1, &mut info,
            );
        }
        check_lapack_info("qr", "zgeqrf(work query)", info)?;
        let lwork = work_len(query[0].re, "qr", "zgeqrf")?;
        let mut work = vec![Complex64::new(0.0, 0.0); lwork as usize];
        unsafe {
            lapack::zgeqrf(
                m_i32, n_i32, &mut qr, m_i32, &mut tau, &mut work, lwork, &mut info,
            );
        }
        check_lapack_info("qr", "zgeqrf", info)?;

        let r = leading_upper_triangle_from_lapack(&qr, m, k, n);
        let mut q = Vec::with_capacity(m * k);
        for col in 0..k {
            let start = col * m;
            q.extend_from_slice(&qr[start..start + m]);
        }

        let mut query = vec![Complex64::new(0.0, 0.0); 1];
        unsafe {
            lapack::zungqr(
                m_i32, k_i32, k_i32, &mut q, m_i32, &tau, &mut query, -1, &mut info,
            );
        }
        check_lapack_info("qr", "zungqr(work query)", info)?;
        let lwork = work_len(query[0].re, "qr", "zungqr")?;
        let mut work = vec![Complex64::new(0.0, 0.0); lwork as usize];
        unsafe {
            lapack::zungqr(
                m_i32, k_i32, k_i32, &mut q, m_i32, &tau, &mut work, lwork, &mut info,
            );
        }
        check_lapack_info("qr", "zungqr", info)?;

        Ok(vec![
            tensor_from_vec_with_template(vec![m, k], q, input),
            tensor_from_vec_with_template(vec![k, n], r, input),
        ])
    }
}

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
