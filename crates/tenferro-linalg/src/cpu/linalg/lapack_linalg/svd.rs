use num_complex::{Complex32, Complex64};

use tenferro_cpu::linalg_interop::{BufferPool, PoolScalar};
use tenferro_tensor::TypedTensor;

use super::helpers::{
    batch_element_count, batched_multi, check_lapack_info, checked_product, checked_slice_range,
    dim_i32, has_zero_dim, matrix_dims, matrix_with_batch_shape, split_core_and_batch_result,
    tensor_from_vec_with_template, vector_with_batch_shape, work_len,
};

pub(crate) trait LapackSvd: Clone + Copy + Default + PoolScalar {
    type Real: Clone + Copy + Default;

    fn svd_2d(
        buffers: &mut BufferPool,
        input: &TypedTensor<Self>,
    ) -> tenferro_tensor::Result<Vec<TypedTensor<Self>>>;
    fn svd_values_2d(
        buffers: &mut BufferPool,
        input: &TypedTensor<Self>,
    ) -> tenferro_tensor::Result<TypedTensor<Self::Real>>;
}

macro_rules! impl_real_svd {
    ($scalar:ty, $gesvd:path, $routine:literal) => {
        impl LapackSvd for $scalar {
            type Real = $scalar;

            fn svd_2d(
                _buffers: &mut BufferPool,
                input: &TypedTensor<Self>,
            ) -> tenferro_tensor::Result<Vec<TypedTensor<Self>>> {
                let (m, n) = matrix_dims(input, "svd")?;
                let k = m.min(n);
                let m_i32 = dim_i32(m, "svd")?;
                let n_i32 = dim_i32(n, "svd")?;
                let k_i32 = dim_i32(k, "svd")?;

                let mut a = input.host_data()?.to_vec();
                let mut s = vec![0.0 as $scalar; k];
                let mut u = vec![0.0 as $scalar; m * k];
                let mut vt = vec![0.0 as $scalar; k * n];
                let mut query = vec![0.0 as $scalar; 1];
                let mut info = 0;
                // SAFETY: `a`, `s`, `u`, and `vt` match the validated SVD
                // dimensions, and `lwork = -1` makes `query` the workspace output.
                unsafe {
                    $gesvd(
                        b'S', b'S', m_i32, n_i32, &mut a, m_i32, &mut s, &mut u, m_i32, &mut vt,
                        k_i32, &mut query, -1, &mut info,
                    );
                }
                check_lapack_info("svd", concat!($routine, "(work query)"), info)?;
                let lwork = work_len(query[0] as f64, "svd", $routine)?;
                let mut work = vec![0.0 as $scalar; lwork as usize];
                // SAFETY: buffers and leading dimensions match the validated
                // SVD problem, and `work` uses the queried workspace length.
                unsafe {
                    $gesvd(
                        b'S', b'S', m_i32, n_i32, &mut a, m_i32, &mut s, &mut u, m_i32, &mut vt,
                        k_i32, &mut work, lwork, &mut info,
                    );
                }
                check_lapack_info("svd", $routine, info)?;

                Ok(vec![
                    tensor_from_vec_with_template(vec![m, k], u, input)?,
                    tensor_from_vec_with_template(vec![k], s, input)?,
                    tensor_from_vec_with_template(vec![k, n], vt, input)?,
                ])
            }

            fn svd_values_2d(
                _buffers: &mut BufferPool,
                input: &TypedTensor<Self>,
            ) -> tenferro_tensor::Result<TypedTensor<Self::Real>> {
                let (m, n) = matrix_dims(input, "svd_values")?;
                let k = m.min(n);
                let m_i32 = dim_i32(m, "svd_values")?;
                let n_i32 = dim_i32(n, "svd_values")?;

                let mut a = input.host_data()?.to_vec();
                let mut s = vec![0.0 as $scalar; k];
                let mut query = vec![0.0 as $scalar; 1];
                let mut info = 0;
                // SAFETY: `a` and `s` match the validated SVD dimensions;
                // no-vector mode ignores the dummy U/VT buffers, and `lwork = -1` queries workspace.
                unsafe {
                    $gesvd(
                        b'N',
                        b'N',
                        m_i32,
                        n_i32,
                        &mut a,
                        m_i32,
                        &mut s,
                        &mut [],
                        1,
                        &mut [],
                        1,
                        &mut query,
                        -1,
                        &mut info,
                    );
                }
                check_lapack_info("svd_values", concat!($routine, "(work query)"), info)?;
                let lwork = work_len(query[0] as f64, "svd_values", $routine)?;
                let mut work = vec![0.0 as $scalar; lwork as usize];
                // SAFETY: `a`, `s`, dummy no-vector buffers, and `work`
                // satisfy the validated SVD dimensions and queried workspace length.
                unsafe {
                    $gesvd(
                        b'N',
                        b'N',
                        m_i32,
                        n_i32,
                        &mut a,
                        m_i32,
                        &mut s,
                        &mut [],
                        1,
                        &mut [],
                        1,
                        &mut work,
                        lwork,
                        &mut info,
                    );
                }
                check_lapack_info("svd_values", $routine, info)?;

                tensor_from_vec_with_template(vec![k], s, input)
            }
        }
    };
}

macro_rules! impl_complex_svd {
    ($complex:ty, $real:ty, $gesvd:path, $routine:literal) => {
        impl LapackSvd for $complex {
            type Real = $real;

            fn svd_2d(
                _buffers: &mut BufferPool,
                input: &TypedTensor<Self>,
            ) -> tenferro_tensor::Result<Vec<TypedTensor<Self>>> {
                let (m, n) = matrix_dims(input, "svd")?;
                let k = m.min(n);
                let m_i32 = dim_i32(m, "svd")?;
                let n_i32 = dim_i32(n, "svd")?;
                let k_i32 = dim_i32(k, "svd")?;

                let mut a = input.host_data()?.to_vec();
                let mut s = vec![0.0 as $real; k];
                let mut u = vec![<$complex>::new(0.0, 0.0); m * k];
                let mut vt = vec![<$complex>::new(0.0, 0.0); k * n];
                let mut query = vec![<$complex>::new(0.0, 0.0); 1];
                let mut rwork = vec![0.0 as $real; 5 * k.max(1)];
                let mut info = 0;
                // SAFETY: `a`, `s`, `u`, `vt`, and `rwork` match the
                // validated complex SVD dimensions; `lwork = -1` queries workspace.
                unsafe {
                    $gesvd(
                        b'S', b'S', m_i32, n_i32, &mut a, m_i32, &mut s, &mut u, m_i32, &mut vt,
                        k_i32, &mut query, -1, &mut rwork, &mut info,
                    );
                }
                check_lapack_info("svd", concat!($routine, "(work query)"), info)?;
                let lwork = work_len(query[0].re as f64, "svd", $routine)?;
                let mut work = vec![<$complex>::new(0.0, 0.0); lwork as usize];
                // SAFETY: buffers, real workspace, and leading dimensions
                // match the validated complex SVD problem and queried workspace length.
                unsafe {
                    $gesvd(
                        b'S', b'S', m_i32, n_i32, &mut a, m_i32, &mut s, &mut u, m_i32, &mut vt,
                        k_i32, &mut work, lwork, &mut rwork, &mut info,
                    );
                }
                check_lapack_info("svd", $routine, info)?;

                Ok(vec![
                    tensor_from_vec_with_template(vec![m, k], u, input)?,
                    tensor_from_vec_with_template(
                        vec![k],
                        s.into_iter()
                            .map(|value| <$complex>::new(value, 0.0))
                            .collect(),
                        input,
                    )?,
                    tensor_from_vec_with_template(vec![k, n], vt, input)?,
                ])
            }

            fn svd_values_2d(
                _buffers: &mut BufferPool,
                input: &TypedTensor<Self>,
            ) -> tenferro_tensor::Result<TypedTensor<Self::Real>> {
                let (m, n) = matrix_dims(input, "svd_values")?;
                let k = m.min(n);
                let m_i32 = dim_i32(m, "svd_values")?;
                let n_i32 = dim_i32(n, "svd_values")?;

                let mut a = input.host_data()?.to_vec();
                let mut s = vec![0.0 as $real; k];
                let mut query = vec![<$complex>::new(0.0, 0.0); 1];
                let mut rwork = vec![0.0 as $real; 5 * k.max(1)];
                let mut info = 0;
                // SAFETY: `a`, `s`, and `rwork` match the validated complex
                // SVD dimensions; no-vector mode ignores dummy U/VT buffers, and `lwork = -1` queries workspace.
                unsafe {
                    $gesvd(
                        b'N',
                        b'N',
                        m_i32,
                        n_i32,
                        &mut a,
                        m_i32,
                        &mut s,
                        &mut [],
                        1,
                        &mut [],
                        1,
                        &mut query,
                        -1,
                        &mut rwork,
                        &mut info,
                    );
                }
                check_lapack_info("svd_values", concat!($routine, "(work query)"), info)?;
                let lwork = work_len(query[0].re as f64, "svd_values", $routine)?;
                let mut work = vec![<$complex>::new(0.0, 0.0); lwork as usize];
                // SAFETY: `a`, `s`, dummy no-vector buffers, `work`, and
                // `rwork` satisfy the validated complex SVD dimensions and queried workspace length.
                unsafe {
                    $gesvd(
                        b'N',
                        b'N',
                        m_i32,
                        n_i32,
                        &mut a,
                        m_i32,
                        &mut s,
                        &mut [],
                        1,
                        &mut [],
                        1,
                        &mut work,
                        lwork,
                        &mut rwork,
                        &mut info,
                    );
                }
                check_lapack_info("svd_values", $routine, info)?;

                tensor_from_vec_with_template(vec![k], s, input)
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
) -> tenferro_tensor::Result<Vec<TypedTensor<T>>> {
    T::svd_2d(buffers, input)
}

pub(crate) fn svd<T: LapackSvd>(
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> tenferro_tensor::Result<Vec<TypedTensor<T>>> {
    if has_zero_dim(input.shape()) {
        let (matrix_shape, batch_shape) = split_core_and_batch_result(input, 2, "svd")?;
        let m = matrix_shape[0];
        let n = matrix_shape[1];
        let k = m.min(n);
        return Ok(vec![
            tensor_from_vec_with_template(
                matrix_with_batch_shape(m, k, batch_shape),
                Vec::new(),
                input,
            )?,
            tensor_from_vec_with_template(
                vector_with_batch_shape(k, batch_shape),
                Vec::new(),
                input,
            )?,
            tensor_from_vec_with_template(
                matrix_with_batch_shape(k, n, batch_shape),
                Vec::new(),
                input,
            )?,
        ]);
    }
    batched_multi("svd", buffers, input, svd_2d)
}

fn svd_values_2d<T: LapackSvd>(
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> tenferro_tensor::Result<TypedTensor<T::Real>> {
    T::svd_values_2d(buffers, input)
}

pub(crate) fn svd_values<T: LapackSvd>(
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> tenferro_tensor::Result<TypedTensor<T::Real>> {
    if has_zero_dim(input.shape()) {
        let (matrix_shape, batch_shape) = split_core_and_batch_result(input, 2, "svd_values")?;
        let k = matrix_shape[0].min(matrix_shape[1]);
        return tensor_from_vec_with_template(
            vector_with_batch_shape(k, batch_shape),
            Vec::new(),
            input,
        );
    }

    let (core_shape, batch_shape) = split_core_and_batch_result(input, 2, "svd_values")?;
    if batch_shape.is_empty() {
        return svd_values_2d(buffers, input);
    }

    let slice_size = checked_product("svd_values", "core shape", core_shape)?;
    let batch_total = batch_element_count("svd_values", batch_shape)?;
    let k = core_shape[0].min(core_shape[1]);
    let mut data = Vec::with_capacity(checked_product(
        "svd_values",
        "values output",
        &[k, batch_total],
    )?);
    for batch in 0..batch_total {
        let range = checked_slice_range("svd_values", batch, slice_size)?;
        let batch_input = tensor_from_vec_with_template(
            core_shape.to_vec(),
            input.host_data()?[range].to_vec(),
            input,
        )?;
        let values = svd_values_2d(buffers, &batch_input)?;
        data.extend_from_slice(values.host_data()?);
    }
    tensor_from_vec_with_template(vector_with_batch_shape(k, batch_shape), data, input)
}
