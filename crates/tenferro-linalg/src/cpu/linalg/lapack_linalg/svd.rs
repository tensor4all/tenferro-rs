use num_complex::{Complex32, Complex64};

use tenferro_cpu::linalg_interop::{BufferPool, PoolScalar};
use tenferro_tensor::TypedTensor;

use super::helpers::{
    batched_multi, batched_multi_convert, check_lapack_info, checked_product, dim_i32,
    has_zero_dim, matrix_dims, matrix_with_batch_shape, split_core_and_batch_result,
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

#[cfg(not(feature = "provider-inject"))]
fn gesdd_iwork_len(k: usize) -> tenferro_tensor::Result<usize> {
    checked_product("svd", "integer workspace", &[8, k.max(1)])
}

#[cfg(not(feature = "provider-inject"))]
fn complex_gesdd_rwork_len(jobz: u8, m: usize, n: usize) -> tenferro_tensor::Result<usize> {
    let mn = m.min(n);
    let mx = m.max(n);
    if jobz == b'N' {
        return checked_product("svd", "real workspace", &[5, mn.max(1)]);
    }
    let threshold = checked_product("svd", "workspace crossover", &[10, mn])?;
    let square_term = checked_product("svd", "real workspace square term", &[5, mn, mn])?;
    let linear_term = checked_product("svd", "real workspace linear term", &[5, mn])?;
    let small_shape_len = square_term.checked_add(linear_term).ok_or_else(|| {
        tenferro_tensor::Error::validation("svd", tenferro_tensor::ValidationError::IntegerOverflow)
    })?;
    if mx > threshold {
        return Ok(small_shape_len);
    }
    let rectangular_term = checked_product("svd", "real workspace rectangular term", &[2, mx, mn])?;
    let second_square_term =
        checked_product("svd", "real workspace secondary square term", &[2, mn, mn])?;
    let large_shape_len = rectangular_term
        .checked_add(second_square_term)
        .and_then(|len| len.checked_add(mn))
        .ok_or_else(|| {
            tenferro_tensor::Error::validation(
                "svd",
                tenferro_tensor::ValidationError::IntegerOverflow,
            )
        })?;
    Ok(small_shape_len.max(large_shape_len))
}

#[cfg(feature = "provider-inject")]
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
                let u_len = checked_product("svd", "left singular vectors", &[m, k])?;
                let mut u = vec![0.0 as $scalar; u_len];
                let vt_len = checked_product("svd", "right singular vectors", &[k, n])?;
                let mut vt = vec![0.0 as $scalar; vt_len];
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

#[cfg(not(feature = "provider-inject"))]
macro_rules! impl_real_svd {
    ($scalar:ty, $gesdd:path, $routine:literal) => {
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
                let u_len = checked_product("svd", "left singular vectors", &[m, k])?;
                let mut u = vec![0.0 as $scalar; u_len];
                let vt_len = checked_product("svd", "right singular vectors", &[k, n])?;
                let mut vt = vec![0.0 as $scalar; vt_len];
                let mut query = vec![0.0 as $scalar; 1];
                let mut iwork = vec![0; gesdd_iwork_len(k)?];
                let mut info = 0;
                // SAFETY: `a`, `s`, `u`, and `vt` match the validated SVD
                // dimensions, and `lwork = -1` makes `query` the workspace output.
                unsafe {
                    $gesdd(
                        b'S', m_i32, n_i32, &mut a, m_i32, &mut s, &mut u, m_i32, &mut vt, k_i32,
                        &mut query, -1, &mut iwork, &mut info,
                    );
                }
                check_lapack_info("svd", concat!($routine, "(work query)"), info)?;
                let lwork = work_len(query[0] as f64, "svd", $routine)?;
                let mut work = vec![0.0 as $scalar; lwork as usize];
                // SAFETY: buffers and leading dimensions match the validated
                // SVD problem, and `work` uses the queried workspace length.
                unsafe {
                    $gesdd(
                        b'S', m_i32, n_i32, &mut a, m_i32, &mut s, &mut u, m_i32, &mut vt, k_i32,
                        &mut work, lwork, &mut iwork, &mut info,
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
                let mut iwork = vec![0; gesdd_iwork_len(k)?];
                let mut info = 0;
                // SAFETY: `a` and `s` match the validated SVD dimensions;
                // no-vector mode ignores the dummy U/VT buffers, and `lwork = -1` queries workspace.
                unsafe {
                    $gesdd(
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
                        &mut iwork,
                        &mut info,
                    );
                }
                check_lapack_info("svd_values", concat!($routine, "(work query)"), info)?;
                let lwork = work_len(query[0] as f64, "svd_values", $routine)?;
                let mut work = vec![0.0 as $scalar; lwork as usize];
                // SAFETY: `a`, `s`, dummy no-vector buffers, and `work`
                // satisfy the validated SVD dimensions and queried workspace length.
                unsafe {
                    $gesdd(
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
                        &mut iwork,
                        &mut info,
                    );
                }
                check_lapack_info("svd_values", $routine, info)?;

                tensor_from_vec_with_template(vec![k], s, input)
            }
        }
    };
}

#[cfg(feature = "provider-inject")]
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
                let u_len = checked_product("svd", "left singular vectors", &[m, k])?;
                let mut u = vec![<$complex>::new(0.0, 0.0); u_len];
                let vt_len = checked_product("svd", "right singular vectors", &[k, n])?;
                let mut vt = vec![<$complex>::new(0.0, 0.0); vt_len];
                let mut query = vec![<$complex>::new(0.0, 0.0); 1];
                let rwork_len = checked_product("svd", "real workspace", &[5, k.max(1)])?;
                let mut rwork = vec![0.0 as $real; rwork_len];
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
                let rwork_len = checked_product("svd", "real workspace", &[5, k.max(1)])?;
                let mut rwork = vec![0.0 as $real; rwork_len];
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

#[cfg(not(feature = "provider-inject"))]
macro_rules! impl_complex_svd {
    ($complex:ty, $real:ty, $gesdd:path, $routine:literal) => {
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
                let u_len = checked_product("svd", "left singular vectors", &[m, k])?;
                let mut u = vec![<$complex>::new(0.0, 0.0); u_len];
                let vt_len = checked_product("svd", "right singular vectors", &[k, n])?;
                let mut vt = vec![<$complex>::new(0.0, 0.0); vt_len];
                let mut query = vec![<$complex>::new(0.0, 0.0); 1];
                let mut rwork = vec![0.0 as $real; complex_gesdd_rwork_len(b'S', m, n)?];
                let mut iwork = vec![0; gesdd_iwork_len(k)?];
                let mut info = 0;
                // SAFETY: `a`, `s`, `u`, `vt`, and `rwork` match the
                // validated complex SVD dimensions; `lwork = -1` queries workspace.
                unsafe {
                    $gesdd(
                        b'S', m_i32, n_i32, &mut a, m_i32, &mut s, &mut u, m_i32, &mut vt, k_i32,
                        &mut query, -1, &mut rwork, &mut iwork, &mut info,
                    );
                }
                check_lapack_info("svd", concat!($routine, "(work query)"), info)?;
                let lwork = work_len(query[0].re as f64, "svd", $routine)?;
                let mut work = vec![<$complex>::new(0.0, 0.0); lwork as usize];
                // SAFETY: buffers, real workspace, and leading dimensions
                // match the validated complex SVD problem and queried workspace length.
                unsafe {
                    $gesdd(
                        b'S', m_i32, n_i32, &mut a, m_i32, &mut s, &mut u, m_i32, &mut vt, k_i32,
                        &mut work, lwork, &mut rwork, &mut iwork, &mut info,
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
                let mut rwork = vec![0.0 as $real; complex_gesdd_rwork_len(b'N', m, n)?];
                let mut iwork = vec![0; gesdd_iwork_len(k)?];
                let mut info = 0;
                // SAFETY: `a`, `s`, and `rwork` match the validated complex
                // SVD dimensions; no-vector mode ignores dummy U/VT buffers, and `lwork = -1` queries workspace.
                unsafe {
                    $gesdd(
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
                        &mut iwork,
                        &mut info,
                    );
                }
                check_lapack_info("svd_values", concat!($routine, "(work query)"), info)?;
                let lwork = work_len(query[0].re as f64, "svd_values", $routine)?;
                let mut work = vec![<$complex>::new(0.0, 0.0); lwork as usize];
                // SAFETY: `a`, `s`, dummy no-vector buffers, `work`, and
                // `rwork` satisfy the validated complex SVD dimensions and queried workspace length.
                unsafe {
                    $gesdd(
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
                        &mut iwork,
                        &mut info,
                    );
                }
                check_lapack_info("svd_values", $routine, info)?;

                tensor_from_vec_with_template(vec![k], s, input)
            }
        }
    };
}

#[cfg(not(feature = "provider-inject"))]
impl_real_svd!(f32, lapack::sgesdd, "sgesdd");
#[cfg(not(feature = "provider-inject"))]
impl_real_svd!(f64, lapack::dgesdd, "dgesdd");
#[cfg(not(feature = "provider-inject"))]
impl_complex_svd!(Complex32, f32, lapack::cgesdd, "cgesdd");
#[cfg(not(feature = "provider-inject"))]
impl_complex_svd!(Complex64, f64, lapack::zgesdd, "zgesdd");

#[cfg(feature = "provider-inject")]
impl_real_svd!(f32, lapack::sgesvd, "sgesvd");
#[cfg(feature = "provider-inject")]
impl_real_svd!(f64, lapack::dgesvd, "dgesvd");
#[cfg(feature = "provider-inject")]
impl_complex_svd!(Complex32, f32, lapack::cgesvd, "cgesvd");
#[cfg(feature = "provider-inject")]
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

    let mut outputs =
        batched_multi_convert("svd_values", buffers, input, |buffers, batch_input| {
            Ok(vec![svd_values_2d(buffers, batch_input)?])
        })?;
    match outputs.pop() {
        Some(values) if outputs.is_empty() => Ok(values),
        _ => Err(tenferro_tensor::Error::Internal(
            "svd_values: expected exactly one output from batched singular-value helper".into(),
        )),
    }
}
