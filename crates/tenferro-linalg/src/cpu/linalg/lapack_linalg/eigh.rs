use num_complex::{Complex32, Complex64};

use tenferro_cpu::linalg_interop::{BufferPool, PoolScalar};
use tenferro_tensor::TypedTensor;

use super::helpers::{
    batched_multi, batched_multi_convert, check_lapack_info, dim_i32, has_zero_dim,
    matrix_with_batch_shape, square_core_and_batch_result, square_matrix_dim,
    tensor_from_vec_with_template, vector_with_batch_shape, work_len,
};

pub(crate) trait LapackEigh: Clone + Copy + Default + PoolScalar {
    type Real: Clone + Copy + Default;

    fn eigh_2d(
        buffers: &mut BufferPool,
        input: &TypedTensor<Self>,
    ) -> tenferro_tensor::Result<Vec<TypedTensor<Self>>>;
    fn eigh_values_2d(
        buffers: &mut BufferPool,
        input: &TypedTensor<Self>,
    ) -> tenferro_tensor::Result<TypedTensor<Self::Real>>;
}

fn iwork_len(query: i32, op: &'static str, routine: &'static str) -> tenferro_tensor::Result<i32> {
    if query < 1 {
        return Err(tenferro_tensor::Error::backend_failure(
            op,
            format!("LAPACK {routine} returned invalid integer workspace size {query}"),
        ));
    }
    Ok(query)
}

fn queried_iwork_len(
    query: &[i32],
    op: &'static str,
    routine: &'static str,
) -> tenferro_tensor::Result<i32> {
    let query = query.first().copied().ok_or_else(|| {
        tenferro_tensor::Error::backend_failure(
            op,
            format!("LAPACK {routine} did not return an integer workspace size"),
        )
    })?;
    iwork_len(query, op, routine)
}

fn iwork_capacity(
    len: i32,
    op: &'static str,
    routine: &'static str,
) -> tenferro_tensor::Result<usize> {
    usize::try_from(len).map_err(|_| {
        tenferro_tensor::Error::backend_failure(
            op,
            format!("LAPACK {routine} integer workspace size {len} does not fit usize"),
        )
    })
}

macro_rules! impl_real_eigh {
    ($scalar:ty, $syevd:path, $routine:literal) => {
        impl LapackEigh for $scalar {
            type Real = $scalar;

            fn eigh_2d(
                _buffers: &mut BufferPool,
                input: &TypedTensor<Self>,
            ) -> tenferro_tensor::Result<Vec<TypedTensor<Self>>> {
                let n = square_matrix_dim(input, "eigh")?;
                let n_i32 = dim_i32(n, "eigh")?;
                let mut vectors = input.host_data()?.to_vec();
                let mut values = vec![0.0 as $scalar; n];
                let mut query = vec![0.0 as $scalar; 1];
                let mut iquery = vec![0; 1];
                let mut info = 0;
                // SAFETY: `vectors` is a mutable column-major `n x n` buffer,
                // `values` has `n` entries, and workspace queries write only
                // the first `work` and `iwork` slots.
                unsafe {
                    $syevd(
                        b'V',
                        b'L',
                        n_i32,
                        &mut vectors,
                        n_i32,
                        &mut values,
                        &mut query,
                        -1,
                        &mut iquery,
                        -1,
                        &mut info,
                    );
                }
                check_lapack_info("eigh", concat!($routine, "(work query)"), info)?;
                let lwork = work_len(query[0] as f64, "eigh", $routine)?;
                let liwork = queried_iwork_len(&iquery, "eigh", $routine)?;
                let liwork_capacity = iwork_capacity(liwork, "eigh", $routine)?;
                let mut work = vec![0.0 as $scalar; lwork as usize];
                let mut iwork = vec![0; liwork_capacity];
                // SAFETY: dimensions and workspace lengths come from validated
                // shape metadata plus the LAPACK query; all mutable buffers are live.
                unsafe {
                    $syevd(
                        b'V',
                        b'L',
                        n_i32,
                        &mut vectors,
                        n_i32,
                        &mut values,
                        &mut work,
                        lwork,
                        &mut iwork,
                        liwork,
                        &mut info,
                    );
                }
                check_lapack_info("eigh", $routine, info)?;

                Ok(vec![
                    tensor_from_vec_with_template(vec![n], values, input)?,
                    tensor_from_vec_with_template(vec![n, n], vectors, input)?,
                ])
            }

            fn eigh_values_2d(
                _buffers: &mut BufferPool,
                input: &TypedTensor<Self>,
            ) -> tenferro_tensor::Result<TypedTensor<Self::Real>> {
                let n = square_matrix_dim(input, "eigh_values")?;
                let n_i32 = dim_i32(n, "eigh_values")?;
                let mut work_matrix = input.host_data()?.to_vec();
                let mut values = vec![0.0 as $scalar; n];
                let mut query = vec![0.0 as $scalar; 1];
                let mut iquery = vec![0; 1];
                let mut info = 0;
                // SAFETY: `work_matrix` is a mutable column-major `n x n`
                // buffer, `values` has `n` entries, and workspace queries write
                // only the first `work` and `iwork` slots.
                unsafe {
                    $syevd(
                        b'N',
                        b'L',
                        n_i32,
                        &mut work_matrix,
                        n_i32,
                        &mut values,
                        &mut query,
                        -1,
                        &mut iquery,
                        -1,
                        &mut info,
                    );
                }
                check_lapack_info("eigh_values", concat!($routine, "(work query)"), info)?;
                let lwork = work_len(query[0] as f64, "eigh_values", $routine)?;
                let liwork = queried_iwork_len(&iquery, "eigh_values", $routine)?;
                let liwork_capacity = iwork_capacity(liwork, "eigh_values", $routine)?;
                let mut work = vec![0.0 as $scalar; lwork as usize];
                let mut iwork = vec![0; liwork_capacity];
                // SAFETY: dimensions and workspace lengths come from validated
                // shape metadata plus the LAPACK query; all mutable buffers are live.
                unsafe {
                    $syevd(
                        b'N',
                        b'L',
                        n_i32,
                        &mut work_matrix,
                        n_i32,
                        &mut values,
                        &mut work,
                        lwork,
                        &mut iwork,
                        liwork,
                        &mut info,
                    );
                }
                check_lapack_info("eigh_values", $routine, info)?;

                tensor_from_vec_with_template(vec![n], values, input)
            }
        }
    };
}

macro_rules! impl_complex_eigh {
    ($complex:ty, $real:ty, $heev:path, $routine:literal) => {
        impl LapackEigh for $complex {
            type Real = $real;

            fn eigh_2d(
                _buffers: &mut BufferPool,
                input: &TypedTensor<Self>,
            ) -> tenferro_tensor::Result<Vec<TypedTensor<Self>>> {
                let n = square_matrix_dim(input, "eigh")?;
                let n_i32 = dim_i32(n, "eigh")?;
                let mut vectors = input.host_data()?.to_vec();
                let mut values = vec![0.0 as $real; n];
                let mut query = vec![<$complex>::new(0.0, 0.0); 1];
                let mut rwork = vec![0.0 as $real; (3 * n).saturating_sub(2).max(1)];
                let mut info = 0;
                // SAFETY: `vectors`, `values`, and `rwork` satisfy LAPACK's
                // Hermitian eigensolver dimensions; `lwork = -1` writes only `query`.
                unsafe {
                    $heev(
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
                check_lapack_info("eigh", concat!($routine, "(work query)"), info)?;
                let lwork = work_len(query[0].re as f64, "eigh", $routine)?;
                let mut work = vec![<$complex>::new(0.0, 0.0); lwork as usize];
                // SAFETY: `vectors`, `values`, `work`, and `rwork` match the
                // validated `n x n` problem and queried workspace length.
                unsafe {
                    $heev(
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
                check_lapack_info("eigh", $routine, info)?;

                Ok(vec![
                    tensor_from_vec_with_template(
                        vec![n],
                        values
                            .into_iter()
                            .map(|value| <$complex>::new(value, 0.0))
                            .collect(),
                        input,
                    )?,
                    tensor_from_vec_with_template(vec![n, n], vectors, input)?,
                ])
            }

            fn eigh_values_2d(
                _buffers: &mut BufferPool,
                input: &TypedTensor<Self>,
            ) -> tenferro_tensor::Result<TypedTensor<Self::Real>> {
                let n = square_matrix_dim(input, "eigh_values")?;
                let n_i32 = dim_i32(n, "eigh_values")?;
                let mut work_matrix = input.host_data()?.to_vec();
                let mut values = vec![0.0 as $real; n];
                let mut query = vec![<$complex>::new(0.0, 0.0); 1];
                let mut rwork = vec![0.0 as $real; (3 * n).saturating_sub(2).max(1)];
                let mut info = 0;
                // SAFETY: `work_matrix`, `values`, and `rwork` satisfy LAPACK's
                // Hermitian eigensolver dimensions; `lwork = -1` writes only `query`.
                unsafe {
                    $heev(
                        b'N',
                        b'L',
                        n_i32,
                        &mut work_matrix,
                        n_i32,
                        &mut values,
                        &mut query,
                        -1,
                        &mut rwork,
                        &mut info,
                    );
                }
                check_lapack_info("eigh_values", concat!($routine, "(work query)"), info)?;
                let lwork = work_len(query[0].re as f64, "eigh_values", $routine)?;
                let mut work = vec![<$complex>::new(0.0, 0.0); lwork as usize];
                // SAFETY: `work_matrix`, `values`, `work`, and `rwork` match
                // the validated `n x n` problem and queried workspace length.
                unsafe {
                    $heev(
                        b'N',
                        b'L',
                        n_i32,
                        &mut work_matrix,
                        n_i32,
                        &mut values,
                        &mut work,
                        lwork,
                        &mut rwork,
                        &mut info,
                    );
                }
                check_lapack_info("eigh_values", $routine, info)?;

                tensor_from_vec_with_template(vec![n], values, input)
            }
        }
    };
}

impl_real_eigh!(f32, lapack::ssyevd, "ssyevd");
impl_real_eigh!(f64, lapack::dsyevd, "dsyevd");
impl_complex_eigh!(Complex32, f32, lapack::cheev, "cheev");
impl_complex_eigh!(Complex64, f64, lapack::zheev, "zheev");

fn eigh_2d<T: LapackEigh>(
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> tenferro_tensor::Result<Vec<TypedTensor<T>>> {
    T::eigh_2d(buffers, input)
}

pub(crate) fn eigh<T: LapackEigh>(
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> tenferro_tensor::Result<Vec<TypedTensor<T>>> {
    if has_zero_dim(input.shape()) {
        let (n, batch_shape) = square_core_and_batch_result(input, "eigh")?;
        return Ok(vec![
            tensor_from_vec_with_template(
                vector_with_batch_shape(n, batch_shape),
                Vec::new(),
                input,
            )?,
            tensor_from_vec_with_template(
                matrix_with_batch_shape(n, n, batch_shape),
                Vec::new(),
                input,
            )?,
        ]);
    }
    batched_multi("eigh", buffers, input, eigh_2d)
}

fn eigh_values_2d<T: LapackEigh>(
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> tenferro_tensor::Result<TypedTensor<T::Real>> {
    T::eigh_values_2d(buffers, input)
}

pub(crate) fn eigh_values<T: LapackEigh>(
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> tenferro_tensor::Result<TypedTensor<T::Real>> {
    if has_zero_dim(input.shape()) {
        let (n, batch_shape) = square_core_and_batch_result(input, "eigh_values")?;
        return tensor_from_vec_with_template(
            vector_with_batch_shape(n, batch_shape),
            Vec::new(),
            input,
        );
    }

    let mut outputs =
        batched_multi_convert("eigh_values", buffers, input, |buffers, batch_input| {
            Ok(vec![eigh_values_2d(buffers, batch_input)?])
        })?;
    match outputs.pop() {
        Some(values) if outputs.is_empty() => Ok(values),
        _ => Err(tenferro_tensor::Error::InvalidConfig {
            op: "eigh_values",
            message: "expected exactly one output from batched eigenvalue helper".into(),
        }),
    }
}
