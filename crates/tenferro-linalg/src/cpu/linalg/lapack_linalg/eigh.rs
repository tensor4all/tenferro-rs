use num_complex::{Complex32, Complex64};

use tenferro_cpu::linalg_interop::BufferPool;
use tenferro_tensor::TypedTensor;

use super::helpers::{
    batched_multi, check_lapack_info, dim_i32, has_zero_dim, matrix_with_batch_shape,
    square_core_and_batch_result, square_matrix_dim, tensor_from_vec_with_template,
    vector_with_batch_shape, work_len,
};

pub(crate) trait LapackEigh: Clone + Copy + Default {
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

macro_rules! impl_real_eigh {
    ($scalar:ty, $syev:path, $routine:literal) => {
        impl LapackEigh for $scalar {
            type Real = $scalar;

            fn eigh_2d(
                _buffers: &mut BufferPool,
                input: &TypedTensor<Self>,
            ) -> tenferro_tensor::Result<Vec<TypedTensor<Self>>> {
                let n = square_matrix_dim(input, "eigh")?;
                let n_i32 = dim_i32(n, "eigh")?;
                let mut vectors = input.host_data().to_vec();
                let mut values = vec![0.0 as $scalar; n];
                let mut query = vec![0.0 as $scalar; 1];
                let mut info = 0;
                unsafe {
                    $syev(
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
                check_lapack_info("eigh", concat!($routine, "(work query)"), info)?;
                let lwork = work_len(query[0] as f64, "eigh", $routine)?;
                let mut work = vec![0.0 as $scalar; lwork as usize];
                unsafe {
                    $syev(
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
                check_lapack_info("eigh", $routine, info)?;

                Ok(vec![
                    tensor_from_vec_with_template(vec![n], values, input),
                    tensor_from_vec_with_template(vec![n, n], vectors, input),
                ])
            }

            fn eigh_values_2d(
                _buffers: &mut BufferPool,
                input: &TypedTensor<Self>,
            ) -> tenferro_tensor::Result<TypedTensor<Self::Real>> {
                let n = square_matrix_dim(input, "eigh_values")?;
                let n_i32 = dim_i32(n, "eigh_values")?;
                let mut work_matrix = input.host_data().to_vec();
                let mut values = vec![0.0 as $scalar; n];
                let mut query = vec![0.0 as $scalar; 1];
                let mut info = 0;
                unsafe {
                    $syev(
                        b'N',
                        b'L',
                        n_i32,
                        &mut work_matrix,
                        n_i32,
                        &mut values,
                        &mut query,
                        -1,
                        &mut info,
                    );
                }
                check_lapack_info("eigh_values", concat!($routine, "(work query)"), info)?;
                let lwork = work_len(query[0] as f64, "eigh_values", $routine)?;
                let mut work = vec![0.0 as $scalar; lwork as usize];
                unsafe {
                    $syev(
                        b'N',
                        b'L',
                        n_i32,
                        &mut work_matrix,
                        n_i32,
                        &mut values,
                        &mut work,
                        lwork,
                        &mut info,
                    );
                }
                check_lapack_info("eigh_values", $routine, info)?;

                Ok(tensor_from_vec_with_template(vec![n], values, input))
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
                let mut vectors = input.host_data().to_vec();
                let mut values = vec![0.0 as $real; n];
                let mut query = vec![<$complex>::new(0.0, 0.0); 1];
                let mut rwork = vec![0.0 as $real; (3 * n).saturating_sub(2).max(1)];
                let mut info = 0;
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
                    ),
                    tensor_from_vec_with_template(vec![n, n], vectors, input),
                ])
            }

            fn eigh_values_2d(
                _buffers: &mut BufferPool,
                input: &TypedTensor<Self>,
            ) -> tenferro_tensor::Result<TypedTensor<Self::Real>> {
                let n = square_matrix_dim(input, "eigh_values")?;
                let n_i32 = dim_i32(n, "eigh_values")?;
                let mut work_matrix = input.host_data().to_vec();
                let mut values = vec![0.0 as $real; n];
                let mut query = vec![<$complex>::new(0.0, 0.0); 1];
                let mut rwork = vec![0.0 as $real; (3 * n).saturating_sub(2).max(1)];
                let mut info = 0;
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

                Ok(tensor_from_vec_with_template(vec![n], values, input))
            }
        }
    };
}

impl_real_eigh!(f32, lapack::ssyev, "ssyev");
impl_real_eigh!(f64, lapack::dsyev, "dsyev");
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
            ),
            tensor_from_vec_with_template(
                matrix_with_batch_shape(n, n, batch_shape),
                Vec::new(),
                input,
            ),
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
        return Ok(tensor_from_vec_with_template(
            vector_with_batch_shape(n, batch_shape),
            Vec::new(),
            input,
        ));
    }

    let (core_shape, batch_shape) =
        super::helpers::split_core_and_batch_result(input, 2, "eigh_values")?;
    if batch_shape.is_empty() {
        return eigh_values_2d(buffers, input);
    }

    let n = core_shape[0];
    let slice_size = n * n;
    let batch_total: usize = batch_shape.iter().product();
    let mut data = Vec::with_capacity(n * batch_total);
    for batch in 0..batch_total {
        let start = batch * slice_size;
        let end = start + slice_size;
        let batch_input = tensor_from_vec_with_template(
            core_shape.to_vec(),
            input.host_data()[start..end].to_vec(),
            input,
        );
        let values = eigh_values_2d(buffers, &batch_input)?;
        data.extend_from_slice(values.host_data());
    }

    Ok(tensor_from_vec_with_template(
        vector_with_batch_shape(n, batch_shape),
        data,
        input,
    ))
}
