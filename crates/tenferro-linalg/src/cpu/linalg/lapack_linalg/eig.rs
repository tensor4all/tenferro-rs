use num_complex::{Complex32, Complex64};

use tenferro_cpu::linalg_interop::BufferPool;
use tenferro_tensor::{Tensor, TypedTensor};

use super::helpers::{
    batched_multi_convert, check_lapack_info, dim_i32, has_zero_dim, square_matrix_dim,
    tensor_from_vec_with_template, vector_with_batch_shape, work_len, zero_dim_eig_outputs,
};

fn eig_imag_is_effectively_zero(real: f64, imag: f64, eps: f64) -> bool {
    imag.abs() <= eps * real.abs().max(1.0)
}

macro_rules! impl_real_eig_to_complex_outputs {
    ($name:ident, $real:ty, $complex:ty) => {
        fn $name(
            u_real: &[$real],
            s_re: &[$real],
            s_im: &[$real],
            n: usize,
        ) -> (Vec<$complex>, Vec<$complex>) {
            let mut vectors = vec![<$complex>::new(0.0, 0.0); n * n];
            let mut values = vec![<$complex>::new(0.0, 0.0); n];
            let mut col = 0;
            while col < n {
                if col + 1 >= n
                    || eig_imag_is_effectively_zero(
                        s_re[col] as f64,
                        s_im[col] as f64,
                        <$real>::EPSILON as f64,
                    )
                {
                    values[col] = <$complex>::new(s_re[col], 0.0);
                    for row in 0..n {
                        vectors[row + col * n] = <$complex>::new(u_real[row + col * n], 0.0);
                    }
                    col += 1;
                } else {
                    values[col] = <$complex>::new(s_re[col], s_im[col]);
                    values[col + 1] = <$complex>::new(s_re[col], -s_im[col]);
                    for row in 0..n {
                        vectors[row + col * n] =
                            <$complex>::new(u_real[row + col * n], u_real[row + (col + 1) * n]);
                        vectors[row + (col + 1) * n] =
                            <$complex>::new(u_real[row + col * n], -u_real[row + (col + 1) * n]);
                    }
                    col += 2;
                }
            }
            (vectors, values)
        }
    };
}

macro_rules! impl_real_eig_to_complex_values {
    ($name:ident, $real:ty, $complex:ty) => {
        fn $name(
            buffers: &mut BufferPool,
            s_re: &[$real],
            s_im: &[$real],
            n: usize,
        ) -> Vec<$complex> {
            let mut values = buffers.acquire_with_capacity::<$complex>(n);
            for idx in 0..n {
                values.push(<$complex>::new(s_re[idx], s_im[idx]));
            }
            values
        }
    };
}

macro_rules! impl_eig_real_2d {
    ($name:ident, $real:ty, $complex:ty, $geev:path, $routine:literal, $convert:ident) => {
        fn $name(
            _buffers: &mut BufferPool,
            input: &TypedTensor<$real>,
        ) -> tenferro_tensor::Result<Vec<TypedTensor<$complex>>> {
            let n = square_matrix_dim(input, "eig")?;
            let n_i32 = dim_i32(n, "eig")?;
            let mut a = input.host_data()?.to_vec();
            let mut values_re = vec![0.0 as $real; n];
            let mut values_im = vec![0.0 as $real; n];
            let mut vl = vec![0.0 as $real; 1];
            let mut vectors_real = vec![0.0 as $real; n * n];
            let mut query = vec![0.0 as $real; 1];
            let mut info = 0;
            // SAFETY: all matrix/vector buffers match the validated `n x n`
            // problem, and `lwork = -1` makes `query` the only workspace output.
            unsafe {
                $geev(
                    b'N',
                    b'V',
                    n_i32,
                    &mut a,
                    n_i32,
                    &mut values_re,
                    &mut values_im,
                    &mut vl,
                    1,
                    &mut vectors_real,
                    n_i32,
                    &mut query,
                    -1,
                    &mut info,
                );
            }
            check_lapack_info("eig", concat!($routine, "(work query)"), info)?;
            let lwork = work_len(query[0] as f64, "eig", $routine)?;
            let mut work = vec![0.0 as $real; lwork as usize];
            // SAFETY: buffers and leading dimensions match the validated
            // problem, and `work` uses the length returned by the LAPACK query.
            unsafe {
                $geev(
                    b'N',
                    b'V',
                    n_i32,
                    &mut a,
                    n_i32,
                    &mut values_re,
                    &mut values_im,
                    &mut vl,
                    1,
                    &mut vectors_real,
                    n_i32,
                    &mut work,
                    lwork,
                    &mut info,
                );
            }
            check_lapack_info("eig", $routine, info)?;
            let (vectors, values) = $convert(&vectors_real, &values_re, &values_im, n);

            Ok(vec![
                tensor_from_vec_with_template(vec![n], values, input)?,
                tensor_from_vec_with_template(vec![n, n], vectors, input)?,
            ])
        }
    };
}

macro_rules! impl_eig_values_real_2d {
    ($name:ident, $real:ty, $complex:ty, $geev:path, $routine:literal, $convert:ident) => {
        fn $name(
            buffers: &mut BufferPool,
            input: &TypedTensor<$real>,
        ) -> tenferro_tensor::Result<TypedTensor<$complex>> {
            let n = square_matrix_dim(input, "eig_values")?;
            let n_i32 = dim_i32(n, "eig_values")?;
            let mut a = input.host_data()?.to_vec();
            let mut values_re = vec![0.0 as $real; n];
            let mut values_im = vec![0.0 as $real; n];
            let mut vl = vec![0.0 as $real; 1];
            let mut vr = vec![0.0 as $real; 1];
            let mut query = vec![0.0 as $real; 1];
            let mut info = 0;
            // SAFETY: all matrix/vector buffers match the validated `n x n`
            // problem, and `lwork = -1` makes `query` the only workspace output.
            unsafe {
                $geev(
                    b'N',
                    b'N',
                    n_i32,
                    &mut a,
                    n_i32,
                    &mut values_re,
                    &mut values_im,
                    &mut vl,
                    1,
                    &mut vr,
                    1,
                    &mut query,
                    -1,
                    &mut info,
                );
            }
            check_lapack_info("eig_values", concat!($routine, "(work query)"), info)?;
            let lwork = work_len(query[0] as f64, "eig_values", $routine)?;
            let mut work = vec![0.0 as $real; lwork as usize];
            // SAFETY: buffers and leading dimensions match the validated
            // problem, and `work` uses the length returned by the LAPACK query.
            unsafe {
                $geev(
                    b'N',
                    b'N',
                    n_i32,
                    &mut a,
                    n_i32,
                    &mut values_re,
                    &mut values_im,
                    &mut vl,
                    1,
                    &mut vr,
                    1,
                    &mut work,
                    lwork,
                    &mut info,
                );
            }
            check_lapack_info("eig_values", $routine, info)?;
            let values = $convert(buffers, &values_re, &values_im, n);

            tensor_from_vec_with_template(vec![n], values, input)
        }
    };
}

macro_rules! impl_eig_complex_2d {
    ($name:ident, $complex:ty, $real:ty, $geev:path, $routine:literal) => {
        fn $name(
            _buffers: &mut BufferPool,
            input: &TypedTensor<$complex>,
        ) -> tenferro_tensor::Result<Vec<TypedTensor<$complex>>> {
            let n = square_matrix_dim(input, "eig")?;
            let n_i32 = dim_i32(n, "eig")?;
            let mut a = input.host_data()?.to_vec();
            let mut values = vec![<$complex>::new(0.0, 0.0); n];
            let mut vl = vec![<$complex>::new(0.0, 0.0); 1];
            let mut vectors = vec![<$complex>::new(0.0, 0.0); n * n];
            let mut query = vec![<$complex>::new(0.0, 0.0); 1];
            let mut rwork = vec![0.0 as $real; 2 * n.max(1)];
            let mut info = 0;
            // SAFETY: all complex matrix/vector buffers and real workspace
            // match the validated `n x n` problem; `lwork = -1` queries workspace.
            unsafe {
                $geev(
                    b'N',
                    b'V',
                    n_i32,
                    &mut a,
                    n_i32,
                    &mut values,
                    &mut vl,
                    1,
                    &mut vectors,
                    n_i32,
                    &mut query,
                    -1,
                    &mut rwork,
                    &mut info,
                );
            }
            check_lapack_info("eig", concat!($routine, "(work query)"), info)?;
            let lwork = work_len(query[0].re as f64, "eig", $routine)?;
            let mut work = vec![<$complex>::new(0.0, 0.0); lwork as usize];
            // SAFETY: buffers, real workspace, and leading dimensions match
            // the validated problem, and `work` has the queried length.
            unsafe {
                $geev(
                    b'N',
                    b'V',
                    n_i32,
                    &mut a,
                    n_i32,
                    &mut values,
                    &mut vl,
                    1,
                    &mut vectors,
                    n_i32,
                    &mut work,
                    lwork,
                    &mut rwork,
                    &mut info,
                );
            }
            check_lapack_info("eig", $routine, info)?;

            Ok(vec![
                tensor_from_vec_with_template(vec![n], values, input)?,
                tensor_from_vec_with_template(vec![n, n], vectors, input)?,
            ])
        }
    };
}

macro_rules! impl_eig_values_complex_2d {
    ($name:ident, $complex:ty, $real:ty, $geev:path, $routine:literal) => {
        fn $name(
            _buffers: &mut BufferPool,
            input: &TypedTensor<$complex>,
        ) -> tenferro_tensor::Result<TypedTensor<$complex>> {
            let n = square_matrix_dim(input, "eig_values")?;
            let n_i32 = dim_i32(n, "eig_values")?;
            let mut a = input.host_data()?.to_vec();
            let mut values = vec![<$complex>::new(0.0, 0.0); n];
            let mut vl = vec![<$complex>::new(0.0, 0.0); 1];
            let mut vr = vec![<$complex>::new(0.0, 0.0); 1];
            let mut query = vec![<$complex>::new(0.0, 0.0); 1];
            let mut rwork = vec![0.0 as $real; 2 * n.max(1)];
            let mut info = 0;
            // SAFETY: all complex matrix/vector buffers and real workspace
            // match the validated `n x n` problem; `lwork = -1` queries workspace.
            unsafe {
                $geev(
                    b'N',
                    b'N',
                    n_i32,
                    &mut a,
                    n_i32,
                    &mut values,
                    &mut vl,
                    1,
                    &mut vr,
                    1,
                    &mut query,
                    -1,
                    &mut rwork,
                    &mut info,
                );
            }
            check_lapack_info("eig_values", concat!($routine, "(work query)"), info)?;
            let lwork = work_len(query[0].re as f64, "eig_values", $routine)?;
            let mut work = vec![<$complex>::new(0.0, 0.0); lwork as usize];
            // SAFETY: buffers, real workspace, and leading dimensions match
            // the validated problem, and `work` has the queried length.
            unsafe {
                $geev(
                    b'N',
                    b'N',
                    n_i32,
                    &mut a,
                    n_i32,
                    &mut values,
                    &mut vl,
                    1,
                    &mut vr,
                    1,
                    &mut work,
                    lwork,
                    &mut rwork,
                    &mut info,
                );
            }
            check_lapack_info("eig_values", $routine, info)?;

            tensor_from_vec_with_template(vec![n], values, input)
        }
    };
}

impl_real_eig_to_complex_outputs!(real32_eig_to_complex_outputs, f32, Complex32);
impl_real_eig_to_complex_outputs!(real64_eig_to_complex_outputs, f64, Complex64);
impl_real_eig_to_complex_values!(real32_eig_to_complex_values, f32, Complex32);
impl_real_eig_to_complex_values!(real64_eig_to_complex_values, f64, Complex64);
impl_eig_real_2d!(
    eig_real32_2d,
    f32,
    Complex32,
    lapack::sgeev,
    "sgeev",
    real32_eig_to_complex_outputs
);
impl_eig_real_2d!(
    eig_real64_2d,
    f64,
    Complex64,
    lapack::dgeev,
    "dgeev",
    real64_eig_to_complex_outputs
);
impl_eig_complex_2d!(eig_complex32_2d, Complex32, f32, lapack::cgeev, "cgeev");
impl_eig_complex_2d!(eig_complex64_2d, Complex64, f64, lapack::zgeev, "zgeev");
impl_eig_values_real_2d!(
    eig_values_real32_2d,
    f32,
    Complex32,
    lapack::sgeev,
    "sgeev",
    real32_eig_to_complex_values
);
impl_eig_values_real_2d!(
    eig_values_real64_2d,
    f64,
    Complex64,
    lapack::dgeev,
    "dgeev",
    real64_eig_to_complex_values
);
impl_eig_values_complex_2d!(
    eig_values_complex32_2d,
    Complex32,
    f32,
    lapack::cgeev,
    "cgeev"
);
impl_eig_values_complex_2d!(
    eig_values_complex64_2d,
    Complex64,
    f64,
    lapack::zgeev,
    "zgeev"
);

pub(crate) fn eig(
    buffers: &mut BufferPool,
    input: &Tensor,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    if has_zero_dim(input.shape()) {
        return zero_dim_eig_outputs(input);
    }

    match input {
        Tensor::F32(t) => Ok(batched_multi_convert("eig", buffers, t, eig_real32_2d)?
            .into_iter()
            .map(Tensor::C32)
            .collect()),
        Tensor::F64(t) => Ok(batched_multi_convert("eig", buffers, t, eig_real64_2d)?
            .into_iter()
            .map(Tensor::C64)
            .collect()),
        Tensor::C32(t) => Ok(batched_multi_convert("eig", buffers, t, eig_complex32_2d)?
            .into_iter()
            .map(Tensor::C32)
            .collect()),
        Tensor::C64(t) => Ok(batched_multi_convert("eig", buffers, t, eig_complex64_2d)?
            .into_iter()
            .map(Tensor::C64)
            .collect()),
        _ => Err(tenferro_tensor::Error::backend_failure(
            "eig",
            format!("unsupported dtype {:?}", input.dtype()),
        )),
    }
}

fn zero_dim_eig_values_output(input: &Tensor) -> tenferro_tensor::Result<Tensor> {
    let shape = input.shape();
    if shape.len() < 2 {
        return Err(tenferro_tensor::Error::RankMismatch {
            op: "eig_values",
            expected: 2,
            actual: shape.len(),
        });
    }
    let n = shape[0];
    if shape[1] != n {
        return Err(tenferro_tensor::Error::ShapeMismatch {
            op: "eig_values",
            lhs: vec![n],
            rhs: vec![shape[1]],
        });
    }
    let value_shape = vector_with_batch_shape(n, &shape[2..]);
    match input {
        Tensor::F32(_) | Tensor::C32(_) => Ok(Tensor::C32(TypedTensor::from_vec_col_major(
            value_shape,
            Vec::new(),
        )?)),
        Tensor::F64(_) | Tensor::C64(_) => Ok(Tensor::C64(TypedTensor::from_vec_col_major(
            value_shape,
            Vec::new(),
        )?)),
        _ => Err(tenferro_tensor::Error::backend_failure(
            "eig_values",
            format!("unsupported dtype {:?}", input.dtype()),
        )),
    }
}

pub(crate) fn eig_values(
    buffers: &mut BufferPool,
    input: &Tensor,
) -> tenferro_tensor::Result<Tensor> {
    if has_zero_dim(input.shape()) {
        return zero_dim_eig_values_output(input);
    }

    match input {
        Tensor::F32(t) => {
            let mut outputs = batched_multi_convert("eig_values", buffers, t, |buffers, batch| {
                eig_values_real32_2d(buffers, batch).map(|values| vec![values])
            })?;
            Ok(Tensor::C32(outputs.remove(0)))
        }
        Tensor::F64(t) => {
            let mut outputs = batched_multi_convert("eig_values", buffers, t, |buffers, batch| {
                eig_values_real64_2d(buffers, batch).map(|values| vec![values])
            })?;
            Ok(Tensor::C64(outputs.remove(0)))
        }
        Tensor::C32(t) => {
            let mut outputs = batched_multi_convert("eig_values", buffers, t, |buffers, batch| {
                eig_values_complex32_2d(buffers, batch).map(|values| vec![values])
            })?;
            Ok(Tensor::C32(outputs.remove(0)))
        }
        Tensor::C64(t) => {
            let mut outputs = batched_multi_convert("eig_values", buffers, t, |buffers, batch| {
                eig_values_complex64_2d(buffers, batch).map(|values| vec![values])
            })?;
            Ok(Tensor::C64(outputs.remove(0)))
        }
        _ => Err(tenferro_tensor::Error::backend_failure(
            "eig_values",
            format!("unsupported dtype {:?}", input.dtype()),
        )),
    }
}
