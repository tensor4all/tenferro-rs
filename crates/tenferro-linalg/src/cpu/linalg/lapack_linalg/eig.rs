use num_complex::{Complex32, Complex64};

use tenferro_cpu::linalg_interop::BufferPool;
use tenferro_tensor::{Tensor, TypedTensor};

use super::helpers::{
    batched_multi_convert, check_lapack_info, dim_i32, has_zero_dim, square_matrix_dim,
    tensor_from_vec_with_template, work_len, zero_dim_eig_outputs,
};

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
                if s_im[col] == 0.0 {
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

macro_rules! impl_eig_real_2d {
    ($name:ident, $real:ty, $complex:ty, $geev:path, $routine:literal, $convert:ident) => {
        fn $name(
            _buffers: &mut BufferPool,
            input: &TypedTensor<$real>,
        ) -> tenferro_tensor::Result<Vec<TypedTensor<$complex>>> {
            let n = square_matrix_dim(input, "eig")?;
            let n_i32 = dim_i32(n, "eig")?;
            let mut a = input.host_data().to_vec();
            let mut values_re = vec![0.0 as $real; n];
            let mut values_im = vec![0.0 as $real; n];
            let mut vl = vec![0.0 as $real; 1];
            let mut vectors_real = vec![0.0 as $real; n * n];
            let mut query = vec![0.0 as $real; 1];
            let mut info = 0;
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
                tensor_from_vec_with_template(vec![n], values, input),
                tensor_from_vec_with_template(vec![n, n], vectors, input),
            ])
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
            let mut a = input.host_data().to_vec();
            let mut values = vec![<$complex>::new(0.0, 0.0); n];
            let mut vl = vec![<$complex>::new(0.0, 0.0); 1];
            let mut vectors = vec![<$complex>::new(0.0, 0.0); n * n];
            let mut query = vec![<$complex>::new(0.0, 0.0); 1];
            let mut rwork = vec![0.0 as $real; 2 * n.max(1)];
            let mut info = 0;
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
                tensor_from_vec_with_template(vec![n], values, input),
                tensor_from_vec_with_template(vec![n, n], vectors, input),
            ])
        }
    };
}

impl_real_eig_to_complex_outputs!(real32_eig_to_complex_outputs, f32, Complex32);
impl_real_eig_to_complex_outputs!(real64_eig_to_complex_outputs, f64, Complex64);
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
