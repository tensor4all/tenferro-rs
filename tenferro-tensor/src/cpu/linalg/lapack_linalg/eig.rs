use num_complex::Complex64;

use crate::buffer_pool::BufferPool;
use crate::{Tensor, TypedTensor};

use super::helpers::{
    batched_multi_convert, check_lapack_info, dim_i32, has_zero_dim, square_matrix_dim,
    tensor_from_vec_with_template, work_len, zero_dim_eig_outputs,
};

fn real_eig_to_complex_outputs(
    u_real: &[f64],
    s_re: &[f64],
    s_im: &[f64],
    n: usize,
) -> (Vec<Complex64>, Vec<Complex64>) {
    let mut vectors = vec![Complex64::new(0.0, 0.0); n * n];
    let mut values = vec![Complex64::new(0.0, 0.0); n];
    let mut col = 0;
    while col < n {
        if s_im[col] == 0.0 {
            values[col] = Complex64::new(s_re[col], 0.0);
            for row in 0..n {
                vectors[row + col * n] = Complex64::new(u_real[row + col * n], 0.0);
            }
            col += 1;
        } else {
            values[col] = Complex64::new(s_re[col], s_im[col]);
            values[col + 1] = Complex64::new(s_re[col], -s_im[col]);
            for row in 0..n {
                vectors[row + col * n] =
                    Complex64::new(u_real[row + col * n], u_real[row + (col + 1) * n]);
                vectors[row + (col + 1) * n] =
                    Complex64::new(u_real[row + col * n], -u_real[row + (col + 1) * n]);
            }
            col += 2;
        }
    }
    (vectors, values)
}

fn eig_real_2d(
    _buffers: &mut BufferPool,
    input: &TypedTensor<f64>,
) -> crate::Result<Vec<TypedTensor<Complex64>>> {
    let n = square_matrix_dim(input, "eig")?;
    let n_i32 = dim_i32(n, "eig")?;
    let mut a = input.host_data().to_vec();
    let mut values_re = vec![0.0; n];
    let mut values_im = vec![0.0; n];
    let mut vl = vec![0.0; 1];
    let mut vectors_real = vec![0.0; n * n];
    let mut query = vec![0.0; 1];
    let mut info = 0;
    unsafe {
        lapack::dgeev(
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
    check_lapack_info("eig", "dgeev(work query)", info)?;
    let lwork = work_len(query[0], "eig", "dgeev")?;
    let mut work = vec![0.0; lwork as usize];
    unsafe {
        lapack::dgeev(
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
    check_lapack_info("eig", "dgeev", info)?;
    let (vectors, values) = real_eig_to_complex_outputs(&vectors_real, &values_re, &values_im, n);

    Ok(vec![
        tensor_from_vec_with_template(vec![n], values, input),
        tensor_from_vec_with_template(vec![n, n], vectors, input),
    ])
}

fn eig_complex_2d(
    _buffers: &mut BufferPool,
    input: &TypedTensor<Complex64>,
) -> crate::Result<Vec<TypedTensor<Complex64>>> {
    let n = square_matrix_dim(input, "eig")?;
    let n_i32 = dim_i32(n, "eig")?;
    let mut a = input.host_data().to_vec();
    let mut values = vec![Complex64::new(0.0, 0.0); n];
    let mut vl = vec![Complex64::new(0.0, 0.0); 1];
    let mut vectors = vec![Complex64::new(0.0, 0.0); n * n];
    let mut query = vec![Complex64::new(0.0, 0.0); 1];
    let mut rwork = vec![0.0; 2 * n.max(1)];
    let mut info = 0;
    unsafe {
        lapack::zgeev(
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
    check_lapack_info("eig", "zgeev(work query)", info)?;
    let lwork = work_len(query[0].re, "eig", "zgeev")?;
    let mut work = vec![Complex64::new(0.0, 0.0); lwork as usize];
    unsafe {
        lapack::zgeev(
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
    check_lapack_info("eig", "zgeev", info)?;

    Ok(vec![
        tensor_from_vec_with_template(vec![n], values, input),
        tensor_from_vec_with_template(vec![n, n], vectors, input),
    ])
}

pub(crate) fn eig(buffers: &mut BufferPool, input: &Tensor) -> crate::Result<Vec<Tensor>> {
    if has_zero_dim(input.shape()) {
        return zero_dim_eig_outputs(input);
    }

    match input {
        Tensor::F64(t) => Ok(batched_multi_convert("eig", buffers, t, eig_real_2d)?
            .into_iter()
            .map(Tensor::C64)
            .collect()),
        Tensor::C64(t) => Ok(batched_multi_convert("eig", buffers, t, eig_complex_2d)?
            .into_iter()
            .map(Tensor::C64)
            .collect()),
        _ => Err(crate::Error::BackendFailure {
            op: "eig",
            message: format!("unsupported dtype {:?}", input.dtype()),
        }),
    }
}
