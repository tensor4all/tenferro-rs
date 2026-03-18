use super::*;

/// Solve the least squares problem: `x = argmin ||Ax - b||²`.
pub fn lstsq<T: KernelLinalgScalar, C>(
    ctx: &mut C,
    a: &Tensor<T>,
    b: &Tensor<T>,
) -> Result<LstsqResult<T>>
where
    T: KernelLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    require_linalg_support::<T, C>(backend::LinalgCapabilityOp::Lstsq, "lstsq")?;

    let (m, n, batch_dims) = validate_2d(a)?;
    if m < n {
        return Err(Error::InvalidArgument(format!(
            "lstsq requires m >= n, got m={m}, n={n}"
        )));
    }
    validate_lstsq_rhs(b, m, batch_dims)?;

    let qr_result = qr(ctx, a)?;
    let q_input = ensure_col_major(&qr_result.q);
    let r_input = ensure_col_major(&qr_result.r);
    let b_input = ensure_col_major(b);

    let q_data = extract_slice(&q_input)?;
    let r_data = extract_slice(&r_input)?;
    let b_data = extract_slice(&b_input)?;
    let q_off = q_input.offset() as usize;
    let r_off = r_input.offset() as usize;
    let b_off = b_input.offset() as usize;

    let k = m.min(n);
    let bc = batch_count(batch_dims);

    let mut x_data = vec![T::zero(); n * bc];
    let mut res_data = vec![T::zero(); m * bc];
    let mut x_buf = vec![T::zero(); k];

    for batch in 0..bc {
        let q_b = &q_data[q_off + batch * m * k..q_off + (batch + 1) * m * k];
        let r_b = &r_data[r_off + batch * k * n..r_off + (batch + 1) * k * n];
        let b_b = &b_data[b_off + batch * m..b_off + (batch + 1) * m];

        let mut qtb = vec![T::zero(); k];
        for i in 0..k {
            let mut sum = T::zero();
            for j in 0..m {
                sum = sum + q_b[j + i * m] * b_b[j];
            }
            qtb[i] = sum;
        }

        let x_solution = backend::slice_bridge::solve_triangular_vec(ctx, r_b, &qtb, k, 1, true)?;
        if x_solution.len() != x_buf.len() {
            return Err(Error::DeviceError(format!(
                "solve_triangular_vec returned unexpected size: expected {}, got {}",
                x_buf.len(),
                x_solution.len()
            )));
        }
        x_buf.copy_from_slice(&x_solution);
        x_data[batch * n..(batch + 1) * n].copy_from_slice(&x_buf);

        let a_contiguous = a.contiguous(MemoryOrder::ColumnMajor);
        let a_slice = extract_slice(&a_contiguous)?;
        let a_off = a_contiguous.offset() as usize;
        let a_data_local = &a_slice[a_off + batch * m * n..a_off + (batch + 1) * m * n];
        for i in 0..m {
            let mut ax_i = T::zero();
            for j in 0..n {
                ax_i = ax_i + a_data_local[i + j * m] * x_buf[j];
            }
            res_data[batch * m + i] = b_b[i] - ax_i;
        }
    }

    let x_dims = output_dims(&[n], batch_dims);
    let res_dims = output_dims(&[m], batch_dims);

    Ok(LstsqResult {
        x: tensor_from_data(x_data, &x_dims)?,
        residual: tensor_from_data(res_data, &res_dims)?,
    })
}

/// Compute the Cholesky decomposition of a Hermitian positive-definite matrix.
pub fn cholesky<T: KernelLinalgScalar, C>(ctx: &mut C, tensor: &Tensor<T>) -> Result<Tensor<T>>
where
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    <C::Backend as backend::TensorLinalgBackend<T>>::cholesky(ctx, tensor)
}

/// Compute the Cholesky decomposition with numerical status information.
pub fn cholesky_ex<T: KernelLinalgScalar, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
) -> Result<CholeskyExResult<T>>
where
    T: KernelLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    require_linalg_support::<T, C>(backend::LinalgCapabilityOp::CholeskyEx, "cholesky_ex")?;

    let (n, batch_dims) = validate_square(tensor)?;
    let bc = batch_count(batch_dims);
    let input = ensure_col_major(tensor);
    let data = extract_slice(&input)?;
    let offset = input.offset() as usize;
    let mat_size = n * n;

    let mut factors = vec![T::zero(); mat_size * bc];
    let mut info = vec![0_i32; bc];

    for batch in 0..bc {
        let start = offset + batch * mat_size;
        let a_slice = &data[start..start + mat_size];
        let l_out = &mut factors[batch * mat_size..(batch + 1) * mat_size];
        match backend::slice_bridge::cholesky_vec(ctx, a_slice, n) {
            Ok(factor_b) => l_out.copy_from_slice(&factor_b),
            Err(_) => {
                l_out.fill(T::zero());
                info[batch] = 1;
            }
        }
    }

    Ok(CholeskyExResult {
        l: tensor_from_data(factors, &output_dims(&[n, n], batch_dims))?,
        info,
    })
}
