use super::*;

/// Solve the least squares problem: `x = argmin ||Ax - b||²`.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::lstsq;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let mut ctx = CpuContext::new(1);
/// let col = MemoryOrder::ColumnMajor;
/// let a = Tensor::<f64>::from_slice(&[1.0, 0.0, 1.0, 1.0], &[2, 2], col).unwrap();
/// let b = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], col).unwrap();
/// let result = lstsq(&mut ctx, &a, &b).unwrap();
/// assert_eq!(result.x.dims(), &[2]);
/// ```
pub fn lstsq<T: KernelLinalgScalar, C>(
    ctx: &mut C,
    a: &Tensor<T>,
    b: &Tensor<T>,
) -> Result<LstsqResult<T>>
where
    T: KernelLinalgScalar + tenferro_algebra::Conjugate,
    C: backend::TensorLinalgContextFor<T>
        + tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<T>>
        + tenferro_prims::TensorSemiringContextFor<tenferro_algebra::Standard<T>>,
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

    let k = m.min(n);
    let rhs_matrix = b_input.unsqueeze(1)?;

    let mut q_perm = vec![1, 0];
    q_perm.extend(2..q_input.ndim());
    let q_adj = q_input.conj().permute(&q_perm)?;
    let qtb =
        crate::prims_bridge::batched_gemm_with_semiring_tensors(ctx, &q_adj, &rhs_matrix, k, m, 1)?;
    let x_matrix = solve_triangular(ctx, &r_input, &qtb, true)?;
    let x = x_matrix.squeeze_dim(1)?;
    let projected_rhs =
        crate::prims_bridge::batched_gemm_with_semiring_tensors(ctx, &q_input, &qtb, m, k, 1)?
            .squeeze_dim(1)?;
    let residual = crate::prims_bridge::scalar_binary_same_shape(
        ctx,
        &b_input,
        &projected_rhs,
        tenferro_prims::ScalarBinaryOp::Sub,
    )?;

    Ok(LstsqResult { x, residual })
}

/// Compute the Cholesky decomposition of a Hermitian positive-definite matrix.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::cholesky;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let mut ctx = CpuContext::new(1);
/// let col = MemoryOrder::ColumnMajor;
/// let a = Tensor::<f64>::from_slice(&[4.0, 2.0, 2.0, 3.0], &[2, 2], col).unwrap();
/// let l = cholesky(&mut ctx, &a).unwrap();
/// assert_eq!(l.dims(), &[2, 2]);
/// ```
pub fn cholesky<T: KernelLinalgScalar, C>(ctx: &mut C, tensor: &Tensor<T>) -> Result<Tensor<T>>
where
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    <C::Backend as backend::TensorLinalgBackend<T>>::cholesky(ctx, tensor)
}

/// Compute the Cholesky decomposition with numerical status information.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::cholesky_ex;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let mut ctx = CpuContext::new(1);
/// let col = MemoryOrder::ColumnMajor;
/// let a = Tensor::<f64>::from_slice(&[4.0, 2.0, 2.0, 3.0], &[2, 2], col).unwrap();
/// let result = cholesky_ex(&mut ctx, &a).unwrap();
/// assert_eq!(result.l.dims(), &[2, 2]);
/// assert_eq!(result.info.len(), 1);
/// ```
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
    let result = <C::Backend as backend::TensorLinalgBackend<T>>::cholesky_ex(ctx, tensor)?;
    Ok(CholeskyExResult {
        l: result.l,
        info: result.info,
    })
}
