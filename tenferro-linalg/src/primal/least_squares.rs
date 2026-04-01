use super::*;
use tenferro_tensor::MemoryOrder;

pub(crate) fn residual_summary_output_dims(
    batch_dims: &[usize],
    nrhs: usize,
    rhs_is_vector: bool,
) -> Vec<usize> {
    if rhs_is_vector {
        batch_dims.to_vec()
    } else {
        output_dims(&[nrhs], batch_dims)
    }
}

pub(crate) fn empty_residual_summary<T: LinalgScalar>(
    memory: tenferro_device::LogicalMemorySpace,
) -> Result<Tensor<T>> {
    Tensor::<T>::zeros(&[0], memory, MemoryOrder::ColumnMajor)
}

pub(crate) fn full_rank_residual_summaries<T: KernelLinalgScalar>(
    residual_matrix: &Tensor<T>,
    m: usize,
    nrhs: usize,
    batch_dims: &[usize],
    rhs_is_vector: bool,
) -> Result<Tensor<T::Real>>
where
    T::Real: LinalgScalar<Real = T::Real> + num_traits::Float,
{
    let bc = batch_count(batch_dims);
    let (residual_data, _) =
        extract_data(residual_matrix).map_err(|e| Error::InvalidArgument(e.to_string()))?;
    let mut summary = vec![T::Real::zero(); bc * nrhs];
    for batch in 0..bc {
        for col in 0..nrhs {
            let mut acc = T::Real::zero();
            for row in 0..m {
                let value = residual_data[batch * m * nrhs + row + col * m].abs_real();
                acc = acc + value * value;
            }
            summary[batch * nrhs + col] = acc;
        }
    }
    let dims = residual_summary_output_dims(batch_dims, nrhs, rhs_is_vector);
    tensor_from_data(summary, &dims)
}

pub(crate) fn lstsq_has_full_rank<
    T: LinalgScalar<Real = T> + num_traits::Float + tenferro_tensor::KeepCountScalar,
>(
    rank: &Tensor<T>,
    expected_rank: usize,
) -> Result<bool> {
    let (rank_data, _) = extract_data(rank).map_err(|e| Error::InvalidArgument(e.to_string()))?;
    let expected_rank = scalar_from::<T>(expected_rank as f64)?;
    Ok(rank_data.iter().all(|value| *value == expected_rank))
}

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
/// assert_eq!(result.solution.dims(), &[2]);
/// ```
pub fn lstsq<T: KernelLinalgScalar, C>(
    ctx: &mut C,
    a: &Tensor<T>,
    b: &Tensor<T>,
) -> Result<LstsqResult<T, T::Real>>
where
    T: KernelLinalgScalar + tenferro_algebra::Conjugate,
    T::Real: LinalgScalar<Real = T::Real> + num_traits::Float + tenferro_tensor::KeepCountScalar,
    C: backend::TensorLinalgContextFor<T>
        + tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<T>>
        + tenferro_prims::TensorSemiringContextFor<tenferro_algebra::Standard<T>>
        + tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<T::Real>>,
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
    let rhs_is_vector = b_input.ndim() == 1 + batch_dims.len();

    let k = m.min(n);
    let rhs_matrix = if rhs_is_vector {
        b_input.unsqueeze(1)?
    } else {
        b_input.clone()
    };

    let mut q_perm = vec![1, 0];
    q_perm.extend(2..q_input.ndim());
    let q_adj = q_input.conj().permute(&q_perm)?;
    let nrhs = rhs_matrix.dims()[1];
    let qtb = crate::prims_bridge::batched_gemm_with_semiring_tensors(
        ctx,
        &q_adj,
        &rhs_matrix,
        k,
        m,
        nrhs,
    )?;
    let x_matrix = solve_triangular(ctx, &r_input, &qtb, true)?;
    let x = if rhs_is_vector {
        x_matrix.squeeze_dim(1)?
    } else {
        x_matrix
    };
    let projected_rhs =
        crate::prims_bridge::batched_gemm_with_semiring_tensors(ctx, &q_input, &qtb, m, k, nrhs)?;
    let residual_matrix = crate::prims_bridge::scalar_binary_same_shape(
        ctx,
        &projected_rhs,
        &rhs_matrix,
        tenferro_prims::ScalarBinaryOp::Sub,
    )?;
    let aux = lstsq_aux(ctx, a)?;
    let residuals = if m > n && lstsq_has_full_rank(&aux.rank, n)? {
        full_rank_residual_summaries(&residual_matrix, m, nrhs, batch_dims, rhs_is_vector)?
    } else {
        empty_residual_summary::<T::Real>(a.logical_memory_space())?
    };

    Ok(LstsqResult {
        solution: x,
        residuals,
    })
}

/// Compute least-squares auxiliary metadata.
///
/// This returns the singular values used for numerical rank estimation together
/// with a batch-shaped count tensor containing the effective rank.
pub fn lstsq_aux<T: KernelLinalgScalar, C>(
    ctx: &mut C,
    a: &Tensor<T>,
) -> Result<LstsqAuxResult<T::Real>>
where
    T: KernelLinalgScalar + tenferro_algebra::Conjugate,
    T::Real: tenferro_tensor::KeepCountScalar + num_traits::Float,
    C: backend::TensorLinalgContextFor<T>
        + tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<T::Real>>,
    C::Backend: 'static,
{
    let singular_values = svdvals(ctx, a)?;
    let (_, _, batch_dims) = validate_2d(a)?;
    let rank = lstsq_rank_counts_tensor(
        ctx,
        &singular_values,
        a.dims()[0].max(a.dims()[1]),
        batch_dims,
    )?;
    Ok(LstsqAuxResult {
        rank,
        singular_values,
    })
}

fn lstsq_rank_counts_tensor<R, C>(
    ctx: &mut C,
    singular_values: &Tensor<R>,
    scale: usize,
    batch_dims: &[usize],
) -> Result<Tensor<R>>
where
    R: LinalgScalar<Real = R> + num_traits::Float + tenferro_tensor::KeepCountScalar,
    C: tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<R>>,
{
    let k = singular_values.dims().first().copied().unwrap_or(0);
    if k == 0 {
        return crate::prims_bridge::full_like_constant(
            R::zero(),
            batch_dims,
            singular_values.logical_memory_space(),
        );
    }

    let kept_axes: Vec<usize> = (1..singular_values.ndim()).collect();
    let max_sigma = crate::prims_bridge::scalar_reduce_keep_axes(
        ctx,
        singular_values,
        &kept_axes,
        tenferro_prims::ScalarReductionOp::Max,
    )?;
    let scaled_eps = scalar_from::<R>(scale as f64)? * R::epsilon();
    let scaled_eps_tensor = crate::prims_bridge::full_like_constant(
        scaled_eps,
        max_sigma.dims(),
        max_sigma.logical_memory_space(),
    )?;
    let tol_by_batch = crate::prims_bridge::scalar_binary_same_shape(
        ctx,
        &max_sigma,
        &scaled_eps_tensor,
        tenferro_prims::ScalarBinaryOp::Mul,
    )?;
    let tol = broadcast_lstsq_batch_control(&tol_by_batch, singular_values.dims())?;
    let active = crate::prims_bridge::scalar_binary_same_shape(
        ctx,
        singular_values,
        &tol,
        tenferro_prims::ScalarBinaryOp::Greater,
    )?;
    crate::prims_bridge::scalar_sum_keep_axes(ctx, &active, &kept_axes)
}

fn broadcast_lstsq_batch_control<R: LinalgScalar>(
    value_by_batch: &Tensor<R>,
    singular_dims: &[usize],
) -> Result<Tensor<R>> {
    if singular_dims.len() <= 1 {
        return value_by_batch.reshape(&[1])?.broadcast(singular_dims);
    }

    let mut reshape_dims = vec![1];
    reshape_dims.extend_from_slice(&singular_dims[1..]);
    value_by_batch
        .reshape(&reshape_dims)?
        .broadcast(singular_dims)
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
