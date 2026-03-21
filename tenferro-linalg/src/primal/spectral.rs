use super::*;

/// Compute the eigendecomposition of a general (non-symmetric) square matrix.
pub fn eig<
    T: KernelLinalgScalar<Real = T, Complex = num_complex::Complex<T>> + num_traits::Float,
    C,
>(
    ctx: &mut C,
    tensor: &Tensor<T>,
) -> Result<EigResult<T>>
where
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    let result = <C::Backend as backend::TensorLinalgBackend<T>>::eig(ctx, tensor)?;
    Ok(EigResult {
        values: result.values,
        vectors: result.vectors,
    })
}

pub(crate) fn require_linalg_support<T: KernelLinalgScalar, C>(
    capability: backend::LinalgCapabilityOp,
    op: &str,
) -> Result<()>
where
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    if <C::Backend as backend::TensorLinalgBackend<T>>::has_linalg_support(capability) {
        return Ok(());
    }

    Err(Error::DeviceError(format!(
        "{op} is not supported on the current linalg backend"
    )))
}

/// Compute the Moore-Penrose pseudoinverse of a matrix.
///
/// # Examples
///
/// ```
/// use tenferro_device::LogicalMemorySpace;
/// use tenferro_linalg::pinv;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::from_slice(
///     &[1.0, 2.0, 3.0, 4.0],
///     &[2, 2],
///     MemoryOrder::ColumnMajor,
/// ).unwrap();
/// let ap = pinv(&mut ctx, &a, None).unwrap();
/// assert_eq!(ap.logical_memory_space(), LogicalMemorySpace::MainMemory);
/// ```
pub fn pinv<T: KernelLinalgScalar<Real = T> + num_traits::Float, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    rcond: Option<f64>,
) -> Result<Tensor<T>>
where
    T: KernelLinalgScalar,
    C: backend::TensorLinalgContextFor<T>
        + tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<T::Real>>
        + tenferro_prims::TensorSemiringContextFor<tenferro_algebra::Standard<T>>,
    C::Backend: 'static,
    T::Real: tenferro_tensor::KeepCountScalar,
{
    require_linalg_support::<T, C>(backend::LinalgCapabilityOp::Pinv, "pinv")?;

    let (m, n, batch_dims) = validate_2d(tensor)?;
    let k = m.min(n);
    if k == 0 {
        let dims = output_dims(&[n, m], batch_dims);
        return Ok(Tensor::zeros(
            &dims,
            tensor.logical_memory_space(),
            MemoryOrder::ColumnMajor,
        ));
    }

    let svd_result = svd(ctx, tensor, None)?;
    let u_input = ensure_col_major(&svd_result.u);
    let s_input = ensure_col_major(&svd_result.s);
    let vt_input = ensure_col_major(&svd_result.vt);
    let s_max_axes: Vec<usize> = (1..s_input.ndim()).collect();
    let s_max = crate::prims_bridge::scalar_reduce_keep_axes(
        ctx,
        &s_input,
        &s_max_axes,
        tenferro_prims::ScalarReductionOp::Max,
    )?;
    let threshold: T = scalar_from(rcond.unwrap_or(1e-15))?;
    let threshold_tensor = crate::prims_bridge::full_like_constant(
        threshold,
        s_max.dims(),
        s_max.logical_memory_space(),
    )?;
    let cutoff = crate::prims_bridge::scalar_binary_same_shape(
        ctx,
        &s_max,
        &threshold_tensor,
        tenferro_prims::ScalarBinaryOp::Mul,
    )?;
    let cutoff = cutoff.unsqueeze(0)?.broadcast(s_input.dims())?;
    let keep_mask = crate::prims_bridge::scalar_binary_same_shape(
        ctx,
        &s_input,
        &cutoff,
        tenferro_prims::ScalarBinaryOp::Greater,
    )?;
    let keep_counts = crate::prims_bridge::scalar_sum_keep_axes(ctx, &keep_mask, &s_max_axes)?;

    let mut sinv = crate::prims_bridge::scalar_unary_same_shape(
        ctx,
        &s_input,
        tenferro_prims::ScalarUnaryOp::Reciprocal,
    )?;
    sinv = backend::tensor_helpers::zero_trailing_by_counts(&sinv, &keep_counts, 0, 1)?;

    let sinv_for_vt = sinv.unsqueeze(1)?.broadcast(vt_input.dims())?;
    let vt_scaled = crate::prims_bridge::scalar_binary_same_shape(
        ctx,
        &vt_input,
        &sinv_for_vt,
        tenferro_prims::ScalarBinaryOp::Mul,
    )?;

    let mut perm = vec![1, 0];
    perm.extend(2..u_input.ndim());
    let u_t = u_input.permute(&perm)?;
    let vt_t = vt_scaled.permute(&perm)?;

    crate::prims_bridge::batched_gemm_with_semiring_tensors(ctx, &vt_t, &u_t, n, k, m)
}

/// Compute the matrix exponential `exp(A)` of a square matrix.
pub fn matrix_exp<T: KernelLinalgScalar, C>(ctx: &mut C, tensor: &Tensor<T>) -> Result<Tensor<T>>
where
    T: KernelLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    require_linalg_support::<T, C>(backend::LinalgCapabilityOp::MatrixExp, "matrix_exp")?;

    let (n, batch_dims) = validate_square(tensor)?;
    let input = ensure_col_major(tensor);
    let data = extract_slice(&input)?;
    let offset = input.offset() as usize;
    let bc = batch_count(batch_dims);
    let mat_size = n * n;

    let mut result_data = vec![T::zero(); mat_size * bc];

    for b in 0..bc {
        let start = offset + b * mat_size;
        let a_slice = &data[start..start + mat_size];
        let exp_a = matrix_exp_single(ctx, a_slice, n)?;
        result_data[b * mat_size..(b + 1) * mat_size].copy_from_slice(&exp_a);
    }

    let dims = output_dims(&[n, n], batch_dims);
    tensor_from_data(result_data, &dims)
}
