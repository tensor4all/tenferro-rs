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
pub fn pinv<
    T: KernelLinalgScalar
        + crate::prims_bridge::ScaleTensorByRealSameShape<C>
        + tenferro_algebra::Conjugate,
    C,
>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    rcond: Option<f64>,
) -> Result<Tensor<T>>
where
    C: backend::TensorLinalgContextFor<T>
        + tenferro_prims::TensorResolveConjContextFor<T>
        + tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<T::Real>>
        + tenferro_prims::TensorSemiringContextFor<tenferro_algebra::Standard<T>>,
    C::Backend: 'static,
    T::Real: num_traits::Float + tenferro_tensor::KeepCountScalar,
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
    let threshold: T::Real = scalar_from(rcond.unwrap_or(1e-15))?;
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
    let one_mask = crate::prims_bridge::full_like_constant(
        <T::Real as num_traits::One>::one(),
        keep_mask.dims(),
        keep_mask.logical_memory_space(),
    )?;
    let drop_mask = crate::prims_bridge::scalar_binary_same_shape(
        ctx,
        &one_mask,
        &keep_mask,
        tenferro_prims::ScalarBinaryOp::Sub,
    )?;
    let kept_s = crate::prims_bridge::scalar_binary_same_shape(
        ctx,
        &s_input,
        &keep_mask,
        tenferro_prims::ScalarBinaryOp::Mul,
    )?;
    let safe_s = crate::prims_bridge::scalar_binary_same_shape(
        ctx,
        &kept_s,
        &drop_mask,
        tenferro_prims::ScalarBinaryOp::Add,
    )?;

    let sinv = crate::prims_bridge::scalar_unary_same_shape(
        ctx,
        &safe_s,
        tenferro_prims::ScalarUnaryOp::Reciprocal,
    )?;
    let sinv = crate::prims_bridge::scalar_binary_same_shape(
        ctx,
        &sinv,
        &keep_mask,
        tenferro_prims::ScalarBinaryOp::Mul,
    )?;

    let sinv_for_vt = sinv.unsqueeze(1)?.broadcast(vt_input.dims())?;
    let vt_scaled = crate::prims_bridge::complex_scale_same_shape(ctx, &vt_input, &sinv_for_vt)?;

    let mut perm = vec![1, 0];
    perm.extend(2..u_input.ndim());
    let u_t = crate::prims_bridge::resolve_conj(ctx, &u_input.conj().permute(&perm)?);
    let vt_t = crate::prims_bridge::resolve_conj(ctx, &vt_scaled.conj().permute(&perm)?);

    crate::prims_bridge::batched_gemm_with_semiring_tensors(ctx, &vt_t, &u_t, n, k, m)
}

/// Compute the matrix exponential `exp(A)` of a square matrix.
#[allow(private_bounds)]
pub fn matrix_exp<T, C>(ctx: &mut C, tensor: &Tensor<T>) -> Result<Tensor<T>>
where
    T: KernelLinalgScalar
        + crate::prims_bridge::ScaleTensorByRealSameShape<C>
        + MatrixExpAbsTensor<C>,
    C: backend::TensorLinalgContextFor<T>,
    C: tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<T>>,
    C: tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<T::Real>>,
    C: tenferro_prims::TensorSemiringContextFor<tenferro_algebra::Standard<T>>,
    C::Backend: 'static,
{
    require_linalg_support::<T, C>(backend::LinalgCapabilityOp::MatrixExp, "matrix_exp")?;

    let (n, batch_dims) = validate_square(tensor)?;
    let input = ensure_col_major(tensor);
    if n == 0 {
        let dims = output_dims(&[n, n], batch_dims);
        return Ok(Tensor::zeros(
            &dims,
            input.logical_memory_space(),
            MemoryOrder::ColumnMajor,
        ));
    }

    let global_norm = crate::ad_helpers::matrix_exp_global_1_norm_tensor(ctx, &input)?;
    let global_norm_cpu =
        global_norm.to_memory_space_async(tenferro_device::LogicalMemorySpace::MainMemory)?;
    let global_norm_slice = global_norm_cpu.buffer().as_slice().ok_or_else(|| {
        Error::InvalidArgument(
            "matrix_exp: expected scalar norm tensor to materialize on host".into(),
        )
    })?;
    let norm_value = *global_norm_slice
        .first()
        .ok_or_else(|| Error::InvalidArgument("matrix_exp: missing scalar norm value".into()))?;
    let norm_f64: f64 = num_traits::NumCast::from(norm_value)
        .ok_or_else(|| Error::InvalidArgument("matrix_exp: cannot convert 1-norm to f64".into()))?;

    let s: usize = if norm_f64 <= crate::ad_helpers::THETA_13 {
        0
    } else {
        (norm_f64 / crate::ad_helpers::THETA_13)
            .log2()
            .ceil()
            .max(0.0) as usize
    };

    let scale_denom = (1u64 << s.min(63)) as f64;
    let scale_inv = crate::ad_helpers::scalar_from::<T::Real>(1.0 / scale_denom)?;
    let scale_tensor = crate::prims_bridge::full_like_constant(
        scale_inv,
        input.dims(),
        input.logical_memory_space(),
    )?;
    let scaled_input =
        <T as crate::prims_bridge::ScaleTensorByRealSameShape<C>>::scale_tensor_by_real_same_shape(
            ctx,
            &input,
            &scale_tensor,
        )?;

    crate::ad_helpers::matrix_exp_tensor_native(ctx, &scaled_input, n, batch_dims, s)
}
