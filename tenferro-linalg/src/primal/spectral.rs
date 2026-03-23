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

fn control_tensor_from_host_values<R>(
    values: &[R],
    dims: &[usize],
    memory_space: tenferro_device::LogicalMemorySpace,
) -> Result<Tensor<R>>
where
    R: LinalgScalar<Real = R>,
{
    let host = if dims.is_empty() {
        let value = *values
            .first()
            .ok_or_else(|| Error::InvalidArgument("matrix_exp: missing control value".into()))?;
        Tensor::from_vec(vec![value], &[], &[], 0)?
    } else {
        Tensor::from_slice(values, dims, MemoryOrder::ColumnMajor)?
    };

    if memory_space == tenferro_device::LogicalMemorySpace::MainMemory {
        Ok(host)
    } else {
        host.to_memory_space_async(memory_space)
    }
}

fn broadcast_batch_control_to_matrix<T: KernelLinalgScalar>(
    value_by_batch: &Tensor<T::Real>,
    batch_dims: &[usize],
    matrix_dims: &[usize],
) -> Result<Tensor<T::Real>> {
    let mut reshape_dims = vec![1, 1];
    reshape_dims.extend_from_slice(batch_dims);
    value_by_batch
        .reshape(&reshape_dims)?
        .broadcast(matrix_dims)
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
        )?);
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
        )?);
    }

    let batch_norms = crate::ad_helpers::matrix_exp_batch_1_norms_tensor(ctx, &input)?;
    let batch_norms_cpu =
        batch_norms.to_memory_space_async(tenferro_device::LogicalMemorySpace::MainMemory)?;
    let batch_norms_slice = batch_norms_cpu.buffer().as_slice().ok_or_else(|| {
        Error::InvalidArgument(
            "matrix_exp: expected batch norm tensor to materialize on host".into(),
        )
    })?;
    let s_by_batch: Vec<usize> = batch_norms_slice
        .iter()
        .map(|&norm_value| {
            let norm_f64: f64 = num_traits::NumCast::from(norm_value).ok_or_else(|| {
                Error::InvalidArgument("matrix_exp: cannot convert 1-norm to f64".into())
            })?;
            Ok(if norm_f64 <= crate::ad_helpers::THETA_13 {
                0
            } else {
                (norm_f64 / crate::ad_helpers::THETA_13)
                    .log2()
                    .ceil()
                    .max(0.0) as usize
            })
        })
        .collect::<Result<_>>()?;
    let s_max = s_by_batch.iter().copied().max().unwrap_or(0);

    let scale_values: Vec<T::Real> = s_by_batch
        .iter()
        .map(|&s| {
            let scale_denom = (1u64 << s.min(63)) as f64;
            crate::ad_helpers::scalar_from::<T::Real>(1.0 / scale_denom)
        })
        .collect::<Result<_>>()?;
    let scale_tensor = if batch_dims.is_empty() {
        crate::prims_bridge::full_like_constant(
            *scale_values.first().ok_or_else(|| {
                Error::InvalidArgument("matrix_exp: missing batch scaling factor".into())
            })?,
            input.dims(),
            input.logical_memory_space(),
        )?
    } else {
        let scale_by_batch = control_tensor_from_host_values::<T::Real>(
            &scale_values,
            batch_dims,
            input.logical_memory_space(),
        )?;
        broadcast_batch_control_to_matrix::<T>(&scale_by_batch, batch_dims, input.dims())?
    };
    let scaled_input =
        <T as crate::prims_bridge::ScaleTensorByRealSameShape<C>>::scale_tensor_by_real_same_shape(
            ctx,
            &input,
            &scale_tensor,
        )?;

    let mut result =
        crate::ad_helpers::matrix_exp_tensor_native(ctx, &scaled_input, n, batch_dims, 0)?;
    let one = crate::ad_helpers::scalar_from::<T::Real>(1.0)?;
    let zero = crate::ad_helpers::scalar_from::<T::Real>(0.0)?;
    for round in 0..s_max {
        let squared = crate::prims_bridge::batched_gemm_with_semiring_tensors(
            ctx, &result, &result, n, n, n,
        )?;
        let mask_values: Vec<T::Real> = s_by_batch
            .iter()
            .map(|&count| if count > round { one } else { zero })
            .collect();
        let mask_tensor = if batch_dims.is_empty() {
            crate::prims_bridge::full_like_constant(
                *mask_values.first().ok_or_else(|| {
                    Error::InvalidArgument("matrix_exp: missing squaring mask value".into())
                })?,
                result.dims(),
                result.logical_memory_space(),
            )?
        } else {
            let mask_by_batch = control_tensor_from_host_values::<T::Real>(
                &mask_values,
                batch_dims,
                result.logical_memory_space(),
            )?;
            broadcast_batch_control_to_matrix::<T>(&mask_by_batch, batch_dims, result.dims())?
        };
        result = crate::ad_helpers::blend_tensor_by_real_mask_same_shape(
            ctx,
            &squared,
            &result,
            &mask_tensor,
        )?;
    }

    Ok(result)
}
