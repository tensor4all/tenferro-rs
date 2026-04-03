use super::*;

/// Forward-mode AD rule for matrix inverse (JVP / pushforward).
///
/// # Examples
///
/// ```
/// use tenferro_linalg::inv_frule;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::eye(3, mem, col).unwrap();
/// let da = Tensor::<f64>::ones(&[3, 3], mem, col).unwrap();
/// let (a_inv, da_inv) = inv_frule(&mut ctx, &a, &da).unwrap();
/// ```
pub fn inv_frule<T: KernelLinalgScalar, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    tangent: &Tensor<T>,
) -> AdResult<(Tensor<T>, Tensor<T>)>
where
    T: KernelLinalgScalar,
    C: backend::TensorLinalgContextFor<T>
        + tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<T>>,
    C::Backend: 'static,
    <C as tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<T>>>::ScalarBackend:
        'static + tenferro_prims::TensorAnalyticPrims<tenferro_algebra::Standard<T>, Context = C>,
{
    require_linalg_support::<T, C>(backend::LinalgCapabilityOp::Inv, "inv_frule")
        .map_err(to_ad_err)?;

    // dB = -B dA B where B = A^{-1}
    let b_inv = inv(ctx, tensor).map_err(to_ad_err)?;
    let (n, _) = validate_square(tensor).map_err(to_ad_err)?;

    let b_da = prims_bridge::batched_gemm_with_semiring_tensors(ctx, &b_inv, tangent, n, n, n)
        .map_err(to_ad_err)?;
    let db = prims_bridge::batched_gemm_alpha_tensors(ctx, &b_da, &b_inv, n, n, n, -T::one())
        .map_err(to_ad_err)?;

    Ok((b_inv, db))
}

/// Forward-mode AD rule for determinant (JVP / pushforward).
///
/// # Examples
///
/// ```
/// use tenferro_linalg::det_frule;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::eye(3, mem, col).unwrap();
/// let da = Tensor::<f64>::ones(&[3, 3], mem, col).unwrap();
/// let (d, dd) = det_frule(&mut ctx, &a, &da).unwrap();
/// ```
pub fn det_frule<T: KernelLinalgScalar, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    tangent: &Tensor<T>,
) -> AdResult<(Tensor<T>, Tensor<T>)>
where
    T: KernelLinalgScalar + crate::prims_bridge::ScaleTensorByRealSameShape<C>,
    C: backend::TensorLinalgContextFor<T>
        + tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<T>>
        + tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<T::Real>>
        + tenferro_prims::TensorMetadataContextFor,
    C::Backend: 'static,
    C::MetadataBackend: tenferro_prims::TensorMetadataPrims<Context = C>,
    <C as tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<T>>>::ScalarBackend:
        tenferro_prims::TensorMetadataCastPrims<T, Context = C>,
    <C as tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<T::Real>>>::ScalarBackend:
        tenferro_prims::TensorMetadataCastPrims<T::Real, Context = C>,
{
    require_linalg_support::<T, C>(backend::LinalgCapabilityOp::Det, "det_frule")
        .map_err(to_ad_err)?;

    // d(det) = det(A) * tr(A^{-1} dA)
    let d = det(ctx, tensor).map_err(to_ad_err)?;
    let (n, _) = validate_square(tensor).map_err(to_ad_err)?;
    if n == 0 {
        let dd = Tensor::zeros(
            d.dims(),
            tensor.logical_memory_space(),
            MemoryOrder::ColumnMajor,
        )
        .map_err(to_ad_err)?;
        return Ok((d, dd));
    }

    let a_inv = inv(ctx, tensor).map_err(to_ad_err)?;
    let a_inv_da = prims_bridge::batched_gemm_with_semiring_tensors(ctx, &a_inv, tangent, n, n, n)
        .map_err(to_ad_err)?;
    let trace = trace_tensor(ctx, &a_inv_da).map_err(to_ad_err)?;

    // dd = det(A) * trace
    let dd = prims_bridge::scalar_binary_same_shape(
        ctx,
        &d,
        &trace,
        tenferro_prims::ScalarBinaryOp::Mul,
    )
    .map_err(to_ad_err)?;

    Ok((d, dd))
}
