use super::*;
use tenferro_algebra::Conjugate;

/// Reverse-mode AD rule for matrix inverse (VJP / pullback).
///
/// `Ā = -A⁻ᵀ · cotangent · A⁻ᵀ`.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::inv_rrule;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::eye(3, mem, col).unwrap();
/// let cotangent = Tensor::<f64>::ones(&[3, 3], mem, col).unwrap();
/// let grad_a = inv_rrule(&mut ctx, &a, &cotangent).unwrap();
/// ```
pub fn inv_rrule<T: KernelLinalgScalar + tenferro_algebra::Conjugate, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    cotangent: &Tensor<T>,
) -> AdResult<Tensor<T>>
where
    T: KernelLinalgScalar + tenferro_algebra::Conjugate,
    C: backend::TensorLinalgContextFor<T> + tenferro_prims::TensorResolveConjContextFor<T>,
    C::Backend: 'static,
{
    require_linalg_support::<T, C>(backend::LinalgCapabilityOp::Inv, "inv_rrule")
        .map_err(to_ad_err)?;

    // dA = -B^H dB B^H where B = A^{-1}
    let b_inv = inv(ctx, tensor).map_err(to_ad_err)?;
    let (n, _) = validate_square(tensor).map_err(to_ad_err)?;

    let bt = matrix_adjoint_eager(ctx, &b_inv).map_err(to_ad_err)?;
    let bt_db = prims_bridge::batched_gemm_with_semiring_tensors(ctx, &bt, cotangent, n, n, n)
        .map_err(to_ad_err)?;
    let grad_a = prims_bridge::batched_gemm_alpha_tensors(ctx, &bt_db, &bt, n, n, n, -T::one())
        .map_err(to_ad_err)?;

    Ok(grad_a)
}

/// Reverse-mode AD rule for determinant (VJP / pullback).
///
/// `Ā = det(A) · cotangent · A⁻ᵀ`.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::det_rrule;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::eye(3, mem, col).unwrap();
/// let cotangent = Tensor::<f64>::ones(&[], mem, col).unwrap();
/// let grad_a = det_rrule(&mut ctx, &a, &cotangent).unwrap();
/// ```
pub fn det_rrule<T: KernelLinalgScalar + Conjugate, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    cotangent: &Tensor<T>,
) -> AdResult<Tensor<T>>
where
    T: KernelLinalgScalar + crate::prims_bridge::ScaleTensorByRealSameShape<C>,
    C: backend::TensorLinalgContextFor<T>
        + tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<T>>
        + tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<T::Real>>
        + tenferro_prims::TensorMetadataContextFor
        + tenferro_prims::TensorResolveConjContextFor<T>,
    C::Backend: 'static,
    C::MetadataBackend: tenferro_prims::TensorMetadataPrims<Context = C>,
    <C as tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<T>>>::ScalarBackend:
        tenferro_prims::TensorMetadataCastPrims<T, Context = C>,
    <C as tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<T::Real>>>::ScalarBackend:
        tenferro_prims::TensorMetadataCastPrims<T::Real, Context = C>,
{
    require_linalg_support::<T, C>(backend::LinalgCapabilityOp::Det, "det_rrule")
        .map_err(to_ad_err)?;

    let (n, _) = validate_square(tensor).map_err(to_ad_err)?;
    if n == 0 {
        return Tensor::zeros(
            tensor.dims(),
            tensor.logical_memory_space(),
            MemoryOrder::ColumnMajor,
        )
        .map_err(to_ad_err);
    }

    // dA = cotangent * conj(det(A)) * A^{-H}
    let det_val = det(ctx, tensor).map_err(to_ad_err)?;
    let a_inv = inv(ctx, tensor).map_err(to_ad_err)?;
    let a_inv_h = matrix_adjoint_eager(ctx, &a_inv).map_err(to_ad_err)?;
    let det_conj =
        prims_bridge::scalar_unary_same_shape(ctx, &det_val, tenferro_prims::ScalarUnaryOp::Conj)
            .map_err(to_ad_err)?;

    // scale = cotangent * conj(det(A)), shape [batch...]
    let scale = prims_bridge::scalar_binary_same_shape(
        ctx,
        cotangent,
        &det_conj,
        tenferro_prims::ScalarBinaryOp::Mul,
    )
    .map_err(to_ad_err)?;

    // broadcast scale [batch...] → [1, 1, batch...] → [n, n, batch...]
    let scale_expanded = scale
        .unsqueeze(0)
        .map_err(to_ad_err)?
        .unsqueeze(0)
        .map_err(to_ad_err)?
        .broadcast(a_inv_h.dims())
        .map_err(to_ad_err)?;
    let grad_a = prims_bridge::scalar_binary_same_shape(
        ctx,
        &scale_expanded,
        &a_inv_h,
        tenferro_prims::ScalarBinaryOp::Mul,
    )
    .map_err(to_ad_err)?;

    Ok(grad_a)
}
