use super::*;
use num_complex::{Complex32, Complex64, ComplexFloat};
use tenferro_algebra::Conjugate;

#[doc(hidden)]
pub trait SlogdetRruleDispatch<C>: crate::SlogdetDispatch<C> {
    fn slogdet_rrule_dispatch(
        ctx: &mut C,
        tensor: &Tensor<Self>,
        cotangent: &SlogdetCotangent<Self, Self::Real>,
    ) -> AdResult<Tensor<Self>>;
}

fn slogdet_rrule_real_impl<T, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    cotangent: &SlogdetCotangent<T, T>,
) -> AdResult<Tensor<T>>
where
    T: KernelLinalgScalar<Real = T> + num_traits::Float + crate::SlogdetDispatch<C>,
    C: backend::TensorLinalgContextFor<T>
        + tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<T>>,
    C::Backend: 'static,
{
    require_linalg_support::<T, C>(backend::LinalgCapabilityOp::Slogdet, "slogdet_rrule")
        .map_err(to_ad_err)?;

    let (n, batch_dims) = validate_square(tensor).map_err(to_ad_err)?;
    if n == 0 {
        let dims = output_dims(&[n, n], batch_dims);
        return Tensor::zeros(
            &dims,
            tensor.logical_memory_space(),
            MemoryOrder::ColumnMajor,
        )
        .map_err(to_ad_err);
    }

    if let Some(ref dlog) = cotangent.logabsdet {
        let a_inv = inv(ctx, tensor).map_err(to_ad_err)?;
        let a_inv_t = matrix_transpose(&a_inv).map_err(to_ad_err)?;
        let dlog_expanded = dlog
            .unsqueeze(0)
            .map_err(to_ad_err)?
            .unsqueeze(0)
            .map_err(to_ad_err)?
            .broadcast(a_inv_t.dims())
            .map_err(to_ad_err)?;
        let grad_a = prims_bridge::scalar_binary_same_shape(
            ctx,
            &dlog_expanded,
            &a_inv_t,
            tenferro_prims::ScalarBinaryOp::Mul,
        )
        .map_err(to_ad_err)?;
        Ok(grad_a)
    } else {
        let dims = output_dims(&[n, n], batch_dims);
        Tensor::zeros(
            &dims,
            tensor.logical_memory_space(),
            MemoryOrder::ColumnMajor,
        )
        .map_err(to_ad_err)
    }
}

fn slogdet_rrule_complex_impl<T, R, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    cotangent: &SlogdetCotangent<T, R>,
) -> AdResult<Tensor<T>>
where
    T: KernelLinalgScalar<Real = R>
        + Conjugate
        + ComplexFloat<Real = R>
        + crate::SlogdetDispatch<C>,
    T: crate::prims_bridge::ScaleTensorByRealSameShape<C>,
    C: backend::TensorLinalgContextFor<T>
        + tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<R>>
        + tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<T>>
        + tenferro_prims::TensorComplexRealContextFor<T>
        + tenferro_prims::TensorResolveConjContextFor<T>
        + tenferro_prims::TensorMetadataContextFor,
    C::Backend: 'static,
    R: KernelLinalgScalar<Real = R> + num_traits::Float,
    <C as tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<R>>>::ScalarBackend:
        'static
            + tenferro_prims::TensorAnalyticPrims<tenferro_algebra::Standard<R>, Context = C>
            + tenferro_prims::TensorMetadataCastPrims<R, Context = C>,
    <C as tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<T>>>::ScalarBackend:
        tenferro_prims::TensorMetadataCastPrims<T, Context = C>,
    <C as tenferro_prims::TensorComplexRealContextFor<T>>::ComplexRealBackend:
        tenferro_prims::TensorComplexRealPrims<T, Context = C, Real = R>,
    <C as tenferro_prims::TensorMetadataContextFor>::MetadataBackend:
        tenferro_prims::TensorMetadataPrims<Context = C>,
{
    require_linalg_support::<T, C>(backend::LinalgCapabilityOp::Slogdet, "slogdet_rrule")
        .map_err(to_ad_err)?;

    let (n, batch_dims) = validate_square(tensor).map_err(to_ad_err)?;
    if n == 0 {
        let dims = output_dims(&[n, n], batch_dims);
        return Tensor::zeros(
            &dims,
            tensor.logical_memory_space(),
            MemoryOrder::ColumnMajor,
        )
        .map_err(to_ad_err);
    }

    let a_inv = inv(ctx, tensor).map_err(to_ad_err)?;
    let a_inv_h = matrix_adjoint_eager(ctx, &a_inv).map_err(to_ad_err)?;
    let dims = output_dims(&[n, n], batch_dims);
    let mut grad_a = Tensor::zeros(
        &dims,
        tensor.logical_memory_space(),
        MemoryOrder::ColumnMajor,
    )
    .map_err(to_ad_err)?;

    if let Some(ref dlog) = cotangent.logabsdet {
        let dlog_expanded = dlog
            .unsqueeze(0)
            .map_err(to_ad_err)?
            .unsqueeze(0)
            .map_err(to_ad_err)?
            .broadcast(a_inv_h.dims())
            .map_err(to_ad_err)?;
        grad_a = prims_bridge::complex_scale_same_shape(ctx, &a_inv_h, &dlog_expanded)
            .map_err(to_ad_err)?;
    }

    if let Some(ref dsign) = cotangent.sign {
        let result = slogdet(ctx, tensor).map_err(to_ad_err)?;
        let dsign_conj =
            prims_bridge::scalar_unary_same_shape(ctx, dsign, tenferro_prims::ScalarUnaryOp::Conj)
                .map_err(to_ad_err)?;
        let alpha = prims_bridge::scalar_binary_same_shape(
            ctx,
            &dsign_conj,
            &result.sign,
            tenferro_prims::ScalarBinaryOp::Mul,
        )
        .map_err(to_ad_err)?;
        let alpha_conj =
            prims_bridge::scalar_unary_same_shape(ctx, &alpha, tenferro_prims::ScalarUnaryOp::Conj)
                .map_err(to_ad_err)?;
        let alpha_skew = prims_bridge::scalar_binary_same_shape(
            ctx,
            &alpha_conj,
            &alpha,
            tenferro_prims::ScalarBinaryOp::Sub,
        )
        .map_err(to_ad_err)?;
        let half = prims_bridge::full_like_constant(
            scalar_from::<R>(0.5).map_err(to_ad_err)?,
            alpha_skew.dims(),
            tensor.logical_memory_space(),
        )
        .map_err(to_ad_err)?;
        let sign_scale =
            prims_bridge::complex_scale_same_shape(ctx, &alpha_skew, &half).map_err(to_ad_err)?;
        let sign_scale_expanded = sign_scale
            .unsqueeze(0)
            .map_err(to_ad_err)?
            .unsqueeze(0)
            .map_err(to_ad_err)?
            .broadcast(a_inv_h.dims())
            .map_err(to_ad_err)?;
        let sign_grad = prims_bridge::scalar_binary_same_shape(
            ctx,
            &sign_scale_expanded,
            &a_inv_h,
            tenferro_prims::ScalarBinaryOp::Mul,
        )
        .map_err(to_ad_err)?;
        grad_a = prims_bridge::scalar_binary_same_shape(
            ctx,
            &grad_a,
            &sign_grad,
            tenferro_prims::ScalarBinaryOp::Add,
        )
        .map_err(to_ad_err)?;
    }

    Ok(grad_a)
}

/// Reverse-mode AD rule for slogdet (VJP / pullback).
///
/// `Ā = cotangent_logabsdet · A⁻ᵀ`.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::{slogdet_rrule, SlogdetCotangent};
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::eye(3, mem, col).unwrap();
/// let cotangent = SlogdetCotangent {
///     logabsdet: Some(Tensor::ones(&[], mem, col).unwrap()),
/// };
/// let grad_a = slogdet_rrule(&mut ctx, &a, &cotangent).unwrap();
/// ```
pub fn slogdet_rrule<T, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    cotangent: &SlogdetCotangent<T, T::Real>,
) -> AdResult<Tensor<T>>
where
    T: SlogdetRruleDispatch<C>,
{
    T::slogdet_rrule_dispatch(ctx, tensor, cotangent)
}

impl<C> SlogdetRruleDispatch<C> for f32
where
    f32: crate::SlogdetDispatch<C>,
    C: backend::TensorLinalgContextFor<f32>
        + tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<f32>>,
    C::Backend: 'static,
{
    fn slogdet_rrule_dispatch(
        ctx: &mut C,
        tensor: &Tensor<Self>,
        cotangent: &SlogdetCotangent<Self, Self::Real>,
    ) -> AdResult<Tensor<Self>> {
        slogdet_rrule_real_impl(ctx, tensor, cotangent)
    }
}

impl<C> SlogdetRruleDispatch<C> for f64
where
    f64: crate::SlogdetDispatch<C>,
    C: backend::TensorLinalgContextFor<f64>
        + tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<f64>>,
    C::Backend: 'static,
{
    fn slogdet_rrule_dispatch(
        ctx: &mut C,
        tensor: &Tensor<Self>,
        cotangent: &SlogdetCotangent<Self, Self::Real>,
    ) -> AdResult<Tensor<Self>> {
        slogdet_rrule_real_impl(ctx, tensor, cotangent)
    }
}

impl<C> SlogdetRruleDispatch<C> for Complex32
where
    Complex32: crate::SlogdetDispatch<C> + crate::prims_bridge::ScaleTensorByRealSameShape<C>,
    C: backend::TensorLinalgContextFor<Complex32>
        + tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<f32>>
        + tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<Complex32>>
        + tenferro_prims::TensorComplexRealContextFor<Complex32>
        + tenferro_prims::TensorResolveConjContextFor<Complex32>
        + tenferro_prims::TensorMetadataContextFor,
    C::Backend: 'static,
    <C as tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<f32>>>::ScalarBackend:
        'static
            + tenferro_prims::TensorAnalyticPrims<tenferro_algebra::Standard<f32>, Context = C>
            + tenferro_prims::TensorMetadataCastPrims<f32, Context = C>,
    <C as tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<Complex32>>>::ScalarBackend:
        tenferro_prims::TensorMetadataCastPrims<Complex32, Context = C>,
    <C as tenferro_prims::TensorComplexRealContextFor<Complex32>>::ComplexRealBackend:
        tenferro_prims::TensorComplexRealPrims<Complex32, Context = C, Real = f32>,
    <C as tenferro_prims::TensorMetadataContextFor>::MetadataBackend:
        tenferro_prims::TensorMetadataPrims<Context = C>,
{
    fn slogdet_rrule_dispatch(
        ctx: &mut C,
        tensor: &Tensor<Self>,
        cotangent: &SlogdetCotangent<Self, Self::Real>,
    ) -> AdResult<Tensor<Self>> {
        slogdet_rrule_complex_impl::<Complex32, f32, C>(ctx, tensor, cotangent)
    }
}

impl<C> SlogdetRruleDispatch<C> for Complex64
where
    Complex64: crate::SlogdetDispatch<C> + crate::prims_bridge::ScaleTensorByRealSameShape<C>,
    C: backend::TensorLinalgContextFor<Complex64>
        + tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<f64>>
        + tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<Complex64>>
        + tenferro_prims::TensorComplexRealContextFor<Complex64>
        + tenferro_prims::TensorResolveConjContextFor<Complex64>
        + tenferro_prims::TensorMetadataContextFor,
    C::Backend: 'static,
    <C as tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<f64>>>::ScalarBackend:
        'static
            + tenferro_prims::TensorAnalyticPrims<tenferro_algebra::Standard<f64>, Context = C>
            + tenferro_prims::TensorMetadataCastPrims<f64, Context = C>,
    <C as tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<Complex64>>>::ScalarBackend:
        tenferro_prims::TensorMetadataCastPrims<Complex64, Context = C>,
    <C as tenferro_prims::TensorComplexRealContextFor<Complex64>>::ComplexRealBackend:
        tenferro_prims::TensorComplexRealPrims<Complex64, Context = C, Real = f64>,
    <C as tenferro_prims::TensorMetadataContextFor>::MetadataBackend:
        tenferro_prims::TensorMetadataPrims<Context = C>,
{
    fn slogdet_rrule_dispatch(
        ctx: &mut C,
        tensor: &Tensor<Self>,
        cotangent: &SlogdetCotangent<Self, Self::Real>,
    ) -> AdResult<Tensor<Self>> {
        slogdet_rrule_complex_impl::<Complex64, f64, C>(ctx, tensor, cotangent)
    }
}
