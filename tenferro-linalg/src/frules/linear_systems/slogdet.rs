use super::*;
use num_complex::{Complex32, Complex64, ComplexFloat};
use tenferro_algebra::Conjugate;

#[doc(hidden)]
pub trait SlogdetFruleDispatch<C>: crate::SlogdetDispatch<C> {
    fn slogdet_frule_dispatch(
        ctx: &mut C,
        tensor: &Tensor<Self>,
        tangent: &Tensor<Self>,
    ) -> AdResult<(
        SlogdetResult<Self, Self::Real>,
        SlogdetResult<Self, Self::Real>,
    )>;
}

fn slogdet_frule_real_impl<T, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    tangent: &Tensor<T>,
) -> AdResult<(SlogdetResult<T, T::Real>, SlogdetResult<T, T::Real>)>
where
    T: KernelLinalgScalar<Real = T> + num_traits::Float + crate::SlogdetDispatch<C>,
    C: backend::TensorLinalgContextFor<T>
        + tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<T>>,
    C::Backend: 'static,
    <C as tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<T>>>::ScalarBackend:
        'static + tenferro_prims::TensorAnalyticPrims<tenferro_algebra::Standard<T>, Context = C>,
{
    require_linalg_support::<T, C>(backend::LinalgCapabilityOp::Slogdet, "slogdet_frule")
        .map_err(to_ad_err)?;

    let result = slogdet(ctx, tensor).map_err(to_ad_err)?;
    let (n, _batch_dims) = validate_square(tensor).map_err(to_ad_err)?;
    if n == 0 {
        let dsign = Tensor::zeros(
            result.sign.dims(),
            tensor.logical_memory_space(),
            MemoryOrder::ColumnMajor,
        )
        .map_err(to_ad_err)?;
        let dlog = Tensor::zeros(
            result.logabsdet.dims(),
            tensor.logical_memory_space(),
            MemoryOrder::ColumnMajor,
        )
        .map_err(to_ad_err)?;
        return Ok((
            result,
            SlogdetResult {
                sign: dsign,
                logabsdet: dlog,
            },
        ));
    }

    let a_inv = inv(ctx, tensor).map_err(to_ad_err)?;
    let a_inv_da = prims_bridge::batched_gemm_with_semiring_tensors(ctx, &a_inv, tangent, n, n, n)
        .map_err(to_ad_err)?;
    let dlog = trace_tensor(ctx, &a_inv_da).map_err(to_ad_err)?;
    let dsign = Tensor::zeros(
        result.sign.dims(),
        tensor.logical_memory_space(),
        MemoryOrder::ColumnMajor,
    )
    .map_err(to_ad_err)?;

    let dresult = SlogdetResult {
        sign: dsign,
        logabsdet: dlog,
    };
    Ok((result, dresult))
}

fn slogdet_frule_complex_impl<T, R, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    tangent: &Tensor<T>,
) -> AdResult<(SlogdetResult<T, R>, SlogdetResult<T, R>)>
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
    require_linalg_support::<T, C>(backend::LinalgCapabilityOp::Slogdet, "slogdet_frule")
        .map_err(to_ad_err)?;

    let result = slogdet(ctx, tensor).map_err(to_ad_err)?;
    let (n, _batch_dims) = validate_square(tensor).map_err(to_ad_err)?;
    if n == 0 {
        let dsign = Tensor::zeros(
            result.sign.dims(),
            tensor.logical_memory_space(),
            MemoryOrder::ColumnMajor,
        )
        .map_err(to_ad_err)?;
        let dlog = Tensor::zeros(
            result.logabsdet.dims(),
            tensor.logical_memory_space(),
            MemoryOrder::ColumnMajor,
        )
        .map_err(to_ad_err)?;
        return Ok((
            result,
            SlogdetResult {
                sign: dsign,
                logabsdet: dlog,
            },
        ));
    }

    let a_inv = inv(ctx, tensor).map_err(to_ad_err)?;
    let a_inv_da = prims_bridge::batched_gemm_with_semiring_tensors(ctx, &a_inv, tangent, n, n, n)
        .map_err(to_ad_err)?;
    let trace = trace_tensor(ctx, &a_inv_da).map_err(to_ad_err)?;
    let dlog = prims_bridge::complex_real_unary_same_shape(
        ctx,
        &trace,
        tenferro_prims::ComplexRealUnaryOp::Real,
    )
    .map_err(to_ad_err)?;
    let trace_conj =
        prims_bridge::scalar_unary_same_shape(ctx, &trace, tenferro_prims::ScalarUnaryOp::Conj)
            .map_err(to_ad_err)?;
    let trace_diff = prims_bridge::scalar_binary_same_shape(
        ctx,
        &trace,
        &trace_conj,
        tenferro_prims::ScalarBinaryOp::Sub,
    )
    .map_err(to_ad_err)?;
    let half = prims_bridge::full_like_constant(
        scalar_from::<R>(0.5).map_err(to_ad_err)?,
        trace_diff.dims(),
        tensor.logical_memory_space(),
    )
    .map_err(to_ad_err)?;
    let phase_tangent =
        prims_bridge::complex_scale_same_shape(ctx, &trace_diff, &half).map_err(to_ad_err)?;
    let dsign = prims_bridge::scalar_binary_same_shape(
        ctx,
        &result.sign,
        &phase_tangent,
        tenferro_prims::ScalarBinaryOp::Mul,
    )
    .map_err(to_ad_err)?;

    let dresult = SlogdetResult {
        sign: dsign,
        logabsdet: dlog,
    };
    Ok((result, dresult))
}

impl<C> SlogdetFruleDispatch<C> for f32
where
    f32: crate::SlogdetDispatch<C>,
    C: backend::TensorLinalgContextFor<f32>
        + tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<f32>>,
    C::Backend: 'static,
    <C as tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<f32>>>::ScalarBackend:
        'static + tenferro_prims::TensorAnalyticPrims<tenferro_algebra::Standard<f32>, Context = C>,
{
    fn slogdet_frule_dispatch(
        ctx: &mut C,
        tensor: &Tensor<Self>,
        tangent: &Tensor<Self>,
    ) -> AdResult<(
        SlogdetResult<Self, Self::Real>,
        SlogdetResult<Self, Self::Real>,
    )> {
        slogdet_frule_real_impl(ctx, tensor, tangent)
    }
}

impl<C> SlogdetFruleDispatch<C> for f64
where
    f64: crate::SlogdetDispatch<C>,
    C: backend::TensorLinalgContextFor<f64>
        + tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<f64>>,
    C::Backend: 'static,
    <C as tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<f64>>>::ScalarBackend:
        'static + tenferro_prims::TensorAnalyticPrims<tenferro_algebra::Standard<f64>, Context = C>,
{
    fn slogdet_frule_dispatch(
        ctx: &mut C,
        tensor: &Tensor<Self>,
        tangent: &Tensor<Self>,
    ) -> AdResult<(
        SlogdetResult<Self, Self::Real>,
        SlogdetResult<Self, Self::Real>,
    )> {
        slogdet_frule_real_impl(ctx, tensor, tangent)
    }
}

impl<C> SlogdetFruleDispatch<C> for Complex32
where
    Complex32: crate::SlogdetDispatch<C> + crate::prims_bridge::ScaleTensorByRealSameShape<C>,
    C: backend::TensorLinalgContextFor<Complex32>
        + tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<f32>>
        + tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<Complex32>>
        + tenferro_prims::TensorComplexRealContextFor<Complex32>
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
    fn slogdet_frule_dispatch(
        ctx: &mut C,
        tensor: &Tensor<Self>,
        tangent: &Tensor<Self>,
    ) -> AdResult<(
        SlogdetResult<Self, Self::Real>,
        SlogdetResult<Self, Self::Real>,
    )> {
        slogdet_frule_complex_impl::<Complex32, f32, C>(ctx, tensor, tangent)
    }
}

impl<C> SlogdetFruleDispatch<C> for Complex64
where
    Complex64: crate::SlogdetDispatch<C> + crate::prims_bridge::ScaleTensorByRealSameShape<C>,
    C: backend::TensorLinalgContextFor<Complex64>
        + tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<f64>>
        + tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<Complex64>>
        + tenferro_prims::TensorComplexRealContextFor<Complex64>
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
    fn slogdet_frule_dispatch(
        ctx: &mut C,
        tensor: &Tensor<Self>,
        tangent: &Tensor<Self>,
    ) -> AdResult<(
        SlogdetResult<Self, Self::Real>,
        SlogdetResult<Self, Self::Real>,
    )> {
        slogdet_frule_complex_impl::<Complex64, f64, C>(ctx, tensor, tangent)
    }
}

/// Forward-mode AD rule for slogdet (JVP / pushforward).
///
/// # Examples
///
/// ```
/// use tenferro_linalg::slogdet_frule;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::eye(3, mem, col).unwrap();
/// let da = Tensor::<f64>::ones(&[3, 3], mem, col).unwrap();
/// let (result, dresult) = slogdet_frule(&mut ctx, &a, &da).unwrap();
/// ```
pub fn slogdet_frule<T, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    tangent: &Tensor<T>,
) -> AdResult<(SlogdetResult<T, T::Real>, SlogdetResult<T, T::Real>)>
where
    T: SlogdetFruleDispatch<C>,
{
    T::slogdet_frule_dispatch(ctx, tensor, tangent)
}
