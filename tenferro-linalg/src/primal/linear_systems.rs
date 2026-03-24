use super::*;
use crate::primal::linear_systems_sign::*;
use num_complex::{Complex32, Complex64, ComplexFloat};
use num_traits::{NumCast, One};
use tenferro_algebra::Conjugate;
use tenferro_prims::{TensorMetadataCastPrims, TensorMetadataContextFor, TensorMetadataPrims};
fn inverse_rhs<T: KernelLinalgScalar>(
    n: usize,
    batch_dims: &[usize],
    memory_space: tenferro_device::LogicalMemorySpace,
) -> Result<Tensor<T>> {
    let mut rhs = crate::prims_bridge::identity_matrix(n, memory_space)?;
    for _ in batch_dims {
        rhs = rhs.unsqueeze(-1)?;
    }
    rhs.broadcast(&output_dims(&[n, n], batch_dims))
}

/// Solve a square linear system `A x = b`.
pub fn solve<T: KernelLinalgScalar, C>(
    ctx: &mut C,
    a: &Tensor<T>,
    b: &Tensor<T>,
) -> Result<Tensor<T>>
where
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    <C::Backend as backend::TensorLinalgBackend<T>>::solve(ctx, a, b)
}

/// Solve a square linear system with numerical status information.
pub fn solve_ex<T: KernelLinalgScalar, C>(
    ctx: &mut C,
    a: &Tensor<T>,
    b: &Tensor<T>,
) -> Result<SolveExResult<T>>
where
    T: KernelLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    require_linalg_support::<T, C>(backend::LinalgCapabilityOp::SolveEx, "solve_ex")?;
    let result = <C::Backend as backend::TensorLinalgBackend<T>>::solve_ex(ctx, a, b)?;
    Ok(SolveExResult {
        solution: result.solution,
        info: result.info,
    })
}

/// Compute the inverse of a square matrix.
pub fn inv<T: KernelLinalgScalar, C>(ctx: &mut C, tensor: &Tensor<T>) -> Result<Tensor<T>>
where
    T: KernelLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    require_linalg_support::<T, C>(backend::LinalgCapabilityOp::Inv, "inv")?;

    let (n, batch_dims) = validate_square(tensor)?;
    let rhs = inverse_rhs::<T>(n, batch_dims, tensor.logical_memory_space())?;
    if n == 0 {
        return Ok(rhs);
    }
    solve(ctx, tensor, &rhs)
}

/// Compute the inverse with numerical status information.
pub fn inv_ex<T: KernelLinalgScalar, C>(ctx: &mut C, tensor: &Tensor<T>) -> Result<InvExResult<T>>
where
    T: KernelLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    require_linalg_support::<T, C>(backend::LinalgCapabilityOp::Inv, "inv_ex")?;
    let (n, batch_dims) = validate_square(tensor)?;
    let rhs = inverse_rhs::<T>(n, batch_dims, tensor.logical_memory_space())?;
    if n == 0 {
        return Ok(InvExResult {
            inverse: rhs,
            info: crate::backend::tensor_helpers::info_tensor_from_vec_on_space(
                vec![0; batch_count(batch_dims)],
                batch_dims,
                tensor.logical_memory_space(),
            )?,
        });
    }
    let result = solve_ex(ctx, tensor, &rhs)?;
    Ok(InvExResult {
        inverse: result.solution,
        info: result.info,
    })
}

/// Compute the determinant of a square matrix.
pub fn det<T: KernelLinalgScalar, C>(ctx: &mut C, tensor: &Tensor<T>) -> Result<Tensor<T>>
where
    C: backend::TensorLinalgContextFor<T>
        + tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<T>>
        + tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<T::Real>>
        + TensorMetadataContextFor,
    T: crate::prims_bridge::ScaleTensorByRealSameShape<C>,
    T::Real: Scalar + NumCast + One + 'static,
    C::MetadataBackend: TensorMetadataPrims<Context = C>,
    <C as tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<T::Real>>>::ScalarBackend:
        TensorMetadataCastPrims<T::Real, Context = C>,
    C::Backend: 'static,
{
    require_linalg_support::<T, C>(backend::LinalgCapabilityOp::Det, "det")?;

    let (_n, batch_dims) = validate_square(tensor)?;
    let lu = <C::Backend as backend::TensorLinalgBackend<T>>::lu_factor(ctx, tensor)?;
    let diagonal = lu.u.diagonal(&[(0, 1)])?;
    let kept_axes: Vec<usize> = (0..batch_dims.len()).collect();
    let diagonal_prod = crate::prims_bridge::scalar_reduce_keep_axes(
        ctx,
        &diagonal,
        &kept_axes,
        tenferro_prims::ScalarReductionOp::Prod,
    )?;
    let sign_tensor = lu_permutation_sign_tensor::<T, C>(ctx, &lu.pivots)?;

    <T as crate::prims_bridge::ScaleTensorByRealSameShape<C>>::scale_tensor_by_real_same_shape(
        ctx,
        &diagonal_prod,
        &sign_tensor,
    )
}

#[doc(hidden)]
pub trait SlogdetDispatch<C>: KernelLinalgScalar {
    fn slogdet_dispatch(
        ctx: &mut C,
        tensor: &Tensor<Self>,
    ) -> Result<SlogdetResult<Self, Self::Real>>;
}

fn slogdet_real_impl<T, C>(ctx: &mut C, tensor: &Tensor<T>) -> Result<SlogdetResult<T, T::Real>>
where
    T: KernelLinalgScalar<Real = T> + num_traits::Float,
    C: backend::TensorLinalgContextFor<T>
        + tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<T>>
        + TensorMetadataContextFor,
    C::Backend: 'static,
    <C as tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<T>>>::ScalarBackend:
        'static
            + tenferro_prims::TensorAnalyticPrims<tenferro_algebra::Standard<T>, Context = C>
            + TensorMetadataCastPrims<T, Context = C>,
    C::MetadataBackend: TensorMetadataPrims<Context = C>,
{
    require_linalg_support::<T, C>(backend::LinalgCapabilityOp::Slogdet, "slogdet")?;

    let (_n, batch_dims) = validate_square(tensor)?;
    let lu = <C::Backend as backend::TensorLinalgBackend<T>>::lu_factor(ctx, tensor)?;
    let diagonal = lu.u.diagonal(&[(0, 1)])?;
    let abs_diagonal = crate::prims_bridge::scalar_unary_same_shape(
        ctx,
        &diagonal,
        tenferro_prims::ScalarUnaryOp::Abs,
    )?;
    let logabsdet_factor = crate::prims_bridge::analytic_unary_same_shape(
        ctx,
        &abs_diagonal,
        tenferro_prims::AnalyticUnaryOp::Log,
    )?;
    let kept_axes: Vec<usize> = (0..batch_dims.len()).collect();
    let logabsdet = crate::prims_bridge::scalar_reduce_keep_axes(
        ctx,
        &logabsdet_factor,
        &kept_axes,
        tenferro_prims::ScalarReductionOp::Sum,
    )?;
    let sign_perm = lu_permutation_sign_tensor::<T, C>(ctx, &lu.pivots)?;

    let zero_diagonal = crate::prims_bridge::full_like_constant(
        T::zero(),
        diagonal.dims(),
        tensor.logical_memory_space(),
    )?;
    let negative_mask = crate::prims_bridge::scalar_binary_same_shape(
        ctx,
        &zero_diagonal,
        &diagonal,
        tenferro_prims::ScalarBinaryOp::Greater,
    )?;
    let double_negative = crate::prims_bridge::scalar_binary_same_shape(
        ctx,
        &negative_mask,
        &negative_mask,
        tenferro_prims::ScalarBinaryOp::Add,
    )?;
    let one = crate::prims_bridge::full_like_constant(
        T::one(),
        diagonal.dims(),
        tensor.logical_memory_space(),
    )?;
    let sign_factors = crate::prims_bridge::scalar_binary_same_shape(
        ctx,
        &one,
        &double_negative,
        tenferro_prims::ScalarBinaryOp::Sub,
    )?;
    let sign_from_diag = crate::prims_bridge::scalar_reduce_keep_axes(
        ctx,
        &sign_factors,
        &kept_axes,
        tenferro_prims::ScalarReductionOp::Prod,
    )?;
    let sign = crate::prims_bridge::scalar_binary_same_shape(
        ctx,
        &sign_perm,
        &sign_from_diag,
        tenferro_prims::ScalarBinaryOp::Mul,
    )?;

    Ok(SlogdetResult { sign, logabsdet })
}

fn slogdet_complex_impl<T, R, C>(ctx: &mut C, tensor: &Tensor<T>) -> Result<SlogdetResult<T, R>>
where
    T: KernelLinalgScalar<Real = R> + Conjugate + ComplexFloat<Real = R>,
    T: crate::prims_bridge::ScaleTensorByRealSameShape<C>,
    C: backend::TensorLinalgContextFor<T>
        + tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<R>>
        + tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<T>>
        + tenferro_prims::TensorComplexRealContextFor<T>,
    C: TensorMetadataContextFor,
    C::Backend: 'static,
    R: KernelLinalgScalar<Real = R> + num_traits::Float,
    <C as tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<R>>>::ScalarBackend:
        'static
            + tenferro_prims::TensorAnalyticPrims<tenferro_algebra::Standard<R>, Context = C>
            + TensorMetadataCastPrims<R, Context = C>,
    <C as tenferro_prims::TensorComplexRealContextFor<T>>::ComplexRealBackend:
        tenferro_prims::TensorComplexRealPrims<T, Context = C, Real = R>,
    C::MetadataBackend: TensorMetadataPrims<Context = C>,
{
    require_linalg_support::<T, C>(backend::LinalgCapabilityOp::Slogdet, "slogdet")?;

    let (_n, batch_dims) = validate_square(tensor)?;
    let lu = <C::Backend as backend::TensorLinalgBackend<T>>::lu_factor(ctx, tensor)?;
    let diagonal = lu.u.diagonal(&[(0, 1)])?;
    let abs_diagonal = crate::prims_bridge::complex_real_unary_same_shape(
        ctx,
        &diagonal,
        tenferro_prims::ComplexRealUnaryOp::Abs,
    )?;
    let logabsdet_factor = crate::prims_bridge::analytic_unary_same_shape(
        ctx,
        &abs_diagonal,
        tenferro_prims::AnalyticUnaryOp::Log,
    )?;
    let kept_axes: Vec<usize> = (0..batch_dims.len()).collect();
    let logabsdet = crate::prims_bridge::scalar_reduce_keep_axes(
        ctx,
        &logabsdet_factor,
        &kept_axes,
        tenferro_prims::ScalarReductionOp::Sum,
    )?;
    let sign_perm = lu_permutation_sign_tensor::<T, C>(ctx, &lu.pivots)?;

    let zero_real = crate::prims_bridge::full_like_constant(
        R::zero(),
        abs_diagonal.dims(),
        tensor.logical_memory_space(),
    )?;
    let positive_mask = crate::prims_bridge::scalar_binary_same_shape(
        ctx,
        &abs_diagonal,
        &zero_real,
        tenferro_prims::ScalarBinaryOp::Greater,
    )?;
    let reciprocal_abs = crate::prims_bridge::scalar_unary_same_shape(
        ctx,
        &abs_diagonal,
        tenferro_prims::ScalarUnaryOp::Reciprocal,
    )?;
    let zero_recip = crate::prims_bridge::full_like_constant(
        R::zero(),
        abs_diagonal.dims(),
        tensor.logical_memory_space(),
    )?;
    let safe_recip = crate::prims_bridge::scalar_where_same_shape(
        ctx,
        &positive_mask,
        &reciprocal_abs,
        &zero_recip,
    )?;
    let phase_factors = crate::prims_bridge::complex_scale_same_shape(ctx, &diagonal, &safe_recip)?;
    let sign_from_diag = crate::prims_bridge::scalar_reduce_keep_axes(
        ctx,
        &phase_factors,
        &kept_axes,
        tenferro_prims::ScalarReductionOp::Prod,
    )?;
    let sign = crate::prims_bridge::complex_scale_same_shape(ctx, &sign_from_diag, &sign_perm)?;

    Ok(SlogdetResult { sign, logabsdet })
}

impl<C> SlogdetDispatch<C> for f32
where
    C: backend::TensorLinalgContextFor<f32>
        + tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<f32>>
        + TensorMetadataContextFor,
    C::Backend: 'static,
    <C as tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<f32>>>::ScalarBackend:
        'static
            + tenferro_prims::TensorAnalyticPrims<tenferro_algebra::Standard<f32>, Context = C>
            + TensorMetadataCastPrims<f32, Context = C>,
    C::MetadataBackend: TensorMetadataPrims<Context = C>,
{
    fn slogdet_dispatch(
        ctx: &mut C,
        tensor: &Tensor<Self>,
    ) -> Result<SlogdetResult<Self, Self::Real>> {
        slogdet_real_impl(ctx, tensor)
    }
}

impl<C> SlogdetDispatch<C> for f64
where
    C: backend::TensorLinalgContextFor<f64>
        + tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<f64>>
        + TensorMetadataContextFor,
    C::Backend: 'static,
    <C as tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<f64>>>::ScalarBackend:
        'static
            + tenferro_prims::TensorAnalyticPrims<tenferro_algebra::Standard<f64>, Context = C>
            + TensorMetadataCastPrims<f64, Context = C>,
    C::MetadataBackend: TensorMetadataPrims<Context = C>,
{
    fn slogdet_dispatch(
        ctx: &mut C,
        tensor: &Tensor<Self>,
    ) -> Result<SlogdetResult<Self, Self::Real>> {
        slogdet_real_impl(ctx, tensor)
    }
}

impl<C> SlogdetDispatch<C> for Complex32
where
    Complex32: crate::prims_bridge::ScaleTensorByRealSameShape<C>,
    C: backend::TensorLinalgContextFor<Complex32>
        + tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<f32>>
        + tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<Complex32>>
        + tenferro_prims::TensorComplexRealContextFor<Complex32>
        + TensorMetadataContextFor,
    C::Backend: 'static,
    <C as tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<f32>>>::ScalarBackend:
        'static
            + tenferro_prims::TensorAnalyticPrims<tenferro_algebra::Standard<f32>, Context = C>
            + TensorMetadataCastPrims<f32, Context = C>,
    <C as tenferro_prims::TensorComplexRealContextFor<Complex32>>::ComplexRealBackend:
        tenferro_prims::TensorComplexRealPrims<Complex32, Context = C, Real = f32>,
    C::MetadataBackend: TensorMetadataPrims<Context = C>,
{
    fn slogdet_dispatch(
        ctx: &mut C,
        tensor: &Tensor<Self>,
    ) -> Result<SlogdetResult<Self, Self::Real>> {
        slogdet_complex_impl::<Complex32, f32, C>(ctx, tensor)
    }
}

impl<C> SlogdetDispatch<C> for Complex64
where
    Complex64: crate::prims_bridge::ScaleTensorByRealSameShape<C>,
    C: backend::TensorLinalgContextFor<Complex64>
        + tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<f64>>
        + tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<Complex64>>
        + tenferro_prims::TensorComplexRealContextFor<Complex64>
        + TensorMetadataContextFor,
    C::Backend: 'static,
    <C as tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<f64>>>::ScalarBackend:
        'static
            + tenferro_prims::TensorAnalyticPrims<tenferro_algebra::Standard<f64>, Context = C>
            + TensorMetadataCastPrims<f64, Context = C>,
    <C as tenferro_prims::TensorComplexRealContextFor<Complex64>>::ComplexRealBackend:
        tenferro_prims::TensorComplexRealPrims<Complex64, Context = C, Real = f64>,
    C::MetadataBackend: TensorMetadataPrims<Context = C>,
{
    fn slogdet_dispatch(
        ctx: &mut C,
        tensor: &Tensor<Self>,
    ) -> Result<SlogdetResult<Self, Self::Real>> {
        slogdet_complex_impl::<Complex64, f64, C>(ctx, tensor)
    }
}

/// Compute sign and log-absolute-determinant of a square matrix.
pub fn slogdet<T: KernelLinalgScalar, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
) -> Result<SlogdetResult<T, T::Real>>
where
    T: SlogdetDispatch<C>,
{
    T::slogdet_dispatch(ctx, tensor)
}
