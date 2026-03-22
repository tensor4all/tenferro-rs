use super::*;
use num_complex::{Complex32, Complex64, ComplexFloat};

pub(crate) fn norm_real_impl<T, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    kind: NormKind,
) -> Result<Tensor<T>>
where
    T: KernelLinalgScalar<Real = T> + num_traits::Float,
    C: backend::TensorLinalgContextFor<T>
        + tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<T>>,
    <C as tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<T>>>::ScalarBackend:
        tenferro_prims::TensorAnalyticPrims<tenferro_algebra::Standard<T>, Context = C>,
    C::Backend: 'static,
{
    require_linalg_support::<T, C>(backend::LinalgCapabilityOp::Norm, "norm")?;

    if tensor.ndim() == 1 {
        match kind {
            NormKind::L1 => {
                let abs = crate::prims_bridge::scalar_unary_same_shape(
                    ctx,
                    tensor,
                    tenferro_prims::ScalarUnaryOp::Abs,
                )?;
                return crate::prims_bridge::scalar_reduce_keep_axes(
                    ctx,
                    &abs,
                    &[],
                    tenferro_prims::ScalarReductionOp::Sum,
                );
            }
            NormKind::Inf => {
                if tensor.dims()[0] == 0 {
                    return crate::prims_bridge::full_like_constant(
                        T::zero(),
                        &[],
                        tensor.logical_memory_space(),
                    );
                }
                let abs = crate::prims_bridge::scalar_unary_same_shape(
                    ctx,
                    tensor,
                    tenferro_prims::ScalarUnaryOp::Abs,
                )?;
                return crate::prims_bridge::scalar_reduce_keep_axes(
                    ctx,
                    &abs,
                    &[],
                    tenferro_prims::ScalarReductionOp::Max,
                );
            }
            NormKind::Nuclear | NormKind::Spectral => {
                return Err(matrix_only_norm_kind_error(kind));
            }
            NormKind::Fro => {
                let squared = crate::prims_bridge::scalar_binary_same_shape(
                    ctx,
                    tensor,
                    tensor,
                    tenferro_prims::ScalarBinaryOp::Mul,
                )?;
                let squared_sum = crate::prims_bridge::scalar_reduce_keep_axes(
                    ctx,
                    &squared,
                    &[],
                    tenferro_prims::ScalarReductionOp::Sum,
                )?;
                return crate::prims_bridge::analytic_unary_same_shape(
                    ctx,
                    &squared_sum,
                    tenferro_prims::AnalyticUnaryOp::Sqrt,
                );
            }
            NormKind::Lp(_) => {}
        }

        let NormKind::Lp(p) = kind else {
            unreachable!();
        };
        if p < 1.0 {
            return Err(invalid_vector_lp_exponent_error(p));
        }
        let p_t = scalar_from::<T>(p)?;
        let abs = crate::prims_bridge::scalar_unary_same_shape(
            ctx,
            tensor,
            tenferro_prims::ScalarUnaryOp::Abs,
        )?;
        let p_tensor = crate::prims_bridge::full_like_constant(
            p_t,
            tensor.dims(),
            tensor.logical_memory_space(),
        )?;
        let abs_pow_p = crate::prims_bridge::analytic_binary_same_shape(
            ctx,
            &abs,
            &p_tensor,
            tenferro_prims::AnalyticBinaryOp::Pow,
        )?;
        let sum = crate::prims_bridge::scalar_reduce_keep_axes(
            ctx,
            &abs_pow_p,
            &[],
            tenferro_prims::ScalarReductionOp::Sum,
        )?;
        let inv_p_tensor = crate::prims_bridge::full_like_constant(
            T::one() / p_t,
            &[],
            sum.logical_memory_space(),
        )?;
        return crate::prims_bridge::analytic_binary_same_shape(
            ctx,
            &sum,
            &inv_p_tensor,
            tenferro_prims::AnalyticBinaryOp::Pow,
        );
    }

    let (m, n, batch_dims) = validate_2d(tensor)?;
    let out_dims = if batch_dims.is_empty() {
        vec![]
    } else {
        batch_dims.to_vec()
    };

    match kind {
        NormKind::Nuclear => {
            let singular_values = svdvals(ctx, tensor)?;
            let kept_axes: Vec<usize> = (1..singular_values.ndim()).collect();
            crate::prims_bridge::scalar_reduce_keep_axes(
                ctx,
                &singular_values,
                &kept_axes,
                tenferro_prims::ScalarReductionOp::Sum,
            )
        }
        NormKind::Spectral => {
            let singular_values = svdvals(ctx, tensor)?;
            let kept_axes: Vec<usize> = (1..singular_values.ndim()).collect();
            crate::prims_bridge::scalar_reduce_keep_axes(
                ctx,
                &singular_values,
                &kept_axes,
                tenferro_prims::ScalarReductionOp::Max,
            )
        }
        NormKind::L1 => {
            if m == 0 || n == 0 {
                return crate::prims_bridge::full_like_constant(
                    T::zero(),
                    &out_dims,
                    tensor.logical_memory_space(),
                );
            }
            let abs = crate::prims_bridge::scalar_unary_same_shape(
                ctx,
                tensor,
                tenferro_prims::ScalarUnaryOp::Abs,
            )?;
            let kept_axes: Vec<usize> = std::iter::once(1).chain(2..abs.ndim()).collect();
            let column_sums = crate::prims_bridge::scalar_reduce_keep_axes(
                ctx,
                &abs,
                &kept_axes,
                tenferro_prims::ScalarReductionOp::Sum,
            )?;
            let batch_axes: Vec<usize> = (1..column_sums.ndim()).collect();
            crate::prims_bridge::scalar_reduce_keep_axes(
                ctx,
                &column_sums,
                &batch_axes,
                tenferro_prims::ScalarReductionOp::Max,
            )
        }
        NormKind::Inf => {
            if m == 0 || n == 0 {
                return crate::prims_bridge::full_like_constant(
                    T::zero(),
                    &out_dims,
                    tensor.logical_memory_space(),
                );
            }
            let abs = crate::prims_bridge::scalar_unary_same_shape(
                ctx,
                tensor,
                tenferro_prims::ScalarUnaryOp::Abs,
            )?;
            let kept_axes: Vec<usize> = std::iter::once(0).chain(2..abs.ndim()).collect();
            let row_sums = crate::prims_bridge::scalar_reduce_keep_axes(
                ctx,
                &abs,
                &kept_axes,
                tenferro_prims::ScalarReductionOp::Sum,
            )?;
            let batch_axes: Vec<usize> = (1..row_sums.ndim()).collect();
            crate::prims_bridge::scalar_reduce_keep_axes(
                ctx,
                &row_sums,
                &batch_axes,
                tenferro_prims::ScalarReductionOp::Max,
            )
        }
        NormKind::Fro => {
            let squared = crate::prims_bridge::scalar_binary_same_shape(
                ctx,
                tensor,
                tensor,
                tenferro_prims::ScalarBinaryOp::Mul,
            )?;
            let kept_axes: Vec<usize> = (2..squared.ndim()).collect();
            let squared_sum = crate::prims_bridge::scalar_reduce_keep_axes(
                ctx,
                &squared,
                &kept_axes,
                tenferro_prims::ScalarReductionOp::Sum,
            )?;
            crate::prims_bridge::analytic_unary_same_shape(
                ctx,
                &squared_sum,
                tenferro_prims::AnalyticUnaryOp::Sqrt,
            )
        }
        _ => Err(Error::InvalidArgument(format!(
            "norm kind {kind:?} not yet implemented"
        ))),
    }
}

fn norm_complex_impl<T, R, C>(ctx: &mut C, tensor: &Tensor<T>, kind: NormKind) -> Result<Tensor<R>>
where
    T: KernelLinalgScalar<Real = R> + ComplexFloat<Real = R>,
    R: KernelLinalgScalar + num_traits::Float,
    C: backend::TensorLinalgContextFor<T>
        + tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<R>>
        + tenferro_prims::TensorComplexRealContextFor<T>,
    <C as tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<R>>>::ScalarBackend:
        tenferro_prims::TensorAnalyticPrims<tenferro_algebra::Standard<R>, Context = C>,
    C::ComplexRealBackend: tenferro_prims::TensorComplexRealPrims<T, Context = C, Real = R>,
    C::Backend: 'static,
{
    require_linalg_support::<T, C>(backend::LinalgCapabilityOp::Norm, "norm")?;

    if tensor.ndim() == 1 {
        match kind {
            NormKind::L1 => {
                return crate::prims_bridge::complex_real_reduce_keep_axes(
                    ctx,
                    tensor,
                    tenferro_prims::ComplexRealUnaryOp::Abs,
                    &[],
                    tenferro_prims::ScalarReductionOp::Sum,
                );
            }
            NormKind::Inf => {
                if tensor.dims()[0] == 0 {
                    return crate::prims_bridge::full_like_constant(
                        R::zero(),
                        &[],
                        tensor.logical_memory_space(),
                    );
                }
                return crate::prims_bridge::complex_real_reduce_keep_axes(
                    ctx,
                    tensor,
                    tenferro_prims::ComplexRealUnaryOp::Abs,
                    &[],
                    tenferro_prims::ScalarReductionOp::Max,
                );
            }
            NormKind::Nuclear | NormKind::Spectral => {
                return Err(matrix_only_norm_kind_error(kind));
            }
            NormKind::Fro => {
                let abs = crate::prims_bridge::complex_real_unary_same_shape(
                    ctx,
                    tensor,
                    tenferro_prims::ComplexRealUnaryOp::Abs,
                )?;
                let squared = crate::prims_bridge::scalar_binary_same_shape(
                    ctx,
                    &abs,
                    &abs,
                    tenferro_prims::ScalarBinaryOp::Mul,
                )?;
                let squared_sum = crate::prims_bridge::scalar_reduce_keep_axes(
                    ctx,
                    &squared,
                    &[],
                    tenferro_prims::ScalarReductionOp::Sum,
                )?;
                return crate::prims_bridge::analytic_unary_same_shape(
                    ctx,
                    &squared_sum,
                    tenferro_prims::AnalyticUnaryOp::Sqrt,
                );
            }
            NormKind::Lp(_) => {}
        }

        let NormKind::Lp(p) = kind else {
            unreachable!();
        };
        if p < 1.0 {
            return Err(invalid_vector_lp_exponent_error(p));
        }
        let p_t = scalar_from::<R>(p)?;
        let abs = crate::prims_bridge::complex_real_unary_same_shape(
            ctx,
            tensor,
            tenferro_prims::ComplexRealUnaryOp::Abs,
        )?;
        let p_tensor = crate::prims_bridge::full_like_constant(
            p_t,
            tensor.dims(),
            tensor.logical_memory_space(),
        )?;
        let abs_pow_p = crate::prims_bridge::analytic_binary_same_shape(
            ctx,
            &abs,
            &p_tensor,
            tenferro_prims::AnalyticBinaryOp::Pow,
        )?;
        let sum = crate::prims_bridge::scalar_reduce_keep_axes(
            ctx,
            &abs_pow_p,
            &[],
            tenferro_prims::ScalarReductionOp::Sum,
        )?;
        let inv_p_tensor = crate::prims_bridge::full_like_constant(
            R::one() / p_t,
            &[],
            sum.logical_memory_space(),
        )?;
        return crate::prims_bridge::analytic_binary_same_shape(
            ctx,
            &sum,
            &inv_p_tensor,
            tenferro_prims::AnalyticBinaryOp::Pow,
        );
    }

    let (m, n, batch_dims) = validate_2d(tensor)?;
    let out_dims = if batch_dims.is_empty() {
        vec![]
    } else {
        batch_dims.to_vec()
    };

    match kind {
        NormKind::Nuclear => {
            let singular_values = svdvals(ctx, tensor)?;
            let kept_axes: Vec<usize> = (1..singular_values.ndim()).collect();
            crate::prims_bridge::scalar_reduce_keep_axes(
                ctx,
                &singular_values,
                &kept_axes,
                tenferro_prims::ScalarReductionOp::Sum,
            )
        }
        NormKind::Spectral => {
            let singular_values = svdvals(ctx, tensor)?;
            let kept_axes: Vec<usize> = (1..singular_values.ndim()).collect();
            crate::prims_bridge::scalar_reduce_keep_axes(
                ctx,
                &singular_values,
                &kept_axes,
                tenferro_prims::ScalarReductionOp::Max,
            )
        }
        NormKind::L1 => {
            if m == 0 || n == 0 {
                return crate::prims_bridge::full_like_constant(
                    R::zero(),
                    &out_dims,
                    tensor.logical_memory_space(),
                );
            }
            let abs = crate::prims_bridge::complex_real_unary_same_shape(
                ctx,
                tensor,
                tenferro_prims::ComplexRealUnaryOp::Abs,
            )?;
            let kept_axes: Vec<usize> = std::iter::once(1).chain(2..abs.ndim()).collect();
            let column_sums = crate::prims_bridge::scalar_reduce_keep_axes(
                ctx,
                &abs,
                &kept_axes,
                tenferro_prims::ScalarReductionOp::Sum,
            )?;
            let batch_axes: Vec<usize> = (1..column_sums.ndim()).collect();
            crate::prims_bridge::scalar_reduce_keep_axes(
                ctx,
                &column_sums,
                &batch_axes,
                tenferro_prims::ScalarReductionOp::Max,
            )
        }
        NormKind::Inf => {
            if m == 0 || n == 0 {
                return crate::prims_bridge::full_like_constant(
                    R::zero(),
                    &out_dims,
                    tensor.logical_memory_space(),
                );
            }
            let abs = crate::prims_bridge::complex_real_unary_same_shape(
                ctx,
                tensor,
                tenferro_prims::ComplexRealUnaryOp::Abs,
            )?;
            let kept_axes: Vec<usize> = std::iter::once(0).chain(2..abs.ndim()).collect();
            let row_sums = crate::prims_bridge::scalar_reduce_keep_axes(
                ctx,
                &abs,
                &kept_axes,
                tenferro_prims::ScalarReductionOp::Sum,
            )?;
            let batch_axes: Vec<usize> = (1..row_sums.ndim()).collect();
            crate::prims_bridge::scalar_reduce_keep_axes(
                ctx,
                &row_sums,
                &batch_axes,
                tenferro_prims::ScalarReductionOp::Max,
            )
        }
        NormKind::Fro => {
            let abs = crate::prims_bridge::complex_real_unary_same_shape(
                ctx,
                tensor,
                tenferro_prims::ComplexRealUnaryOp::Abs,
            )?;
            let squared = crate::prims_bridge::scalar_binary_same_shape(
                ctx,
                &abs,
                &abs,
                tenferro_prims::ScalarBinaryOp::Mul,
            )?;
            let kept_axes: Vec<usize> = (2..squared.ndim()).collect();
            let squared_sum = crate::prims_bridge::scalar_reduce_keep_axes(
                ctx,
                &squared,
                &kept_axes,
                tenferro_prims::ScalarReductionOp::Sum,
            )?;
            crate::prims_bridge::analytic_unary_same_shape(
                ctx,
                &squared_sum,
                tenferro_prims::AnalyticUnaryOp::Sqrt,
            )
        }
        _ => Err(Error::InvalidArgument(format!(
            "norm kind {kind:?} not yet implemented"
        ))),
    }
}

macro_rules! impl_norm_primal_real {
    ($ty:ty) => {
        impl<C> NormPrimal<C> for $ty
        where
            C: backend::TensorLinalgContextFor<$ty>
                + tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<$ty>>,
            <C as tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<$ty>>>::ScalarBackend:
                tenferro_prims::TensorAnalyticPrims<tenferro_algebra::Standard<$ty>, Context = C>,
            C::Backend: 'static,
        {
            fn norm_primal(
                ctx: &mut C,
                tensor: &Tensor<Self>,
                kind: NormKind,
            ) -> Result<Tensor<Self::Real>> {
                norm_real_impl::<$ty, C>(ctx, tensor, kind)
            }
        }
    };
}

macro_rules! impl_norm_primal_complex {
    ($ty:ty, $real_ty:ty) => {
        impl<C> NormPrimal<C> for $ty
        where
            C: backend::TensorLinalgContextFor<$ty>
                + tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<$real_ty>>
                + tenferro_prims::TensorComplexRealContextFor<$ty>,
            <C as tenferro_prims::TensorScalarContextFor<
                tenferro_algebra::Standard<$real_ty>,
            >>::ScalarBackend:
                tenferro_prims::TensorAnalyticPrims<
                    tenferro_algebra::Standard<$real_ty>,
                    Context = C,
                >,
            C::ComplexRealBackend:
                tenferro_prims::TensorComplexRealPrims<$ty, Context = C, Real = $real_ty>,
            C::Backend: 'static,
        {
            fn norm_primal(
                ctx: &mut C,
                tensor: &Tensor<Self>,
                kind: NormKind,
            ) -> Result<Tensor<Self::Real>> {
                norm_complex_impl::<$ty, $real_ty, C>(ctx, tensor, kind)
            }
        }
    };
}

impl_norm_primal_real!(f32);
impl_norm_primal_real!(f64);
impl_norm_primal_complex!(Complex32, f32);
impl_norm_primal_complex!(Complex64, f64);
