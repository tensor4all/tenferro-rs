use super::*;

/// Solve a triangular linear system `A x = b`.
pub fn solve_triangular<T: KernelLinalgScalar, C>(
    ctx: &mut C,
    a: &Tensor<T>,
    b: &Tensor<T>,
    upper: bool,
) -> Result<Tensor<T>>
where
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    <C::Backend as backend::TensorLinalgBackend<T>>::solve_triangular(ctx, a, b, upper)
}

/// Compute a norm.
pub fn norm<T: KernelLinalgScalar<Real = T> + num_traits::Float, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    kind: NormKind,
) -> Result<Tensor<T>>
where
    T: KernelLinalgScalar,
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

        let value = match kind {
            NormKind::Fro => {
                let mut sum = T::zero();
                let input = ensure_col_major(tensor);
                let offset = input.offset() as usize;
                let len = tensor.dims()[0];
                let vec_data = &extract_slice(&input)?[offset..offset + len];
                for &v in vec_data {
                    sum = sum + v * v;
                }
                sum.sqrt()
            }
            NormKind::Lp(p) => {
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
            NormKind::L1 | NormKind::Inf | NormKind::Nuclear | NormKind::Spectral => unreachable!(),
        };

        return tensor_from_data(vec![value], &[]);
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

/// Compute the matrix condition number with a selected norm convention.
pub fn cond<T: KernelLinalgScalar<Real = T> + num_traits::Float, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    kind: NormKind,
) -> Result<Tensor<T>>
where
    T: KernelLinalgScalar,
    C: backend::TensorLinalgContextFor<T>
        + tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<T>>,
    <C as tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<T>>>::ScalarBackend:
        tenferro_prims::TensorAnalyticPrims<tenferro_algebra::Standard<T>, Context = C>,
    C::Backend: 'static,
{
    match kind {
        NormKind::Fro | NormKind::L1 | NormKind::Inf | NormKind::Spectral => {}
        _ => {
            return Err(Error::InvalidArgument(format!(
                "cond only supports Fro, L1, Inf, and Spectral norms, got {kind:?}"
            )));
        }
    }

    validate_square(tensor)?;
    let lhs = norm(ctx, tensor, kind)?;
    let inverse = inv(ctx, tensor)?;
    let rhs = norm(ctx, &inverse, kind)?;
    crate::prims_bridge::scalar_binary_same_shape(
        ctx,
        &lhs,
        &rhs,
        tenferro_prims::ScalarBinaryOp::Mul,
    )
}
