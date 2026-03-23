use super::*;

/// Forward-mode AD rule for matrix exponential (JVP / pushforward).
///
/// Computes `exp(A)` and the Frechet derivative `d(exp(A))` in the direction `dA`.
/// Uses the auxiliary 2n x 2n matrix trick (PyTorch approach):
///
/// ```text
/// M = [[A, dA], [0, A]]
/// exp(A)    = top-left  n×n block of exp(M)
/// d(exp(A)) = top-right n×n block of exp(M)
/// ```
///
/// # Examples
///
/// ```
/// use tenferro_linalg::matrix_exp_frule;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::zeros(&[3, 3], mem, col).unwrap();
/// let da = Tensor::<f64>::ones(&[3, 3], mem, col).unwrap();
/// let (exp_a, dexp_a) = matrix_exp_frule(&mut ctx, &a, &da).unwrap();
/// ```
pub fn matrix_exp_frule<T: KernelLinalgScalar<Real = T> + num_traits::Float, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    tangent: &Tensor<T>,
) -> AdResult<(Tensor<T>, Tensor<T>)>
where
    T: KernelLinalgScalar
        + crate::prims_bridge::ScaleTensorByRealSameShape<C>
        + crate::ad_helpers::MatrixExpAbsTensor<C>,
    C: backend::TensorLinalgContextFor<T>
        + tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<T>>
        + tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<T::Real>>
        + tenferro_prims::TensorSemiringContextFor<tenferro_algebra::Standard<T>>,
    <C as tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<T::Real>>>::ScalarBackend:
        tenferro_prims::TensorAnalyticPrims<tenferro_algebra::Standard<T::Real>, Context = C>,
    C::Backend: 'static,
{
    require_linalg_support::<T, C>(backend::LinalgCapabilityOp::MatrixExp, "matrix_exp_frule")
        .map_err(to_ad_err)?;

    let (n, batch_dims) = validate_square(tensor).map_err(to_ad_err)?;
    let zero = Tensor::<T>::zeros(
        &output_dims(&[n, n], batch_dims),
        tensor.logical_memory_space(),
        MemoryOrder::ColumnMajor,
    )
    .map_err(to_ad_err)?;
    let top = Tensor::cat(&[tensor, tangent], 1).map_err(to_ad_err)?;
    let bottom = Tensor::cat(&[&zero, tensor], 1).map_err(to_ad_err)?;
    let m = Tensor::cat(&[&top, &bottom], 0).map_err(to_ad_err)?;
    let exp_m = matrix_exp(ctx, &m).map_err(to_ad_err)?;

    let result = exp_m
        .narrow(0, 0, n)
        .and_then(|t| t.narrow(1, 0, n))
        .map_err(to_ad_err)?;
    let tang = exp_m
        .narrow(0, 0, n)
        .and_then(|t| t.narrow(1, n, n))
        .map_err(to_ad_err)?;
    Ok((result, tang))
}
