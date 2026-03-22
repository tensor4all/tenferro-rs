use super::*;

mod norm_impl;

pub(crate) use norm_impl::norm_real_impl;

#[doc(hidden)]
pub trait NormPrimal<C>: KernelLinalgScalar {
    fn norm_primal(
        ctx: &mut C,
        tensor: &Tensor<Self>,
        kind: NormKind,
    ) -> Result<Tensor<Self::Real>>;
}

/// Solve a triangular linear system `A x = b`.
///
/// # Examples
///
/// ```
/// use tenferro_device::LogicalMemorySpace;
/// use tenferro_linalg::solve_triangular;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::from_slice(
///     &[1.0, 0.0, 0.0, 2.0],
///     &[2, 2],
///     MemoryOrder::ColumnMajor,
/// ).unwrap();
/// let b = Tensor::<f64>::from_slice(&[1.0, 4.0], &[2], MemoryOrder::ColumnMajor).unwrap();
/// let x = solve_triangular(&mut ctx, &a, &b, true).unwrap();
/// assert_eq!(x.logical_memory_space(), LogicalMemorySpace::MainMemory);
/// ```
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
///
/// Complex inputs return a tensor over the associated real scalar type.
///
/// # Examples
///
/// ```
/// use num_complex::Complex64;
/// use tenferro_device::LogicalMemorySpace;
/// use tenferro_linalg::{norm, NormKind};
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::from_slice(
///     &[Complex64::new(3.0, 4.0), Complex64::new(0.0, 0.0)],
///     &[2],
///     MemoryOrder::ColumnMajor,
/// )
/// .unwrap();
/// let n: Tensor<f64> = norm(&mut ctx, &a, NormKind::Fro).unwrap();
/// assert_eq!(n.logical_memory_space(), LogicalMemorySpace::MainMemory);
/// ```
#[allow(private_bounds)]
pub fn norm<T, C>(ctx: &mut C, tensor: &Tensor<T>, kind: NormKind) -> Result<Tensor<T::Real>>
where
    T: KernelLinalgScalar + NormPrimal<C>,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    T::norm_primal(ctx, tensor, kind)
}

/// Compute the matrix condition number with a selected norm convention.
///
/// Complex inputs return a real-valued tensor.
///
/// # Examples
///
/// ```
/// use num_complex::Complex64;
/// use tenferro_device::LogicalMemorySpace;
/// use tenferro_linalg::{cond, NormKind};
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::from_slice(
///     &[
///         Complex64::new(3.0, 4.0),
///         Complex64::new(0.0, 0.0),
///         Complex64::new(0.0, 0.0),
///         Complex64::new(2.0, 0.0),
///     ],
///     &[2, 2],
///     MemoryOrder::ColumnMajor,
/// )
/// .unwrap();
/// let c: Tensor<f64> = cond(&mut ctx, &a, NormKind::Fro).unwrap();
/// assert_eq!(c.logical_memory_space(), LogicalMemorySpace::MainMemory);
/// ```
#[allow(private_bounds)]
pub fn cond<T, C>(ctx: &mut C, tensor: &Tensor<T>, kind: NormKind) -> Result<Tensor<T::Real>>
where
    T: KernelLinalgScalar + NormPrimal<C>,
    C: backend::TensorLinalgContextFor<T>
        + tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<T::Real>>,
    C::Backend: 'static,
{
    match kind {
        NormKind::Fro | NormKind::L1 | NormKind::Inf | NormKind::Spectral | NormKind::Nuclear => {}
        _ => {
            return Err(Error::InvalidArgument(format!(
                "cond only supports Fro, L1, Inf, Spectral, and Nuclear norms, got {kind:?}"
            )));
        }
    }

    validate_square(tensor)?;
    if matches!(kind, NormKind::Nuclear) {
        let singular_values = svdvals(ctx, tensor)?;
        let kept_axes: Vec<usize> = (1..singular_values.ndim()).collect();
        let nuclear_norm = crate::prims_bridge::scalar_reduce_keep_axes(
            ctx,
            &singular_values,
            &kept_axes,
            tenferro_prims::ScalarReductionOp::Sum,
        )?;
        let reciprocal = crate::prims_bridge::scalar_unary_same_shape(
            ctx,
            &singular_values,
            tenferro_prims::ScalarUnaryOp::Reciprocal,
        )?;
        let reciprocal_norm = crate::prims_bridge::scalar_reduce_keep_axes(
            ctx,
            &reciprocal,
            &kept_axes,
            tenferro_prims::ScalarReductionOp::Sum,
        )?;
        return crate::prims_bridge::scalar_binary_same_shape(
            ctx,
            &nuclear_norm,
            &reciprocal_norm,
            tenferro_prims::ScalarBinaryOp::Mul,
        );
    }

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
