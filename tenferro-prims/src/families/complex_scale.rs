use num_complex::ComplexFloat;
use tenferro_algebra::Scalar;
use tenferro_device::Result;
use tenferro_tensor::Tensor;

#[cfg(not(feature = "cuda"))]
use crate::{CudaBackend, CudaContext};
use crate::{RocmBackend, RocmContext};

/// Cross-dtype complex-by-real pointwise operations.
///
/// # Examples
///
/// ```rust
/// use tenferro_prims::ComplexScalePrimsDescriptor;
///
/// let desc = ComplexScalePrimsDescriptor::PointwiseMul;
/// assert!(matches!(desc, ComplexScalePrimsDescriptor::PointwiseMul));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ComplexScalePrimsDescriptor {
    /// Multiply a complex tensor by a real tensor elementwise.
    PointwiseMul,
}

/// Cross-dtype complex-by-real pointwise family.
///
/// The left-hand side and output are complex-valued, while the right-hand side
/// is real-valued. The execution contract matches the rest of tenferro prims:
/// `output <- alpha * (lhs * rhs) + beta * output`.
///
/// # Examples
///
/// ```ignore
/// use num_complex::Complex64;
/// use tenferro_prims::{
///     ComplexScalePrimsDescriptor, CpuBackend, CpuContext, TensorComplexScalePrims,
/// };
///
/// let mut ctx = CpuContext::new(1);
/// let desc = ComplexScalePrimsDescriptor::PointwiseMul;
/// let _plan = <CpuBackend as TensorComplexScalePrims<Complex64>>::plan(
///     &mut ctx,
///     &desc,
///     &[&[2, 2], &[2, 2], &[2, 2]],
/// )
/// .unwrap();
/// ```
pub trait TensorComplexScalePrims<Input: ComplexFloat + Scalar>
where
    Input::Real: Scalar + Send + Sync,
{
    /// Backend plan type.
    type Plan;
    /// Backend execution context.
    type Context;

    /// Plan a complex-by-real pointwise operation for the given shapes.
    fn plan(
        ctx: &mut Self::Context,
        desc: &ComplexScalePrimsDescriptor,
        shapes: &[&[usize]],
    ) -> Result<Self::Plan>;

    /// Execute a previously planned complex-by-real pointwise operation.
    fn execute(
        ctx: &mut Self::Context,
        plan: &Self::Plan,
        alpha: Input,
        lhs: &Tensor<Input>,
        rhs: &Tensor<Input::Real>,
        beta: Input,
        output: &mut Tensor<Input>,
    ) -> Result<()>;

    /// Report whether the backend advertises support for the given descriptor.
    fn has_complex_scale_support(desc: ComplexScalePrimsDescriptor) -> bool;
}

#[cfg(not(feature = "cuda"))]
impl<Input> TensorComplexScalePrims<Input> for CudaBackend
where
    Input: ComplexFloat + Scalar,
    Input::Real: Scalar + Send + Sync,
{
    type Plan = ();
    type Context = CudaContext;

    fn plan(
        _ctx: &mut Self::Context,
        desc: &ComplexScalePrimsDescriptor,
        _shapes: &[&[usize]],
    ) -> Result<Self::Plan> {
        Err(tenferro_device::Error::InvalidArgument(format!(
            "complex-scale family descriptor {desc:?} is not implemented on CudaBackend in phase 1"
        )))
    }

    fn execute(
        _ctx: &mut Self::Context,
        _plan: &Self::Plan,
        _alpha: Input,
        _lhs: &Tensor<Input>,
        _rhs: &Tensor<Input::Real>,
        _beta: Input,
        _output: &mut Tensor<Input>,
    ) -> Result<()> {
        Err(tenferro_device::Error::InvalidArgument(
            "complex-scale family execution is not implemented on CudaBackend in phase 1".into(),
        ))
    }

    fn has_complex_scale_support(_desc: ComplexScalePrimsDescriptor) -> bool {
        false
    }
}

impl<Input> TensorComplexScalePrims<Input> for RocmBackend
where
    Input: ComplexFloat + Scalar,
    Input::Real: Scalar + Send + Sync,
{
    type Plan = ();
    type Context = RocmContext;

    fn plan(
        _ctx: &mut Self::Context,
        desc: &ComplexScalePrimsDescriptor,
        _shapes: &[&[usize]],
    ) -> Result<Self::Plan> {
        Err(tenferro_device::Error::InvalidArgument(format!(
            "complex-scale family descriptor {desc:?} is not implemented on RocmBackend in phase 1"
        )))
    }

    fn execute(
        _ctx: &mut Self::Context,
        _plan: &Self::Plan,
        _alpha: Input,
        _lhs: &Tensor<Input>,
        _rhs: &Tensor<Input::Real>,
        _beta: Input,
        _output: &mut Tensor<Input>,
    ) -> Result<()> {
        Err(tenferro_device::Error::InvalidArgument(
            "complex-scale family execution is not implemented on RocmBackend in phase 1".into(),
        ))
    }

    fn has_complex_scale_support(_desc: ComplexScalePrimsDescriptor) -> bool {
        false
    }
}
