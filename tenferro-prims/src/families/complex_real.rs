use num_complex::ComplexFloat;
use tenferro_algebra::Scalar;
use tenferro_device::{Error, Result};
use tenferro_tensor::Tensor;

#[cfg(not(feature = "cuda"))]
use crate::{CudaBackend, CudaContext};
use crate::{RocmBackend, RocmContext};

/// Cross-dtype complex-to-real unary operations.
///
/// # Examples
///
/// ```rust
/// use tenferro_prims::ComplexRealUnaryOp;
///
/// let op = ComplexRealUnaryOp::Abs;
/// assert_eq!(op, ComplexRealUnaryOp::Abs);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ComplexRealUnaryOp {
    Abs,
}

/// Descriptor for complex-to-real unary planning.
///
/// # Examples
///
/// ```rust
/// use tenferro_prims::{ComplexRealPrimsDescriptor, ComplexRealUnaryOp};
///
/// let desc = ComplexRealPrimsDescriptor::PointwiseUnary {
///     op: ComplexRealUnaryOp::Abs,
/// };
/// assert!(matches!(desc, ComplexRealPrimsDescriptor::PointwiseUnary { .. }));
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ComplexRealPrimsDescriptor {
    /// Apply a complex-to-real unary operation to one input tensor.
    PointwiseUnary {
        /// The unary operation to apply.
        op: ComplexRealUnaryOp,
    },
}

/// Cross-dtype complex-to-real unary protocol family.
///
/// The input tensor is complex-valued and the output tensor is real-valued.
///
/// # Examples
///
/// ```ignore
/// use num_complex::Complex64;
/// use tenferro_prims::{
///     ComplexRealPrimsDescriptor, ComplexRealUnaryOp, CpuBackend, CpuContext,
///     TensorComplexRealPrims,
/// };
///
/// let mut ctx = CpuContext::new(1);
/// let desc = ComplexRealPrimsDescriptor::PointwiseUnary {
///     op: ComplexRealUnaryOp::Abs,
/// };
/// let _plan = <CpuBackend as TensorComplexRealPrims<Complex64>>::plan(
///     &mut ctx,
///     &desc,
///     &[&[2, 2], &[2, 2]],
/// )
/// .unwrap();
/// ```
pub trait TensorComplexRealPrims<Input: ComplexFloat + Scalar> {
    /// The real-valued output scalar type.
    type Real: Scalar + Send + Sync;
    /// Backend plan type.
    type Plan;
    /// Backend execution context.
    type Context;

    /// Plan a complex-to-real unary operation for the given input/output shapes.
    fn plan(
        ctx: &mut Self::Context,
        desc: &ComplexRealPrimsDescriptor,
        shapes: &[&[usize]],
    ) -> Result<Self::Plan>;

    /// Execute a previously planned complex-to-real unary operation.
    ///
    /// The execution contract matches the rest of tenferro prims:
    /// `output <- alpha * op(inputs) + beta * output`.
    fn execute(
        ctx: &mut Self::Context,
        plan: &Self::Plan,
        alpha: Input::Real,
        inputs: &[&Tensor<Input>],
        beta: Input::Real,
        output: &mut Tensor<Self::Real>,
    ) -> Result<()>;

    /// Report whether the backend advertises support for the given descriptor.
    fn has_complex_real_support(desc: ComplexRealPrimsDescriptor) -> bool;
}

#[cfg(not(feature = "cuda"))]
impl<Input> TensorComplexRealPrims<Input> for CudaBackend
where
    Input: ComplexFloat + Scalar,
    Input::Real: Scalar,
{
    type Real = Input::Real;
    type Plan = ();
    type Context = CudaContext;

    fn plan(
        _ctx: &mut Self::Context,
        desc: &ComplexRealPrimsDescriptor,
        _shapes: &[&[usize]],
    ) -> Result<Self::Plan> {
        Err(Error::InvalidArgument(format!(
            "complex-real family descriptor {desc:?} is not implemented on CudaBackend in phase 1"
        )))
    }

    fn execute(
        _ctx: &mut Self::Context,
        _plan: &Self::Plan,
        _alpha: Input::Real,
        _inputs: &[&Tensor<Input>],
        _beta: Input::Real,
        _output: &mut Tensor<Self::Real>,
    ) -> Result<()> {
        Err(Error::InvalidArgument(
            "complex-real family execution is not implemented on CudaBackend in phase 1".into(),
        ))
    }

    fn has_complex_real_support(_desc: ComplexRealPrimsDescriptor) -> bool {
        false
    }
}

impl<Input> TensorComplexRealPrims<Input> for RocmBackend
where
    Input: ComplexFloat + Scalar,
    Input::Real: Scalar,
{
    type Real = Input::Real;
    type Plan = ();
    type Context = RocmContext;

    fn plan(
        _ctx: &mut Self::Context,
        desc: &ComplexRealPrimsDescriptor,
        _shapes: &[&[usize]],
    ) -> Result<Self::Plan> {
        Err(Error::InvalidArgument(format!(
            "complex-real family descriptor {desc:?} is not implemented on RocmBackend in phase 1"
        )))
    }

    fn execute(
        _ctx: &mut Self::Context,
        _plan: &Self::Plan,
        _alpha: Input::Real,
        _inputs: &[&Tensor<Input>],
        _beta: Input::Real,
        _output: &mut Tensor<Self::Real>,
    ) -> Result<()> {
        Err(Error::InvalidArgument(
            "complex-real family execution is not implemented on RocmBackend in phase 1".into(),
        ))
    }

    fn has_complex_real_support(_desc: ComplexRealPrimsDescriptor) -> bool {
        false
    }
}
