use tenferro_algebra::{Algebra, Scalar, Standard};
use tenferro_device::{Error, Result};
use tenferro_tensor::Tensor;

#[cfg(not(feature = "cuda"))]
use crate::{CudaBackend, CudaContext};
use crate::{RocmBackend, RocmContext};

/// Pointwise scalar unary operations.
///
/// # Examples
///
/// ```
/// use tenferro_prims::ScalarUnaryOp;
///
/// let op = ScalarUnaryOp::Reciprocal;
/// assert_eq!(op, ScalarUnaryOp::Reciprocal);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScalarUnaryOp {
    Neg,
    Conj,
    Abs,
    Reciprocal,
    Real,
    Imag,
    Square,
}

/// Pointwise scalar binary operations.
///
/// # Examples
///
/// ```
/// use tenferro_prims::ScalarBinaryOp;
///
/// let op = ScalarBinaryOp::Mul;
/// assert_eq!(op, ScalarBinaryOp::Mul);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScalarBinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Maximum,
    Minimum,
    ClampMin,
    ClampMax,
}

/// Scalar reduction operations.
///
/// # Examples
///
/// ```
/// use tenferro_prims::ScalarReductionOp;
///
/// let op = ScalarReductionOp::Sum;
/// assert_eq!(op, ScalarReductionOp::Sum);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScalarReductionOp {
    Sum,
    Prod,
    Mean,
    Max,
    Min,
}

/// Descriptor for scalar-pointwise and scalar-reduction planning.
///
/// # Examples
///
/// ```
/// use tenferro_prims::{ScalarPrimsDescriptor, ScalarUnaryOp};
///
/// let desc = ScalarPrimsDescriptor::PointwiseUnary {
///     op: ScalarUnaryOp::Reciprocal,
/// };
/// assert!(matches!(desc, ScalarPrimsDescriptor::PointwiseUnary { .. }));
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ScalarPrimsDescriptor {
    /// Apply a unary pointwise operation to one input tensor.
    PointwiseUnary {
        /// The unary scalar operation to apply.
        op: ScalarUnaryOp,
    },
    /// Apply a binary pointwise operation to two input tensors.
    PointwiseBinary {
        /// The binary scalar operation to apply.
        op: ScalarBinaryOp,
    },
    /// Reduce one tensor into an output tensor over the dropped modes.
    Reduction {
        /// Input modes associated with the source tensor.
        modes_a: Vec<u32>,
        /// Output modes that remain after reduction.
        modes_c: Vec<u32>,
        /// Reduction operator to use.
        op: ScalarReductionOp,
    },
}

/// Scalar pointwise and reduction protocol family.
///
/// # Examples
///
/// ```ignore
/// use tenferro_algebra::Standard;
/// use tenferro_prims::{CpuBackend, CpuContext, ScalarPrimsDescriptor, ScalarUnaryOp, TensorScalarPrims};
///
/// let mut ctx = CpuContext::new(1);
/// let desc = ScalarPrimsDescriptor::PointwiseUnary {
///     op: ScalarUnaryOp::Reciprocal,
/// };
/// let _plan = <CpuBackend as TensorScalarPrims<Standard<f64>>>::plan(
///     &mut ctx,
///     &desc,
///     &[&[2, 2], &[2, 2]],
/// )
/// .unwrap();
/// ```
pub trait TensorScalarPrims<Alg: Algebra> {
    type Plan;
    type Context;

    /// Plan a scalar-family operation for the given input/output shapes.
    ///
    /// Backends may reject descriptors that are reserved in the public
    /// vocabulary but not yet wired to the current execution substrate.
    fn plan(
        ctx: &mut Self::Context,
        desc: &ScalarPrimsDescriptor,
        shapes: &[&[usize]],
    ) -> Result<Self::Plan>;

    /// Execute a previously planned scalar-family operation.
    ///
    /// The execution contract matches the rest of tenferro prims:
    /// `output <- alpha * op(inputs) + beta * output`.
    fn execute(
        ctx: &mut Self::Context,
        plan: &Self::Plan,
        alpha: Alg::Scalar,
        inputs: &[&Tensor<Alg::Scalar>],
        beta: Alg::Scalar,
        output: &mut Tensor<Alg::Scalar>,
    ) -> Result<()>;

    /// Report whether the backend advertises support for the given descriptor.
    ///
    /// This query is about the backend family surface, not whether a specific
    /// shape instance is valid.
    fn has_scalar_support(desc: ScalarPrimsDescriptor) -> bool;
}

#[cfg(not(feature = "cuda"))]
impl<S: Scalar> TensorScalarPrims<Standard<S>> for CudaBackend {
    type Plan = ();
    type Context = CudaContext;

    fn plan(
        _ctx: &mut Self::Context,
        desc: &ScalarPrimsDescriptor,
        _shapes: &[&[usize]],
    ) -> Result<Self::Plan> {
        Err(Error::InvalidArgument(format!(
            "scalar family descriptor {desc:?} is not implemented on CudaBackend in phase 1"
        )))
    }

    fn execute(
        _ctx: &mut Self::Context,
        _plan: &Self::Plan,
        _alpha: S,
        _inputs: &[&Tensor<S>],
        _beta: S,
        _output: &mut Tensor<S>,
    ) -> Result<()> {
        Err(Error::InvalidArgument(
            "scalar family execution is not implemented on CudaBackend in phase 1".into(),
        ))
    }

    fn has_scalar_support(_desc: ScalarPrimsDescriptor) -> bool {
        false
    }
}

impl<S: Scalar> TensorScalarPrims<Standard<S>> for RocmBackend {
    type Plan = ();
    type Context = RocmContext;

    fn plan(
        _ctx: &mut Self::Context,
        desc: &ScalarPrimsDescriptor,
        _shapes: &[&[usize]],
    ) -> Result<Self::Plan> {
        Err(Error::InvalidArgument(format!(
            "scalar family descriptor {desc:?} is not implemented on RocmBackend in phase 1"
        )))
    }

    fn execute(
        _ctx: &mut Self::Context,
        _plan: &Self::Plan,
        _alpha: S,
        _inputs: &[&Tensor<S>],
        _beta: S,
        _output: &mut Tensor<S>,
    ) -> Result<()> {
        Err(Error::InvalidArgument(
            "scalar family execution is not implemented on RocmBackend in phase 1".into(),
        ))
    }

    fn has_scalar_support(_desc: ScalarPrimsDescriptor) -> bool {
        false
    }
}
