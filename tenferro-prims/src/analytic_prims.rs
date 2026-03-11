use tenferro_algebra::{Algebra, Scalar, Standard};
use tenferro_device::{Error, Result};
use tenferro_tensor::Tensor;

#[cfg(test)]
use crate::{CudaBackend, CudaContext, PrimDescriptor, RocmBackend, RocmContext, UnaryOp};
#[cfg(not(test))]
use crate::{CudaBackend, CudaContext, RocmBackend, RocmContext};

/// Analytic unary operations.
///
/// # Examples
///
/// ```
/// use tenferro_prims::AnalyticUnaryOp;
///
/// let op = AnalyticUnaryOp::Sqrt;
/// assert_eq!(op, AnalyticUnaryOp::Sqrt);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AnalyticUnaryOp {
    Sqrt,
    Rsqrt,
    Exp,
    Expm1,
    Log,
    Log1p,
    Sin,
    Cos,
    Tan,
    Tanh,
}

/// Analytic binary operations.
///
/// # Examples
///
/// ```
/// use tenferro_prims::AnalyticBinaryOp;
///
/// let op = AnalyticBinaryOp::Pow;
/// assert_eq!(op, AnalyticBinaryOp::Pow);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AnalyticBinaryOp {
    Pow,
    Atan2,
    Hypot,
    Xlogy,
}

/// Analytic reduction operations.
///
/// # Examples
///
/// ```
/// use tenferro_prims::AnalyticReductionOp;
///
/// let op = AnalyticReductionOp::Var;
/// assert_eq!(op, AnalyticReductionOp::Var);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AnalyticReductionOp {
    Var,
    Std,
}

/// Descriptor for analytic-pointwise and analytic-reduction planning.
///
/// # Examples
///
/// ```
/// use tenferro_prims::{AnalyticPrimsDescriptor, AnalyticUnaryOp};
///
/// let desc = AnalyticPrimsDescriptor::PointwiseUnary {
///     op: AnalyticUnaryOp::Sqrt,
/// };
/// assert!(matches!(desc, AnalyticPrimsDescriptor::PointwiseUnary { .. }));
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AnalyticPrimsDescriptor {
    /// Apply an analytic unary pointwise operation to one input tensor.
    PointwiseUnary {
        /// The unary analytic operation to apply.
        op: AnalyticUnaryOp,
    },
    /// Apply an analytic binary pointwise operation to two input tensors.
    PointwiseBinary {
        /// The binary analytic operation to apply.
        op: AnalyticBinaryOp,
    },
    /// Reduce one tensor into an output tensor over the dropped modes.
    Reduction {
        /// Input modes associated with the source tensor.
        modes_a: Vec<u32>,
        /// Output modes that remain after reduction.
        modes_c: Vec<u32>,
        /// Reduction operator to use.
        op: AnalyticReductionOp,
    },
}

impl AnalyticPrimsDescriptor {
    #[cfg(test)]
    pub(crate) fn to_legacy(&self) -> Result<PrimDescriptor> {
        match self {
            Self::PointwiseUnary {
                op: AnalyticUnaryOp::Sqrt,
            } => Ok(PrimDescriptor::ElementwiseUnary { op: UnaryOp::Sqrt }),
            Self::PointwiseUnary { op } => Err(Error::InvalidArgument(format!(
                "analytic unary operation {op:?} is not wired to the legacy prim surface yet"
            ))),
            Self::PointwiseBinary { op } => Err(Error::InvalidArgument(format!(
                "analytic binary operation {op:?} is not wired to the legacy prim surface yet"
            ))),
            Self::Reduction { op, .. } => Err(Error::InvalidArgument(format!(
                "analytic reduction {op:?} is not wired to the legacy prim surface yet"
            ))),
        }
    }
}

/// Analytic pointwise and reduction protocol family.
///
/// # Examples
///
/// ```ignore
/// use tenferro_algebra::Standard;
/// use tenferro_prims::{AnalyticPrimsDescriptor, AnalyticUnaryOp, CpuBackend, CpuContext, TensorAnalyticPrims};
///
/// let mut ctx = CpuContext::new(1);
/// let desc = AnalyticPrimsDescriptor::PointwiseUnary {
///     op: AnalyticUnaryOp::Sqrt,
/// };
/// let _plan = <CpuBackend as TensorAnalyticPrims<Standard<f64>>>::plan(
///     &mut ctx,
///     &desc,
///     &[&[2, 2], &[2, 2]],
/// )
/// .unwrap();
/// ```
pub trait TensorAnalyticPrims<Alg: Algebra> {
    type Plan;
    type Context;

    /// Plan an analytic-family operation for the given input/output shapes.
    ///
    /// Public vocabulary may be broader than the currently wired execution
    /// surface while the workspace migrates away from legacy `TensorPrims<A>`.
    fn plan(
        ctx: &mut Self::Context,
        desc: &AnalyticPrimsDescriptor,
        shapes: &[&[usize]],
    ) -> Result<Self::Plan>;

    /// Execute a previously planned analytic-family operation.
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
    /// This is a family-level capability check and does not validate every
    /// shape-specific precondition.
    fn has_analytic_support(desc: AnalyticPrimsDescriptor) -> bool;
}

impl<S: Scalar> TensorAnalyticPrims<Standard<S>> for CudaBackend {
    type Plan = ();
    type Context = CudaContext;

    fn plan(
        _ctx: &mut Self::Context,
        desc: &AnalyticPrimsDescriptor,
        _shapes: &[&[usize]],
    ) -> Result<Self::Plan> {
        Err(Error::InvalidArgument(format!(
            "analytic family descriptor {desc:?} is not implemented on CudaBackend in phase 1"
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
            "analytic family execution is not implemented on CudaBackend in phase 1".into(),
        ))
    }

    fn has_analytic_support(_desc: AnalyticPrimsDescriptor) -> bool {
        false
    }
}

impl<S: Scalar> TensorAnalyticPrims<Standard<S>> for RocmBackend {
    type Plan = ();
    type Context = RocmContext;

    fn plan(
        _ctx: &mut Self::Context,
        desc: &AnalyticPrimsDescriptor,
        _shapes: &[&[usize]],
    ) -> Result<Self::Plan> {
        Err(Error::InvalidArgument(format!(
            "analytic family descriptor {desc:?} is not implemented on RocmBackend in phase 1"
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
            "analytic family execution is not implemented on RocmBackend in phase 1".into(),
        ))
    }

    fn has_analytic_support(_desc: AnalyticPrimsDescriptor) -> bool {
        false
    }
}
