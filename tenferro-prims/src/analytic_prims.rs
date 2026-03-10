use tenferro_algebra::Algebra;
use tenferro_device::{Error, Result};
use tenferro_tensor::Tensor;

use crate::{PrimDescriptor, TensorPrims, UnaryOp};

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
    PointwiseUnary {
        op: AnalyticUnaryOp,
    },
    PointwiseBinary {
        op: AnalyticBinaryOp,
    },
    Reduction {
        modes_a: Vec<u32>,
        modes_c: Vec<u32>,
        op: AnalyticReductionOp,
    },
}

impl AnalyticPrimsDescriptor {
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

    fn plan(
        ctx: &mut Self::Context,
        desc: &AnalyticPrimsDescriptor,
        shapes: &[&[usize]],
    ) -> Result<Self::Plan>;

    fn execute(
        ctx: &mut Self::Context,
        plan: &Self::Plan,
        alpha: Alg::Scalar,
        inputs: &[&Tensor<Alg::Scalar>],
        beta: Alg::Scalar,
        output: &mut Tensor<Alg::Scalar>,
    ) -> Result<()>;

    fn has_analytic_support(desc: AnalyticPrimsDescriptor) -> bool;
}

impl<Alg, B> TensorAnalyticPrims<Alg> for B
where
    Alg: Algebra,
    B: TensorPrims<Alg>,
{
    type Plan = <B as TensorPrims<Alg>>::Plan;
    type Context = <B as TensorPrims<Alg>>::Context;

    fn plan(
        ctx: &mut Self::Context,
        desc: &AnalyticPrimsDescriptor,
        shapes: &[&[usize]],
    ) -> Result<Self::Plan> {
        let legacy = desc.to_legacy()?;
        <B as TensorPrims<Alg>>::plan(ctx, &legacy, shapes)
    }

    fn execute(
        ctx: &mut Self::Context,
        plan: &Self::Plan,
        alpha: Alg::Scalar,
        inputs: &[&Tensor<Alg::Scalar>],
        beta: Alg::Scalar,
        output: &mut Tensor<Alg::Scalar>,
    ) -> Result<()> {
        <B as TensorPrims<Alg>>::execute(ctx, plan, alpha, inputs, beta, output)
    }

    fn has_analytic_support(desc: AnalyticPrimsDescriptor) -> bool {
        matches!(
            desc,
            AnalyticPrimsDescriptor::PointwiseUnary {
                op: AnalyticUnaryOp::Sqrt,
            }
        )
    }
}
