use tenferro_algebra::Semiring;
use tenferro_device::{Error, Result};
use tenferro_tensor::Tensor;

use crate::{Extension, PrimDescriptor, TensorPrims};

/// Semiring-valid optional binary fast-path operations.
///
/// # Examples
///
/// ```
/// use tenferro_prims::SemiringBinaryOp;
///
/// let op = SemiringBinaryOp::Mul;
/// assert_eq!(op, SemiringBinaryOp::Mul);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SemiringBinaryOp {
    /// Elementwise semiring addition.
    Add,
    /// Elementwise semiring multiplication.
    Mul,
}

/// Descriptor for optional semiring fast paths.
///
/// # Examples
///
/// ```
/// use tenferro_prims::{SemiringBinaryOp, SemiringFastPathDescriptor};
///
/// let desc = SemiringFastPathDescriptor::ElementwiseBinary {
///     op: SemiringBinaryOp::Mul,
/// };
/// assert!(matches!(desc, SemiringFastPathDescriptor::ElementwiseBinary { .. }));
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SemiringFastPathDescriptor {
    /// Optional contraction fast path.
    Contract {
        /// Mode labels for input A.
        modes_a: Vec<u32>,
        /// Mode labels for input B.
        modes_b: Vec<u32>,
        /// Mode labels for output C.
        modes_c: Vec<u32>,
    },
    /// Optional elementwise semiring binary fast path.
    ElementwiseBinary {
        /// The semiring binary operation.
        op: SemiringBinaryOp,
    },
}

impl SemiringFastPathDescriptor {
    pub(crate) fn to_legacy(&self) -> Result<PrimDescriptor> {
        match self {
            Self::Contract {
                modes_a,
                modes_b,
                modes_c,
            } => Ok(PrimDescriptor::Contract {
                modes_a: modes_a.clone(),
                modes_b: modes_b.clone(),
                modes_c: modes_c.clone(),
            }),
            Self::ElementwiseBinary {
                op: SemiringBinaryOp::Mul,
            } => Ok(PrimDescriptor::ElementwiseMul),
            Self::ElementwiseBinary {
                op: SemiringBinaryOp::Add,
            } => Err(Error::InvalidArgument(
                "ElementwiseBinary(Add) is not wired to the legacy prim surface yet".into(),
            )),
        }
    }
}

/// Optional semiring performance paths.
///
/// # Examples
///
/// ```ignore
/// use tenferro_algebra::Standard;
/// use tenferro_prims::{
///     CpuBackend, SemiringBinaryOp, SemiringFastPathDescriptor, TensorSemiringFastPath,
/// };
///
/// let supported =
///     <CpuBackend as TensorSemiringFastPath<Standard<f64>>>::has_fast_path(SemiringFastPathDescriptor::ElementwiseBinary {
///         op: SemiringBinaryOp::Mul,
///     });
/// assert!(supported);
/// ```
pub trait TensorSemiringFastPath<Alg: Semiring> {
    /// Backend-specific plan type.
    type Plan;

    /// Backend-specific execution context.
    type Context;

    /// Plan an optional semiring fast path.
    fn plan(
        ctx: &mut Self::Context,
        desc: &SemiringFastPathDescriptor,
        shapes: &[&[usize]],
    ) -> Result<Self::Plan>;

    /// Execute an optional semiring fast path.
    fn execute(
        ctx: &mut Self::Context,
        plan: &Self::Plan,
        alpha: Alg::Scalar,
        inputs: &[&Tensor<Alg::Scalar>],
        beta: Alg::Scalar,
        output: &mut Tensor<Alg::Scalar>,
    ) -> Result<()>;

    /// Query whether the optional path is available.
    fn has_fast_path(desc: SemiringFastPathDescriptor) -> bool;
}

impl<Alg, B> TensorSemiringFastPath<Alg> for B
where
    Alg: Semiring,
    B: TensorPrims<Alg>,
{
    type Plan = <B as TensorPrims<Alg>>::Plan;
    type Context = <B as TensorPrims<Alg>>::Context;

    fn plan(
        ctx: &mut Self::Context,
        desc: &SemiringFastPathDescriptor,
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

    fn has_fast_path(desc: SemiringFastPathDescriptor) -> bool {
        match desc {
            SemiringFastPathDescriptor::Contract { .. } => {
                <B as TensorPrims<Alg>>::has_extension_for(Extension::Contract)
            }
            SemiringFastPathDescriptor::ElementwiseBinary {
                op: SemiringBinaryOp::Mul,
            } => <B as TensorPrims<Alg>>::has_extension_for(Extension::ElementwiseMul),
            SemiringFastPathDescriptor::ElementwiseBinary {
                op: SemiringBinaryOp::Add,
            } => false,
        }
    }
}
