use computegraph::types::{LocalValId, OpMode, ValRef};
use computegraph::OpEmitter;
use tenferro_tensor::DType;

use crate::ad::context::ShapeGuardContext;
use crate::std_tensor_op::StdTensorOp;

/// Describes how a core [`StdTensorOp`] participates in AD.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AdRuleSupport {
    /// The op has direct `linearize` and `transpose_rule` coverage.
    DirectTranspose,
    /// Reverse-mode support is obtained by linearizing the primal op first.
    ///
    /// A direct `transpose_rule` arm is not expected for these primal ops.
    SupportedViaLinearize,
    /// The op has no differentiable tangent contribution by construction.
    NoTangent,
    /// The extension object owns its own AD support contract.
    DelegatedToExtension,
}

/// Return the AD support strategy for a core tensor op.
pub(crate) fn ad_rule_support(op: &StdTensorOp) -> AdRuleSupport {
    match op {
        StdTensorOp::Constant { .. } | StdTensorOp::Compare(_) | StdTensorOp::ShapeOf { .. } => {
            AdRuleSupport::NoTangent
        }

        StdTensorOp::Cholesky
        | StdTensorOp::Lu
        | StdTensorOp::FullPivLu
        | StdTensorOp::Svd { .. }
        | StdTensorOp::Qr
        | StdTensorOp::Eigh { .. }
        | StdTensorOp::Eig { .. } => AdRuleSupport::SupportedViaLinearize,

        StdTensorOp::Extension(_) => AdRuleSupport::DelegatedToExtension,

        StdTensorOp::Add
        | StdTensorOp::Mul
        | StdTensorOp::Neg
        | StdTensorOp::Conj
        | StdTensorOp::DotGeneral { .. }
        | StdTensorOp::Transpose { .. }
        | StdTensorOp::Reshape { .. }
        | StdTensorOp::BroadcastInDim { .. }
        | StdTensorOp::Convert { .. }
        | StdTensorOp::ReduceSum { .. }
        | StdTensorOp::Div
        | StdTensorOp::Abs
        | StdTensorOp::Sign
        | StdTensorOp::Maximum
        | StdTensorOp::Minimum
        | StdTensorOp::Select
        | StdTensorOp::Clamp
        | StdTensorOp::Exp
        | StdTensorOp::Log
        | StdTensorOp::Sin
        | StdTensorOp::Cos
        | StdTensorOp::Tanh
        | StdTensorOp::Sqrt
        | StdTensorOp::Rsqrt
        | StdTensorOp::Pow
        | StdTensorOp::Expm1
        | StdTensorOp::Log1p
        | StdTensorOp::ExtractDiag { .. }
        | StdTensorOp::EmbedDiag { .. }
        | StdTensorOp::Tril { .. }
        | StdTensorOp::Triu { .. }
        | StdTensorOp::Gather(_)
        | StdTensorOp::GatherDynamicSliceSizes { .. }
        | StdTensorOp::Scatter(_)
        | StdTensorOp::Slice(_)
        | StdTensorOp::DynamicSlice { .. }
        | StdTensorOp::DynamicUpdateSlice
        | StdTensorOp::Pad(_)
        | StdTensorOp::NaryEinsum { .. }
        | StdTensorOp::Concatenate { .. }
        | StdTensorOp::Reverse { .. }
        | StdTensorOp::DynamicTruncate { .. }
        | StdTensorOp::PadToMatch { .. }
        | StdTensorOp::ReduceProd { .. }
        | StdTensorOp::ReduceMax { .. }
        | StdTensorOp::ReduceMin { .. }
        | StdTensorOp::Solve { .. }
        | StdTensorOp::FullPivLuSolve { .. }
        | StdTensorOp::TriangularSolve { .. }
        | StdTensorOp::ValidateNonsingular => AdRuleSupport::DirectTranspose,
    }
}

pub(crate) fn is_real_dtype(dtype: DType) -> bool {
    matches!(dtype, DType::F32 | DType::F64)
}

pub(crate) fn conjugate_primal_if_complex(
    emitter: &mut impl OpEmitter<StdTensorOp>,
    input: ValRef<StdTensorOp>,
    ctx: &mut ShapeGuardContext,
) -> ValRef<StdTensorOp> {
    let dtype = ctx.dtype_of(&input);
    conjugate_primal_if_dtype_complex(emitter, input, dtype)
}

pub(crate) fn conjugate_primal_if_dtype_complex(
    emitter: &mut impl OpEmitter<StdTensorOp>,
    input: ValRef<StdTensorOp>,
    dtype: DType,
) -> ValRef<StdTensorOp> {
    if is_real_dtype(dtype) {
        input
    } else {
        ValRef::Local(emitter.add_op(StdTensorOp::Conj, vec![input], OpMode::Primal)[0])
    }
}

pub(crate) fn conjugate_linear_if_dtype_complex(
    emitter: &mut impl OpEmitter<StdTensorOp>,
    input: LocalValId,
    dtype: DType,
) -> LocalValId {
    if is_real_dtype(dtype) {
        input
    } else {
        emitter.add_op(
            StdTensorOp::Conj,
            vec![ValRef::Local(input)],
            OpMode::Linear {
                active_mask: vec![true],
            },
        )[0]
    }
}
