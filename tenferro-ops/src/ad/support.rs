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
        | StdTensorOp::FullPivLuSolve { .. }
        | StdTensorOp::TriangularSolve { .. }
        | StdTensorOp::ValidateNonsingular => AdRuleSupport::DirectTranspose,
    }
}
