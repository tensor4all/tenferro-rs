use super::ResidualSpec;
use crate::std_tensor_op::StdTensorOp;
use tenferro_core_ops::PrimitiveOpKind;

/// Return the tensor retention declaration for a core primitive.
/// Extension declarations remain owned by their context's rule set.
///
/// # Examples
///
/// ```
/// use tenferro_ops::{ad::primitive_residual_spec, std_tensor_op::StdTensorOp};
/// assert!(primitive_residual_spec(&StdTensorOp::Add).unwrap().is_empty());
/// ```
pub fn primitive_residual_spec(op: &StdTensorOp) -> Option<ResidualSpec> {
    op.primitive_kind().map(residual_spec_for_kind)
}

// These retention declarations are shared metadata, not differentiation rules.
pub(super) fn residual_spec_for_kind(kind: PrimitiveOpKind) -> ResidualSpec {
    match kind {
        PrimitiveOpKind::Add => ResidualSpec::none(),
        PrimitiveOpKind::Sub => ResidualSpec::none(),
        PrimitiveOpKind::Mul => ResidualSpec::all_inputs(),
        PrimitiveOpKind::Neg => ResidualSpec::none(),
        PrimitiveOpKind::Conj => ResidualSpec::none(),
        PrimitiveOpKind::Div => ResidualSpec::all_inputs(),
        PrimitiveOpKind::Rem => ResidualSpec::none(),
        PrimitiveOpKind::Abs => ResidualSpec::input(0),
        PrimitiveOpKind::Sign => ResidualSpec::none(),
        PrimitiveOpKind::Maximum => ResidualSpec::all_inputs(),
        PrimitiveOpKind::Minimum => ResidualSpec::all_inputs(),
        PrimitiveOpKind::Compare => ResidualSpec::none(),
        PrimitiveOpKind::Select => ResidualSpec::input(0),
        PrimitiveOpKind::Clamp => ResidualSpec::all_inputs(),
        PrimitiveOpKind::Exp => ResidualSpec::input(0),
        PrimitiveOpKind::Log => ResidualSpec::input(0),
        PrimitiveOpKind::Sin => ResidualSpec::input(0),
        PrimitiveOpKind::Cos => ResidualSpec::input(0),
        PrimitiveOpKind::Tanh => ResidualSpec::input(0),
        PrimitiveOpKind::Sqrt => ResidualSpec::input(0),
        PrimitiveOpKind::Rsqrt => ResidualSpec::input(0),
        PrimitiveOpKind::Pow => ResidualSpec::all_inputs(),
        PrimitiveOpKind::Expm1 => ResidualSpec::input(0),
        PrimitiveOpKind::Log1p => ResidualSpec::input(0),
        PrimitiveOpKind::Erf => ResidualSpec::input(0),
        PrimitiveOpKind::DotGeneral => ResidualSpec::all_inputs(),
        PrimitiveOpKind::ReduceSum => ResidualSpec::input(0),
        PrimitiveOpKind::ReduceSumSquares => ResidualSpec::input(0),
        PrimitiveOpKind::ReduceProd => ResidualSpec::input(0),
        PrimitiveOpKind::ReduceMax => ResidualSpec::input(0),
        PrimitiveOpKind::ReduceMin => ResidualSpec::input(0),
        PrimitiveOpKind::Transpose => ResidualSpec::none(),
        PrimitiveOpKind::Reshape => ResidualSpec::input(0),
        PrimitiveOpKind::BroadcastInDim => ResidualSpec::input(0),
        PrimitiveOpKind::Convert => ResidualSpec::none(),
        PrimitiveOpKind::ExtractDiag => ResidualSpec::input(0),
        PrimitiveOpKind::EmbedDiag => ResidualSpec::none(),
        PrimitiveOpKind::Tril => ResidualSpec::none(),
        PrimitiveOpKind::Triu => ResidualSpec::none(),
        PrimitiveOpKind::Gather => ResidualSpec::all_inputs(),
        PrimitiveOpKind::GatherDynamicSliceSizes => ResidualSpec::all_inputs(),
        PrimitiveOpKind::Scatter => ResidualSpec::all_inputs(),
        PrimitiveOpKind::Slice => ResidualSpec::input(0),
        PrimitiveOpKind::DynamicSlice => ResidualSpec::all_inputs(),
        PrimitiveOpKind::DynamicUpdateSlice => ResidualSpec::all_inputs(),
        PrimitiveOpKind::Pad => ResidualSpec::input(0),
        PrimitiveOpKind::Concatenate => ResidualSpec::all_inputs(),
        PrimitiveOpKind::Reverse => ResidualSpec::none(),
        PrimitiveOpKind::ShapeOf => ResidualSpec::none(),
        PrimitiveOpKind::DynamicTruncate => ResidualSpec::all_inputs(),
        PrimitiveOpKind::PadToMatch => ResidualSpec::all_inputs(),
        PrimitiveOpKind::Constant => ResidualSpec::none(),
    }
}
