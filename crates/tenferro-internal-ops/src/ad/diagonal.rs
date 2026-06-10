use crate::ad::context::ShapeGuardContext;
use crate::ad::PrimitiveRuleBuilder;
use computegraph::types::{LocalValueId, OperationRole, ValueRef};

use crate::std_tensor_op::StdTensorOp;

pub fn linearize_extract_diag(
    builder: &mut dyn PrimitiveRuleBuilder,
    tangent_in: &[Option<LocalValueId>],
    axis_a: usize,
    axis_b: usize,
) -> Vec<Option<LocalValueId>> {
    // TODO: ExtractDiag/EmbedDiag could be replaced by Gather/Scatter
    match tangent_in[0] {
        Some(dx) => {
            let out = builder.add_operation(
                StdTensorOp::ExtractDiag { axis_a, axis_b },
                vec![ValueRef::Local(dx)],
                OperationRole::Linearized {
                    active_mask: vec![true],
                },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}

pub fn linearize_embed_diag(
    builder: &mut dyn PrimitiveRuleBuilder,
    tangent_in: &[Option<LocalValueId>],
    axis_a: usize,
    axis_b: usize,
) -> Vec<Option<LocalValueId>> {
    // TODO: ExtractDiag/EmbedDiag could be replaced by Gather/Scatter
    match tangent_in[0] {
        Some(dx) => {
            let out = builder.add_operation(
                StdTensorOp::EmbedDiag { axis_a, axis_b },
                vec![ValueRef::Local(dx)],
                OperationRole::Linearized {
                    active_mask: vec![true],
                },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}

pub fn transpose_extract_diag(
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[ValueRef<StdTensorOp>],
    axis_a: usize,
    axis_b: usize,
    _ctx: &mut ShapeGuardContext,
) -> Vec<Option<LocalValueId>> {
    // TODO: ExtractDiag/EmbedDiag could be replaced by Gather/Scatter
    match cotangent_out[0] {
        Some(ct) => {
            let out = builder.add_operation(
                StdTensorOp::EmbedDiag { axis_a, axis_b },
                vec![ValueRef::Local(ct)],
                OperationRole::Linearized {
                    active_mask: vec![true],
                },
            );
            let padded_axis_a = builder.add_operation(
                StdTensorOp::PadToMatch { axis: axis_a },
                vec![ValueRef::Local(out[0]), inputs[0].clone()],
                OperationRole::Linearized {
                    active_mask: vec![true, false],
                },
            );
            let padded_axis_b = builder.add_operation(
                StdTensorOp::PadToMatch { axis: axis_b },
                vec![ValueRef::Local(padded_axis_a[0]), inputs[0].clone()],
                OperationRole::Linearized {
                    active_mask: vec![true, false],
                },
            );
            vec![Some(padded_axis_b[0])]
        }
        None => vec![None],
    }
}

pub fn transpose_embed_diag(
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[ValueRef<StdTensorOp>],
    axis_a: usize,
    axis_b: usize,
    ctx: &mut ShapeGuardContext,
) -> Vec<Option<LocalValueId>> {
    // TODO: ExtractDiag/EmbedDiag could be replaced by Gather/Scatter
    match cotangent_out[0] {
        Some(ct) => {
            let source_axis = if axis_b <= axis_a { axis_a + 1 } else { axis_a };
            let out = builder.add_operation(
                StdTensorOp::ExtractDiag {
                    axis_a: source_axis,
                    axis_b,
                },
                vec![ValueRef::Local(ct)],
                OperationRole::Linearized {
                    active_mask: vec![true],
                },
            );
            if axis_b < axis_a {
                let rank = ctx.shape_of(&inputs[0]).len();
                let mut perm: Vec<usize> = (0..rank).collect();
                let diag_axis = perm.remove(axis_b);
                perm.insert(axis_a, diag_axis);
                let transposed = builder.add_operation(
                    StdTensorOp::Transpose { perm },
                    vec![ValueRef::Local(out[0])],
                    OperationRole::Linearized {
                        active_mask: vec![true],
                    },
                );
                return vec![Some(transposed[0])];
            }
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}
