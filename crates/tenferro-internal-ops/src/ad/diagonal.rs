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
    axis_a: usize,
    axis_b: usize,
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
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}

pub fn transpose_embed_diag(
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    axis_a: usize,
    axis_b: usize,
) -> Vec<Option<LocalValueId>> {
    // TODO: ExtractDiag/EmbedDiag could be replaced by Gather/Scatter
    match cotangent_out[0] {
        Some(ct) => {
            let out = builder.add_operation(
                StdTensorOp::ExtractDiag { axis_a, axis_b },
                vec![ValueRef::Local(ct)],
                OperationRole::Linearized {
                    active_mask: vec![true],
                },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}
