use computegraph::fragment::FragmentBuilder;
use computegraph::types::{LocalValId, OpMode, ValRef};

use crate::std_tensor_op::StdTensorOp;

pub fn linearize_extract_diag(
    builder: &mut FragmentBuilder<StdTensorOp>,
    tangent_in: &[Option<LocalValId>],
    axis_a: usize,
    axis_b: usize,
) -> Vec<Option<LocalValId>> {
    // TODO: ExtractDiag/EmbedDiag will be replaced by Gather/Scatter (Tier-2)
    match tangent_in[0] {
        Some(dx) => {
            let out = builder.add_op(
                StdTensorOp::ExtractDiag { axis_a, axis_b },
                vec![ValRef::Local(dx)],
                OpMode::Linear {
                    active_mask: vec![true],
                },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}

pub fn linearize_embed_diag(
    builder: &mut FragmentBuilder<StdTensorOp>,
    tangent_in: &[Option<LocalValId>],
    axis_a: usize,
    axis_b: usize,
) -> Vec<Option<LocalValId>> {
    // TODO: ExtractDiag/EmbedDiag will be replaced by Gather/Scatter (Tier-2)
    match tangent_in[0] {
        Some(dx) => {
            let out = builder.add_op(
                StdTensorOp::EmbedDiag { axis_a, axis_b },
                vec![ValRef::Local(dx)],
                OpMode::Linear {
                    active_mask: vec![true],
                },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}

pub fn transpose_extract_diag(
    builder: &mut FragmentBuilder<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    axis_a: usize,
    axis_b: usize,
) -> Vec<Option<LocalValId>> {
    // TODO: ExtractDiag/EmbedDiag will be replaced by Gather/Scatter (Tier-2)
    match cotangent_out[0] {
        Some(ct) => {
            let out = builder.add_op(
                StdTensorOp::EmbedDiag { axis_a, axis_b },
                vec![ValRef::Local(ct)],
                OpMode::Linear {
                    active_mask: vec![true],
                },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}

pub fn transpose_embed_diag(
    builder: &mut FragmentBuilder<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    axis_a: usize,
    axis_b: usize,
) -> Vec<Option<LocalValId>> {
    // TODO: ExtractDiag/EmbedDiag will be replaced by Gather/Scatter (Tier-2)
    match cotangent_out[0] {
        Some(ct) => {
            let out = builder.add_op(
                StdTensorOp::ExtractDiag { axis_a, axis_b },
                vec![ValRef::Local(ct)],
                OpMode::Linear {
                    active_mask: vec![true],
                },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}
