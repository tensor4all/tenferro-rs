use computegraph::fragment::FragmentBuilder;
use computegraph::types::{GlobalValKey, LocalValId, OpMode, ValRef};
use computegraph::OpEmitter;

use crate::std_tensor_op::StdTensorOp;

pub fn linearize_dynamic_truncate(
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    axis: usize,
) -> Vec<Option<LocalValId>> {
    match tangent_in[0] {
        Some(dx) => {
            let out = builder.add_op(
                StdTensorOp::DynamicTruncate { axis },
                vec![ValRef::Local(dx), ValRef::External(primal_in[1].clone())],
                OpMode::Linear {
                    active_mask: vec![true, false],
                },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}

pub fn transpose_dynamic_truncate(
    emitter: &mut impl OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    inputs: &[ValRef<StdTensorOp>],
    axis: usize,
) -> Vec<Option<LocalValId>> {
    match cotangent_out[0] {
        Some(ct) => {
            let out = emitter.add_op(
                StdTensorOp::PadToMatch { axis },
                vec![ValRef::Local(ct), inputs[0].clone()],
                OpMode::Linear {
                    active_mask: vec![true, false],
                },
            );
            vec![Some(out[0]), None]
        }
        None => vec![None, None],
    }
}

pub fn linearize_pad_to_match(
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    axis: usize,
) -> Vec<Option<LocalValId>> {
    match tangent_in[0] {
        Some(dx) => {
            let out = builder.add_op(
                StdTensorOp::PadToMatch { axis },
                vec![ValRef::Local(dx), ValRef::External(primal_in[1].clone())],
                OpMode::Linear {
                    active_mask: vec![true, false],
                },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}

pub fn transpose_pad_to_match(
    emitter: &mut impl OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    inputs: &[ValRef<StdTensorOp>],
    axis: usize,
) -> Vec<Option<LocalValId>> {
    match cotangent_out[0] {
        Some(ct) => {
            let size = emitter.add_op(
                StdTensorOp::ShapeOf { axis },
                vec![inputs[0].clone()],
                OpMode::Linear {
                    active_mask: vec![false],
                },
            );
            let out = emitter.add_op(
                StdTensorOp::DynamicTruncate { axis },
                vec![ValRef::Local(ct), ValRef::Local(size[0])],
                OpMode::Linear {
                    active_mask: vec![true, false],
                },
            );
            vec![Some(out[0]), None]
        }
        None => vec![None, None],
    }
}
