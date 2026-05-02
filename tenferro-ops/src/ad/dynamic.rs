use computegraph::fragment::FragmentBuilder;
use computegraph::types::{GlobalValKey, LocalValId, OpMode, ValRef};
use computegraph::OpEmitter;
use tenferro_tensor::SliceConfig;

use crate::ad::context::ShapeGuardContext;
use crate::std_tensor_op::StdTensorOp;

pub fn linearize_dynamic_truncate(
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    primal_out: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    axis: usize,
    ctx: &mut ShapeGuardContext,
) -> Vec<Option<LocalValId>> {
    match tangent_in[0] {
        Some(dx) => {
            // Checkpointed dynamic truncation records the concrete runtime
            // output extent. Preserve that narrower tangent shape when it is
            // available so later checkpointed steps compile against the alias
            // shape instead of the pre-truncation static upper bound.
            if let Some(limits) = static_truncated_shape(primal_in, primal_out, axis, ctx) {
                let rank = limits.len();
                let out = builder.add_op(
                    StdTensorOp::Slice(SliceConfig {
                        starts: vec![0; rank],
                        limits,
                        strides: vec![1; rank],
                    }),
                    vec![ValRef::Local(dx)],
                    OpMode::Linear {
                        active_mask: vec![true],
                    },
                );
                return vec![Some(out[0])];
            }

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
    mode: &OpMode,
    axis: usize,
    ctx: &mut ShapeGuardContext,
) -> Vec<Option<LocalValId>> {
    let Some(ct) = cotangent_out[0] else {
        return vec![None, None];
    };
    if !first_input_active(mode) {
        return vec![None, None];
    }

    // For concrete input metadata the adjoint is just a prefix slice back to
    // the original input shape. This keeps the transposed graph statically
    // shape-exact across checkpoint aliases.
    if let Some(input_shape) = ctx.try_shape_of(&inputs[0]) {
        assert!(
            axis < input_shape.len(),
            "transpose_pad_to_match: axis {axis} out of bounds for rank {}",
            input_shape.len()
        );
        if let Some(limits) = input_shape
            .iter()
            .map(|dim| dim.constant_value())
            .collect::<Option<Vec<_>>>()
        {
            let rank = limits.len();
            let out = emitter.add_op(
                StdTensorOp::Slice(SliceConfig {
                    starts: vec![0; rank],
                    limits,
                    strides: vec![1; rank],
                }),
                vec![ValRef::Local(ct)],
                OpMode::Linear {
                    active_mask: vec![true],
                },
            );
            return vec![Some(out[0]), None];
        }
    }

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

fn first_input_active(mode: &OpMode) -> bool {
    matches!(
        mode,
        OpMode::Linear { active_mask } if active_mask.first().copied().unwrap_or(false)
    )
}

fn static_truncated_shape(
    primal_in: &[GlobalValKey<StdTensorOp>],
    primal_out: &[GlobalValKey<StdTensorOp>],
    axis: usize,
    ctx: &mut ShapeGuardContext,
) -> Option<Vec<usize>> {
    let input_shape = ctx
        .try_shape_of(&ValRef::External(primal_in[0].clone()))?
        .to_vec();
    let output_shape = ctx
        .try_shape_of(&ValRef::External(primal_out[0].clone()))?
        .to_vec();
    if axis >= input_shape.len() || input_shape.len() != output_shape.len() {
        return None;
    }

    let input_extent = input_shape[axis].constant_value()?;
    let output_extent = output_shape[axis].constant_value()?;
    if output_extent >= input_extent {
        return None;
    }

    output_shape
        .iter()
        .map(|dim| dim.constant_value())
        .collect::<Option<Vec<_>>>()
}
