use computegraph::types::{LocalValueId, OperationRole, ValueKey, ValueRef};
use tenferro_tensor::SliceConfig;

use crate::ad::context::ShapeGuardContext;
use crate::ad::PrimitiveRuleBuilder;
use crate::std_tensor_op::StdTensorOp;

pub fn linearize_dynamic_truncate(
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    axis: usize,
    ctx: &mut ShapeGuardContext,
) -> Vec<Option<LocalValueId>> {
    match tangent_in[0] {
        Some(dx) => {
            // Checkpointed dynamic truncation records the concrete runtime
            // output extent. Preserve that narrower tangent shape when it is
            // available so later checkpointed steps compile against the alias
            // shape instead of the pre-truncation static upper bound.
            if let Some(limits) = static_truncated_shape(primal_in, primal_out, axis, ctx) {
                let rank = limits.len();
                let out = builder.add_operation(
                    StdTensorOp::Slice(SliceConfig {
                        starts: vec![0; rank],
                        limits,
                        strides: vec![1; rank],
                    }),
                    vec![ValueRef::Local(dx)],
                    OperationRole::Linearized {
                        active_mask: vec![true],
                    },
                );
                return vec![Some(out[0])];
            }

            let out = builder.add_operation(
                StdTensorOp::DynamicTruncate { axis },
                vec![
                    ValueRef::Local(dx),
                    ValueRef::External(primal_in[1].clone()),
                ],
                OperationRole::Linearized {
                    active_mask: vec![true, false],
                },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}

pub fn transpose_dynamic_truncate(
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[ValueRef<StdTensorOp>],
    axis: usize,
) -> Vec<Option<LocalValueId>> {
    match cotangent_out[0] {
        Some(ct) => {
            let out = builder.add_operation(
                StdTensorOp::PadToMatch { axis },
                vec![ValueRef::Local(ct), inputs[0].clone()],
                OperationRole::Linearized {
                    active_mask: vec![true, false],
                },
            );
            vec![Some(out[0]), None]
        }
        None => vec![None, None],
    }
}

pub fn linearize_pad_to_match(
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    axis: usize,
) -> Vec<Option<LocalValueId>> {
    match tangent_in[0] {
        Some(dx) => {
            let out = builder.add_operation(
                StdTensorOp::PadToMatch { axis },
                vec![
                    ValueRef::Local(dx),
                    ValueRef::External(primal_in[1].clone()),
                ],
                OperationRole::Linearized {
                    active_mask: vec![true, false],
                },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}

pub fn transpose_pad_to_match(
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[ValueRef<StdTensorOp>],
    mode: &OperationRole,
    axis: usize,
    ctx: &mut ShapeGuardContext,
) -> Vec<Option<LocalValueId>> {
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
            let out = builder.add_operation(
                StdTensorOp::Slice(SliceConfig {
                    starts: vec![0; rank],
                    limits,
                    strides: vec![1; rank],
                }),
                vec![ValueRef::Local(ct)],
                OperationRole::Linearized {
                    active_mask: vec![true],
                },
            );
            return vec![Some(out[0]), None];
        }
    }

    let size = builder.add_operation(
        StdTensorOp::ShapeOf { axis },
        vec![inputs[0].clone()],
        OperationRole::Linearized {
            active_mask: vec![false],
        },
    );
    let out = builder.add_operation(
        StdTensorOp::DynamicTruncate { axis },
        vec![ValueRef::Local(ct), ValueRef::Local(size[0])],
        OperationRole::Linearized {
            active_mask: vec![true, false],
        },
    );
    vec![Some(out[0]), None]
}

fn first_input_active(mode: &OperationRole) -> bool {
    matches!(
        mode,
        OperationRole::Linearized { active_mask } if active_mask.first().copied().unwrap_or(false)
    )
}

fn static_truncated_shape(
    primal_in: &[ValueKey<StdTensorOp>],
    primal_out: &[ValueKey<StdTensorOp>],
    axis: usize,
    ctx: &mut ShapeGuardContext,
) -> Option<Vec<usize>> {
    let input_shape = ctx
        .try_shape_of(&ValueRef::External(primal_in[0].clone()))?
        .to_vec();
    let output_shape = ctx
        .try_shape_of(&ValueRef::External(primal_out[0].clone()))?
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
