use computegraph::fragment::FragmentBuilder;
use computegraph::types::{GlobalValKey, LocalValId, OpMode, ValRef};
use computegraph::OpEmitter;
use tenferro_tensor::PadConfig;

use crate::ad::context::ShapeGuardContext;
use crate::dim_expr::DimExpr;
use crate::std_tensor_op::StdTensorOp;

pub fn linearize_transpose(
    builder: &mut FragmentBuilder<StdTensorOp>,
    tangent_in: &[Option<LocalValId>],
    perm: &[usize],
) -> Vec<Option<LocalValId>> {
    match tangent_in[0] {
        Some(dx) => {
            let out = builder.add_op(
                StdTensorOp::Transpose {
                    perm: perm.to_vec(),
                },
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

pub fn linearize_reshape(
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    op: &StdTensorOp,
) -> Vec<Option<LocalValId>> {
    let StdTensorOp::Reshape { to_shape } = op else {
        unreachable!("linearize_reshape expects Reshape");
    };

    match tangent_in[0] {
        Some(dx) => {
            let needs_shape_source =
                DimExpr::max_input_idx_all(to_shape).is_some_and(|idx| idx > 0);
            let mut op_inputs = vec![ValRef::Local(dx)];
            let active_mask = if needs_shape_source {
                op_inputs.push(ValRef::External(primal_in[1].clone()));
                vec![true, false]
            } else {
                vec![true]
            };
            let out = builder.add_op(
                StdTensorOp::Reshape {
                    to_shape: to_shape.clone(),
                },
                op_inputs,
                OpMode::Linear { active_mask },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}

pub fn linearize_broadcast_in_dim(
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    shape: &[DimExpr],
    dims: &[usize],
) -> Vec<Option<LocalValId>> {
    match tangent_in[0] {
        Some(dx) => {
            let needs_shape_source = DimExpr::max_input_idx_all(shape).is_some_and(|idx| idx > 0);
            let mut op_inputs = vec![ValRef::Local(dx)];
            let active_mask = if needs_shape_source {
                op_inputs.push(ValRef::External(primal_in[1].clone()));
                vec![true, false]
            } else {
                vec![true]
            };
            let out = builder.add_op(
                StdTensorOp::BroadcastInDim {
                    shape: shape.to_vec(),
                    dims: dims.to_vec(),
                },
                op_inputs,
                OpMode::Linear { active_mask },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}

pub fn linearize_convert(
    builder: &mut FragmentBuilder<StdTensorOp>,
    tangent_in: &[Option<LocalValId>],
    from: tenferro_tensor::DType,
    to: tenferro_tensor::DType,
) -> Vec<Option<LocalValId>> {
    match tangent_in[0] {
        Some(dt) => {
            let out = builder.add_op(
                StdTensorOp::Convert { from, to },
                vec![ValRef::Local(dt)],
                OpMode::Linear {
                    active_mask: vec![true],
                },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}

pub fn linearize_tril(
    builder: &mut FragmentBuilder<StdTensorOp>,
    tangent_in: &[Option<LocalValId>],
    k: i64,
) -> Vec<Option<LocalValId>> {
    match tangent_in[0] {
        Some(dx) => {
            let out = builder.add_op(
                StdTensorOp::Tril { k },
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

pub fn linearize_triu(
    builder: &mut FragmentBuilder<StdTensorOp>,
    tangent_in: &[Option<LocalValId>],
    k: i64,
) -> Vec<Option<LocalValId>> {
    match tangent_in[0] {
        Some(dx) => {
            let out = builder.add_op(
                StdTensorOp::Triu { k },
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

pub fn linearize_pad(
    builder: &mut FragmentBuilder<StdTensorOp>,
    tangent_in: &[Option<LocalValId>],
    config: &PadConfig,
) -> Vec<Option<LocalValId>> {
    match tangent_in[0] {
        Some(dx) => {
            let out = builder.add_op(
                StdTensorOp::Pad(config.clone()),
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

pub fn transpose_transpose(
    emitter: &mut impl OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    perm: &[usize],
) -> Vec<Option<LocalValId>> {
    let mut inv = vec![0; perm.len()];
    for (index, &value) in perm.iter().enumerate() {
        inv[value] = index;
    }

    match cotangent_out[0] {
        Some(ct) => {
            let out = emitter.add_op(
                StdTensorOp::Transpose { perm: inv },
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

pub fn transpose_reshape(
    emitter: &mut impl OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    op: &StdTensorOp,
    inputs: &[ValRef<StdTensorOp>],
    ctx: &ShapeGuardContext,
) -> Vec<Option<LocalValId>> {
    let StdTensorOp::Reshape { to_shape: _ } = op else {
        unreachable!("transpose_reshape expects Reshape");
    };

    match cotangent_out[0] {
        Some(ct) => {
            let input_rank = ctx.shape_of(&inputs[0]).len();
            let remapped_to_shape = DimExpr::input_shape(1, input_rank);
            let needs_shape_source =
                DimExpr::max_input_idx_all(&remapped_to_shape).is_some_and(|idx| idx > 0);
            let mut op_inputs = vec![ValRef::Local(ct)];
            let active_mask = if needs_shape_source {
                op_inputs.push(inputs[0].clone());
                vec![true, false]
            } else {
                vec![true]
            };
            let out = emitter.add_op(
                StdTensorOp::Reshape {
                    to_shape: remapped_to_shape,
                },
                op_inputs,
                OpMode::Linear { active_mask },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}

pub fn transpose_broadcast_in_dim(
    emitter: &mut impl OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    shape: &[DimExpr],
    dims: &[usize],
) -> Vec<Option<LocalValId>> {
    let output_rank = shape.len();
    let broadcast_axes: Vec<usize> = (0..output_rank).filter(|dim| !dims.contains(dim)).collect();

    match cotangent_out[0] {
        Some(ct) if broadcast_axes.is_empty() => vec![Some(ct)],
        Some(ct) => {
            let out = emitter.add_op(
                StdTensorOp::ReduceSum {
                    axes: broadcast_axes,
                    input_shape: shape.to_vec(),
                },
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

pub fn transpose_convert(
    emitter: &mut impl OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    mode: &OpMode,
    from: tenferro_tensor::DType,
    to: tenferro_tensor::DType,
) -> Vec<Option<LocalValId>> {
    let is_active = matches!(
        mode,
        OpMode::Linear { active_mask } if active_mask.first().copied().unwrap_or(false)
    );
    if !is_active {
        return vec![None];
    }

    match cotangent_out[0] {
        Some(ct) => {
            let out = emitter.add_op(
                StdTensorOp::Convert { from: to, to: from },
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

pub fn transpose_tril(
    emitter: &mut impl OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    k: i64,
) -> Vec<Option<LocalValId>> {
    match cotangent_out[0] {
        Some(ct) => {
            let out = emitter.add_op(
                StdTensorOp::Tril { k },
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

pub fn transpose_triu(
    emitter: &mut impl OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    k: i64,
) -> Vec<Option<LocalValId>> {
    match cotangent_out[0] {
        Some(ct) => {
            let out = emitter.add_op(
                StdTensorOp::Triu { k },
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
