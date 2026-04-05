use computegraph::fragment::FragmentBuilder;
use computegraph::types::{GlobalValKey, LocalValId, OpMode, ValRef};
use tenferro_tensor::DotGeneralConfig;

use crate::std_tensor_op::StdTensorOp;

pub fn linearize_dot_general(
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    config: &DotGeneralConfig,
) -> Vec<Option<LocalValId>> {
    let mut terms = Vec::with_capacity(2);

    if let Some(dx) = tangent_in[0] {
        let term = builder.add_op(
            StdTensorOp::DotGeneral(config.clone()),
            vec![ValRef::Local(dx), ValRef::External(primal_in[1].clone())],
            OpMode::Linear {
                active_mask: vec![true, false],
            },
        );
        terms.push(term[0]);
    }

    if let Some(dy) = tangent_in[1] {
        let term = builder.add_op(
            StdTensorOp::DotGeneral(config.clone()),
            vec![ValRef::External(primal_in[0].clone()), ValRef::Local(dy)],
            OpMode::Linear {
                active_mask: vec![false, true],
            },
        );
        terms.push(term[0]);
    }

    match terms.as_slice() {
        [] => vec![None],
        [only] => vec![Some(*only)],
        [lhs, rhs] => {
            let sum = builder.add_op(
                StdTensorOp::Add,
                vec![ValRef::Local(*lhs), ValRef::Local(*rhs)],
                OpMode::Linear {
                    active_mask: vec![true, true],
                },
            );
            vec![Some(sum[0])]
        }
        _ => unreachable!("dot_general linearization creates at most two terms"),
    }
}

pub fn linearize_reduce_sum(
    builder: &mut FragmentBuilder<StdTensorOp>,
    tangent_in: &[Option<LocalValId>],
    op: &StdTensorOp,
    _axes: &[usize],
) -> Vec<Option<LocalValId>> {
    match tangent_in[0] {
        Some(dx) => {
            let out = builder.add_op(
                op.clone(),
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

pub fn transpose_dot_general(
    builder: &mut FragmentBuilder<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    inputs: &[ValRef<StdTensorOp>],
    mode: &OpMode,
    config: &DotGeneralConfig,
) -> Vec<Option<LocalValId>> {
    let ct = match cotangent_out[0] {
        Some(ct) => ct,
        None => return vec![None, None],
    };

    let active_mask = match mode {
        OpMode::Linear { active_mask } => active_mask,
        OpMode::Primal => return vec![None, None],
    };

    assert_supported_matrix_dot(config);

    match active_mask.as_slice() {
        [true, false] => {
            let rhs_t = transpose_matrix(builder, inputs[1].clone());
            let out = builder.add_op(
                StdTensorOp::DotGeneral(transpose_config_for_lhs(config)),
                vec![ValRef::Local(ct), ValRef::Local(rhs_t)],
                OpMode::Linear {
                    active_mask: vec![true, false],
                },
            );
            vec![Some(out[0]), None]
        }
        [false, true] => {
            let lhs_t = transpose_matrix(builder, inputs[0].clone());
            let out = builder.add_op(
                StdTensorOp::DotGeneral(transpose_config_for_rhs(config)),
                vec![ValRef::Local(lhs_t), ValRef::Local(ct)],
                OpMode::Linear {
                    active_mask: vec![false, true],
                },
            );
            vec![None, Some(out[0])]
        }
        [false, false] => vec![None, None],
        _ => todo!("transpose_dot_general only supports single-active matrix terms"),
    }
}

pub fn transpose_reduce_sum(
    builder: &mut FragmentBuilder<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    op: &StdTensorOp,
) -> Vec<Option<LocalValId>> {
    let StdTensorOp::ReduceSum { axes, input_shape } = op else {
        unreachable!("transpose_reduce_sum expects ReduceSum");
    };

    match cotangent_out[0] {
        Some(ct) => {
            let kept_dims = (0..input_shape.len())
                .filter(|dim| !axes.contains(dim))
                .collect::<Vec<_>>();
            let cotangent = if kept_dims.is_empty() {
                let scalar = builder.add_op(
                    StdTensorOp::Reshape {
                        from_shape: vec![1],
                        to_shape: vec![],
                    },
                    vec![ValRef::Local(ct)],
                    OpMode::Linear {
                        active_mask: vec![true],
                    },
                );
                ValRef::Local(scalar[0])
            } else {
                ValRef::Local(ct)
            };
            let out = builder.add_op(
                StdTensorOp::BroadcastInDim {
                    shape: input_shape.clone(),
                    dims: kept_dims,
                },
                vec![cotangent],
                OpMode::Linear {
                    active_mask: vec![true],
                },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}

fn assert_supported_matrix_dot(config: &DotGeneralConfig) {
    assert!(
        config.lhs_batch_dims.is_empty() && config.rhs_batch_dims.is_empty(),
        "transpose_dot_general only supports non-batched matrix contractions"
    );
    assert_eq!(
        config.lhs_contracting_dims,
        vec![1],
        "transpose_dot_general only supports lhs_contracting_dims=[1]"
    );
    assert_eq!(
        config.rhs_contracting_dims,
        vec![0],
        "transpose_dot_general only supports rhs_contracting_dims=[0]"
    );
}

fn transpose_matrix(
    builder: &mut FragmentBuilder<StdTensorOp>,
    input: ValRef<StdTensorOp>,
) -> LocalValId {
    let out = builder.add_op(
        StdTensorOp::Transpose { perm: vec![1, 0] },
        vec![input],
        OpMode::Linear {
            active_mask: vec![true],
        },
    );
    out[0]
}

fn transpose_config_for_lhs(_config: &DotGeneralConfig) -> DotGeneralConfig {
    DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    }
}

fn transpose_config_for_rhs(_config: &DotGeneralConfig) -> DotGeneralConfig {
    DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    }
}
