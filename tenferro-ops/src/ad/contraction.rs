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

    let lhs_free = compute_free_dims(
        config.lhs_rank,
        &config.lhs_contracting_dims,
        &config.lhs_batch_dims,
    );
    let rhs_free = compute_free_dims(
        config.rhs_rank,
        &config.rhs_contracting_dims,
        &config.rhs_batch_dims,
    );
    let output_rank = config.lhs_batch_dims.len() + lhs_free.len() + rhs_free.len();
    let cotangent = normalize_scalar_cotangent(builder, ct, output_rank);

    let mut result = vec![None, None];

    if active_mask[0] {
        let rhs_conj =
            builder.add_op(StdTensorOp::Conj, vec![inputs[1].clone()], OpMode::Primal)[0];
        let (transpose_config, perm) = transpose_plan_for_lhs(config, &lhs_free, &rhs_free);
        let out = builder.add_op(
            StdTensorOp::DotGeneral(transpose_config),
            vec![cotangent.clone(), ValRef::Local(rhs_conj)],
            OpMode::Linear {
                active_mask: vec![true, false],
            },
        );
        result[0] = Some(add_transpose_if_needed(builder, out[0], &perm));
    }

    if active_mask[1] {
        let lhs_conj =
            builder.add_op(StdTensorOp::Conj, vec![inputs[0].clone()], OpMode::Primal)[0];
        let (transpose_config, perm) = transpose_plan_for_rhs(config, &lhs_free, &rhs_free);
        let out = builder.add_op(
            StdTensorOp::DotGeneral(transpose_config),
            vec![ValRef::Local(lhs_conj), cotangent],
            OpMode::Linear {
                active_mask: vec![false, true],
            },
        );
        result[1] = Some(add_transpose_if_needed(builder, out[0], &perm));
    }

    result
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

fn compute_free_dims(rank: usize, contracting: &[usize], batch: &[usize]) -> Vec<usize> {
    let mut is_bound = vec![false; rank];
    for &dim in batch {
        is_bound[dim] = true;
    }
    for &dim in contracting {
        is_bound[dim] = true;
    }

    (0..rank).filter(|&dim| !is_bound[dim]).collect()
}

fn normalize_scalar_cotangent(
    builder: &mut FragmentBuilder<StdTensorOp>,
    cotangent: LocalValId,
    output_rank: usize,
) -> ValRef<StdTensorOp> {
    if output_rank == 0 {
        let scalar = builder.add_op(
            StdTensorOp::Reshape {
                from_shape: vec![1],
                to_shape: vec![],
            },
            vec![ValRef::Local(cotangent)],
            OpMode::Linear {
                active_mask: vec![true],
            },
        );
        ValRef::Local(scalar[0])
    } else {
        ValRef::Local(cotangent)
    }
}

fn transpose_plan_for_lhs(
    config: &DotGeneralConfig,
    lhs_free: &[usize],
    rhs_free: &[usize],
) -> (DotGeneralConfig, Vec<usize>) {
    let n_batch = config.lhs_batch_dims.len();
    let output_rank = n_batch + lhs_free.len() + rhs_free.len();
    let ct_rhs_free_positions: Vec<usize> = (n_batch + lhs_free.len()..output_rank).collect();

    let rhs_contracting_order =
        compute_free_dims(config.rhs_rank, rhs_free, &config.rhs_batch_dims);
    let mut result_order = Vec::with_capacity(config.lhs_rank);
    result_order.extend(config.lhs_batch_dims.iter().copied());
    result_order.extend(lhs_free.iter().copied());
    for rhs_dim in rhs_contracting_order {
        let pair_idx = config
            .rhs_contracting_dims
            .iter()
            .position(|&dim| dim == rhs_dim)
            .expect("rhs contracting dimension must be paired");
        result_order.push(config.lhs_contracting_dims[pair_idx]);
    }

    (
        DotGeneralConfig {
            lhs_contracting_dims: ct_rhs_free_positions,
            rhs_contracting_dims: rhs_free.to_vec(),
            lhs_batch_dims: (0..n_batch).collect(),
            rhs_batch_dims: config.rhs_batch_dims.clone(),
            lhs_rank: output_rank,
            rhs_rank: config.rhs_rank,
        },
        permutation_to_original_order(config.lhs_rank, &result_order),
    )
}

fn transpose_plan_for_rhs(
    config: &DotGeneralConfig,
    lhs_free: &[usize],
    rhs_free: &[usize],
) -> (DotGeneralConfig, Vec<usize>) {
    let n_batch = config.lhs_batch_dims.len();
    let ct_lhs_free_positions: Vec<usize> = (n_batch..n_batch + lhs_free.len()).collect();

    let lhs_contracting_order =
        compute_free_dims(config.lhs_rank, lhs_free, &config.lhs_batch_dims);
    let mut result_order = Vec::with_capacity(config.rhs_rank);
    result_order.extend(config.rhs_batch_dims.iter().copied());
    for lhs_dim in lhs_contracting_order {
        let pair_idx = config
            .lhs_contracting_dims
            .iter()
            .position(|&dim| dim == lhs_dim)
            .expect("lhs contracting dimension must be paired");
        result_order.push(config.rhs_contracting_dims[pair_idx]);
    }
    result_order.extend(rhs_free.iter().copied());

    let output_rank = n_batch + lhs_free.len() + rhs_free.len();
    (
        DotGeneralConfig {
            lhs_contracting_dims: lhs_free.to_vec(),
            rhs_contracting_dims: ct_lhs_free_positions,
            lhs_batch_dims: config.lhs_batch_dims.clone(),
            rhs_batch_dims: (0..n_batch).collect(),
            lhs_rank: config.lhs_rank,
            rhs_rank: output_rank,
        },
        permutation_to_original_order(config.rhs_rank, &result_order),
    )
}

fn permutation_to_original_order(rank: usize, result_order: &[usize]) -> Vec<usize> {
    let mut perm = vec![0; rank];
    for (result_axis, &original_dim) in result_order.iter().enumerate() {
        perm[original_dim] = result_axis;
    }
    perm
}

fn add_transpose_if_needed(
    builder: &mut FragmentBuilder<StdTensorOp>,
    input: LocalValId,
    perm: &[usize],
) -> LocalValId {
    if perm.iter().enumerate().all(|(dim, &axis)| dim == axis) {
        return input;
    }

    let out = builder.add_op(
        StdTensorOp::Transpose {
            perm: perm.to_vec(),
        },
        vec![ValRef::Local(input)],
        OpMode::Linear {
            active_mask: vec![true],
        },
    );
    out[0]
}
