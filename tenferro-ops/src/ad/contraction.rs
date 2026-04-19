use computegraph::fragment::FragmentBuilder;
use computegraph::types::{GlobalValKey, LocalValId, OpMode, ValRef};
use computegraph::OpEmitter;
use tenferro_tensor::{CompareDir, DotGeneralConfig};

use crate::dim_expr::DimExpr;
use crate::std_tensor_op::StdTensorOp;

pub fn linearize_dot_general(
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    config: &DotGeneralConfig,
) -> Vec<Option<LocalValId>> {
    let mut terms = Vec::with_capacity(2);
    let lhs_rank = config.lhs_rank;
    let rhs_rank = config.rhs_rank;

    if let Some(dx) = tangent_in[0] {
        let term = builder.add_op(
            StdTensorOp::DotGeneral {
                config: config.clone(),
                lhs_rank,
                rhs_rank,
            },
            vec![ValRef::Local(dx), ValRef::External(primal_in[1].clone())],
            OpMode::Linear {
                active_mask: vec![true, false],
            },
        );
        terms.push(term[0]);
    }

    if let Some(dy) = tangent_in[1] {
        let term = builder.add_op(
            StdTensorOp::DotGeneral {
                config: config.clone(),
                lhs_rank,
                rhs_rank,
            },
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

pub fn linearize_nary_einsum(
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    subscripts: &str,
) -> Vec<Option<LocalValId>> {
    let mut terms = Vec::new();

    for (active_idx, tangent) in tangent_in.iter().enumerate() {
        let Some(dt) = tangent else {
            continue;
        };

        let mut inputs = Vec::with_capacity(primal_in.len());
        for (input_idx, key) in primal_in.iter().enumerate() {
            if input_idx == active_idx {
                inputs.push(ValRef::Local(*dt));
            } else {
                inputs.push(ValRef::External(key.clone()));
            }
        }

        let out = builder.add_op(
            StdTensorOp::NaryEinsum {
                subscripts: subscripts.to_string(),
                n_inputs: primal_in.len(),
            },
            inputs,
            OpMode::Linear {
                active_mask: (0..primal_in.len()).map(|idx| idx == active_idx).collect(),
            },
        );
        terms.push(out[0]);
    }

    match terms.as_slice() {
        [] => vec![None],
        [only] => vec![Some(*only)],
        [head, tail @ ..] => {
            let mut result = *head;
            for &term in tail {
                let sum = builder.add_op(
                    StdTensorOp::Add,
                    vec![ValRef::Local(result), ValRef::Local(term)],
                    OpMode::Linear {
                        active_mask: vec![true, true],
                    },
                );
                result = sum[0];
            }
            vec![Some(result)]
        }
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

pub fn linearize_reduce_prod(
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    primal_out: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    axes: &[usize],
    input_shape: &[DimExpr],
) -> Vec<Option<LocalValId>> {
    let Some(dx) = tangent_in[0] else {
        return vec![None];
    };

    let kept_dims = kept_dims(input_shape.len(), axes);
    let prod_broadcast = broadcast_reduction_output(
        builder,
        ValRef::External(primal_out[0].clone()),
        ValRef::External(primal_in[0].clone()),
        input_shape,
        &kept_dims,
    );
    let coeff = builder.add_op(
        StdTensorOp::Div,
        vec![
            ValRef::Local(prod_broadcast),
            ValRef::External(primal_in[0].clone()),
        ],
        OpMode::Primal,
    )[0];
    let scaled_tangent = builder.add_op(
        StdTensorOp::Mul,
        vec![ValRef::Local(coeff), ValRef::Local(dx)],
        OpMode::Linear {
            active_mask: vec![false, true],
        },
    )[0];
    let out = builder.add_op(
        StdTensorOp::ReduceSum {
            axes: axes.to_vec(),
            input_shape: input_shape.to_vec(),
        },
        vec![ValRef::Local(scaled_tangent)],
        OpMode::Linear {
            active_mask: vec![true],
        },
    )[0];
    vec![Some(out)]
}

pub fn linearize_reduce_chooser(
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    primal_out: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    axes: &[usize],
    input_shape: &[DimExpr],
) -> Vec<Option<LocalValId>> {
    let Some(dx) = tangent_in[0] else {
        return vec![None];
    };

    let kept_dims = kept_dims(input_shape.len(), axes);
    let answer_broadcast = broadcast_reduction_output(
        builder,
        ValRef::External(primal_out[0].clone()),
        ValRef::External(primal_in[0].clone()),
        input_shape,
        &kept_dims,
    );
    let indicators = reduction_location_indicators(
        builder,
        ValRef::External(primal_in[0].clone()),
        ValRef::Local(answer_broadcast),
    );
    let weighted_tangent = builder.add_op(
        StdTensorOp::Mul,
        vec![ValRef::Local(indicators), ValRef::Local(dx)],
        OpMode::Linear {
            active_mask: vec![false, true],
        },
    )[0];
    let tangent_sum = builder.add_op(
        StdTensorOp::ReduceSum {
            axes: axes.to_vec(),
            input_shape: input_shape.to_vec(),
        },
        vec![ValRef::Local(weighted_tangent)],
        OpMode::Linear {
            active_mask: vec![true],
        },
    )[0];
    let counts = reduction_location_counts(builder, indicators, axes, input_shape);
    let out = builder.add_op(
        StdTensorOp::Div,
        vec![ValRef::Local(tangent_sum), ValRef::Local(counts)],
        OpMode::Linear {
            active_mask: vec![true, false],
        },
    )[0];
    vec![Some(out)]
}

pub fn transpose_dot_general(
    emitter: &mut impl OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    inputs: &[ValRef<StdTensorOp>],
    mode: &OpMode,
    config: &DotGeneralConfig,
    lhs_rank: usize,
    rhs_rank: usize,
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
        lhs_rank,
        &config.lhs_contracting_dims,
        &config.lhs_batch_dims,
    );
    let rhs_free = compute_free_dims(
        rhs_rank,
        &config.rhs_contracting_dims,
        &config.rhs_batch_dims,
    );
    let output_rank = config.lhs_batch_dims.len() + lhs_free.len() + rhs_free.len();
    let cotangent = normalize_scalar_cotangent(emitter, ct, output_rank);

    let mut result = vec![None, None];

    if active_mask[0] {
        let rhs_conj =
            emitter.add_op(StdTensorOp::Conj, vec![inputs[1].clone()], OpMode::Primal)[0];
        let (transpose_config, new_lhs_rank, new_rhs_rank, perm) =
            transpose_plan_for_lhs(config, lhs_rank, rhs_rank, &lhs_free, &rhs_free);
        let out = emitter.add_op(
            StdTensorOp::DotGeneral {
                config: transpose_config,
                lhs_rank: new_lhs_rank,
                rhs_rank: new_rhs_rank,
            },
            vec![cotangent.clone(), ValRef::Local(rhs_conj)],
            OpMode::Linear {
                active_mask: vec![true, false],
            },
        );
        result[0] = Some(add_transpose_if_needed(emitter, out[0], &perm));
    }

    if active_mask[1] {
        let lhs_conj =
            emitter.add_op(StdTensorOp::Conj, vec![inputs[0].clone()], OpMode::Primal)[0];
        let (transpose_config, new_lhs_rank, new_rhs_rank, perm) =
            transpose_plan_for_rhs(config, lhs_rank, rhs_rank, &lhs_free, &rhs_free);
        let out = emitter.add_op(
            StdTensorOp::DotGeneral {
                config: transpose_config,
                lhs_rank: new_lhs_rank,
                rhs_rank: new_rhs_rank,
            },
            vec![ValRef::Local(lhs_conj), cotangent],
            OpMode::Linear {
                active_mask: vec![false, true],
            },
        );
        result[1] = Some(add_transpose_if_needed(emitter, out[0], &perm));
    }

    result
}

pub fn transpose_nary_einsum(
    emitter: &mut impl OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    inputs: &[ValRef<StdTensorOp>],
    mode: &OpMode,
    subscripts: &str,
    n_inputs: usize,
) -> Vec<Option<LocalValId>> {
    let ct = match cotangent_out[0] {
        Some(ct) => ct,
        None => return vec![None; n_inputs],
    };

    let active_mask = match mode {
        OpMode::Linear { active_mask } => active_mask,
        OpMode::Primal => return vec![None; n_inputs],
    };

    let Some((input_part, output_labels)) = subscripts.split_once("->") else {
        panic!("NaryEinsum subscripts must contain '->': {subscripts}");
    };
    let input_labels: Vec<&str> = input_part.split(',').collect();
    assert_eq!(
        input_labels.len(),
        n_inputs,
        "NaryEinsum input count mismatch for subscripts {subscripts}"
    );

    let mut result = Vec::with_capacity(n_inputs);
    for active_idx in 0..n_inputs {
        if !active_mask[active_idx] {
            result.push(None);
            continue;
        }

        let mut vjp_input_parts = Vec::with_capacity(n_inputs);
        let mut vjp_inputs = Vec::with_capacity(n_inputs);

        vjp_input_parts.push(output_labels.to_string());
        vjp_inputs.push(ValRef::Local(ct));

        for input_idx in 0..n_inputs {
            if input_idx == active_idx {
                continue;
            }

            vjp_input_parts.push(input_labels[input_idx].to_string());
            let conjugated = emitter.add_op(
                StdTensorOp::Conj,
                vec![inputs[input_idx].clone()],
                OpMode::Primal,
            );
            vjp_inputs.push(ValRef::Local(conjugated[0]));
        }

        let out = emitter.add_op(
            StdTensorOp::NaryEinsum {
                subscripts: format!(
                    "{}->{}",
                    vjp_input_parts.join(","),
                    input_labels[active_idx]
                ),
                n_inputs: vjp_inputs.len(),
            },
            vjp_inputs,
            OpMode::Linear {
                active_mask: std::iter::once(true)
                    .chain(std::iter::repeat_n(false, n_inputs.saturating_sub(1)))
                    .collect(),
            },
        );
        result.push(Some(out[0]));
    }

    result
}

pub fn transpose_reduce_sum(
    emitter: &mut impl OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    op: &StdTensorOp,
    inputs: &[ValRef<StdTensorOp>],
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
                let scalar = emitter.add_op(
                    StdTensorOp::Reshape {
                        from_shape: DimExpr::from_concrete(&[1]),
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
            let shape = DimExpr::remap_all(input_shape, 0, 1);
            let needs_shape_source = DimExpr::max_input_idx_all(&shape).is_some_and(|idx| idx > 0);
            let mut op_inputs = vec![cotangent];
            let active_mask = if needs_shape_source {
                op_inputs.push(inputs[0].clone());
                vec![true, false]
            } else {
                vec![true]
            };
            let out = emitter.add_op(
                StdTensorOp::BroadcastInDim {
                    shape,
                    dims: kept_dims,
                },
                op_inputs,
                OpMode::Linear { active_mask },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}

pub fn transpose_reduce_prod(
    emitter: &mut impl OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    inputs: &[ValRef<StdTensorOp>],
    op: &StdTensorOp,
) -> Vec<Option<LocalValId>> {
    let StdTensorOp::ReduceProd { axes, input_shape } = op else {
        unreachable!("transpose_reduce_prod expects ReduceProd");
    };

    match cotangent_out[0] {
        Some(ct) => {
            let kept_dims = kept_dims(input_shape.len(), axes);
            let cotangent = normalize_reduction_cotangent(emitter, ct, &kept_dims);
            let shape = DimExpr::remap_all(input_shape, 0, 1);
            let needs_shape_source = DimExpr::max_input_idx_all(&shape).is_some_and(|idx| idx > 0);
            let mut op_inputs = vec![cotangent];
            let active_mask = if needs_shape_source {
                op_inputs.push(inputs[0].clone());
                vec![true, false]
            } else {
                vec![true]
            };
            let cotangent = emitter.add_op(
                StdTensorOp::BroadcastInDim {
                    shape,
                    dims: kept_dims.clone(),
                },
                op_inputs,
                OpMode::Linear { active_mask },
            )[0];
            let prod = emitter.add_op(op.clone(), vec![inputs[0].clone()], OpMode::Primal)[0];
            let prod_broadcast = broadcast_reduction_output(
                emitter,
                ValRef::Local(prod),
                inputs[0].clone(),
                input_shape,
                &kept_dims,
            );
            let coeff = emitter.add_op(
                StdTensorOp::Div,
                vec![ValRef::Local(prod_broadcast), inputs[0].clone()],
                OpMode::Primal,
            )[0];
            let coeff_conj = emitter.add_op(
                StdTensorOp::Conj,
                vec![ValRef::Local(coeff)],
                OpMode::Primal,
            )[0];
            let out = emitter.add_op(
                StdTensorOp::Mul,
                vec![ValRef::Local(coeff_conj), ValRef::Local(cotangent)],
                OpMode::Linear {
                    active_mask: vec![false, true],
                },
            )[0];
            vec![Some(out)]
        }
        None => vec![None],
    }
}

pub fn transpose_reduce_chooser(
    emitter: &mut impl OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    inputs: &[ValRef<StdTensorOp>],
    op: &StdTensorOp,
) -> Vec<Option<LocalValId>> {
    let (axes, input_shape) = match op {
        StdTensorOp::ReduceMax { axes, input_shape }
        | StdTensorOp::ReduceMin { axes, input_shape } => (axes, input_shape),
        _ => unreachable!("transpose_reduce_chooser expects ReduceMax or ReduceMin"),
    };

    match cotangent_out[0] {
        Some(ct) => {
            let kept_dims = kept_dims(input_shape.len(), axes);
            let cotangent = normalize_reduction_cotangent(emitter, ct, &kept_dims);
            let shape = DimExpr::remap_all(input_shape, 0, 1);
            let needs_shape_source = DimExpr::max_input_idx_all(&shape).is_some_and(|idx| idx > 0);
            let mut op_inputs = vec![cotangent];
            let active_mask = if needs_shape_source {
                op_inputs.push(inputs[0].clone());
                vec![true, false]
            } else {
                vec![true]
            };
            let cotangent = emitter.add_op(
                StdTensorOp::BroadcastInDim {
                    shape,
                    dims: kept_dims.clone(),
                },
                op_inputs,
                OpMode::Linear { active_mask },
            )[0];
            let answer = emitter.add_op(op.clone(), vec![inputs[0].clone()], OpMode::Primal)[0];
            let answer_broadcast = broadcast_reduction_output(
                emitter,
                ValRef::Local(answer),
                inputs[0].clone(),
                input_shape,
                &kept_dims,
            );
            let indicators = reduction_location_indicators(
                emitter,
                inputs[0].clone(),
                ValRef::Local(answer_broadcast),
            );
            let counts = reduction_location_counts(emitter, indicators, axes, input_shape);
            let counts_broadcast = broadcast_reduction_output(
                emitter,
                ValRef::Local(counts),
                inputs[0].clone(),
                input_shape,
                &kept_dims,
            );
            let weights = emitter.add_op(
                StdTensorOp::Div,
                vec![ValRef::Local(indicators), ValRef::Local(counts_broadcast)],
                OpMode::Primal,
            )[0];
            let weights_conj = emitter.add_op(
                StdTensorOp::Conj,
                vec![ValRef::Local(weights)],
                OpMode::Primal,
            )[0];
            let out = emitter.add_op(
                StdTensorOp::Mul,
                vec![ValRef::Local(weights_conj), ValRef::Local(cotangent)],
                OpMode::Linear {
                    active_mask: vec![false, true],
                },
            )[0];
            vec![Some(out)]
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

fn kept_dims(rank: usize, axes: &[usize]) -> Vec<usize> {
    (0..rank).filter(|dim| !axes.contains(dim)).collect()
}

fn normalize_reduction_cotangent(
    emitter: &mut impl OpEmitter<StdTensorOp>,
    cotangent: LocalValId,
    kept_dims: &[usize],
) -> ValRef<StdTensorOp> {
    if kept_dims.is_empty() {
        let scalar = emitter.add_op(
            StdTensorOp::Reshape {
                from_shape: DimExpr::from_concrete(&[1]),
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

fn broadcast_reduction_output(
    emitter: &mut impl OpEmitter<StdTensorOp>,
    output: ValRef<StdTensorOp>,
    shape_source: ValRef<StdTensorOp>,
    input_shape: &[DimExpr],
    kept_dims: &[usize],
) -> LocalValId {
    let shape = DimExpr::remap_all(input_shape, 0, 1);
    let needs_shape_source = DimExpr::max_input_idx_all(&shape).is_some_and(|idx| idx > 0);
    let inputs = if needs_shape_source {
        vec![output, shape_source]
    } else {
        vec![output]
    };
    emitter.add_op(
        StdTensorOp::BroadcastInDim {
            shape,
            dims: kept_dims.to_vec(),
        },
        inputs,
        OpMode::Primal,
    )[0]
}

fn reduction_location_indicators(
    emitter: &mut impl OpEmitter<StdTensorOp>,
    input: ValRef<StdTensorOp>,
    answer_broadcast: ValRef<StdTensorOp>,
) -> LocalValId {
    emitter.add_op(
        StdTensorOp::Compare(CompareDir::Eq),
        vec![input, answer_broadcast],
        OpMode::Primal,
    )[0]
}

fn reduction_location_counts(
    emitter: &mut impl OpEmitter<StdTensorOp>,
    indicators: LocalValId,
    axes: &[usize],
    input_shape: &[DimExpr],
) -> LocalValId {
    emitter.add_op(
        StdTensorOp::ReduceSum {
            axes: axes.to_vec(),
            input_shape: input_shape.to_vec(),
        },
        vec![ValRef::Local(indicators)],
        OpMode::Primal,
    )[0]
}

fn normalize_scalar_cotangent(
    emitter: &mut impl OpEmitter<StdTensorOp>,
    cotangent: LocalValId,
    output_rank: usize,
) -> ValRef<StdTensorOp> {
    if output_rank == 0 {
        let scalar = emitter.add_op(
            StdTensorOp::Reshape {
                from_shape: DimExpr::from_concrete(&[1]),
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
    lhs_rank: usize,
    rhs_rank: usize,
    lhs_free: &[usize],
    rhs_free: &[usize],
) -> (
    DotGeneralConfig,
    /* new_lhs_rank */ usize,
    /* new_rhs_rank */ usize,
    Vec<usize>,
) {
    let n_batch = config.lhs_batch_dims.len();
    let output_rank = lhs_free.len() + rhs_free.len() + n_batch;
    let ct_rhs_free_positions: Vec<usize> =
        (lhs_free.len()..lhs_free.len() + rhs_free.len()).collect();

    let rhs_contracting_order = compute_free_dims(rhs_rank, rhs_free, &config.rhs_batch_dims);
    let mut result_order = Vec::with_capacity(lhs_rank);
    result_order.extend(lhs_free.iter().copied());
    for rhs_dim in rhs_contracting_order {
        let pair_idx = config
            .rhs_contracting_dims
            .iter()
            .position(|&dim| dim == rhs_dim)
            .expect("rhs contracting dimension must be paired");
        result_order.push(config.lhs_contracting_dims[pair_idx]);
    }
    result_order.extend(config.lhs_batch_dims.iter().copied());

    let new_config = DotGeneralConfig {
        lhs_contracting_dims: ct_rhs_free_positions,
        rhs_contracting_dims: rhs_free.to_vec(),
        lhs_batch_dims: (lhs_free.len() + rhs_free.len()..output_rank).collect(),
        rhs_batch_dims: config.rhs_batch_dims.clone(),
        // These two fields are still required by the struct until Task 10.
        lhs_rank: output_rank,
        rhs_rank,
    };
    (
        new_config,
        output_rank,
        rhs_rank,
        permutation_to_original_order(lhs_rank, &result_order),
    )
}

fn transpose_plan_for_rhs(
    config: &DotGeneralConfig,
    lhs_rank: usize,
    rhs_rank: usize,
    lhs_free: &[usize],
    rhs_free: &[usize],
) -> (
    DotGeneralConfig,
    /* new_lhs_rank */ usize,
    /* new_rhs_rank */ usize,
    Vec<usize>,
) {
    let n_batch = config.lhs_batch_dims.len();
    let ct_lhs_free_positions: Vec<usize> = (0..lhs_free.len()).collect();

    let lhs_contracting_order = compute_free_dims(lhs_rank, lhs_free, &config.lhs_batch_dims);
    let mut result_order = Vec::with_capacity(rhs_rank);
    for lhs_dim in lhs_contracting_order {
        let pair_idx = config
            .lhs_contracting_dims
            .iter()
            .position(|&dim| dim == lhs_dim)
            .expect("lhs contracting dimension must be paired");
        result_order.push(config.rhs_contracting_dims[pair_idx]);
    }
    result_order.extend(rhs_free.iter().copied());
    result_order.extend(config.rhs_batch_dims.iter().copied());

    let output_rank = lhs_free.len() + rhs_free.len() + n_batch;
    let new_config = DotGeneralConfig {
        lhs_contracting_dims: lhs_free.to_vec(),
        rhs_contracting_dims: ct_lhs_free_positions,
        lhs_batch_dims: config.lhs_batch_dims.clone(),
        rhs_batch_dims: (lhs_free.len() + rhs_free.len()..output_rank).collect(),
        lhs_rank,
        rhs_rank: output_rank,
    };
    (
        new_config,
        lhs_rank,
        output_rank,
        permutation_to_original_order(rhs_rank, &result_order),
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
    emitter: &mut impl OpEmitter<StdTensorOp>,
    input: LocalValId,
    perm: &[usize],
) -> LocalValId {
    if perm.iter().enumerate().all(|(dim, &axis)| dim == axis) {
        return input;
    }

    let out = emitter.add_op(
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
