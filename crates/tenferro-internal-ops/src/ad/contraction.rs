use computegraph::types::{LocalValueId, OperationRole, ValueKey, ValueRef};
use tenferro_tensor::{CompareDir, DType, DotGeneralConfig};
use tidu::{ADRuleError, ADRuleKind, ADRuleResult};

use crate::ad::context::ShapeGuardContext;
use crate::ad::support::{
    conjugate_primal_if_complex, conjugate_primal_if_dtype_complex, convert_fixed_ref_to_dtype,
    convert_linear_to_dtype, dtype_of_or_real, project_linear_to_dtype, promote_dtype,
};
use crate::ad::transpose_input::{
    linearized_inputs_with_inactive_shape_sources, metadata_value_refs,
    shape_exprs_for_value_extent, TransposeInputRef,
};
use crate::ad::zeros::{build_one_like, build_zero_like};
use crate::ad::PrimitiveRuleBuilder;
use crate::std_tensor_op::StdTensorOp;

pub fn linearize_dot_general(
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    config: &DotGeneralConfig,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let lhs_rank = ctx.rank_of(&ValueRef::External(primal_in[0].clone()))?;
    let rhs_rank = ctx.rank_of(&ValueRef::External(primal_in[1].clone()))?;
    config
        .validate_dims_with_ranks(lhs_rank, rhs_rank)
        .map_err(|err| ad_rule_error(format!(
            "invalid DotGeneral config during linearize: {err} (lhs_rank={lhs_rank}, rhs_rank={rhs_rank})"
        ), ADRuleKind::Jvp))?;
    let lhs_dtype = dtype_of_or_real(ctx, &ValueRef::External(primal_in[0].clone()));
    let rhs_dtype = dtype_of_or_real(ctx, &ValueRef::External(primal_in[1].clone()));
    let output_dtype = promote_dtype(lhs_dtype, rhs_dtype);
    let mut terms = Vec::with_capacity(2);

    if let Some(dx) = tangent_in[0] {
        let dx = convert_linear_to_dtype(builder, dx, lhs_dtype, output_dtype);
        let rhs = convert_fixed_ref_to_dtype(
            builder,
            ValueRef::External(primal_in[1].clone()),
            rhs_dtype,
            output_dtype,
        );
        let term = builder.add_operation(
            StdTensorOp::DotGeneral {
                config: config.clone(),
            },
            vec![ValueRef::Local(dx), rhs],
            OperationRole::Linearized {
                active_mask: vec![true, false],
            },
        );
        terms.push(term[0]);
    }

    if let Some(dy) = tangent_in[1] {
        let lhs = convert_fixed_ref_to_dtype(
            builder,
            ValueRef::External(primal_in[0].clone()),
            lhs_dtype,
            output_dtype,
        );
        let dy = convert_linear_to_dtype(builder, dy, rhs_dtype, output_dtype);
        let term = builder.add_operation(
            StdTensorOp::DotGeneral {
                config: config.clone(),
            },
            vec![lhs, ValueRef::Local(dy)],
            OperationRole::Linearized {
                active_mask: vec![false, true],
            },
        );
        terms.push(term[0]);
    }

    match terms.as_slice() {
        [] => Ok(vec![None]),
        [only] => Ok(vec![Some(*only)]),
        [lhs, rhs] => {
            let sum = builder.add_operation(
                StdTensorOp::Add,
                vec![ValueRef::Local(*lhs), ValueRef::Local(*rhs)],
                OperationRole::Linearized {
                    active_mask: vec![true, true],
                },
            );
            Ok(vec![Some(sum[0])])
        }
        _ => unreachable!("dot_general linearization creates at most two terms"),
    }
}

pub fn linearize_reduce_sum(
    builder: &mut dyn PrimitiveRuleBuilder,
    tangent_in: &[Option<LocalValueId>],
    op: &StdTensorOp,
    _axes: &[usize],
) -> Vec<Option<LocalValueId>> {
    match tangent_in[0] {
        Some(dx) => {
            let out = builder.add_operation(
                op.clone(),
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

pub fn linearize_reduce_prod(
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    axes: &[usize],
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let Some(dx) = tangent_in[0] else {
        return Ok(vec![None]);
    };

    let input = ValueRef::External(primal_in[0].clone());
    let input_rank = ctx.rank_of(&input)?;
    let kept_dims = kept_dims(input_rank, axes);
    let prod_broadcast = broadcast_reduction_output(
        builder,
        ValueRef::External(primal_out[0].clone()),
        input.clone(),
        input_rank,
        &kept_dims,
        ctx,
    );
    let input_dtype = ctx.dtype_of(&input)?;
    let coeff = reduce_prod_derivative_coeff(
        builder,
        input,
        ValueRef::Local(prod_broadcast),
        ReductionShapeSpec {
            input_rank,
            kept_dims: &kept_dims,
            axes,
        },
        input_dtype,
        ctx,
    );
    let scaled_tangent = builder.add_operation(
        StdTensorOp::Mul,
        vec![ValueRef::Local(coeff), ValueRef::Local(dx)],
        OperationRole::Linearized {
            active_mask: vec![false, true],
        },
    )[0];
    let out = builder.add_operation(
        StdTensorOp::ReduceSum {
            axes: axes.to_vec(),
        },
        vec![ValueRef::Local(scaled_tangent)],
        OperationRole::Linearized {
            active_mask: vec![true],
        },
    )[0];
    Ok(vec![Some(out)])
}

pub fn linearize_reduce_chooser(
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    axes: &[usize],
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let Some(dx) = tangent_in[0] else {
        return Ok(vec![None]);
    };

    let input = ValueRef::External(primal_in[0].clone());
    let input_rank = ctx.rank_of(&input)?;
    let kept_dims = kept_dims(input_rank, axes);
    let answer_broadcast = broadcast_reduction_output(
        builder,
        ValueRef::External(primal_out[0].clone()),
        input.clone(),
        input_rank,
        &kept_dims,
        ctx,
    );
    let indicators =
        reduction_location_indicators(builder, input.clone(), ValueRef::Local(answer_broadcast));
    let input_dtype = ctx.dtype_of(&input)?;
    let numeric_indicators = numeric_indicators(builder, indicators, input_dtype);
    let weighted_tangent = builder.add_operation(
        StdTensorOp::Mul,
        vec![ValueRef::Local(numeric_indicators), ValueRef::Local(dx)],
        OperationRole::Linearized {
            active_mask: vec![false, true],
        },
    )[0];
    let tangent_sum = builder.add_operation(
        StdTensorOp::ReduceSum {
            axes: axes.to_vec(),
        },
        vec![ValueRef::Local(weighted_tangent)],
        OperationRole::Linearized {
            active_mask: vec![true],
        },
    )[0];
    let counts = reduction_location_counts(builder, numeric_indicators, axes);
    let out = builder.add_operation(
        StdTensorOp::Div,
        vec![ValueRef::Local(tangent_sum), ValueRef::Local(counts)],
        OperationRole::Linearized {
            active_mask: vec![true, false],
        },
    )[0];
    Ok(vec![Some(out)])
}

pub fn transpose_dot_general(
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[TransposeInputRef<'_>],
    mode: &OperationRole,
    config: &DotGeneralConfig,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let ct = match cotangent_out[0] {
        Some(ct) => ct,
        None => return Ok(vec![None, None]),
    };

    let metadata_inputs = metadata_value_refs(inputs);
    let lhs_rank = ctx.rank_of(&metadata_inputs[0])?;
    let rhs_rank = ctx.rank_of(&metadata_inputs[1])?;
    let lhs_dtype = dtype_of_or_real(ctx, &metadata_inputs[0]);
    let rhs_dtype = dtype_of_or_real(ctx, &metadata_inputs[1]);
    let output_dtype = promote_dtype(lhs_dtype, rhs_dtype);

    let active_mask = match mode {
        OperationRole::Linearized { active_mask } => active_mask,
        OperationRole::Primary => return Ok(vec![None, None]),
    };

    config
        .validate_dims_with_ranks(lhs_rank, rhs_rank)
        .map_err(|err| ad_rule_error(format!(
            "invalid DotGeneral config during transpose: {err} (lhs_rank={lhs_rank}, rhs_rank={rhs_rank})"
        ), ADRuleKind::Transpose))?;

    let lhs_free = compute_free_dims(
        lhs_rank,
        &config.lhs_contracting_dims,
        &config.lhs_batch_dims,
    )?;
    let rhs_free = compute_free_dims(
        rhs_rank,
        &config.rhs_contracting_dims,
        &config.rhs_batch_dims,
    )?;
    let output_rank = config.lhs_batch_dims.len() + lhs_free.len() + rhs_free.len();
    let cotangent = normalize_scalar_cotangent(builder, ct, output_rank);

    let mut result = vec![None, None];

    if active_mask[0] {
        let rhs = inputs[1].fixed_value("dot_general", 1)?;
        let rhs_conj = conjugate_primal_if_complex(builder, rhs, ctx)?;
        let rhs_conj = convert_fixed_ref_to_dtype(builder, rhs_conj, rhs_dtype, output_dtype);
        let (transpose_config, new_lhs_rank, new_rhs_rank, perm) =
            transpose_plan_for_lhs(config, lhs_rank, rhs_rank, &lhs_free, &rhs_free)?;
        let out = builder.add_operation(
            StdTensorOp::DotGeneral {
                config: transpose_config,
            },
            vec![cotangent.clone(), rhs_conj],
            OperationRole::Linearized {
                active_mask: vec![true, false],
            },
        );
        let _ = (new_lhs_rank, new_rhs_rank);
        let cotangent = add_transpose_if_needed(builder, out[0], &perm);
        result[0] = Some(project_linear_to_dtype(
            builder,
            cotangent,
            output_dtype,
            lhs_dtype,
        ));
    }

    if active_mask[1] {
        let lhs = inputs[0].fixed_value("dot_general", 0)?;
        let lhs_conj = conjugate_primal_if_complex(builder, lhs, ctx)?;
        let lhs_conj = convert_fixed_ref_to_dtype(builder, lhs_conj, lhs_dtype, output_dtype);
        let (transpose_config, new_lhs_rank, new_rhs_rank, perm) =
            transpose_plan_for_rhs(config, lhs_rank, rhs_rank, &lhs_free, &rhs_free)?;
        let out = builder.add_operation(
            StdTensorOp::DotGeneral {
                config: transpose_config,
            },
            vec![lhs_conj, cotangent],
            OperationRole::Linearized {
                active_mask: vec![false, true],
            },
        );
        let _ = (new_lhs_rank, new_rhs_rank);
        let cotangent = add_transpose_if_needed(builder, out[0], &perm);
        result[1] = Some(project_linear_to_dtype(
            builder,
            cotangent,
            output_dtype,
            rhs_dtype,
        ));
    }

    Ok(result)
}

pub fn transpose_reduce_sum_input(
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    op: &StdTensorOp,
    input: &TransposeInputRef<'_>,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let StdTensorOp::ReduceSum { axes } = op else {
        unreachable!("transpose_reduce_sum_input expects ReduceSum");
    };

    match cotangent_out[0] {
        Some(ct) => {
            let input_key = ValueRef::External(input.key().clone());
            let input_rank = ctx.rank_of(&input_key)?;
            let kept_dims = kept_dims(input_rank, axes);
            let cotangent = if kept_dims.is_empty() {
                let scalar = builder.add_operation(
                    StdTensorOp::Reshape { to_shape: vec![] },
                    vec![ValueRef::Local(ct)],
                    OperationRole::Linearized {
                        active_mask: vec![true],
                    },
                );
                ValueRef::Local(scalar[0])
            } else {
                ValueRef::Local(ct)
            };
            let (shape, shape_sources) = input.shape_operand(input_rank, 1, ctx)?;
            let (op_inputs, active_mask) =
                linearized_inputs_with_inactive_shape_sources(cotangent, shape_sources);
            let out = builder.add_operation(
                StdTensorOp::BroadcastInDim {
                    shape,
                    dims: kept_dims,
                },
                op_inputs,
                OperationRole::Linearized { active_mask },
            );
            Ok(vec![Some(out[0])])
        }
        None => Ok(vec![None]),
    }
}

pub fn transpose_reduce_prod(
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[ValueRef<StdTensorOp>],
    op: &StdTensorOp,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let StdTensorOp::ReduceProd { axes } = op else {
        unreachable!("transpose_reduce_prod expects ReduceProd");
    };

    match cotangent_out[0] {
        Some(ct) => {
            let input_rank = ctx.rank_of(&inputs[0])?;
            let kept_dims = kept_dims(input_rank, axes);
            let cotangent = normalize_reduction_cotangent(builder, ct, &kept_dims);
            let (shape, shape_sources) =
                shape_exprs_for_value_extent(&inputs[0], input_rank, 1, ctx);
            let (op_inputs, active_mask) =
                linearized_inputs_with_inactive_shape_sources(cotangent, shape_sources);
            let cotangent = builder.add_operation(
                StdTensorOp::BroadcastInDim {
                    shape,
                    dims: kept_dims.clone(),
                },
                op_inputs,
                OperationRole::Linearized { active_mask },
            )[0];
            let prod =
                builder.add_operation(op.clone(), vec![inputs[0].clone()], OperationRole::Primary)
                    [0];
            let prod_broadcast = broadcast_reduction_output(
                builder,
                ValueRef::Local(prod),
                inputs[0].clone(),
                input_rank,
                &kept_dims,
                ctx,
            );
            let input_dtype = ctx.dtype_of(&inputs[0])?;
            let coeff = reduce_prod_derivative_coeff(
                builder,
                inputs[0].clone(),
                ValueRef::Local(prod_broadcast),
                ReductionShapeSpec {
                    input_rank,
                    kept_dims: &kept_dims,
                    axes,
                },
                input_dtype,
                ctx,
            );
            let coeff_conj =
                conjugate_primal_if_dtype_complex(builder, ValueRef::Local(coeff), input_dtype);
            let out = builder.add_operation(
                StdTensorOp::Mul,
                vec![coeff_conj, ValueRef::Local(cotangent)],
                OperationRole::Linearized {
                    active_mask: vec![false, true],
                },
            )[0];
            Ok(vec![Some(out)])
        }
        None => Ok(vec![None]),
    }
}

pub fn transpose_reduce_chooser(
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[ValueRef<StdTensorOp>],
    op: &StdTensorOp,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let axes = match op {
        StdTensorOp::ReduceMax { axes } | StdTensorOp::ReduceMin { axes } => axes,
        _ => unreachable!("transpose_reduce_chooser expects ReduceMax or ReduceMin"),
    };

    match cotangent_out[0] {
        Some(ct) => {
            let input_rank = ctx.rank_of(&inputs[0])?;
            let kept_dims = kept_dims(input_rank, axes);
            let cotangent = normalize_reduction_cotangent(builder, ct, &kept_dims);
            let (shape, shape_sources) =
                shape_exprs_for_value_extent(&inputs[0], input_rank, 1, ctx);
            let (op_inputs, active_mask) =
                linearized_inputs_with_inactive_shape_sources(cotangent, shape_sources);
            let cotangent = builder.add_operation(
                StdTensorOp::BroadcastInDim {
                    shape,
                    dims: kept_dims.clone(),
                },
                op_inputs,
                OperationRole::Linearized { active_mask },
            )[0];
            let answer =
                builder.add_operation(op.clone(), vec![inputs[0].clone()], OperationRole::Primary)
                    [0];
            let answer_broadcast = broadcast_reduction_output(
                builder,
                ValueRef::Local(answer),
                inputs[0].clone(),
                input_rank,
                &kept_dims,
                ctx,
            );
            let indicators = reduction_location_indicators(
                builder,
                inputs[0].clone(),
                ValueRef::Local(answer_broadcast),
            );
            let input_dtype = ctx.dtype_of(&inputs[0])?;
            let numeric_indicators = numeric_indicators(builder, indicators, input_dtype);
            let counts = reduction_location_counts(builder, numeric_indicators, axes);
            let counts_broadcast = broadcast_reduction_output(
                builder,
                ValueRef::Local(counts),
                inputs[0].clone(),
                input_rank,
                &kept_dims,
                ctx,
            );
            let weights = builder.add_operation(
                StdTensorOp::Div,
                vec![
                    ValueRef::Local(numeric_indicators),
                    ValueRef::Local(counts_broadcast),
                ],
                OperationRole::Primary,
            )[0];
            let weights_conj =
                conjugate_primal_if_dtype_complex(builder, ValueRef::Local(weights), input_dtype);
            let out = builder.add_operation(
                StdTensorOp::Mul,
                vec![weights_conj, ValueRef::Local(cotangent)],
                OperationRole::Linearized {
                    active_mask: vec![false, true],
                },
            )[0];
            Ok(vec![Some(out)])
        }
        None => Ok(vec![None]),
    }
}

fn compute_free_dims(
    rank: usize,
    contracting: &[usize],
    batch: &[usize],
) -> ADRuleResult<Vec<usize>> {
    let mut is_bound = vec![false; rank];
    for &dim in batch {
        if dim >= rank {
            return Err(ad_rule_error(
                format!("batch dimension {dim} out of bounds for rank {rank}"),
                ADRuleKind::Transpose,
            ));
        }
        is_bound[dim] = true;
    }
    for &dim in contracting {
        if dim >= rank {
            return Err(ad_rule_error(
                format!("contracting dimension {dim} out of bounds for rank {rank}"),
                ADRuleKind::Transpose,
            ));
        }
        is_bound[dim] = true;
    }

    Ok((0..rank).filter(|&dim| !is_bound[dim]).collect())
}

fn kept_dims(rank: usize, axes: &[usize]) -> Vec<usize> {
    // INVARIANT: both sequences are tensor axes, so the scan is rank-bounded;
    // avoiding a HashSet preserves the small-rank fast path.
    (0..rank).filter(|dim| !axes.contains(dim)).collect()
}

fn normalize_reduction_cotangent(
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent: LocalValueId,
    kept_dims: &[usize],
) -> ValueRef<StdTensorOp> {
    if kept_dims.is_empty() {
        let scalar = builder.add_operation(
            StdTensorOp::Reshape { to_shape: vec![] },
            vec![ValueRef::Local(cotangent)],
            OperationRole::Linearized {
                active_mask: vec![true],
            },
        );
        ValueRef::Local(scalar[0])
    } else {
        ValueRef::Local(cotangent)
    }
}

fn broadcast_reduction_output(
    builder: &mut dyn PrimitiveRuleBuilder,
    output: ValueRef<StdTensorOp>,
    shape_source: ValueRef<StdTensorOp>,
    input_rank: usize,
    kept_dims: &[usize],
    ctx: &mut ShapeGuardContext,
) -> LocalValueId {
    let (shape, shape_sources) = shape_exprs_for_value_extent(&shape_source, input_rank, 1, ctx);
    let mut inputs = Vec::with_capacity(1 + shape_sources.len());
    inputs.push(output);
    inputs.extend(shape_sources);
    builder.add_operation(
        StdTensorOp::BroadcastInDim {
            shape,
            dims: kept_dims.to_vec(),
        },
        inputs,
        OperationRole::Primary,
    )[0]
}

struct ReductionShapeSpec<'a> {
    input_rank: usize,
    kept_dims: &'a [usize],
    axes: &'a [usize],
}

fn reduce_prod_derivative_coeff(
    builder: &mut dyn PrimitiveRuleBuilder,
    input: ValueRef<StdTensorOp>,
    prod_broadcast: ValueRef<StdTensorOp>,
    shape: ReductionShapeSpec<'_>,
    dtype: DType,
    ctx: &mut ShapeGuardContext,
) -> LocalValueId {
    let zero = build_zero_like(builder, dtype, input.clone(), shape.input_rank);
    let one = build_one_like(builder, dtype, input.clone(), shape.input_rank);
    let zero_mask = builder.add_operation(
        StdTensorOp::Compare(CompareDir::Eq),
        vec![input.clone(), ValueRef::Local(zero)],
        OperationRole::Primary,
    )[0];
    let numeric_zero_mask = numeric_indicators(builder, zero_mask, dtype);
    let zero_count = builder.add_operation(
        StdTensorOp::ReduceSum {
            axes: shape.axes.to_vec(),
        },
        vec![ValueRef::Local(numeric_zero_mask)],
        OperationRole::Primary,
    )[0];
    let zero_count_broadcast = broadcast_reduction_output(
        builder,
        ValueRef::Local(zero_count),
        input.clone(),
        shape.input_rank,
        shape.kept_dims,
        ctx,
    );
    let safe_input = builder.add_operation(
        StdTensorOp::Select,
        vec![
            ValueRef::Local(zero_mask),
            ValueRef::Local(one),
            input.clone(),
        ],
        OperationRole::Primary,
    )[0];
    let nonzero_prod = builder.add_operation(
        StdTensorOp::ReduceProd {
            axes: shape.axes.to_vec(),
        },
        vec![ValueRef::Local(safe_input)],
        OperationRole::Primary,
    )[0];
    let nonzero_prod_broadcast = broadcast_reduction_output(
        builder,
        ValueRef::Local(nonzero_prod),
        input.clone(),
        shape.input_rank,
        shape.kept_dims,
        ctx,
    );
    let quotient = builder.add_operation(
        StdTensorOp::Div,
        vec![prod_broadcast, ValueRef::Local(safe_input)],
        OperationRole::Primary,
    )[0];
    let zero_coeff = build_zero_like(builder, dtype, input.clone(), shape.input_rank);
    let single_zero_coeff = builder.add_operation(
        StdTensorOp::Select,
        vec![
            ValueRef::Local(zero_mask),
            ValueRef::Local(nonzero_prod_broadcast),
            ValueRef::Local(zero_coeff),
        ],
        OperationRole::Primary,
    )[0];
    let zero_count_zero = build_zero_like(builder, dtype, input.clone(), shape.input_rank);
    let zero_count_is_zero = builder.add_operation(
        StdTensorOp::Compare(CompareDir::Eq),
        vec![
            ValueRef::Local(zero_count_broadcast),
            ValueRef::Local(zero_count_zero),
        ],
        OperationRole::Primary,
    )[0];
    let zero_count_one = build_one_like(builder, dtype, input, shape.input_rank);
    let zero_count_is_one = builder.add_operation(
        StdTensorOp::Compare(CompareDir::Eq),
        vec![
            ValueRef::Local(zero_count_broadcast),
            ValueRef::Local(zero_count_one),
        ],
        OperationRole::Primary,
    )[0];
    let zero_for_multiple = build_zero_like(
        builder,
        dtype,
        ValueRef::Local(single_zero_coeff),
        shape.input_rank,
    );
    let zero_case_coeff = builder.add_operation(
        StdTensorOp::Select,
        vec![
            ValueRef::Local(zero_count_is_one),
            ValueRef::Local(single_zero_coeff),
            ValueRef::Local(zero_for_multiple),
        ],
        OperationRole::Primary,
    )[0];
    builder.add_operation(
        StdTensorOp::Select,
        vec![
            ValueRef::Local(zero_count_is_zero),
            ValueRef::Local(quotient),
            ValueRef::Local(zero_case_coeff),
        ],
        OperationRole::Primary,
    )[0]
}

fn reduction_location_indicators(
    builder: &mut dyn PrimitiveRuleBuilder,
    input: ValueRef<StdTensorOp>,
    answer_broadcast: ValueRef<StdTensorOp>,
) -> LocalValueId {
    builder.add_operation(
        StdTensorOp::Compare(CompareDir::Eq),
        vec![input, answer_broadcast],
        OperationRole::Primary,
    )[0]
}

fn reduction_location_counts(
    builder: &mut dyn PrimitiveRuleBuilder,
    indicators: LocalValueId,
    axes: &[usize],
) -> LocalValueId {
    builder.add_operation(
        StdTensorOp::ReduceSum {
            axes: axes.to_vec(),
        },
        vec![ValueRef::Local(indicators)],
        OperationRole::Primary,
    )[0]
}

fn numeric_indicators(
    builder: &mut dyn PrimitiveRuleBuilder,
    indicators: LocalValueId,
    dtype: DType,
) -> LocalValueId {
    builder.add_operation(
        StdTensorOp::Convert {
            from: DType::Bool,
            to: dtype,
        },
        vec![ValueRef::Local(indicators)],
        OperationRole::Primary,
    )[0]
}

fn normalize_scalar_cotangent(
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent: LocalValueId,
    output_rank: usize,
) -> ValueRef<StdTensorOp> {
    if output_rank == 0 {
        let scalar = builder.add_operation(
            StdTensorOp::Reshape { to_shape: vec![] },
            vec![ValueRef::Local(cotangent)],
            OperationRole::Linearized {
                active_mask: vec![true],
            },
        );
        ValueRef::Local(scalar[0])
    } else {
        ValueRef::Local(cotangent)
    }
}

fn transpose_plan_for_lhs(
    config: &DotGeneralConfig,
    lhs_rank: usize,
    rhs_rank: usize,
    lhs_free: &[usize],
    rhs_free: &[usize],
) -> ADRuleResult<(
    DotGeneralConfig,
    /* new_lhs_rank */ usize,
    /* new_rhs_rank */ usize,
    Vec<usize>,
)> {
    let n_batch = config.lhs_batch_dims.len();
    let output_rank = lhs_free.len() + rhs_free.len() + n_batch;
    let ct_rhs_free_positions: Vec<usize> =
        (lhs_free.len()..lhs_free.len() + rhs_free.len()).collect();

    let rhs_contracting_order = compute_free_dims(rhs_rank, rhs_free, &config.rhs_batch_dims)?;
    let mut result_order = Vec::with_capacity(lhs_rank);
    result_order.extend(lhs_free.iter().copied());
    for rhs_dim in rhs_contracting_order {
        let pair_idx = config
            .rhs_contracting_dims
            .iter()
            .position(|&dim| dim == rhs_dim)
            .ok_or_else(|| {
                ad_rule_error(
                    format!("rhs contracting dimension {rhs_dim} has no lhs pair during transpose"),
                    ADRuleKind::Transpose,
                )
            })?;
        result_order.push(config.lhs_contracting_dims[pair_idx]);
    }
    result_order.extend(config.lhs_batch_dims.iter().copied());

    let new_config = DotGeneralConfig {
        lhs_contracting_dims: ct_rhs_free_positions,
        rhs_contracting_dims: rhs_free.to_vec(),
        lhs_batch_dims: (lhs_free.len() + rhs_free.len()..output_rank).collect(),
        rhs_batch_dims: config.rhs_batch_dims.clone(),
    };
    Ok((
        new_config,
        output_rank,
        rhs_rank,
        permutation_to_original_order(lhs_rank, &result_order),
    ))
}

fn transpose_plan_for_rhs(
    config: &DotGeneralConfig,
    lhs_rank: usize,
    rhs_rank: usize,
    lhs_free: &[usize],
    rhs_free: &[usize],
) -> ADRuleResult<(
    DotGeneralConfig,
    /* new_lhs_rank */ usize,
    /* new_rhs_rank */ usize,
    Vec<usize>,
)> {
    let n_batch = config.lhs_batch_dims.len();
    let ct_lhs_free_positions: Vec<usize> = (0..lhs_free.len()).collect();

    let lhs_contracting_order = compute_free_dims(lhs_rank, lhs_free, &config.lhs_batch_dims)?;
    let mut result_order = Vec::with_capacity(rhs_rank);
    for lhs_dim in lhs_contracting_order {
        let pair_idx = config
            .lhs_contracting_dims
            .iter()
            .position(|&dim| dim == lhs_dim)
            .ok_or_else(|| {
                ad_rule_error(
                    format!("lhs contracting dimension {lhs_dim} has no rhs pair during transpose"),
                    ADRuleKind::Transpose,
                )
            })?;
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
    };
    Ok((
        new_config,
        lhs_rank,
        output_rank,
        permutation_to_original_order(rhs_rank, &result_order),
    ))
}

fn permutation_to_original_order(rank: usize, result_order: &[usize]) -> Vec<usize> {
    let mut perm = vec![0; rank];
    for (result_axis, &original_dim) in result_order.iter().enumerate() {
        perm[original_dim] = result_axis;
    }
    perm
}

fn ad_rule_error(message: impl Into<String>, kind: ADRuleKind) -> ADRuleError {
    ADRuleError::unsupported(message.into(), kind)
}

fn add_transpose_if_needed(
    builder: &mut dyn PrimitiveRuleBuilder,
    input: LocalValueId,
    perm: &[usize],
) -> LocalValueId {
    if perm.iter().enumerate().all(|(dim, &axis)| dim == axis) {
        return input;
    }

    let out = builder.add_operation(
        StdTensorOp::Transpose {
            perm: perm.to_vec(),
        },
        vec![ValueRef::Local(input)],
        OperationRole::Linearized {
            active_mask: vec![true],
        },
    );
    out[0]
}
