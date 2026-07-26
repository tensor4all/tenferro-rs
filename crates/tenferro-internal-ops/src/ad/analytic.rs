use crate::ad::context::ShapeGuardContext;
use crate::ad::support::{
    conjugate_primal_if_any_dtype_complex, convert_fixed_ref_to_dtype, convert_linear_to_dtype,
    dtype_of_or_real, project_linear_to_dtype, promote_dtype_div_like,
};
use crate::ad::zeros::build_one_like;
use crate::ad::ADRuleResult;
use crate::ad::PrimitiveRuleBuilder;
use computegraph::types::{LocalValueId, OperationRole, ValueKey, ValueRef};

use crate::std_tensor_op::StdTensorOp;

fn emit_fixed_unary(
    builder: &mut dyn PrimitiveRuleBuilder,
    op: StdTensorOp,
    input: ValueRef<StdTensorOp>,
) -> LocalValueId {
    builder.add_operation(
        op,
        vec![input],
        OperationRole::Linearized {
            active_mask: vec![false],
        },
    )[0]
}

fn emit_fixed_binary(
    builder: &mut dyn PrimitiveRuleBuilder,
    op: StdTensorOp,
    lhs: ValueRef<StdTensorOp>,
    rhs: ValueRef<StdTensorOp>,
) -> LocalValueId {
    builder.add_operation(
        op,
        vec![lhs, rhs],
        OperationRole::Linearized {
            active_mask: vec![false, false],
        },
    )[0]
}

fn emit_fixed_neg(
    builder: &mut dyn PrimitiveRuleBuilder,
    input: ValueRef<StdTensorOp>,
) -> LocalValueId {
    emit_fixed_unary(builder, StdTensorOp::Neg, input)
}

fn emit_fixed_add(
    builder: &mut dyn PrimitiveRuleBuilder,
    lhs: ValueRef<StdTensorOp>,
    rhs: ValueRef<StdTensorOp>,
) -> LocalValueId {
    emit_fixed_binary(builder, StdTensorOp::Add, lhs, rhs)
}

fn emit_fixed_sub(
    builder: &mut dyn PrimitiveRuleBuilder,
    lhs: ValueRef<StdTensorOp>,
    rhs: ValueRef<StdTensorOp>,
) -> LocalValueId {
    let neg_rhs = emit_fixed_neg(builder, rhs);
    emit_fixed_add(builder, lhs, ValueRef::Local(neg_rhs))
}

fn emit_fixed_mul(
    builder: &mut dyn PrimitiveRuleBuilder,
    lhs: ValueRef<StdTensorOp>,
    rhs: ValueRef<StdTensorOp>,
) -> LocalValueId {
    emit_fixed_binary(builder, StdTensorOp::Mul, lhs, rhs)
}

fn emit_fixed_div(
    builder: &mut dyn PrimitiveRuleBuilder,
    lhs: ValueRef<StdTensorOp>,
    rhs: ValueRef<StdTensorOp>,
) -> LocalValueId {
    emit_fixed_binary(builder, StdTensorOp::Div, lhs, rhs)
}

fn emit_one_like_fixed(
    builder: &mut dyn PrimitiveRuleBuilder,
    anchor: ValueRef<StdTensorOp>,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<LocalValueId> {
    let dtype = dtype_of_or_real(ctx, &anchor);
    let rank = ctx.rank_of(&anchor)?;
    Ok(build_one_like(builder, dtype, anchor, rank))
}

fn emit_linear_mul_fixed(
    builder: &mut dyn PrimitiveRuleBuilder,
    fixed: ValueRef<StdTensorOp>,
    active: LocalValueId,
) -> LocalValueId {
    builder.add_operation(
        StdTensorOp::Mul,
        vec![fixed, ValueRef::Local(active)],
        OperationRole::Linearized {
            active_mask: vec![false, true],
        },
    )[0]
}

fn emit_linear_div_fixed_denominator(
    builder: &mut dyn PrimitiveRuleBuilder,
    active: LocalValueId,
    denominator: ValueRef<StdTensorOp>,
) -> LocalValueId {
    builder.add_operation(
        StdTensorOp::Div,
        vec![ValueRef::Local(active), denominator],
        OperationRole::Linearized {
            active_mask: vec![true, false],
        },
    )[0]
}

fn conjugate_for_unary_input_dtype(
    builder: &mut dyn PrimitiveRuleBuilder,
    input: ValueRef<StdTensorOp>,
    inputs: &[ValueRef<StdTensorOp>],
    ctx: &mut ShapeGuardContext,
) -> ValueRef<StdTensorOp> {
    let dtype = dtype_of_or_real(ctx, &inputs[0]);
    conjugate_primal_if_any_dtype_complex(builder, input, &[dtype])
}

fn conjugate_for_binary_input_dtypes(
    builder: &mut dyn PrimitiveRuleBuilder,
    input: ValueRef<StdTensorOp>,
    inputs: &[ValueRef<StdTensorOp>],
    ctx: &mut ShapeGuardContext,
) -> ValueRef<StdTensorOp> {
    let lhs_dtype = dtype_of_or_real(ctx, &inputs[0]);
    let rhs_dtype = dtype_of_or_real(ctx, &inputs[1]);
    conjugate_primal_if_any_dtype_complex(builder, input, &[lhs_dtype, rhs_dtype])
}

fn unary_is_active(mode: &OperationRole) -> bool {
    match mode {
        OperationRole::Linearized { active_mask } => active_mask[0],
        OperationRole::Primary => false,
    }
}

pub fn linearize_exp(
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
) -> Vec<Option<LocalValueId>> {
    match tangent_in[0] {
        Some(dx) => {
            let out = builder.add_operation(
                StdTensorOp::Mul,
                vec![
                    ValueRef::External(primal_out[0].clone()),
                    ValueRef::Local(dx),
                ],
                OperationRole::Linearized {
                    active_mask: vec![false, true],
                },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}

pub fn linearize_log(
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
) -> Vec<Option<LocalValueId>> {
    match tangent_in[0] {
        Some(dx) => {
            let out = builder.add_operation(
                StdTensorOp::Div,
                vec![
                    ValueRef::Local(dx),
                    ValueRef::External(primal_in[0].clone()),
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

pub fn linearize_sin(
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
) -> Vec<Option<LocalValueId>> {
    match tangent_in[0] {
        Some(dx) => {
            let cos_x = emit_fixed_unary(
                builder,
                StdTensorOp::Cos,
                ValueRef::External(primal_in[0].clone()),
            );
            vec![Some(emit_linear_mul_fixed(
                builder,
                ValueRef::Local(cos_x),
                dx,
            ))]
        }
        None => vec![None],
    }
}

pub fn linearize_cos(
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
) -> Vec<Option<LocalValueId>> {
    match tangent_in[0] {
        Some(dx) => {
            let sin_x = emit_fixed_unary(
                builder,
                StdTensorOp::Sin,
                ValueRef::External(primal_in[0].clone()),
            );
            let neg_sin_x = emit_fixed_neg(builder, ValueRef::Local(sin_x));
            vec![Some(emit_linear_mul_fixed(
                builder,
                ValueRef::Local(neg_sin_x),
                dx,
            ))]
        }
        None => vec![None],
    }
}

pub fn linearize_tanh(
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    match tangent_in[0] {
        Some(dx) => {
            let y = ValueRef::External(primal_out[0].clone());
            let y_sq = emit_fixed_mul(builder, y.clone(), y.clone());
            let one = emit_one_like_fixed(builder, y, ctx)?;
            let neg_y_sq = emit_fixed_neg(builder, ValueRef::Local(y_sq));
            let coeff = emit_fixed_add(builder, ValueRef::Local(one), ValueRef::Local(neg_y_sq));
            Ok(vec![Some(emit_linear_mul_fixed(
                builder,
                ValueRef::Local(coeff),
                dx,
            ))])
        }
        None => Ok(vec![None]),
    }
}

pub fn linearize_sqrt(
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
) -> Vec<Option<LocalValueId>> {
    match tangent_in[0] {
        Some(dx) => {
            let y = ValueRef::External(primal_out[0].clone());
            let two_y = emit_fixed_add(builder, y.clone(), y);
            vec![Some(emit_linear_div_fixed_denominator(
                builder,
                dx,
                ValueRef::Local(two_y),
            ))]
        }
        None => vec![None],
    }
}

pub fn linearize_rsqrt(
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
) -> Vec<Option<LocalValueId>> {
    match tangent_in[0] {
        Some(dx) => {
            let neg_rsqrt_x = emit_fixed_neg(builder, ValueRef::External(primal_out[0].clone()));
            let x = ValueRef::External(primal_in[0].clone());
            let two_x = emit_fixed_add(builder, x.clone(), x);
            let coeff = emit_fixed_div(
                builder,
                ValueRef::Local(neg_rsqrt_x),
                ValueRef::Local(two_x),
            );
            vec![Some(emit_linear_mul_fixed(
                builder,
                ValueRef::Local(coeff),
                dx,
            ))]
        }
        None => vec![None],
    }
}

pub fn linearize_pow(
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let lhs_dtype = dtype_of_or_real(ctx, &ValueRef::External(primal_in[0].clone()));
    let rhs_dtype = dtype_of_or_real(ctx, &ValueRef::External(primal_in[1].clone()));
    let output_dtype = promote_dtype_div_like(lhs_dtype, rhs_dtype);
    let mut terms = Vec::with_capacity(2);

    if let Some(dx) = tangent_in[0] {
        let one = emit_one_like_fixed(builder, ValueRef::External(primal_in[1].clone()), ctx)?;
        let exponent_minus_one = emit_fixed_sub(
            builder,
            ValueRef::External(primal_in[1].clone()),
            ValueRef::Local(one),
        );
        let promoted_base = convert_fixed_ref_to_dtype(
            builder,
            ValueRef::External(primal_in[0].clone()),
            lhs_dtype,
            output_dtype,
        );
        let promoted_exponent = convert_fixed_ref_to_dtype(
            builder,
            ValueRef::Local(exponent_minus_one),
            rhs_dtype,
            output_dtype,
        );
        let pow_x_y_minus_one =
            emit_fixed_binary(builder, StdTensorOp::Pow, promoted_base, promoted_exponent);
        let promoted_exponent_input = convert_fixed_ref_to_dtype(
            builder,
            ValueRef::External(primal_in[1].clone()),
            rhs_dtype,
            output_dtype,
        );
        let coeff = emit_fixed_mul(
            builder,
            promoted_exponent_input,
            ValueRef::Local(pow_x_y_minus_one),
        );
        let dx = convert_linear_to_dtype(builder, dx, lhs_dtype, output_dtype);
        terms.push(emit_linear_mul_fixed(builder, ValueRef::Local(coeff), dx));
    }

    if let Some(dy) = tangent_in[1] {
        let log_x = emit_fixed_unary(
            builder,
            StdTensorOp::Log,
            ValueRef::External(primal_in[0].clone()),
        );
        let log_x =
            convert_fixed_ref_to_dtype(builder, ValueRef::Local(log_x), lhs_dtype, output_dtype);
        let pow_xy = if lhs_dtype == output_dtype && rhs_dtype == output_dtype {
            ValueRef::External(primal_out[0].clone())
        } else {
            let promoted_base = convert_fixed_ref_to_dtype(
                builder,
                ValueRef::External(primal_in[0].clone()),
                lhs_dtype,
                output_dtype,
            );
            let promoted_exponent = convert_fixed_ref_to_dtype(
                builder,
                ValueRef::External(primal_in[1].clone()),
                rhs_dtype,
                output_dtype,
            );
            ValueRef::Local(emit_fixed_binary(
                builder,
                StdTensorOp::Pow,
                promoted_base,
                promoted_exponent,
            ))
        };
        let coeff = emit_fixed_mul(builder, log_x, pow_xy);
        let dy = convert_linear_to_dtype(builder, dy, rhs_dtype, output_dtype);
        terms.push(emit_linear_mul_fixed(builder, ValueRef::Local(coeff), dy));
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
        _ => unreachable!("pow linearization creates at most two terms"),
    }
}

pub fn linearize_expm1(
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    match tangent_in[0] {
        Some(dx) => {
            let y = ValueRef::External(primal_out[0].clone());
            let one = emit_one_like_fixed(builder, y.clone(), ctx)?;
            let coeff = emit_fixed_add(builder, y, ValueRef::Local(one));
            Ok(vec![Some(emit_linear_mul_fixed(
                builder,
                ValueRef::Local(coeff),
                dx,
            ))])
        }
        None => Ok(vec![None]),
    }
}

pub fn linearize_log1p(
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    match tangent_in[0] {
        Some(dx) => {
            let x = ValueRef::External(primal_in[0].clone());
            let one = emit_one_like_fixed(builder, x.clone(), ctx)?;
            let denom = emit_fixed_add(builder, x, ValueRef::Local(one));
            Ok(vec![Some(emit_linear_div_fixed_denominator(
                builder,
                dx,
                ValueRef::Local(denom),
            ))])
        }
        None => Ok(vec![None]),
    }
}

pub fn transpose_exp(
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[ValueRef<StdTensorOp>],
    mode: &OperationRole,
    ctx: &mut ShapeGuardContext,
) -> Vec<Option<LocalValueId>> {
    if !unary_is_active(mode) {
        return vec![None];
    }
    match cotangent_out[0] {
        Some(ct) => {
            let exp_x = emit_fixed_unary(builder, StdTensorOp::Exp, inputs[0].clone());
            let coeff =
                conjugate_for_unary_input_dtype(builder, ValueRef::Local(exp_x), inputs, ctx);
            vec![Some(emit_linear_mul_fixed(builder, coeff, ct))]
        }
        None => vec![None],
    }
}

pub fn transpose_log(
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[ValueRef<StdTensorOp>],
    mode: &OperationRole,
    ctx: &mut ShapeGuardContext,
) -> Vec<Option<LocalValueId>> {
    if !unary_is_active(mode) {
        return vec![None];
    }
    match cotangent_out[0] {
        Some(ct) => {
            let denominator =
                conjugate_for_unary_input_dtype(builder, inputs[0].clone(), inputs, ctx);
            let out = builder.add_operation(
                StdTensorOp::Div,
                vec![ValueRef::Local(ct), denominator],
                OperationRole::Linearized {
                    active_mask: vec![true, false],
                },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}

pub fn transpose_sin(
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[ValueRef<StdTensorOp>],
    mode: &OperationRole,
    ctx: &mut ShapeGuardContext,
) -> Vec<Option<LocalValueId>> {
    if !unary_is_active(mode) {
        return vec![None];
    }
    match cotangent_out[0] {
        Some(ct) => {
            let cos_x = emit_fixed_unary(builder, StdTensorOp::Cos, inputs[0].clone());
            let coeff =
                conjugate_for_unary_input_dtype(builder, ValueRef::Local(cos_x), inputs, ctx);
            vec![Some(emit_linear_mul_fixed(builder, coeff, ct))]
        }
        None => vec![None],
    }
}

pub fn transpose_cos(
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[ValueRef<StdTensorOp>],
    mode: &OperationRole,
    ctx: &mut ShapeGuardContext,
) -> Vec<Option<LocalValueId>> {
    if !unary_is_active(mode) {
        return vec![None];
    }
    match cotangent_out[0] {
        Some(ct) => {
            let sin_x = emit_fixed_unary(builder, StdTensorOp::Sin, inputs[0].clone());
            let neg_sin_x = emit_fixed_neg(builder, ValueRef::Local(sin_x));
            let coeff =
                conjugate_for_unary_input_dtype(builder, ValueRef::Local(neg_sin_x), inputs, ctx);
            vec![Some(emit_linear_mul_fixed(builder, coeff, ct))]
        }
        None => vec![None],
    }
}

pub fn transpose_tanh(
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[ValueRef<StdTensorOp>],
    mode: &OperationRole,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    if !unary_is_active(mode) {
        return Ok(vec![None]);
    }
    match cotangent_out[0] {
        Some(ct) => {
            let tanh_x = emit_fixed_unary(builder, StdTensorOp::Tanh, inputs[0].clone());
            let tanh_sq = emit_fixed_mul(builder, ValueRef::Local(tanh_x), ValueRef::Local(tanh_x));
            let one = emit_one_like_fixed(builder, inputs[0].clone(), ctx)?;
            let neg_tanh_sq = emit_fixed_neg(builder, ValueRef::Local(tanh_sq));
            let coeff = emit_fixed_add(builder, ValueRef::Local(one), ValueRef::Local(neg_tanh_sq));
            let coeff =
                conjugate_for_unary_input_dtype(builder, ValueRef::Local(coeff), inputs, ctx);
            Ok(vec![Some(emit_linear_mul_fixed(builder, coeff, ct))])
        }
        None => Ok(vec![None]),
    }
}

pub fn transpose_sqrt(
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[ValueRef<StdTensorOp>],
    mode: &OperationRole,
    ctx: &mut ShapeGuardContext,
) -> Vec<Option<LocalValueId>> {
    if !unary_is_active(mode) {
        return vec![None];
    }
    match cotangent_out[0] {
        Some(ct) => {
            let sqrt_x = emit_fixed_unary(builder, StdTensorOp::Sqrt, inputs[0].clone());
            let two_sqrt_x =
                emit_fixed_add(builder, ValueRef::Local(sqrt_x), ValueRef::Local(sqrt_x));
            let denominator =
                conjugate_for_unary_input_dtype(builder, ValueRef::Local(two_sqrt_x), inputs, ctx);
            let out = builder.add_operation(
                StdTensorOp::Div,
                vec![ValueRef::Local(ct), denominator],
                OperationRole::Linearized {
                    active_mask: vec![true, false],
                },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}

pub fn transpose_rsqrt(
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[ValueRef<StdTensorOp>],
    mode: &OperationRole,
    ctx: &mut ShapeGuardContext,
) -> Vec<Option<LocalValueId>> {
    if !unary_is_active(mode) {
        return vec![None];
    }
    match cotangent_out[0] {
        Some(ct) => {
            let rsqrt_x = emit_fixed_unary(builder, StdTensorOp::Rsqrt, inputs[0].clone());
            let neg_rsqrt_x = emit_fixed_neg(builder, ValueRef::Local(rsqrt_x));
            let two_x = emit_fixed_add(builder, inputs[0].clone(), inputs[0].clone());
            let coeff = emit_fixed_div(
                builder,
                ValueRef::Local(neg_rsqrt_x),
                ValueRef::Local(two_x),
            );
            let coeff =
                conjugate_for_unary_input_dtype(builder, ValueRef::Local(coeff), inputs, ctx);
            vec![Some(emit_linear_mul_fixed(builder, coeff, ct))]
        }
        None => vec![None],
    }
}

pub fn transpose_pow(
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[ValueRef<StdTensorOp>],
    mode: &OperationRole,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let ct = match cotangent_out[0] {
        Some(ct) => ct,
        None => return Ok(vec![None, None]),
    };

    let active_mask = match mode {
        OperationRole::Linearized { active_mask } => active_mask,
        OperationRole::Primary => return Ok(vec![None, None]),
    };

    let lhs_dtype = dtype_of_or_real(ctx, &inputs[0]);
    let rhs_dtype = dtype_of_or_real(ctx, &inputs[1]);
    let output_dtype = promote_dtype_div_like(lhs_dtype, rhs_dtype);
    let mut result = vec![None, None];

    if active_mask[0] {
        let one = emit_one_like_fixed(builder, inputs[1].clone(), ctx)?;
        let exponent_minus_one = emit_fixed_sub(builder, inputs[1].clone(), ValueRef::Local(one));
        let promoted_base =
            convert_fixed_ref_to_dtype(builder, inputs[0].clone(), lhs_dtype, output_dtype);
        let promoted_exponent = convert_fixed_ref_to_dtype(
            builder,
            ValueRef::Local(exponent_minus_one),
            rhs_dtype,
            output_dtype,
        );
        let pow_x_y_minus_one =
            emit_fixed_binary(builder, StdTensorOp::Pow, promoted_base, promoted_exponent);
        let promoted_exponent_input =
            convert_fixed_ref_to_dtype(builder, inputs[1].clone(), rhs_dtype, output_dtype);
        let coeff = emit_fixed_mul(
            builder,
            promoted_exponent_input,
            ValueRef::Local(pow_x_y_minus_one),
        );
        let coeff = conjugate_for_binary_input_dtypes(builder, ValueRef::Local(coeff), inputs, ctx);
        let cotangent = emit_linear_mul_fixed(builder, coeff, ct);
        result[0] = Some(project_linear_to_dtype(
            builder,
            cotangent,
            output_dtype,
            lhs_dtype,
        ));
    }

    if active_mask[1] {
        let log_x = emit_fixed_unary(builder, StdTensorOp::Log, inputs[0].clone());
        let promoted_log_x =
            convert_fixed_ref_to_dtype(builder, ValueRef::Local(log_x), lhs_dtype, output_dtype);
        let promoted_base =
            convert_fixed_ref_to_dtype(builder, inputs[0].clone(), lhs_dtype, output_dtype);
        let promoted_exponent =
            convert_fixed_ref_to_dtype(builder, inputs[1].clone(), rhs_dtype, output_dtype);
        let pow_xy = emit_fixed_binary(builder, StdTensorOp::Pow, promoted_base, promoted_exponent);
        let coeff = emit_fixed_mul(builder, promoted_log_x, ValueRef::Local(pow_xy));
        let coeff = conjugate_for_binary_input_dtypes(builder, ValueRef::Local(coeff), inputs, ctx);
        let cotangent = emit_linear_mul_fixed(builder, coeff, ct);
        result[1] = Some(project_linear_to_dtype(
            builder,
            cotangent,
            output_dtype,
            rhs_dtype,
        ));
    }

    Ok(result)
}

pub fn transpose_expm1(
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[ValueRef<StdTensorOp>],
    mode: &OperationRole,
    ctx: &mut ShapeGuardContext,
) -> Vec<Option<LocalValueId>> {
    if !unary_is_active(mode) {
        return vec![None];
    }
    match cotangent_out[0] {
        Some(ct) => {
            let exp_x = emit_fixed_unary(builder, StdTensorOp::Exp, inputs[0].clone());
            let coeff =
                conjugate_for_unary_input_dtype(builder, ValueRef::Local(exp_x), inputs, ctx);
            vec![Some(emit_linear_mul_fixed(builder, coeff, ct))]
        }
        None => vec![None],
    }
}

pub fn transpose_log1p(
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[ValueRef<StdTensorOp>],
    mode: &OperationRole,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    if !unary_is_active(mode) {
        return Ok(vec![None]);
    }
    match cotangent_out[0] {
        Some(ct) => {
            let one = emit_one_like_fixed(builder, inputs[0].clone(), ctx)?;
            let denom = emit_fixed_add(builder, inputs[0].clone(), ValueRef::Local(one));
            let denominator =
                conjugate_for_unary_input_dtype(builder, ValueRef::Local(denom), inputs, ctx);
            let out = builder.add_operation(
                StdTensorOp::Div,
                vec![ValueRef::Local(ct), denominator],
                OperationRole::Linearized {
                    active_mask: vec![true, false],
                },
            );
            Ok(vec![Some(out[0])])
        }
        None => Ok(vec![None]),
    }
}
