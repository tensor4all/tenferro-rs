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
) -> LocalValueId {
    let neg = emit_fixed_neg(builder, anchor.clone());
    let zero = emit_fixed_add(builder, anchor, ValueRef::Local(neg));
    emit_fixed_unary(builder, StdTensorOp::Exp, ValueRef::Local(zero))
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
) -> Vec<Option<LocalValueId>> {
    match tangent_in[0] {
        Some(dx) => {
            let y = ValueRef::External(primal_out[0].clone());
            let y_sq = emit_fixed_mul(builder, y.clone(), y.clone());
            let one = emit_one_like_fixed(builder, y);
            let neg_y_sq = emit_fixed_neg(builder, ValueRef::Local(y_sq));
            let coeff = emit_fixed_add(builder, ValueRef::Local(one), ValueRef::Local(neg_y_sq));
            vec![Some(emit_linear_mul_fixed(
                builder,
                ValueRef::Local(coeff),
                dx,
            ))]
        }
        None => vec![None],
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
) -> Vec<Option<LocalValueId>> {
    let mut terms = Vec::with_capacity(2);

    if let Some(dx) = tangent_in[0] {
        let pow_over_x = emit_fixed_div(
            builder,
            ValueRef::External(primal_out[0].clone()),
            ValueRef::External(primal_in[0].clone()),
        );
        let coeff = emit_fixed_mul(
            builder,
            ValueRef::External(primal_in[1].clone()),
            ValueRef::Local(pow_over_x),
        );
        terms.push(emit_linear_mul_fixed(builder, ValueRef::Local(coeff), dx));
    }

    if let Some(dy) = tangent_in[1] {
        let log_x = emit_fixed_unary(
            builder,
            StdTensorOp::Log,
            ValueRef::External(primal_in[0].clone()),
        );
        let coeff = emit_fixed_mul(
            builder,
            ValueRef::Local(log_x),
            ValueRef::External(primal_out[0].clone()),
        );
        terms.push(emit_linear_mul_fixed(builder, ValueRef::Local(coeff), dy));
    }

    match terms.as_slice() {
        [] => vec![None],
        [only] => vec![Some(*only)],
        [lhs, rhs] => {
            let sum = builder.add_operation(
                StdTensorOp::Add,
                vec![ValueRef::Local(*lhs), ValueRef::Local(*rhs)],
                OperationRole::Linearized {
                    active_mask: vec![true, true],
                },
            );
            vec![Some(sum[0])]
        }
        _ => unreachable!("pow linearization creates at most two terms"),
    }
}

pub fn linearize_expm1(
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
) -> Vec<Option<LocalValueId>> {
    match tangent_in[0] {
        Some(dx) => {
            let y = ValueRef::External(primal_out[0].clone());
            let one = emit_one_like_fixed(builder, y.clone());
            let coeff = emit_fixed_add(builder, y, ValueRef::Local(one));
            vec![Some(emit_linear_mul_fixed(
                builder,
                ValueRef::Local(coeff),
                dx,
            ))]
        }
        None => vec![None],
    }
}

pub fn linearize_log1p(
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
) -> Vec<Option<LocalValueId>> {
    match tangent_in[0] {
        Some(dx) => {
            let x = ValueRef::External(primal_in[0].clone());
            let one = emit_one_like_fixed(builder, x.clone());
            let denom = emit_fixed_add(builder, x, ValueRef::Local(one));
            vec![Some(emit_linear_div_fixed_denominator(
                builder,
                dx,
                ValueRef::Local(denom),
            ))]
        }
        None => vec![None],
    }
}

pub fn transpose_exp(
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[ValueRef<StdTensorOp>],
    mode: &OperationRole,
) -> Vec<Option<LocalValueId>> {
    if !unary_is_active(mode) {
        return vec![None];
    }
    match cotangent_out[0] {
        Some(ct) => {
            let exp_x = emit_fixed_unary(builder, StdTensorOp::Exp, inputs[0].clone());
            vec![Some(emit_linear_mul_fixed(
                builder,
                ValueRef::Local(exp_x),
                ct,
            ))]
        }
        None => vec![None],
    }
}

pub fn transpose_log(
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[ValueRef<StdTensorOp>],
    mode: &OperationRole,
) -> Vec<Option<LocalValueId>> {
    if !unary_is_active(mode) {
        return vec![None];
    }
    match cotangent_out[0] {
        Some(ct) => {
            let out = builder.add_operation(
                StdTensorOp::Div,
                vec![ValueRef::Local(ct), inputs[0].clone()],
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
) -> Vec<Option<LocalValueId>> {
    if !unary_is_active(mode) {
        return vec![None];
    }
    match cotangent_out[0] {
        Some(ct) => {
            let cos_x = emit_fixed_unary(builder, StdTensorOp::Cos, inputs[0].clone());
            vec![Some(emit_linear_mul_fixed(
                builder,
                ValueRef::Local(cos_x),
                ct,
            ))]
        }
        None => vec![None],
    }
}

pub fn transpose_cos(
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[ValueRef<StdTensorOp>],
    mode: &OperationRole,
) -> Vec<Option<LocalValueId>> {
    if !unary_is_active(mode) {
        return vec![None];
    }
    match cotangent_out[0] {
        Some(ct) => {
            let sin_x = emit_fixed_unary(builder, StdTensorOp::Sin, inputs[0].clone());
            let neg_sin_x = emit_fixed_neg(builder, ValueRef::Local(sin_x));
            vec![Some(emit_linear_mul_fixed(
                builder,
                ValueRef::Local(neg_sin_x),
                ct,
            ))]
        }
        None => vec![None],
    }
}

pub fn transpose_tanh(
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[ValueRef<StdTensorOp>],
    mode: &OperationRole,
) -> Vec<Option<LocalValueId>> {
    if !unary_is_active(mode) {
        return vec![None];
    }
    match cotangent_out[0] {
        Some(ct) => {
            let tanh_x = emit_fixed_unary(builder, StdTensorOp::Tanh, inputs[0].clone());
            let tanh_sq = emit_fixed_mul(builder, ValueRef::Local(tanh_x), ValueRef::Local(tanh_x));
            let one = emit_one_like_fixed(builder, inputs[0].clone());
            let neg_tanh_sq = emit_fixed_neg(builder, ValueRef::Local(tanh_sq));
            let coeff = emit_fixed_add(builder, ValueRef::Local(one), ValueRef::Local(neg_tanh_sq));
            vec![Some(emit_linear_mul_fixed(
                builder,
                ValueRef::Local(coeff),
                ct,
            ))]
        }
        None => vec![None],
    }
}

pub fn transpose_sqrt(
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[ValueRef<StdTensorOp>],
    mode: &OperationRole,
) -> Vec<Option<LocalValueId>> {
    if !unary_is_active(mode) {
        return vec![None];
    }
    match cotangent_out[0] {
        Some(ct) => {
            let sqrt_x = emit_fixed_unary(builder, StdTensorOp::Sqrt, inputs[0].clone());
            let two_sqrt_x =
                emit_fixed_add(builder, ValueRef::Local(sqrt_x), ValueRef::Local(sqrt_x));
            let out = builder.add_operation(
                StdTensorOp::Div,
                vec![ValueRef::Local(ct), ValueRef::Local(two_sqrt_x)],
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
            vec![Some(emit_linear_mul_fixed(
                builder,
                ValueRef::Local(coeff),
                ct,
            ))]
        }
        None => vec![None],
    }
}

pub fn transpose_pow(
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[ValueRef<StdTensorOp>],
    mode: &OperationRole,
) -> Vec<Option<LocalValueId>> {
    let ct = match cotangent_out[0] {
        Some(ct) => ct,
        None => return vec![None, None],
    };

    let active_mask = match mode {
        OperationRole::Linearized { active_mask } => active_mask,
        OperationRole::Primary => return vec![None, None],
    };

    let mut result = vec![None, None];

    if active_mask[0] {
        let pow_xy = emit_fixed_binary(
            builder,
            StdTensorOp::Pow,
            inputs[0].clone(),
            inputs[1].clone(),
        );
        let pow_over_x = emit_fixed_div(builder, ValueRef::Local(pow_xy), inputs[0].clone());
        let coeff = emit_fixed_mul(builder, inputs[1].clone(), ValueRef::Local(pow_over_x));
        result[0] = Some(emit_linear_mul_fixed(builder, ValueRef::Local(coeff), ct));
    }

    if active_mask[1] {
        let log_x = emit_fixed_unary(builder, StdTensorOp::Log, inputs[0].clone());
        let pow_xy = emit_fixed_binary(
            builder,
            StdTensorOp::Pow,
            inputs[0].clone(),
            inputs[1].clone(),
        );
        let coeff = emit_fixed_mul(builder, ValueRef::Local(log_x), ValueRef::Local(pow_xy));
        result[1] = Some(emit_linear_mul_fixed(builder, ValueRef::Local(coeff), ct));
    }

    result
}

pub fn transpose_expm1(
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[ValueRef<StdTensorOp>],
    mode: &OperationRole,
) -> Vec<Option<LocalValueId>> {
    if !unary_is_active(mode) {
        return vec![None];
    }
    match cotangent_out[0] {
        Some(ct) => {
            let exp_x = emit_fixed_unary(builder, StdTensorOp::Exp, inputs[0].clone());
            vec![Some(emit_linear_mul_fixed(
                builder,
                ValueRef::Local(exp_x),
                ct,
            ))]
        }
        None => vec![None],
    }
}

pub fn transpose_log1p(
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[ValueRef<StdTensorOp>],
    mode: &OperationRole,
) -> Vec<Option<LocalValueId>> {
    if !unary_is_active(mode) {
        return vec![None];
    }
    match cotangent_out[0] {
        Some(ct) => {
            let one = emit_one_like_fixed(builder, inputs[0].clone());
            let denom = emit_fixed_add(builder, inputs[0].clone(), ValueRef::Local(one));
            let out = builder.add_operation(
                StdTensorOp::Div,
                vec![ValueRef::Local(ct), ValueRef::Local(denom)],
                OperationRole::Linearized {
                    active_mask: vec![true, false],
                },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}
