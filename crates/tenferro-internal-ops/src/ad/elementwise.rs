use crate::ad::context::ShapeGuardContext;
use crate::ad::support::{
    conjugate_primal_if_any_dtype_complex, convert_fixed_ref_to_dtype, convert_linear_to_dtype,
    dtype_of_or_real, project_linear_to_dtype, promote_dtype_div_like,
};
use crate::ad::PrimitiveRuleBuilder;
use computegraph::types::{LocalValueId, OperationRole, ValueKey, ValueRef};
use tenferro_tensor::CompareDir;

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

fn emit_fixed_div(
    builder: &mut dyn PrimitiveRuleBuilder,
    lhs: ValueRef<StdTensorOp>,
    rhs: ValueRef<StdTensorOp>,
) -> LocalValueId {
    emit_fixed_binary(builder, StdTensorOp::Div, lhs, rhs)
}

fn emit_fixed_compare(
    builder: &mut dyn PrimitiveRuleBuilder,
    dir: CompareDir,
    lhs: ValueRef<StdTensorOp>,
    rhs: ValueRef<StdTensorOp>,
) -> LocalValueId {
    emit_fixed_binary(builder, StdTensorOp::Compare(dir), lhs, rhs)
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

fn emit_linear_select(
    builder: &mut dyn PrimitiveRuleBuilder,
    condition: ValueRef<StdTensorOp>,
    on_true: LocalValueId,
    on_false: LocalValueId,
) -> LocalValueId {
    builder.add_operation(
        StdTensorOp::Select,
        vec![
            condition,
            ValueRef::Local(on_true),
            ValueRef::Local(on_false),
        ],
        OperationRole::Linearized {
            active_mask: vec![false, true, true],
        },
    )[0]
}

fn emit_zero_from_active(
    builder: &mut dyn PrimitiveRuleBuilder,
    active: LocalValueId,
) -> LocalValueId {
    let neg = builder.add_operation(
        StdTensorOp::Neg,
        vec![ValueRef::Local(active)],
        OperationRole::Linearized {
            active_mask: vec![true],
        },
    );
    builder.add_operation(
        StdTensorOp::Add,
        vec![ValueRef::Local(active), ValueRef::Local(neg[0])],
        OperationRole::Linearized {
            active_mask: vec![true, true],
        },
    )[0]
}

fn conjugate_for_input_dtypes(
    builder: &mut dyn PrimitiveRuleBuilder,
    input: ValueRef<StdTensorOp>,
    inputs: &[ValueRef<StdTensorOp>],
    indices: &[usize],
    ctx: &mut ShapeGuardContext,
) -> ValueRef<StdTensorOp> {
    let dtypes: Vec<_> = indices
        .iter()
        .map(|&index| dtype_of_or_real(ctx, &inputs[index]))
        .collect();
    conjugate_primal_if_any_dtype_complex(builder, input, &dtypes)
}

fn select_tangents(
    builder: &mut dyn PrimitiveRuleBuilder,
    condition: ValueRef<StdTensorOp>,
    on_true: Option<LocalValueId>,
    on_false: Option<LocalValueId>,
) -> Option<LocalValueId> {
    match (on_true, on_false) {
        (Some(t), Some(f)) => Some(emit_linear_select(builder, condition, t, f)),
        (Some(t), None) => {
            let zero = emit_zero_from_active(builder, t);
            Some(emit_linear_select(builder, condition, t, zero))
        }
        (None, Some(f)) => {
            let zero = emit_zero_from_active(builder, f);
            Some(emit_linear_select(builder, condition, zero, f))
        }
        (None, None) => None,
    }
}

fn split_cotangent_by_mask(
    builder: &mut dyn PrimitiveRuleBuilder,
    condition: ValueRef<StdTensorOp>,
    cotangent: LocalValueId,
    true_active: bool,
    false_active: bool,
) -> (Option<LocalValueId>, Option<LocalValueId>) {
    if !true_active && !false_active {
        return (None, None);
    }

    let zero = emit_zero_from_active(builder, cotangent);
    let true_ct =
        true_active.then(|| emit_linear_select(builder, condition.clone(), cotangent, zero));
    let false_ct = false_active.then(|| emit_linear_select(builder, condition, zero, cotangent));
    (true_ct, false_ct)
}

fn unary_is_active(mode: &OperationRole) -> bool {
    match mode {
        OperationRole::Linearized { active_mask } => active_mask[0],
        OperationRole::Primary => false,
    }
}

fn active_mask(mode: &OperationRole, len: usize) -> Vec<bool> {
    match mode {
        OperationRole::Linearized { active_mask } => active_mask.clone(),
        OperationRole::Primary => vec![false; len],
    }
}

pub fn linearize_div(
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    ctx: &mut ShapeGuardContext,
) -> Vec<Option<LocalValueId>> {
    let lhs_dtype = dtype_of_or_real(ctx, &ValueRef::External(primal_in[0].clone()));
    let rhs_dtype = dtype_of_or_real(ctx, &ValueRef::External(primal_in[1].clone()));
    let output_dtype = promote_dtype_div_like(lhs_dtype, rhs_dtype);
    let mut terms = Vec::with_capacity(2);

    if let Some(dx) = tangent_in[0] {
        let dx = convert_linear_to_dtype(builder, dx, lhs_dtype, output_dtype);
        let rhs = convert_fixed_ref_to_dtype(
            builder,
            ValueRef::External(primal_in[1].clone()),
            rhs_dtype,
            output_dtype,
        );
        let out = builder.add_operation(
            StdTensorOp::Div,
            vec![ValueRef::Local(dx), rhs],
            OperationRole::Linearized {
                active_mask: vec![true, false],
            },
        );
        terms.push(out[0]);
    }

    if let Some(dy) = tangent_in[1] {
        let quotient = if lhs_dtype == output_dtype && rhs_dtype == output_dtype {
            ValueRef::External(primal_out[0].clone())
        } else {
            let lhs = convert_fixed_ref_to_dtype(
                builder,
                ValueRef::External(primal_in[0].clone()),
                lhs_dtype,
                output_dtype,
            );
            let rhs = convert_fixed_ref_to_dtype(
                builder,
                ValueRef::External(primal_in[1].clone()),
                rhs_dtype,
                output_dtype,
            );
            ValueRef::Local(emit_fixed_div(builder, lhs, rhs))
        };
        let rhs = convert_fixed_ref_to_dtype(
            builder,
            ValueRef::External(primal_in[1].clone()),
            rhs_dtype,
            output_dtype,
        );
        let quotient_over_rhs = emit_fixed_div(builder, quotient, rhs);
        let neg_coeff = emit_fixed_neg(builder, ValueRef::Local(quotient_over_rhs));
        let dy = convert_linear_to_dtype(builder, dy, rhs_dtype, output_dtype);
        terms.push(emit_linear_mul_fixed(
            builder,
            ValueRef::Local(neg_coeff),
            dy,
        ));
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
        _ => unreachable!("div linearization creates at most two terms"),
    }
}

pub fn linearize_abs(
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
) -> Vec<Option<LocalValueId>> {
    match tangent_in[0] {
        Some(dx) => {
            let sign_x = emit_fixed_unary(
                builder,
                StdTensorOp::Sign,
                ValueRef::External(primal_in[0].clone()),
            );
            vec![Some(emit_linear_mul_fixed(
                builder,
                ValueRef::Local(sign_x),
                dx,
            ))]
        }
        None => vec![None],
    }
}

pub fn linearize_sign(
    builder: &mut dyn PrimitiveRuleBuilder,
    tangent_in: &[Option<LocalValueId>],
) -> Vec<Option<LocalValueId>> {
    match tangent_in[0] {
        Some(dx) => vec![Some(emit_zero_from_active(builder, dx))],
        None => vec![None],
    }
}

pub fn linearize_maximum(
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
) -> Vec<Option<LocalValueId>> {
    if tangent_in[0].is_none() && tangent_in[1].is_none() {
        return vec![None];
    }

    let mask = emit_fixed_compare(
        builder,
        CompareDir::Ge,
        ValueRef::External(primal_in[0].clone()),
        ValueRef::External(primal_in[1].clone()),
    );
    vec![select_tangents(
        builder,
        ValueRef::Local(mask),
        tangent_in[0],
        tangent_in[1],
    )]
}

pub fn linearize_minimum(
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
) -> Vec<Option<LocalValueId>> {
    if tangent_in[0].is_none() && tangent_in[1].is_none() {
        return vec![None];
    }

    let mask = emit_fixed_compare(
        builder,
        CompareDir::Le,
        ValueRef::External(primal_in[0].clone()),
        ValueRef::External(primal_in[1].clone()),
    );
    vec![select_tangents(
        builder,
        ValueRef::Local(mask),
        tangent_in[0],
        tangent_in[1],
    )]
}

pub fn linearize_select(
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
) -> Vec<Option<LocalValueId>> {
    vec![select_tangents(
        builder,
        ValueRef::External(primal_in[0].clone()),
        tangent_in[1],
        tangent_in[2],
    )]
}

pub fn linearize_clamp(
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
) -> Vec<Option<LocalValueId>> {
    if tangent_in.iter().all(Option::is_none) {
        return vec![None];
    }

    let upper_mask = emit_fixed_compare(
        builder,
        CompareDir::Le,
        ValueRef::External(primal_in[0].clone()),
        ValueRef::External(primal_in[2].clone()),
    );
    let inner_tangent = select_tangents(
        builder,
        ValueRef::Local(upper_mask),
        tangent_in[0],
        tangent_in[2],
    );

    let inner_primal = emit_fixed_binary(
        builder,
        StdTensorOp::Minimum,
        ValueRef::External(primal_in[0].clone()),
        ValueRef::External(primal_in[2].clone()),
    );
    let lower_mask = emit_fixed_compare(
        builder,
        CompareDir::Ge,
        ValueRef::External(primal_in[1].clone()),
        ValueRef::Local(inner_primal),
    );

    vec![select_tangents(
        builder,
        ValueRef::Local(lower_mask),
        tangent_in[1],
        inner_tangent,
    )]
}

pub fn transpose_div(
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[ValueRef<StdTensorOp>],
    mode: &OperationRole,
    ctx: &mut ShapeGuardContext,
) -> Vec<Option<LocalValueId>> {
    let ct = match cotangent_out[0] {
        Some(ct) => ct,
        None => return vec![None, None],
    };

    let active_mask = match mode {
        OperationRole::Linearized { active_mask } => active_mask,
        OperationRole::Primary => return vec![None, None],
    };

    let lhs_dtype = dtype_of_or_real(ctx, &inputs[0]);
    let rhs_dtype = dtype_of_or_real(ctx, &inputs[1]);
    let output_dtype = promote_dtype_div_like(lhs_dtype, rhs_dtype);
    let mut result = vec![None, None];

    if active_mask[0] {
        let denominator = conjugate_for_input_dtypes(builder, inputs[1].clone(), inputs, &[1], ctx);
        let denominator = convert_fixed_ref_to_dtype(builder, denominator, rhs_dtype, output_dtype);
        let out = builder.add_operation(
            StdTensorOp::Div,
            vec![ValueRef::Local(ct), denominator],
            OperationRole::Linearized {
                active_mask: vec![true, false],
            },
        );
        result[0] = Some(project_linear_to_dtype(
            builder,
            out[0],
            output_dtype,
            lhs_dtype,
        ));
    }

    if active_mask[1] {
        let numerator =
            convert_fixed_ref_to_dtype(builder, inputs[0].clone(), lhs_dtype, output_dtype);
        let denominator =
            convert_fixed_ref_to_dtype(builder, inputs[1].clone(), rhs_dtype, output_dtype);
        let quotient = emit_fixed_div(builder, numerator, denominator.clone());
        let neg_quotient = emit_fixed_neg(builder, ValueRef::Local(quotient));
        let coeff = emit_fixed_div(builder, ValueRef::Local(neg_quotient), denominator);
        let coeff =
            conjugate_for_input_dtypes(builder, ValueRef::Local(coeff), inputs, &[0, 1], ctx);
        let cotangent = emit_linear_mul_fixed(builder, coeff, ct);
        result[1] = Some(project_linear_to_dtype(
            builder,
            cotangent,
            output_dtype,
            rhs_dtype,
        ));
    }

    result
}

pub fn transpose_abs(
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
            let sign_x = emit_fixed_unary(builder, StdTensorOp::Sign, inputs[0].clone());
            vec![Some(emit_linear_mul_fixed(
                builder,
                ValueRef::Local(sign_x),
                ct,
            ))]
        }
        None => vec![None],
    }
}

pub fn transpose_sign(
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    mode: &OperationRole,
) -> Vec<Option<LocalValueId>> {
    if !unary_is_active(mode) {
        return vec![None];
    }
    match cotangent_out[0] {
        Some(ct) => vec![Some(emit_zero_from_active(builder, ct))],
        None => vec![None],
    }
}

pub fn transpose_maximum(
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[ValueRef<StdTensorOp>],
    mode: &OperationRole,
) -> Vec<Option<LocalValueId>> {
    let Some(ct) = cotangent_out[0] else {
        return vec![None, None];
    };
    let active = active_mask(mode, 2);
    if active.iter().all(|is_active| !is_active) {
        return vec![None, None];
    }
    let mask = emit_fixed_compare(
        builder,
        CompareDir::Ge,
        inputs[0].clone(),
        inputs[1].clone(),
    );
    let lhs_active = active.first().copied().unwrap_or(false);
    let rhs_active = active.get(1).copied().unwrap_or(false);
    let (lhs, rhs) =
        split_cotangent_by_mask(builder, ValueRef::Local(mask), ct, lhs_active, rhs_active);
    vec![lhs, rhs]
}

pub fn transpose_minimum(
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[ValueRef<StdTensorOp>],
    mode: &OperationRole,
) -> Vec<Option<LocalValueId>> {
    let Some(ct) = cotangent_out[0] else {
        return vec![None, None];
    };
    let active = active_mask(mode, 2);
    if active.iter().all(|is_active| !is_active) {
        return vec![None, None];
    }
    let mask = emit_fixed_compare(
        builder,
        CompareDir::Le,
        inputs[0].clone(),
        inputs[1].clone(),
    );
    let lhs_active = active.first().copied().unwrap_or(false);
    let rhs_active = active.get(1).copied().unwrap_or(false);
    let (lhs, rhs) =
        split_cotangent_by_mask(builder, ValueRef::Local(mask), ct, lhs_active, rhs_active);
    vec![lhs, rhs]
}

pub fn transpose_select(
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[ValueRef<StdTensorOp>],
    mode: &OperationRole,
) -> Vec<Option<LocalValueId>> {
    let Some(ct) = cotangent_out[0] else {
        return vec![None, None, None];
    };
    let active = active_mask(mode, 3);
    let true_active = active.get(1).copied().unwrap_or(false);
    let false_active = active.get(2).copied().unwrap_or(false);
    if !true_active && !false_active {
        return vec![None, None, None];
    }
    let (on_true, on_false) =
        split_cotangent_by_mask(builder, inputs[0].clone(), ct, true_active, false_active);
    vec![None, on_true, on_false]
}

pub fn transpose_clamp(
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[ValueRef<StdTensorOp>],
    mode: &OperationRole,
) -> Vec<Option<LocalValueId>> {
    let Some(ct) = cotangent_out[0] else {
        return vec![None, None, None];
    };
    let active = active_mask(mode, 3);
    if active.iter().all(|is_active| !is_active) {
        return vec![None, None, None];
    }
    let input_active = active.first().copied().unwrap_or(false);
    let lower_active = active.get(1).copied().unwrap_or(false);
    let upper_active = active.get(2).copied().unwrap_or(false);

    let inner_primal = emit_fixed_binary(
        builder,
        StdTensorOp::Minimum,
        inputs[0].clone(),
        inputs[2].clone(),
    );
    let lower_mask = emit_fixed_compare(
        builder,
        CompareDir::Ge,
        inputs[1].clone(),
        ValueRef::Local(inner_primal),
    );
    let (lower_ct, inner_ct) = split_cotangent_by_mask(
        builder,
        ValueRef::Local(lower_mask),
        ct,
        lower_active,
        input_active || upper_active,
    );

    let (input_ct, upper_ct) = match inner_ct {
        Some(inner_ct) => {
            let upper_mask = emit_fixed_compare(
                builder,
                CompareDir::Le,
                inputs[0].clone(),
                inputs[2].clone(),
            );
            split_cotangent_by_mask(
                builder,
                ValueRef::Local(upper_mask),
                inner_ct,
                input_active,
                upper_active,
            )
        }
        None => (None, None),
    };

    vec![input_ct, lower_ct, upper_ct]
}
