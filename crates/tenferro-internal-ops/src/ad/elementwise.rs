use crate::ad::context::ShapeGuardContext;
use crate::ad::support::{
    conjugate_primal_if_any_dtype_complex, convert_fixed_ref_to_dtype, convert_linear_to_dtype,
    dtype_of_or_real, project_linear_to_dtype, promote_dtype_div_like,
};
use crate::ad::transpose_input::{metadata_value_refs, TransposeInputRef};
use crate::ad::PrimitiveRuleBuilder;
use computegraph::types::{LocalValueId, OperationRole, ValueKey, ValueRef};
use tenferro_tensor::{CompareDir, DType};
use tidu::ADRuleResult;

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

fn emit_linear_div_fixed(
    builder: &mut dyn PrimitiveRuleBuilder,
    active: LocalValueId,
    fixed: ValueRef<StdTensorOp>,
) -> LocalValueId {
    builder.add_operation(
        StdTensorOp::Div,
        vec![ValueRef::Local(active), fixed],
        OperationRole::Linearized {
            active_mask: vec![true, false],
        },
    )[0]
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

fn emit_scalar_constant(
    builder: &mut dyn PrimitiveRuleBuilder,
    dtype: DType,
    value: f64,
) -> LocalValueId {
    let bytes = match dtype {
        DType::F32 => (value as f32).to_le_bytes().to_vec(),
        DType::F64 => value.to_le_bytes().to_vec(),
        DType::I32 => (value as i32).to_le_bytes().to_vec(),
        DType::I64 => (value as i64).to_le_bytes().to_vec(),
        DType::Bool => vec![(value != 0.0) as u8],
        DType::C32 => {
            let mut bytes = Vec::with_capacity(8);
            bytes.extend_from_slice(&(value as f32).to_le_bytes());
            bytes.extend_from_slice(&0.0_f32.to_le_bytes());
            bytes
        }
        DType::C64 => {
            let mut bytes = Vec::with_capacity(16);
            bytes.extend_from_slice(&value.to_le_bytes());
            bytes.extend_from_slice(&0.0_f64.to_le_bytes());
            bytes
        }
    };
    builder.add_operation(
        StdTensorOp::Constant { dtype, bytes },
        vec![],
        OperationRole::Primary,
    )[0]
}

fn sum_linear_terms(
    builder: &mut dyn PrimitiveRuleBuilder,
    terms: &[LocalValueId],
) -> Option<LocalValueId> {
    match terms {
        [] => None,
        [only] => Some(*only),
        [first, rest @ ..] => {
            let mut acc = *first;
            for term in rest {
                acc = builder.add_operation(
                    StdTensorOp::Add,
                    vec![ValueRef::Local(acc), ValueRef::Local(*term)],
                    OperationRole::Linearized {
                        active_mask: vec![true, true],
                    },
                )[0];
            }
            Some(acc)
        }
    }
}

fn mask_active_by_conditions(
    builder: &mut dyn PrimitiveRuleBuilder,
    active: LocalValueId,
    conditions: &[LocalValueId],
) -> LocalValueId {
    let zero = emit_zero_from_active(builder, active);
    let mut value = active;
    for condition in conditions {
        value = emit_linear_select(builder, ValueRef::Local(*condition), value, zero);
    }
    value
}

fn balanced_extrema_contribution(
    builder: &mut dyn PrimitiveRuleBuilder,
    active: LocalValueId,
    self_eq_output: LocalValueId,
    other_eq_output: LocalValueId,
    dtype: DType,
) -> LocalValueId {
    let selected = mask_active_by_conditions(builder, active, &[self_eq_output]);
    let two = emit_scalar_constant(builder, dtype, 2.0);
    let half = emit_linear_div_fixed(builder, selected, ValueRef::Local(two));
    emit_linear_select(builder, ValueRef::Local(other_eq_output), half, selected)
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

fn abs_output_dtype(input_dtype: DType) -> DType {
    match input_dtype {
        DType::C32 => DType::F32,
        DType::C64 => DType::F64,
        other => other,
    }
}

fn is_complex_dtype(dtype: DType) -> bool {
    matches!(dtype, DType::C32 | DType::C64)
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
    ctx: &mut ShapeGuardContext,
) -> Vec<Option<LocalValueId>> {
    match tangent_in[0] {
        Some(dx) => {
            let input_ref = ValueRef::External(primal_in[0].clone());
            let input_dtype = dtype_of_or_real(ctx, &input_ref);
            let output_dtype = abs_output_dtype(input_dtype);
            let sign_x = emit_fixed_unary(builder, StdTensorOp::Sign, input_ref);
            let coeff = if is_complex_dtype(input_dtype) {
                emit_fixed_unary(builder, StdTensorOp::Conj, ValueRef::Local(sign_x))
            } else {
                sign_x
            };
            let tangent = emit_linear_mul_fixed(builder, ValueRef::Local(coeff), dx);
            vec![Some(project_linear_to_dtype(
                builder,
                tangent,
                input_dtype,
                output_dtype,
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
    ctx: &mut ShapeGuardContext,
) -> Vec<Option<LocalValueId>> {
    if tangent_in[0].is_none() && tangent_in[1].is_none() {
        return vec![None];
    }

    let lhs = ValueRef::External(primal_in[0].clone());
    let rhs = ValueRef::External(primal_in[1].clone());
    let dtype = dtype_of_or_real(ctx, &lhs);
    let output = emit_fixed_binary(builder, StdTensorOp::Maximum, lhs.clone(), rhs.clone());
    let lhs_eq_output = emit_fixed_compare(
        builder,
        CompareDir::Eq,
        lhs.clone(),
        ValueRef::Local(output),
    );
    let rhs_eq_output = emit_fixed_compare(builder, CompareDir::Eq, rhs, ValueRef::Local(output));

    let mut terms = Vec::new();
    if let Some(lhs_tangent) = tangent_in[0] {
        terms.push(balanced_extrema_contribution(
            builder,
            lhs_tangent,
            lhs_eq_output,
            rhs_eq_output,
            dtype,
        ));
    }
    if let Some(rhs_tangent) = tangent_in[1] {
        terms.push(balanced_extrema_contribution(
            builder,
            rhs_tangent,
            rhs_eq_output,
            lhs_eq_output,
            dtype,
        ));
    }
    vec![sum_linear_terms(builder, &terms)]
}

pub fn linearize_minimum(
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    ctx: &mut ShapeGuardContext,
) -> Vec<Option<LocalValueId>> {
    if tangent_in[0].is_none() && tangent_in[1].is_none() {
        return vec![None];
    }

    let lhs = ValueRef::External(primal_in[0].clone());
    let rhs = ValueRef::External(primal_in[1].clone());
    let dtype = dtype_of_or_real(ctx, &lhs);
    let output = emit_fixed_binary(builder, StdTensorOp::Minimum, lhs.clone(), rhs.clone());
    let lhs_eq_output = emit_fixed_compare(
        builder,
        CompareDir::Eq,
        lhs.clone(),
        ValueRef::Local(output),
    );
    let rhs_eq_output = emit_fixed_compare(builder, CompareDir::Eq, rhs, ValueRef::Local(output));

    let mut terms = Vec::new();
    if let Some(lhs_tangent) = tangent_in[0] {
        terms.push(balanced_extrema_contribution(
            builder,
            lhs_tangent,
            lhs_eq_output,
            rhs_eq_output,
            dtype,
        ));
    }
    if let Some(rhs_tangent) = tangent_in[1] {
        terms.push(balanced_extrema_contribution(
            builder,
            rhs_tangent,
            rhs_eq_output,
            lhs_eq_output,
            dtype,
        ));
    }
    vec![sum_linear_terms(builder, &terms)]
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

    let input = ValueRef::External(primal_in[0].clone());
    let lower = ValueRef::External(primal_in[1].clone());
    let upper = ValueRef::External(primal_in[2].clone());
    let input_gt_lower = emit_fixed_compare(builder, CompareDir::Gt, input.clone(), lower.clone());
    let input_lt_upper = emit_fixed_compare(builder, CompareDir::Lt, input.clone(), upper.clone());
    let lower_gt_input = emit_fixed_compare(builder, CompareDir::Gt, lower.clone(), input.clone());
    let lower_lt_upper = emit_fixed_compare(builder, CompareDir::Lt, lower.clone(), upper.clone());
    let max_input_lower = emit_fixed_binary(builder, StdTensorOp::Maximum, input, lower);
    let upper_lt_max_input_lower = emit_fixed_compare(
        builder,
        CompareDir::Lt,
        upper,
        ValueRef::Local(max_input_lower),
    );

    let mut terms = Vec::new();
    if let Some(d_input) = tangent_in[0] {
        terms.push(mask_active_by_conditions(
            builder,
            d_input,
            &[input_gt_lower, input_lt_upper],
        ));
    }
    if let Some(d_lower) = tangent_in[1] {
        terms.push(mask_active_by_conditions(
            builder,
            d_lower,
            &[lower_gt_input, lower_lt_upper],
        ));
    }
    if let Some(d_upper) = tangent_in[2] {
        terms.push(mask_active_by_conditions(
            builder,
            d_upper,
            &[upper_lt_max_input_lower],
        ));
    }

    vec![sum_linear_terms(builder, &terms)]
}

pub fn transpose_div(
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[TransposeInputRef<'_>],
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

    let metadata_inputs = metadata_value_refs(inputs);
    let lhs_dtype = dtype_of_or_real(ctx, &metadata_inputs[0]);
    let rhs_dtype = dtype_of_or_real(ctx, &metadata_inputs[1]);
    let output_dtype = promote_dtype_div_like(lhs_dtype, rhs_dtype);
    let mut result = vec![None, None];

    if active_mask[0] {
        let rhs = inputs[1].fixed_value("div", 1)?;
        let denominator = conjugate_for_input_dtypes(builder, rhs, &metadata_inputs, &[1], ctx);
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
        let lhs = inputs[0].fixed_value("div", 0)?;
        let rhs = inputs[1].fixed_value("div", 1)?;
        let numerator = convert_fixed_ref_to_dtype(builder, lhs, lhs_dtype, output_dtype);
        let denominator = convert_fixed_ref_to_dtype(builder, rhs, rhs_dtype, output_dtype);
        let quotient = emit_fixed_div(builder, numerator, denominator.clone());
        let neg_quotient = emit_fixed_neg(builder, ValueRef::Local(quotient));
        let coeff = emit_fixed_div(builder, ValueRef::Local(neg_quotient), denominator);
        let coeff = conjugate_for_input_dtypes(
            builder,
            ValueRef::Local(coeff),
            &metadata_inputs,
            &[0, 1],
            ctx,
        );
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

pub fn transpose_abs(
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
            let input_dtype = dtype_of_or_real(ctx, &inputs[0]);
            let output_dtype = abs_output_dtype(input_dtype);
            let sign_x = emit_fixed_unary(builder, StdTensorOp::Sign, inputs[0].clone());
            let ct = convert_linear_to_dtype(builder, ct, output_dtype, input_dtype);
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
    ctx: &mut ShapeGuardContext,
) -> Vec<Option<LocalValueId>> {
    let Some(ct) = cotangent_out[0] else {
        return vec![None, None];
    };
    let active = active_mask(mode, 2);
    if active.iter().all(|is_active| !is_active) {
        return vec![None, None];
    }
    let dtype = dtype_of_or_real(ctx, &inputs[0]);
    let output = emit_fixed_binary(
        builder,
        StdTensorOp::Maximum,
        inputs[0].clone(),
        inputs[1].clone(),
    );
    let lhs_eq_output = emit_fixed_compare(
        builder,
        CompareDir::Eq,
        inputs[0].clone(),
        ValueRef::Local(output),
    );
    let rhs_eq_output = emit_fixed_compare(
        builder,
        CompareDir::Eq,
        inputs[1].clone(),
        ValueRef::Local(output),
    );
    let lhs_active = active.first().copied().unwrap_or(false);
    let rhs_active = active.get(1).copied().unwrap_or(false);
    let lhs = lhs_active
        .then(|| balanced_extrema_contribution(builder, ct, lhs_eq_output, rhs_eq_output, dtype));
    let rhs = rhs_active
        .then(|| balanced_extrema_contribution(builder, ct, rhs_eq_output, lhs_eq_output, dtype));
    vec![lhs, rhs]
}

pub fn transpose_minimum(
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[ValueRef<StdTensorOp>],
    mode: &OperationRole,
    ctx: &mut ShapeGuardContext,
) -> Vec<Option<LocalValueId>> {
    let Some(ct) = cotangent_out[0] else {
        return vec![None, None];
    };
    let active = active_mask(mode, 2);
    if active.iter().all(|is_active| !is_active) {
        return vec![None, None];
    }
    let dtype = dtype_of_or_real(ctx, &inputs[0]);
    let output = emit_fixed_binary(
        builder,
        StdTensorOp::Minimum,
        inputs[0].clone(),
        inputs[1].clone(),
    );
    let lhs_eq_output = emit_fixed_compare(
        builder,
        CompareDir::Eq,
        inputs[0].clone(),
        ValueRef::Local(output),
    );
    let rhs_eq_output = emit_fixed_compare(
        builder,
        CompareDir::Eq,
        inputs[1].clone(),
        ValueRef::Local(output),
    );
    let lhs_active = active.first().copied().unwrap_or(false);
    let rhs_active = active.get(1).copied().unwrap_or(false);
    let lhs = lhs_active
        .then(|| balanced_extrema_contribution(builder, ct, lhs_eq_output, rhs_eq_output, dtype));
    let rhs = rhs_active
        .then(|| balanced_extrema_contribution(builder, ct, rhs_eq_output, lhs_eq_output, dtype));
    vec![lhs, rhs]
}

pub fn transpose_select(
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[TransposeInputRef<'_>],
    mode: &OperationRole,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let Some(ct) = cotangent_out[0] else {
        return Ok(vec![None, None, None]);
    };
    let active = active_mask(mode, 3);
    let true_active = active.get(1).copied().unwrap_or(false);
    let false_active = active.get(2).copied().unwrap_or(false);
    if !true_active && !false_active {
        return Ok(vec![None, None, None]);
    }
    let condition = inputs[0].fixed_value("select", 0)?;
    let (on_true, on_false) =
        split_cotangent_by_mask(builder, condition, ct, true_active, false_active);
    Ok(vec![None, on_true, on_false])
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

    let input_gt_lower = emit_fixed_compare(
        builder,
        CompareDir::Gt,
        inputs[0].clone(),
        inputs[1].clone(),
    );
    let input_lt_upper = emit_fixed_compare(
        builder,
        CompareDir::Lt,
        inputs[0].clone(),
        inputs[2].clone(),
    );
    let lower_gt_input = emit_fixed_compare(
        builder,
        CompareDir::Gt,
        inputs[1].clone(),
        inputs[0].clone(),
    );
    let lower_lt_upper = emit_fixed_compare(
        builder,
        CompareDir::Lt,
        inputs[1].clone(),
        inputs[2].clone(),
    );
    let max_input_lower = emit_fixed_binary(
        builder,
        StdTensorOp::Maximum,
        inputs[0].clone(),
        inputs[1].clone(),
    );
    let upper_lt_max_input_lower = emit_fixed_compare(
        builder,
        CompareDir::Lt,
        inputs[2].clone(),
        ValueRef::Local(max_input_lower),
    );

    let input_ct = input_active
        .then(|| mask_active_by_conditions(builder, ct, &[input_gt_lower, input_lt_upper]));
    let lower_ct = lower_active
        .then(|| mask_active_by_conditions(builder, ct, &[lower_gt_input, lower_lt_upper]));
    let upper_ct =
        upper_active.then(|| mask_active_by_conditions(builder, ct, &[upper_lt_max_input_lower]));

    vec![input_ct, lower_ct, upper_ct]
}
