use computegraph::fragment::FragmentBuilder;
use computegraph::types::{GlobalValKey, LocalValId, OpMode, ValRef};
use computegraph::OpEmitter;
use tenferro_tensor::CompareDir;

use crate::std_tensor_op::StdTensorOp;

fn emit_fixed_unary(
    emitter: &mut impl OpEmitter<StdTensorOp>,
    op: StdTensorOp,
    input: ValRef<StdTensorOp>,
) -> LocalValId {
    emitter.add_op(
        op,
        vec![input],
        OpMode::Linear {
            active_mask: vec![false],
        },
    )[0]
}

fn emit_fixed_binary(
    emitter: &mut impl OpEmitter<StdTensorOp>,
    op: StdTensorOp,
    lhs: ValRef<StdTensorOp>,
    rhs: ValRef<StdTensorOp>,
) -> LocalValId {
    emitter.add_op(
        op,
        vec![lhs, rhs],
        OpMode::Linear {
            active_mask: vec![false, false],
        },
    )[0]
}

fn emit_fixed_neg(
    emitter: &mut impl OpEmitter<StdTensorOp>,
    input: ValRef<StdTensorOp>,
) -> LocalValId {
    emit_fixed_unary(emitter, StdTensorOp::Neg, input)
}

fn emit_fixed_div(
    emitter: &mut impl OpEmitter<StdTensorOp>,
    lhs: ValRef<StdTensorOp>,
    rhs: ValRef<StdTensorOp>,
) -> LocalValId {
    emit_fixed_binary(emitter, StdTensorOp::Div, lhs, rhs)
}

fn emit_fixed_compare(
    emitter: &mut impl OpEmitter<StdTensorOp>,
    dir: CompareDir,
    lhs: ValRef<StdTensorOp>,
    rhs: ValRef<StdTensorOp>,
) -> LocalValId {
    emit_fixed_binary(emitter, StdTensorOp::Compare(dir), lhs, rhs)
}

fn emit_linear_mul_fixed(
    emitter: &mut impl OpEmitter<StdTensorOp>,
    fixed: ValRef<StdTensorOp>,
    active: LocalValId,
) -> LocalValId {
    emitter.add_op(
        StdTensorOp::Mul,
        vec![fixed, ValRef::Local(active)],
        OpMode::Linear {
            active_mask: vec![false, true],
        },
    )[0]
}

fn emit_linear_select(
    emitter: &mut impl OpEmitter<StdTensorOp>,
    condition: ValRef<StdTensorOp>,
    on_true: LocalValId,
    on_false: LocalValId,
) -> LocalValId {
    emitter.add_op(
        StdTensorOp::Select,
        vec![condition, ValRef::Local(on_true), ValRef::Local(on_false)],
        OpMode::Linear {
            active_mask: vec![false, true, true],
        },
    )[0]
}

fn emit_zero_from_active(
    emitter: &mut impl OpEmitter<StdTensorOp>,
    active: LocalValId,
) -> LocalValId {
    let neg = emitter.add_op(
        StdTensorOp::Neg,
        vec![ValRef::Local(active)],
        OpMode::Linear {
            active_mask: vec![true],
        },
    );
    emitter.add_op(
        StdTensorOp::Add,
        vec![ValRef::Local(active), ValRef::Local(neg[0])],
        OpMode::Linear {
            active_mask: vec![true, true],
        },
    )[0]
}

fn select_tangents(
    emitter: &mut impl OpEmitter<StdTensorOp>,
    condition: ValRef<StdTensorOp>,
    on_true: Option<LocalValId>,
    on_false: Option<LocalValId>,
) -> Option<LocalValId> {
    match (on_true, on_false) {
        (Some(t), Some(f)) => Some(emit_linear_select(emitter, condition, t, f)),
        (Some(t), None) => {
            let zero = emit_zero_from_active(emitter, t);
            Some(emit_linear_select(emitter, condition, t, zero))
        }
        (None, Some(f)) => {
            let zero = emit_zero_from_active(emitter, f);
            Some(emit_linear_select(emitter, condition, zero, f))
        }
        (None, None) => None,
    }
}

fn split_cotangent_by_mask(
    emitter: &mut impl OpEmitter<StdTensorOp>,
    condition: ValRef<StdTensorOp>,
    cotangent: LocalValId,
    true_active: bool,
    false_active: bool,
) -> (Option<LocalValId>, Option<LocalValId>) {
    if !true_active && !false_active {
        return (None, None);
    }

    let zero = emit_zero_from_active(emitter, cotangent);
    let true_ct =
        true_active.then(|| emit_linear_select(emitter, condition.clone(), cotangent, zero));
    let false_ct = false_active.then(|| emit_linear_select(emitter, condition, zero, cotangent));
    (true_ct, false_ct)
}

fn unary_is_active(mode: &OpMode) -> bool {
    match mode {
        OpMode::Linear { active_mask } => active_mask[0],
        OpMode::Primal => false,
    }
}

fn active_mask(mode: &OpMode, len: usize) -> Vec<bool> {
    match mode {
        OpMode::Linear { active_mask } => active_mask.clone(),
        OpMode::Primal => vec![false; len],
    }
}

pub fn linearize_div(
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    primal_out: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
) -> Vec<Option<LocalValId>> {
    let mut terms = Vec::with_capacity(2);

    if let Some(dx) = tangent_in[0] {
        let out = builder.add_op(
            StdTensorOp::Div,
            vec![ValRef::Local(dx), ValRef::External(primal_in[1].clone())],
            OpMode::Linear {
                active_mask: vec![true, false],
            },
        );
        terms.push(out[0]);
    }

    if let Some(dy) = tangent_in[1] {
        let quotient_over_rhs = emit_fixed_div(
            builder,
            ValRef::External(primal_out[0].clone()),
            ValRef::External(primal_in[1].clone()),
        );
        let neg_coeff = emit_fixed_neg(builder, ValRef::Local(quotient_over_rhs));
        terms.push(emit_linear_mul_fixed(builder, ValRef::Local(neg_coeff), dy));
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
        _ => unreachable!("div linearization creates at most two terms"),
    }
}

pub fn linearize_abs(
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
) -> Vec<Option<LocalValId>> {
    match tangent_in[0] {
        Some(dx) => {
            let sign_x = emit_fixed_unary(
                builder,
                StdTensorOp::Sign,
                ValRef::External(primal_in[0].clone()),
            );
            vec![Some(emit_linear_mul_fixed(
                builder,
                ValRef::Local(sign_x),
                dx,
            ))]
        }
        None => vec![None],
    }
}

pub fn linearize_sign(
    builder: &mut FragmentBuilder<StdTensorOp>,
    tangent_in: &[Option<LocalValId>],
) -> Vec<Option<LocalValId>> {
    match tangent_in[0] {
        Some(dx) => vec![Some(emit_zero_from_active(builder, dx))],
        None => vec![None],
    }
}

pub fn linearize_maximum(
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
) -> Vec<Option<LocalValId>> {
    if tangent_in[0].is_none() && tangent_in[1].is_none() {
        return vec![None];
    }

    let mask = emit_fixed_compare(
        builder,
        CompareDir::Ge,
        ValRef::External(primal_in[0].clone()),
        ValRef::External(primal_in[1].clone()),
    );
    vec![select_tangents(
        builder,
        ValRef::Local(mask),
        tangent_in[0],
        tangent_in[1],
    )]
}

pub fn linearize_minimum(
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
) -> Vec<Option<LocalValId>> {
    if tangent_in[0].is_none() && tangent_in[1].is_none() {
        return vec![None];
    }

    let mask = emit_fixed_compare(
        builder,
        CompareDir::Le,
        ValRef::External(primal_in[0].clone()),
        ValRef::External(primal_in[1].clone()),
    );
    vec![select_tangents(
        builder,
        ValRef::Local(mask),
        tangent_in[0],
        tangent_in[1],
    )]
}

pub fn linearize_select(
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
) -> Vec<Option<LocalValId>> {
    vec![select_tangents(
        builder,
        ValRef::External(primal_in[0].clone()),
        tangent_in[1],
        tangent_in[2],
    )]
}

pub fn linearize_clamp(
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
) -> Vec<Option<LocalValId>> {
    if tangent_in.iter().all(Option::is_none) {
        return vec![None];
    }

    let upper_mask = emit_fixed_compare(
        builder,
        CompareDir::Le,
        ValRef::External(primal_in[0].clone()),
        ValRef::External(primal_in[2].clone()),
    );
    let inner_tangent = select_tangents(
        builder,
        ValRef::Local(upper_mask),
        tangent_in[0],
        tangent_in[2],
    );

    let inner_primal = emit_fixed_binary(
        builder,
        StdTensorOp::Minimum,
        ValRef::External(primal_in[0].clone()),
        ValRef::External(primal_in[2].clone()),
    );
    let lower_mask = emit_fixed_compare(
        builder,
        CompareDir::Ge,
        ValRef::External(primal_in[1].clone()),
        ValRef::Local(inner_primal),
    );

    vec![select_tangents(
        builder,
        ValRef::Local(lower_mask),
        tangent_in[1],
        inner_tangent,
    )]
}

pub fn transpose_div(
    emitter: &mut impl OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    inputs: &[ValRef<StdTensorOp>],
    mode: &OpMode,
) -> Vec<Option<LocalValId>> {
    let ct = match cotangent_out[0] {
        Some(ct) => ct,
        None => return vec![None, None],
    };

    let active_mask = match mode {
        OpMode::Linear { active_mask } => active_mask,
        OpMode::Primal => return vec![None, None],
    };

    let mut result = vec![None, None];

    if active_mask[0] {
        let out = emitter.add_op(
            StdTensorOp::Div,
            vec![ValRef::Local(ct), inputs[1].clone()],
            OpMode::Linear {
                active_mask: vec![true, false],
            },
        );
        result[0] = Some(out[0]);
    }

    if active_mask[1] {
        let quotient = emit_fixed_div(emitter, inputs[0].clone(), inputs[1].clone());
        let neg_quotient = emit_fixed_neg(emitter, ValRef::Local(quotient));
        let coeff = emit_fixed_div(emitter, ValRef::Local(neg_quotient), inputs[1].clone());
        result[1] = Some(emit_linear_mul_fixed(emitter, ValRef::Local(coeff), ct));
    }

    result
}

pub fn transpose_abs(
    emitter: &mut impl OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    inputs: &[ValRef<StdTensorOp>],
    mode: &OpMode,
) -> Vec<Option<LocalValId>> {
    if !unary_is_active(mode) {
        return vec![None];
    }
    match cotangent_out[0] {
        Some(ct) => {
            let sign_x = emit_fixed_unary(emitter, StdTensorOp::Sign, inputs[0].clone());
            vec![Some(emit_linear_mul_fixed(
                emitter,
                ValRef::Local(sign_x),
                ct,
            ))]
        }
        None => vec![None],
    }
}

pub fn transpose_sign(
    emitter: &mut impl OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    mode: &OpMode,
) -> Vec<Option<LocalValId>> {
    if !unary_is_active(mode) {
        return vec![None];
    }
    match cotangent_out[0] {
        Some(ct) => vec![Some(emit_zero_from_active(emitter, ct))],
        None => vec![None],
    }
}

pub fn transpose_maximum(
    emitter: &mut impl OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    inputs: &[ValRef<StdTensorOp>],
    mode: &OpMode,
) -> Vec<Option<LocalValId>> {
    let Some(ct) = cotangent_out[0] else {
        return vec![None, None];
    };
    let active = active_mask(mode, 2);
    if active.iter().all(|is_active| !is_active) {
        return vec![None, None];
    }
    let mask = emit_fixed_compare(
        emitter,
        CompareDir::Ge,
        inputs[0].clone(),
        inputs[1].clone(),
    );
    let lhs_active = active.first().copied().unwrap_or(false);
    let rhs_active = active.get(1).copied().unwrap_or(false);
    let (lhs, rhs) =
        split_cotangent_by_mask(emitter, ValRef::Local(mask), ct, lhs_active, rhs_active);
    vec![lhs, rhs]
}

pub fn transpose_minimum(
    emitter: &mut impl OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    inputs: &[ValRef<StdTensorOp>],
    mode: &OpMode,
) -> Vec<Option<LocalValId>> {
    let Some(ct) = cotangent_out[0] else {
        return vec![None, None];
    };
    let active = active_mask(mode, 2);
    if active.iter().all(|is_active| !is_active) {
        return vec![None, None];
    }
    let mask = emit_fixed_compare(
        emitter,
        CompareDir::Le,
        inputs[0].clone(),
        inputs[1].clone(),
    );
    let lhs_active = active.first().copied().unwrap_or(false);
    let rhs_active = active.get(1).copied().unwrap_or(false);
    let (lhs, rhs) =
        split_cotangent_by_mask(emitter, ValRef::Local(mask), ct, lhs_active, rhs_active);
    vec![lhs, rhs]
}

pub fn transpose_select(
    emitter: &mut impl OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    inputs: &[ValRef<StdTensorOp>],
    mode: &OpMode,
) -> Vec<Option<LocalValId>> {
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
        split_cotangent_by_mask(emitter, inputs[0].clone(), ct, true_active, false_active);
    vec![None, on_true, on_false]
}

pub fn transpose_clamp(
    emitter: &mut impl OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    inputs: &[ValRef<StdTensorOp>],
    mode: &OpMode,
) -> Vec<Option<LocalValId>> {
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
        emitter,
        StdTensorOp::Minimum,
        inputs[0].clone(),
        inputs[2].clone(),
    );
    let lower_mask = emit_fixed_compare(
        emitter,
        CompareDir::Ge,
        inputs[1].clone(),
        ValRef::Local(inner_primal),
    );
    let (lower_ct, inner_ct) = split_cotangent_by_mask(
        emitter,
        ValRef::Local(lower_mask),
        ct,
        lower_active,
        input_active || upper_active,
    );

    let (input_ct, upper_ct) = match inner_ct {
        Some(inner_ct) => {
            let upper_mask = emit_fixed_compare(
                emitter,
                CompareDir::Le,
                inputs[0].clone(),
                inputs[2].clone(),
            );
            split_cotangent_by_mask(
                emitter,
                ValRef::Local(upper_mask),
                inner_ct,
                input_active,
                upper_active,
            )
        }
        None => (None, None),
    };

    vec![input_ct, lower_ct, upper_ct]
}
