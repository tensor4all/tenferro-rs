use computegraph::fragment::FragmentBuilder;
use computegraph::types::{GlobalValKey, LocalValId, OpMode, ValRef};

use crate::std_tensor_op::StdTensorOp;

fn emit_fixed_unary(
    builder: &mut FragmentBuilder<StdTensorOp>,
    op: StdTensorOp,
    input: ValRef<StdTensorOp>,
) -> LocalValId {
    builder.add_op(
        op,
        vec![input],
        OpMode::Linear {
            active_mask: vec![false],
        },
    )[0]
}

fn emit_fixed_binary(
    builder: &mut FragmentBuilder<StdTensorOp>,
    op: StdTensorOp,
    lhs: ValRef<StdTensorOp>,
    rhs: ValRef<StdTensorOp>,
) -> LocalValId {
    builder.add_op(
        op,
        vec![lhs, rhs],
        OpMode::Linear {
            active_mask: vec![false, false],
        },
    )[0]
}

fn emit_fixed_neg(
    builder: &mut FragmentBuilder<StdTensorOp>,
    input: ValRef<StdTensorOp>,
) -> LocalValId {
    emit_fixed_unary(builder, StdTensorOp::Neg, input)
}

fn emit_fixed_div(
    builder: &mut FragmentBuilder<StdTensorOp>,
    lhs: ValRef<StdTensorOp>,
    rhs: ValRef<StdTensorOp>,
) -> LocalValId {
    emit_fixed_binary(builder, StdTensorOp::Div, lhs, rhs)
}

fn emit_linear_mul_fixed(
    builder: &mut FragmentBuilder<StdTensorOp>,
    fixed: ValRef<StdTensorOp>,
    active: LocalValId,
) -> LocalValId {
    builder.add_op(
        StdTensorOp::Mul,
        vec![fixed, ValRef::Local(active)],
        OpMode::Linear {
            active_mask: vec![false, true],
        },
    )[0]
}

fn emit_zero_from_active(
    builder: &mut FragmentBuilder<StdTensorOp>,
    active: LocalValId,
) -> LocalValId {
    let neg = builder.add_op(
        StdTensorOp::Neg,
        vec![ValRef::Local(active)],
        OpMode::Linear {
            active_mask: vec![true],
        },
    );
    builder.add_op(
        StdTensorOp::Add,
        vec![ValRef::Local(active), ValRef::Local(neg[0])],
        OpMode::Linear {
            active_mask: vec![true, true],
        },
    )[0]
}

fn unary_is_active(mode: &OpMode) -> bool {
    match mode {
        OpMode::Linear { active_mask } => active_mask[0],
        OpMode::Primal => false,
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

pub fn linearize_scale(
    builder: &mut FragmentBuilder<StdTensorOp>,
    tangent_in: &[Option<LocalValId>],
    factor: f64,
) -> Vec<Option<LocalValId>> {
    match tangent_in[0] {
        Some(dx) => {
            let out = builder.add_op(
                StdTensorOp::Scale { factor },
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

pub fn linearize_scale_complex(
    builder: &mut FragmentBuilder<StdTensorOp>,
    tangent_in: &[Option<LocalValId>],
    re: f64,
    im: f64,
) -> Vec<Option<LocalValId>> {
    match tangent_in[0] {
        Some(dx) => {
            let out = builder.add_op(
                StdTensorOp::ScaleComplex { re, im },
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

pub fn transpose_div(
    builder: &mut FragmentBuilder<StdTensorOp>,
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
        let out = builder.add_op(
            StdTensorOp::Div,
            vec![ValRef::Local(ct), inputs[1].clone()],
            OpMode::Linear {
                active_mask: vec![true, false],
            },
        );
        result[0] = Some(out[0]);
    }

    if active_mask[1] {
        let quotient = emit_fixed_div(builder, inputs[0].clone(), inputs[1].clone());
        let neg_quotient = emit_fixed_neg(builder, ValRef::Local(quotient));
        let coeff = emit_fixed_div(builder, ValRef::Local(neg_quotient), inputs[1].clone());
        result[1] = Some(emit_linear_mul_fixed(builder, ValRef::Local(coeff), ct));
    }

    result
}

pub fn transpose_abs(
    builder: &mut FragmentBuilder<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    inputs: &[ValRef<StdTensorOp>],
    mode: &OpMode,
) -> Vec<Option<LocalValId>> {
    if !unary_is_active(mode) {
        return vec![None];
    }
    match cotangent_out[0] {
        Some(ct) => {
            let sign_x = emit_fixed_unary(builder, StdTensorOp::Sign, inputs[0].clone());
            vec![Some(emit_linear_mul_fixed(
                builder,
                ValRef::Local(sign_x),
                ct,
            ))]
        }
        None => vec![None],
    }
}

pub fn transpose_sign(
    builder: &mut FragmentBuilder<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    mode: &OpMode,
) -> Vec<Option<LocalValId>> {
    if !unary_is_active(mode) {
        return vec![None];
    }
    match cotangent_out[0] {
        Some(ct) => vec![Some(emit_zero_from_active(builder, ct))],
        None => vec![None],
    }
}

pub fn transpose_scale(
    builder: &mut FragmentBuilder<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    mode: &OpMode,
    factor: f64,
) -> Vec<Option<LocalValId>> {
    if !unary_is_active(mode) {
        return vec![None];
    }
    match cotangent_out[0] {
        Some(ct) => {
            let out = builder.add_op(
                StdTensorOp::Scale { factor },
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

pub fn transpose_scale_complex(
    builder: &mut FragmentBuilder<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    mode: &OpMode,
    re: f64,
    im: f64,
) -> Vec<Option<LocalValId>> {
    if !unary_is_active(mode) {
        return vec![None];
    }
    match cotangent_out[0] {
        Some(ct) => {
            let out = builder.add_op(
                StdTensorOp::ScaleComplex { re, im: -im },
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
