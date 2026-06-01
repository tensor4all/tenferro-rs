use computegraph::types::{GlobalValKey, LocalValId, OpMode, ValRef};
use computegraph::OpEmitter;

use crate::std_tensor_op::StdTensorOp;

fn emit_fixed_unary(
    emitter: &mut dyn OpEmitter<StdTensorOp>,
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
    emitter: &mut dyn OpEmitter<StdTensorOp>,
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
    emitter: &mut dyn OpEmitter<StdTensorOp>,
    input: ValRef<StdTensorOp>,
) -> LocalValId {
    emit_fixed_unary(emitter, StdTensorOp::Neg, input)
}

fn emit_fixed_add(
    emitter: &mut dyn OpEmitter<StdTensorOp>,
    lhs: ValRef<StdTensorOp>,
    rhs: ValRef<StdTensorOp>,
) -> LocalValId {
    emit_fixed_binary(emitter, StdTensorOp::Add, lhs, rhs)
}

fn emit_fixed_mul(
    emitter: &mut dyn OpEmitter<StdTensorOp>,
    lhs: ValRef<StdTensorOp>,
    rhs: ValRef<StdTensorOp>,
) -> LocalValId {
    emit_fixed_binary(emitter, StdTensorOp::Mul, lhs, rhs)
}

fn emit_fixed_div(
    emitter: &mut dyn OpEmitter<StdTensorOp>,
    lhs: ValRef<StdTensorOp>,
    rhs: ValRef<StdTensorOp>,
) -> LocalValId {
    emit_fixed_binary(emitter, StdTensorOp::Div, lhs, rhs)
}

fn emit_one_like_fixed(
    emitter: &mut dyn OpEmitter<StdTensorOp>,
    anchor: ValRef<StdTensorOp>,
) -> LocalValId {
    let neg = emit_fixed_neg(emitter, anchor.clone());
    let zero = emit_fixed_add(emitter, anchor, ValRef::Local(neg));
    emit_fixed_unary(emitter, StdTensorOp::Exp, ValRef::Local(zero))
}

fn emit_linear_mul_fixed(
    emitter: &mut dyn OpEmitter<StdTensorOp>,
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

fn emit_linear_div_fixed_denominator(
    emitter: &mut dyn OpEmitter<StdTensorOp>,
    active: LocalValId,
    denominator: ValRef<StdTensorOp>,
) -> LocalValId {
    emitter.add_op(
        StdTensorOp::Div,
        vec![ValRef::Local(active), denominator],
        OpMode::Linear {
            active_mask: vec![true, false],
        },
    )[0]
}

fn unary_is_active(mode: &OpMode) -> bool {
    match mode {
        OpMode::Linear { active_mask } => active_mask[0],
        OpMode::Primal => false,
    }
}

pub fn linearize_exp(
    builder: &mut dyn OpEmitter<StdTensorOp>,
    primal_out: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
) -> Vec<Option<LocalValId>> {
    match tangent_in[0] {
        Some(dx) => {
            let out = builder.add_op(
                StdTensorOp::Mul,
                vec![ValRef::External(primal_out[0].clone()), ValRef::Local(dx)],
                OpMode::Linear {
                    active_mask: vec![false, true],
                },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}

pub fn linearize_log(
    builder: &mut dyn OpEmitter<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
) -> Vec<Option<LocalValId>> {
    match tangent_in[0] {
        Some(dx) => {
            let out = builder.add_op(
                StdTensorOp::Div,
                vec![ValRef::Local(dx), ValRef::External(primal_in[0].clone())],
                OpMode::Linear {
                    active_mask: vec![true, false],
                },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}

pub fn linearize_sin(
    builder: &mut dyn OpEmitter<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
) -> Vec<Option<LocalValId>> {
    match tangent_in[0] {
        Some(dx) => {
            let cos_x = emit_fixed_unary(
                builder,
                StdTensorOp::Cos,
                ValRef::External(primal_in[0].clone()),
            );
            vec![Some(emit_linear_mul_fixed(
                builder,
                ValRef::Local(cos_x),
                dx,
            ))]
        }
        None => vec![None],
    }
}

pub fn linearize_cos(
    builder: &mut dyn OpEmitter<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
) -> Vec<Option<LocalValId>> {
    match tangent_in[0] {
        Some(dx) => {
            let sin_x = emit_fixed_unary(
                builder,
                StdTensorOp::Sin,
                ValRef::External(primal_in[0].clone()),
            );
            let neg_sin_x = emit_fixed_neg(builder, ValRef::Local(sin_x));
            vec![Some(emit_linear_mul_fixed(
                builder,
                ValRef::Local(neg_sin_x),
                dx,
            ))]
        }
        None => vec![None],
    }
}

pub fn linearize_tanh(
    builder: &mut dyn OpEmitter<StdTensorOp>,
    primal_out: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
) -> Vec<Option<LocalValId>> {
    match tangent_in[0] {
        Some(dx) => {
            let y = ValRef::External(primal_out[0].clone());
            let y_sq = emit_fixed_mul(builder, y.clone(), y.clone());
            let one = emit_one_like_fixed(builder, y);
            let neg_y_sq = emit_fixed_neg(builder, ValRef::Local(y_sq));
            let coeff = emit_fixed_add(builder, ValRef::Local(one), ValRef::Local(neg_y_sq));
            vec![Some(emit_linear_mul_fixed(
                builder,
                ValRef::Local(coeff),
                dx,
            ))]
        }
        None => vec![None],
    }
}

pub fn linearize_sqrt(
    builder: &mut dyn OpEmitter<StdTensorOp>,
    primal_out: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
) -> Vec<Option<LocalValId>> {
    match tangent_in[0] {
        Some(dx) => {
            let y = ValRef::External(primal_out[0].clone());
            let two_y = emit_fixed_add(builder, y.clone(), y);
            vec![Some(emit_linear_div_fixed_denominator(
                builder,
                dx,
                ValRef::Local(two_y),
            ))]
        }
        None => vec![None],
    }
}

pub fn linearize_rsqrt(
    builder: &mut dyn OpEmitter<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    primal_out: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
) -> Vec<Option<LocalValId>> {
    match tangent_in[0] {
        Some(dx) => {
            let neg_rsqrt_x = emit_fixed_neg(builder, ValRef::External(primal_out[0].clone()));
            let x = ValRef::External(primal_in[0].clone());
            let two_x = emit_fixed_add(builder, x.clone(), x);
            let coeff = emit_fixed_div(builder, ValRef::Local(neg_rsqrt_x), ValRef::Local(two_x));
            vec![Some(emit_linear_mul_fixed(
                builder,
                ValRef::Local(coeff),
                dx,
            ))]
        }
        None => vec![None],
    }
}

pub fn linearize_pow(
    builder: &mut dyn OpEmitter<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    primal_out: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
) -> Vec<Option<LocalValId>> {
    let mut terms = Vec::with_capacity(2);

    if let Some(dx) = tangent_in[0] {
        let pow_over_x = emit_fixed_div(
            builder,
            ValRef::External(primal_out[0].clone()),
            ValRef::External(primal_in[0].clone()),
        );
        let coeff = emit_fixed_mul(
            builder,
            ValRef::External(primal_in[1].clone()),
            ValRef::Local(pow_over_x),
        );
        terms.push(emit_linear_mul_fixed(builder, ValRef::Local(coeff), dx));
    }

    if let Some(dy) = tangent_in[1] {
        let log_x = emit_fixed_unary(
            builder,
            StdTensorOp::Log,
            ValRef::External(primal_in[0].clone()),
        );
        let coeff = emit_fixed_mul(
            builder,
            ValRef::Local(log_x),
            ValRef::External(primal_out[0].clone()),
        );
        terms.push(emit_linear_mul_fixed(builder, ValRef::Local(coeff), dy));
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
        _ => unreachable!("pow linearization creates at most two terms"),
    }
}

pub fn linearize_expm1(
    builder: &mut dyn OpEmitter<StdTensorOp>,
    primal_out: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
) -> Vec<Option<LocalValId>> {
    match tangent_in[0] {
        Some(dx) => {
            let y = ValRef::External(primal_out[0].clone());
            let one = emit_one_like_fixed(builder, y.clone());
            let coeff = emit_fixed_add(builder, y, ValRef::Local(one));
            vec![Some(emit_linear_mul_fixed(
                builder,
                ValRef::Local(coeff),
                dx,
            ))]
        }
        None => vec![None],
    }
}

pub fn linearize_log1p(
    builder: &mut dyn OpEmitter<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
) -> Vec<Option<LocalValId>> {
    match tangent_in[0] {
        Some(dx) => {
            let x = ValRef::External(primal_in[0].clone());
            let one = emit_one_like_fixed(builder, x.clone());
            let denom = emit_fixed_add(builder, x, ValRef::Local(one));
            vec![Some(emit_linear_div_fixed_denominator(
                builder,
                dx,
                ValRef::Local(denom),
            ))]
        }
        None => vec![None],
    }
}

pub fn transpose_exp(
    emitter: &mut dyn OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    inputs: &[ValRef<StdTensorOp>],
    mode: &OpMode,
) -> Vec<Option<LocalValId>> {
    if !unary_is_active(mode) {
        return vec![None];
    }
    match cotangent_out[0] {
        Some(ct) => {
            let exp_x = emit_fixed_unary(emitter, StdTensorOp::Exp, inputs[0].clone());
            vec![Some(emit_linear_mul_fixed(
                emitter,
                ValRef::Local(exp_x),
                ct,
            ))]
        }
        None => vec![None],
    }
}

pub fn transpose_log(
    emitter: &mut dyn OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    inputs: &[ValRef<StdTensorOp>],
    mode: &OpMode,
) -> Vec<Option<LocalValId>> {
    if !unary_is_active(mode) {
        return vec![None];
    }
    match cotangent_out[0] {
        Some(ct) => {
            let out = emitter.add_op(
                StdTensorOp::Div,
                vec![ValRef::Local(ct), inputs[0].clone()],
                OpMode::Linear {
                    active_mask: vec![true, false],
                },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}

pub fn transpose_sin(
    emitter: &mut dyn OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    inputs: &[ValRef<StdTensorOp>],
    mode: &OpMode,
) -> Vec<Option<LocalValId>> {
    if !unary_is_active(mode) {
        return vec![None];
    }
    match cotangent_out[0] {
        Some(ct) => {
            let cos_x = emit_fixed_unary(emitter, StdTensorOp::Cos, inputs[0].clone());
            vec![Some(emit_linear_mul_fixed(
                emitter,
                ValRef::Local(cos_x),
                ct,
            ))]
        }
        None => vec![None],
    }
}

pub fn transpose_cos(
    emitter: &mut dyn OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    inputs: &[ValRef<StdTensorOp>],
    mode: &OpMode,
) -> Vec<Option<LocalValId>> {
    if !unary_is_active(mode) {
        return vec![None];
    }
    match cotangent_out[0] {
        Some(ct) => {
            let sin_x = emit_fixed_unary(emitter, StdTensorOp::Sin, inputs[0].clone());
            let neg_sin_x = emit_fixed_neg(emitter, ValRef::Local(sin_x));
            vec![Some(emit_linear_mul_fixed(
                emitter,
                ValRef::Local(neg_sin_x),
                ct,
            ))]
        }
        None => vec![None],
    }
}

pub fn transpose_tanh(
    emitter: &mut dyn OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    inputs: &[ValRef<StdTensorOp>],
    mode: &OpMode,
) -> Vec<Option<LocalValId>> {
    if !unary_is_active(mode) {
        return vec![None];
    }
    match cotangent_out[0] {
        Some(ct) => {
            let tanh_x = emit_fixed_unary(emitter, StdTensorOp::Tanh, inputs[0].clone());
            let tanh_sq = emit_fixed_mul(emitter, ValRef::Local(tanh_x), ValRef::Local(tanh_x));
            let one = emit_one_like_fixed(emitter, inputs[0].clone());
            let neg_tanh_sq = emit_fixed_neg(emitter, ValRef::Local(tanh_sq));
            let coeff = emit_fixed_add(emitter, ValRef::Local(one), ValRef::Local(neg_tanh_sq));
            vec![Some(emit_linear_mul_fixed(
                emitter,
                ValRef::Local(coeff),
                ct,
            ))]
        }
        None => vec![None],
    }
}

pub fn transpose_sqrt(
    emitter: &mut dyn OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    inputs: &[ValRef<StdTensorOp>],
    mode: &OpMode,
) -> Vec<Option<LocalValId>> {
    if !unary_is_active(mode) {
        return vec![None];
    }
    match cotangent_out[0] {
        Some(ct) => {
            let sqrt_x = emit_fixed_unary(emitter, StdTensorOp::Sqrt, inputs[0].clone());
            let two_sqrt_x = emit_fixed_add(emitter, ValRef::Local(sqrt_x), ValRef::Local(sqrt_x));
            let out = emitter.add_op(
                StdTensorOp::Div,
                vec![ValRef::Local(ct), ValRef::Local(two_sqrt_x)],
                OpMode::Linear {
                    active_mask: vec![true, false],
                },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}

pub fn transpose_rsqrt(
    emitter: &mut dyn OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    inputs: &[ValRef<StdTensorOp>],
    mode: &OpMode,
) -> Vec<Option<LocalValId>> {
    if !unary_is_active(mode) {
        return vec![None];
    }
    match cotangent_out[0] {
        Some(ct) => {
            let rsqrt_x = emit_fixed_unary(emitter, StdTensorOp::Rsqrt, inputs[0].clone());
            let neg_rsqrt_x = emit_fixed_neg(emitter, ValRef::Local(rsqrt_x));
            let two_x = emit_fixed_add(emitter, inputs[0].clone(), inputs[0].clone());
            let coeff = emit_fixed_div(emitter, ValRef::Local(neg_rsqrt_x), ValRef::Local(two_x));
            vec![Some(emit_linear_mul_fixed(
                emitter,
                ValRef::Local(coeff),
                ct,
            ))]
        }
        None => vec![None],
    }
}

pub fn transpose_pow(
    emitter: &mut dyn OpEmitter<StdTensorOp>,
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
        let pow_xy = emit_fixed_binary(
            emitter,
            StdTensorOp::Pow,
            inputs[0].clone(),
            inputs[1].clone(),
        );
        let pow_over_x = emit_fixed_div(emitter, ValRef::Local(pow_xy), inputs[0].clone());
        let coeff = emit_fixed_mul(emitter, inputs[1].clone(), ValRef::Local(pow_over_x));
        result[0] = Some(emit_linear_mul_fixed(emitter, ValRef::Local(coeff), ct));
    }

    if active_mask[1] {
        let log_x = emit_fixed_unary(emitter, StdTensorOp::Log, inputs[0].clone());
        let pow_xy = emit_fixed_binary(
            emitter,
            StdTensorOp::Pow,
            inputs[0].clone(),
            inputs[1].clone(),
        );
        let coeff = emit_fixed_mul(emitter, ValRef::Local(log_x), ValRef::Local(pow_xy));
        result[1] = Some(emit_linear_mul_fixed(emitter, ValRef::Local(coeff), ct));
    }

    result
}

pub fn transpose_expm1(
    emitter: &mut dyn OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    inputs: &[ValRef<StdTensorOp>],
    mode: &OpMode,
) -> Vec<Option<LocalValId>> {
    if !unary_is_active(mode) {
        return vec![None];
    }
    match cotangent_out[0] {
        Some(ct) => {
            let exp_x = emit_fixed_unary(emitter, StdTensorOp::Exp, inputs[0].clone());
            vec![Some(emit_linear_mul_fixed(
                emitter,
                ValRef::Local(exp_x),
                ct,
            ))]
        }
        None => vec![None],
    }
}

pub fn transpose_log1p(
    emitter: &mut dyn OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    inputs: &[ValRef<StdTensorOp>],
    mode: &OpMode,
) -> Vec<Option<LocalValId>> {
    if !unary_is_active(mode) {
        return vec![None];
    }
    match cotangent_out[0] {
        Some(ct) => {
            let one = emit_one_like_fixed(emitter, inputs[0].clone());
            let denom = emit_fixed_add(emitter, inputs[0].clone(), ValRef::Local(one));
            let out = emitter.add_op(
                StdTensorOp::Div,
                vec![ValRef::Local(ct), ValRef::Local(denom)],
                OpMode::Linear {
                    active_mask: vec![true, false],
                },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}
