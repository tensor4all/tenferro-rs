use computegraph::fragment::FragmentBuilder;
use computegraph::types::{GlobalValKey, LocalValId, OpMode, ValRef};

use crate::std_tensor_op::StdTensorOp;

pub fn linearize_add(
    builder: &mut FragmentBuilder<StdTensorOp>,
    tangent_in: &[Option<LocalValId>],
) -> Vec<Option<LocalValId>> {
    match (tangent_in[0], tangent_in[1]) {
        (Some(dx), Some(dy)) => {
            let out = builder.add_op(
                StdTensorOp::Add,
                vec![ValRef::Local(dx), ValRef::Local(dy)],
                OpMode::Linear {
                    active_mask: vec![true, true],
                },
            );
            vec![Some(out[0])]
        }
        (Some(dx), None) => vec![Some(dx)],
        (None, Some(dy)) => vec![Some(dy)],
        (None, None) => vec![None],
    }
}

pub fn linearize_mul(
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
) -> Vec<Option<LocalValId>> {
    let mut terms = Vec::with_capacity(2);

    if let Some(dx) = tangent_in[0] {
        let term = builder.add_op(
            StdTensorOp::Mul,
            vec![ValRef::Local(dx), ValRef::External(primal_in[1].clone())],
            OpMode::Linear {
                active_mask: vec![true, false],
            },
        );
        terms.push(term[0]);
    }

    if let Some(dy) = tangent_in[1] {
        let term = builder.add_op(
            StdTensorOp::Mul,
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
        _ => unreachable!("mul linearization creates at most two terms"),
    }
}

pub fn linearize_neg(
    builder: &mut FragmentBuilder<StdTensorOp>,
    tangent_in: &[Option<LocalValId>],
) -> Vec<Option<LocalValId>> {
    match tangent_in[0] {
        Some(dx) => {
            let out = builder.add_op(
                StdTensorOp::Neg,
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

pub fn linearize_conj(
    builder: &mut FragmentBuilder<StdTensorOp>,
    tangent_in: &[Option<LocalValId>],
) -> Vec<Option<LocalValId>> {
    match tangent_in[0] {
        Some(dx) => {
            let out = builder.add_op(
                StdTensorOp::Conj,
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

pub fn transpose_add(cotangent_out: &[Option<LocalValId>]) -> Vec<Option<LocalValId>> {
    match cotangent_out[0] {
        Some(ct) => vec![Some(ct), Some(ct)],
        None => vec![None, None],
    }
}

pub fn transpose_mul(
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
            StdTensorOp::Mul,
            vec![inputs[1].clone(), ValRef::Local(ct)],
            OpMode::Linear {
                active_mask: vec![false, true],
            },
        );
        result[0] = Some(out[0]);
    }

    if active_mask[1] {
        let out = builder.add_op(
            StdTensorOp::Mul,
            vec![inputs[0].clone(), ValRef::Local(ct)],
            OpMode::Linear {
                active_mask: vec![false, true],
            },
        );
        result[1] = Some(out[0]);
    }

    result
}

pub fn transpose_neg(
    builder: &mut FragmentBuilder<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
) -> Vec<Option<LocalValId>> {
    match cotangent_out[0] {
        Some(ct) => {
            let out = builder.add_op(
                StdTensorOp::Neg,
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

pub fn transpose_conj(
    builder: &mut FragmentBuilder<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
) -> Vec<Option<LocalValId>> {
    match cotangent_out[0] {
        Some(ct) => {
            let out = builder.add_op(
                StdTensorOp::Conj,
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
