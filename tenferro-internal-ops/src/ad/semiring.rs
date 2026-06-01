use computegraph::types::{GlobalValKey, LocalValId, OpMode, ValRef};
use computegraph::OpEmitter;

use crate::ad::context::ShapeGuardContext;
use crate::ad::support::{conjugate_linear_if_dtype_complex, conjugate_primal_if_complex};
use crate::std_tensor_op::StdTensorOp;

pub fn linearize_add(
    builder: &mut dyn OpEmitter<StdTensorOp>,
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
    builder: &mut dyn OpEmitter<StdTensorOp>,
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
    builder: &mut dyn OpEmitter<StdTensorOp>,
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
    builder: &mut dyn OpEmitter<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    ctx: &mut ShapeGuardContext,
) -> Vec<Option<LocalValId>> {
    match tangent_in[0] {
        Some(dx) => {
            let dtype = ctx.dtype_of(&ValRef::External(primal_in[0].clone()));
            vec![Some(conjugate_linear_if_dtype_complex(builder, dx, dtype))]
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
    emitter: &mut dyn OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    inputs: &[ValRef<StdTensorOp>],
    mode: &OpMode,
    ctx: &mut ShapeGuardContext,
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
        let rhs_conj = conjugate_primal_if_complex(emitter, inputs[1].clone(), ctx);
        let out = emitter.add_op(
            StdTensorOp::Mul,
            vec![rhs_conj, ValRef::Local(ct)],
            OpMode::Linear {
                active_mask: vec![false, true],
            },
        );
        result[0] = Some(out[0]);
    }

    if active_mask[1] {
        let lhs_conj = conjugate_primal_if_complex(emitter, inputs[0].clone(), ctx);
        let out = emitter.add_op(
            StdTensorOp::Mul,
            vec![lhs_conj, ValRef::Local(ct)],
            OpMode::Linear {
                active_mask: vec![false, true],
            },
        );
        result[1] = Some(out[0]);
    }

    result
}

pub fn transpose_neg(
    emitter: &mut dyn OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
) -> Vec<Option<LocalValId>> {
    match cotangent_out[0] {
        Some(ct) => {
            let out = emitter.add_op(
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
    emitter: &mut dyn OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    inputs: &[ValRef<StdTensorOp>],
    ctx: &mut ShapeGuardContext,
) -> Vec<Option<LocalValId>> {
    match cotangent_out[0] {
        Some(ct) => {
            let dtype = ctx.dtype_of(&inputs[0]);
            vec![Some(conjugate_linear_if_dtype_complex(emitter, ct, dtype))]
        }
        None => vec![None],
    }
}
