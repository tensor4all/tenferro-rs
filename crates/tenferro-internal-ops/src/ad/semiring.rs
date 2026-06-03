use computegraph::types::{LocalValueId, OperationRole, ValueKey, ValueRef};

use crate::ad::context::ShapeGuardContext;
use crate::ad::support::{conjugate_linear_if_dtype_complex, conjugate_primal_if_complex};
use crate::ad::PrimitiveRuleBuilder;
use crate::std_tensor_op::StdTensorOp;

pub fn linearize_add(
    builder: &mut dyn PrimitiveRuleBuilder,
    tangent_in: &[Option<LocalValueId>],
) -> Vec<Option<LocalValueId>> {
    match (tangent_in[0], tangent_in[1]) {
        (Some(dx), Some(dy)) => {
            let out = builder.add_operation(
                StdTensorOp::Add,
                vec![ValueRef::Local(dx), ValueRef::Local(dy)],
                OperationRole::Linearized {
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
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
) -> Vec<Option<LocalValueId>> {
    let mut terms = Vec::with_capacity(2);

    if let Some(dx) = tangent_in[0] {
        let term = builder.add_operation(
            StdTensorOp::Mul,
            vec![
                ValueRef::Local(dx),
                ValueRef::External(primal_in[1].clone()),
            ],
            OperationRole::Linearized {
                active_mask: vec![true, false],
            },
        );
        terms.push(term[0]);
    }

    if let Some(dy) = tangent_in[1] {
        let term = builder.add_operation(
            StdTensorOp::Mul,
            vec![
                ValueRef::External(primal_in[0].clone()),
                ValueRef::Local(dy),
            ],
            OperationRole::Linearized {
                active_mask: vec![false, true],
            },
        );
        terms.push(term[0]);
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
        _ => unreachable!("mul linearization creates at most two terms"),
    }
}

pub fn linearize_neg(
    builder: &mut dyn PrimitiveRuleBuilder,
    tangent_in: &[Option<LocalValueId>],
) -> Vec<Option<LocalValueId>> {
    match tangent_in[0] {
        Some(dx) => {
            let out = builder.add_operation(
                StdTensorOp::Neg,
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

pub fn linearize_conj(
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    ctx: &mut ShapeGuardContext,
) -> Vec<Option<LocalValueId>> {
    match tangent_in[0] {
        Some(dx) => {
            let dtype = ctx.dtype_of(&ValueRef::External(primal_in[0].clone()));
            vec![Some(conjugate_linear_if_dtype_complex(builder, dx, dtype))]
        }
        None => vec![None],
    }
}

pub fn transpose_add(cotangent_out: &[Option<LocalValueId>]) -> Vec<Option<LocalValueId>> {
    match cotangent_out[0] {
        Some(ct) => vec![Some(ct), Some(ct)],
        None => vec![None, None],
    }
}

pub fn transpose_mul(
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

    let mut result = vec![None, None];

    if active_mask[0] {
        let rhs_conj = conjugate_primal_if_complex(builder, inputs[1].clone(), ctx);
        let out = builder.add_operation(
            StdTensorOp::Mul,
            vec![rhs_conj, ValueRef::Local(ct)],
            OperationRole::Linearized {
                active_mask: vec![false, true],
            },
        );
        result[0] = Some(out[0]);
    }

    if active_mask[1] {
        let lhs_conj = conjugate_primal_if_complex(builder, inputs[0].clone(), ctx);
        let out = builder.add_operation(
            StdTensorOp::Mul,
            vec![lhs_conj, ValueRef::Local(ct)],
            OperationRole::Linearized {
                active_mask: vec![false, true],
            },
        );
        result[1] = Some(out[0]);
    }

    result
}

pub fn transpose_neg(
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
) -> Vec<Option<LocalValueId>> {
    match cotangent_out[0] {
        Some(ct) => {
            let out = builder.add_operation(
                StdTensorOp::Neg,
                vec![ValueRef::Local(ct)],
                OperationRole::Linearized {
                    active_mask: vec![true],
                },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}

pub fn transpose_conj(
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[ValueRef<StdTensorOp>],
    ctx: &mut ShapeGuardContext,
) -> Vec<Option<LocalValueId>> {
    match cotangent_out[0] {
        Some(ct) => {
            let dtype = ctx.dtype_of(&inputs[0]);
            vec![Some(conjugate_linear_if_dtype_complex(builder, ct, dtype))]
        }
        None => vec![None],
    }
}
