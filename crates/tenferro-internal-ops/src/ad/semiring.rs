use computegraph::types::{LocalValueId, OperationRole, ValueKey, ValueRef};

use crate::ad::context::ShapeGuardContext;
use crate::ad::support::{
    conjugate_linear_if_dtype_complex, conjugate_primal_if_complex, convert_fixed_ref_to_dtype,
    convert_linear_to_dtype, dtype_of_or_real, project_linear_to_dtype, promote_dtype,
};
use crate::ad::PrimitiveRuleBuilder;
use crate::std_tensor_op::StdTensorOp;

pub fn linearize_add(
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    ctx: &mut ShapeGuardContext,
) -> Vec<Option<LocalValueId>> {
    let lhs_dtype = dtype_of_or_real(ctx, &ValueRef::External(primal_in[0].clone()));
    let rhs_dtype = dtype_of_or_real(ctx, &ValueRef::External(primal_in[1].clone()));
    let output_dtype = promote_dtype(lhs_dtype, rhs_dtype);
    let lhs_tangent =
        tangent_in[0].map(|dx| convert_linear_to_dtype(builder, dx, lhs_dtype, output_dtype));
    let rhs_tangent =
        tangent_in[1].map(|dy| convert_linear_to_dtype(builder, dy, rhs_dtype, output_dtype));

    match (lhs_tangent, rhs_tangent) {
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
    ctx: &mut ShapeGuardContext,
) -> Vec<Option<LocalValueId>> {
    let lhs_dtype = dtype_of_or_real(ctx, &ValueRef::External(primal_in[0].clone()));
    let rhs_dtype = dtype_of_or_real(ctx, &ValueRef::External(primal_in[1].clone()));
    let output_dtype = promote_dtype(lhs_dtype, rhs_dtype);
    let mut terms = Vec::with_capacity(2);

    if let Some(dx) = tangent_in[0] {
        let dx = convert_linear_to_dtype(builder, dx, lhs_dtype, output_dtype);
        let rhs = convert_fixed_ref_to_dtype(
            builder,
            ValueRef::External(primal_in[1].clone()),
            rhs_dtype,
            output_dtype,
        );
        let term = builder.add_operation(
            StdTensorOp::Mul,
            vec![ValueRef::Local(dx), rhs],
            OperationRole::Linearized {
                active_mask: vec![true, false],
            },
        );
        terms.push(term[0]);
    }

    if let Some(dy) = tangent_in[1] {
        let lhs = convert_fixed_ref_to_dtype(
            builder,
            ValueRef::External(primal_in[0].clone()),
            lhs_dtype,
            output_dtype,
        );
        let dy = convert_linear_to_dtype(builder, dy, rhs_dtype, output_dtype);
        let term = builder.add_operation(
            StdTensorOp::Mul,
            vec![lhs, ValueRef::Local(dy)],
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

pub fn transpose_add(
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[ValueRef<StdTensorOp>],
    ctx: &mut ShapeGuardContext,
) -> Vec<Option<LocalValueId>> {
    match cotangent_out[0] {
        Some(ct) => {
            let lhs_dtype = dtype_of_or_real(ctx, &inputs[0]);
            let rhs_dtype = dtype_of_or_real(ctx, &inputs[1]);
            let output_dtype = promote_dtype(lhs_dtype, rhs_dtype);
            vec![
                Some(project_linear_to_dtype(
                    builder,
                    ct,
                    output_dtype,
                    lhs_dtype,
                )),
                Some(project_linear_to_dtype(
                    builder,
                    ct,
                    output_dtype,
                    rhs_dtype,
                )),
            ]
        }
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

    let lhs_dtype = dtype_of_or_real(ctx, &inputs[0]);
    let rhs_dtype = dtype_of_or_real(ctx, &inputs[1]);
    let output_dtype = promote_dtype(lhs_dtype, rhs_dtype);
    let mut result = vec![None, None];

    if active_mask[0] {
        let rhs_conj = conjugate_primal_if_complex(builder, inputs[1].clone(), ctx);
        let rhs_conj = convert_fixed_ref_to_dtype(builder, rhs_conj, rhs_dtype, output_dtype);
        let out = builder.add_operation(
            StdTensorOp::Mul,
            vec![rhs_conj, ValueRef::Local(ct)],
            OperationRole::Linearized {
                active_mask: vec![false, true],
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
        let lhs_conj = conjugate_primal_if_complex(builder, inputs[0].clone(), ctx);
        let lhs_conj = convert_fixed_ref_to_dtype(builder, lhs_conj, lhs_dtype, output_dtype);
        let out = builder.add_operation(
            StdTensorOp::Mul,
            vec![lhs_conj, ValueRef::Local(ct)],
            OperationRole::Linearized {
                active_mask: vec![false, true],
            },
        );
        result[1] = Some(project_linear_to_dtype(
            builder,
            out[0],
            output_dtype,
            rhs_dtype,
        ));
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
