use super::*;

pub(crate) fn linearize_factor(
    primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let packed = ctx
        .is_value_active_in_linearize(&primal_out[0])
        .then_some(tangent_in[0])
        .flatten();
    Ok(vec![packed, None])
}

pub(crate) fn linearize_from_factors(
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    if !ctx.is_value_active_in_linearize(&primal_out[0]) {
        return Ok(vec![None, None]);
    }
    let dq = tangent_in[0].map(|dq| {
        matmul_linear(
            builder,
            ValueRef::Local(dq),
            ValueRef::External(primal_in[1].clone()),
            vec![true, false],
            2,
        )
    });
    let q_dr = tangent_in[1].map(|dr| {
        let upper = linear_unary(builder, StdTensorOp::Triu { k: 0 }, dr);
        matmul_linear(
            builder,
            ValueRef::External(primal_in[0].clone()),
            ValueRef::Local(upper),
            vec![false, true],
            2,
        )
    });
    let packed = match (dq, q_dr) {
        (Some(dq), Some(q_dr)) => Some(linear_add(builder, dq, q_dr)),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    };
    Ok(vec![packed, None])
}

pub(crate) fn linearize_append(
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    if !ctx.is_value_active_in_linearize(&primal_out[0]) {
        return Ok(vec![None, None]);
    }
    let old_active = tangent_in[0].is_some();
    let block_active = tangent_in[2].is_some();
    if !old_active && !block_active {
        return Ok(vec![None, None]);
    }
    let old = tangent_in[0].unwrap_or_else(|| {
        fixed_binary(
            builder,
            StdTensorOp::Sub,
            ValueRef::External(primal_in[0].clone()),
            ValueRef::External(primal_in[0].clone()),
        )
    });
    let block = tangent_in[2].unwrap_or_else(|| {
        fixed_binary(
            builder,
            StdTensorOp::Sub,
            ValueRef::External(primal_in[2].clone()),
            ValueRef::External(primal_in[2].clone()),
        )
    });
    let packed = builder.add_operation(
        linalg_std_op(LinalgOp::HouseholderQrAppendTangent),
        vec![
            ValueRef::Local(old),
            ValueRef::Local(block),
            ValueRef::External(primal_in[0].clone()),
            ValueRef::External(primal_in[2].clone()),
        ],
        OperationRole::Linearized {
            active_mask: vec![old_active, block_active, false, false],
        },
    )[0];
    Ok(vec![Some(packed), None])
}

fn recover_factors(
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    gauge: QrGauge,
) -> (LocalValueId, LocalValueId) {
    let inputs = vec![
        ValueRef::External(primal_in[0].clone()),
        ValueRef::External(primal_in[1].clone()),
    ];
    let q = builder.add_operation(
        linalg_std_op(LinalgOp::HouseholderQrThinQ { gauge }),
        inputs.clone(),
        OperationRole::Primary,
    )[0];
    let r = builder.add_operation(
        linalg_std_op(LinalgOp::HouseholderQrR { gauge }),
        inputs,
        OperationRole::Primary,
    )[0];
    (q, r)
}

pub(crate) fn linearize_r(
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    gauge: QrGauge,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let Some(da) = tangent_in[0] else {
        return Ok(vec![None]);
    };
    if !ctx.is_value_active_in_linearize(&primal_out[0]) {
        return Ok(vec![None]);
    }
    let (q, r) = recover_factors(builder, primal_in, gauge);
    let derivatives = linearize_qr_with_factors(
        builder,
        primal_in,
        da,
        ValueRef::Local(q),
        ValueRef::Local(r),
        false,
        true,
        ctx,
    )?;
    Ok(vec![derivatives[1]])
}

// INVARIANT: these arguments directly mirror the selected-Q operation and shared QR rule.
#[allow(clippy::too_many_arguments)]
pub(crate) fn linearize_q_columns(
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    start: usize,
    end: usize,
    gauge: QrGauge,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let Some(da) = tangent_in[0] else {
        return Ok(vec![None]);
    };
    if !ctx.is_value_active_in_linearize(&primal_out[0]) {
        return Ok(vec![None]);
    }
    let Some(input_shape) = primal_matrix_input_shape(ctx, primal_in)? else {
        return Ok(vec![None]);
    };
    let k = DimExpr::min(input_shape[0].clone(), input_shape[1].clone());
    let dtype = ctx.dtype_of(&ValueRef::External(primal_in[0].clone()))?;
    let (q, r) = recover_factors(builder, primal_in, gauge);
    let derivatives = linearize_qr_with_factors(
        builder,
        primal_in,
        da,
        ValueRef::Local(q),
        ValueRef::Local(r),
        true,
        false,
        ctx,
    )?;
    let Some(dq) = derivatives[0] else {
        return Ok(vec![None]);
    };
    let concrete_k = match (&input_shape[0], &input_shape[1]) {
        (DimExpr::Const(m), DimExpr::Const(n)) => Some((*m).min(*n)),
        _ => None,
    };
    if start == 0 && concrete_k == Some(end) {
        return Ok(vec![Some(dq)]);
    }
    let selector = column_range_selector_symbolic(
        builder,
        dtype,
        k,
        start,
        end,
        &[],
        ValueRef::External(primal_in[0].clone()),
    );
    let selector_h = adjoint_matrix_fixed(builder, ValueRef::Local(selector), 2, dtype);
    let selected = matmul_linear(
        builder,
        ValueRef::Local(dq),
        ValueRef::Local(selector_h),
        vec![true, false],
        2,
    );
    Ok(vec![Some(selected)])
}
