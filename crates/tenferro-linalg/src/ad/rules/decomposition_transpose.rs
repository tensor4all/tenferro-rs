use computegraph::types::{LocalValueId, OperationRole, ValueRef};
use tenferro_ops::ad::context::{resolve_and_guard, ShapeGuardContext};
use tenferro_ops::ad::PrimitiveRuleBuilder;
use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_tensor::DType;
use tidu::{ADRuleError, ADRuleKind, ADRuleResult};

use crate::extension::LinalgOp;

use super::support::*;
use super::{conjugate_primal_if_dtype_complex, linalg_std_op};

fn transpose_input_active(mode: &OperationRole, input_index: usize) -> bool {
    match mode {
        OperationRole::Primary => true,
        OperationRole::Linearized { active_mask } => {
            active_mask.get(input_index).copied().unwrap_or(false)
        }
    }
}

fn matrix_shape_from_input(
    ctx: &mut ShapeGuardContext,
    input: &ValueRef<StdTensorOp>,
) -> ADRuleResult<Option<Vec<DimExpr>>> {
    if let Some(exact_shape) = ctx.shape_if_available(input) {
        return if let Some(concrete) = exact_shape
            .iter()
            .map(|dim| dim.constant_value())
            .collect::<Option<Vec<_>>>()
        {
            Ok((concrete.len() >= 2).then(|| DimExpr::from_concrete(&concrete)))
        } else {
            Ok((exact_shape.len() >= 2).then(|| DimExpr::input_shape(0, exact_shape.len())))
        };
    }

    let rank = ctx.rank_of(input)?;
    Ok((rank >= 2).then(|| DimExpr::input_shape(0, rank)))
}

fn invalid_transpose_dim_expr(op: &'static str, err: impl std::fmt::Display) -> ADRuleError {
    ADRuleError::invalid_input(
        format!("tenferro-linalg.{op}"),
        ADRuleKind::Transpose,
        format!("invalid matrix dimension expression: {err}"),
    )
}

fn expected_primal_outputs_error(op: &'static str, expected: usize) -> ADRuleError {
    ADRuleError::invalid_input(
        format!("tenferro-linalg.{op}"),
        ADRuleKind::Transpose,
        format!("expected {expected} primal outputs for primary transpose"),
    )
}

fn lu_primal_outputs(
    builder: &mut dyn PrimitiveRuleBuilder,
    input: ValueRef<StdTensorOp>,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<(
    ValueRef<StdTensorOp>,
    ValueRef<StdTensorOp>,
    ValueRef<StdTensorOp>,
)> {
    if let Some(primal_outputs) = ctx.transpose_primal_outputs() {
        if primal_outputs.len() < 4 {
            return Err(expected_primal_outputs_error("lu", 4));
        }
        return Ok((
            ValueRef::External(primal_outputs[0].clone()),
            ValueRef::External(primal_outputs[1].clone()),
            ValueRef::External(primal_outputs[2].clone()),
        ));
    }

    let outputs = builder.add_operation(
        linalg_std_op(LinalgOp::Lu),
        vec![input],
        OperationRole::Primary,
    );
    Ok((
        ValueRef::Local(outputs[0]),
        ValueRef::Local(outputs[1]),
        ValueRef::Local(outputs[2]),
    ))
}

fn qr_primal_outputs(
    builder: &mut dyn PrimitiveRuleBuilder,
    input: ValueRef<StdTensorOp>,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<(ValueRef<StdTensorOp>, ValueRef<StdTensorOp>)> {
    if let Some(primal_outputs) = ctx.transpose_primal_outputs() {
        if primal_outputs.len() < 2 {
            return Err(expected_primal_outputs_error("qr", 2));
        }
        return Ok((
            ValueRef::External(primal_outputs[0].clone()),
            ValueRef::External(primal_outputs[1].clone()),
        ));
    }

    let outputs = builder.add_operation(
        linalg_std_op(LinalgOp::Qr),
        vec![input],
        OperationRole::Primary,
    );
    Ok((ValueRef::Local(outputs[0]), ValueRef::Local(outputs[1])))
}

fn add_optional_linear(
    builder: &mut dyn PrimitiveRuleBuilder,
    lhs: Option<LocalValueId>,
    rhs: LocalValueId,
) -> LocalValueId {
    match lhs {
        Some(lhs) => linear_add(builder, lhs, rhs),
        None => rhs,
    }
}

fn left_solve_lower_adjoint(
    builder: &mut dyn PrimitiveRuleBuilder,
    lower: ValueRef<StdTensorOp>,
    rhs: LocalValueId,
    unit_diagonal: bool,
    dtype: DType,
) -> LocalValueId {
    let lower = conjugate_primal_if_dtype_complex(builder, lower, dtype);
    builder.add_operation(
        linalg_std_op(LinalgOp::TriangularSolve {
            left_side: true,
            lower: true,
            transpose_a: true,
            unit_diagonal,
        }),
        vec![lower, ValueRef::Local(rhs)],
        OperationRole::Linearized {
            active_mask: vec![false, true],
        },
    )[0]
}

fn right_solve_upper_adjoint(
    builder: &mut dyn PrimitiveRuleBuilder,
    upper: ValueRef<StdTensorOp>,
    rhs: LocalValueId,
    dtype: DType,
) -> LocalValueId {
    let upper = conjugate_primal_if_dtype_complex(builder, upper, dtype);
    builder.add_operation(
        linalg_std_op(LinalgOp::TriangularSolve {
            left_side: false,
            lower: false,
            transpose_a: true,
            unit_diagonal: false,
        }),
        vec![upper, ValueRef::Local(rhs)],
        OperationRole::Linearized {
            active_mask: vec![false, true],
        },
    )[0]
}

fn tril_im_inv_adj_skew_linear(
    builder: &mut dyn PrimitiveRuleBuilder,
    input: LocalValueId,
    rank: usize,
    dtype: DType,
) -> LocalValueId {
    let input_h = adjoint_matrix_linear(builder, input, rank, dtype);
    let skew = linear_sub(builder, input, input_h);
    let strict_lower = linear_unary(builder, StdTensorOp::Tril { k: -1 }, skew);
    let diag = extract_diag_linear(builder, skew);
    let half_diag = linear_scale(builder, diag, 0.5);
    let half_diag_mat = embed_diag_linear(builder, half_diag);
    linear_add(builder, strict_lower, half_diag_mat)
}

pub(crate) fn transpose_lu(
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[ValueRef<StdTensorOp>],
    mode: &OperationRole,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    if !transpose_input_active(mode, 0) {
        return Ok(vec![None]);
    }

    let g_l = cotangent_out.get(1).copied().flatten();
    let g_u = cotangent_out.get(2).copied().flatten();
    if g_l.is_none() && g_u.is_none() {
        return Ok(vec![None]);
    }

    let input = inputs.first().ok_or_else(|| {
        ADRuleError::invalid_input(
            "tenferro-linalg.lu",
            ADRuleKind::Transpose,
            "expected one matrix input",
        )
    })?;
    let Some(input_shape) = matrix_shape_from_input(ctx, input)? else {
        return Ok(vec![None]);
    };
    let input_shape = input_shape.as_slice();
    let (m, n, batch_shape) = matrix_shape_parts(input_shape, "transpose_lu");
    let (m_size, n_size) =
        resolve_and_guard(m, n, ctx).map_err(|err| invalid_transpose_dim_expr("lu", err))?;
    let k = DimExpr::min(m.clone(), n.clone());
    let k_size = m_size.min(n_size);
    let rank = input_shape.len();
    let dtype = ctx.dtype_of(input)?;
    let (p, l, u) = lu_primal_outputs(builder, input.clone(), ctx)?;
    let l_shape = matrix_shape(m, &k, batch_shape);
    let u_shape = matrix_shape(&k, n, batch_shape);
    let l_square = augment_unit_lower_to_square_fixed(
        builder,
        l.clone(),
        m_size,
        k_size,
        batch_shape,
        &l_shape,
        rank,
    );
    let u_square = augment_upper_to_square_fixed(
        builder,
        u.clone(),
        k_size,
        n_size,
        batch_shape,
        &u_shape,
        rank,
    );

    let mut f_sum = None;
    if let Some(g_l) = g_l {
        let g_l = if n_size > k_size {
            pad_linear(
                builder,
                g_l,
                rank,
                vec![0; rank],
                pad_vec(rank, 1, (n_size - k_size) as i64),
            )
        } else {
            g_l
        };
        let l_h = adjoint_matrix_fixed(builder, ValueRef::Local(l_square), rank, dtype);
        let term = matmul_linear(
            builder,
            ValueRef::Local(l_h),
            ValueRef::Local(g_l),
            vec![false, true],
            rank,
        );
        let term = linear_unary(builder, StdTensorOp::Tril { k: -1 }, term);
        f_sum = Some(add_optional_linear(builder, f_sum, term));
    }
    if let Some(g_u) = g_u {
        let g_u = if m_size > k_size {
            pad_linear(
                builder,
                g_u,
                rank,
                vec![0; rank],
                pad_vec(rank, 0, (m_size - k_size) as i64),
            )
        } else {
            g_u
        };
        let u_h = adjoint_matrix_fixed(builder, ValueRef::Local(u_square), rank, dtype);
        let term = matmul_linear(
            builder,
            ValueRef::Local(g_u),
            ValueRef::Local(u_h),
            vec![true, false],
            rank,
        );
        let term = linear_unary(builder, StdTensorOp::Triu { k: 0 }, term);
        f_sum = Some(add_optional_linear(builder, f_sum, term));
    }

    let Some(f_sum) = f_sum else {
        return Ok(vec![None]);
    };
    let y = left_solve_lower_adjoint(builder, ValueRef::Local(l_square), f_sum, true, dtype);
    let z = right_solve_upper_adjoint(builder, ValueRef::Local(u_square), y, dtype);
    let p_t = transpose_matrix_fixed(builder, p, rank);
    let da = matmul_linear(
        builder,
        ValueRef::Local(p_t),
        ValueRef::Local(z),
        vec![false, true],
        rank,
    );
    Ok(vec![Some(da)])
}

pub(crate) fn transpose_qr(
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[ValueRef<StdTensorOp>],
    mode: &OperationRole,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    if !transpose_input_active(mode, 0) {
        return Ok(vec![None]);
    }

    let g_q = cotangent_out.first().copied().flatten();
    let g_r = cotangent_out.get(1).copied().flatten();
    if g_q.is_none() && g_r.is_none() {
        return Ok(vec![None]);
    }

    let input = inputs.first().ok_or_else(|| {
        ADRuleError::invalid_input(
            "tenferro-linalg.qr",
            ADRuleKind::Transpose,
            "expected one matrix input",
        )
    })?;
    let Some(input_shape) = matrix_shape_from_input(ctx, input)? else {
        return Ok(vec![None]);
    };
    let input_shape = input_shape.as_slice();
    let (m, n, batch_shape) = matrix_shape_parts(input_shape, "transpose_qr");
    let (m_size, n_size) =
        resolve_and_guard(m, n, ctx).map_err(|err| invalid_transpose_dim_expr("qr", err))?;
    let rank = input_shape.len();
    let dtype = ctx.dtype_of(input)?;
    let (q, r) = qr_primal_outputs(builder, input.clone(), ctx)?;

    if m_size >= n_size {
        let mut w_sum = None;
        if let Some(g_r) = g_r {
            let g_r_h = adjoint_matrix_linear(builder, g_r, rank, dtype);
            let term = matmul_linear(
                builder,
                r.clone(),
                ValueRef::Local(g_r_h),
                vec![false, true],
                rank,
            );
            w_sum = Some(add_optional_linear(builder, w_sum, term));
        }
        if let Some(g_q) = g_q {
            let g_q_h = adjoint_matrix_linear(builder, g_q, rank, dtype);
            let term = matmul_linear(
                builder,
                ValueRef::Local(g_q_h),
                q.clone(),
                vec![true, false],
                rank,
            );
            let term = linear_neg(builder, term);
            w_sum = Some(add_optional_linear(builder, w_sum, term));
        }
        let Some(w_sum) = w_sum else {
            return Ok(vec![None]);
        };
        let h_sum = self_adjoint_from_lower_linear(builder, w_sum, rank, dtype);
        let q_h_sum = matmul_linear(
            builder,
            q.clone(),
            ValueRef::Local(h_sum),
            vec![false, true],
            rank,
        );
        let rhs = match g_q {
            Some(g_q) => linear_add(builder, g_q, q_h_sum),
            None => q_h_sum,
        };
        let da = right_solve_upper_adjoint(builder, r, rhs, dtype);
        return Ok(vec![Some(da)]);
    }

    let mut result = None;
    if let Some(g_r) = g_r {
        let direct = matmul_linear(
            builder,
            q.clone(),
            ValueRef::Local(g_r),
            vec![false, true],
            rank,
        );
        result = Some(add_optional_linear(builder, result, direct));
    }

    let mut x = None;
    if let Some(g_q) = g_q {
        let q_h = adjoint_matrix_fixed(builder, q.clone(), rank, dtype);
        let term = matmul_linear(
            builder,
            ValueRef::Local(q_h),
            ValueRef::Local(g_q),
            vec![false, true],
            rank,
        );
        x = Some(add_optional_linear(builder, x, term));
    }
    if let Some(g_r) = g_r {
        let r_h = adjoint_matrix_fixed(builder, r.clone(), rank, dtype);
        let term = matmul_linear(
            builder,
            ValueRef::Local(g_r),
            ValueRef::Local(r_h),
            vec![true, false],
            rank,
        );
        let term = linear_neg(builder, term);
        x = Some(add_optional_linear(builder, x, term));
    }
    if let Some(x) = x {
        let helper = tril_im_inv_adj_skew_linear(builder, x, rank, dtype);
        let q_helper = matmul_linear(builder, q, ValueRef::Local(helper), vec![false, true], rank);
        let leading_selector = leading_column_selector_fixed(
            builder,
            m_size,
            n_size,
            batch_shape,
            r.clone(),
            input_shape,
        );
        let leading_selector_t =
            transpose_matrix_fixed(builder, ValueRef::Local(leading_selector), rank);
        let r_leading = matmul_fixed(builder, r, ValueRef::Local(leading_selector_t), rank);
        let lead = right_solve_upper_adjoint(builder, ValueRef::Local(r_leading), q_helper, dtype);
        let padded = pad_linear(
            builder,
            lead,
            rank,
            vec![0; rank],
            pad_vec(rank, 1, (n_size - m_size) as i64),
        );
        result = Some(add_optional_linear(builder, result, padded));
    }

    Ok(vec![result])
}
