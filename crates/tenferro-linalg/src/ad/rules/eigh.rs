use computegraph::types::{LocalValueId, OperationRole, ValueRef};
use tenferro_ops::ad::context::ShapeGuardContext;
use tenferro_ops::ad::PrimitiveRuleBuilder;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_tensor::DType;
use tidu::{ADRuleError, ADRuleKind, ADRuleResult};

use crate::extension::LinalgOp;

use super::linalg_std_op;
use super::support::*;

fn linear_transpose_input_active(mode: &OperationRole, input_index: usize) -> bool {
    match mode {
        OperationRole::Primary => true,
        OperationRole::Linearized { active_mask } => {
            active_mask.get(input_index).copied().unwrap_or(false)
        }
    }
}

/// `dA = V @ diag(gL) @ V^H`, symmetrized for real inputs.
fn emit_eigenvalue_cotangent_to_input(
    builder: &mut dyn PrimitiveRuleBuilder,
    g_l: LocalValueId,
    v: ValueRef<StdTensorOp>,
    rank: usize,
    dtype: DType,
    w_dtype: DType,
) -> ADRuleResult<LocalValueId> {
    let g_l_cast = convert_linear_to_dtype(builder, g_l, w_dtype, dtype);
    let g_l_mat = embed_diag_linear(builder, g_l_cast);
    let v_scaled = matmul_linear(
        builder,
        v.clone(),
        ValueRef::Local(g_l_mat),
        vec![false, true],
        rank,
    );
    let vh = adjoint_matrix_fixed(builder, v, rank, dtype);
    let da = matmul_linear(
        builder,
        ValueRef::Local(v_scaled),
        ValueRef::Local(vh),
        vec![true, false],
        rank,
    );
    Ok(self_adjoint_from_lower_linear(builder, da, rank, dtype))
}

struct EighTransposeContext {
    w: ValueRef<StdTensorOp>,
    v: ValueRef<StdTensorOp>,
    eps: f64,
    rank: usize,
    dtype: DType,
    w_dtype: DType,
}

fn emit_full_eigh_cotangent_to_input(
    builder: &mut dyn PrimitiveRuleBuilder,
    g_l: Option<LocalValueId>,
    g_v: LocalValueId,
    ctx: EighTransposeContext,
) -> ADRuleResult<LocalValueId> {
    let EighTransposeContext {
        w,
        v,
        eps,
        rank,
        dtype,
        w_dtype,
    } = ctx;
    let vh = adjoint_matrix_fixed(builder, v.clone(), rank, dtype);
    let vhgv = matmul_linear(
        builder,
        ValueRef::Local(vh),
        ValueRef::Local(g_v),
        vec![false, true],
        rank,
    );
    let vhgv_h = adjoint_matrix_linear(builder, vhgv, rank, dtype);
    let skew_diff = linear_sub(builder, vhgv, vhgv_h);
    let skew = linear_scale(builder, skew_diff, 0.5);

    let diag_w = embed_diag_fixed(builder, w.clone());
    let ones_mat = one_like_fixed(builder, ValueRef::Local(diag_w));
    let w_col = matmul_fixed(
        builder,
        ValueRef::Local(diag_w),
        ValueRef::Local(ones_mat),
        rank,
    );
    let w_row = matmul_fixed(
        builder,
        ValueRef::Local(ones_mat),
        ValueRef::Local(diag_w),
        rank,
    );
    let diff = fixed_sub(builder, ValueRef::Local(w_row), ValueRef::Local(w_col));
    let diff_sq = fixed_mul(builder, ValueRef::Local(diff), ValueRef::Local(diff));
    let eps_sq = fixed_scale(builder, ValueRef::Local(ones_mat), eps * eps);
    let safe_diff = fixed_add(builder, ValueRef::Local(diff_sq), ValueRef::Local(eps_sq));
    let f = fixed_div(builder, ValueRef::Local(diff), ValueRef::Local(safe_diff));
    let f = convert_fixed_ref_to_dtype(builder, ValueRef::Local(f), w_dtype, dtype);

    let mut g_a_inner = hadamard_fixed_linear(builder, f, skew);
    if let Some(g_l) = g_l {
        let g_l_cast = convert_linear_to_dtype(builder, g_l, w_dtype, dtype);
        let g_l_diag = embed_diag_linear(builder, g_l_cast);
        g_a_inner = linear_add(builder, g_a_inner, g_l_diag);
    }

    let tmp = matmul_linear(
        builder,
        v.clone(),
        ValueRef::Local(g_a_inner),
        vec![false, true],
        rank,
    );
    let vh = adjoint_matrix_fixed(builder, v, rank, dtype);
    let da = matmul_linear(
        builder,
        ValueRef::Local(tmp),
        ValueRef::Local(vh),
        vec![true, false],
        rank,
    );
    Ok(self_adjoint_from_lower_linear(builder, da, rank, dtype))
}

pub(crate) fn transpose_eigh(
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[ValueRef<StdTensorOp>],
    mode: &OperationRole,
    eps: f64,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    if !linear_transpose_input_active(mode, 0) {
        return Ok(vec![None]);
    }

    let g_l = cotangent_out.first().copied().flatten();
    let g_v = cotangent_out.get(1).copied().flatten();
    if g_l.is_none() && g_v.is_none() {
        return Ok(vec![None]);
    }

    let a = inputs.first().ok_or_else(|| {
        ADRuleError::invalid_input(
            "tenferro-linalg.eigh",
            ADRuleKind::Transpose,
            "expected one matrix input",
        )
    })?;
    let rank = ctx.rank_of(a)?;
    if rank < 2 {
        return Ok(vec![None]);
    }
    let dtype = ctx.dtype_of(a)?;

    let (w, v) = if let Some(primal_outputs) = ctx.transpose_primal_outputs() {
        if primal_outputs.len() < 2 {
            return Err(ADRuleError::invalid_input(
                "tenferro-linalg.eigh",
                ADRuleKind::Transpose,
                "expected two primal outputs for primary transpose",
            ));
        }
        (
            ValueRef::External(primal_outputs[0].clone()),
            ValueRef::External(primal_outputs[1].clone()),
        )
    } else {
        let eigh_outputs = builder.add_operation(
            linalg_std_op(LinalgOp::Eigh { eps }),
            vec![a.clone()],
            OperationRole::Primary,
        );
        (
            ValueRef::Local(eigh_outputs[0]),
            ValueRef::Local(eigh_outputs[1]),
        )
    };
    let w_dtype = ctx.dtype_of(&w)?;

    let da = match (g_l, g_v) {
        (Some(g_l), None) => {
            emit_eigenvalue_cotangent_to_input(builder, g_l, v, rank, dtype, w_dtype)?
        }
        (None, Some(g_v)) => emit_full_eigh_cotangent_to_input(
            builder,
            None,
            g_v,
            EighTransposeContext {
                w,
                v,
                eps,
                rank,
                dtype,
                w_dtype,
            },
        )?,
        (Some(g_l), Some(g_v)) => emit_full_eigh_cotangent_to_input(
            builder,
            Some(g_l),
            g_v,
            EighTransposeContext {
                w,
                v,
                eps,
                rank,
                dtype,
                w_dtype,
            },
        )?,
        (None, None) => return Ok(vec![None]),
    };

    Ok(vec![Some(da)])
}
