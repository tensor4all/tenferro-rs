// Graph-emitting linalg AD rules (`linearize` / `transpose_rule`).
//
// Correctness here is validated NUMERICALLY, not by line coverage. Every
// differentiable output of these rules is checked against finite-difference
// and/or Torch oracles, and each op's per-output support status is asserted
// against the machine-readable manifest:
//   - finite-diff sweep: ../../../tests/traced_ad_explicit.rs
//     (`remaining_linalg_ops_jvp_match_finite_diff_except_full_piv_lu` covers
//     cholesky, qr, eig, eigh, lu, full_piv_lu, solve, triangular_solve;
//     plus svd/eigh/eig *values* tests)
//   - manifest: `super::support::all_linalg_ad_support` asserted by
//     ../../../tests/ad_support_manifest.rs
//   - oracle table: docs/oracle/tensor-ad-oracles-support.md
//
// These files therefore carry intentionally below-default per-file thresholds
// in coverage-thresholds.json; the uncovered lines are dtype-guard arms
// (real->complex casts, integer-dtype early returns) and F32/error branches the
// oracles do not exercise. See the "AD Rule Coverage" section of
// REPOSITORY_RULES.md before changing a rule or its threshold.
use std::sync::Arc;
use tenferro_ops::ad::PrimitiveRuleBuilder;

use computegraph::types::{LocalValueId, OperationRole, ValueKey, ValueRef};

use crate::extension::{EighGauge, LinalgExtensionOp, LinalgOp, SvdGauge};
use tenferro_ops::ad::context::{resolve_and_guard, ShapeGuardContext};
pub(crate) use tenferro_ops::ad::support::{
    conjugate_linear_if_dtype_complex, conjugate_primal_if_dtype_complex,
};
use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_tensor::DType;
use tidu::{ADRuleError, ADRuleKind, ADRuleResult};

mod solve;
mod support;

pub(crate) use solve::{
    linearize_full_piv_lu_solve, linearize_lu_solve_prepared, linearize_triangular_solve,
    transpose_full_piv_lu_solve, transpose_lu_solve_prepared, transpose_triangular_solve,
    TriangularSolveFlags,
};
use support::*;

/// Derive an op-local `Vec<DimExpr>` for the primal input of a 1-input linalg op.
///
/// Category C `input_shape` snapshots have been removed from the linalg op
/// variants; instead, AD rules resolve the primal input's shape through the
/// [`ShapeGuardContext`] metadata surface. Concrete shapes collapse to
/// `DimExpr::Const`; non-constant exact shapes and rank-known bounded metadata
/// fall back to `DimExpr::input_shape(0, rank)` so graph-emitting rules that only
/// need rank can preserve runtime shape sources.
fn primal_input_shape(
    ctx: &mut ShapeGuardContext,
    primal_in: &[ValueKey<StdTensorOp>],
) -> ADRuleResult<Option<Vec<DimExpr>>> {
    let input = ValueRef::External(primal_in[0].clone());
    if let Some(exact_shape) = ctx.shape_if_available(&input) {
        return if let Some(concrete) = exact_shape
            .iter()
            .map(|dim| dim.constant_value())
            .collect::<Option<Vec<_>>>()
        {
            Ok(Some(DimExpr::from_concrete(&concrete)))
        } else {
            Ok(Some(DimExpr::input_shape(0, exact_shape.len())))
        };
    }

    let rank = ctx.rank_of(&input)?;
    Ok(Some(DimExpr::input_shape(0, rank)))
}

fn primal_matrix_input_shape(
    ctx: &mut ShapeGuardContext,
    primal_in: &[ValueKey<StdTensorOp>],
) -> ADRuleResult<Option<Vec<DimExpr>>> {
    let Some(input_shape) = primal_input_shape(ctx, primal_in)? else {
        return Ok(None);
    };
    Ok((input_shape.len() >= 2).then_some(input_shape))
}

fn linalg_std_op(op: LinalgOp) -> StdTensorOp {
    StdTensorOp::Extension(Arc::new(LinalgExtensionOp::new(op)))
}

fn invalid_dim_expr(op: &'static str, err: impl std::fmt::Display) -> ADRuleError {
    ADRuleError::invalid_input(
        format!("tenferro-linalg.{op}"),
        ADRuleKind::Jvp,
        format!("invalid matrix dimension expression: {err}"),
    )
}

pub(crate) fn linearize_lu(
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let Some(da) = tangent_in[0] else {
        return Ok(vec![None, None, None, None]);
    };

    let l_active = ctx.is_value_active_in_linearize(&primal_out[1]);
    let u_active = ctx.is_value_active_in_linearize(&primal_out[2]);
    if !l_active && !u_active {
        return Ok(vec![None, None, None, None]);
    }

    let Some(input_shape) = primal_matrix_input_shape(ctx, primal_in)? else {
        return Ok(vec![None, None, None, None]);
    };
    let input_shape = input_shape.as_slice();
    let (m, n, batch_shape) = matrix_shape_parts(input_shape, "linearize_lu");
    let (m_size, n_size) =
        resolve_and_guard(m, n, ctx).map_err(|err| invalid_dim_expr("linearize_lu", err))?;
    let k_size = m_size.min(n_size);
    let rank = input_shape.len();
    let dtype = ctx.dtype_of(&ValueRef::External(primal_in[0].clone()))?;
    let p = ValueRef::External(primal_out[0].clone());
    let l = ValueRef::External(primal_out[1].clone());
    let u = ValueRef::External(primal_out[2].clone());
    let l_square = if m_size == k_size {
        l.clone()
    } else {
        ValueRef::Local(augment_unit_lower_to_square_fixed(
            builder,
            l.clone(),
            SquareAugmentSpec::new(dtype, m_size, k_size, batch_shape, rank),
        ))
    };
    let u_square = if k_size == n_size {
        u.clone()
    } else {
        ValueRef::Local(augment_upper_to_square_fixed(
            builder,
            u.clone(),
            SquareAugmentSpec::new(dtype, k_size, n_size, batch_shape, rank),
        ))
    };

    let pd_a = matmul_linear(builder, p, ValueRef::Local(da), vec![false, true], rank);
    let la = builder.add_operation(
        linalg_std_op(LinalgOp::TriangularSolve {
            left_side: true,
            lower: true,
            transpose_a: false,
            unit_diagonal: true,
        }),
        vec![l_square.clone(), ValueRef::Local(pd_a)],
        OperationRole::Linearized {
            active_mask: vec![false, true],
        },
    )[0];
    let x = builder.add_operation(
        linalg_std_op(LinalgOp::TriangularSolve {
            left_side: false,
            lower: false,
            transpose_a: false,
            unit_diagonal: false,
        }),
        vec![u_square.clone(), ValueRef::Local(la)],
        OperationRole::Linearized {
            active_mask: vec![false, true],
        },
    )[0];

    let dl = if l_active {
        let x_lower = linear_unary(builder, StdTensorOp::Tril { k: -1 }, x);
        let dl_full = matmul_linear(
            builder,
            l_square,
            ValueRef::Local(x_lower),
            vec![false, true],
            rank,
        );
        Some(if n_size > k_size {
            take_leading_cols_linear(
                builder,
                dl_full,
                LeadingMatrixSlice::new(k_size, n_size, dtype, batch_shape, l.clone(), rank),
            )
        } else {
            dl_full
        })
    } else {
        None
    };
    let du = if u_active {
        let x_upper = linear_unary(builder, StdTensorOp::Triu { k: 0 }, x);
        let du_full = matmul_linear(
            builder,
            ValueRef::Local(x_upper),
            u_square,
            vec![true, false],
            rank,
        );
        Some(if m_size > k_size {
            take_leading_rows_linear(
                builder,
                du_full,
                LeadingMatrixSlice::new(k_size, m_size, dtype, batch_shape, u.clone(), rank),
            )
        } else {
            du_full
        })
    } else {
        None
    };

    Ok(vec![None, dl, du, None])
}

pub(crate) fn linearize_full_piv_lu(
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let Some(da) = tangent_in[0] else {
        return Ok(vec![None, None, None, None, None]);
    };

    let Some(input_shape) = primal_matrix_input_shape(ctx, primal_in)? else {
        return Ok(vec![None, None, None, None, None]);
    };
    let input_shape = input_shape.as_slice();
    let (rows, cols, _batch_shape) = matrix_shape_parts(input_shape, "linearize_full_piv_lu");
    let (rows_size, cols_size) = resolve_and_guard(rows, cols, ctx)
        .map_err(|err| invalid_dim_expr("linearize_full_piv_lu", err))?;
    if rows_size != cols_size {
        return Ok(vec![None, None, None, None, None]);
    }

    let rank = input_shape.len();
    let p = ValueRef::External(primal_out[0].clone());
    let l = ValueRef::External(primal_out[1].clone());
    let u = ValueRef::External(primal_out[2].clone());
    let q = ValueRef::External(primal_out[3].clone());
    let q_t = transpose_matrix_fixed(builder, q, rank);

    let pd_a = matmul_linear(builder, p, ValueRef::Local(da), vec![false, true], rank);
    let pd_a_qt = matmul_linear(
        builder,
        ValueRef::Local(pd_a),
        ValueRef::Local(q_t),
        vec![true, false],
        rank,
    );
    let la = builder.add_operation(
        linalg_std_op(LinalgOp::TriangularSolve {
            left_side: true,
            lower: true,
            transpose_a: false,
            unit_diagonal: true,
        }),
        vec![l.clone(), ValueRef::Local(pd_a_qt)],
        OperationRole::Linearized {
            active_mask: vec![false, true],
        },
    )[0];
    let x = builder.add_operation(
        linalg_std_op(LinalgOp::TriangularSolve {
            left_side: false,
            lower: false,
            transpose_a: false,
            unit_diagonal: false,
        }),
        vec![u.clone(), ValueRef::Local(la)],
        OperationRole::Linearized {
            active_mask: vec![false, true],
        },
    )[0];

    let x_lower = linear_unary(builder, StdTensorOp::Tril { k: -1 }, x);
    let x_upper = linear_unary(builder, StdTensorOp::Triu { k: 0 }, x);
    let dl = matmul_linear(
        builder,
        l,
        ValueRef::Local(x_lower),
        vec![false, true],
        rank,
    );
    let du = matmul_linear(
        builder,
        ValueRef::Local(x_upper),
        u,
        vec![true, false],
        rank,
    );

    Ok(vec![None, Some(dl), Some(du), None, None])
}

pub(crate) fn linearize_eig(
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    input_dtype: DType,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let Some(da) = tangent_in[0] else {
        return Ok(vec![None, None]);
    };

    if !ctx.is_value_active_in_linearize(&primal_out[0]) {
        return Ok(vec![None, None]);
    }

    let Some(input_shape) = primal_matrix_input_shape(ctx, primal_in)? else {
        return Ok(vec![None, None]);
    };
    let input_shape = input_shape.as_slice();
    let rank = input_shape.len();
    let v = ValueRef::External(primal_out[1].clone());
    let da_complex = match input_dtype {
        DType::F64 => builder.add_operation(
            StdTensorOp::Convert {
                from: input_dtype,
                to: DType::C64,
            },
            vec![ValueRef::Local(da)],
            OperationRole::Linearized {
                active_mask: vec![true],
            },
        )[0],
        DType::F32 => builder.add_operation(
            StdTensorOp::Convert {
                from: input_dtype,
                to: DType::C32,
            },
            vec![ValueRef::Local(da)],
            OperationRole::Linearized {
                active_mask: vec![true],
            },
        )[0],
        DType::C64 | DType::C32 => da,
        DType::I32 | DType::I64 | DType::Bool => return Ok(vec![None, None]),
    };

    let dav = matmul_linear(
        builder,
        ValueRef::Local(da_complex),
        v.clone(),
        vec![true, false],
        rank,
    );
    let projected = solve_in_graph(builder, v, ValueRef::Local(dav), rank);
    let dw = extract_diag_linear(builder, projected);
    Ok(vec![Some(dw), None])
}

pub(crate) fn linearize_eig_values(
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    input_dtype: DType,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let Some(da) = tangent_in[0] else {
        return Ok(vec![None]);
    };

    let Some(input_shape) = primal_matrix_input_shape(ctx, primal_in)? else {
        return Ok(vec![None]);
    };
    let input_shape = input_shape.as_slice();
    let rank = input_shape.len();
    let eig_outputs = builder.add_operation(
        linalg_std_op(LinalgOp::Eig { input_dtype }),
        vec![ValueRef::External(primal_in[0].clone())],
        OperationRole::Primary,
    );
    let v = ValueRef::Local(eig_outputs[1]);
    let da_complex = match input_dtype {
        DType::F64 => builder.add_operation(
            StdTensorOp::Convert {
                from: input_dtype,
                to: DType::C64,
            },
            vec![ValueRef::Local(da)],
            OperationRole::Linearized {
                active_mask: vec![true],
            },
        )[0],
        DType::F32 => builder.add_operation(
            StdTensorOp::Convert {
                from: input_dtype,
                to: DType::C32,
            },
            vec![ValueRef::Local(da)],
            OperationRole::Linearized {
                active_mask: vec![true],
            },
        )[0],
        DType::C64 | DType::C32 => da,
        DType::I32 | DType::I64 | DType::Bool => return Ok(vec![None]),
    };

    let dav = matmul_linear(
        builder,
        ValueRef::Local(da_complex),
        v.clone(),
        vec![true, false],
        rank,
    );
    let projected = solve_in_graph(builder, v, ValueRef::Local(dav), rank);
    let dw = extract_diag_linear(builder, projected);
    Ok(vec![Some(dw)])
}

pub(crate) fn linearize_svd(
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    eps: f64,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let Some(da) = tangent_in[0] else {
        return Ok(vec![None, None, None]);
    };

    let u_active = ctx.is_value_active_in_linearize(&primal_out[0]);
    let s_active = ctx.is_value_active_in_linearize(&primal_out[1]);
    let vt_active = ctx.is_value_active_in_linearize(&primal_out[2]);
    if !u_active && !s_active && !vt_active {
        return Ok(vec![None, None, None]);
    }

    let Some(input_shape) = primal_matrix_input_shape(ctx, primal_in)? else {
        return Ok(vec![None, None, None]);
    };
    let input_shape = input_shape.as_slice();
    let (m, n, batch_shape) = matrix_shape_parts(input_shape, "linearize_svd");
    let (m_size, n_size) =
        resolve_and_guard(m, n, ctx).map_err(|err| invalid_dim_expr("linearize_svd", err))?;
    let matrix_rank = input_shape.len();
    let dtype = ctx.dtype_of(&ValueRef::External(primal_in[0].clone()))?;
    let k = DimExpr::min(m.clone(), n.clone());
    let u = ValueRef::External(primal_out[0].clone());
    let s = ValueRef::External(primal_out[1].clone());
    let vt = ValueRef::External(primal_out[2].clone());
    let s_dtype = ctx.dtype_of(&s)?;

    let uh = adjoint_matrix_fixed(builder, u.clone(), matrix_rank, dtype);
    let v = adjoint_matrix_fixed(builder, vt.clone(), matrix_rank, dtype);
    let tmp = matmul_linear(
        builder,
        ValueRef::Local(uh),
        ValueRef::Local(da),
        vec![false, true],
        matrix_rank,
    );
    let ds_mat = matmul_linear(
        builder,
        ValueRef::Local(tmp),
        ValueRef::Local(v),
        vec![true, false],
        matrix_rank,
    );
    let ds = s_active.then(|| extract_diag_linear(builder, ds_mat));

    if !u_active && !vt_active {
        return Ok(vec![None, ds, None]);
    }

    let diag_s = embed_diag_fixed(builder, s.clone());
    let ones_mat = one_like_fixed(builder, s_dtype, ValueRef::Local(diag_s), matrix_rank);
    let s_dim = matmul_fixed(
        builder,
        ValueRef::Local(ones_mat),
        ValueRef::Local(diag_s),
        matrix_rank,
    );
    let s_dim_t = matmul_fixed(
        builder,
        ValueRef::Local(diag_s),
        ValueRef::Local(ones_mat),
        matrix_rank,
    );
    let s_sum = fixed_add(builder, ValueRef::Local(s_dim), ValueRef::Local(s_dim_t));
    let s_diff = fixed_sub(builder, ValueRef::Local(s_dim), ValueRef::Local(s_dim_t));
    let s_gap = fixed_mul(builder, ValueRef::Local(s_sum), ValueRef::Local(s_diff));
    let s_gap_sq = fixed_mul(builder, ValueRef::Local(s_gap), ValueRef::Local(s_gap));
    let eps_sq = fixed_scale(
        builder,
        ValueRef::Local(ones_mat),
        eps * eps,
        matrix_shape(k.clone(), k.clone(), batch_shape),
    );
    let safe_gap = fixed_add(builder, ValueRef::Local(s_gap_sq), ValueRef::Local(eps_sq));
    let f = fixed_div(builder, ValueRef::Local(s_gap), ValueRef::Local(safe_gap));

    let du = if u_active {
        let s_ones = one_like_fixed(builder, s_dtype, s.clone(), matrix_rank - 1);
        let s_sq = fixed_mul(builder, s.clone(), s.clone());
        let s_eps_sq = fixed_scale(
            builder,
            ValueRef::Local(s_ones),
            eps * eps,
            vector_shape(k.clone(), batch_shape),
        );
        let safe_s_sq = fixed_add(builder, ValueRef::Local(s_sq), ValueRef::Local(s_eps_sq));
        let s_inv = fixed_div(builder, s.clone(), ValueRef::Local(safe_s_sq));
        let s_inv_mat = embed_diag_fixed(builder, ValueRef::Local(s_inv));

        let ds_h = adjoint_matrix_linear(builder, ds_mat, matrix_rank, dtype);
        let anti_hermitian = linear_sub(builder, ds_mat, ds_h);
        // Matches JAX's complex gauge correction: 0.5 * (dS - dS^H) * diag(1 / s).
        let d_udv_diag = hadamard_fixed_linear(builder, ValueRef::Local(s_inv_mat), anti_hermitian);
        let d_udv_diag = linear_scale(
            builder,
            d_udv_diag,
            0.5,
            matrix_shape(k.clone(), k.clone(), batch_shape),
        );

        let dss = hadamard_fixed_linear(builder, ValueRef::Local(s_dim), ds_mat);
        let dss_h = adjoint_matrix_linear(builder, dss, matrix_rank, dtype);
        let du_inner_sum = linear_add(builder, dss, dss_h);
        let du_inner = hadamard_fixed_linear(builder, ValueRef::Local(f), du_inner_sum);
        let du_inner = linear_add(builder, du_inner, d_udv_diag);
        let mut du = matmul_linear(
            builder,
            u.clone(),
            ValueRef::Local(du_inner),
            vec![false, true],
            matrix_rank,
        );

        if m_size > n_size {
            let d_av = matmul_linear(
                builder,
                ValueRef::Local(da),
                ValueRef::Local(v),
                vec![true, false],
                matrix_rank,
            );
            let ut_d_av = matmul_linear(
                builder,
                ValueRef::Local(uh),
                ValueRef::Local(d_av),
                vec![false, true],
                matrix_rank,
            );
            let u_ut_d_av = matmul_linear(
                builder,
                u.clone(),
                ValueRef::Local(ut_d_av),
                vec![false, true],
                matrix_rank,
            );
            let proj = linear_sub(builder, d_av, u_ut_d_av);
            let s_broadcast = broadcast_in_dim_fixed(
                builder,
                s.clone(),
                matrix_shape(m, &k, batch_shape),
                vector_to_matrix_broadcast_dims(batch_shape.len()),
            );
            let correction = linear_div_fixed(builder, proj, ValueRef::Local(s_broadcast));
            du = linear_add(builder, du, correction);
        }
        Some(du)
    } else {
        None
    };

    let dvt = if vt_active {
        let sds = hadamard_fixed_linear(builder, ValueRef::Local(s_dim_t), ds_mat);
        let sds_h = adjoint_matrix_linear(builder, sds, matrix_rank, dtype);
        let dv_inner_sum = linear_add(builder, sds, sds_h);
        let dv_inner = hadamard_fixed_linear(builder, ValueRef::Local(f), dv_inner_sum);
        let mut dv = matmul_linear(
            builder,
            ValueRef::Local(v),
            ValueRef::Local(dv_inner),
            vec![false, true],
            matrix_rank,
        );

        if n_size > m_size {
            let da_h = adjoint_matrix_linear(builder, da, matrix_rank, dtype);
            let d_ahu = matmul_linear(
                builder,
                ValueRef::Local(da_h),
                u.clone(),
                vec![true, false],
                matrix_rank,
            );
            let vt_d_ahu = matmul_linear(
                builder,
                vt.clone(),
                ValueRef::Local(d_ahu),
                vec![false, true],
                matrix_rank,
            );
            let v_vt_d_ahu = matmul_linear(
                builder,
                ValueRef::Local(v),
                ValueRef::Local(vt_d_ahu),
                vec![false, true],
                matrix_rank,
            );
            let proj = linear_sub(builder, d_ahu, v_vt_d_ahu);
            let s_broadcast = broadcast_in_dim_fixed(
                builder,
                s.clone(),
                matrix_shape(n, &k, batch_shape),
                vector_to_matrix_broadcast_dims(batch_shape.len()),
            );
            let correction = linear_div_fixed(builder, proj, ValueRef::Local(s_broadcast));
            dv = linear_add(builder, dv, correction);
        }

        Some(adjoint_matrix_linear(builder, dv, matrix_rank, dtype))
    } else {
        None
    };

    Ok(vec![du, ds, dvt])
}

pub(crate) fn linearize_svd_values(
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    eps: f64,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let Some(da) = tangent_in[0] else {
        return Ok(vec![None]);
    };

    let Some(input_shape) = primal_matrix_input_shape(ctx, primal_in)? else {
        return Ok(vec![None]);
    };
    let input_shape = input_shape.as_slice();
    let matrix_rank = input_shape.len();
    let dtype = ctx.dtype_of(&ValueRef::External(primal_in[0].clone()))?;
    let svd_outputs = builder.add_operation(
        linalg_std_op(LinalgOp::Svd {
            derivative_eps: eps,
            gauge: SvdGauge::Raw,
        }),
        vec![ValueRef::External(primal_in[0].clone())],
        OperationRole::Primary,
    );
    let u = ValueRef::Local(svd_outputs[0]);
    let vt = ValueRef::Local(svd_outputs[2]);
    let uh = adjoint_matrix_fixed(builder, u, matrix_rank, dtype);
    let v = adjoint_matrix_fixed(builder, vt, matrix_rank, dtype);
    let tmp = matmul_linear(
        builder,
        ValueRef::Local(uh),
        ValueRef::Local(da),
        vec![false, true],
        matrix_rank,
    );
    let ds_mat = matmul_linear(
        builder,
        ValueRef::Local(tmp),
        ValueRef::Local(v),
        vec![true, false],
        matrix_rank,
    );
    Ok(vec![Some(extract_diag_linear(builder, ds_mat))])
}

pub(crate) fn linearize_eigh(
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    eps: f64,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let Some(da) = tangent_in[0] else {
        return Ok(vec![None, None]);
    };

    let w_active = ctx.is_value_active_in_linearize(&primal_out[0]);
    let v_active = ctx.is_value_active_in_linearize(&primal_out[1]);
    if !w_active && !v_active {
        return Ok(vec![None, None]);
    }

    let Some(input_shape) = primal_matrix_input_shape(ctx, primal_in)? else {
        return Ok(vec![None, None]);
    };
    let input_shape = input_shape.as_slice();
    let (n, _, batch_shape) = matrix_shape_parts(input_shape, "linearize_eigh");
    let matrix_rank = input_shape.len();
    let dtype = ctx.dtype_of(&ValueRef::External(primal_in[0].clone()))?;
    let w = ValueRef::External(primal_out[0].clone());
    let w_dtype = ctx.dtype_of(&w)?;
    let v = ValueRef::External(primal_out[1].clone());
    let da_self_adjoint = self_adjoint_from_lower_linear(
        builder,
        da,
        matrix_rank,
        dtype,
        vector_shape(n.clone(), batch_shape),
    );

    let vh = adjoint_matrix_fixed(builder, v.clone(), matrix_rank, dtype);
    let tmp = matmul_linear(
        builder,
        ValueRef::Local(vh),
        ValueRef::Local(da_self_adjoint),
        vec![false, true],
        matrix_rank,
    );
    let projected = matmul_linear(
        builder,
        ValueRef::Local(tmp),
        v.clone(),
        vec![true, false],
        matrix_rank,
    );
    let dw = if w_active {
        let dw = extract_diag_linear(builder, projected);
        Some(convert_linear_to_dtype(builder, dw, dtype, w_dtype))
    } else {
        None
    };

    if !v_active {
        return Ok(vec![dw, None]);
    }

    let diag_w = embed_diag_fixed(builder, w.clone());
    let ones_mat = one_like_fixed(builder, w_dtype, ValueRef::Local(diag_w), matrix_rank);
    let w_col = matmul_fixed(
        builder,
        ValueRef::Local(diag_w),
        ValueRef::Local(ones_mat),
        matrix_rank,
    );
    let w_row = matmul_fixed(
        builder,
        ValueRef::Local(ones_mat),
        ValueRef::Local(diag_w),
        matrix_rank,
    );
    let diff = fixed_sub(builder, ValueRef::Local(w_row), ValueRef::Local(w_col));
    let diff_sq = fixed_mul(builder, ValueRef::Local(diff), ValueRef::Local(diff));
    let eps_sq = fixed_scale(
        builder,
        ValueRef::Local(ones_mat),
        eps * eps,
        matrix_shape(n.clone(), n.clone(), batch_shape),
    );
    let safe_diff = fixed_add(builder, ValueRef::Local(diff_sq), ValueRef::Local(eps_sq));
    let f = fixed_div(builder, ValueRef::Local(diff), ValueRef::Local(safe_diff));
    let f = convert_fixed_ref_to_dtype(builder, ValueRef::Local(f), w_dtype, dtype);
    let fm = hadamard_fixed_linear(builder, f, projected);
    let dv = matmul_linear(
        builder,
        v,
        ValueRef::Local(fm),
        vec![false, true],
        matrix_rank,
    );

    Ok(vec![dw, Some(dv)])
}

pub(crate) fn linearize_eigh_values(
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    eps: f64,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let Some(da) = tangent_in[0] else {
        return Ok(vec![None]);
    };

    let Some(input_shape) = primal_matrix_input_shape(ctx, primal_in)? else {
        return Ok(vec![None]);
    };
    let input_shape = input_shape.as_slice();
    let matrix_rank = input_shape.len();
    let (n, _, batch_shape) = matrix_shape_parts(input_shape, "linearize_eigh_values");
    let dtype = ctx.dtype_of(&ValueRef::External(primal_in[0].clone()))?;
    let eigh_outputs = builder.add_operation(
        linalg_std_op(LinalgOp::Eigh {
            derivative_eps: eps,
            gauge: EighGauge::Raw,
        }),
        vec![ValueRef::External(primal_in[0].clone())],
        OperationRole::Primary,
    );
    let v = ValueRef::Local(eigh_outputs[1]);
    let da_self_adjoint = self_adjoint_from_lower_linear(
        builder,
        da,
        matrix_rank,
        dtype,
        vector_shape(n.clone(), batch_shape),
    );
    let vh = adjoint_matrix_fixed(builder, v.clone(), matrix_rank, dtype);
    let tmp = matmul_linear(
        builder,
        ValueRef::Local(vh),
        ValueRef::Local(da_self_adjoint),
        vec![false, true],
        matrix_rank,
    );
    let projected = matmul_linear(
        builder,
        ValueRef::Local(tmp),
        v,
        vec![true, false],
        matrix_rank,
    );

    Ok(vec![Some(extract_diag_linear(builder, projected))])
}

pub(crate) fn linearize_cholesky(
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let Some(da) = tangent_in[0] else {
        return Ok(vec![None]);
    };

    let Some(input_shape) = primal_matrix_input_shape(ctx, primal_in)? else {
        return Ok(vec![None]);
    };
    let input_shape = input_shape.as_slice();
    let matrix_rank = input_shape.len();
    let (n, _, batch_shape) = matrix_shape_parts(input_shape, "linearize_cholesky");
    let dtype = ctx.dtype_of(&ValueRef::External(primal_in[0].clone()))?;
    let l = ValueRef::External(primal_out[0].clone());
    let da_self_adjoint = self_adjoint_from_lower_linear(
        builder,
        da,
        matrix_rank,
        dtype,
        vector_shape(n.clone(), batch_shape),
    );
    let l_conj = conjugate_primal_if_dtype_complex(builder, l.clone(), dtype);

    let tmp = builder.add_operation(
        linalg_std_op(LinalgOp::TriangularSolve {
            left_side: false,
            lower: true,
            transpose_a: true,
            unit_diagonal: false,
        }),
        vec![l_conj, ValueRef::Local(da_self_adjoint)],
        OperationRole::Linearized {
            active_mask: vec![false, true],
        },
    )[0];
    let s = builder.add_operation(
        linalg_std_op(LinalgOp::TriangularSolve {
            left_side: true,
            lower: true,
            transpose_a: false,
            unit_diagonal: false,
        }),
        vec![l.clone(), ValueRef::Local(tmp)],
        OperationRole::Linearized {
            active_mask: vec![false, true],
        },
    )[0];

    let strict_lower = builder.add_operation(
        StdTensorOp::Tril { k: -1 },
        vec![ValueRef::Local(s)],
        OperationRole::Linearized {
            active_mask: vec![true],
        },
    )[0];
    let diag_s = extract_diag_linear(builder, s);
    let half_diag = linear_scale(builder, diag_s, 0.5, vector_shape(n.clone(), batch_shape));
    let half_diag_mat = embed_diag_linear(builder, half_diag);
    let phi_s = linear_add(builder, strict_lower, half_diag_mat);
    let dl = matmul_linear(
        builder,
        l,
        ValueRef::Local(phi_s),
        vec![false, true],
        matrix_rank,
    );

    Ok(vec![Some(dl)])
}

pub(crate) fn linearize_qr(
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let Some(da) = tangent_in[0] else {
        return Ok(vec![None, None]);
    };

    let q_active = ctx.is_value_active_in_linearize(&primal_out[0]);
    let r_active = ctx.is_value_active_in_linearize(&primal_out[1]);
    if !q_active && !r_active {
        return Ok(vec![None, None]);
    }

    let Some(input_shape) = primal_matrix_input_shape(ctx, primal_in)? else {
        return Ok(vec![None, None]);
    };
    let input_shape = input_shape.as_slice();
    let (m, n, batch_shape) = matrix_shape_parts(input_shape, "linearize_qr");
    let (m_size, n_size) =
        resolve_and_guard(m, n, ctx).map_err(|err| invalid_dim_expr("linearize_qr", err))?;
    let matrix_rank = input_shape.len();
    let dtype = ctx.dtype_of(&ValueRef::External(primal_in[0].clone()))?;
    let q = ValueRef::External(primal_out[0].clone());
    let r = ValueRef::External(primal_out[1].clone());

    if n_size > m_size {
        let qh = adjoint_matrix_fixed(builder, q.clone(), matrix_rank, dtype);
        let leading_selector =
            leading_column_selector_fixed(builder, dtype, m_size, n_size, batch_shape, r.clone());
        let leading_selector_t =
            transpose_matrix_fixed(builder, ValueRef::Local(leading_selector), matrix_rank);
        let da_leading = matmul_linear(
            builder,
            ValueRef::Local(da),
            ValueRef::Local(leading_selector_t),
            vec![true, false],
            matrix_rank,
        );
        let r_leading = matmul_fixed(
            builder,
            r.clone(),
            ValueRef::Local(leading_selector_t),
            matrix_rank,
        );
        let r_leading_h =
            adjoint_matrix_fixed(builder, ValueRef::Local(r_leading), matrix_rank, dtype);
        let da_leading_h = adjoint_matrix_linear(builder, da_leading, matrix_rank, dtype);
        let dx_rinv_h = builder.add_operation(
            linalg_std_op(LinalgOp::TriangularSolve {
                left_side: true,
                lower: true,
                transpose_a: false,
                unit_diagonal: false,
            }),
            vec![ValueRef::Local(r_leading_h), ValueRef::Local(da_leading_h)],
            OperationRole::Linearized {
                active_mask: vec![false, true],
            },
        )[0];
        let dx_rinv = adjoint_matrix_linear(builder, dx_rinv_h, matrix_rank, dtype);
        let qt_dx_rinv = matmul_linear(
            builder,
            ValueRef::Local(qh),
            ValueRef::Local(dx_rinv),
            vec![false, true],
            matrix_rank,
        );
        let qt_dx_rinv_h = adjoint_matrix_linear(builder, qt_dx_rinv, matrix_rank, dtype);
        let sym = linear_add(builder, qt_dx_rinv, qt_dx_rinv_h);
        let upper = builder.add_operation(
            StdTensorOp::Triu { k: 1 },
            vec![ValueRef::Local(sym)],
            OperationRole::Linearized {
                active_mask: vec![true],
            },
        )[0];
        let sym_diag = extract_diag_linear(builder, sym);
        let half_sym_diag =
            linear_scale(builder, sym_diag, 0.5, vector_shape(m.clone(), batch_shape));
        let half_sym_diag_mat = embed_diag_linear(builder, half_sym_diag);
        let dr_leading_hat = linear_add(builder, upper, half_sym_diag_mat);

        let dq = if q_active {
            let q_dr_leading_hat = matmul_linear(
                builder,
                q.clone(),
                ValueRef::Local(dr_leading_hat),
                vec![false, true],
                matrix_rank,
            );
            Some(linear_sub(builder, dx_rinv, q_dr_leading_hat))
        } else {
            None
        };

        let mut dr = if r_active {
            let dr_leading = matmul_linear(
                builder,
                ValueRef::Local(dr_leading_hat),
                ValueRef::Local(r_leading),
                vec![true, false],
                matrix_rank,
            );
            Some(matmul_linear(
                builder,
                ValueRef::Local(dr_leading),
                ValueRef::Local(leading_selector),
                vec![true, false],
                matrix_rank,
            ))
        } else {
            None
        };

        let trailing_cols = n_size - m_size;
        if r_active && trailing_cols > 0 {
            let trailing_selector = trailing_column_selector_fixed(
                builder,
                dtype,
                m_size,
                trailing_cols,
                batch_shape,
                r.clone(),
            );
            let trailing_selector_t =
                transpose_matrix_fixed(builder, ValueRef::Local(trailing_selector), matrix_rank);
            let da_trailing = matmul_linear(
                builder,
                ValueRef::Local(da),
                ValueRef::Local(trailing_selector_t),
                vec![true, false],
                matrix_rank,
            );
            let r_trailing = matmul_fixed(
                builder,
                r.clone(),
                ValueRef::Local(trailing_selector_t),
                matrix_rank,
            );
            let qt_da_trailing = matmul_linear(
                builder,
                ValueRef::Local(qh),
                ValueRef::Local(da_trailing),
                vec![false, true],
                matrix_rank,
            );
            let omega = linear_sub(builder, qt_dx_rinv, dr_leading_hat);
            let omega_r_trailing = matmul_linear(
                builder,
                ValueRef::Local(omega),
                ValueRef::Local(r_trailing),
                vec![true, false],
                matrix_rank,
            );
            let dr_trailing = linear_sub(builder, qt_da_trailing, omega_r_trailing);
            let dr_trailing_full = matmul_linear(
                builder,
                ValueRef::Local(dr_trailing),
                ValueRef::Local(trailing_selector),
                vec![true, false],
                matrix_rank,
            );
            if let Some(current_dr) = dr {
                dr = Some(linear_add(builder, current_dr, dr_trailing_full));
            }
        }

        return Ok(vec![dq, dr]);
    }

    let r_h = adjoint_matrix_fixed(builder, r.clone(), matrix_rank, dtype);
    let da_h = adjoint_matrix_linear(builder, da, matrix_rank, dtype);
    let dx_rinv_h = builder.add_operation(
        linalg_std_op(LinalgOp::TriangularSolve {
            left_side: true,
            lower: true,
            transpose_a: false,
            unit_diagonal: false,
        }),
        vec![ValueRef::Local(r_h), ValueRef::Local(da_h)],
        OperationRole::Linearized {
            active_mask: vec![false, true],
        },
    )[0];
    let dx_rinv = adjoint_matrix_linear(builder, dx_rinv_h, matrix_rank, dtype);
    let qh = adjoint_matrix_fixed(builder, q.clone(), matrix_rank, dtype);
    let qt_dx_rinv = matmul_linear(
        builder,
        ValueRef::Local(qh),
        ValueRef::Local(dx_rinv),
        vec![false, true],
        matrix_rank,
    );
    let qt_dx_rinv_h = adjoint_matrix_linear(builder, qt_dx_rinv, matrix_rank, dtype);
    let sym = linear_add(builder, qt_dx_rinv, qt_dx_rinv_h);
    let upper = builder.add_operation(
        StdTensorOp::Triu { k: 1 },
        vec![ValueRef::Local(sym)],
        OperationRole::Linearized {
            active_mask: vec![true],
        },
    )[0];
    let sym_diag = extract_diag_linear(builder, sym);
    let half_sym_diag = linear_scale(builder, sym_diag, 0.5, vector_shape(n.clone(), batch_shape));
    let half_sym_diag_mat = embed_diag_linear(builder, half_sym_diag);
    let dr_hat = linear_add(builder, upper, half_sym_diag_mat);

    let dq = if q_active {
        let q_dr_hat = matmul_linear(
            builder,
            q.clone(),
            ValueRef::Local(dr_hat),
            vec![false, true],
            matrix_rank,
        );
        Some(linear_sub(builder, dx_rinv, q_dr_hat))
    } else {
        None
    };
    let dr = if r_active {
        Some(matmul_linear(
            builder,
            ValueRef::Local(dr_hat),
            r,
            vec![true, false],
            matrix_rank,
        ))
    } else {
        None
    };

    Ok(vec![dq, dr])
}
