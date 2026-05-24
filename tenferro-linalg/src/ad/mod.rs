use std::sync::Arc;

use computegraph::fragment::FragmentBuilder;
use computegraph::types::{GlobalValKey, LocalValId, OpMode, ValRef};
use computegraph::OpEmitter;
use tenferro_tensor::DType;

use crate::extension::{LinalgExtensionOp, LinalgOp};
use tenferro_ops::ad::context::{resolve_and_guard, ShapeGuardContext};
use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::std_tensor_op::StdTensorOp;

mod solve;
mod support;

pub(crate) use solve::{
    linearize_full_piv_lu_solve, linearize_solve, linearize_triangular_solve,
    transpose_full_piv_lu_solve, transpose_solve, transpose_triangular_solve,
};
use support::*;

/// Derive an op-local `Vec<DimExpr>` for the primal input of a 1-input linalg op.
///
/// Category C `input_shape` snapshots have been removed from the linalg op
/// variants; instead, AD rules resolve the primal input's shape through the
/// [`ShapeGuardContext`] metadata surface. Concrete shapes collapse to
/// `DimExpr::Const`; symbolic shapes fall back to `DimExpr::input_shape(0, rank)`
/// so downstream op constructors that still carry shape fields (e.g.
/// `TriangularSolve`) see a well-formed shape expression.
fn primal_input_shape(
    ctx: &mut ShapeGuardContext,
    primal_in: &[GlobalValKey<StdTensorOp>],
) -> Vec<DimExpr> {
    let shape = ctx.shape_of(&ValRef::External(primal_in[0].clone()));
    if let Some(concrete) = shape
        .iter()
        .map(|dim| dim.constant_value())
        .collect::<Option<Vec<_>>>()
    {
        DimExpr::from_concrete(&concrete)
    } else {
        DimExpr::input_shape(0, shape.len())
    }
}

fn linalg_std_op(op: LinalgOp) -> StdTensorOp {
    StdTensorOp::Extension(Arc::new(LinalgExtensionOp::new(op)))
}

fn is_real_dtype(dtype: DType) -> bool {
    matches!(dtype, DType::F32 | DType::F64)
}

fn conjugate_primal_if_dtype_complex(
    emitter: &mut impl OpEmitter<StdTensorOp>,
    input: ValRef<StdTensorOp>,
    dtype: DType,
) -> ValRef<StdTensorOp> {
    if is_real_dtype(dtype) {
        input
    } else {
        ValRef::Local(emitter.add_op(StdTensorOp::Conj, vec![input], OpMode::Primal)[0])
    }
}

fn conjugate_linear_if_dtype_complex(
    emitter: &mut impl OpEmitter<StdTensorOp>,
    input: LocalValId,
    dtype: DType,
) -> LocalValId {
    if is_real_dtype(dtype) {
        input
    } else {
        emitter.add_op(
            StdTensorOp::Conj,
            vec![ValRef::Local(input)],
            OpMode::Linear {
                active_mask: vec![true],
            },
        )[0]
    }
}

pub(crate) fn linearize_lu(
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    primal_out: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    ctx: &mut ShapeGuardContext,
) -> Vec<Option<LocalValId>> {
    let Some(da) = tangent_in[0] else {
        return vec![None, None, None, None];
    };

    let input_shape = primal_input_shape(ctx, primal_in);
    let input_shape = input_shape.as_slice();
    let (m, n, batch_shape) = matrix_shape_parts(input_shape, "linearize_lu");
    let (m_size, n_size) = resolve_and_guard(m, n, ctx);
    let k = DimExpr::min(m.clone(), n.clone());
    let k_size = m_size.min(n_size);
    let rank = input_shape.len();
    let l_shape = matrix_shape(m, &k, batch_shape);
    let u_shape = matrix_shape(&k, n, batch_shape);
    let p = ValRef::External(primal_out[0].clone());
    let l = ValRef::External(primal_out[1].clone());
    let u = ValRef::External(primal_out[2].clone());
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

    let pd_a = matmul_linear(builder, p, ValRef::Local(da), vec![false, true], rank);
    let la = builder.add_op(
        linalg_std_op(LinalgOp::TriangularSolve {
            left_side: true,
            lower: true,
            transpose_a: false,
            unit_diagonal: true,
        }),
        vec![ValRef::Local(l_square), ValRef::Local(pd_a)],
        OpMode::Linear {
            active_mask: vec![false, true],
        },
    )[0];
    let x = builder.add_op(
        linalg_std_op(LinalgOp::TriangularSolve {
            left_side: false,
            lower: false,
            transpose_a: false,
            unit_diagonal: false,
        }),
        vec![ValRef::Local(u_square), ValRef::Local(la)],
        OpMode::Linear {
            active_mask: vec![false, true],
        },
    )[0];

    let x_lower = linear_unary(builder, StdTensorOp::Tril { k: -1 }, x);
    let x_upper = linear_unary(builder, StdTensorOp::Triu { k: 0 }, x);
    let dl_full = matmul_linear(
        builder,
        ValRef::Local(l_square),
        ValRef::Local(x_lower),
        vec![false, true],
        rank,
    );
    let du_full = matmul_linear(
        builder,
        ValRef::Local(x_upper),
        ValRef::Local(u_square),
        vec![true, false],
        rank,
    );
    let dl = if n_size > k_size {
        take_leading_cols_linear(
            builder,
            dl_full,
            k_size,
            n_size,
            batch_shape,
            l.clone(),
            &l_shape,
            rank,
        )
    } else {
        dl_full
    };
    let du = if m_size > k_size {
        take_leading_rows_linear(
            builder,
            du_full,
            k_size,
            m_size,
            batch_shape,
            u.clone(),
            &u_shape,
            rank,
        )
    } else {
        du_full
    };

    vec![None, Some(dl), Some(du), None]
}

pub(crate) fn linearize_full_piv_lu(
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    primal_out: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    ctx: &mut ShapeGuardContext,
) -> Vec<Option<LocalValId>> {
    let Some(da) = tangent_in[0] else {
        return vec![None, None, None, None, None];
    };

    let input_shape = primal_input_shape(ctx, primal_in);
    let input_shape = input_shape.as_slice();
    let (rows, cols, _batch_shape) = matrix_shape_parts(input_shape, "linearize_full_piv_lu");
    let (rows_size, cols_size) = resolve_and_guard(rows, cols, ctx);
    assert_eq!(
        rows_size, cols_size,
        "linearize_full_piv_lu: expected square matrix"
    );

    let rank = input_shape.len();
    let p = ValRef::External(primal_out[0].clone());
    let l = ValRef::External(primal_out[1].clone());
    let u = ValRef::External(primal_out[2].clone());
    let q = ValRef::External(primal_out[3].clone());
    let q_t = transpose_matrix_fixed(builder, q, rank);

    let pd_a = matmul_linear(builder, p, ValRef::Local(da), vec![false, true], rank);
    let pd_a_qt = matmul_linear(
        builder,
        ValRef::Local(pd_a),
        ValRef::Local(q_t),
        vec![true, false],
        rank,
    );
    let la = builder.add_op(
        linalg_std_op(LinalgOp::TriangularSolve {
            left_side: true,
            lower: true,
            transpose_a: false,
            unit_diagonal: true,
        }),
        vec![l.clone(), ValRef::Local(pd_a_qt)],
        OpMode::Linear {
            active_mask: vec![false, true],
        },
    )[0];
    let x = builder.add_op(
        linalg_std_op(LinalgOp::TriangularSolve {
            left_side: false,
            lower: false,
            transpose_a: false,
            unit_diagonal: false,
        }),
        vec![u.clone(), ValRef::Local(la)],
        OpMode::Linear {
            active_mask: vec![false, true],
        },
    )[0];

    let x_lower = linear_unary(builder, StdTensorOp::Tril { k: -1 }, x);
    let x_upper = linear_unary(builder, StdTensorOp::Triu { k: 0 }, x);
    let dl = matmul_linear(builder, l, ValRef::Local(x_lower), vec![false, true], rank);
    let du = matmul_linear(builder, ValRef::Local(x_upper), u, vec![true, false], rank);

    vec![None, Some(dl), Some(du), None, None]
}

pub(crate) fn linearize_eig(
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    primal_out: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    input_dtype: DType,
    ctx: &mut ShapeGuardContext,
) -> Vec<Option<LocalValId>> {
    let Some(da) = tangent_in[0] else {
        return vec![None, None];
    };

    let input_shape = primal_input_shape(ctx, primal_in);
    let input_shape = input_shape.as_slice();
    let rank = input_shape.len();
    let v = ValRef::External(primal_out[1].clone());
    let da_complex = match input_dtype {
        DType::F64 | DType::F32 => builder.add_op(
            StdTensorOp::Convert {
                from: input_dtype,
                to: match input_dtype {
                    DType::F64 => DType::C64,
                    DType::F32 => DType::C32,
                    _ => unreachable!("real dtype branch"),
                },
            },
            vec![ValRef::Local(da)],
            OpMode::Linear {
                active_mask: vec![true],
            },
        )[0],
        DType::C64 | DType::C32 => da,
        DType::I32 | DType::I64 | DType::Bool => return vec![None, None],
    };

    let dav = matmul_linear(
        builder,
        ValRef::Local(da_complex),
        v.clone(),
        vec![true, false],
        rank,
    );
    let projected = solve_in_graph(builder, v, ValRef::Local(dav), rank);
    let dw = extract_diag_linear(builder, projected);
    vec![Some(dw), None]
}

pub(crate) fn linearize_svd(
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    primal_out: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    eps: f64,
    ctx: &mut ShapeGuardContext,
) -> Vec<Option<LocalValId>> {
    let Some(da) = tangent_in[0] else {
        return vec![None, None, None];
    };

    let input_shape = primal_input_shape(ctx, primal_in);
    let input_shape = input_shape.as_slice();
    let (m, n, batch_shape) = matrix_shape_parts(input_shape, "linearize_svd");
    let (m_size, n_size) = resolve_and_guard(m, n, ctx);
    let matrix_rank = input_shape.len();
    let dtype = ctx.dtype_of(&ValRef::External(primal_in[0].clone()));
    let k = DimExpr::min(m.clone(), n.clone());
    let u = ValRef::External(primal_out[0].clone());
    let s = ValRef::External(primal_out[1].clone());
    let vt = ValRef::External(primal_out[2].clone());

    let uh = adjoint_matrix_fixed(builder, u.clone(), matrix_rank, dtype);
    let v = adjoint_matrix_fixed(builder, vt.clone(), matrix_rank, dtype);
    let tmp = matmul_linear(
        builder,
        ValRef::Local(uh),
        ValRef::Local(da),
        vec![false, true],
        matrix_rank,
    );
    let ds_mat = matmul_linear(
        builder,
        ValRef::Local(tmp),
        ValRef::Local(v),
        vec![true, false],
        matrix_rank,
    );
    let ds = extract_diag_linear(builder, ds_mat);

    let diag_s = embed_diag_fixed(builder, s.clone());
    let ones_mat = one_like_fixed(builder, ValRef::Local(diag_s));
    let s_dim = matmul_fixed(
        builder,
        ValRef::Local(ones_mat),
        ValRef::Local(diag_s),
        matrix_rank,
    );
    let s_dim_t = matmul_fixed(
        builder,
        ValRef::Local(diag_s),
        ValRef::Local(ones_mat),
        matrix_rank,
    );
    let s_sum = fixed_add(builder, ValRef::Local(s_dim), ValRef::Local(s_dim_t));
    let s_diff = fixed_sub(builder, ValRef::Local(s_dim), ValRef::Local(s_dim_t));
    let s_gap = fixed_mul(builder, ValRef::Local(s_sum), ValRef::Local(s_diff));
    let s_gap_sq = fixed_mul(builder, ValRef::Local(s_gap), ValRef::Local(s_gap));
    let eps_sq = fixed_scale(builder, ValRef::Local(ones_mat), eps * eps);
    let safe_gap = fixed_add(builder, ValRef::Local(s_gap_sq), ValRef::Local(eps_sq));
    let f = fixed_div(builder, ValRef::Local(s_gap), ValRef::Local(safe_gap));

    let s_ones = one_like_fixed(builder, s.clone());
    let s_sq = fixed_mul(builder, s.clone(), s.clone());
    let s_eps_sq = fixed_scale(builder, ValRef::Local(s_ones), eps * eps);
    let safe_s_sq = fixed_add(builder, ValRef::Local(s_sq), ValRef::Local(s_eps_sq));
    let s_inv = fixed_div(builder, s.clone(), ValRef::Local(safe_s_sq));
    let s_inv_mat = embed_diag_fixed(builder, ValRef::Local(s_inv));

    let ds_h = adjoint_matrix_linear(builder, ds_mat, matrix_rank, dtype);
    let anti_hermitian = linear_sub(builder, ds_mat, ds_h);
    // Matches JAX's complex gauge correction: 0.5 * (dS - dS^H) * diag(1 / s).
    let d_udv_diag = hadamard_fixed_linear(builder, ValRef::Local(s_inv_mat), anti_hermitian);
    let d_udv_diag = linear_scale(builder, d_udv_diag, 0.5);

    let dss = hadamard_fixed_linear(builder, ValRef::Local(s_dim), ds_mat);
    let dss_h = adjoint_matrix_linear(builder, dss, matrix_rank, dtype);
    let du_inner_sum = linear_add(builder, dss, dss_h);
    let du_inner = hadamard_fixed_linear(builder, ValRef::Local(f), du_inner_sum);
    let du_inner = linear_add(builder, du_inner, d_udv_diag);
    let mut du = matmul_linear(
        builder,
        u.clone(),
        ValRef::Local(du_inner),
        vec![false, true],
        matrix_rank,
    );

    let sds = hadamard_fixed_linear(builder, ValRef::Local(s_dim_t), ds_mat);
    let sds_h = adjoint_matrix_linear(builder, sds, matrix_rank, dtype);
    let dv_inner_sum = linear_add(builder, sds, sds_h);
    let dv_inner = hadamard_fixed_linear(builder, ValRef::Local(f), dv_inner_sum);
    let mut dv = matmul_linear(
        builder,
        ValRef::Local(v),
        ValRef::Local(dv_inner),
        vec![false, true],
        matrix_rank,
    );

    if m_size > n_size {
        let d_av = matmul_linear(
            builder,
            ValRef::Local(da),
            ValRef::Local(v),
            vec![true, false],
            matrix_rank,
        );
        let ut_d_av = matmul_linear(
            builder,
            ValRef::Local(uh),
            ValRef::Local(d_av),
            vec![false, true],
            matrix_rank,
        );
        let u_ut_d_av = matmul_linear(
            builder,
            u.clone(),
            ValRef::Local(ut_d_av),
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
        let correction = linear_div_fixed(builder, proj, ValRef::Local(s_broadcast));
        du = linear_add(builder, du, correction);
    }

    if n_size > m_size {
        let da_h = adjoint_matrix_linear(builder, da, matrix_rank, dtype);
        let d_ahu = matmul_linear(
            builder,
            ValRef::Local(da_h),
            u.clone(),
            vec![true, false],
            matrix_rank,
        );
        let vt_d_ahu = matmul_linear(
            builder,
            vt.clone(),
            ValRef::Local(d_ahu),
            vec![false, true],
            matrix_rank,
        );
        let v_vt_d_ahu = matmul_linear(
            builder,
            ValRef::Local(v),
            ValRef::Local(vt_d_ahu),
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
        let correction = linear_div_fixed(builder, proj, ValRef::Local(s_broadcast));
        dv = linear_add(builder, dv, correction);
    }

    let dvt = adjoint_matrix_linear(builder, dv, matrix_rank, dtype);

    vec![Some(du), Some(ds), Some(dvt)]
}

pub(crate) fn linearize_eigh(
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    primal_out: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    eps: f64,
    ctx: &mut ShapeGuardContext,
) -> Vec<Option<LocalValId>> {
    let Some(da) = tangent_in[0] else {
        return vec![None, None];
    };

    let input_shape = primal_input_shape(ctx, primal_in);
    let input_shape = input_shape.as_slice();
    let matrix_rank = input_shape.len();
    let dtype = ctx.dtype_of(&ValRef::External(primal_in[0].clone()));
    let w = ValRef::External(primal_out[0].clone());
    let v = ValRef::External(primal_out[1].clone());
    let da_self_adjoint = self_adjoint_from_lower_linear(builder, da, matrix_rank, dtype);

    let vh = adjoint_matrix_fixed(builder, v.clone(), matrix_rank, dtype);
    let tmp = matmul_linear(
        builder,
        ValRef::Local(vh),
        ValRef::Local(da_self_adjoint),
        vec![false, true],
        matrix_rank,
    );
    let projected = matmul_linear(
        builder,
        ValRef::Local(tmp),
        v.clone(),
        vec![true, false],
        matrix_rank,
    );
    let dw = extract_diag_linear(builder, projected);

    let diag_w = embed_diag_fixed(builder, w.clone());
    let ones_mat = one_like_fixed(builder, ValRef::Local(diag_w));
    let w_col = matmul_fixed(
        builder,
        ValRef::Local(diag_w),
        ValRef::Local(ones_mat),
        matrix_rank,
    );
    let w_row = matmul_fixed(
        builder,
        ValRef::Local(ones_mat),
        ValRef::Local(diag_w),
        matrix_rank,
    );
    let diff = fixed_sub(builder, ValRef::Local(w_row), ValRef::Local(w_col));
    let diff_sq = fixed_mul(builder, ValRef::Local(diff), ValRef::Local(diff));
    let eps_sq = fixed_scale(builder, ValRef::Local(ones_mat), eps * eps);
    let safe_diff = fixed_add(builder, ValRef::Local(diff_sq), ValRef::Local(eps_sq));
    let f = fixed_div(builder, ValRef::Local(diff), ValRef::Local(safe_diff));
    let fm = hadamard_fixed_linear(builder, ValRef::Local(f), projected);
    let dv = matmul_linear(
        builder,
        v,
        ValRef::Local(fm),
        vec![false, true],
        matrix_rank,
    );

    vec![Some(dw), Some(dv)]
}

pub(crate) fn linearize_cholesky(
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    primal_out: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    ctx: &mut ShapeGuardContext,
) -> Vec<Option<LocalValId>> {
    let Some(da) = tangent_in[0] else {
        return vec![None];
    };

    let input_shape = primal_input_shape(ctx, primal_in);
    let input_shape = input_shape.as_slice();
    let matrix_rank = input_shape.len();
    let dtype = ctx.dtype_of(&ValRef::External(primal_in[0].clone()));
    let l = ValRef::External(primal_out[0].clone());
    let da_self_adjoint = self_adjoint_from_lower_linear(builder, da, matrix_rank, dtype);
    let l_conj = conjugate_primal_if_dtype_complex(builder, l.clone(), dtype);

    let tmp = builder.add_op(
        linalg_std_op(LinalgOp::TriangularSolve {
            left_side: false,
            lower: true,
            transpose_a: true,
            unit_diagonal: false,
        }),
        vec![l_conj, ValRef::Local(da_self_adjoint)],
        OpMode::Linear {
            active_mask: vec![false, true],
        },
    )[0];
    let s = builder.add_op(
        linalg_std_op(LinalgOp::TriangularSolve {
            left_side: true,
            lower: true,
            transpose_a: false,
            unit_diagonal: false,
        }),
        vec![l.clone(), ValRef::Local(tmp)],
        OpMode::Linear {
            active_mask: vec![false, true],
        },
    )[0];

    let strict_lower = builder.add_op(
        StdTensorOp::Tril { k: -1 },
        vec![ValRef::Local(s)],
        OpMode::Linear {
            active_mask: vec![true],
        },
    )[0];
    let diag_s = extract_diag_linear(builder, s);
    let half_diag = linear_scale(builder, diag_s, 0.5);
    let half_diag_mat = embed_diag_linear(builder, half_diag);
    let phi_s = linear_add(builder, strict_lower, half_diag_mat);
    let dl = matmul_linear(
        builder,
        l,
        ValRef::Local(phi_s),
        vec![false, true],
        matrix_rank,
    );

    vec![Some(dl)]
}

pub(crate) fn linearize_qr(
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    primal_out: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    ctx: &mut ShapeGuardContext,
) -> Vec<Option<LocalValId>> {
    let Some(da) = tangent_in[0] else {
        return vec![None, None];
    };

    let input_shape = primal_input_shape(ctx, primal_in);
    let input_shape = input_shape.as_slice();
    let (m, n, batch_shape) = matrix_shape_parts(input_shape, "linearize_qr");
    let (m_size, n_size) = resolve_and_guard(m, n, ctx);
    let matrix_rank = input_shape.len();
    let dtype = ctx.dtype_of(&ValRef::External(primal_in[0].clone()));
    let q = ValRef::External(primal_out[0].clone());
    let r = ValRef::External(primal_out[1].clone());

    if n_size > m_size {
        let qh = adjoint_matrix_fixed(builder, q.clone(), matrix_rank, dtype);
        let leading_selector = leading_column_selector_fixed(
            builder,
            m_size,
            n_size,
            batch_shape,
            r.clone(),
            input_shape,
        );
        let leading_selector_t =
            transpose_matrix_fixed(builder, ValRef::Local(leading_selector), matrix_rank);
        let da_leading = matmul_linear(
            builder,
            ValRef::Local(da),
            ValRef::Local(leading_selector_t),
            vec![true, false],
            matrix_rank,
        );
        let r_leading = matmul_fixed(
            builder,
            r.clone(),
            ValRef::Local(leading_selector_t),
            matrix_rank,
        );
        let r_leading_h =
            adjoint_matrix_fixed(builder, ValRef::Local(r_leading), matrix_rank, dtype);
        let da_leading_h = adjoint_matrix_linear(builder, da_leading, matrix_rank, dtype);
        let dx_rinv_h = builder.add_op(
            linalg_std_op(LinalgOp::TriangularSolve {
                left_side: true,
                lower: true,
                transpose_a: false,
                unit_diagonal: false,
            }),
            vec![ValRef::Local(r_leading_h), ValRef::Local(da_leading_h)],
            OpMode::Linear {
                active_mask: vec![false, true],
            },
        )[0];
        let dx_rinv = adjoint_matrix_linear(builder, dx_rinv_h, matrix_rank, dtype);
        let qt_dx_rinv = matmul_linear(
            builder,
            ValRef::Local(qh),
            ValRef::Local(dx_rinv),
            vec![false, true],
            matrix_rank,
        );
        let qt_dx_rinv_h = adjoint_matrix_linear(builder, qt_dx_rinv, matrix_rank, dtype);
        let sym = linear_add(builder, qt_dx_rinv, qt_dx_rinv_h);
        let upper = builder.add_op(
            StdTensorOp::Triu { k: 1 },
            vec![ValRef::Local(sym)],
            OpMode::Linear {
                active_mask: vec![true],
            },
        )[0];
        let sym_diag = extract_diag_linear(builder, sym);
        let half_sym_diag = linear_scale(builder, sym_diag, 0.5);
        let half_sym_diag_mat = embed_diag_linear(builder, half_sym_diag);
        let dr_leading_hat = linear_add(builder, upper, half_sym_diag_mat);

        let q_dr_leading_hat = matmul_linear(
            builder,
            q.clone(),
            ValRef::Local(dr_leading_hat),
            vec![false, true],
            matrix_rank,
        );
        let dq = linear_sub(builder, dx_rinv, q_dr_leading_hat);
        let dr_leading = matmul_linear(
            builder,
            ValRef::Local(dr_leading_hat),
            ValRef::Local(r_leading),
            vec![true, false],
            matrix_rank,
        );
        let mut dr = matmul_linear(
            builder,
            ValRef::Local(dr_leading),
            ValRef::Local(leading_selector),
            vec![true, false],
            matrix_rank,
        );

        let trailing_cols = n_size - m_size;
        if trailing_cols > 0 {
            let trailing_selector = trailing_column_selector_fixed(
                builder,
                m_size,
                trailing_cols,
                batch_shape,
                r.clone(),
                input_shape,
            );
            let trailing_selector_t =
                transpose_matrix_fixed(builder, ValRef::Local(trailing_selector), matrix_rank);
            let da_trailing = matmul_linear(
                builder,
                ValRef::Local(da),
                ValRef::Local(trailing_selector_t),
                vec![true, false],
                matrix_rank,
            );
            let r_trailing = matmul_fixed(
                builder,
                r.clone(),
                ValRef::Local(trailing_selector_t),
                matrix_rank,
            );
            let qt_da_trailing = matmul_linear(
                builder,
                ValRef::Local(qh),
                ValRef::Local(da_trailing),
                vec![false, true],
                matrix_rank,
            );
            let omega = linear_sub(builder, qt_dx_rinv, dr_leading_hat);
            let omega_r_trailing = matmul_linear(
                builder,
                ValRef::Local(omega),
                ValRef::Local(r_trailing),
                vec![true, false],
                matrix_rank,
            );
            let dr_trailing = linear_sub(builder, qt_da_trailing, omega_r_trailing);
            let dr_trailing_full = matmul_linear(
                builder,
                ValRef::Local(dr_trailing),
                ValRef::Local(trailing_selector),
                vec![true, false],
                matrix_rank,
            );
            dr = linear_add(builder, dr, dr_trailing_full);
        }

        return vec![Some(dq), Some(dr)];
    }

    let r_h = adjoint_matrix_fixed(builder, r.clone(), matrix_rank, dtype);
    let da_h = adjoint_matrix_linear(builder, da, matrix_rank, dtype);
    let dx_rinv_h = builder.add_op(
        linalg_std_op(LinalgOp::TriangularSolve {
            left_side: true,
            lower: true,
            transpose_a: false,
            unit_diagonal: false,
        }),
        vec![ValRef::Local(r_h), ValRef::Local(da_h)],
        OpMode::Linear {
            active_mask: vec![false, true],
        },
    )[0];
    let dx_rinv = adjoint_matrix_linear(builder, dx_rinv_h, matrix_rank, dtype);
    let qh = adjoint_matrix_fixed(builder, q.clone(), matrix_rank, dtype);
    let qt_dx_rinv = matmul_linear(
        builder,
        ValRef::Local(qh),
        ValRef::Local(dx_rinv),
        vec![false, true],
        matrix_rank,
    );
    let qt_dx_rinv_h = adjoint_matrix_linear(builder, qt_dx_rinv, matrix_rank, dtype);
    let sym = linear_add(builder, qt_dx_rinv, qt_dx_rinv_h);
    let upper = builder.add_op(
        StdTensorOp::Triu { k: 1 },
        vec![ValRef::Local(sym)],
        OpMode::Linear {
            active_mask: vec![true],
        },
    )[0];
    let sym_diag = extract_diag_linear(builder, sym);
    let half_sym_diag = linear_scale(builder, sym_diag, 0.5);
    let half_sym_diag_mat = embed_diag_linear(builder, half_sym_diag);
    let dr_hat = linear_add(builder, upper, half_sym_diag_mat);

    let q_dr_hat = matmul_linear(
        builder,
        q.clone(),
        ValRef::Local(dr_hat),
        vec![false, true],
        matrix_rank,
    );
    let dq = linear_sub(builder, dx_rinv, q_dr_hat);
    let dr = matmul_linear(
        builder,
        ValRef::Local(dr_hat),
        r,
        vec![true, false],
        matrix_rank,
    );

    vec![Some(dq), Some(dr)]
}
