use computegraph::fragment::FragmentBuilder;
use computegraph::types::{GlobalValKey, LocalValId, OpMode, ValRef};
use computegraph::OpEmitter;
use tenferro_tensor::{DType, DotGeneralConfig, PadConfig};

use super::context::{resolve_and_guard, ShapeGuardContext};
use crate::dim_expr::DimExpr;
use crate::std_tensor_op::StdTensorOp;

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

pub fn linearize_lu(
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
        StdTensorOp::TriangularSolve {
            left_side: true,
            lower: true,
            transpose_a: false,
            unit_diagonal: true,
        },
        vec![ValRef::Local(l_square), ValRef::Local(pd_a)],
        OpMode::Linear {
            active_mask: vec![false, true],
        },
    )[0];
    let x = builder.add_op(
        StdTensorOp::TriangularSolve {
            left_side: false,
            lower: false,
            transpose_a: false,
            unit_diagonal: false,
        },
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

pub fn linearize_eig(
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
        DType::I64 => return vec![None, None],
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

pub fn linearize_triangular_solve(
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    primal_out: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    left_side: bool,
    lower: bool,
    transpose_a: bool,
    unit_diagonal: bool,
    ctx: &mut ShapeGuardContext,
) -> Vec<Option<LocalValId>> {
    // Equation: op(A) @ X = B  (left_side=true)
    //       or  X @ op(A) = B  (left_side=false)
    // where op = identity (transpose_a=false) or transpose (transpose_a=true).
    //
    // Linearize: op(A) @ dX = dB - d(op(A)) @ X  (left_side=true)
    //        or  dX @ op(A) = dB - X @ d(op(A))  (left_side=false)
    //
    // When tangent_in[0] (dA) is present, we compute the correction:
    //   -d(op(A)) @ X  or  -X @ d(op(A))
    let lhs_rank = ctx.shape_of(&ValRef::External(primal_in[0].clone())).len();
    let rhs_rank = ctx.shape_of(&ValRef::External(primal_in[1].clone())).len();
    assert!(
        lhs_rank >= 2 && rhs_rank >= 2,
        "linearize_triangular_solve: expected matrix operands"
    );
    assert_eq!(
        lhs_rank, rhs_rank,
        "linearize_triangular_solve: rank mismatch between lhs and rhs"
    );
    let rank = lhs_rank;
    let rhs_tangent = triangular_solve_rhs_tangent(
        builder,
        primal_out,
        tangent_in,
        left_side,
        lower,
        transpose_a,
        unit_diagonal,
        rank,
    );
    let Some(rhs_tangent) = rhs_tangent else {
        return vec![None];
    };

    let out = builder.add_op(
        StdTensorOp::TriangularSolve {
            left_side,
            lower,
            transpose_a,
            unit_diagonal,
        },
        vec![
            ValRef::External(primal_in[0].clone()),
            ValRef::Local(rhs_tangent),
        ],
        OpMode::Linear {
            active_mask: vec![false, true],
        },
    );
    vec![Some(out[0])]
}

pub fn linearize_full_piv_lu_solve(
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    primal_out: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    transpose_a: bool,
    ctx: &mut ShapeGuardContext,
) -> Vec<Option<LocalValId>> {
    let lhs_rank = ctx.shape_of(&ValRef::External(primal_in[0].clone())).len();
    let rhs_rank = ctx.shape_of(&ValRef::External(primal_in[1].clone())).len();
    assert!(
        lhs_rank >= 2 && rhs_rank >= 2,
        "linearize_full_piv_lu_solve: expected matrix operands"
    );
    assert_eq!(
        lhs_rank, rhs_rank,
        "linearize_full_piv_lu_solve: rank mismatch between lhs and rhs"
    );
    let rank = lhs_rank;
    let mut rhs_tangent = tangent_in[1];

    if let Some(da) = tangent_in[0] {
        let d_op_a = if transpose_a {
            transpose_matrix_linear(builder, da, rank)
        } else {
            da
        };
        let x = ValRef::External(primal_out[0].clone());
        let correction = matmul_linear(builder, ValRef::Local(d_op_a), x, vec![true, false], rank);
        let neg_correction = linear_neg(builder, correction);
        rhs_tangent = Some(match rhs_tangent {
            Some(db) => linear_add(builder, db, neg_correction),
            None => neg_correction,
        });
    }

    let Some(rhs_tangent) = rhs_tangent else {
        return vec![None];
    };

    let out = builder.add_op(
        StdTensorOp::FullPivLuSolve { transpose_a },
        vec![
            ValRef::External(primal_in[0].clone()),
            ValRef::Local(rhs_tangent),
        ],
        OpMode::Linear {
            active_mask: vec![false, true],
        },
    );
    vec![Some(out[0])]
}

fn triangular_solve_rhs_tangent(
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_out: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    left_side: bool,
    lower: bool,
    transpose_a: bool,
    unit_diagonal: bool,
    rank: usize,
) -> Option<LocalValId> {
    let mut rhs_tangent = tangent_in[1];

    if let Some(da) = tangent_in[0] {
        let da = project_triangular_operand_linear(builder, da, lower, unit_diagonal);
        // d(op(A)) = op(dA), with op = identity or transpose.
        let d_op_a = if transpose_a {
            transpose_matrix_linear(builder, da, rank)
        } else {
            da
        };

        // Correction = d(op(A)) @ X  or  X @ d(op(A))
        let x = ValRef::External(primal_out[0].clone());
        let correction = if left_side {
            matmul_linear(builder, ValRef::Local(d_op_a), x, vec![true, false], rank)
        } else {
            matmul_linear(builder, x, ValRef::Local(d_op_a), vec![false, true], rank)
        };
        let neg_correction = linear_neg(builder, correction);
        rhs_tangent = Some(match rhs_tangent {
            Some(db) => linear_add(builder, db, neg_correction),
            None => neg_correction,
        });
    }

    rhs_tangent
}

pub fn linearize_svd(
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
    let k = DimExpr::min(m.clone(), n.clone());
    let u = ValRef::External(primal_out[0].clone());
    let s = ValRef::External(primal_out[1].clone());
    let vt = ValRef::External(primal_out[2].clone());

    let uh = adjoint_matrix_fixed(builder, u.clone(), matrix_rank);
    let v = adjoint_matrix_fixed(builder, vt.clone(), matrix_rank);
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

    let ds_h = adjoint_matrix_linear(builder, ds_mat, matrix_rank);
    let anti_hermitian = linear_sub(builder, ds_mat, ds_h);
    // Matches JAX's complex gauge correction: 0.5 * (dS - dS^H) * diag(1 / s).
    let d_udv_diag = hadamard_fixed_linear(builder, ValRef::Local(s_inv_mat), anti_hermitian);
    let d_udv_diag = linear_scale(builder, d_udv_diag, 0.5);

    let dss = hadamard_fixed_linear(builder, ValRef::Local(s_dim), ds_mat);
    let dss_h = adjoint_matrix_linear(builder, dss, matrix_rank);
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
    let sds_h = adjoint_matrix_linear(builder, sds, matrix_rank);
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
        let da_h = adjoint_matrix_linear(builder, da, matrix_rank);
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

    let dvt = adjoint_matrix_linear(builder, dv, matrix_rank);

    vec![Some(du), Some(ds), Some(dvt)]
}

pub fn linearize_eigh(
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
    let w = ValRef::External(primal_out[0].clone());
    let v = ValRef::External(primal_out[1].clone());
    let da_self_adjoint = self_adjoint_from_lower_linear(builder, da, matrix_rank);

    let vh = adjoint_matrix_fixed(builder, v.clone(), matrix_rank);
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

pub fn linearize_cholesky(
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
    let l = ValRef::External(primal_out[0].clone());
    let da_self_adjoint = self_adjoint_from_lower_linear(builder, da, matrix_rank);
    let l_conj = fixed_unary(builder, StdTensorOp::Conj, l.clone());

    let tmp = builder.add_op(
        StdTensorOp::TriangularSolve {
            left_side: false,
            lower: true,
            transpose_a: true,
            unit_diagonal: false,
        },
        vec![ValRef::Local(l_conj), ValRef::Local(da_self_adjoint)],
        OpMode::Linear {
            active_mask: vec![false, true],
        },
    )[0];
    let s = builder.add_op(
        StdTensorOp::TriangularSolve {
            left_side: true,
            lower: true,
            transpose_a: false,
            unit_diagonal: false,
        },
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

pub fn linearize_qr(
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
    let q = ValRef::External(primal_out[0].clone());
    let r = ValRef::External(primal_out[1].clone());

    if n_size > m_size {
        let qh = adjoint_matrix_fixed(builder, q.clone(), matrix_rank);
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
        let r_leading_h = adjoint_matrix_fixed(builder, ValRef::Local(r_leading), matrix_rank);
        let da_leading_h = adjoint_matrix_linear(builder, da_leading, matrix_rank);
        let dx_rinv_h = builder.add_op(
            StdTensorOp::TriangularSolve {
                left_side: true,
                lower: true,
                transpose_a: false,
                unit_diagonal: false,
            },
            vec![ValRef::Local(r_leading_h), ValRef::Local(da_leading_h)],
            OpMode::Linear {
                active_mask: vec![false, true],
            },
        )[0];
        let dx_rinv = adjoint_matrix_linear(builder, dx_rinv_h, matrix_rank);
        let qt_dx_rinv = matmul_linear(
            builder,
            ValRef::Local(qh),
            ValRef::Local(dx_rinv),
            vec![false, true],
            matrix_rank,
        );
        let qt_dx_rinv_h = adjoint_matrix_linear(builder, qt_dx_rinv, matrix_rank);
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

    let r_h = adjoint_matrix_fixed(builder, r.clone(), matrix_rank);
    let da_h = adjoint_matrix_linear(builder, da, matrix_rank);
    let dx_rinv_h = builder.add_op(
        StdTensorOp::TriangularSolve {
            left_side: true,
            lower: true,
            transpose_a: false,
            unit_diagonal: false,
        },
        vec![ValRef::Local(r_h), ValRef::Local(da_h)],
        OpMode::Linear {
            active_mask: vec![false, true],
        },
    )[0];
    let dx_rinv = adjoint_matrix_linear(builder, dx_rinv_h, matrix_rank);
    let qh = adjoint_matrix_fixed(builder, q.clone(), matrix_rank);
    let qt_dx_rinv = matmul_linear(
        builder,
        ValRef::Local(qh),
        ValRef::Local(dx_rinv),
        vec![false, true],
        matrix_rank,
    );
    let qt_dx_rinv_h = adjoint_matrix_linear(builder, qt_dx_rinv, matrix_rank);
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

pub fn transpose_triangular_solve(
    emitter: &mut impl OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    inputs: &[ValRef<StdTensorOp>],
    mode: &OpMode,
    left_side: bool,
    lower: bool,
    transpose_a: bool,
    unit_diagonal: bool,
) -> Vec<Option<LocalValId>> {
    let Some(ct) = cotangent_out[0] else {
        return vec![None, None];
    };
    let OpMode::Linear { active_mask } = mode else {
        return vec![None, None];
    };

    let mut result = vec![None, None];
    if active_mask[1] {
        let conjugated_a =
            emitter.add_op(StdTensorOp::Conj, vec![inputs[0].clone()], OpMode::Primal)[0];
        let out = emitter.add_op(
            StdTensorOp::TriangularSolve {
                left_side,
                lower,
                transpose_a: !transpose_a,
                unit_diagonal,
            },
            vec![ValRef::Local(conjugated_a), ValRef::Local(ct)],
            OpMode::Linear {
                active_mask: vec![false, true],
            },
        );
        result[1] = Some(out[0]);
    }

    result
}

pub fn transpose_full_piv_lu_solve(
    emitter: &mut impl OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    inputs: &[ValRef<StdTensorOp>],
    mode: &OpMode,
    transpose_a: bool,
) -> Vec<Option<LocalValId>> {
    let Some(ct) = cotangent_out[0] else {
        return vec![None, None];
    };
    let OpMode::Linear { active_mask } = mode else {
        return vec![None, None];
    };

    let mut result = vec![None, None];
    if active_mask[1] {
        let conjugated_a =
            emitter.add_op(StdTensorOp::Conj, vec![inputs[0].clone()], OpMode::Primal)[0];
        let out = emitter.add_op(
            StdTensorOp::FullPivLuSolve {
                transpose_a: !transpose_a,
            },
            vec![ValRef::Local(conjugated_a), ValRef::Local(ct)],
            OpMode::Linear {
                active_mask: vec![false, true],
            },
        );
        result[1] = Some(out[0]);
    }

    result
}

fn solve_in_graph(
    builder: &mut FragmentBuilder<StdTensorOp>,
    a: ValRef<StdTensorOp>,
    b: ValRef<StdTensorOp>,
    rank: usize,
) -> LocalValId {
    let lu_outputs = builder.add_op(StdTensorOp::Lu, vec![a], OpMode::Primal);
    let p = lu_outputs[0];
    let l = lu_outputs[1];
    let u = lu_outputs[2];
    let pb = matmul_linear(builder, ValRef::Local(p), b, vec![false, true], rank);
    let z = builder.add_op(
        StdTensorOp::TriangularSolve {
            left_side: true,
            lower: true,
            transpose_a: false,
            unit_diagonal: true,
        },
        vec![ValRef::Local(l), ValRef::Local(pb)],
        OpMode::Linear {
            active_mask: vec![false, true],
        },
    )[0];
    builder.add_op(
        StdTensorOp::TriangularSolve {
            left_side: true,
            lower: false,
            transpose_a: false,
            unit_diagonal: false,
        },
        vec![ValRef::Local(u), ValRef::Local(z)],
        OpMode::Linear {
            active_mask: vec![false, true],
        },
    )[0]
}

fn fixed_unary(
    emitter: &mut impl OpEmitter<StdTensorOp>,
    op: StdTensorOp,
    input: ValRef<StdTensorOp>,
) -> LocalValId {
    emitter.add_op(op, vec![input], OpMode::Primal)[0]
}

fn fixed_binary(
    builder: &mut FragmentBuilder<StdTensorOp>,
    op: StdTensorOp,
    lhs: ValRef<StdTensorOp>,
    rhs: ValRef<StdTensorOp>,
) -> LocalValId {
    builder.add_op(op, vec![lhs, rhs], OpMode::Primal)[0]
}

fn linear_unary(
    builder: &mut FragmentBuilder<StdTensorOp>,
    op: StdTensorOp,
    input: LocalValId,
) -> LocalValId {
    builder.add_op(
        op,
        vec![ValRef::Local(input)],
        OpMode::Linear {
            active_mask: vec![true],
        },
    )[0]
}

fn linear_binary(
    builder: &mut FragmentBuilder<StdTensorOp>,
    op: StdTensorOp,
    lhs: LocalValId,
    rhs: LocalValId,
) -> LocalValId {
    builder.add_op(
        op,
        vec![ValRef::Local(lhs), ValRef::Local(rhs)],
        OpMode::Linear {
            active_mask: vec![true, true],
        },
    )[0]
}

fn fixed_add(
    builder: &mut FragmentBuilder<StdTensorOp>,
    lhs: ValRef<StdTensorOp>,
    rhs: ValRef<StdTensorOp>,
) -> LocalValId {
    fixed_binary(builder, StdTensorOp::Add, lhs, rhs)
}

fn fixed_mul(
    builder: &mut FragmentBuilder<StdTensorOp>,
    lhs: ValRef<StdTensorOp>,
    rhs: ValRef<StdTensorOp>,
) -> LocalValId {
    fixed_binary(builder, StdTensorOp::Mul, lhs, rhs)
}

fn fixed_div(
    builder: &mut FragmentBuilder<StdTensorOp>,
    lhs: ValRef<StdTensorOp>,
    rhs: ValRef<StdTensorOp>,
) -> LocalValId {
    fixed_binary(builder, StdTensorOp::Div, lhs, rhs)
}

fn fixed_scale(
    builder: &mut FragmentBuilder<StdTensorOp>,
    input: ValRef<StdTensorOp>,
    factor: f64,
) -> LocalValId {
    let constant = builder.add_op(StdTensorOp::constant_f64(factor), vec![], OpMode::Primal);
    builder.add_op(
        StdTensorOp::Mul,
        vec![ValRef::Local(constant[0]), input],
        OpMode::Primal,
    )[0]
}

fn broadcast_in_dim_fixed(
    builder: &mut FragmentBuilder<StdTensorOp>,
    input: ValRef<StdTensorOp>,
    shape: Vec<DimExpr>,
    dims: Vec<usize>,
) -> LocalValId {
    fixed_unary(builder, StdTensorOp::BroadcastInDim { shape, dims }, input)
}

fn reduce_sum_fixed(
    builder: &mut FragmentBuilder<StdTensorOp>,
    input: ValRef<StdTensorOp>,
    axes: Vec<usize>,
) -> LocalValId {
    fixed_unary(builder, StdTensorOp::ReduceSum { axes }, input)
}

fn pad_fixed(
    builder: &mut FragmentBuilder<StdTensorOp>,
    input: ValRef<StdTensorOp>,
    rank: usize,
    edge_padding_low: Vec<i64>,
    edge_padding_high: Vec<i64>,
) -> LocalValId {
    fixed_unary(
        builder,
        StdTensorOp::Pad(PadConfig {
            edge_padding_low,
            edge_padding_high,
            interior_padding: vec![0; rank],
        }),
        input,
    )
}

fn fixed_sub(
    builder: &mut FragmentBuilder<StdTensorOp>,
    lhs: ValRef<StdTensorOp>,
    rhs: ValRef<StdTensorOp>,
) -> LocalValId {
    let neg_rhs = fixed_unary(builder, StdTensorOp::Neg, rhs);
    fixed_add(builder, lhs, ValRef::Local(neg_rhs))
}

fn linear_add(
    builder: &mut FragmentBuilder<StdTensorOp>,
    lhs: LocalValId,
    rhs: LocalValId,
) -> LocalValId {
    linear_binary(builder, StdTensorOp::Add, lhs, rhs)
}

fn linear_neg(builder: &mut FragmentBuilder<StdTensorOp>, input: LocalValId) -> LocalValId {
    linear_unary(builder, StdTensorOp::Neg, input)
}

fn linear_scale(
    builder: &mut FragmentBuilder<StdTensorOp>,
    input: LocalValId,
    factor: f64,
) -> LocalValId {
    let constant = builder.add_op(StdTensorOp::constant_f64(factor), vec![], OpMode::Primal);
    builder.add_op(
        StdTensorOp::Mul,
        vec![ValRef::Local(constant[0]), ValRef::Local(input)],
        OpMode::Linear {
            active_mask: vec![false, true],
        },
    )[0]
}

fn linear_sub(
    builder: &mut FragmentBuilder<StdTensorOp>,
    lhs: LocalValId,
    rhs: LocalValId,
) -> LocalValId {
    let neg_rhs = linear_neg(builder, rhs);
    linear_add(builder, lhs, neg_rhs)
}

fn linear_div_fixed(
    builder: &mut FragmentBuilder<StdTensorOp>,
    lhs: LocalValId,
    rhs: ValRef<StdTensorOp>,
) -> LocalValId {
    builder.add_op(
        StdTensorOp::Div,
        vec![ValRef::Local(lhs), rhs],
        OpMode::Linear {
            active_mask: vec![true, false],
        },
    )[0]
}

fn hadamard_fixed_linear(
    builder: &mut FragmentBuilder<StdTensorOp>,
    fixed: ValRef<StdTensorOp>,
    active: LocalValId,
) -> LocalValId {
    builder.add_op(
        StdTensorOp::Mul,
        vec![fixed, ValRef::Local(active)],
        OpMode::Linear {
            active_mask: vec![false, true],
        },
    )[0]
}

fn one_like_fixed(
    builder: &mut FragmentBuilder<StdTensorOp>,
    anchor: ValRef<StdTensorOp>,
) -> LocalValId {
    let zero = fixed_sub(builder, anchor.clone(), anchor);
    fixed_unary(builder, StdTensorOp::Exp, ValRef::Local(zero))
}

fn extract_diag_linear(
    builder: &mut FragmentBuilder<StdTensorOp>,
    input: LocalValId,
) -> LocalValId {
    builder.add_op(
        StdTensorOp::ExtractDiag {
            axis_a: 0,
            axis_b: 1,
        },
        vec![ValRef::Local(input)],
        OpMode::Linear {
            active_mask: vec![true],
        },
    )[0]
}

fn embed_diag_linear(builder: &mut FragmentBuilder<StdTensorOp>, input: LocalValId) -> LocalValId {
    builder.add_op(
        StdTensorOp::EmbedDiag {
            axis_a: 0,
            axis_b: 1,
        },
        vec![ValRef::Local(input)],
        OpMode::Linear {
            active_mask: vec![true],
        },
    )[0]
}

fn embed_diag_fixed(
    builder: &mut FragmentBuilder<StdTensorOp>,
    input: ValRef<StdTensorOp>,
) -> LocalValId {
    fixed_unary(
        builder,
        StdTensorOp::EmbedDiag {
            axis_a: 0,
            axis_b: 1,
        },
        input,
    )
}

fn transpose_matrix_fixed(
    emitter: &mut impl OpEmitter<StdTensorOp>,
    input: ValRef<StdTensorOp>,
    rank: usize,
) -> LocalValId {
    fixed_unary(
        emitter,
        StdTensorOp::Transpose {
            perm: matrix_transpose_perm(rank),
        },
        input,
    )
}

fn adjoint_matrix_fixed(
    builder: &mut FragmentBuilder<StdTensorOp>,
    input: ValRef<StdTensorOp>,
    rank: usize,
) -> LocalValId {
    let conjugated = fixed_unary(builder, StdTensorOp::Conj, input);
    transpose_matrix_fixed(builder, ValRef::Local(conjugated), rank)
}

fn adjoint_matrix_linear(
    builder: &mut FragmentBuilder<StdTensorOp>,
    input: LocalValId,
    rank: usize,
) -> LocalValId {
    let conjugated = linear_unary(builder, StdTensorOp::Conj, input);
    builder.add_op(
        StdTensorOp::Transpose {
            perm: matrix_transpose_perm(rank),
        },
        vec![ValRef::Local(conjugated)],
        OpMode::Linear {
            active_mask: vec![true],
        },
    )[0]
}

fn transpose_matrix_linear(
    emitter: &mut impl OpEmitter<StdTensorOp>,
    input: LocalValId,
    rank: usize,
) -> LocalValId {
    emitter.add_op(
        StdTensorOp::Transpose {
            perm: matrix_transpose_perm(rank),
        },
        vec![ValRef::Local(input)],
        OpMode::Linear {
            active_mask: vec![true],
        },
    )[0]
}

fn project_triangular_operand_linear(
    builder: &mut FragmentBuilder<StdTensorOp>,
    input: LocalValId,
    lower: bool,
    unit_diagonal: bool,
) -> LocalValId {
    let op = if lower {
        StdTensorOp::Tril {
            k: if unit_diagonal { -1 } else { 0 },
        }
    } else {
        StdTensorOp::Triu {
            k: if unit_diagonal { 1 } else { 0 },
        }
    };
    linear_unary(builder, op, input)
}

fn self_adjoint_from_lower_linear(
    builder: &mut FragmentBuilder<StdTensorOp>,
    input: LocalValId,
    rank: usize,
) -> LocalValId {
    let strict_lower = builder.add_op(
        StdTensorOp::Tril { k: -1 },
        vec![ValRef::Local(input)],
        OpMode::Linear {
            active_mask: vec![true],
        },
    )[0];
    let strict_lower_h = adjoint_matrix_linear(builder, strict_lower, rank);
    let offdiag = linear_add(builder, strict_lower, strict_lower_h);

    let diag = extract_diag_linear(builder, input);
    let diag_h = linear_unary(builder, StdTensorOp::Conj, diag);
    let diag_sum = linear_add(builder, diag, diag_h);
    let real_diag = linear_scale(builder, diag_sum, 0.5);
    let diag_mat = embed_diag_linear(builder, real_diag);
    linear_add(builder, offdiag, diag_mat)
}

fn matmul_fixed(
    builder: &mut FragmentBuilder<StdTensorOp>,
    lhs: ValRef<StdTensorOp>,
    rhs: ValRef<StdTensorOp>,
    rank: usize,
) -> LocalValId {
    builder.add_op(
        StdTensorOp::DotGeneral {
            config: matrix_multiply_config(rank),
        },
        vec![lhs, rhs],
        OpMode::Primal,
    )[0]
}

fn matmul_linear(
    builder: &mut FragmentBuilder<StdTensorOp>,
    lhs: ValRef<StdTensorOp>,
    rhs: ValRef<StdTensorOp>,
    active_mask: Vec<bool>,
    rank: usize,
) -> LocalValId {
    builder.add_op(
        StdTensorOp::DotGeneral {
            config: matrix_multiply_config(rank),
        },
        vec![lhs, rhs],
        OpMode::Linear { active_mask },
    )[0]
}

fn matrix_multiply_config(rank: usize) -> DotGeneralConfig {
    assert!(rank >= 2, "matrix_multiply_config expects rank >= 2");
    let batch_dims: Vec<usize> = (2..rank).collect();
    DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: batch_dims.clone(),
        rhs_batch_dims: batch_dims,
    }
}

fn matrix_transpose_perm(rank: usize) -> Vec<usize> {
    assert!(rank >= 2, "matrix_transpose_perm expects rank >= 2");
    let mut perm: Vec<usize> = (0..rank).collect();
    perm.swap(0, 1);
    perm
}

fn matrix_shape_parts<'a>(
    shape: &'a [DimExpr],
    op: &str,
) -> (&'a DimExpr, &'a DimExpr, &'a [DimExpr]) {
    assert!(shape.len() >= 2, "{op}: expected rank >= 2");
    (&shape[0], &shape[1], &shape[2..])
}

fn matrix_shape(
    rows: impl Into<DimExpr>,
    cols: impl Into<DimExpr>,
    batch_shape: &[DimExpr],
) -> Vec<DimExpr> {
    let mut shape = vec![rows.into(), cols.into()];
    shape.extend_from_slice(batch_shape);
    shape
}

fn vector_shape(len: impl Into<DimExpr>, batch_shape: &[DimExpr]) -> Vec<DimExpr> {
    let mut shape = vec![len.into()];
    shape.extend_from_slice(batch_shape);
    shape
}

fn vector_to_matrix_broadcast_dims(batch_ndim: usize) -> Vec<usize> {
    let mut dims = Vec::with_capacity(1 + batch_ndim);
    dims.push(1);
    dims.extend(2..(2 + batch_ndim));
    dims
}

fn leading_column_selector_fixed(
    builder: &mut FragmentBuilder<StdTensorOp>,
    leading_cols: usize,
    total_cols: usize,
    batch_shape: &[DimExpr],
    anchor: ValRef<StdTensorOp>,
    anchor_shape: &[DimExpr],
) -> LocalValId {
    let eye = identity_matrix_fixed(builder, leading_cols, batch_shape, anchor, anchor_shape);
    let rank = 2 + batch_shape.len();
    pad_fixed(
        builder,
        ValRef::Local(eye),
        rank,
        vec![0; rank],
        pad_vec(rank, 1, (total_cols - leading_cols) as i64),
    )
}

fn trailing_column_selector_fixed(
    builder: &mut FragmentBuilder<StdTensorOp>,
    leading_cols: usize,
    trailing_cols: usize,
    batch_shape: &[DimExpr],
    anchor: ValRef<StdTensorOp>,
    anchor_shape: &[DimExpr],
) -> LocalValId {
    let eye = identity_matrix_fixed(builder, trailing_cols, batch_shape, anchor, anchor_shape);
    let rank = 2 + batch_shape.len();
    pad_fixed(
        builder,
        ValRef::Local(eye),
        rank,
        pad_vec(rank, 1, leading_cols as i64),
        vec![0; rank],
    )
}

fn identity_matrix_fixed(
    builder: &mut FragmentBuilder<StdTensorOp>,
    size: usize,
    batch_shape: &[DimExpr],
    anchor: ValRef<StdTensorOp>,
    anchor_shape: &[DimExpr],
) -> LocalValId {
    let one_scalar = scalar_one_fixed(builder, anchor, anchor_shape);
    let ones = broadcast_in_dim_fixed(
        builder,
        ValRef::Local(one_scalar),
        vector_shape(size, batch_shape),
        vec![],
    );
    embed_diag_fixed(builder, ValRef::Local(ones))
}

fn scalar_one_fixed(
    builder: &mut FragmentBuilder<StdTensorOp>,
    anchor: ValRef<StdTensorOp>,
    anchor_shape: &[DimExpr],
) -> LocalValId {
    let zero = fixed_sub(builder, anchor.clone(), anchor);
    let zero_scalar = if anchor_shape.is_empty() {
        zero
    } else {
        reduce_sum_fixed(
            builder,
            ValRef::Local(zero),
            (0..anchor_shape.len()).collect(),
        )
    };
    fixed_unary(builder, StdTensorOp::Exp, ValRef::Local(zero_scalar))
}

fn pad_vec(rank: usize, axis: usize, amount: i64) -> Vec<i64> {
    let mut padding = vec![0; rank];
    padding[axis] = amount;
    padding
}

fn pad_matrix_low(rank: usize, row_amount: i64, col_amount: i64) -> Vec<i64> {
    let mut padding = vec![0; rank];
    padding[0] = row_amount;
    padding[1] = col_amount;
    padding
}

fn augment_unit_lower_to_square_fixed(
    builder: &mut FragmentBuilder<StdTensorOp>,
    l: ValRef<StdTensorOp>,
    rows: usize,
    cols: usize,
    batch_shape: &[DimExpr],
    l_shape: &[DimExpr],
    rank: usize,
) -> LocalValId {
    let strict_lower = fixed_unary(builder, StdTensorOp::Tril { k: -1 }, l.clone());
    let strict_lower_square = if cols < rows {
        pad_fixed(
            builder,
            ValRef::Local(strict_lower),
            rank,
            vec![0; rank],
            pad_vec(rank, 1, (rows - cols) as i64),
        )
    } else {
        strict_lower
    };
    let identity = identity_matrix_fixed(builder, rows, batch_shape, l, l_shape);
    fixed_add(
        builder,
        ValRef::Local(strict_lower_square),
        ValRef::Local(identity),
    )
}

fn augment_upper_to_square_fixed(
    builder: &mut FragmentBuilder<StdTensorOp>,
    u: ValRef<StdTensorOp>,
    rows: usize,
    cols: usize,
    batch_shape: &[DimExpr],
    u_shape: &[DimExpr],
    rank: usize,
) -> LocalValId {
    let upper = fixed_unary(builder, StdTensorOp::Triu { k: 0 }, u.clone());
    if rows == cols {
        return upper;
    }

    let upper_square = pad_fixed(
        builder,
        ValRef::Local(upper),
        rank,
        vec![0; rank],
        pad_vec(rank, 0, (cols - rows) as i64),
    );
    let trailing_eye = identity_matrix_fixed(builder, cols - rows, batch_shape, u, u_shape);
    let trailing_eye = pad_fixed(
        builder,
        ValRef::Local(trailing_eye),
        rank,
        pad_matrix_low(rank, rows as i64, rows as i64),
        vec![0; rank],
    );
    fixed_add(
        builder,
        ValRef::Local(upper_square),
        ValRef::Local(trailing_eye),
    )
}

fn take_leading_cols_linear(
    builder: &mut FragmentBuilder<StdTensorOp>,
    input: LocalValId,
    cols: usize,
    total_cols: usize,
    batch_shape: &[DimExpr],
    anchor: ValRef<StdTensorOp>,
    anchor_shape: &[DimExpr],
    rank: usize,
) -> LocalValId {
    let selector =
        leading_column_selector_fixed(builder, cols, total_cols, batch_shape, anchor, anchor_shape);
    let selector_t = transpose_matrix_fixed(builder, ValRef::Local(selector), rank);
    matmul_linear(
        builder,
        ValRef::Local(input),
        ValRef::Local(selector_t),
        vec![true, false],
        rank,
    )
}

fn take_leading_rows_linear(
    builder: &mut FragmentBuilder<StdTensorOp>,
    input: LocalValId,
    rows: usize,
    total_rows: usize,
    batch_shape: &[DimExpr],
    anchor: ValRef<StdTensorOp>,
    anchor_shape: &[DimExpr],
    rank: usize,
) -> LocalValId {
    let selector =
        leading_column_selector_fixed(builder, rows, total_rows, batch_shape, anchor, anchor_shape);
    matmul_linear(
        builder,
        ValRef::Local(selector),
        ValRef::Local(input),
        vec![false, true],
        rank,
    )
}
