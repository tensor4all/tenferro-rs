use computegraph::fragment::FragmentBuilder;
use computegraph::types::{GlobalValKey, LocalValId, OpMode, ValRef};
use tenferro_tensor::DotGeneralConfig;

use crate::std_tensor_op::StdTensorOp;

const F_MATRIX_EPS: f64 = 1.0e-12;

pub fn linearize_solve(
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    primal_out: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
) -> Vec<Option<LocalValId>> {
    let rhs_tangent = solve_rhs_tangent(builder, primal_out, tangent_in);
    let Some(rhs_tangent) = rhs_tangent else {
        return vec![None];
    };

    let out = builder.add_op(
        StdTensorOp::Solve,
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

pub fn linearize_triangular_solve(
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    left_side: bool,
    lower: bool,
    transpose_a: bool,
    unit_diagonal: bool,
) -> Vec<Option<LocalValId>> {
    match tangent_in[1] {
        Some(db) => {
            let out = builder.add_op(
                StdTensorOp::TriangularSolve {
                    left_side,
                    lower,
                    transpose_a,
                    unit_diagonal,
                },
                vec![ValRef::External(primal_in[0].clone()), ValRef::Local(db)],
                OpMode::Linear {
                    active_mask: vec![false, true],
                },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}

pub fn linearize_svd(
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_out: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
) -> Vec<Option<LocalValId>> {
    let Some(da) = tangent_in[0] else {
        return vec![None, None, None];
    };

    let u = ValRef::External(primal_out[0].clone());
    let s = ValRef::External(primal_out[1].clone());
    let vt = ValRef::External(primal_out[2].clone());

    let uh = adjoint_2d_fixed(builder, u.clone());
    let v = adjoint_2d_fixed(builder, vt);
    let tmp = matmul_linear(
        builder,
        ValRef::Local(uh),
        ValRef::Local(da),
        vec![false, true],
    );
    let ds_mat = matmul_linear(
        builder,
        ValRef::Local(tmp),
        ValRef::Local(v),
        vec![true, false],
    );
    let ds = extract_diag_linear(builder, ds_mat);

    let diag_s = embed_diag_fixed(builder, s.clone());
    let ones_mat = one_like_fixed(builder, ValRef::Local(diag_s));
    let s_col = matmul_fixed(builder, ValRef::Local(diag_s), ValRef::Local(ones_mat));
    let s_row = matmul_fixed(builder, ValRef::Local(ones_mat), ValRef::Local(diag_s));
    let s_sum = fixed_add(builder, ValRef::Local(s_col), ValRef::Local(s_row));
    let s_diff = fixed_sub(builder, ValRef::Local(s_col), ValRef::Local(s_row));
    let s_gap = fixed_mul(builder, ValRef::Local(s_sum), ValRef::Local(s_diff));
    let s_gap_sq = fixed_mul(builder, ValRef::Local(s_gap), ValRef::Local(s_gap));
    let eps_sq = fixed_scale(
        builder,
        ValRef::Local(ones_mat),
        F_MATRIX_EPS * F_MATRIX_EPS,
    );
    let safe_gap = fixed_add(builder, ValRef::Local(s_gap_sq), ValRef::Local(eps_sq));
    let f = fixed_div(builder, ValRef::Local(s_gap), ValRef::Local(safe_gap));

    // TODO: add the complex dUdV diagonal correction and rectangular JAX correction terms.
    let dss = hadamard_fixed_linear(builder, ValRef::Local(s_col), ds_mat);
    let dss_h = adjoint_2d_linear(builder, dss);
    let du_inner_sum = linear_add(builder, dss, dss_h);
    let du_inner = hadamard_fixed_linear(builder, ValRef::Local(f), du_inner_sum);
    let du = matmul_linear(builder, u, ValRef::Local(du_inner), vec![false, true]);

    let sds = hadamard_fixed_linear(builder, ValRef::Local(s_row), ds_mat);
    let sds_h = adjoint_2d_linear(builder, sds);
    let dv_inner_sum = linear_add(builder, sds, sds_h);
    let dv_inner = hadamard_fixed_linear(builder, ValRef::Local(f), dv_inner_sum);
    let dv = matmul_linear(
        builder,
        ValRef::Local(v),
        ValRef::Local(dv_inner),
        vec![false, true],
    );
    let dvt = adjoint_2d_linear(builder, dv);

    vec![Some(du), Some(ds), Some(dvt)]
}

pub fn linearize_eigh(
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_out: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
) -> Vec<Option<LocalValId>> {
    let Some(da) = tangent_in[0] else {
        return vec![None, None];
    };

    let w = ValRef::External(primal_out[0].clone());
    let v = ValRef::External(primal_out[1].clone());
    let da_self_adjoint = self_adjoint_from_lower_linear(builder, da);

    let vh = adjoint_2d_fixed(builder, v.clone());
    let tmp = matmul_linear(
        builder,
        ValRef::Local(vh),
        ValRef::Local(da_self_adjoint),
        vec![false, true],
    );
    let projected = matmul_linear(builder, ValRef::Local(tmp), v.clone(), vec![true, false]);
    let dw = extract_diag_linear(builder, projected);

    let diag_w = embed_diag_fixed(builder, w.clone());
    let ones_mat = one_like_fixed(builder, ValRef::Local(diag_w));
    let w_col = matmul_fixed(builder, ValRef::Local(diag_w), ValRef::Local(ones_mat));
    let w_row = matmul_fixed(builder, ValRef::Local(ones_mat), ValRef::Local(diag_w));
    let diff = fixed_sub(builder, ValRef::Local(w_row), ValRef::Local(w_col));
    let diff_sq = fixed_mul(builder, ValRef::Local(diff), ValRef::Local(diff));
    let eps_sq = fixed_scale(
        builder,
        ValRef::Local(ones_mat),
        F_MATRIX_EPS * F_MATRIX_EPS,
    );
    let safe_diff = fixed_add(builder, ValRef::Local(diff_sq), ValRef::Local(eps_sq));
    let f = fixed_div(builder, ValRef::Local(diff), ValRef::Local(safe_diff));
    let fm = hadamard_fixed_linear(builder, ValRef::Local(f), projected);
    let dv = matmul_linear(builder, v, ValRef::Local(fm), vec![false, true]);

    vec![Some(dw), Some(dv)]
}

pub fn linearize_cholesky(
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_out: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
) -> Vec<Option<LocalValId>> {
    let Some(da) = tangent_in[0] else {
        return vec![None];
    };

    let l = ValRef::External(primal_out[0].clone());
    let da_self_adjoint = self_adjoint_from_lower_linear(builder, da);
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
    let half_diag = builder.add_op(
        StdTensorOp::Scale { factor: 0.5 },
        vec![ValRef::Local(diag_s)],
        OpMode::Linear {
            active_mask: vec![true],
        },
    )[0];
    let half_diag_mat = embed_diag_linear(builder, half_diag);
    let phi_s = linear_add(builder, strict_lower, half_diag_mat);
    let dl = matmul_linear(builder, l, ValRef::Local(phi_s), vec![false, true]);

    vec![Some(dl)]
}

pub fn linearize_qr(
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_out: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
) -> Vec<Option<LocalValId>> {
    let Some(da) = tangent_in[0] else {
        return vec![None, None];
    };

    let q = ValRef::External(primal_out[0].clone());
    let r = ValRef::External(primal_out[1].clone());

    let dx_rinv = builder.add_op(
        StdTensorOp::TriangularSolve {
            left_side: false,
            lower: false,
            transpose_a: false,
            unit_diagonal: false,
        },
        vec![r.clone(), ValRef::Local(da)],
        OpMode::Linear {
            active_mask: vec![false, true],
        },
    )[0];
    let qh = adjoint_2d_fixed(builder, q.clone());
    let qt_dx_rinv = matmul_linear(
        builder,
        ValRef::Local(qh),
        ValRef::Local(dx_rinv),
        vec![false, true],
    );
    let lower = builder.add_op(
        StdTensorOp::Tril { k: -1 },
        vec![ValRef::Local(qt_dx_rinv)],
        OpMode::Linear {
            active_mask: vec![true],
        },
    )[0];
    let lower_h = adjoint_2d_linear(builder, lower);
    let d_omega = linear_sub(builder, lower, lower_h);

    let d_omega_minus_qt_dx = linear_sub(builder, d_omega, qt_dx_rinv);
    let q_term = matmul_linear(
        builder,
        q.clone(),
        ValRef::Local(d_omega_minus_qt_dx),
        vec![false, true],
    );
    let dq = linear_add(builder, q_term, dx_rinv);

    let qt_dx_minus_d_omega = linear_sub(builder, qt_dx_rinv, d_omega);
    let dr = matmul_linear(
        builder,
        ValRef::Local(qt_dx_minus_d_omega),
        r,
        vec![true, false],
    );

    vec![Some(dq), Some(dr)]
}

pub fn transpose_solve(
    builder: &mut FragmentBuilder<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    inputs: &[ValRef<StdTensorOp>],
    mode: &OpMode,
) -> Vec<Option<LocalValId>> {
    let Some(ct) = cotangent_out[0] else {
        return vec![None, None];
    };
    let OpMode::Linear { active_mask } = mode else {
        return vec![None, None];
    };

    let mut result = vec![None, None];
    if active_mask[1] {
        let a_h = adjoint_2d_fixed(builder, inputs[0].clone());
        let out = builder.add_op(
            StdTensorOp::Solve,
            vec![ValRef::Local(a_h), ValRef::Local(ct)],
            OpMode::Linear {
                active_mask: vec![false, true],
            },
        );
        result[1] = Some(out[0]);
    }

    result
}

pub fn transpose_triangular_solve(
    builder: &mut FragmentBuilder<StdTensorOp>,
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
        let conjugated_a = fixed_unary(builder, StdTensorOp::Conj, inputs[0].clone());
        let out = builder.add_op(
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

fn solve_rhs_tangent(
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_out: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
) -> Option<LocalValId> {
    let mut rhs_tangent = tangent_in[1];

    if let Some(da) = tangent_in[0] {
        let dax = matmul_linear(
            builder,
            ValRef::Local(da),
            ValRef::External(primal_out[0].clone()),
            vec![true, false],
        );
        let neg_dax = linear_neg(builder, dax);
        rhs_tangent = Some(match rhs_tangent {
            Some(db) => linear_add(builder, db, neg_dax),
            None => neg_dax,
        });
    }

    rhs_tangent
}

fn fixed_unary(
    builder: &mut FragmentBuilder<StdTensorOp>,
    op: StdTensorOp,
    input: ValRef<StdTensorOp>,
) -> LocalValId {
    builder.add_op(op, vec![input], OpMode::Primal)[0]
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
    fixed_unary(builder, StdTensorOp::Scale { factor }, input)
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

fn linear_sub(
    builder: &mut FragmentBuilder<StdTensorOp>,
    lhs: LocalValId,
    rhs: LocalValId,
) -> LocalValId {
    let neg_rhs = linear_neg(builder, rhs);
    linear_add(builder, lhs, neg_rhs)
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

fn transpose_2d_fixed(
    builder: &mut FragmentBuilder<StdTensorOp>,
    input: ValRef<StdTensorOp>,
) -> LocalValId {
    fixed_unary(builder, StdTensorOp::Transpose { perm: vec![1, 0] }, input)
}

fn adjoint_2d_fixed(
    builder: &mut FragmentBuilder<StdTensorOp>,
    input: ValRef<StdTensorOp>,
) -> LocalValId {
    let conjugated = fixed_unary(builder, StdTensorOp::Conj, input);
    transpose_2d_fixed(builder, ValRef::Local(conjugated))
}

fn adjoint_2d_linear(builder: &mut FragmentBuilder<StdTensorOp>, input: LocalValId) -> LocalValId {
    let conjugated = linear_unary(builder, StdTensorOp::Conj, input);
    builder.add_op(
        StdTensorOp::Transpose { perm: vec![1, 0] },
        vec![ValRef::Local(conjugated)],
        OpMode::Linear {
            active_mask: vec![true],
        },
    )[0]
}

fn self_adjoint_from_lower_linear(
    builder: &mut FragmentBuilder<StdTensorOp>,
    input: LocalValId,
) -> LocalValId {
    let strict_lower = builder.add_op(
        StdTensorOp::Tril { k: -1 },
        vec![ValRef::Local(input)],
        OpMode::Linear {
            active_mask: vec![true],
        },
    )[0];
    let strict_lower_h = adjoint_2d_linear(builder, strict_lower);
    let offdiag = linear_add(builder, strict_lower, strict_lower_h);

    let diag = extract_diag_linear(builder, input);
    let diag_h = linear_unary(builder, StdTensorOp::Conj, diag);
    let diag_sum = linear_add(builder, diag, diag_h);
    let real_diag = builder.add_op(
        StdTensorOp::Scale { factor: 0.5 },
        vec![ValRef::Local(diag_sum)],
        OpMode::Linear {
            active_mask: vec![true],
        },
    )[0];
    let diag_mat = embed_diag_linear(builder, real_diag);
    linear_add(builder, offdiag, diag_mat)
}

fn matmul_fixed(
    builder: &mut FragmentBuilder<StdTensorOp>,
    lhs: ValRef<StdTensorOp>,
    rhs: ValRef<StdTensorOp>,
) -> LocalValId {
    builder.add_op(
        StdTensorOp::DotGeneral(matrix_multiply_config()),
        vec![lhs, rhs],
        OpMode::Primal,
    )[0]
}

fn matmul_linear(
    builder: &mut FragmentBuilder<StdTensorOp>,
    lhs: ValRef<StdTensorOp>,
    rhs: ValRef<StdTensorOp>,
    active_mask: Vec<bool>,
) -> LocalValId {
    builder.add_op(
        StdTensorOp::DotGeneral(matrix_multiply_config()),
        vec![lhs, rhs],
        OpMode::Linear { active_mask },
    )[0]
}

fn matrix_multiply_config() -> DotGeneralConfig {
    DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
        lhs_rank: 2,
        rhs_rank: 2,
    }
}
