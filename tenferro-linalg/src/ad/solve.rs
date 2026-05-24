use computegraph::fragment::FragmentBuilder;
use computegraph::types::{GlobalValKey, LocalValId, OpMode, ValRef};
use computegraph::OpEmitter;
use tenferro_ops::ad::context::ShapeGuardContext;
use tenferro_ops::std_tensor_op::StdTensorOp;

use crate::extension::LinalgOp;

use super::support::*;
use super::{conjugate_primal_if_dtype_complex, linalg_std_op};

pub(crate) fn linearize_triangular_solve(
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
        linalg_std_op(LinalgOp::TriangularSolve {
            left_side,
            lower,
            transpose_a,
            unit_diagonal,
        }),
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

#[derive(Clone, Copy)]
enum LinearSolveOp {
    Solve,
    FullPivLuSolve,
}

fn linear_solve_op(kind: LinearSolveOp, transpose_a: bool) -> StdTensorOp {
    match kind {
        LinearSolveOp::Solve => linalg_std_op(LinalgOp::Solve { transpose_a }),
        LinearSolveOp::FullPivLuSolve => linalg_std_op(LinalgOp::FullPivLuSolve { transpose_a }),
    }
}

fn linear_solve_op_name(kind: LinearSolveOp) -> &'static str {
    match kind {
        LinearSolveOp::Solve => "linearize_solve",
        LinearSolveOp::FullPivLuSolve => "linearize_full_piv_lu_solve",
    }
}

fn linearize_linear_solve(
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    primal_out: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    transpose_a: bool,
    ctx: &mut ShapeGuardContext,
    kind: LinearSolveOp,
) -> Vec<Option<LocalValId>> {
    let lhs_rank = ctx.shape_of(&ValRef::External(primal_in[0].clone())).len();
    let rhs_rank = ctx.shape_of(&ValRef::External(primal_in[1].clone())).len();
    let op_name = linear_solve_op_name(kind);
    assert!(
        lhs_rank >= 2 && rhs_rank >= 2,
        "{op_name}: expected matrix operands"
    );
    assert_eq!(
        lhs_rank, rhs_rank,
        "{op_name}: rank mismatch between lhs and rhs"
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
        linear_solve_op(kind, transpose_a),
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

pub(crate) fn linearize_solve(
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    primal_out: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    transpose_a: bool,
    ctx: &mut ShapeGuardContext,
) -> Vec<Option<LocalValId>> {
    linearize_linear_solve(
        builder,
        primal_in,
        primal_out,
        tangent_in,
        transpose_a,
        ctx,
        LinearSolveOp::Solve,
    )
}

pub(crate) fn linearize_full_piv_lu_solve(
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    primal_out: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    transpose_a: bool,
    ctx: &mut ShapeGuardContext,
) -> Vec<Option<LocalValId>> {
    linearize_linear_solve(
        builder,
        primal_in,
        primal_out,
        tangent_in,
        transpose_a,
        ctx,
        LinearSolveOp::FullPivLuSolve,
    )
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

pub(crate) fn transpose_triangular_solve(
    emitter: &mut impl OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    inputs: &[ValRef<StdTensorOp>],
    mode: &OpMode,
    left_side: bool,
    lower: bool,
    transpose_a: bool,
    unit_diagonal: bool,
    ctx: &mut ShapeGuardContext,
) -> Vec<Option<LocalValId>> {
    let Some(ct) = cotangent_out[0] else {
        return vec![None, None];
    };
    let OpMode::Linear { active_mask } = mode else {
        return vec![None, None];
    };

    let mut result = vec![None, None];
    if active_mask[0] || active_mask[1] {
        let dtype = ctx.dtype_of(&inputs[0]);
        let conjugated_a = conjugate_primal_if_dtype_complex(emitter, inputs[0].clone(), dtype);
        let out = emitter.add_op(
            linalg_std_op(LinalgOp::TriangularSolve {
                left_side,
                lower,
                transpose_a: !transpose_a,
                unit_diagonal,
            }),
            vec![conjugated_a, ValRef::Local(ct)],
            OpMode::Linear {
                active_mask: vec![false, true],
            },
        );
        let rhs_cotangent = out[0];
        if active_mask[0] {
            let rank = ctx.shape_of(&inputs[0]).len();
            let solution = emitter.add_op(
                linalg_std_op(LinalgOp::TriangularSolve {
                    left_side,
                    lower,
                    transpose_a,
                    unit_diagonal,
                }),
                inputs.to_vec(),
                OpMode::Primal,
            )[0];
            let matrix_cotangent = solve_matrix_cotangent(
                emitter,
                rhs_cotangent,
                ValRef::Local(solution),
                left_side,
                transpose_a,
                rank,
                dtype,
            );
            let matrix_cotangent =
                project_triangular_operand_linear(emitter, matrix_cotangent, lower, unit_diagonal);
            result[0] = Some(matrix_cotangent);
        }
        if active_mask[1] {
            result[1] = Some(rhs_cotangent);
        }
    }

    result
}

fn transpose_linear_solve(
    emitter: &mut impl OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    inputs: &[ValRef<StdTensorOp>],
    mode: &OpMode,
    transpose_a: bool,
    ctx: &mut ShapeGuardContext,
    kind: LinearSolveOp,
) -> Vec<Option<LocalValId>> {
    let Some(ct) = cotangent_out[0] else {
        return vec![None, None];
    };
    let OpMode::Linear { active_mask } = mode else {
        return vec![None, None];
    };

    let mut result = vec![None, None];
    if active_mask[0] || active_mask[1] {
        let dtype = ctx.dtype_of(&inputs[0]);
        let conjugated_a = conjugate_primal_if_dtype_complex(emitter, inputs[0].clone(), dtype);
        let out = emitter.add_op(
            linear_solve_op(kind, !transpose_a),
            vec![conjugated_a, ValRef::Local(ct)],
            OpMode::Linear {
                active_mask: vec![false, true],
            },
        );
        let rhs_cotangent = out[0];
        if active_mask[0] {
            let rank = ctx.shape_of(&inputs[0]).len();
            let solution = emitter.add_op(
                linear_solve_op(kind, transpose_a),
                inputs.to_vec(),
                OpMode::Primal,
            )[0];
            result[0] = Some(solve_matrix_cotangent(
                emitter,
                rhs_cotangent,
                ValRef::Local(solution),
                true,
                transpose_a,
                rank,
                dtype,
            ));
        }
        if active_mask[1] {
            result[1] = Some(rhs_cotangent);
        }
    }

    result
}

pub(crate) fn transpose_solve(
    emitter: &mut impl OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    inputs: &[ValRef<StdTensorOp>],
    mode: &OpMode,
    transpose_a: bool,
    ctx: &mut ShapeGuardContext,
) -> Vec<Option<LocalValId>> {
    transpose_linear_solve(
        emitter,
        cotangent_out,
        inputs,
        mode,
        transpose_a,
        ctx,
        LinearSolveOp::Solve,
    )
}

pub(crate) fn transpose_full_piv_lu_solve(
    emitter: &mut impl OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    inputs: &[ValRef<StdTensorOp>],
    mode: &OpMode,
    transpose_a: bool,
    ctx: &mut ShapeGuardContext,
) -> Vec<Option<LocalValId>> {
    transpose_linear_solve(
        emitter,
        cotangent_out,
        inputs,
        mode,
        transpose_a,
        ctx,
        LinearSolveOp::FullPivLuSolve,
    )
}
