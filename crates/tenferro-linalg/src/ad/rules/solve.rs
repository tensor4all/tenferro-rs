use computegraph::types::{LocalValueId, OperationRole, ValueKey, ValueRef};
use tenferro_ops::ad::context::ShapeGuardContext;
use tenferro_ops::ad::PrimitiveRuleBuilder;
use tenferro_ops::ad::{ADRuleError, ADRuleKind, ADRuleResult};
use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::shape_extent::ShapeExtent;
use tenferro_ops::std_tensor_op::StdTensorOp;

use crate::extension::LinalgOp;

use super::support::*;
use super::{conjugate_linear_if_dtype_complex, conjugate_primal_if_dtype_complex, linalg_std_op};

#[derive(Clone, Copy)]
pub(crate) struct TriangularSolveFlags {
    pub(crate) left_side: bool,
    pub(crate) lower: bool,
    pub(crate) transpose_a: bool,
    pub(crate) unit_diagonal: bool,
}

impl TriangularSolveFlags {
    pub(crate) fn new(
        left_side: bool,
        lower: bool,
        transpose_a: bool,
        unit_diagonal: bool,
    ) -> Self {
        Self {
            left_side,
            lower,
            transpose_a,
            unit_diagonal,
        }
    }

    fn transposed(self) -> Self {
        Self {
            transpose_a: !self.transpose_a,
            ..self
        }
    }

    fn std_op(self) -> StdTensorOp {
        linalg_std_op(LinalgOp::TriangularSolve {
            left_side: self.left_side,
            lower: self.lower,
            transpose_a: self.transpose_a,
            unit_diagonal: self.unit_diagonal,
        })
    }
}

pub(crate) fn linearize_triangular_solve(
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    flags: TriangularSolveFlags,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    // Equation: op(A) @ X = B  (left_side=true)
    //       or  X @ op(A) = B  (left_side=false)
    // where op = identity (transpose_a=false) or transpose (transpose_a=true).
    //
    // Linearize: op(A) @ dX = dB - d(op(A)) @ X  (left_side=true)
    //        or  dX @ op(A) = dB - X @ d(op(A))  (left_side=false)
    //
    // When tangent_in[0] (dA) is present, we compute the correction:
    //   -d(op(A)) @ X  or  -X @ d(op(A))
    let lhs_ref = ValueRef::External(primal_in[0].clone());
    let rhs_ref = ValueRef::External(primal_in[1].clone());
    let lhs_rank = ctx.rank_of(&lhs_ref)?;
    let rhs_rank = ctx.rank_of(&rhs_ref)?;
    validate_matrix_operands("triangular_solve", ADRuleKind::Jvp, lhs_rank, rhs_rank)?;
    validate_square_matrix_input("triangular_solve", ADRuleKind::Jvp, &lhs_ref, ctx)?;
    let rank = lhs_rank;
    let rhs_tangent = triangular_solve_rhs_tangent(builder, primal_out, tangent_in, flags, rank);
    let Some(rhs_tangent) = rhs_tangent else {
        return Ok(vec![None]);
    };

    let out = builder.add_operation(
        flags.std_op(),
        vec![
            ValueRef::External(primal_in[0].clone()),
            ValueRef::Local(rhs_tangent),
        ],
        OperationRole::Linearized {
            active_mask: vec![false, true],
        },
    );
    Ok(vec![Some(out[0])])
}

#[derive(Clone, Copy)]
enum LinearSolveOp {
    FullPivLuSolve,
    Solve,
}

fn linear_solve_op(kind: LinearSolveOp, transpose_a: bool) -> StdTensorOp {
    match kind {
        LinearSolveOp::FullPivLuSolve => linalg_std_op(LinalgOp::FullPivLuSolve { transpose_a }),
        // The plain partial-pivot solve has no transpose flag; the adjoint
        // solve transposes the matrix before the plain solve (see
        // `transpose_linear_solve`), so `transpose_a` is always false here.
        LinearSolveOp::Solve => linalg_std_op(LinalgOp::Solve),
    }
}

fn linearize_linear_solve(
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    transpose_a: bool,
    ctx: &mut ShapeGuardContext,
    kind: LinearSolveOp,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let lhs_ref = ValueRef::External(primal_in[0].clone());
    let rhs_ref = ValueRef::External(primal_in[1].clone());
    let lhs_rank = ctx.rank_of(&lhs_ref)?;
    let rhs_rank = ctx.rank_of(&rhs_ref)?;
    validate_matrix_operands(kind.op_name(), ADRuleKind::Jvp, lhs_rank, rhs_rank)?;
    validate_square_matrix_input(kind.op_name(), ADRuleKind::Jvp, &lhs_ref, ctx)?;
    let rank = lhs_rank;
    let mut rhs_tangent = tangent_in[1];

    if let Some(da) = tangent_in[0] {
        let d_op_a = if transpose_a {
            transpose_matrix_linear(builder, da, rank)
        } else {
            da
        };
        let x = ValueRef::External(primal_out[0].clone());
        let correction =
            matmul_linear(builder, ValueRef::Local(d_op_a), x, vec![true, false], rank);
        let neg_correction = linear_neg(builder, correction);
        rhs_tangent = Some(match rhs_tangent {
            Some(db) => linear_add(builder, db, neg_correction),
            None => neg_correction,
        });
    }

    let Some(rhs_tangent) = rhs_tangent else {
        return Ok(vec![None]);
    };

    let out = builder.add_operation(
        linear_solve_op(kind, transpose_a),
        vec![
            ValueRef::External(primal_in[0].clone()),
            ValueRef::Local(rhs_tangent),
        ],
        OperationRole::Linearized {
            active_mask: vec![false, true],
        },
    );
    Ok(vec![Some(out[0])])
}

pub(crate) fn linearize_lu_solve_prepared(
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    transpose_a: bool,
    conjugate_a: bool,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let lhs_ref = ValueRef::External(primal_in[0].clone());
    let rhs_ref = ValueRef::External(primal_in[3].clone());
    let lhs_rank = ctx.rank_of(&lhs_ref)?;
    let rhs_rank = ctx.rank_of(&rhs_ref)?;
    validate_matrix_operands("lu_solve_prepared", ADRuleKind::Jvp, lhs_rank, rhs_rank)?;
    validate_square_matrix_input("lu_solve_prepared", ADRuleKind::Jvp, &lhs_ref, ctx)?;
    let rank = lhs_rank;
    let mut rhs_tangent = tangent_in[3];

    if let Some(da) = tangent_in[0] {
        let dtype = ctx.dtype_of(&ValueRef::External(primal_in[0].clone()))?;
        let d_op_a = match (transpose_a, conjugate_a) {
            (false, false) => da,
            (true, false) => transpose_matrix_linear(builder, da, rank),
            (true, true) => adjoint_matrix_linear(builder, da, rank, dtype),
            (false, true) => conjugate_linear_if_dtype_complex(builder, da, dtype),
        };
        let x = ValueRef::External(primal_out[0].clone());
        let correction =
            matmul_linear(builder, ValueRef::Local(d_op_a), x, vec![true, false], rank);
        let neg_correction = linear_neg(builder, correction);
        rhs_tangent = Some(match rhs_tangent {
            Some(db) => linear_add(builder, db, neg_correction),
            None => neg_correction,
        });
    }

    let Some(rhs_tangent) = rhs_tangent else {
        return Ok(vec![None]);
    };

    let out = builder.add_operation(
        linalg_std_op(LinalgOp::LuSolvePrepared {
            transpose_a,
            conjugate_a,
        }),
        vec![
            ValueRef::External(primal_in[0].clone()),
            ValueRef::External(primal_in[1].clone()),
            ValueRef::External(primal_in[2].clone()),
            ValueRef::Local(rhs_tangent),
        ],
        OperationRole::Linearized {
            active_mask: vec![false, false, false, true],
        },
    );
    Ok(vec![Some(out[0])])
}

pub(crate) fn linearize_full_piv_lu_solve(
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    transpose_a: bool,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
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

pub(crate) fn linearize_solve(
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    transpose_a: bool,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
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

fn invalid_input(op: &'static str, rule: ADRuleKind, message: impl Into<String>) -> ADRuleError {
    ADRuleError::invalid_input(format!("{LINALG_AD_OP_PREFIX}{op}"), rule, message)
}

const LINALG_AD_OP_PREFIX: &str = "tenferro-linalg.";

impl LinearSolveOp {
    fn op_name(self) -> &'static str {
        match self {
            LinearSolveOp::FullPivLuSolve => "full_piv_lu_solve",
            LinearSolveOp::Solve => "solve",
        }
    }
}

fn validate_matrix_operands(
    op: &'static str,
    rule: ADRuleKind,
    lhs_rank: usize,
    rhs_rank: usize,
) -> ADRuleResult<()> {
    if lhs_rank < 2 || rhs_rank < 2 {
        return Err(invalid_input(
            op,
            rule,
            format!("expected matrix operands with rank >= 2, got ranks {lhs_rank} and {rhs_rank}"),
        ));
    }
    if lhs_rank != rhs_rank {
        return Err(invalid_input(
            op,
            rule,
            format!("expected matrix operands with matching ranks, got {lhs_rank} and {rhs_rank}"),
        ));
    }
    Ok(())
}

fn validate_square_matrix_input(
    op: &'static str,
    rule: ADRuleKind,
    input: &ValueRef<StdTensorOp>,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<()> {
    let extents = ctx.extents_of(input)?;
    let (Some(rows), Some(cols)) = (
        exact_constant_extent(&extents[0]),
        exact_constant_extent(&extents[1]),
    ) else {
        return Ok(());
    };
    let (rows_size, cols_size) = tenferro_ops::ad::context::resolve_and_guard(
        &DimExpr::Const(rows),
        &DimExpr::Const(cols),
        ctx,
    )
    .map_err(|err| {
        invalid_input(
            op,
            rule,
            format!("invalid matrix dimension expression: {err}"),
        )
    })?;
    if rows_size != cols_size {
        return Err(invalid_input(
            op,
            rule,
            format!("expected square matrix operand, got {rows_size}x{cols_size}"),
        ));
    }
    Ok(())
}

fn exact_constant_extent(extent: &ShapeExtent<tenferro_ops::SymDim>) -> Option<usize> {
    match extent {
        ShapeExtent::Exact(dim) => dim.constant_value(),
        ShapeExtent::UpperBound(_) | ShapeExtent::Unknown => None,
    }
}

fn triangular_solve_rhs_tangent(
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    flags: TriangularSolveFlags,
    rank: usize,
) -> Option<LocalValueId> {
    let mut rhs_tangent = tangent_in[1];

    if let Some(da) = tangent_in[0] {
        let da = project_triangular_operand_linear(builder, da, flags.lower, flags.unit_diagonal);
        // d(op(A)) = op(dA), with op = identity or transpose.
        let d_op_a = if flags.transpose_a {
            transpose_matrix_linear(builder, da, rank)
        } else {
            da
        };

        // Correction = d(op(A)) @ X  or  X @ d(op(A))
        let x = ValueRef::External(primal_out[0].clone());
        let correction = if flags.left_side {
            matmul_linear(builder, ValueRef::Local(d_op_a), x, vec![true, false], rank)
        } else {
            matmul_linear(builder, x, ValueRef::Local(d_op_a), vec![false, true], rank)
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
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[ValueRef<StdTensorOp>],
    mode: &OperationRole,
    flags: TriangularSolveFlags,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let Some(ct) = cotangent_out[0] else {
        return Ok(vec![None, None]);
    };
    let OperationRole::Linearized { active_mask } = mode else {
        return Ok(vec![None, None]);
    };

    let mut result = vec![None, None];
    if active_mask[0] || active_mask[1] {
        let rank = ctx.rank_of(&inputs[0])?;
        let rhs_rank = ctx.rank_of(&inputs[1])?;
        validate_matrix_operands("triangular_solve", ADRuleKind::Transpose, rank, rhs_rank)?;
        validate_square_matrix_input("triangular_solve", ADRuleKind::Transpose, &inputs[0], ctx)?;
        let dtype = ctx.dtype_of(&inputs[0])?;
        let conjugated_a = conjugate_primal_if_dtype_complex(builder, inputs[0].clone(), dtype);
        let out = builder.add_operation(
            flags.transposed().std_op(),
            vec![conjugated_a, ValueRef::Local(ct)],
            OperationRole::Linearized {
                active_mask: vec![false, true],
            },
        );
        let rhs_cotangent = out[0];
        if active_mask[0] {
            let solution = builder.add_operation(
                linalg_std_op(LinalgOp::TriangularSolve {
                    left_side: flags.left_side,
                    lower: flags.lower,
                    transpose_a: flags.transpose_a,
                    unit_diagonal: flags.unit_diagonal,
                }),
                inputs.to_vec(),
                OperationRole::Primary,
            )[0];
            let matrix_cotangent = solve_matrix_cotangent(
                builder,
                rhs_cotangent,
                ValueRef::Local(solution),
                flags.left_side,
                flags.transpose_a,
                rank,
                dtype,
            );
            let matrix_cotangent = project_triangular_operand_linear(
                builder,
                matrix_cotangent,
                flags.lower,
                flags.unit_diagonal,
            );
            result[0] = Some(matrix_cotangent);
        }
        if active_mask[1] {
            result[1] = Some(rhs_cotangent);
        }
    }

    Ok(result)
}

fn transpose_linear_solve(
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[ValueRef<StdTensorOp>],
    mode: &OperationRole,
    transpose_a: bool,
    ctx: &mut ShapeGuardContext,
    kind: LinearSolveOp,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let Some(ct) = cotangent_out[0] else {
        return Ok(vec![None, None]);
    };
    let OperationRole::Linearized { active_mask } = mode else {
        return Ok(vec![None, None]);
    };

    let mut result = vec![None, None];
    if active_mask[0] || active_mask[1] {
        let rank = ctx.rank_of(&inputs[0])?;
        let rhs_rank = ctx.rank_of(&inputs[1])?;
        validate_matrix_operands(kind.op_name(), ADRuleKind::Transpose, rank, rhs_rank)?;
        validate_square_matrix_input(kind.op_name(), ADRuleKind::Transpose, &inputs[0], ctx)?;
        let dtype = ctx.dtype_of(&inputs[0])?;
        let conjugated_a = conjugate_primal_if_dtype_complex(builder, inputs[0].clone(), dtype);
        // `LinalgOp::Solve` carries no transpose flag (partial-pivot plain
        // solve), so the adjoint solve op(A)^T y = ct is expressed as a plain
        // solve of the transposed (and, for complex dtypes, conjugated)
        // matrix instead of a transposed-solve op.
        let (adjoint_a, adjoint_transpose_a) = match kind {
            LinearSolveOp::Solve => {
                let transposed = transpose_matrix_fixed(builder, conjugated_a, rank);
                (ValueRef::Local(transposed), false)
            }
            LinearSolveOp::FullPivLuSolve => (conjugated_a, !transpose_a),
        };
        let out = builder.add_operation(
            linear_solve_op(kind, adjoint_transpose_a),
            vec![adjoint_a, ValueRef::Local(ct)],
            OperationRole::Linearized {
                active_mask: vec![false, true],
            },
        );
        let rhs_cotangent = out[0];
        if active_mask[0] {
            let solution = builder.add_operation(
                linear_solve_op(kind, transpose_a),
                inputs.to_vec(),
                OperationRole::Primary,
            )[0];
            result[0] = Some(solve_matrix_cotangent(
                builder,
                rhs_cotangent,
                ValueRef::Local(solution),
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

    Ok(result)
}

pub(crate) fn transpose_lu_solve_prepared(
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[ValueRef<StdTensorOp>],
    mode: &OperationRole,
    transpose_a: bool,
    conjugate_a: bool,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let Some(ct) = cotangent_out[0] else {
        return Ok(vec![None, None, None, None]);
    };
    let OperationRole::Linearized { active_mask } = mode else {
        return Ok(vec![None, None, None, None]);
    };
    if active_mask.len() != 4 {
        return Err(invalid_input(
            "lu_solve_prepared",
            ADRuleKind::Transpose,
            format!(
                "expected 4 active-mask entries for prepared LU solve, got {}",
                active_mask.len()
            ),
        ));
    }
    let mut result = vec![None, None, None, None];
    if active_mask[0] || active_mask[3] {
        // Packed LU may be an active intermediate when `solve` is lowered
        // through factorization. Pivot/parity slots are fixed residuals.
        let rank = ctx.rank_of(&inputs[0])?;
        let rhs_rank = ctx.rank_of(&inputs[3])?;
        validate_matrix_operands("lu_solve_prepared", ADRuleKind::Transpose, rank, rhs_rank)?;
        validate_square_matrix_input("lu_solve_prepared", ADRuleKind::Transpose, &inputs[0], ctx)?;
        let dtype = ctx.dtype_of(&inputs[0])?;
        let (adjoint_transpose_a, adjoint_conjugate_a) =
            adjoint_lu_solve_flags(transpose_a, conjugate_a);
        let out = builder.add_operation(
            linalg_std_op(LinalgOp::LuSolvePrepared {
                transpose_a: adjoint_transpose_a,
                conjugate_a: adjoint_conjugate_a,
            }),
            vec![
                inputs[0].clone(),
                inputs[1].clone(),
                inputs[2].clone(),
                ValueRef::Local(ct),
            ],
            OperationRole::Linearized {
                active_mask: vec![false, false, false, true],
            },
        );
        let rhs_cotangent = out[0];
        if active_mask[0] {
            let solution = builder.add_operation(
                linalg_std_op(LinalgOp::LuSolvePrepared {
                    transpose_a,
                    conjugate_a,
                }),
                inputs.to_vec(),
                OperationRole::Primary,
            )[0];
            let op_matrix_cotangent = solve_matrix_cotangent(
                builder,
                rhs_cotangent,
                ValueRef::Local(solution),
                true,
                false,
                rank,
                dtype,
            );
            result[0] = Some(adjoint_lu_operand_cotangent(
                builder,
                op_matrix_cotangent,
                transpose_a,
                conjugate_a,
                rank,
                dtype,
            ));
        }
        if active_mask[3] {
            result[3] = Some(rhs_cotangent);
        }
    }
    Ok(result)
}

fn adjoint_lu_operand_cotangent(
    builder: &mut dyn PrimitiveRuleBuilder,
    op_matrix_cotangent: LocalValueId,
    transpose_a: bool,
    conjugate_a: bool,
    rank: usize,
    dtype: tenferro_tensor::DType,
) -> LocalValueId {
    match (transpose_a, conjugate_a) {
        (false, false) => op_matrix_cotangent,
        (true, false) => transpose_matrix_linear(builder, op_matrix_cotangent, rank),
        (false, true) => conjugate_linear_if_dtype_complex(builder, op_matrix_cotangent, dtype),
        (true, true) => adjoint_matrix_linear(builder, op_matrix_cotangent, rank, dtype),
    }
}

pub(crate) fn transpose_full_piv_lu_solve(
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[ValueRef<StdTensorOp>],
    mode: &OperationRole,
    transpose_a: bool,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    transpose_linear_solve(
        builder,
        cotangent_out,
        inputs,
        mode,
        transpose_a,
        ctx,
        LinearSolveOp::FullPivLuSolve,
    )
}

pub(crate) fn transpose_solve(
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[ValueRef<StdTensorOp>],
    mode: &OperationRole,
    transpose_a: bool,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    transpose_linear_solve(
        builder,
        cotangent_out,
        inputs,
        mode,
        transpose_a,
        ctx,
        LinearSolveOp::Solve,
    )
}

fn adjoint_lu_solve_flags(transpose_a: bool, conjugate_a: bool) -> (bool, bool) {
    (!transpose_a, !conjugate_a)
}

#[cfg(test)]
mod tests;
