use computegraph::fragment::FragmentBuilder;
use computegraph::types::{LocalValId, OpMode, ValRef};
use computegraph::OpEmitter;
use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_tensor::{DType, DotGeneralConfig, PadConfig};

use crate::extension::LinalgOp;

use super::{conjugate_linear_if_dtype_complex, conjugate_primal_if_dtype_complex, linalg_std_op};

pub(super) fn solve_matrix_cotangent(
    emitter: &mut impl OpEmitter<StdTensorOp>,
    rhs_cotangent: LocalValId,
    solution: ValRef<StdTensorOp>,
    left_side: bool,
    transpose_a: bool,
    rank: usize,
    dtype: DType,
) -> LocalValId {
    let negative_rhs_cotangent = linear_neg(emitter, rhs_cotangent);
    let solution_h = adjoint_matrix_fixed(emitter, solution, rank, dtype);
    let op_matrix_cotangent = if left_side {
        matmul_linear(
            emitter,
            ValRef::Local(negative_rhs_cotangent),
            ValRef::Local(solution_h),
            vec![true, false],
            rank,
        )
    } else {
        matmul_linear(
            emitter,
            ValRef::Local(solution_h),
            ValRef::Local(negative_rhs_cotangent),
            vec![false, true],
            rank,
        )
    };
    if transpose_a {
        transpose_matrix_linear(emitter, op_matrix_cotangent, rank)
    } else {
        op_matrix_cotangent
    }
}

pub(super) fn solve_in_graph(
    builder: &mut FragmentBuilder<StdTensorOp>,
    a: ValRef<StdTensorOp>,
    b: ValRef<StdTensorOp>,
    rank: usize,
) -> LocalValId {
    let lu_outputs = builder.add_op(linalg_std_op(LinalgOp::Lu), vec![a], OpMode::Primal);
    let p = lu_outputs[0];
    let l = lu_outputs[1];
    let u = lu_outputs[2];
    let pb = matmul_linear(builder, ValRef::Local(p), b, vec![false, true], rank);
    let z = builder.add_op(
        linalg_std_op(LinalgOp::TriangularSolve {
            left_side: true,
            lower: true,
            transpose_a: false,
            unit_diagonal: true,
        }),
        vec![ValRef::Local(l), ValRef::Local(pb)],
        OpMode::Linear {
            active_mask: vec![false, true],
        },
    )[0];
    builder.add_op(
        linalg_std_op(LinalgOp::TriangularSolve {
            left_side: true,
            lower: false,
            transpose_a: false,
            unit_diagonal: false,
        }),
        vec![ValRef::Local(u), ValRef::Local(z)],
        OpMode::Linear {
            active_mask: vec![false, true],
        },
    )[0]
}

pub(super) fn fixed_unary(
    emitter: &mut impl OpEmitter<StdTensorOp>,
    op: StdTensorOp,
    input: ValRef<StdTensorOp>,
) -> LocalValId {
    emitter.add_op(op, vec![input], OpMode::Primal)[0]
}

pub(super) fn fixed_binary(
    builder: &mut FragmentBuilder<StdTensorOp>,
    op: StdTensorOp,
    lhs: ValRef<StdTensorOp>,
    rhs: ValRef<StdTensorOp>,
) -> LocalValId {
    builder.add_op(op, vec![lhs, rhs], OpMode::Primal)[0]
}

pub(super) fn linear_unary(
    builder: &mut impl OpEmitter<StdTensorOp>,
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

pub(super) fn linear_binary(
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

pub(super) fn fixed_add(
    builder: &mut FragmentBuilder<StdTensorOp>,
    lhs: ValRef<StdTensorOp>,
    rhs: ValRef<StdTensorOp>,
) -> LocalValId {
    fixed_binary(builder, StdTensorOp::Add, lhs, rhs)
}

pub(super) fn fixed_mul(
    builder: &mut FragmentBuilder<StdTensorOp>,
    lhs: ValRef<StdTensorOp>,
    rhs: ValRef<StdTensorOp>,
) -> LocalValId {
    fixed_binary(builder, StdTensorOp::Mul, lhs, rhs)
}

pub(super) fn fixed_div(
    builder: &mut FragmentBuilder<StdTensorOp>,
    lhs: ValRef<StdTensorOp>,
    rhs: ValRef<StdTensorOp>,
) -> LocalValId {
    fixed_binary(builder, StdTensorOp::Div, lhs, rhs)
}

pub(super) fn fixed_scale(
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

pub(super) fn broadcast_in_dim_fixed(
    builder: &mut FragmentBuilder<StdTensorOp>,
    input: ValRef<StdTensorOp>,
    shape: Vec<DimExpr>,
    dims: Vec<usize>,
) -> LocalValId {
    fixed_unary(builder, StdTensorOp::BroadcastInDim { shape, dims }, input)
}

pub(super) fn reduce_sum_fixed(
    builder: &mut FragmentBuilder<StdTensorOp>,
    input: ValRef<StdTensorOp>,
    axes: Vec<usize>,
) -> LocalValId {
    fixed_unary(builder, StdTensorOp::ReduceSum { axes }, input)
}

pub(super) fn pad_fixed(
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

pub(super) fn fixed_sub(
    builder: &mut FragmentBuilder<StdTensorOp>,
    lhs: ValRef<StdTensorOp>,
    rhs: ValRef<StdTensorOp>,
) -> LocalValId {
    let neg_rhs = fixed_unary(builder, StdTensorOp::Neg, rhs);
    fixed_add(builder, lhs, ValRef::Local(neg_rhs))
}

pub(super) fn linear_add(
    builder: &mut FragmentBuilder<StdTensorOp>,
    lhs: LocalValId,
    rhs: LocalValId,
) -> LocalValId {
    linear_binary(builder, StdTensorOp::Add, lhs, rhs)
}

pub(super) fn linear_neg(
    builder: &mut impl OpEmitter<StdTensorOp>,
    input: LocalValId,
) -> LocalValId {
    linear_unary(builder, StdTensorOp::Neg, input)
}

pub(super) fn linear_scale(
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

pub(super) fn linear_sub(
    builder: &mut FragmentBuilder<StdTensorOp>,
    lhs: LocalValId,
    rhs: LocalValId,
) -> LocalValId {
    let neg_rhs = linear_neg(builder, rhs);
    linear_add(builder, lhs, neg_rhs)
}

pub(super) fn linear_div_fixed(
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

pub(super) fn hadamard_fixed_linear(
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

pub(super) fn one_like_fixed(
    builder: &mut FragmentBuilder<StdTensorOp>,
    anchor: ValRef<StdTensorOp>,
) -> LocalValId {
    let zero = fixed_sub(builder, anchor.clone(), anchor);
    fixed_unary(builder, StdTensorOp::Exp, ValRef::Local(zero))
}

pub(super) fn extract_diag_linear(
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

pub(super) fn embed_diag_linear(
    builder: &mut FragmentBuilder<StdTensorOp>,
    input: LocalValId,
) -> LocalValId {
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

pub(super) fn embed_diag_fixed(
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

pub(super) fn transpose_matrix_fixed(
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

pub(super) fn adjoint_matrix_fixed(
    builder: &mut impl OpEmitter<StdTensorOp>,
    input: ValRef<StdTensorOp>,
    rank: usize,
    dtype: DType,
) -> LocalValId {
    let input = conjugate_primal_if_dtype_complex(builder, input, dtype);
    transpose_matrix_fixed(builder, input, rank)
}

pub(super) fn adjoint_matrix_linear(
    builder: &mut FragmentBuilder<StdTensorOp>,
    input: LocalValId,
    rank: usize,
    dtype: DType,
) -> LocalValId {
    let conjugated = conjugate_linear_if_dtype_complex(builder, input, dtype);
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

pub(super) fn transpose_matrix_linear(
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

pub(super) fn project_triangular_operand_linear(
    builder: &mut impl OpEmitter<StdTensorOp>,
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

pub(super) fn self_adjoint_from_lower_linear(
    builder: &mut FragmentBuilder<StdTensorOp>,
    input: LocalValId,
    rank: usize,
    dtype: DType,
) -> LocalValId {
    let strict_lower = builder.add_op(
        StdTensorOp::Tril { k: -1 },
        vec![ValRef::Local(input)],
        OpMode::Linear {
            active_mask: vec![true],
        },
    )[0];
    let strict_lower_h = adjoint_matrix_linear(builder, strict_lower, rank, dtype);
    let offdiag = linear_add(builder, strict_lower, strict_lower_h);

    let diag = extract_diag_linear(builder, input);
    let diag = if matches!(dtype, DType::F32 | DType::F64) {
        diag
    } else {
        let diag_h = conjugate_linear_if_dtype_complex(builder, diag, dtype);
        let diag_sum = linear_add(builder, diag, diag_h);
        linear_scale(builder, diag_sum, 0.5)
    };
    let diag_mat = embed_diag_linear(builder, diag);
    linear_add(builder, offdiag, diag_mat)
}

pub(super) fn matmul_fixed(
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

pub(super) fn matmul_linear(
    builder: &mut impl OpEmitter<StdTensorOp>,
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

pub(super) fn matrix_multiply_config(rank: usize) -> DotGeneralConfig {
    assert!(rank >= 2, "matrix_multiply_config expects rank >= 2");
    let batch_dims: Vec<usize> = (2..rank).collect();
    DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: batch_dims.clone(),
        rhs_batch_dims: batch_dims,
    }
}

pub(super) fn matrix_transpose_perm(rank: usize) -> Vec<usize> {
    assert!(rank >= 2, "matrix_transpose_perm expects rank >= 2");
    let mut perm: Vec<usize> = (0..rank).collect();
    perm.swap(0, 1);
    perm
}

pub(super) fn matrix_shape_parts<'a>(
    shape: &'a [DimExpr],
    op: &str,
) -> (&'a DimExpr, &'a DimExpr, &'a [DimExpr]) {
    assert!(shape.len() >= 2, "{op}: expected rank >= 2");
    (&shape[0], &shape[1], &shape[2..])
}

pub(super) fn matrix_shape(
    rows: impl Into<DimExpr>,
    cols: impl Into<DimExpr>,
    batch_shape: &[DimExpr],
) -> Vec<DimExpr> {
    let mut shape = vec![rows.into(), cols.into()];
    shape.extend_from_slice(batch_shape);
    shape
}

pub(super) fn vector_shape(len: impl Into<DimExpr>, batch_shape: &[DimExpr]) -> Vec<DimExpr> {
    let mut shape = vec![len.into()];
    shape.extend_from_slice(batch_shape);
    shape
}

pub(super) fn vector_to_matrix_broadcast_dims(batch_ndim: usize) -> Vec<usize> {
    let mut dims = Vec::with_capacity(1 + batch_ndim);
    dims.push(1);
    dims.extend(2..(2 + batch_ndim));
    dims
}

pub(super) fn leading_column_selector_fixed(
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

pub(super) fn trailing_column_selector_fixed(
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

pub(super) fn identity_matrix_fixed(
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

pub(super) fn scalar_one_fixed(
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

pub(super) fn pad_vec(rank: usize, axis: usize, amount: i64) -> Vec<i64> {
    let mut padding = vec![0; rank];
    padding[axis] = amount;
    padding
}

pub(super) fn pad_matrix_low(rank: usize, row_amount: i64, col_amount: i64) -> Vec<i64> {
    let mut padding = vec![0; rank];
    padding[0] = row_amount;
    padding[1] = col_amount;
    padding
}

pub(super) fn augment_unit_lower_to_square_fixed(
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

pub(super) fn augment_upper_to_square_fixed(
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

pub(super) fn take_leading_cols_linear(
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

pub(super) fn take_leading_rows_linear(
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
