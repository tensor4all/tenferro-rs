use computegraph::types::{LocalValueId, OperationRole, ValueRef};
use tenferro_ops::ad::PrimitiveRuleBuilder;
use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_tensor::{DType, DotGeneralConfig, PadConfig};

use crate::extension::LinalgOp;

use super::{conjugate_linear_if_dtype_complex, conjugate_primal_if_dtype_complex, linalg_std_op};

pub(super) fn solve_matrix_cotangent(
    builder: &mut dyn PrimitiveRuleBuilder,
    rhs_cotangent: LocalValueId,
    solution: ValueRef<StdTensorOp>,
    left_side: bool,
    transpose_a: bool,
    rank: usize,
    dtype: DType,
) -> LocalValueId {
    let negative_rhs_cotangent = linear_neg(builder, rhs_cotangent);
    let solution_h = adjoint_matrix_fixed(builder, solution, rank, dtype);
    let op_matrix_cotangent = if left_side {
        matmul_linear(
            builder,
            ValueRef::Local(negative_rhs_cotangent),
            ValueRef::Local(solution_h),
            vec![true, false],
            rank,
        )
    } else {
        matmul_linear(
            builder,
            ValueRef::Local(solution_h),
            ValueRef::Local(negative_rhs_cotangent),
            vec![false, true],
            rank,
        )
    };
    if transpose_a {
        transpose_matrix_linear(builder, op_matrix_cotangent, rank)
    } else {
        op_matrix_cotangent
    }
}

pub(super) fn solve_in_graph(
    builder: &mut dyn PrimitiveRuleBuilder,
    a: ValueRef<StdTensorOp>,
    b: ValueRef<StdTensorOp>,
    _rank: usize,
) -> LocalValueId {
    let lu_outputs = builder.add_operation(
        linalg_std_op(LinalgOp::LuFactor),
        vec![a.clone()],
        OperationRole::Primary,
    );
    builder.add_operation(
        linalg_std_op(LinalgOp::LuSolvePrepared {
            transpose_a: false,
            conjugate_a: false,
        }),
        vec![
            a,
            ValueRef::Local(lu_outputs[0]),
            ValueRef::Local(lu_outputs[1]),
            b,
        ],
        OperationRole::Linearized {
            active_mask: vec![false, false, false, true],
        },
    )[0]
}

pub(super) fn fixed_unary(
    builder: &mut dyn PrimitiveRuleBuilder,
    op: StdTensorOp,
    input: ValueRef<StdTensorOp>,
) -> LocalValueId {
    builder.add_operation(op, vec![input], OperationRole::Primary)[0]
}

pub(super) fn fixed_binary(
    builder: &mut dyn PrimitiveRuleBuilder,
    op: StdTensorOp,
    lhs: ValueRef<StdTensorOp>,
    rhs: ValueRef<StdTensorOp>,
) -> LocalValueId {
    builder.add_operation(op, vec![lhs, rhs], OperationRole::Primary)[0]
}

pub(super) fn linear_unary(
    builder: &mut dyn PrimitiveRuleBuilder,
    op: StdTensorOp,
    input: LocalValueId,
) -> LocalValueId {
    builder.add_operation(
        op,
        vec![ValueRef::Local(input)],
        OperationRole::Linearized {
            active_mask: vec![true],
        },
    )[0]
}

pub(super) fn linear_binary(
    builder: &mut dyn PrimitiveRuleBuilder,
    op: StdTensorOp,
    lhs: LocalValueId,
    rhs: LocalValueId,
) -> LocalValueId {
    builder.add_operation(
        op,
        vec![ValueRef::Local(lhs), ValueRef::Local(rhs)],
        OperationRole::Linearized {
            active_mask: vec![true, true],
        },
    )[0]
}

pub(super) fn fixed_add(
    builder: &mut dyn PrimitiveRuleBuilder,
    lhs: ValueRef<StdTensorOp>,
    rhs: ValueRef<StdTensorOp>,
) -> LocalValueId {
    fixed_binary(builder, StdTensorOp::Add, lhs, rhs)
}

pub(super) fn fixed_mul(
    builder: &mut dyn PrimitiveRuleBuilder,
    lhs: ValueRef<StdTensorOp>,
    rhs: ValueRef<StdTensorOp>,
) -> LocalValueId {
    fixed_binary(builder, StdTensorOp::Mul, lhs, rhs)
}

pub(super) fn fixed_div(
    builder: &mut dyn PrimitiveRuleBuilder,
    lhs: ValueRef<StdTensorOp>,
    rhs: ValueRef<StdTensorOp>,
) -> LocalValueId {
    fixed_binary(builder, StdTensorOp::Div, lhs, rhs)
}

pub(super) fn fixed_scale(
    builder: &mut dyn PrimitiveRuleBuilder,
    input: ValueRef<StdTensorOp>,
    factor: f64,
) -> LocalValueId {
    let constant = builder.add_operation(
        StdTensorOp::constant(factor),
        vec![],
        OperationRole::Primary,
    );
    builder.add_operation(
        StdTensorOp::Mul,
        vec![ValueRef::Local(constant[0]), input],
        OperationRole::Primary,
    )[0]
}

pub(super) fn broadcast_in_dim_fixed(
    builder: &mut dyn PrimitiveRuleBuilder,
    input: ValueRef<StdTensorOp>,
    shape: Vec<DimExpr>,
    dims: Vec<usize>,
) -> LocalValueId {
    fixed_unary(builder, StdTensorOp::BroadcastInDim { shape, dims }, input)
}

pub(super) fn reduce_sum_fixed(
    builder: &mut dyn PrimitiveRuleBuilder,
    input: ValueRef<StdTensorOp>,
    axes: Vec<usize>,
) -> LocalValueId {
    fixed_unary(builder, StdTensorOp::ReduceSum { axes }, input)
}

pub(super) fn pad_fixed(
    builder: &mut dyn PrimitiveRuleBuilder,
    input: ValueRef<StdTensorOp>,
    rank: usize,
    edge_padding_low: Vec<i64>,
    edge_padding_high: Vec<i64>,
) -> LocalValueId {
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
    builder: &mut dyn PrimitiveRuleBuilder,
    lhs: ValueRef<StdTensorOp>,
    rhs: ValueRef<StdTensorOp>,
) -> LocalValueId {
    let neg_rhs = fixed_unary(builder, StdTensorOp::Neg, rhs);
    fixed_add(builder, lhs, ValueRef::Local(neg_rhs))
}

pub(super) fn linear_add(
    builder: &mut dyn PrimitiveRuleBuilder,
    lhs: LocalValueId,
    rhs: LocalValueId,
) -> LocalValueId {
    linear_binary(builder, StdTensorOp::Add, lhs, rhs)
}

pub(super) fn linear_neg(
    builder: &mut dyn PrimitiveRuleBuilder,
    input: LocalValueId,
) -> LocalValueId {
    linear_unary(builder, StdTensorOp::Neg, input)
}

pub(super) fn linear_scale(
    builder: &mut dyn PrimitiveRuleBuilder,
    input: LocalValueId,
    factor: f64,
) -> LocalValueId {
    let constant = builder.add_operation(
        StdTensorOp::constant(factor),
        vec![],
        OperationRole::Primary,
    );
    builder.add_operation(
        StdTensorOp::Mul,
        vec![ValueRef::Local(constant[0]), ValueRef::Local(input)],
        OperationRole::Linearized {
            active_mask: vec![false, true],
        },
    )[0]
}

pub(super) fn linear_sub(
    builder: &mut dyn PrimitiveRuleBuilder,
    lhs: LocalValueId,
    rhs: LocalValueId,
) -> LocalValueId {
    let neg_rhs = linear_neg(builder, rhs);
    linear_add(builder, lhs, neg_rhs)
}

pub(super) fn linear_div_fixed(
    builder: &mut dyn PrimitiveRuleBuilder,
    lhs: LocalValueId,
    rhs: ValueRef<StdTensorOp>,
) -> LocalValueId {
    builder.add_operation(
        StdTensorOp::Div,
        vec![ValueRef::Local(lhs), rhs],
        OperationRole::Linearized {
            active_mask: vec![true, false],
        },
    )[0]
}

pub(super) fn hadamard_fixed_linear(
    builder: &mut dyn PrimitiveRuleBuilder,
    fixed: ValueRef<StdTensorOp>,
    active: LocalValueId,
) -> LocalValueId {
    builder.add_operation(
        StdTensorOp::Mul,
        vec![fixed, ValueRef::Local(active)],
        OperationRole::Linearized {
            active_mask: vec![false, true],
        },
    )[0]
}

pub(super) fn one_like_fixed(
    builder: &mut dyn PrimitiveRuleBuilder,
    anchor: ValueRef<StdTensorOp>,
) -> LocalValueId {
    let zero = fixed_sub(builder, anchor.clone(), anchor);
    fixed_unary(builder, StdTensorOp::Exp, ValueRef::Local(zero))
}

pub(super) fn extract_diag_linear(
    builder: &mut dyn PrimitiveRuleBuilder,
    input: LocalValueId,
) -> LocalValueId {
    builder.add_operation(
        StdTensorOp::ExtractDiag {
            axis_a: 0,
            axis_b: 1,
        },
        vec![ValueRef::Local(input)],
        OperationRole::Linearized {
            active_mask: vec![true],
        },
    )[0]
}

pub(super) fn embed_diag_linear(
    builder: &mut dyn PrimitiveRuleBuilder,
    input: LocalValueId,
) -> LocalValueId {
    builder.add_operation(
        StdTensorOp::EmbedDiag {
            axis_a: 0,
            axis_b: 1,
        },
        vec![ValueRef::Local(input)],
        OperationRole::Linearized {
            active_mask: vec![true],
        },
    )[0]
}

pub(super) fn embed_diag_fixed(
    builder: &mut dyn PrimitiveRuleBuilder,
    input: ValueRef<StdTensorOp>,
) -> LocalValueId {
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
    builder: &mut dyn PrimitiveRuleBuilder,
    input: ValueRef<StdTensorOp>,
    rank: usize,
) -> LocalValueId {
    fixed_unary(
        builder,
        StdTensorOp::Transpose {
            perm: matrix_transpose_perm(rank),
        },
        input,
    )
}

pub(super) fn adjoint_matrix_fixed(
    builder: &mut dyn PrimitiveRuleBuilder,
    input: ValueRef<StdTensorOp>,
    rank: usize,
    dtype: DType,
) -> LocalValueId {
    let input = conjugate_primal_if_dtype_complex(builder, input, dtype);
    transpose_matrix_fixed(builder, input, rank)
}

pub(super) fn adjoint_matrix_linear(
    builder: &mut dyn PrimitiveRuleBuilder,
    input: LocalValueId,
    rank: usize,
    dtype: DType,
) -> LocalValueId {
    let conjugated = conjugate_linear_if_dtype_complex(builder, input, dtype);
    builder.add_operation(
        StdTensorOp::Transpose {
            perm: matrix_transpose_perm(rank),
        },
        vec![ValueRef::Local(conjugated)],
        OperationRole::Linearized {
            active_mask: vec![true],
        },
    )[0]
}

pub(super) fn transpose_matrix_linear(
    builder: &mut dyn PrimitiveRuleBuilder,
    input: LocalValueId,
    rank: usize,
) -> LocalValueId {
    builder.add_operation(
        StdTensorOp::Transpose {
            perm: matrix_transpose_perm(rank),
        },
        vec![ValueRef::Local(input)],
        OperationRole::Linearized {
            active_mask: vec![true],
        },
    )[0]
}

pub(super) fn project_triangular_operand_linear(
    builder: &mut dyn PrimitiveRuleBuilder,
    input: LocalValueId,
    lower: bool,
    unit_diagonal: bool,
) -> LocalValueId {
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
    builder: &mut dyn PrimitiveRuleBuilder,
    input: LocalValueId,
    rank: usize,
    dtype: DType,
) -> LocalValueId {
    let strict_lower = builder.add_operation(
        StdTensorOp::Tril { k: -1 },
        vec![ValueRef::Local(input)],
        OperationRole::Linearized {
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
    builder: &mut dyn PrimitiveRuleBuilder,
    lhs: ValueRef<StdTensorOp>,
    rhs: ValueRef<StdTensorOp>,
    rank: usize,
) -> LocalValueId {
    builder.add_operation(
        StdTensorOp::DotGeneral {
            config: matrix_multiply_config(rank),
        },
        vec![lhs, rhs],
        OperationRole::Primary,
    )[0]
}

pub(super) fn matmul_linear(
    builder: &mut dyn PrimitiveRuleBuilder,
    lhs: ValueRef<StdTensorOp>,
    rhs: ValueRef<StdTensorOp>,
    active_mask: Vec<bool>,
    rank: usize,
) -> LocalValueId {
    builder.add_operation(
        StdTensorOp::DotGeneral {
            config: matrix_multiply_config(rank),
        },
        vec![lhs, rhs],
        OperationRole::Linearized { active_mask },
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
    builder: &mut dyn PrimitiveRuleBuilder,
    leading_cols: usize,
    total_cols: usize,
    batch_shape: &[DimExpr],
    anchor: ValueRef<StdTensorOp>,
    anchor_shape: &[DimExpr],
) -> LocalValueId {
    let eye = identity_matrix_fixed(builder, leading_cols, batch_shape, anchor, anchor_shape);
    let rank = 2 + batch_shape.len();
    pad_fixed(
        builder,
        ValueRef::Local(eye),
        rank,
        vec![0; rank],
        pad_vec(rank, 1, (total_cols - leading_cols) as i64),
    )
}

pub(super) fn trailing_column_selector_fixed(
    builder: &mut dyn PrimitiveRuleBuilder,
    leading_cols: usize,
    trailing_cols: usize,
    batch_shape: &[DimExpr],
    anchor: ValueRef<StdTensorOp>,
    anchor_shape: &[DimExpr],
) -> LocalValueId {
    let eye = identity_matrix_fixed(builder, trailing_cols, batch_shape, anchor, anchor_shape);
    let rank = 2 + batch_shape.len();
    pad_fixed(
        builder,
        ValueRef::Local(eye),
        rank,
        pad_vec(rank, 1, leading_cols as i64),
        vec![0; rank],
    )
}

pub(super) fn identity_matrix_fixed(
    builder: &mut dyn PrimitiveRuleBuilder,
    size: usize,
    batch_shape: &[DimExpr],
    anchor: ValueRef<StdTensorOp>,
    anchor_shape: &[DimExpr],
) -> LocalValueId {
    let one_scalar = scalar_one_fixed(builder, anchor, anchor_shape);
    let ones = broadcast_in_dim_fixed(
        builder,
        ValueRef::Local(one_scalar),
        vector_shape(size, batch_shape),
        vec![],
    );
    embed_diag_fixed(builder, ValueRef::Local(ones))
}

pub(super) fn scalar_one_fixed(
    builder: &mut dyn PrimitiveRuleBuilder,
    anchor: ValueRef<StdTensorOp>,
    anchor_shape: &[DimExpr],
) -> LocalValueId {
    let zero = fixed_sub(builder, anchor.clone(), anchor);
    let zero_scalar = if anchor_shape.is_empty() {
        zero
    } else {
        reduce_sum_fixed(
            builder,
            ValueRef::Local(zero),
            (0..anchor_shape.len()).collect(),
        )
    };
    fixed_unary(builder, StdTensorOp::Exp, ValueRef::Local(zero_scalar))
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
    builder: &mut dyn PrimitiveRuleBuilder,
    l: ValueRef<StdTensorOp>,
    rows: usize,
    cols: usize,
    batch_shape: &[DimExpr],
    l_shape: &[DimExpr],
    rank: usize,
) -> LocalValueId {
    let strict_lower = fixed_unary(builder, StdTensorOp::Tril { k: -1 }, l.clone());
    let strict_lower_square = if cols < rows {
        pad_fixed(
            builder,
            ValueRef::Local(strict_lower),
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
        ValueRef::Local(strict_lower_square),
        ValueRef::Local(identity),
    )
}

pub(super) fn augment_upper_to_square_fixed(
    builder: &mut dyn PrimitiveRuleBuilder,
    u: ValueRef<StdTensorOp>,
    rows: usize,
    cols: usize,
    batch_shape: &[DimExpr],
    u_shape: &[DimExpr],
    rank: usize,
) -> LocalValueId {
    let upper = fixed_unary(builder, StdTensorOp::Triu { k: 0 }, u.clone());
    if rows == cols {
        return upper;
    }

    let upper_square = pad_fixed(
        builder,
        ValueRef::Local(upper),
        rank,
        vec![0; rank],
        pad_vec(rank, 0, (cols - rows) as i64),
    );
    let trailing_eye = identity_matrix_fixed(builder, cols - rows, batch_shape, u, u_shape);
    let trailing_eye = pad_fixed(
        builder,
        ValueRef::Local(trailing_eye),
        rank,
        pad_matrix_low(rank, rows as i64, rows as i64),
        vec![0; rank],
    );
    fixed_add(
        builder,
        ValueRef::Local(upper_square),
        ValueRef::Local(trailing_eye),
    )
}

pub(super) fn take_leading_cols_linear(
    builder: &mut dyn PrimitiveRuleBuilder,
    input: LocalValueId,
    cols: usize,
    total_cols: usize,
    batch_shape: &[DimExpr],
    anchor: ValueRef<StdTensorOp>,
    anchor_shape: &[DimExpr],
    rank: usize,
) -> LocalValueId {
    let selector =
        leading_column_selector_fixed(builder, cols, total_cols, batch_shape, anchor, anchor_shape);
    let selector_t = transpose_matrix_fixed(builder, ValueRef::Local(selector), rank);
    matmul_linear(
        builder,
        ValueRef::Local(input),
        ValueRef::Local(selector_t),
        vec![true, false],
        rank,
    )
}

pub(super) fn take_leading_rows_linear(
    builder: &mut dyn PrimitiveRuleBuilder,
    input: LocalValueId,
    rows: usize,
    total_rows: usize,
    batch_shape: &[DimExpr],
    anchor: ValueRef<StdTensorOp>,
    anchor_shape: &[DimExpr],
    rank: usize,
) -> LocalValueId {
    let selector =
        leading_column_selector_fixed(builder, rows, total_rows, batch_shape, anchor, anchor_shape);
    matmul_linear(
        builder,
        ValueRef::Local(selector),
        ValueRef::Local(input),
        vec![false, true],
        rank,
    )
}
