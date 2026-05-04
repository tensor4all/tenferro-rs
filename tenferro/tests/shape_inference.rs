use tenferro::shape_infer::{infer_output_extents, infer_output_shapes};
use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_ops::ShapeExtent;
use tenferro_tensor::{DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig, SliceConfig};

fn cst(n: usize) -> DimExpr {
    DimExpr::Const(n)
}

#[test]
fn test_add_broadcasts_trivially() {
    let op = StdTensorOp::Add;
    let lhs = vec![cst(3), cst(4)];
    let rhs = vec![cst(3), cst(4)];
    let out = infer_output_shapes(&op, &[&lhs, &rhs]);
    assert_eq!(out, vec![vec![cst(3), cst(4)]]);
}

#[test]
fn test_add_and_mul_preserve_tensor_shape_under_scalar_broadcast() {
    let scalar = vec![];
    let vector = vec![cst(3)];
    let matrix = vec![cst(2), cst(4)];

    for (op, tensor_shape) in [
        (StdTensorOp::Add, vector.clone()),
        (StdTensorOp::Mul, vector.clone()),
        (StdTensorOp::Add, matrix.clone()),
        (StdTensorOp::Mul, matrix.clone()),
    ] {
        assert_eq!(
            infer_output_shapes(&op, &[&scalar, &tensor_shape]),
            vec![tensor_shape.clone()],
            "unexpected shape for left-scalar {op:?}"
        );
        assert_eq!(
            infer_output_shapes(&op, &[&tensor_shape, &scalar]),
            vec![tensor_shape.clone()],
            "unexpected shape for right-scalar {op:?}"
        );
    }
}

#[test]
fn test_add_infers_non_scalar_broadcast_shape() {
    let lhs = vec![cst(3), cst(1)];
    let rhs = vec![cst(1), cst(4)];
    let out = infer_output_shapes(&StdTensorOp::Add, &[&lhs, &rhs]);
    assert_eq!(out, vec![vec![cst(3), cst(4)]]);
}

#[test]
fn test_mul_infers_non_scalar_broadcast_shape() {
    let lhs = vec![cst(3), cst(1)];
    let rhs = vec![cst(1), cst(4)];
    let out = infer_output_shapes(&StdTensorOp::Mul, &[&lhs, &rhs]);
    assert_eq!(out, vec![vec![cst(3), cst(4)]]);
}

#[test]
fn test_transpose_applies_perm() {
    let op = StdTensorOp::Transpose { perm: vec![1, 0] };
    let input = vec![cst(3), cst(4)];
    let out = infer_output_shapes(&op, &[&input]);
    assert_eq!(out, vec![vec![cst(4), cst(3)]]);
}

#[test]
fn test_reshape_uses_to_shape() {
    let op = StdTensorOp::Reshape {
        to_shape: vec![cst(2), cst(3)],
    };
    let input = vec![cst(6)];
    let out = infer_output_shapes(&op, &[&input]);
    assert_eq!(out, vec![vec![cst(2), cst(3)]]);
}

#[test]
fn test_broadcast_in_dim_uses_shape() {
    let op = StdTensorOp::BroadcastInDim {
        shape: vec![cst(3), cst(4)],
        dims: vec![1],
    };
    let input = vec![cst(4)];
    let out = infer_output_shapes(&op, &[&input]);
    assert_eq!(out, vec![vec![cst(3), cst(4)]]);
}

#[test]
fn test_reduce_sum_removes_axes() {
    let op = StdTensorOp::ReduceSum { axes: vec![0] };
    let input = vec![cst(3), cst(4)];
    let out = infer_output_shapes(&op, &[&input]);
    assert_eq!(out, vec![vec![cst(4)]]);
}

#[test]
fn test_dot_general_canonical() {
    let op = StdTensorOp::DotGeneral {
        config: DotGeneralConfig {
            lhs_contracting_dims: vec![1],
            rhs_contracting_dims: vec![0],
            lhs_batch_dims: vec![],
            rhs_batch_dims: vec![],
        },
    };
    let lhs = vec![cst(3), cst(4)];
    let rhs = vec![cst(4), cst(5)];
    let out = infer_output_shapes(&op, &[&lhs, &rhs]);
    assert_eq!(out, vec![vec![cst(3), cst(5)]]);
}

#[test]
fn test_shape_of_returns_scalar() {
    let op = StdTensorOp::ShapeOf { axis: 0 };
    let input = vec![cst(3), cst(4)];
    let out = infer_output_shapes(&op, &[&input]);
    assert_eq!(out, vec![vec![]]);
}

#[test]
fn test_svd_three_outputs() {
    let op = StdTensorOp::Svd { eps: 1e-10 };
    let input = vec![cst(3), cst(4)];
    let out = infer_output_shapes(&op, &[&input]);
    assert_eq!(out.len(), 3);
    assert_eq!(
        out,
        vec![vec![cst(3), cst(3)], vec![cst(3)], vec![cst(3), cst(4)],]
    );
}

#[test]
fn test_shared_passthrough_shape_rules_cover_remaining_elementwise_ops() {
    let unary_input = vec![cst(2), cst(5)];
    for op in [
        StdTensorOp::Neg,
        StdTensorOp::Conj,
        StdTensorOp::Abs,
        StdTensorOp::Sign,
        StdTensorOp::Exp,
        StdTensorOp::Log,
        StdTensorOp::Sin,
        StdTensorOp::Cos,
        StdTensorOp::Tanh,
        StdTensorOp::Sqrt,
        StdTensorOp::Rsqrt,
        StdTensorOp::Expm1,
        StdTensorOp::Log1p,
        StdTensorOp::Tril { k: -1 },
        StdTensorOp::Triu { k: 2 },
        StdTensorOp::Reverse { axes: vec![0] },
        StdTensorOp::ValidateNonsingular,
        StdTensorOp::Cholesky,
    ] {
        let out = infer_output_shapes(&op, &[&unary_input]);
        assert_eq!(
            out,
            vec![unary_input.clone()],
            "unexpected shape for {op:?}"
        );
    }

    let binary_lhs = vec![cst(2), cst(5)];
    let binary_rhs = vec![cst(2), cst(5)];
    for op in [
        StdTensorOp::Mul,
        StdTensorOp::Div,
        StdTensorOp::Maximum,
        StdTensorOp::Minimum,
        StdTensorOp::Compare(tenferro_tensor::CompareDir::Eq),
    ] {
        let out = infer_output_shapes(&op, &[&binary_lhs, &binary_rhs]);
        assert_eq!(out, vec![binary_lhs.clone()], "unexpected shape for {op:?}");
    }

    let scatter = StdTensorOp::Scatter(ScatterConfig {
        update_window_dims: vec![],
        inserted_window_dims: vec![0],
        scatter_dims_to_operand_dims: vec![0],
        index_vector_dim: 1,
    });
    let scatter_updates = vec![cst(2), cst(5)];
    let out = infer_output_shapes(&scatter, &[&binary_lhs, &binary_rhs, &scatter_updates]);
    assert_eq!(out, vec![binary_lhs.clone()]);

    let ternary_a = vec![cst(2), cst(5)];
    let ternary_b = vec![cst(2), cst(5)];
    let ternary_c = vec![cst(2), cst(5)];
    for op in [StdTensorOp::Select, StdTensorOp::Clamp] {
        let out = infer_output_shapes(&op, &[&ternary_a, &ternary_b, &ternary_c]);
        assert_eq!(out, vec![ternary_a.clone()], "unexpected shape for {op:?}");
    }
}

#[test]
fn test_structural_indexing_and_dynamic_shapes() {
    let input = vec![cst(4), cst(5)];
    let gather_indices = vec![cst(3), cst(1)];
    let gather = StdTensorOp::Gather(GatherConfig {
        offset_dims: vec![1],
        collapsed_slice_dims: vec![0],
        start_index_map: vec![0],
        index_vector_dim: 1,
        slice_sizes: vec![1, 5],
    });
    assert_eq!(
        infer_output_shapes(&gather, &[&input, &gather_indices]),
        vec![vec![cst(3), cst(5)]]
    );

    let slice = StdTensorOp::Slice(SliceConfig {
        starts: vec![1, 0],
        limits: vec![4, 5],
        strides: vec![2, 2],
    });
    assert_eq!(
        infer_output_shapes(&slice, &[&input]),
        vec![vec![cst(2), cst(3)]]
    );

    let dynamic_slice = StdTensorOp::DynamicSlice {
        slice_sizes: vec![2, 4],
    };
    let starts = vec![cst(2)];
    assert_eq!(
        infer_output_shapes(&dynamic_slice, &[&input, &starts]),
        vec![vec![cst(2), cst(4)]]
    );

    let pad = StdTensorOp::Pad(PadConfig {
        edge_padding_low: vec![1, 2],
        edge_padding_high: vec![0, 1],
        interior_padding: vec![1, 0],
    });
    assert_eq!(
        infer_output_shapes(&pad, &[&input]),
        vec![vec![cst(8), cst(8)]]
    );

    let extracted = StdTensorOp::ExtractDiag {
        axis_a: 0,
        axis_b: 1,
    };
    assert_eq!(
        infer_output_shapes(&extracted, &[&input]),
        vec![vec![cst(4)]]
    );

    let embedded = StdTensorOp::EmbedDiag {
        axis_a: 0,
        axis_b: 1,
    };
    let diag = vec![cst(4)];
    assert_eq!(
        infer_output_shapes(&embedded, &[&diag]),
        vec![vec![cst(4), cst(4)]]
    );

    let concat = StdTensorOp::Concatenate {
        axis: 1,
        n_inputs: 2,
    };
    let rhs = vec![cst(4), cst(7)];
    assert_eq!(
        infer_output_shapes(&concat, &[&input, &rhs]),
        vec![vec![cst(4), cst(12)]]
    );

    let einsum = StdTensorOp::NaryEinsum {
        subscripts: "ij,jk->ik".into(),
    };
    let lhs = vec![cst(3), cst(4)];
    let rhs = vec![cst(4), cst(6)];
    assert_eq!(
        infer_output_shapes(&einsum, &[&lhs, &rhs]),
        vec![vec![cst(3), cst(6)]]
    );

    let dynamic_truncate = StdTensorOp::DynamicTruncate { axis: 1 };
    let scalar = vec![];
    assert_eq!(
        infer_output_shapes(&dynamic_truncate, &[&input, &scalar]),
        vec![input.clone()]
    );

    let pad_to_match = StdTensorOp::PadToMatch { axis: 1 };
    let reference = vec![cst(4), cst(9)];
    assert_eq!(
        infer_output_shapes(&pad_to_match, &[&input, &reference]),
        vec![vec![cst(4), cst(9)]]
    );
}

#[test]
fn dynamic_truncate_extent_is_upper_bound_on_truncated_axis() {
    let input = vec![cst(4), cst(7)];
    let scalar = vec![];
    let op = StdTensorOp::DynamicTruncate { axis: 1 };

    let extents = infer_output_extents(&op, &[&input, &scalar]);

    assert_eq!(extents.len(), 1);
    assert_eq!(extents[0][0], ShapeExtent::exact(cst(4)));
    assert_eq!(extents[0][1], ShapeExtent::upper_bound(cst(7)));
}

#[test]
fn gather_dynamic_slice_sizes_uses_shape_source_input() {
    let operand = vec![cst(4), cst(5)];
    let indices = vec![cst(1), cst(1)];
    let updates = vec![cst(1), cst(2)];
    let op = StdTensorOp::GatherDynamicSliceSizes {
        offset_dims: vec![1],
        collapsed_slice_dims: vec![0],
        start_index_map: vec![0],
        index_vector_dim: 1,
        slice_sizes: vec![
            cst(1),
            DimExpr::InputDim {
                input_idx: 2,
                axis: 1,
            },
        ],
    };

    assert_eq!(
        infer_output_shapes(&op, &[&operand, &indices, &updates]),
        vec![vec![cst(1), cst(2)]]
    );
}

#[test]
fn test_multi_output_linalg_shape_rules() {
    let input = vec![cst(3), cst(4), cst(2)];

    let qr = StdTensorOp::Qr;
    assert_eq!(
        infer_output_shapes(&qr, &[&input]),
        vec![vec![cst(3), cst(3), cst(2)], vec![cst(3), cst(4), cst(2)]]
    );

    let lu = StdTensorOp::Lu;
    assert_eq!(
        infer_output_shapes(&lu, &[&input]),
        vec![
            vec![cst(3), cst(3), cst(2)],
            vec![cst(3), cst(3), cst(2)],
            vec![cst(3), cst(4), cst(2)],
            vec![cst(2)],
        ]
    );

    let eigh = StdTensorOp::Eigh { eps: 1e-12 };
    let eigh_input = vec![cst(3), cst(3), cst(2)];
    assert_eq!(
        infer_output_shapes(&eigh, &[&eigh_input]),
        vec![vec![cst(3), cst(2)], vec![cst(3), cst(3), cst(2)]]
    );

    let eig = StdTensorOp::Eig {
        input_dtype: tenferro_tensor::DType::F64,
    };
    assert_eq!(
        infer_output_shapes(&eig, &[&eigh_input]),
        vec![vec![cst(3), cst(2)], vec![cst(3), cst(3), cst(2)]]
    );

    let triangular_solve = StdTensorOp::TriangularSolve {
        left_side: true,
        lower: true,
        transpose_a: false,
        unit_diagonal: false,
    };
    let rhs = vec![cst(3), cst(2)];
    assert_eq!(
        infer_output_shapes(&triangular_solve, &[&eigh_input, &rhs]),
        vec![rhs]
    );
}
