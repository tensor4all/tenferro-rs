use super::*;

#[test]
fn promote_same_returns_same() {
    assert_eq!(promote_dtype(DType::F64, DType::F64), DType::F64);
    assert_eq!(promote_dtype(DType::C64, DType::C64), DType::C64);
    assert_eq!(promote_dtype(DType::I64, DType::I64), DType::I64);
}

#[test]
fn promote_i64_to_float() {
    assert_eq!(promote_dtype(DType::I64, DType::F64), DType::F64);
    assert_eq!(promote_dtype(DType::F64, DType::I64), DType::F64);
    assert_eq!(promote_dtype(DType::I64, DType::F32), DType::F64);
    assert_eq!(promote_dtype(DType::F32, DType::I64), DType::F64);
}

#[test]
fn promote_i64_to_complex() {
    assert_eq!(promote_dtype(DType::I64, DType::C64), DType::C64);
    assert_eq!(promote_dtype(DType::C64, DType::I64), DType::C64);
    assert_eq!(promote_dtype(DType::I64, DType::C32), DType::C64);
    assert_eq!(promote_dtype(DType::C32, DType::I64), DType::C64);
}

#[test]
fn promote_float_to_wider_float() {
    assert_eq!(promote_dtype(DType::F32, DType::F64), DType::F64);
    assert_eq!(promote_dtype(DType::F64, DType::F32), DType::F64);
}

#[test]
fn promote_float_to_complex() {
    assert_eq!(promote_dtype(DType::F32, DType::C32), DType::C32);
    assert_eq!(promote_dtype(DType::F64, DType::C64), DType::C64);
    assert_eq!(promote_dtype(DType::F64, DType::C32), DType::C64);
    assert_eq!(promote_dtype(DType::F32, DType::C64), DType::C64);
}

#[test]
fn promote_complex_to_wider_complex() {
    assert_eq!(promote_dtype(DType::C32, DType::C64), DType::C64);
    assert_eq!(promote_dtype(DType::C64, DType::C32), DType::C64);
}

#[test]
fn promote_dtype_div_like_i64_to_f64() {
    assert_eq!(promote_dtype_div_like(DType::I64, DType::I64), DType::F64);
    assert_eq!(promote_dtype_div_like(DType::F64, DType::F64), DType::F64);
    assert_eq!(promote_dtype_div_like(DType::I64, DType::F64), DType::F64);
}

#[test]
fn promote_dtypes_fold() {
    assert_eq!(
        promote_dtypes([DType::I64, DType::F32, DType::C64]),
        DType::C64
    );
    assert_eq!(promote_dtypes([DType::F32, DType::F64]), DType::F64);
    assert_eq!(promote_dtypes([]), DType::F64); // empty -> F64 default
}

#[test]
fn invalid_shape_configs_return_errors_instead_of_panicking() {
    let shape = DimExpr::from_concrete(&[2, 3]);

    let bad_pad = StdTensorOp::Pad(PadConfig {
        edge_padding_low: vec![0],
        edge_padding_high: vec![0, 0],
        interior_padding: vec![0, 0],
    });
    assert!(infer_output_shapes(&bad_pad, &[&shape]).is_err());

    let bad_slice = StdTensorOp::Slice(SliceConfig {
        starts: vec![2],
        limits: vec![1],
        strides: vec![1],
    });
    assert!(infer_output_shapes(&bad_slice, &[&shape[..1]]).is_err());

    let bad_pad_to_match = StdTensorOp::PadToMatch { axis: 2 };
    assert!(infer_output_shapes(&bad_pad_to_match, &[&shape, &shape]).is_err());

    let bad_concat = StdTensorOp::Concatenate {
        axis: 2,
        input_count: 2,
    };
    assert!(infer_output_shapes(&bad_concat, &[&shape, &shape]).is_err());
}

#[test]
fn reduction_shape_inference_rejects_invalid_axes() {
    let shape = DimExpr::from_concrete(&[2, 3]);
    let invalid_ops = [
        StdTensorOp::ReduceSum { axes: vec![2] },
        StdTensorOp::ReduceProd { axes: vec![0, 0] },
        StdTensorOp::ReduceMax { axes: vec![3] },
        StdTensorOp::ReduceMin { axes: vec![1, 1] },
    ];

    for op in invalid_ops {
        assert!(infer_output_shapes(&op, &[&shape]).is_err(), "{op:?}");
    }
}

#[test]
fn transpose_shape_inference_rejects_invalid_permutations() {
    let shape = DimExpr::from_concrete(&[2, 3]);
    let invalid_ops = [
        StdTensorOp::Transpose { perm: vec![0] },
        StdTensorOp::Transpose { perm: vec![0, 0] },
        StdTensorOp::Transpose { perm: vec![0, 2] },
    ];

    for op in invalid_ops {
        assert!(infer_output_shapes(&op, &[&shape]).is_err(), "{op:?}");
    }
}

#[test]
fn dot_general_shape_inference_rejects_invalid_dimension_numbers() {
    let lhs = DimExpr::from_concrete(&[2, 3]);
    let rhs = DimExpr::from_concrete(&[3, 2]);

    let contracting_oob = StdTensorOp::DotGeneral {
        config: DotGeneralConfig {
            lhs_contracting_dims: vec![2],
            rhs_contracting_dims: vec![0],
            lhs_batch_dims: vec![],
            rhs_batch_dims: vec![],
        },
    };
    let batch_oob = StdTensorOp::DotGeneral {
        config: DotGeneralConfig {
            lhs_contracting_dims: vec![1],
            rhs_contracting_dims: vec![0],
            lhs_batch_dims: vec![2],
            rhs_batch_dims: vec![1],
        },
    };

    assert!(infer_output_shapes(&contracting_oob, &[&lhs, &rhs]).is_err());
    assert!(infer_output_shapes(&batch_oob, &[&lhs, &rhs]).is_err());
}

#[test]
fn concatenate_rejects_non_axis_dimension_mismatch() {
    let lhs = DimExpr::from_concrete(&[2, 3]);
    let rhs = DimExpr::from_concrete(&[4, 3]);
    let op = StdTensorOp::Concatenate {
        axis: 1,
        input_count: 2,
    };

    let err = infer_output_shapes(&op, &[&lhs, &rhs]).unwrap_err();

    let message = err.to_string();
    assert!(message.contains("concatenate"), "{message}");
    assert!(message.contains("dimension mismatch"), "{message}");
}

#[test]
fn gather_rejects_duplicate_offset_and_collapsed_slice_dims() {
    let operand = DimExpr::from_concrete(&[4, 5]);
    let indices = DimExpr::from_concrete(&[2, 2]);
    let duplicate_offset_dims = StdTensorOp::Gather(GatherConfig {
        offset_dims: vec![1, 1],
        collapsed_slice_dims: vec![],
        start_index_map: vec![0],
        index_vector_dim: 1,
        slice_sizes: vec![1, 5],
    });
    let duplicate_collapsed_slice_dims = StdTensorOp::Gather(GatherConfig {
        offset_dims: vec![1],
        collapsed_slice_dims: vec![0, 0],
        start_index_map: vec![0],
        index_vector_dim: 1,
        slice_sizes: vec![1, 5],
    });

    assert!(infer_output_shapes(&duplicate_offset_dims, &[&operand, &indices]).is_err());
    assert!(infer_output_shapes(&duplicate_collapsed_slice_dims, &[&operand, &indices]).is_err());
}

#[test]
fn gather_rejects_concrete_slice_sizes_larger_than_operand_dims() {
    let operand = DimExpr::from_concrete(&[4, 5]);
    let indices = DimExpr::from_concrete(&[2, 1]);
    let op = StdTensorOp::Gather(GatherConfig {
        offset_dims: vec![1, 2],
        collapsed_slice_dims: vec![],
        start_index_map: vec![0],
        index_vector_dim: 1,
        slice_sizes: vec![5, 5],
    });

    let err = infer_output_shapes(&op, &[&operand, &indices]).unwrap_err();
    let message = err.to_string();
    assert!(message.contains("gather"), "{message}");
    assert!(message.contains("slice_sizes[0]"), "{message}");
}

#[test]
fn invalid_structural_shape_rules_return_errors() {
    let shape = DimExpr::from_concrete(&[2, 3]);

    let missing_rhs = StdTensorOp::Mul;
    assert!(infer_output_shapes(&missing_rhs, &[&shape]).is_err());

    let extract_axis_oob = StdTensorOp::ExtractDiag {
        axis_a: 0,
        axis_b: 2,
    };
    assert!(infer_output_shapes(&extract_axis_oob, &[&shape]).is_err());

    let extract_duplicate_axis = StdTensorOp::ExtractDiag {
        axis_a: 1,
        axis_b: 1,
    };
    assert!(infer_output_shapes(&extract_duplicate_axis, &[&shape]).is_err());

    let embed_axis_a_oob = StdTensorOp::EmbedDiag {
        axis_a: 2,
        axis_b: 0,
    };
    assert!(infer_output_shapes(&embed_axis_a_oob, &[&shape]).is_err());

    let embed_axis_b_oob = StdTensorOp::EmbedDiag {
        axis_a: 0,
        axis_b: 3,
    };
    assert!(infer_output_shapes(&embed_axis_b_oob, &[&shape]).is_err());

    let truncate_axis_oob = StdTensorOp::DynamicTruncate { axis: 2 };
    assert!(infer_output_shapes(&truncate_axis_oob, &[&shape]).is_err());
    assert!(infer_output_extents(&truncate_axis_oob, &[&shape]).is_err());

    let reference = DimExpr::from_concrete(&[2]);
    let pad_reference_axis_oob = StdTensorOp::PadToMatch { axis: 1 };
    assert!(infer_output_shapes(&pad_reference_axis_oob, &[&shape, &reference]).is_err());
}

#[test]
fn gather_shape_rules_validate_dynamic_slice_metadata() {
    let operand = DimExpr::from_concrete(&[4, 5]);
    let indices = DimExpr::from_concrete(&[2, 2]);

    let slice_rank_mismatch = StdTensorOp::Gather(GatherConfig {
        offset_dims: vec![1],
        collapsed_slice_dims: vec![0],
        start_index_map: vec![0],
        index_vector_dim: 1,
        slice_sizes: vec![1],
    });
    assert!(infer_output_shapes(&slice_rank_mismatch, &[&operand, &indices]).is_err());

    let index_vector_oob = StdTensorOp::Gather(GatherConfig {
        offset_dims: vec![1],
        collapsed_slice_dims: vec![0],
        start_index_map: vec![0],
        index_vector_dim: 3,
        slice_sizes: vec![1, 5],
    });
    assert!(infer_output_shapes(&index_vector_oob, &[&operand, &indices]).is_err());

    let collapsed_axis_oob = StdTensorOp::Gather(GatherConfig {
        offset_dims: vec![1],
        collapsed_slice_dims: vec![2],
        start_index_map: vec![0],
        index_vector_dim: 1,
        slice_sizes: vec![1, 5],
    });
    assert!(infer_output_shapes(&collapsed_axis_oob, &[&operand, &indices]).is_err());

    let offset_len_mismatch = StdTensorOp::Gather(GatherConfig {
        offset_dims: vec![],
        collapsed_slice_dims: vec![0],
        start_index_map: vec![0],
        index_vector_dim: 1,
        slice_sizes: vec![1, 5],
    });
    assert!(infer_output_shapes(&offset_len_mismatch, &[&operand, &indices]).is_err());

    let offset_axis_oob = StdTensorOp::Gather(GatherConfig {
        offset_dims: vec![2],
        collapsed_slice_dims: vec![0],
        start_index_map: vec![0],
        index_vector_dim: 1,
        slice_sizes: vec![1, 5],
    });
    assert!(infer_output_shapes(&offset_axis_oob, &[&operand, &indices]).is_err());

    let unresolved_dynamic_slice_size = StdTensorOp::GatherDynamicSliceSizes {
        offset_dims: vec![1],
        collapsed_slice_dims: vec![0],
        start_index_map: vec![0],
        index_vector_dim: 1,
        slice_sizes: vec![
            DimExpr::Const(1),
            DimExpr::InputDim {
                input_idx: 2,
                axis: 0,
            },
        ],
    };
    assert!(infer_output_shapes(&unresolved_dynamic_slice_size, &[&operand, &indices]).is_err());
}

#[test]
fn slice_and_pad_shape_rules_validate_arithmetic() {
    let shape = DimExpr::from_concrete(&[4, 5]);

    let bad_slice_rank = StdTensorOp::Slice(SliceConfig {
        starts: vec![0],
        limits: vec![4, 5],
        strides: vec![1, 1],
    });
    assert!(infer_output_shapes(&bad_slice_rank, &[&shape]).is_err());

    let bad_slice_bounds = StdTensorOp::Slice(SliceConfig {
        starts: vec![3, 0],
        limits: vec![2, 5],
        strides: vec![1, 1],
    });
    assert!(infer_output_shapes(&bad_slice_bounds, &[&shape]).is_err());

    let zero_slice_stride = StdTensorOp::Slice(SliceConfig {
        starts: vec![0, 0],
        limits: vec![4, 5],
        strides: vec![0, 1],
    });
    assert!(infer_output_shapes(&zero_slice_stride, &[&shape]).is_err());

    let bad_pad_rank = StdTensorOp::Pad(PadConfig {
        edge_padding_low: vec![0],
        edge_padding_high: vec![0, 0],
        interior_padding: vec![0, 0],
    });
    assert!(infer_output_shapes(&bad_pad_rank, &[&shape]).is_err());

    let negative_interior_pad = StdTensorOp::Pad(PadConfig {
        edge_padding_low: vec![0, 0],
        edge_padding_high: vec![0, 0],
        interior_padding: vec![-1, 0],
    });
    assert!(infer_output_shapes(&negative_interior_pad, &[&shape]).is_err());

    let symbolic_shape = vec![DimExpr::InputDim {
        input_idx: 0,
        axis: 0,
    }];
    let symbolic_negative_edge_pad = StdTensorOp::Pad(PadConfig {
        edge_padding_low: vec![-2],
        edge_padding_high: vec![0],
        interior_padding: vec![0],
    });
    assert!(infer_output_shapes(&symbolic_negative_edge_pad, &[&symbolic_shape]).is_ok());

    let symbolic_interior_pad = StdTensorOp::Pad(PadConfig {
        edge_padding_low: vec![1],
        edge_padding_high: vec![2],
        interior_padding: vec![1],
    });
    assert!(infer_output_shapes(&symbolic_interior_pad, &[&symbolic_shape]).is_ok());

    let overflowing_edge_pad = StdTensorOp::Pad(PadConfig {
        edge_padding_low: vec![i64::MAX],
        edge_padding_high: vec![1],
        interior_padding: vec![0],
    });
    assert!(infer_output_shapes(&overflowing_edge_pad, &[&symbolic_shape]).is_err());
}

#[test]
fn shape_arithmetic_overflow_returns_errors_instead_of_wrapping() {
    let huge = vec![DimExpr::Const(usize::MAX), DimExpr::Const(1)];

    let concat = StdTensorOp::Concatenate {
        axis: 0,
        input_count: 2,
    };
    assert!(infer_output_shapes(&concat, &[&huge, &huge]).is_err());

    let pad = StdTensorOp::Pad(PadConfig {
        edge_padding_low: vec![1],
        edge_padding_high: vec![0],
        interior_padding: vec![0],
    });
    assert!(infer_output_shapes(&pad, &[&[DimExpr::Const(usize::MAX)]]).is_err());
}
