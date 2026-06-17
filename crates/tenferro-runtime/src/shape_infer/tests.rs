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
fn promotion_rank_ordering() {
    assert!(promotion_rank(DType::I64) < promotion_rank(DType::F32));
    assert!(promotion_rank(DType::F32) < promotion_rank(DType::F64));
    assert!(promotion_rank(DType::F64) < promotion_rank(DType::C32));
    assert!(promotion_rank(DType::C32) < promotion_rank(DType::C64));
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
