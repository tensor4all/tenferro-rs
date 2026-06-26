use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_runtime::extension::infer_output_dtype;
use tenferro_tensor::{DType, DotGeneralConfig, GatherConfig, ScatterConfig};

#[test]
fn test_add_preserves_f32() {
    let op = StdTensorOp::Add;
    assert_eq!(
        infer_output_dtype(&op, &[DType::F32, DType::F32]).unwrap(),
        DType::F32
    );
}

#[test]
fn test_add_preserves_c32() {
    let op = StdTensorOp::Add;
    assert_eq!(
        infer_output_dtype(&op, &[DType::C32, DType::C32]).unwrap(),
        DType::C32
    );
}

#[test]
fn test_convert_uses_target_dtype() {
    let op = StdTensorOp::Convert {
        from: DType::F32,
        to: DType::F64,
    };
    assert_eq!(infer_output_dtype(&op, &[DType::F32]).unwrap(), DType::F64);
}

#[test]
fn test_constant_uses_variant_dtype() {
    let op = StdTensorOp::Constant {
        dtype: DType::C64,
        bytes: vec![],
    };
    assert_eq!(infer_output_dtype(&op, &[]).unwrap(), DType::C64);
}

#[test]
fn test_abs_complex_outputs_real_dtype() {
    assert_eq!(
        infer_output_dtype(&StdTensorOp::Abs, &[DType::C32]).unwrap(),
        DType::F32
    );
    assert_eq!(
        infer_output_dtype(&StdTensorOp::Abs, &[DType::C64]).unwrap(),
        DType::F64
    );
}

#[test]
fn test_reduce_sum_preserves_dtype() {
    let op = StdTensorOp::ReduceSum { axes: vec![0] };
    assert_eq!(infer_output_dtype(&op, &[DType::F32]).unwrap(), DType::F32);
}

#[test]
fn test_dot_general_preserves_lhs_dtype() {
    let op = StdTensorOp::DotGeneral {
        config: DotGeneralConfig {
            lhs_contracting_dims: vec![1],
            rhs_contracting_dims: vec![0],
            lhs_batch_dims: vec![],
            rhs_batch_dims: vec![],
        },
    };
    assert_eq!(
        infer_output_dtype(&op, &[DType::F32, DType::F32]).unwrap(),
        DType::F32
    );
}

#[test]
fn test_indexing_dtype_inference_ignores_index_operands() {
    let gather = StdTensorOp::Gather(GatherConfig {
        offset_dims: vec![],
        collapsed_slice_dims: vec![0],
        start_index_map: vec![0],
        index_vector_dim: 1,
        slice_sizes: vec![1],
    });
    assert_eq!(
        infer_output_dtype(&gather, &[DType::C64, DType::I64]).unwrap(),
        DType::C64
    );

    let dynamic_slice = StdTensorOp::DynamicSlice {
        slice_sizes: vec![1],
    };
    assert_eq!(
        infer_output_dtype(&dynamic_slice, &[DType::F32, DType::I64]).unwrap(),
        DType::F32
    );

    let scatter = StdTensorOp::Scatter(ScatterConfig {
        update_window_dims: vec![],
        inserted_window_dims: vec![0],
        scatter_dims_to_operand_dims: vec![0],
        index_vector_dim: 1,
    });
    assert_eq!(
        infer_output_dtype(&scatter, &[DType::F32, DType::I64, DType::F32]).unwrap(),
        DType::F32
    );

    assert_eq!(
        infer_output_dtype(
            &StdTensorOp::DynamicUpdateSlice,
            &[DType::F32, DType::F32, DType::I64],
        )
        .unwrap(),
        DType::F32
    );
}

#[test]
fn test_dynamic_shape_ops_preserve_data_dtype() {
    assert_eq!(
        infer_output_dtype(
            &StdTensorOp::DynamicTruncate { axis: 0 },
            &[DType::F32, DType::F64],
        )
        .unwrap(),
        DType::F32
    );

    assert_eq!(
        infer_output_dtype(
            &StdTensorOp::PadToMatch { axis: 0 },
            &[DType::F32, DType::F64],
        )
        .unwrap(),
        DType::F32
    );
}
