use crate::std_tensor_op::StdTensorOp;
use computegraph::GraphOperation;
use num_complex::{Complex32, Complex64};
use tenferro_tensor::{
    CompareDir, DType, DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig,
};

#[test]
fn test_std_tensor_op_input_output_counts() {
    assert_eq!(StdTensorOp::Add.input_count(), 2);
    assert_eq!(StdTensorOp::Mul.input_count(), 2);
    assert_eq!(StdTensorOp::Neg.input_count(), 1);
    assert_eq!(StdTensorOp::Conj.input_count(), 1);
    assert_eq!(
        StdTensorOp::DotGeneral {
            config: DotGeneralConfig {
                lhs_contracting_dims: [1].as_slice().into(),
                rhs_contracting_dims: [0].as_slice().into(),
                lhs_batch_dims: [].as_slice().into(),
                rhs_batch_dims: [].as_slice().into(),
            },
        }
        .input_count(),
        2
    );
    assert_eq!(StdTensorOp::ReduceSum { axes: vec![0] }.input_count(), 1);
    assert_eq!(StdTensorOp::constant(1.0).input_count(), 0);
    assert_eq!(
        StdTensorOp::ExtractDiag {
            axis_a: 0,
            axis_b: 1
        }
        .input_count(),
        1
    );
    assert_eq!(
        StdTensorOp::EmbedDiag {
            axis_a: 0,
            axis_b: 1
        }
        .input_count(),
        1
    );
    assert_eq!(
        StdTensorOp::Gather(GatherConfig {
            offset_dims: vec![],
            collapsed_slice_dims: vec![0],
            start_index_map: vec![0],
            index_vector_dim: 1,
            slice_sizes: vec![1],
        })
        .input_count(),
        2
    );
    assert_eq!(
        StdTensorOp::Scatter(ScatterConfig {
            update_window_dims: vec![],
            inserted_window_dims: vec![0],
            scatter_dims_to_operand_dims: vec![0],
            index_vector_dim: 1,
        })
        .input_count(),
        3
    );
    assert_eq!(
        StdTensorOp::DynamicSlice {
            slice_sizes: vec![1]
        }
        .input_count(),
        2
    );
    assert_eq!(StdTensorOp::DynamicUpdateSlice.input_count(), 3);
    assert_eq!(
        StdTensorOp::Pad(PadConfig {
            edge_padding_low: vec![1],
            edge_padding_high: vec![1],
            interior_padding: vec![0],
        })
        .input_count(),
        1
    );

    assert_eq!(StdTensorOp::Add.output_count(), 1);
    assert_eq!(StdTensorOp::Neg.output_count(), 1);
    assert_eq!(StdTensorOp::Compare(CompareDir::Eq).input_count(), 2);
    assert_eq!(StdTensorOp::Select.input_count(), 3);
    assert_eq!(StdTensorOp::Clamp.input_count(), 3);
    assert_eq!(StdTensorOp::Div.input_count(), 2);
    assert_eq!(StdTensorOp::Pow.input_count(), 2);
    assert_eq!(StdTensorOp::Abs.input_count(), 1);
    assert_eq!(StdTensorOp::Exp.input_count(), 1);
    assert_eq!(StdTensorOp::Log1p.input_count(), 1);
    assert_eq!(StdTensorOp::constant(1.0).output_count(), 1);
    assert_eq!(
        StdTensorOp::EmbedDiag {
            axis_a: 0,
            axis_b: 1
        }
        .output_count(),
        1
    );
    assert_eq!(
        StdTensorOp::Gather(GatherConfig {
            offset_dims: vec![],
            collapsed_slice_dims: vec![0],
            start_index_map: vec![0],
            index_vector_dim: 1,
            slice_sizes: vec![1],
        })
        .output_count(),
        1
    );
    assert_eq!(
        StdTensorOp::Scatter(ScatterConfig {
            update_window_dims: vec![],
            inserted_window_dims: vec![0],
            scatter_dims_to_operand_dims: vec![0],
            index_vector_dim: 1,
        })
        .output_count(),
        1
    );
    assert_eq!(
        StdTensorOp::DynamicSlice {
            slice_sizes: vec![1]
        }
        .output_count(),
        1
    );
    assert_eq!(StdTensorOp::DynamicUpdateSlice.output_count(), 1);
    assert_eq!(
        StdTensorOp::Pad(PadConfig {
            edge_padding_low: vec![1],
            edge_padding_high: vec![1],
            interior_padding: vec![0],
        })
        .output_count(),
        1
    );
    assert_eq!(StdTensorOp::Tril { k: -1 }.input_count(), 1);
    assert_eq!(StdTensorOp::Tril { k: -1 }.output_count(), 1);
    assert_eq!(StdTensorOp::Triu { k: 1 }.input_count(), 1);
    assert_eq!(StdTensorOp::Triu { k: 1 }.output_count(), 1);
}

#[test]
fn test_std_tensor_op_remaining_input_output_counts() {
    let slice = tenferro_tensor::SliceConfig {
        starts: vec![0, 1],
        limits: vec![2, 3],
        strides: vec![1, 1],
    };

    assert_eq!(StdTensorOp::Maximum.input_count(), 2);
    assert_eq!(StdTensorOp::Maximum.output_count(), 1);
    assert_eq!(StdTensorOp::Minimum.input_count(), 2);
    assert_eq!(StdTensorOp::Minimum.output_count(), 1);
    assert_eq!(
        StdTensorOp::Convert {
            from: DType::F64,
            to: DType::C64,
        }
        .input_count(),
        1
    );
    assert_eq!(
        StdTensorOp::Convert {
            from: DType::F64,
            to: DType::C64,
        }
        .output_count(),
        1
    );
    assert_eq!(StdTensorOp::Slice(slice.clone()).input_count(), 1);
    assert_eq!(StdTensorOp::Slice(slice).output_count(), 1);
    assert_eq!(StdTensorOp::Reverse { axes: vec![0, 2] }.input_count(), 1);
    assert_eq!(StdTensorOp::Reverse { axes: vec![0, 2] }.output_count(), 1);
    assert_eq!(StdTensorOp::ShapeOf { axis: 1 }.input_count(), 1);
    assert_eq!(StdTensorOp::ShapeOf { axis: 1 }.output_count(), 1);
    assert_eq!(StdTensorOp::DynamicTruncate { axis: 0 }.input_count(), 2);
    assert_eq!(StdTensorOp::DynamicTruncate { axis: 0 }.output_count(), 1);
    assert_eq!(StdTensorOp::PadToMatch { axis: 0 }.input_count(), 2);
    assert_eq!(StdTensorOp::PadToMatch { axis: 0 }.output_count(), 1);
}

#[test]
fn test_std_tensor_op_concatenate_counts_use_recorded_arity() {
    let op = StdTensorOp::Concatenate {
        axis: 0,
        input_count: 3,
    };

    assert_eq!(op.input_count(), 3);
    assert_eq!(op.output_count(), 1);
}

#[test]
fn test_std_tensor_op_reduction_counts_cover_remaining_variants() {
    assert_eq!(StdTensorOp::ReduceProd { axes: vec![1] }.input_count(), 1);
    assert_eq!(StdTensorOp::ReduceProd { axes: vec![1] }.output_count(), 1);
    assert_eq!(StdTensorOp::ReduceMax { axes: vec![0] }.input_count(), 1);
    assert_eq!(StdTensorOp::ReduceMax { axes: vec![0] }.output_count(), 1);
    assert_eq!(StdTensorOp::ReduceMin { axes: vec![0, 1] }.input_count(), 1);
    assert_eq!(
        StdTensorOp::ReduceMin { axes: vec![0, 1] }.output_count(),
        1
    );
}

#[test]
fn test_std_tensor_op_hash_covers_remaining_variants() {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let variants = [
        StdTensorOp::Compare(CompareDir::Ge),
        StdTensorOp::Tril { k: -2 },
        StdTensorOp::Triu { k: 3 },
        StdTensorOp::Gather(GatherConfig {
            offset_dims: vec![],
            collapsed_slice_dims: vec![0],
            start_index_map: vec![0],
            index_vector_dim: 1,
            slice_sizes: vec![1],
        }),
        StdTensorOp::Scatter(ScatterConfig {
            update_window_dims: vec![],
            inserted_window_dims: vec![0],
            scatter_dims_to_operand_dims: vec![0],
            index_vector_dim: 1,
        }),
        StdTensorOp::Slice(tenferro_tensor::SliceConfig {
            starts: vec![0],
            limits: vec![1],
            strides: vec![1],
        }),
        StdTensorOp::DynamicSlice {
            slice_sizes: vec![1],
        },
        StdTensorOp::DynamicUpdateSlice,
        StdTensorOp::Pad(PadConfig {
            edge_padding_low: vec![0],
            edge_padding_high: vec![0],
            interior_padding: vec![0],
        }),
        StdTensorOp::Convert {
            from: DType::F64,
            to: DType::C64,
        },
        StdTensorOp::Concatenate {
            axis: 1,
            input_count: 3,
        },
        StdTensorOp::Reverse { axes: vec![0] },
        StdTensorOp::ShapeOf { axis: 0 },
        StdTensorOp::DynamicTruncate { axis: 0 },
        StdTensorOp::PadToMatch { axis: 0 },
        StdTensorOp::ReduceProd { axes: vec![0] },
        StdTensorOp::ReduceMax { axes: vec![0] },
        StdTensorOp::ReduceMin { axes: vec![0] },
    ];

    for op in variants {
        let mut hasher = DefaultHasher::new();
        op.hash(&mut hasher);
        assert_ne!(hasher.finish(), 0, "unexpected zero hash for {op:?}");
    }

    let mut lhs = DefaultHasher::new();
    StdTensorOp::constant(1.25).hash(&mut lhs);
    let mut rhs = DefaultHasher::new();
    StdTensorOp::constant(1.25).hash(&mut rhs);
    assert_eq!(lhs.finish(), rhs.finish());
}

#[test]
fn test_std_tensor_op_generic_constant_constructor_encodes_expected_bytes() {
    assert_eq!(
        StdTensorOp::constant(1.25_f64),
        StdTensorOp::Constant {
            dtype: DType::F64,
            bytes: 1.25_f64.to_le_bytes().to_vec(),
        }
    );
    assert_eq!(
        StdTensorOp::constant(1.25_f32),
        StdTensorOp::Constant {
            dtype: DType::F32,
            bytes: 1.25_f32.to_le_bytes().to_vec(),
        }
    );
    assert_eq!(
        StdTensorOp::constant(7_i64),
        StdTensorOp::Constant {
            dtype: DType::I64,
            bytes: 7_i64.to_le_bytes().to_vec(),
        }
    );
    assert_eq!(
        StdTensorOp::constant(7_i32),
        StdTensorOp::Constant {
            dtype: DType::I32,
            bytes: 7_i32.to_le_bytes().to_vec(),
        }
    );
    assert_eq!(
        StdTensorOp::constant(true),
        StdTensorOp::Constant {
            dtype: DType::Bool,
            bytes: vec![1],
        }
    );

    let c64 = Complex64::new(1.0, -2.0);
    let mut c64_bytes = Vec::new();
    c64_bytes.extend_from_slice(&c64.re.to_le_bytes());
    c64_bytes.extend_from_slice(&c64.im.to_le_bytes());
    assert_eq!(
        StdTensorOp::constant(c64),
        StdTensorOp::Constant {
            dtype: DType::C64,
            bytes: c64_bytes,
        }
    );

    let c32 = Complex32::new(1.0, -2.0);
    let mut c32_bytes = Vec::new();
    c32_bytes.extend_from_slice(&c32.re.to_le_bytes());
    c32_bytes.extend_from_slice(&c32.im.to_le_bytes());
    assert_eq!(
        StdTensorOp::constant(c32),
        StdTensorOp::Constant {
            dtype: DType::C32,
            bytes: c32_bytes,
        }
    );
}

#[test]
fn generic_constant_constructor_sets_dtype_for_each_scalar() {
    fn dtype_of(op: StdTensorOp) -> DType {
        match op {
            StdTensorOp::Constant { dtype, .. } => dtype,
            other => panic!("expected constant op, got {other:?}"),
        }
    }

    assert_eq!(dtype_of(StdTensorOp::constant(1.0_f64)), DType::F64);
    assert_eq!(dtype_of(StdTensorOp::constant(1.0_f32)), DType::F32);
    assert_eq!(dtype_of(StdTensorOp::constant(1_i64)), DType::I64);
    assert_eq!(dtype_of(StdTensorOp::constant(1_i32)), DType::I32);
    assert_eq!(dtype_of(StdTensorOp::constant(true)), DType::Bool);
    assert_eq!(
        dtype_of(StdTensorOp::constant(Complex64::new(1.0, -2.0))),
        DType::C64
    );
    assert_eq!(
        dtype_of(StdTensorOp::constant(Complex32::new(1.0, -2.0))),
        DType::C32
    );
}
