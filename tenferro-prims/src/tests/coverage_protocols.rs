use tenferro_device::LogicalMemorySpace;
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::{
    cast_metadata_value, for_each_index_result, supports_metadata_cast,
    validate_metadata_cast_shapes, validate_pointwise_cast_bridge_inputs,
    validate_where_bridge_inputs, MetadataBinaryOp, MetadataCastPrimsDescriptor,
    MetadataConstantValue, MetadataDType, MetadataGenerateOp, MetadataPrimsDescriptor,
    MetadataReductionOp, MetadataScalarTensorRef, MetadataTensorMut, MetadataTensorRef,
    MetadataTernaryOp,
};

const CPU: LogicalMemorySpace = LogicalMemorySpace::MainMemory;

#[test]
fn metadata_descriptors_and_handles_report_expected_dtypes() {
    assert_eq!(MetadataConstantValue::I32(-7).dtype(), MetadataDType::I32);
    assert_eq!(
        MetadataConstantValue::Bool(true).dtype(),
        MetadataDType::Bool
    );
    assert!(matches!(
        MetadataGenerateOp::Constant(MetadataConstantValue::Bool(true)),
        MetadataGenerateOp::Constant(MetadataConstantValue::Bool(true))
    ));
    assert_eq!(MetadataBinaryOp::BitAnd, MetadataBinaryOp::BitAnd);
    assert_eq!(MetadataTernaryOp::Where, MetadataTernaryOp::Where);
    assert_eq!(MetadataReductionOp::Any, MetadataReductionOp::Any);

    let ints = Tensor::<i32>::zeros(&[2], CPU, MemoryOrder::ColumnMajor).unwrap();
    let bools = Tensor::<u8>::zeros(&[2], CPU, MemoryOrder::ColumnMajor).unwrap();
    assert_eq!(MetadataTensorRef::I32(&ints).dtype(), MetadataDType::I32);
    assert_eq!(MetadataTensorRef::Bool(&bools).dtype(), MetadataDType::Bool);

    let mut ints_mut = Tensor::<i32>::zeros(&[2], CPU, MemoryOrder::ColumnMajor).unwrap();
    let mut bools_mut = Tensor::<u8>::zeros(&[2], CPU, MemoryOrder::ColumnMajor).unwrap();
    assert_eq!(
        MetadataTensorMut::I32(&mut ints_mut).dtype(),
        MetadataDType::I32
    );
    assert_eq!(
        MetadataTensorMut::Bool(&mut bools_mut).dtype(),
        MetadataDType::Bool
    );

    let generate = MetadataPrimsDescriptor::Generate {
        op: MetadataGenerateOp::IotaStartZero,
        output_dtype: MetadataDType::I32,
    };
    assert!(matches!(generate, MetadataPrimsDescriptor::Generate { .. }));
    let binary = MetadataPrimsDescriptor::Binary {
        op: MetadataBinaryOp::Equal,
        lhs_dtype: MetadataDType::I32,
        rhs_dtype: MetadataDType::I32,
        output_dtype: MetadataDType::Bool,
    };
    assert!(matches!(binary, MetadataPrimsDescriptor::Binary { .. }));
    let reduction = MetadataPrimsDescriptor::Reduction {
        modes_a: vec![0, 1],
        modes_c: vec![1],
        input_dtype: MetadataDType::Bool,
        output_dtype: MetadataDType::Bool,
        op: MetadataReductionOp::All,
    };
    assert!(matches!(
        reduction,
        MetadataPrimsDescriptor::Reduction { .. }
    ));
}

#[test]
fn metadata_cast_helpers_accept_valid_shapes_and_inputs() {
    let cast_desc = MetadataCastPrimsDescriptor::PointwiseCast {
        input_dtype: MetadataDType::Bool,
    };
    assert!(supports_metadata_cast(&cast_desc));
    validate_metadata_cast_shapes(&cast_desc, &[&[2, 1], &[2, 3]], "MetadataCastPointwise")
        .unwrap();

    let where_desc = MetadataCastPrimsDescriptor::Where {
        cond_dtype: MetadataDType::Bool,
    };
    assert!(supports_metadata_cast(&where_desc));
    validate_metadata_cast_shapes(
        &where_desc,
        &[&[2, 1], &[2, 3], &[1, 3], &[2, 3]],
        "MetadataCastWhere",
    )
    .unwrap();

    assert_eq!(cast_metadata_value::<f32, _>(7_i32, "mask").unwrap(), 7.0);
    assert_eq!(
        cast_metadata_value::<i16, _>(true as u8, "mask").unwrap(),
        1
    );

    let mask = Tensor::<u8>::zeros(&[2, 3], CPU, MemoryOrder::ColumnMajor).unwrap();
    let on_true = Tensor::<f64>::ones(&[2, 3], CPU, MemoryOrder::ColumnMajor).unwrap();
    let on_false = Tensor::<f64>::zeros(&[2, 3], CPU, MemoryOrder::ColumnMajor).unwrap();
    let where_inputs = [
        MetadataScalarTensorRef::Metadata(MetadataTensorRef::Bool(&mask)),
        MetadataScalarTensorRef::Scalar(&on_true),
        MetadataScalarTensorRef::Scalar(&on_false),
    ];
    let (cond, lhs, rhs) = validate_where_bridge_inputs(&where_inputs).unwrap();
    assert_eq!(cond.dtype(), MetadataDType::Bool);
    assert_eq!(lhs.dims(), &[2, 3]);
    assert_eq!(rhs.dims(), &[2, 3]);

    let pointwise_inputs: [MetadataScalarTensorRef<'_, f64>; 1] =
        [MetadataScalarTensorRef::Metadata(MetadataTensorRef::Bool(
            &mask,
        ))];
    assert_eq!(
        validate_pointwise_cast_bridge_inputs(&pointwise_inputs)
            .unwrap()
            .dtype(),
        MetadataDType::Bool
    );

    let mut visits = 0usize;
    for_each_index_result(&[2, 2], |_| {
        visits += 1;
        Ok(())
    })
    .unwrap();
    assert_eq!(visits, 4);
}

#[test]
fn metadata_cast_helpers_reject_invalid_inputs_and_propagate_errors() {
    let unsupported_where = MetadataCastPrimsDescriptor::Where {
        cond_dtype: MetadataDType::I32,
    };
    assert!(!supports_metadata_cast(&unsupported_where));

    let err = validate_metadata_cast_shapes(
        &MetadataCastPrimsDescriptor::PointwiseCast {
            input_dtype: MetadataDType::I32,
        },
        &[&[2]],
        "MetadataCastPointwise",
    )
    .unwrap_err();
    assert!(err.to_string().contains("expects 2 shapes"));

    let err = cast_metadata_value::<u8, _>(-1_i32, "mask").unwrap_err();
    assert!(err.to_string().contains("mask cannot be represented"));

    let err = for_each_index_result(&[2, 2], |idx| {
        if idx == [1, 0] {
            Err(tenferro_device::Error::InvalidArgument("stop".into()))
        } else {
            Ok(())
        }
    })
    .unwrap_err();
    assert!(err.to_string().contains("stop"));

    let scalar = Tensor::<f64>::zeros(&[2], CPU, MemoryOrder::ColumnMajor).unwrap();
    let wrong_where_inputs = [
        MetadataScalarTensorRef::Scalar(&scalar),
        MetadataScalarTensorRef::Scalar(&scalar),
        MetadataScalarTensorRef::Scalar(&scalar),
    ];
    let err = validate_where_bridge_inputs(&wrong_where_inputs).unwrap_err();
    assert!(err.to_string().contains("metadata condition input"));

    let wrong_pointwise_inputs = [MetadataScalarTensorRef::Scalar(&scalar)];
    let err = validate_pointwise_cast_bridge_inputs(&wrong_pointwise_inputs).unwrap_err();
    assert!(err.to_string().contains("metadata input"));
}
