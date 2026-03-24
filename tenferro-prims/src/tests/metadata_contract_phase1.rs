use crate::{
    MetadataBinaryOp, MetadataConstantValue, MetadataDType, MetadataGenerateOp,
    MetadataPrimsDescriptor, MetadataReductionOp, MetadataTensorMut, MetadataTensorRef,
    MetadataTernaryOp,
};
use tenferro_device::LogicalMemorySpace;
use tenferro_tensor::{MemoryOrder, Tensor};

#[test]
fn metadata_family_exposes_dtype_aware_generate_binary_ternary_and_reduction_contracts() {
    let generate = MetadataPrimsDescriptor::Generate {
        op: MetadataGenerateOp::IotaStartZero,
        output_dtype: MetadataDType::I32,
    };
    let generate_i32 = MetadataPrimsDescriptor::Generate {
        op: MetadataGenerateOp::Constant(MetadataConstantValue::I32(3)),
        output_dtype: MetadataDType::I32,
    };
    let generate_bool = MetadataPrimsDescriptor::Generate {
        op: MetadataGenerateOp::Constant(MetadataConstantValue::Bool(true)),
        output_dtype: MetadataDType::Bool,
    };
    let binary = MetadataPrimsDescriptor::Binary {
        op: MetadataBinaryOp::NotEqual,
        lhs_dtype: MetadataDType::I32,
        rhs_dtype: MetadataDType::I32,
        output_dtype: MetadataDType::Bool,
    };
    let bitand = MetadataPrimsDescriptor::Binary {
        op: MetadataBinaryOp::BitAnd,
        lhs_dtype: MetadataDType::I32,
        rhs_dtype: MetadataDType::I32,
        output_dtype: MetadataDType::I32,
    };
    let ternary = MetadataPrimsDescriptor::Ternary {
        op: MetadataTernaryOp::Where,
        cond_dtype: MetadataDType::Bool,
        lhs_dtype: MetadataDType::I32,
        rhs_dtype: MetadataDType::I32,
        output_dtype: MetadataDType::I32,
    };
    let reduction = MetadataPrimsDescriptor::Reduction {
        modes_a: vec![0, 1],
        modes_c: vec![1],
        input_dtype: MetadataDType::Bool,
        output_dtype: MetadataDType::I32,
        op: MetadataReductionOp::Sum,
    };

    assert!(matches!(
        generate,
        MetadataPrimsDescriptor::Generate {
            op: MetadataGenerateOp::IotaStartZero,
            output_dtype: MetadataDType::I32,
        }
    ));
    assert!(matches!(
        generate_i32,
        MetadataPrimsDescriptor::Generate {
            op: MetadataGenerateOp::Constant(MetadataConstantValue::I32(3)),
            output_dtype: MetadataDType::I32,
        }
    ));
    assert!(matches!(
        generate_bool,
        MetadataPrimsDescriptor::Generate {
            op: MetadataGenerateOp::Constant(MetadataConstantValue::Bool(true)),
            output_dtype: MetadataDType::Bool,
        }
    ));
    assert!(matches!(
        binary,
        MetadataPrimsDescriptor::Binary {
            op: MetadataBinaryOp::NotEqual,
            lhs_dtype: MetadataDType::I32,
            rhs_dtype: MetadataDType::I32,
            output_dtype: MetadataDType::Bool,
        }
    ));
    assert!(matches!(
        bitand,
        MetadataPrimsDescriptor::Binary {
            op: MetadataBinaryOp::BitAnd,
            lhs_dtype: MetadataDType::I32,
            rhs_dtype: MetadataDType::I32,
            output_dtype: MetadataDType::I32,
        }
    ));
    assert!(matches!(
        ternary,
        MetadataPrimsDescriptor::Ternary {
            op: MetadataTernaryOp::Where,
            cond_dtype: MetadataDType::Bool,
            lhs_dtype: MetadataDType::I32,
            rhs_dtype: MetadataDType::I32,
            output_dtype: MetadataDType::I32,
        }
    ));
    assert!(matches!(
        reduction,
        MetadataPrimsDescriptor::Reduction {
            modes_a,
            modes_c,
            input_dtype: MetadataDType::Bool,
            output_dtype: MetadataDType::I32,
            op: MetadataReductionOp::Sum,
        } if modes_a == vec![0, 1] && modes_c == vec![1]
    ));
}

#[test]
fn metadata_family_distinguishes_i32_and_bool_tensor_handles() {
    let i32_input = Tensor::<i32>::zeros(
        &[2, 2],
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let bool_input = Tensor::<u8>::zeros(
        &[2, 2],
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let mut i32_output = Tensor::<i32>::zeros(
        &[2, 2],
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let mut bool_output = Tensor::<u8>::zeros(
        &[2, 2],
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    )
    .unwrap();

    let i32_ref = MetadataTensorRef::I32(&i32_input);
    let bool_ref = MetadataTensorRef::Bool(&bool_input);
    let i32_mut = MetadataTensorMut::I32(&mut i32_output);
    let bool_mut = MetadataTensorMut::Bool(&mut bool_output);

    assert_eq!(i32_ref.dtype(), MetadataDType::I32);
    assert_eq!(bool_ref.dtype(), MetadataDType::Bool);
    assert_eq!(i32_mut.dtype(), MetadataDType::I32);
    assert_eq!(bool_mut.dtype(), MetadataDType::Bool);
    assert_ne!(i32_ref.dtype(), bool_ref.dtype());

    fn accepts_ref<'a>(_: MetadataTensorRef<'a>) {}
    fn accepts_mut<'a>(_: MetadataTensorMut<'a>) {}

    accepts_ref(i32_ref);
    accepts_ref(bool_ref);
    accepts_mut(i32_mut);
    accepts_mut(bool_mut);
}
