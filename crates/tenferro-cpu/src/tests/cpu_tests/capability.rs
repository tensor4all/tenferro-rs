use tenferro_core_ops::PrimitiveOpKind;
use tenferro_tensor::{
    BackendId, CapabilityAxis, CapabilityQuery, DType, SupportLevel, TensorBackendCapability,
};

use crate::CpuBackend;

#[test]
fn cpu_capability_table_reports_core_elementwise_reduction_and_dot_support() {
    let backend = CpuBackend::new();

    assert_eq!(backend.backend_id(), BackendId::Cpu);

    let add_i64 = backend
        .capability(CapabilityQuery::new(PrimitiveOpKind::Add, DType::I64))
        .expect("CPU add/i64 should be described");
    assert_eq!(add_i64.result, SupportLevel::Native);
    assert_eq!(add_i64.read_inputs, SupportLevel::Native);
    assert_eq!(add_i64.output_dtype, DType::I64);

    let reduce_prod_c64 = backend
        .capability(CapabilityQuery::new(
            PrimitiveOpKind::ReduceProd,
            DType::C64,
        ))
        .expect("CPU reduce_prod/c64 should be described");
    assert_eq!(reduce_prod_c64.result, SupportLevel::Native);

    let dot_f64 = backend
        .capability(CapabilityQuery::new(
            PrimitiveOpKind::DotGeneral,
            DType::F64,
        ))
        .expect("CPU dot_general/f64 should be described");
    assert_eq!(
        dot_f64.axis(CapabilityAxis::OwnedResult),
        SupportLevel::Native
    );
    assert_eq!(
        dot_f64.axis(CapabilityAxis::ReadInputs),
        SupportLevel::Native
    );
    assert_eq!(
        dot_f64.axis(CapabilityAxis::WriteOutput),
        SupportLevel::Native
    );
    assert_eq!(
        dot_f64.axis(CapabilityAxis::Accumulation),
        SupportLevel::Native
    );

    let neg_i32 = backend
        .capability(CapabilityQuery::new(PrimitiveOpKind::Neg, DType::I32))
        .expect("CPU neg/i32 should be described");
    assert_eq!(neg_i32.result, SupportLevel::Native);

    let neg_i32 = backend
        .require_capability(
            CapabilityQuery::new(PrimitiveOpKind::Neg, DType::I32),
            CapabilityAxis::OwnedResult,
        )
        .unwrap();
    assert_eq!(neg_i32.result, SupportLevel::Native);

    assert!(
        backend
            .capability(CapabilityQuery::new(PrimitiveOpKind::Div, DType::I32))
            .is_none(),
        "descriptor should not claim CPU integer div before #1320 lands"
    );
}
