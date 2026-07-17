use tenferro_core_ops::PrimitiveOpKind;

use crate::{
    capability_output_dtype, BackendId, CapabilityAxis, CapabilityQuery, DType, Error,
    OperationCapability, SupportLevel, TensorBackendCapability,
};

struct DescriptorBackend;

const CAPABILITIES: &[OperationCapability] = &[
    OperationCapability {
        backend: BackendId::Cpu,
        op: PrimitiveOpKind::Add,
        dtype: DType::I32,
        output_dtype: DType::I32,
        result: SupportLevel::Native,
        read_inputs: SupportLevel::Native,
        write_output: SupportLevel::Native,
        strided_output: SupportLevel::Native,
        accumulation: SupportLevel::Unsupported,
    },
    OperationCapability {
        backend: BackendId::Cpu,
        op: PrimitiveOpKind::DotGeneral,
        dtype: DType::F64,
        output_dtype: DType::F64,
        result: SupportLevel::Native,
        read_inputs: SupportLevel::FallbackCopy,
        write_output: SupportLevel::FallbackCopy,
        strided_output: SupportLevel::Unsupported,
        accumulation: SupportLevel::FallbackCopy,
    },
];

impl TensorBackendCapability for DescriptorBackend {
    fn backend_id(&self) -> BackendId {
        BackendId::Cpu
    }

    fn capabilities(&self) -> &'static [OperationCapability] {
        CAPABILITIES
    }
}

#[test]
fn support_levels_order_copy_fallback_between_unsupported_and_native() {
    assert!(!SupportLevel::Unsupported.is_supported());
    assert!(SupportLevel::FallbackCopy.is_supported());
    assert!(SupportLevel::Native.is_supported());
    assert!(SupportLevel::Native > SupportLevel::FallbackCopy);
    assert!(SupportLevel::FallbackCopy > SupportLevel::Unsupported);
}

#[test]
fn backend_capability_query_reports_axis_specific_support() {
    let backend = DescriptorBackend;
    let add = backend
        .capability(CapabilityQuery::new(PrimitiveOpKind::Add, DType::I32))
        .expect("add/i32 entry should exist");

    assert_eq!(add.backend, BackendId::Cpu);
    assert_eq!(add.axis(CapabilityAxis::OwnedResult), SupportLevel::Native);
    assert_eq!(add.axis(CapabilityAxis::ReadInputs), SupportLevel::Native);
    assert_eq!(add.axis(CapabilityAxis::WriteOutput), SupportLevel::Native);
    assert_eq!(
        add.axis(CapabilityAxis::StridedOutput),
        SupportLevel::Native
    );
    assert_eq!(
        add.axis(CapabilityAxis::Accumulation),
        SupportLevel::Unsupported
    );

    let dot = backend
        .capability(CapabilityQuery::new(
            PrimitiveOpKind::DotGeneral,
            DType::F64,
        ))
        .expect("dot/f64 entry should exist");
    assert_eq!(
        dot.axis(CapabilityAxis::ReadInputs),
        SupportLevel::FallbackCopy
    );
    assert_eq!(
        dot.axis(CapabilityAxis::WriteOutput),
        SupportLevel::FallbackCopy
    );
    assert!(backend
        .require_capability(
            CapabilityQuery::new(PrimitiveOpKind::DotGeneral, DType::F64),
            CapabilityAxis::Accumulation,
        )
        .is_ok());

    assert!(
        backend
            .capability(CapabilityQuery::new(PrimitiveOpKind::Add, DType::Bool))
            .is_none(),
        "missing descriptor entries remain explicit rather than guessed"
    );
}

#[test]
fn backend_require_capability_reports_structured_unsupported_errors() {
    let backend = DescriptorBackend;
    let err = backend
        .require_capability(
            CapabilityQuery::new(PrimitiveOpKind::Add, DType::Bool),
            CapabilityAxis::OwnedResult,
        )
        .unwrap_err();

    assert!(matches!(
        err,
        Error::UnsupportedDTypeConversion {
            op: "add",
            from: DType::Bool,
            to: DType::Bool,
            ..
        }
    ));

    let err = backend
        .require_capability(
            CapabilityQuery::new(PrimitiveOpKind::Add, DType::I32),
            CapabilityAxis::Accumulation,
        )
        .unwrap_err();
    assert!(matches!(
        err,
        Error::UnsupportedDTypeConversion {
            op: "add",
            from: DType::I32,
            to: DType::I32,
            ..
        }
    ));
}

#[test]
fn backend_ids_report_stable_names() {
    assert_eq!(BackendId::Cpu.as_str(), "cpu");
    assert_eq!(BackendId::Cuda.as_str(), "cuda");
    assert_eq!(BackendId::WebGpu.as_str(), "webgpu");
    assert_eq!(BackendId::Other("custom").as_str(), "custom");
    assert_eq!(BackendId::Other("custom").to_string(), "custom");
}

#[test]
fn capability_output_dtype_reuses_catalog_dtype_policy() {
    assert_eq!(
        capability_output_dtype(PrimitiveOpKind::Compare, DType::I64),
        Some(DType::Bool)
    );
    assert_eq!(
        capability_output_dtype(PrimitiveOpKind::Abs, DType::C32),
        Some(DType::F32)
    );
    assert_eq!(
        capability_output_dtype(PrimitiveOpKind::Add, DType::C64),
        Some(DType::C64)
    );
    assert_eq!(
        capability_output_dtype(PrimitiveOpKind::Compare, DType::Bool),
        Some(DType::Bool)
    );
    assert_eq!(
        capability_output_dtype(PrimitiveOpKind::Div, DType::I32),
        Some(DType::I32)
    );
    assert_eq!(
        capability_output_dtype(PrimitiveOpKind::Rem, DType::I64),
        Some(DType::I64)
    );
    assert_eq!(
        capability_output_dtype(PrimitiveOpKind::Pow, DType::I32),
        Some(DType::I32)
    );
    assert_eq!(
        capability_output_dtype(PrimitiveOpKind::Add, DType::Bool),
        None
    );
    assert_eq!(
        capability_output_dtype(PrimitiveOpKind::Sqrt, DType::F32),
        Some(DType::F32)
    );
    assert_eq!(
        capability_output_dtype(PrimitiveOpKind::Sqrt, DType::I32),
        None
    );
    assert_eq!(
        capability_output_dtype(PrimitiveOpKind::Exp, DType::C64),
        Some(DType::C64)
    );
    assert_eq!(
        capability_output_dtype(PrimitiveOpKind::Exp, DType::I64),
        None
    );
    assert_eq!(
        capability_output_dtype(PrimitiveOpKind::Abs, DType::F32),
        Some(DType::F32)
    );
    assert_eq!(
        capability_output_dtype(PrimitiveOpKind::Abs, DType::F64),
        Some(DType::F64)
    );
    assert_eq!(
        capability_output_dtype(PrimitiveOpKind::Abs, DType::I32),
        Some(DType::I32)
    );
    assert_eq!(
        capability_output_dtype(PrimitiveOpKind::Abs, DType::I64),
        Some(DType::I64)
    );
    assert_eq!(
        capability_output_dtype(PrimitiveOpKind::Abs, DType::C64),
        Some(DType::F64)
    );
    assert_eq!(
        capability_output_dtype(PrimitiveOpKind::Abs, DType::Bool),
        None
    );
    assert_eq!(
        capability_output_dtype(PrimitiveOpKind::Compare, DType::C64),
        None
    );
    assert_eq!(
        capability_output_dtype(PrimitiveOpKind::Select, DType::C32),
        Some(DType::C32)
    );
    assert_eq!(
        capability_output_dtype(PrimitiveOpKind::Convert, DType::Bool),
        Some(DType::Bool)
    );
    assert_eq!(
        capability_output_dtype(PrimitiveOpKind::Constant, DType::I64),
        Some(DType::I64)
    );
}
