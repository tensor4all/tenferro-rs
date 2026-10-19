use std::mem::{size_of, size_of_val};

use tenferro_runtime::program::CoreSemanticOp;
use tenferro_runtime::{
    CompareDir, DType, DotGeneralConfig, InputSignature, InputSignatureEntry, InputSpecializationProjection,
    InputSpecializationRequirements, LayoutClass, LayoutSpecialization,
    PlacementSpecialization, SpecializationError,
    SpecializationRequirements,
};
use tenferro_tensor::{
    GatherConfig, PadConfig, Placement, ScatterConfig, ShapeVec, SliceConfig,
    StrideVec,
};

use super::{
    checked_specialization_heap_retained_bytes, concrete_axes_for_rank, cpu_operation_kind,
    minimum_specialization_requirements, CpuPreparedKind,
};

#[test]
fn managed_ingress_requires_a_registered_matching_allocation_domain() {
    use tenferro_tensor::{AllocationDomainId, MemoryKind};
    let own = AllocationDomainId::fresh();
    let foreign = AllocationDomainId::fresh();
    let mut placement = Placement {
        memory_kind: MemoryKind::Managed,
        device: None,
        cpu_affinity: None,
    };
    assert!(super::cpu_input_signature(
        &placement,
        Some("managed"),
        Some(own),
        Some(own)
    ));
    for (family, input, registered) in [
        (Some("managed"), Some(foreign), Some(own)),
        (Some("managed"), None, Some(own)),
        (Some("managed"), Some(own), None),
        (None, Some(own), Some(own)),
        (None, None, Some(own)),
    ] {
        assert!(!super::cpu_input_signature(
            &placement, family, input, registered
        ));
    }
    placement.memory_kind = MemoryKind::Device;
    assert!(!super::cpu_input_signature(
        &placement,
        Some("managed"),
        Some(own),
        Some(own)
    ));
}

#[test]
fn public_cpu_runtime_registration_exposes_complete_preparation_capabilities() {
    let backend = crate::CpuEngine::serial();

    let registration = crate::runtime_engine_registration(&backend).expect("CPU registration");

    assert_eq!(registration.engine_id().as_str(), "tenferro-cpu.default.v1");
    let capabilities = registration.capabilities();
    assert!(capabilities.elementwise().is_some());
    assert!(capabilities.reduction().is_some());
    assert!(capabilities.indexing().is_some());
    assert!(capabilities.dot_general().is_some());
    assert!(capabilities.layout().is_some());
    assert_eq!(
        registration
            .provider_device_identity()
            .provider_id()
            .as_str(),
        crate::cpu_provider_id()
    );
    assert_eq!(
        registration.provider_device_identity().target_identity(),
        "host"
    );
}

// A build compiles exactly one CPU backend, so a single backend is registered
// under the caller's engine IDs and reports the compiled provider identity.
#[cfg(any(feature = "native", feature = "blas"))]
#[test]
fn public_cpu_runtime_registration_uses_the_compiled_provider_under_caller_engine_ids() {
    use tenferro_runtime::{EngineId, Runtime};

    let backend = crate::test_support::cpu_engine(1).expect("CPU backend");
    let primary = crate::runtime_engine_registration_with_id(
        &backend,
        EngineId::new("tenferro-cpu.primary.identity.v1").expect("primary engine ID"),
    )
    .expect("primary CPU registration");
    let secondary = crate::runtime_engine_registration_with_id(
        &backend,
        EngineId::new("tenferro-cpu.secondary.identity.v1").expect("secondary engine ID"),
    )
    .expect("secondary CPU registration");

    assert_eq!(
        primary.provider_device_identity().provider_id().as_str(),
        crate::cpu_provider_id()
    );
    assert_eq!(
        primary.provider_device_identity(),
        secondary.provider_device_identity()
    );

    let mut builder = Runtime::builder();
    builder
        .register_engine(primary)
        .expect("register primary CPU engine");
    // One compiled provider owns one device target, so a second registration of
    // the same provider is rejected rather than aliased.
    let duplicate = builder.register_engine(secondary);
    assert!(
        matches!(
            duplicate,
            Err(tenferro_runtime::RuntimeConfigError::DuplicateProviderDeviceTarget { .. })
        ),
        "a second engine for one compiled provider must be rejected: {duplicate:?}"
    );
    assert_eq!(
        builder
            .build()
            .expect("runtime")
            .snapshot()
            .unwrap()
            .engine_count(),
        1
    );
}

#[test]
fn cpu_family_minimum_specialization_matches_adapter_contract() {
    let signature = two_input_signature();

    assert_minimum_requirements(
        &minimum_specialization_requirements(CpuPreparedKind::Elementwise, &signature)
            .expect("elementwise requirements"),
        &[&[], &[]],
        LayoutSpecialization::None,
    );
    assert_minimum_requirements(
        &minimum_specialization_requirements(CpuPreparedKind::Reduction, &signature)
            .expect("reduction requirements"),
        &[&[], &[]],
        LayoutSpecialization::None,
    );
    assert_minimum_requirements(
        &minimum_specialization_requirements(CpuPreparedKind::Indexing, &signature)
            .expect("indexing requirements"),
        &[&[0, 1], &[0]],
        LayoutSpecialization::None,
    );
    assert_minimum_requirements(
        &minimum_specialization_requirements(CpuPreparedKind::DotGeneral, &signature)
            .expect("dot requirements"),
        &[&[0, 1], &[0]],
        LayoutSpecialization::Class,
    );
    assert_minimum_requirements(
        &minimum_specialization_requirements(CpuPreparedKind::Layout, &signature)
            .expect("layout requirements"),
        &[&[], &[]],
        LayoutSpecialization::None,
    );
}

#[test]
fn cpu_specialization_projection_retained_bytes_count_owned_heap_payloads() {
    let signature = two_input_signature();
    let requirements = minimum_specialization_requirements(CpuPreparedKind::DotGeneral, &signature)
        .expect("dot requirements");
    let projection = requirements.project(&signature).expect("projection");

    let expected = size_of_val(requirements.inputs())
        + (2 + 1) * size_of::<u32>()
        + size_of_val(projection.inputs())
        + (2 + 1) * size_of::<(u32, usize)>();
    assert_eq!(
        checked_specialization_heap_retained_bytes(&projection),
        Some(expected)
    );
}

#[test]
fn cpu_specialization_projection_retained_bytes_count_spilled_exact_strides() {
    let signature = high_rank_signature();
    let mut builder = InputSpecializationRequirements::builder();
    builder
        .rank(true)
        .layout(LayoutSpecialization::ExactStrides);
    let requirements =
        SpecializationRequirements::new(vec![builder.build().expect("requirements")]);
    let projection = requirements.project(&signature).expect("projection");

    let expected = size_of::<InputSpecializationRequirements>()
        + size_of::<InputSpecializationProjection>()
        + 9 * size_of::<isize>();
    assert_eq!(
        checked_specialization_heap_retained_bytes(&projection),
        Some(expected)
    );
}

#[test]
fn current_core_vocabulary_has_explicit_cpu_family_classification() {
    let cases = all_current_core_variants();
    assert_eq!(cases.len(), 50);
    for (operation, expected) in cases {
        assert_eq!(
            cpu_operation_kind(&operation),
            Some(expected),
            "{operation:?}"
        );
    }
}

#[test]
fn concrete_axis_projection_overflow_is_reported_before_allocation() {
    let error = concrete_axes_for_rank(3, usize::MAX).expect_err("rank too large");

    assert!(matches!(
        error,
        tenferro_runtime::PrepareError::Specialization {
            source: SpecializationError::ProjectionOverflow {
                input: 3,
                rank: usize::MAX
            }
        }
    ));
}

fn two_input_signature() -> InputSignature {
    InputSignature::new(vec![
        InputSignatureEntry::new(
            DType::F64,
            vec![2, 3].into_iter().collect::<ShapeVec>(),
            Placement::default(),
            layout_class(),
            vec![1, 2].into_iter().collect::<StrideVec>(),
            Some(4),
        )
        .expect("lhs signature"),
        InputSignatureEntry::new(
            DType::F64,
            vec![5].into_iter().collect::<ShapeVec>(),
            Placement::default(),
            layout_class(),
            vec![1].into_iter().collect::<StrideVec>(),
            Some(4),
        )
        .expect("rhs signature"),
    ])
}

fn high_rank_signature() -> InputSignature {
    InputSignature::new(vec![InputSignatureEntry::new(
        DType::F64,
        vec![1; 9].into_iter().collect::<ShapeVec>(),
        Placement::default(),
        layout_class(),
        vec![1; 9].into_iter().collect::<StrideVec>(),
        None,
    )
    .expect("high-rank signature")])
}

fn layout_class() -> LayoutClass {
    LayoutClass::new("tenferro.layout.compact-col-major.v1").expect("layout class")
}

fn assert_minimum_requirements(
    requirements: &SpecializationRequirements,
    expected_axes: &[&[u32]],
    expected_layout: LayoutSpecialization,
) {
    assert_eq!(requirements.inputs().len(), expected_axes.len());
    for (input, axes) in requirements.inputs().iter().zip(expected_axes) {
        assert!(input.specializes_dtype());
        assert!(input.specializes_rank());
        assert_eq!(input.concrete_dimensions(), *axes);
        assert_eq!(input.placement(), PlacementSpecialization::None);
        assert_eq!(input.layout(), expected_layout);
        assert_eq!(input.alignment_log2(), None);
    }
}

fn all_current_core_variants() -> Vec<(CoreSemanticOp, CpuPreparedKind)> {
    use CpuPreparedKind::{DotGeneral, Elementwise, Indexing, Layout, Reduction};

    vec![
        (CoreSemanticOp::Add, Elementwise),
        (CoreSemanticOp::Sub, Elementwise),
        (CoreSemanticOp::Mul, Elementwise),
        (CoreSemanticOp::Neg, Elementwise),
        (CoreSemanticOp::Conj, Elementwise),
        (
            CoreSemanticOp::DotGeneral {
                config: DotGeneralConfig {
                    lhs_contracting_dims: [1].as_slice().into(),
                    rhs_contracting_dims: [0].as_slice().into(),
                    lhs_batch_dims: [].as_slice().into(),
                    rhs_batch_dims: [].as_slice().into(),
                },
            },
            DotGeneral,
        ),
        (CoreSemanticOp::Transpose { perm: vec![1, 0] }, Layout),
        (
            CoreSemanticOp::Reshape {
                to_shape: vec![1_usize.into()],
            },
            Layout,
        ),
        (
            CoreSemanticOp::BroadcastInDim {
                shape: vec![1_usize.into()],
                dims: vec![0],
            },
            Layout,
        ),
        (
            CoreSemanticOp::Convert {
                from: DType::F32,
                to: DType::F64,
            },
            Layout,
        ),
        (
            CoreSemanticOp::Constant {
                dtype: DType::F64,
                bytes: vec![0; 8],
            },
            Layout,
        ),
        (CoreSemanticOp::ReduceSum { axes: vec![0] }, Reduction),
        (CoreSemanticOp::Div, Elementwise),
        (CoreSemanticOp::Rem, Elementwise),
        (CoreSemanticOp::Abs, Elementwise),
        (CoreSemanticOp::Sign, Elementwise),
        (CoreSemanticOp::Maximum, Elementwise),
        (CoreSemanticOp::Minimum, Elementwise),
        (CoreSemanticOp::Compare(CompareDir::Eq), Elementwise),
        (CoreSemanticOp::Select, Elementwise),
        (CoreSemanticOp::Clamp, Elementwise),
        (CoreSemanticOp::Exp, Elementwise),
        (CoreSemanticOp::Log, Elementwise),
        (CoreSemanticOp::Sin, Elementwise),
        (CoreSemanticOp::Cos, Elementwise),
        (CoreSemanticOp::Tanh, Elementwise),
        (CoreSemanticOp::Sqrt, Elementwise),
        (CoreSemanticOp::Rsqrt, Elementwise),
        (CoreSemanticOp::Pow, Elementwise),
        (CoreSemanticOp::Expm1, Elementwise),
        (CoreSemanticOp::Log1p, Elementwise),
        (
            CoreSemanticOp::ExtractDiag {
                axis_a: 0,
                axis_b: 1,
            },
            Layout,
        ),
        (
            CoreSemanticOp::EmbedDiag {
                axis_a: 0,
                axis_b: 1,
            },
            Layout,
        ),
        (CoreSemanticOp::Tril { k: 0 }, Layout),
        (CoreSemanticOp::Triu { k: 0 }, Layout),
        (
            CoreSemanticOp::Gather(GatherConfig {
                offset_dims: vec![1],
                collapsed_slice_dims: vec![0],
                start_index_map: vec![0],
                index_vector_dim: 1,
                slice_sizes: vec![1],
            }),
            Indexing,
        ),
        (
            CoreSemanticOp::GatherDynamicSliceSizes {
                offset_dims: vec![1],
                collapsed_slice_dims: vec![0],
                start_index_map: vec![0],
                index_vector_dim: 1,
                slice_sizes: vec![1_usize.into()],
            },
            Indexing,
        ),
        (
            CoreSemanticOp::Scatter(ScatterConfig {
                update_window_dims: vec![0],
                inserted_window_dims: vec![0],
                scatter_dims_to_operand_dims: vec![0],
                index_vector_dim: 1,
            }),
            Indexing,
        ),
        (
            CoreSemanticOp::Slice(SliceConfig {
                starts: vec![0],
                limits: vec![1],
                strides: vec![1],
            }),
            Indexing,
        ),
        (
            CoreSemanticOp::DynamicSlice {
                slice_sizes: vec![1],
            },
            Indexing,
        ),
        (CoreSemanticOp::DynamicUpdateSlice, Indexing),
        (
            CoreSemanticOp::Pad(PadConfig {
                edge_padding_low: vec![0],
                edge_padding_high: vec![0],
                interior_padding: vec![0],
            }),
            Indexing,
        ),
        (
            CoreSemanticOp::Concatenate {
                axis: 0,
                input_count: 2,
            },
            Indexing,
        ),
        (CoreSemanticOp::Reverse { axes: vec![0] }, Indexing),
        (CoreSemanticOp::ShapeOf { axis: 0 }, Indexing),
        (CoreSemanticOp::DynamicTruncate { axis: 0 }, Indexing),
        (CoreSemanticOp::PadToMatch { axis: 0 }, Indexing),
        (CoreSemanticOp::ReduceProd { axes: vec![0] }, Reduction),
        (CoreSemanticOp::ReduceMax { axes: vec![0] }, Reduction),
        (CoreSemanticOp::ReduceMin { axes: vec![0] }, Reduction),
    ]
}
