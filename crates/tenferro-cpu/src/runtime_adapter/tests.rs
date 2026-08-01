use std::mem::{size_of, size_of_val};
use std::num::NonZeroUsize;
use std::sync::Arc;

use tenferro_runtime::program::CoreSemanticOp;
use tenferro_runtime::{
    CompareDir, DType, DotGeneralConfig, DotGeneralPreparation, ElementwiseRuntime, EngineId,
    IndexingRuntime, InputSignature, InputSignatureEntry, InputSpecializationProjection,
    InputSpecializationRequirements, LayoutClass, LayoutRuntime, LayoutSpecialization,
    PlacementSpecialization, ReductionRuntime, Runtime, RuntimeCacheOwner, SpecializationError,
    SpecializationRequirements,
};
use tenferro_tensor::{
    CpuDomainId, GatherConfig, PadConfig, Placement, ScatterConfig, ShapeVec, SliceConfig,
    StrideVec, Tensor, TensorIndexing,
};

use super::{
    checked_specialization_heap_retained_bytes, concrete_axes_for_rank, cpu_operation_kind,
    minimum_specialization_requirements, CpuPreparedKind,
};
use crate::CpuBackend;

#[test]
fn cpu_backend_coerces_to_all_runtime_traits() {
    let backend = Arc::new(CpuBackend::new());

    let _: Arc<dyn ElementwiseRuntime> = backend.clone();
    let _: Arc<dyn ReductionRuntime> = backend.clone();
    let _: Arc<dyn IndexingRuntime> = backend.clone();
    let _: Arc<dyn DotGeneralPreparation> = backend.clone();
    let _: Arc<dyn LayoutRuntime> = backend.clone();
    let _: Arc<dyn RuntimeCacheOwner> = backend;
}

#[test]
fn public_cpu_runtime_registration_exposes_complete_preparation_capabilities() {
    let backend = CpuBackend::new();

    let registration = crate::runtime_engine_registration(&backend).expect("CPU registration");

    assert_eq!(registration.engine_id().as_str(), "tenferro-cpu.default.v1");
    let capabilities = registration.capabilities();
    assert!(capabilities.elementwise().is_some());
    assert!(capabilities.reduction().is_some());
    assert!(capabilities.indexing().is_some());
    assert!(capabilities.dot_general().is_some());
    assert!(capabilities.layout().is_some());
    let expected_provider_id = match backend.kind() {
        crate::CpuBackendKind::Faer => "tenferro.cpu.faer",
        crate::CpuBackendKind::Blas => "tenferro.cpu.blas",
    };
    assert_eq!(
        registration
            .provider_device_identity()
            .provider_id()
            .as_str(),
        expected_provider_id
    );
    assert_eq!(
        registration.provider_device_identity().target_identity(),
        format!("domain:{}", backend.execution_info().domain_id().as_u64())
    );
}

#[test]
fn public_cpu_runtime_registration_tracks_distinct_selected_cpu_domains() {
    let topology = crate::discover_cpu_topology().expect("CPU topology");
    if topology.nodes().len() < 2 {
        return;
    }
    let first_node = &topology.nodes()[0];
    let second_node = &topology.nodes()[1];
    let first_domain_id = CpuDomainId::new(101);
    let second_domain_id = CpuDomainId::new(102);
    let first_domain = crate::ExternalCpuDomain::new(
        first_domain_id,
        crate::ResolvedCpuPlacement::NumaNode {
            id: first_node.id(),
            cpus: first_node.cpus().clone(),
        },
        Arc::new(crate::CpuContext::with_threads(1).expect("first CPU context")),
        NonZeroUsize::new(1).unwrap(),
        crate::CpuPlacementGuarantee::AdvisoryDeclared,
    )
    .expect("first CPU domain");
    let second_domain = crate::ExternalCpuDomain::new(
        second_domain_id,
        crate::ResolvedCpuPlacement::NumaNode {
            id: second_node.id(),
            cpus: second_node.cpus().clone(),
        },
        Arc::new(crate::CpuContext::with_threads(1).expect("second CPU context")),
        NonZeroUsize::new(1).unwrap(),
        crate::CpuPlacementGuarantee::AdvisoryDeclared,
    )
    .expect("second CPU domain");
    let first = crate::CpuBackend::from_external_managed_domains(first_domain_id, [first_domain])
        .expect("first external CPU backend");
    let second =
        crate::CpuBackend::from_external_managed_domains(second_domain_id, [second_domain])
            .expect("second external CPU backend");

    let first_registration = crate::runtime_engine_registration_with_id(
        &first,
        EngineId::new("tenferro-cpu.domain-first.v1").expect("first engine ID"),
    )
    .expect("first CPU registration");
    let second_registration = crate::runtime_engine_registration_with_id(
        &second,
        EngineId::new("tenferro-cpu.domain-second.v1").expect("second engine ID"),
    )
    .expect("second CPU registration");
    assert_ne!(
        first_registration.provider_device_identity(),
        second_registration.provider_device_identity()
    );
    assert_eq!(
        first_registration
            .provider_device_identity()
            .target_identity(),
        "domain:101"
    );
    assert_eq!(
        second_registration
            .provider_device_identity()
            .target_identity(),
        "domain:102"
    );
}

#[test]
fn public_cpu_runtime_registration_allows_two_selected_cpu_domains_in_one_runtime() {
    let topology = crate::discover_cpu_topology().expect("CPU topology");
    if topology.nodes().len() < 2 {
        return;
    }
    let first_node = &topology.nodes()[0];
    let second_node = &topology.nodes()[1];
    let first_domain_id = CpuDomainId::new(201);
    let second_domain_id = CpuDomainId::new(202);
    let first_domain = crate::ExternalCpuDomain::new(
        first_domain_id,
        crate::ResolvedCpuPlacement::NumaNode {
            id: first_node.id(),
            cpus: first_node.cpus().clone(),
        },
        Arc::new(crate::CpuContext::with_threads(1).expect("first CPU context")),
        NonZeroUsize::new(1).unwrap(),
        crate::CpuPlacementGuarantee::AdvisoryDeclared,
    )
    .expect("first CPU domain");
    let second_domain = crate::ExternalCpuDomain::new(
        second_domain_id,
        crate::ResolvedCpuPlacement::NumaNode {
            id: second_node.id(),
            cpus: second_node.cpus().clone(),
        },
        Arc::new(crate::CpuContext::with_threads(1).expect("second CPU context")),
        NonZeroUsize::new(1).unwrap(),
        crate::CpuPlacementGuarantee::AdvisoryDeclared,
    )
    .expect("second CPU domain");
    let first_backend = CpuBackend::from_external_managed_domains(first_domain_id, [first_domain])
        .expect("first external CPU backend");
    let second_backend =
        CpuBackend::from_external_managed_domains(second_domain_id, [second_domain])
            .expect("second external CPU backend");
    let first_id = EngineId::new("tenferro-cpu.first.v1").expect("first engine ID");
    let second_id = EngineId::new("tenferro-cpu.second.v1").expect("second engine ID");

    let mut builder = Runtime::builder();
    builder
        .register_engine(
            crate::runtime_engine_registration_with_id(&first_backend, first_id.clone())
                .expect("first CPU registration"),
        )
        .expect("register first CPU engine");
    builder
        .register_engine(
            crate::runtime_engine_registration_with_id(&second_backend, second_id.clone())
                .expect("second CPU registration"),
        )
        .expect("register second CPU engine");
    let runtime = builder.build().expect("runtime with two CPU engines");
    let snapshot = runtime.snapshot().expect("runtime snapshot");

    assert_eq!(snapshot.engine_count(), 2);
    assert_eq!(snapshot.engine(&first_id).unwrap().engine_id(), &first_id);
    assert_eq!(snapshot.engine(&second_id).unwrap().engine_id(), &second_id);
}

#[cfg(all(feature = "cpu-faer", feature = "cpu-blas"))]
#[test]
fn public_cpu_runtime_registration_supports_distinct_compiled_provider_kinds() {
    let faer = CpuBackend::with_threads_and_kind(1, crate::CpuBackendKind::Faer)
        .expect("faer CPU backend");
    let blas = CpuBackend::with_threads_and_kind(1, crate::CpuBackendKind::Blas)
        .expect("BLAS CPU backend");

    assert_ne!(faer.kind(), blas.kind());
    let faer_registration = crate::runtime_engine_registration_with_id(
        &faer,
        EngineId::new("tenferro-cpu.faer.identity.v1").expect("faer engine ID"),
    )
    .expect("faer CPU registration");
    let blas_registration = crate::runtime_engine_registration_with_id(
        &blas,
        EngineId::new("tenferro-cpu.blas.identity.v1").expect("BLAS engine ID"),
    )
    .expect("BLAS CPU registration");
    assert_eq!(
        faer_registration
            .provider_device_identity()
            .provider_id()
            .as_str(),
        "tenferro.cpu.faer"
    );
    assert_eq!(
        blas_registration
            .provider_device_identity()
            .provider_id()
            .as_str(),
        "tenferro.cpu.blas"
    );
    assert_ne!(
        faer_registration.provider_device_identity(),
        blas_registration.provider_device_identity()
    );

    let mut builder = Runtime::builder();
    builder
        .register_engine(
            crate::runtime_engine_registration_with_id(
                &faer,
                EngineId::new("tenferro-cpu.faer.v1").expect("faer engine ID"),
            )
            .expect("faer CPU registration"),
        )
        .expect("register faer CPU engine");
    builder
        .register_engine(
            crate::runtime_engine_registration_with_id(
                &blas,
                EngineId::new("tenferro-cpu.blas.v1").expect("BLAS engine ID"),
            )
            .expect("BLAS CPU registration"),
        )
        .expect("register BLAS CPU engine");
    assert_eq!(
        builder
            .build()
            .expect("runtime")
            .snapshot()
            .unwrap()
            .engine_count(),
        2
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
fn cpu_runtime_adapter_source_stays_metadata_only() {
    let source = include_str!("../runtime_adapter.rs");

    assert!(source.contains("assemble_executable_engine_registration("));
    assert!(!source.contains("ExecutableEngineContract::new("));
    assert!(!source.contains("ProviderExecutableBinding::new("));
    assert!(!source.contains("use tenferro_tensor"));
    assert!(!source.contains("Tensor::"));
    assert!(!source.contains("lock_engine_resources"));
    assert!(!source.contains(".install("));
    assert!(!source.contains("with_backend_session"));
    assert!(!source.contains("execute_"));
}

#[test]
fn cpu_runtime_adapter_has_only_the_six_runtime_trait_impls_on_cpu_backend() {
    let source = include_str!("../runtime_adapter.rs");

    for implementation in [
        "impl ElementwiseRuntime for CpuBackend",
        "impl ReductionRuntime for CpuBackend",
        "impl IndexingRuntime for CpuBackend",
        "impl DotGeneralPreparation for CpuBackend",
        "impl LayoutRuntime for CpuBackend",
        "impl RuntimeCacheOwner for CpuBackend",
    ] {
        assert_eq!(
            source.matches(implementation).count(),
            1,
            "{implementation}"
        );
    }
    assert!(!source.contains("pub struct"));
    assert!(!source.contains("pub enum"));
}

#[test]
fn cpu_backend_cache_owner_hooks_report_and_clear_current_engine_caches() {
    let mut backend = CpuBackend::with_threads(1).expect("backend");

    let stats = RuntimeCacheOwner::cache_stats(&backend).expect("cache stats");
    assert_eq!(stats.entries, 0);
    assert_eq!(stats.retained_bytes, 0);
    assert_eq!(stats.hits, 0);
    assert_eq!(stats.misses, 0);
    assert_eq!(stats.evictions, 0);
    assert_eq!(stats.clears, 0);

    let operand = Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]).expect("operand");
    let indices = Tensor::from_vec_col_major(vec![2, 1], vec![0_i64, 2]).expect("indices");
    let config = GatherConfig {
        offset_dims: vec![],
        collapsed_slice_dims: vec![0],
        start_index_map: vec![0],
        index_vector_dim: 1,
        slice_sizes: vec![1],
    };
    backend
        .gather(&operand, &indices, &config)
        .expect("compile gather plan");
    backend
        .gather(&operand, &indices, &config)
        .expect("reuse gather plan");
    let populated = RuntimeCacheOwner::cache_stats(&backend).expect("populated cache stats");
    assert_eq!(populated.entries, 1);
    assert!(populated.retained_bytes > 0);
    assert_eq!(populated.hits, 1);
    assert_eq!(populated.misses, 1);

    RuntimeCacheOwner::clear_caches(&backend).expect("clear caches");
    let cleared = RuntimeCacheOwner::cache_stats(&backend).expect("cleared cache stats");
    assert_eq!(cleared.entries, 0);
    assert_eq!(cleared.retained_bytes, 0);
    assert_eq!(cleared.clears, 1);
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
            ShapeVec::from_vec(vec![2, 3]),
            Placement::default(),
            layout_class(),
            StrideVec::from_vec(vec![1, 2]),
            Some(4),
        )
        .expect("lhs signature"),
        InputSignatureEntry::new(
            DType::F64,
            ShapeVec::from_vec(vec![5]),
            Placement::default(),
            layout_class(),
            StrideVec::from_vec(vec![1]),
            Some(4),
        )
        .expect("rhs signature"),
    ])
}

fn high_rank_signature() -> InputSignature {
    InputSignature::new(vec![InputSignatureEntry::new(
        DType::F64,
        ShapeVec::from_vec(vec![1; 9]),
        Placement::default(),
        layout_class(),
        StrideVec::from_vec(vec![1; 9]),
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
                    lhs_contracting_dims: vec![1],
                    rhs_contracting_dims: vec![0],
                    lhs_batch_dims: vec![],
                    rhs_batch_dims: vec![],
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
