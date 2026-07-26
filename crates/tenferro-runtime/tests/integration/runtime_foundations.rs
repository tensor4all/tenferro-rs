mod identity {
    use std::error::Error as _;
    use std::fmt::Debug;

    use tenferro_runtime::runtime::{
        EngineId, ExecutionContextIdentity, HardwareClassId, IdentityError, IdentityKind,
        RegistrationIdentity, RuntimeEpoch, RuntimeId,
    };

    fn assert_debug<T: Debug>() {}

    #[test]
    fn identity_types_are_public_debug_and_context_typed() {
        assert_debug::<IdentityKind>();
        assert_debug::<IdentityError>();
        assert_debug::<RuntimeId>();
        assert_debug::<RuntimeEpoch>();
        assert_debug::<EngineId>();
        assert_debug::<HardwareClassId>();
        assert_debug::<RegistrationIdentity>();
        assert_debug::<ExecutionContextIdentity>();

        struct First;
        struct Second;
        let first = ExecutionContextIdentity::of::<First>();
        let first_again = ExecutionContextIdentity::of::<First>();
        let second = ExecutionContextIdentity::of::<Second>();

        assert_eq!(first, first_again);
        assert_ne!(first, second);
        assert!(first.type_name().ends_with("::First"));
    }

    #[test]
    fn identity_engine_and_hardware_ids_accept_the_exact_ascii_grammar() {
        for value in [
            "a.b",
            "tenferro.cpu",
            "tenferro-cpu.host_v1",
            "a1.b2.c3",
            "0.9",
        ] {
            assert_eq!(EngineId::new(value).unwrap().as_str(), value);
            assert_eq!(HardwareClassId::new(value).unwrap().as_str(), value);
        }
    }

    #[test]
    fn identity_engine_and_hardware_ids_reject_the_exact_ascii_grammar() {
        for value in [
            "",
            "a",
            ".a",
            "a.",
            "a..b",
            "a.-b",
            "a.b_",
            "a.b-",
            "A.b",
            "a.B",
            "a.b c",
            "a.b/c",
            "a.b\u{00e9}",
        ] {
            let engine_error = EngineId::new(value).unwrap_err();
            let hardware_error = HardwareClassId::new(value).unwrap_err();
            assert_eq!(engine_error.kind(), IdentityKind::Engine);
            assert_eq!(hardware_error.kind(), IdentityKind::HardwareClass);
        }
    }

    #[test]
    fn identity_error_redacts_a_unique_rejected_value_and_has_no_source() {
        let rejected = "caller-SECRET.invalid";
        let error = EngineId::new(rejected).unwrap_err();

        assert_eq!(error.kind(), IdentityKind::Engine);
        assert!(!error.to_string().contains(rejected));
        assert!(!format!("{error:?}").contains(rejected));
        assert!(error.source().is_none());
    }

    #[test]
    fn identity_public_accessors_return_the_stored_values() {
        let engine = EngineId::new("tenferro.cpu.v1").unwrap();
        let hardware = HardwareClassId::new("tenferro.cpu.host").unwrap();
        let context = ExecutionContextIdentity::of::<u64>();

        assert_eq!(engine.as_str(), "tenferro.cpu.v1");
        assert_eq!(hardware.as_str(), "tenferro.cpu.host");
        assert_eq!(context.type_name(), std::any::type_name::<u64>());
    }
}

mod policy {
    use std::fmt::Debug;

    use tenferro_runtime::runtime::{
        CacheInFlightBehavior, Determinism, EngineId, ExecutionPolicy, HardwareClassId,
        IdentityKind, LayoutClass, PlacementConstraintError, PrepareOptions, PrepareOptionsKey,
        ProgramPlacementConstraint, ResolvedPlanningConfig, ResolvedPlanningKey,
        ResolvedProgramPlacement, StorageClass,
    };

    fn assert_debug<T: Debug>() {}

    fn engine(value: &str) -> EngineId {
        EngineId::new(value).unwrap()
    }

    fn storage(value: &str) -> StorageClass {
        StorageClass::new(value).unwrap()
    }

    fn hardware(value: &str) -> HardwareClassId {
        HardwareClassId::new(value).unwrap()
    }

    #[test]
    fn policy_storage_and_layout_ids_share_the_exact_ascii_grammar() {
        for value in [
            "a.b",
            "tenferro.storage.host",
            "tenferro-layout.col_major",
            "a1.b2.c3",
            "0.9",
        ] {
            assert_eq!(StorageClass::new(value).unwrap().as_str(), value);
            assert_eq!(LayoutClass::new(value).unwrap().as_str(), value);
        }

        for value in [
            "",
            "a",
            ".a",
            "a.",
            "a..b",
            "a.-b",
            "a.b_",
            "a.b-",
            "A.b",
            "a.B",
            "a.b c",
            "a.b/c",
            "a.b\u{00e9}",
        ] {
            assert_eq!(
                StorageClass::new(value).unwrap_err().kind(),
                IdentityKind::StorageClass
            );
            assert_eq!(
                LayoutClass::new(value).unwrap_err().kind(),
                IdentityKind::LayoutClass
            );
        }
    }

    #[test]
    fn policy_placement_constraint_reports_first_duplicate_positions() {
        let engines = [
            engine("tenferro.cpu"),
            engine("tenferro.gpu"),
            engine("tenferro.cpu"),
            engine("tenferro.cpu"),
        ];

        let error = ProgramPlacementConstraint::new(
            engines.to_vec(),
            Some(storage("tenferro.storage.host")),
        )
        .unwrap_err();

        assert!(matches!(
            error,
            PlacementConstraintError::DuplicateEngine {
                ref engine_id,
                first_index: 0,
                duplicate_index: 2,
            } if engine_id.as_str() == "tenferro.cpu"
        ));
    }

    #[test]
    fn policy_placement_constraint_preserves_preference_order_and_accessors() {
        let preferred = engine("tenferro.gpu");
        let fallback = engine("tenferro.cpu");
        let storage = storage("tenferro.storage.pinned-host");

        let constraint = ProgramPlacementConstraint::new(
            vec![preferred.clone(), fallback.clone()],
            Some(storage.clone()),
        )
        .unwrap();

        assert_eq!(constraint.allowed_engines(), &[preferred, fallback]);
        assert_eq!(constraint.storage_class(), Some(&storage));
        assert!(ProgramPlacementConstraint::any()
            .allowed_engines()
            .is_empty());
        assert!(ProgramPlacementConstraint::any().storage_class().is_none());
    }

    #[test]
    fn policy_resolution_covers_both_determinism_values_all_seed_extremes_and_workspace_lattice() {
        let hardware = hardware("tenferro.cpu.host");

        let inherited_unlimited = ResolvedPlanningConfig::resolve(
            &ExecutionPolicy::new(Determinism::Fast, None, 0),
            &PrepareOptions::new(),
            hardware.clone(),
        );
        assert_eq!(inherited_unlimited.determinism(), Determinism::Fast);
        assert_eq!(inherited_unlimited.hard_workspace_limit_bytes(), None);
        assert_eq!(inherited_unlimited.planning_seed(), 0);
        assert_eq!(inherited_unlimited.hardware_class(), &hardware);

        let inherited_finite = ResolvedPlanningConfig::resolve(
            &ExecutionPolicy::new(Determinism::Reproducible, Some(11), u64::MAX),
            &PrepareOptions::new(),
            hardware.clone(),
        );
        assert_eq!(inherited_finite.determinism(), Determinism::Reproducible);
        assert_eq!(inherited_finite.hard_workspace_limit_bytes(), Some(11));
        assert_eq!(inherited_finite.planning_seed(), u64::MAX);

        let zero_override = ResolvedPlanningConfig::resolve(
            &ExecutionPolicy::new(Determinism::Fast, Some(11), 5),
            &PrepareOptions::new()
                .with_hard_workspace_limit_bytes(Some(0))
                .with_planning_seed(u64::MAX),
            hardware.clone(),
        );
        assert_eq!(zero_override.hard_workspace_limit_bytes(), Some(0));
        assert_eq!(zero_override.planning_seed(), u64::MAX);

        let finite_override = ResolvedPlanningConfig::resolve(
            &ExecutionPolicy::new(Determinism::Reproducible, None, 9),
            &PrepareOptions::new()
                .with_hard_workspace_limit_bytes(Some(64))
                .with_planning_seed(0),
            hardware,
        );
        assert_eq!(finite_override.determinism(), Determinism::Reproducible);
        assert_eq!(finite_override.hard_workspace_limit_bytes(), Some(64));
        assert_eq!(finite_override.planning_seed(), 0);
    }

    #[test]
    fn policy_prepare_options_none_inherits_and_some_zero_overrides() {
        let hardware = hardware("tenferro.cpu.host");
        let finite_policy = ExecutionPolicy::new(Determinism::Fast, Some(8), 1);

        let inherit_default = ResolvedPlanningConfig::resolve(
            &finite_policy,
            &PrepareOptions::new(),
            hardware.clone(),
        );
        assert_eq!(inherit_default.hard_workspace_limit_bytes(), Some(8));

        let inherit_explicit_none = ResolvedPlanningConfig::resolve(
            &finite_policy,
            &PrepareOptions::new().with_hard_workspace_limit_bytes(None),
            hardware.clone(),
        );
        assert_eq!(inherit_explicit_none.hard_workspace_limit_bytes(), Some(8));

        let zero_override = ResolvedPlanningConfig::resolve(
            &finite_policy,
            &PrepareOptions::new().with_hard_workspace_limit_bytes(Some(0)),
            hardware.clone(),
        );
        assert_eq!(zero_override.hard_workspace_limit_bytes(), Some(0));

        let unlimited = ResolvedPlanningConfig::resolve(
            &ExecutionPolicy::new(Determinism::Fast, None, 1),
            &PrepareOptions::new().with_hard_workspace_limit_bytes(None),
            hardware,
        );
        assert_eq!(unlimited.hard_workspace_limit_bytes(), None);
    }

    #[test]
    fn policy_prepare_options_builders_and_accessors_round_trip() {
        let placement = ProgramPlacementConstraint::new(
            vec![engine("tenferro.cpu")],
            Some(storage("tenferro.storage.host")),
        )
        .unwrap();

        let options = PrepareOptions::new()
            .with_placement(placement.clone())
            .with_hard_workspace_limit_bytes(Some(4096))
            .with_planning_seed(123)
            .with_cache_in_flight(CacheInFlightBehavior::Refuse);

        assert_eq!(options.placement(), Some(&placement));
        assert_eq!(options.hard_workspace_limit_bytes(), Some(4096));
        assert_eq!(options.planning_seed(), Some(123));
        assert_eq!(options.cache_in_flight(), CacheInFlightBehavior::Refuse);

        let reset = options.with_hard_workspace_limit_bytes(None);
        assert_eq!(reset.hard_workspace_limit_bytes(), None);
        assert_eq!(
            PrepareOptions::new().cache_in_flight(),
            CacheInFlightBehavior::Wait
        );
    }

    #[test]
    fn policy_types_are_debug_and_all_public_accessors_are_callable() {
        assert_debug::<PlacementConstraintError>();
        assert_debug::<Determinism>();
        assert_debug::<StorageClass>();
        assert_debug::<LayoutClass>();
        assert_debug::<ProgramPlacementConstraint>();
        assert_debug::<ResolvedProgramPlacement>();
        assert_debug::<CacheInFlightBehavior>();
        assert_debug::<ExecutionPolicy>();
        assert_debug::<PrepareOptions>();
        assert_debug::<PrepareOptionsKey>();
        assert_debug::<ResolvedPlanningConfig>();
        assert_debug::<ResolvedPlanningKey>();

        let policy = ExecutionPolicy::new(Determinism::Fast, Some(12), 34);
        assert_eq!(policy.determinism(), Determinism::Fast);
        assert_eq!(policy.hard_workspace_limit_bytes(), Some(12));
        assert_eq!(policy.planning_seed(), 34);

        let options = PrepareOptions::new();
        let config = ResolvedPlanningConfig::resolve(&policy, &options, hardware("tenferro.cpu"));
        assert_eq!(config.determinism(), Determinism::Fast);
        assert_eq!(config.hard_workspace_limit_bytes(), Some(12));
        assert_eq!(config.planning_seed(), 34);
        assert_eq!(config.hardware_class().as_str(), "tenferro.cpu");
    }
}

mod signature {
    use std::error::Error as _;
    use std::fmt::Debug;
    use std::mem::align_of;
    use std::sync::Arc;

    use tenferro_runtime::runtime::{
        InputSignature, InputSignatureEntry, InputSignatureError, LayoutClass, PrepareError,
    };
    use tenferro_tensor::{
        BackendBuffer, Buffer, BufferHandle, DType, MemoryKind, Placement, ShapeVec, StrideVec,
        Tensor, TensorRead, TensorScalar, TypedTensor, TypedTensorView,
    };

    fn assert_debug<T: Debug>() {}

    fn shape(values: &[usize]) -> ShapeVec {
        values.iter().copied().collect()
    }

    fn strides(values: &[isize]) -> StrideVec {
        values.iter().copied().collect()
    }

    fn layout(value: &str) -> LayoutClass {
        LayoutClass::new(value).unwrap()
    }

    fn entry(
        dtype: DType,
        shape_values: &[usize],
        placement: Placement,
        layout_class: &str,
        stride_values: &[isize],
        alignment_log2: Option<u8>,
    ) -> InputSignatureEntry {
        InputSignatureEntry::new(
            dtype,
            shape(shape_values),
            placement,
            layout(layout_class),
            strides(stride_values),
            alignment_log2,
        )
        .unwrap()
    }

    fn host_alignment_log2<T>(logical_pointer: *const T) -> u8 {
        let address_class = (logical_pointer as usize).trailing_zeros();
        let type_class = align_of::<T>().trailing_zeros();
        address_class.min(type_class).min(usize::BITS - 1) as u8
    }

    #[test]
    fn signature_entry_rejects_shape_stride_rank_mismatch() {
        let error = InputSignatureEntry::new(
            DType::F64,
            shape(&[2, 3]),
            Placement::default(),
            layout("tenferro.layout.strided.v1"),
            strides(&[1]),
            None,
        )
        .unwrap_err();

        assert!(matches!(
            error,
            InputSignatureError::ShapeStrideRankMismatch {
                rank: 2,
                stride_count: 1,
            }
        ));
    }

    #[test]
    fn signature_entry_rejects_out_of_lattice_alignment() {
        let alignment_log2 = u8::try_from(usize::BITS).unwrap();
        let error = InputSignatureEntry::new(
            DType::F32,
            shape(&[1]),
            Placement::default(),
            layout("tenferro.layout.compact-col-major.v1"),
            strides(&[1]),
            Some(alignment_log2),
        )
        .unwrap_err();

        assert!(matches!(
            error,
            InputSignatureError::InvalidAlignmentClass {
                alignment_log2: actual,
            } if actual == alignment_log2
        ));
    }

    #[test]
    fn signature_aggregate_is_infallible_and_distinguishes_unknown_from_one_byte() {
        let unknown = entry(
            DType::Bool,
            &[1],
            Placement::default(),
            "tenferro.layout.compact-col-major.v1",
            &[1],
            None,
        );
        let one_byte = entry(
            DType::Bool,
            &[1],
            Placement::default(),
            "tenferro.layout.compact-col-major.v1",
            &[1],
            Some(0),
        );

        let signature = InputSignature::new(vec![unknown, one_byte]);

        assert_eq!(signature.entries()[0].alignment_log2(), None);
        assert_eq!(signature.entries()[1].alignment_log2(), Some(0));
        assert_ne!(signature.entries()[0], signature.entries()[1]);
    }

    #[test]
    fn signature_from_reads_copies_only_dtype_shape_strides_placement_layout_and_alignment() {
        let mut data = vec![10.0_f64, 20.0, 30.0, 40.0, 50.0, 60.0];
        let expected_placement = Placement::default();
        let expected_alignment = host_alignment_log2(data[1..].as_ptr());

        let signature = {
            let view = TypedTensorView::from_slice([2, 2], [1, 3], 1, &data).unwrap();
            let read = TensorRead::from_view(f64::tensor_view(view));
            InputSignature::from_reads(&[read]).unwrap()
        };
        data.fill(-1.0);

        let actual = &signature.entries()[0];
        assert_eq!(actual.dtype(), DType::F64);
        assert_eq!(actual.shape(), &[2, 2]);
        assert_eq!(actual.strides(), &[1, 3]);
        assert_eq!(actual.placement(), &expected_placement);
        assert_eq!(actual.layout_class().as_str(), "tenferro.layout.strided.v1");
        assert_eq!(actual.alignment_log2(), Some(expected_alignment));
    }

    #[test]
    fn signature_nonzero_host_offset_uses_the_actual_logical_pointer_alignment() {
        let data = [0.0_f64; 4];
        let logical_pointer = data[1..].as_ptr();
        let view = TypedTensorView::from_slice([2], [1], 1, &data).unwrap();
        let read = TensorRead::from_view(f64::tensor_view(view));

        let signature = InputSignature::from_reads(&[read]).unwrap();

        assert_eq!(
            signature.entries()[0].alignment_log2(),
            Some(host_alignment_log2(logical_pointer))
        );
    }

    #[test]
    fn signature_empty_host_read_uses_type_alignment_without_reading_an_element() {
        let data: [i64; 0] = [];
        let view = TypedTensorView::from_col_major(&[0], &data).unwrap();
        let read = TensorRead::from_view(i64::tensor_view(view));

        let signature = InputSignature::from_reads(&[read]).unwrap();

        assert_eq!(
            signature.entries()[0].alignment_log2(),
            Some(align_of::<i64>().trailing_zeros() as u8)
        );
    }

    #[test]
    fn signature_backend_read_records_unknown_alignment_and_retains_no_buffer() {
        let allocation = Arc::new(BufferHandle::<f64>::new_with_len(7, 2));
        let weak = Arc::downgrade(&allocation);
        let erased: Arc<dyn BackendBuffer<f64>> = allocation.clone();
        drop(allocation);

        let signature = {
            let tensor = TypedTensor::from_buffer_col_major(
                vec![2],
                Buffer::Backend(erased),
                Placement {
                    memory_kind: MemoryKind::Device,
                    device: None,
                    cpu_affinity: None,
                },
            )
            .unwrap();
            let read = f64::tensor_read(&tensor);
            InputSignature::from_reads(&[read]).unwrap()
        };

        assert_eq!(signature.entries()[0].alignment_log2(), None);
        assert_eq!(
            signature.entries()[0].placement().memory_kind,
            MemoryKind::Device
        );
        assert!(
            weak.upgrade().is_none(),
            "the value-free signature must not retain the backend buffer"
        );
    }

    #[test]
    fn signature_compact_and_strided_reads_use_exact_layout_classes() {
        let data = [1_i32, 2, 3, 4];
        let compact = TypedTensorView::from_col_major(&[2, 2], &data).unwrap();
        let strided = TypedTensorView::from_slice([2], [2], 0, &data).unwrap();
        let reads = [
            TensorRead::from_view(i32::tensor_view(compact)),
            TensorRead::from_view(i32::tensor_view(strided)),
        ];

        let signature = InputSignature::from_reads(&reads).unwrap();

        assert_eq!(
            signature.entries()[0].layout_class().as_str(),
            "tenferro.layout.compact-col-major.v1"
        );
        assert_eq!(
            signature.entries()[1].layout_class().as_str(),
            "tenferro.layout.strided.v1"
        );
    }

    #[test]
    fn signature_types_are_debug_and_all_public_accessors_are_callable() {
        assert_debug::<InputSignatureError>();
        assert_debug::<PrepareError>();
        assert_debug::<InputSignatureEntry>();
        assert_debug::<InputSignature>();

        let placement = Placement::default();
        let entry = entry(
            DType::I64,
            &[2, 3],
            placement.clone(),
            "tenferro.layout.strided.v1",
            &[1, 4],
            Some(2),
        );
        assert_eq!(entry.dtype(), DType::I64);
        assert_eq!(entry.shape(), &[2, 3]);
        assert_eq!(entry.placement(), &placement);
        assert_eq!(entry.layout_class().as_str(), "tenferro.layout.strided.v1");
        assert_eq!(entry.strides(), &[1, 4]);
        assert_eq!(entry.alignment_log2(), Some(2));

        let signature = InputSignature::new(vec![entry]);
        assert_eq!(signature.entries().len(), 1);
        let _ = format!("{signature:?}");
    }

    #[test]
    fn signature_error_sources_remain_typed() {
        let tensor_error = Tensor::from_vec_col_major(vec![2], vec![1.0_f64]).unwrap_err();
        let error = InputSignatureError::TensorMetadata {
            input: 3,
            source: tensor_error,
        };
        let typed_source = error.source().expect("tensor source must be retained");
        assert!(typed_source
            .downcast_ref::<tenferro_tensor::Error>()
            .is_some());

        let prepare = PrepareError::InputSignature { source: error };
        let signature_source = prepare.source().expect("signature source must be retained");
        assert!(signature_source
            .downcast_ref::<InputSignatureError>()
            .is_some());
    }
}

mod specialization {
    use std::error::Error as _;
    use std::fmt::Debug;

    use tenferro_runtime::runtime::{
        InputSignature, InputSignatureEntry, InputSpecializationProjection,
        InputSpecializationRequirements, InputSpecializationRequirementsBuilder,
        InputSpecializationRequirementsError, LayoutClass, LayoutProjection, LayoutSpecialization,
        PlacementProjection, PlacementSpecialization, PrepareError, RankRequirement,
        SpecializationError, SpecializationProjection, SpecializationRequirements,
    };
    use tenferro_tensor::{
        DType, DeviceId, DeviceKind, GpuBackendKind, MemoryKind, Placement, ShapeVec, StrideVec,
    };

    fn assert_debug<T: Debug>() {}

    fn shape(values: &[usize]) -> ShapeVec {
        values.iter().copied().collect()
    }

    fn strides(values: &[isize]) -> StrideVec {
        values.iter().copied().collect()
    }

    fn layout(value: &str) -> LayoutClass {
        LayoutClass::new(value).unwrap()
    }

    fn signature_entry(
        dtype: DType,
        shape_values: &[usize],
        placement: Placement,
        layout_class: &str,
        stride_values: &[isize],
        alignment_log2: Option<u8>,
    ) -> InputSignatureEntry {
        InputSignatureEntry::new(
            dtype,
            shape(shape_values),
            placement,
            layout(layout_class),
            strides(stride_values),
            alignment_log2,
        )
        .unwrap()
    }

    fn input_requirements(
        configure: impl FnOnce(&mut InputSpecializationRequirementsBuilder),
    ) -> InputSpecializationRequirements {
        let mut builder = InputSpecializationRequirements::builder();
        configure(&mut builder);
        builder.build().unwrap()
    }

    fn one_input_signature(
        shape_values: &[usize],
        placement: Placement,
        alignment_log2: Option<u8>,
    ) -> InputSignature {
        InputSignature::new(vec![signature_entry(
            DType::F64,
            shape_values,
            placement,
            "tenferro.layout.strided.v1",
            &(0..shape_values.len())
                .map(|axis| isize::try_from(axis + 1).unwrap())
                .collect::<Vec<_>>(),
            alignment_log2,
        )])
    }

    fn project_alignment(
        required: Option<u8>,
        actual: Option<u8>,
    ) -> Result<Option<u8>, PrepareError> {
        let requirements = input_requirements(|builder| {
            builder.alignment_log2(required);
        });
        let aggregate = SpecializationRequirements::new(vec![requirements]);
        let signature = one_input_signature(&[2], Placement::default(), actual);
        aggregate
            .project(&signature)
            .map(|projection| projection.inputs()[0].alignment_log2())
    }

    fn storage_projection(memory_kind: MemoryKind) -> PlacementProjection {
        let requirements = input_requirements(|builder| {
            builder.placement(PlacementSpecialization::StorageClass);
        });
        let aggregate = SpecializationRequirements::new(vec![requirements]);
        let signature = one_input_signature(
            &[1],
            Placement {
                memory_kind,
                device: None,
                cpu_affinity: None,
            },
            None,
        );
        aggregate.project(&signature).unwrap().inputs()[0]
            .placement()
            .unwrap()
            .clone()
    }

    #[test]
    fn specialization_builder_reports_first_duplicate_axis_positions() {
        let mut builder = InputSpecializationRequirements::builder();
        builder.rank(true).concrete_dimensions(vec![3, 1, 3, 1]);

        let error = builder.build().unwrap_err();

        assert_eq!(
            error,
            InputSpecializationRequirementsError::DuplicateAxis {
                axis: 3,
                first_index: 0,
                duplicate_index: 2,
            }
        );
    }

    #[test]
    fn specialization_builder_reports_concrete_axis_rank_requirement() {
        let mut builder = InputSpecializationRequirements::builder();
        builder.rank(false).concrete_dimensions(vec![2]);

        let error = builder.build().unwrap_err();

        assert_eq!(
            error,
            InputSpecializationRequirementsError::RankRequired {
                reason: RankRequirement::ConcreteAxis { axis: 2 },
            }
        );
        assert_eq!(
            RankRequirement::ConcreteAxis { axis: 2 }.to_string(),
            "concrete axis 2"
        );
    }

    #[test]
    fn specialization_builder_reports_exact_strides_rank_requirement() {
        let mut builder = InputSpecializationRequirements::builder();
        builder
            .rank(false)
            .layout(LayoutSpecialization::ExactStrides);

        let error = builder.build().unwrap_err();

        assert_eq!(
            error,
            InputSpecializationRequirementsError::RankRequired {
                reason: RankRequirement::ExactStrides,
            }
        );
        assert_eq!(RankRequirement::ExactStrides.to_string(), "exact strides");
    }

    #[test]
    fn specialization_builder_reports_out_of_lattice_alignment() {
        let alignment_log2 = u8::try_from(usize::BITS).unwrap();
        let mut builder = InputSpecializationRequirements::builder();
        builder.alignment_log2(Some(alignment_log2));

        let error = builder.build().unwrap_err();

        assert_eq!(
            error,
            InputSpecializationRequirementsError::InvalidAlignmentClass { alignment_log2 }
        );
    }

    #[test]
    fn specialization_builder_validation_order_is_deterministic() {
        let invalid_alignment = u8::try_from(usize::BITS).unwrap();

        let mut duplicate_first = InputSpecializationRequirements::builder();
        duplicate_first
            .rank(false)
            .concrete_dimensions(vec![4, 4])
            .layout(LayoutSpecialization::ExactStrides)
            .alignment_log2(Some(invalid_alignment));
        assert!(matches!(
            duplicate_first.build(),
            Err(InputSpecializationRequirementsError::DuplicateAxis { .. })
        ));

        let mut concrete_rank_second = InputSpecializationRequirements::builder();
        concrete_rank_second
            .rank(false)
            .concrete_dimensions(vec![4])
            .layout(LayoutSpecialization::ExactStrides)
            .alignment_log2(Some(invalid_alignment));
        assert_eq!(
            concrete_rank_second.build().unwrap_err(),
            InputSpecializationRequirementsError::RankRequired {
                reason: RankRequirement::ConcreteAxis { axis: 4 },
            }
        );

        let mut exact_strides_third = InputSpecializationRequirements::builder();
        exact_strides_third
            .rank(false)
            .layout(LayoutSpecialization::ExactStrides)
            .alignment_log2(Some(invalid_alignment));
        assert_eq!(
            exact_strides_third.build().unwrap_err(),
            InputSpecializationRequirementsError::RankRequired {
                reason: RankRequirement::ExactStrides,
            }
        );

        let mut alignment_last = InputSpecializationRequirements::builder();
        alignment_last
            .rank(true)
            .layout(LayoutSpecialization::ExactStrides)
            .alignment_log2(Some(invalid_alignment));
        assert_eq!(
            alignment_last.build().unwrap_err(),
            InputSpecializationRequirementsError::InvalidAlignmentClass {
                alignment_log2: invalid_alignment,
            }
        );
    }

    #[test]
    fn specialization_aggregate_requirements_construction_is_infallible() {
        let first = input_requirements(|builder| {
            builder.dtype(true);
        });
        let second = input_requirements(|builder| {
            builder.rank(true).concrete_dimensions(vec![0]);
        });

        let aggregate = SpecializationRequirements::new(vec![first.clone(), second.clone()]);

        assert_eq!(aggregate.inputs(), &[first, second]);
        assert_eq!(SpecializationRequirements::polymorphic(2).inputs().len(), 2);
    }

    #[test]
    fn specialization_projection_reports_wrong_input_count() {
        let requirements = SpecializationRequirements::polymorphic(2);
        let signature = one_input_signature(&[1], Placement::default(), None);

        let error = requirements.project(&signature).unwrap_err();

        assert!(matches!(
            error,
            PrepareError::Specialization {
                source: SpecializationError::WrongInputCount {
                    expected: 2,
                    actual: 1,
                },
            }
        ));
    }

    #[test]
    fn specialization_projection_reports_axis_outside_actual_rank() {
        let input = input_requirements(|builder| {
            builder.rank(true).concrete_dimensions(vec![2]);
        });
        let requirements = SpecializationRequirements::new(vec![input]);
        let signature = one_input_signature(&[2, 3], Placement::default(), None);

        let error = requirements.project(&signature).unwrap_err();

        assert!(matches!(
            error,
            PrepareError::Specialization {
                source: SpecializationError::AxisOutOfRange {
                    input: 0,
                    axis: 2,
                    rank: 2,
                },
            }
        ));
    }

    #[test]
    fn specialization_projection_reports_unavailable_alignment() {
        let error = project_alignment(Some(2), None).unwrap_err();

        assert!(matches!(
            error,
            PrepareError::Specialization {
                source: SpecializationError::AlignmentUnavailable {
                    input: 0,
                    required_alignment_log2: 2,
                },
            }
        ));
    }

    #[test]
    fn specialization_projection_covers_all_alignment_rows_and_caps_known_alignment() {
        assert_eq!(project_alignment(None, None).unwrap(), None);
        assert_eq!(project_alignment(None, Some(5)).unwrap(), None);
        assert!(matches!(
            project_alignment(Some(3), None),
            Err(PrepareError::Specialization {
                source: SpecializationError::AlignmentUnavailable {
                    input: 0,
                    required_alignment_log2: 3,
                },
            })
        ));
        assert_eq!(project_alignment(Some(5), Some(2)).unwrap(), Some(2));
        assert_eq!(project_alignment(Some(2), Some(5)).unwrap(), Some(2));
    }

    #[test]
    fn specialization_projection_selects_exact_dtype_rank_dimensions_placement_and_layout() {
        let placement = Placement {
            memory_kind: MemoryKind::Device,
            device: Some(DeviceId {
                kind: DeviceKind::Gpu(GpuBackendKind::Cuda),
                ordinal: 3,
            }),
            cpu_affinity: None,
        };
        let signature = InputSignature::new(vec![signature_entry(
            DType::F64,
            &[2, 3],
            placement.clone(),
            "tenferro.layout.strided.v1",
            &[1, 4],
            Some(5),
        )]);
        let input = input_requirements(|builder| {
            builder
                .dtype(true)
                .rank(true)
                .concrete_dimensions(vec![1, 0])
                .placement(PlacementSpecialization::Device)
                .layout(LayoutSpecialization::ExactStrides)
                .alignment_log2(Some(3));
        });
        let requirements = SpecializationRequirements::new(vec![input]);

        let projection = requirements.project(&signature).unwrap();
        let actual = &projection.inputs()[0];

        assert_eq!(actual.dtype(), Some(DType::F64));
        assert_eq!(actual.rank(), Some(2));
        assert_eq!(actual.concrete_dimensions(), &[(1, 3), (0, 2)]);
        assert_eq!(
            actual.placement(),
            Some(&PlacementProjection::Device(placement))
        );
        assert_eq!(
            actual.layout(),
            Some(&LayoutProjection::ExactStrides(strides(&[1, 4])))
        );
        assert_eq!(actual.alignment_log2(), Some(3));
        assert_eq!(projection.requirements(), &requirements);
    }

    #[test]
    fn specialization_storage_projection_maps_builtin_memory_kinds_exactly() {
        for (memory_kind, expected) in [
            (MemoryKind::Device, "tenferro.storage.device.v1"),
            (MemoryKind::PinnedHost, "tenferro.storage.pinned-host.v1"),
            (
                MemoryKind::UnpinnedHost,
                "tenferro.storage.unpinned-host.v1",
            ),
            (MemoryKind::Managed, "tenferro.storage.managed.v1"),
        ] {
            let projection = storage_projection(memory_kind);
            assert!(matches!(
                projection,
                PlacementProjection::StorageClass(ref storage)
                    if storage.as_str() == expected
            ));
        }
    }

    #[test]
    fn specialization_storage_projection_uses_other_empty_sentinel() {
        let projection = storage_projection(MemoryKind::Other(String::new()));

        assert!(matches!(
            projection,
            PlacementProjection::StorageClass(ref storage)
                if storage.as_str() == "tenferro.storage.other-empty.v1"
        ));
    }

    #[test]
    fn specialization_non_storage_placement_modes_do_not_derive_storage_class() {
        let placement = Placement {
            memory_kind: MemoryKind::Other("payload-that-must-not-be-classified".into()),
            device: Some(DeviceId {
                kind: DeviceKind::Other("accelerator".into()),
                ordinal: 9,
            }),
            cpu_affinity: None,
        };
        let signature = one_input_signature(&[1], placement.clone(), None);

        let none = input_requirements(|builder| {
            builder.placement(PlacementSpecialization::None);
        });
        let none_projection = SpecializationRequirements::new(vec![none])
            .project(&signature)
            .unwrap();
        assert_eq!(none_projection.inputs()[0].placement(), None);

        let device = input_requirements(|builder| {
            builder.placement(PlacementSpecialization::Device);
        });
        let device_projection = SpecializationRequirements::new(vec![device])
            .project(&signature)
            .unwrap();
        assert_eq!(
            device_projection.inputs()[0].placement(),
            Some(&PlacementProjection::Device(placement))
        );
    }

    #[test]
    fn specialization_strict_widening_rejects_equal_lowered_incomparable_and_different_arity_values(
    ) {
        let polymorphic = SpecializationRequirements::new(vec![input_requirements(|_| {})]);
        let dtype = SpecializationRequirements::new(vec![input_requirements(|builder| {
            builder.dtype(true);
        })]);
        let rank = SpecializationRequirements::new(vec![input_requirements(|builder| {
            builder.rank(true);
        })]);

        assert!(polymorphic.strictly_widens(&dtype));
        assert!(!dtype.strictly_widens(&dtype));
        assert!(!dtype.strictly_widens(&polymorphic));
        assert!(!dtype.strictly_widens(&rank));
        assert!(!rank.strictly_widens(&dtype));
        assert!(!SpecializationRequirements::polymorphic(2).strictly_widens(&dtype));
        assert!(!dtype.strictly_widens(&SpecializationRequirements::polymorphic(2)));
    }

    #[test]
    fn specialization_types_are_debug_and_all_public_accessors_are_callable() {
        assert_debug::<RankRequirement>();
        assert_debug::<InputSpecializationRequirementsError>();
        assert_debug::<SpecializationError>();
        assert_debug::<PlacementSpecialization>();
        assert_debug::<LayoutSpecialization>();
        assert_debug::<InputSpecializationRequirements>();
        assert_debug::<InputSpecializationRequirementsBuilder>();
        assert_debug::<SpecializationRequirements>();
        assert_debug::<SpecializationProjection>();
        assert_debug::<InputSpecializationProjection>();
        assert_debug::<PlacementProjection>();
        assert_debug::<LayoutProjection>();

        let mut builder = InputSpecializationRequirementsBuilder::new();
        builder
            .dtype(true)
            .rank(true)
            .concrete_dimensions(vec![0])
            .placement(PlacementSpecialization::StorageClass)
            .layout(LayoutSpecialization::Class)
            .alignment_log2(Some(1));
        let _ = format!("{builder:?}");
        let input = builder.build().unwrap();
        assert!(input.specializes_dtype());
        assert!(input.specializes_rank());
        assert_eq!(input.concrete_dimensions(), &[0]);
        assert_eq!(input.placement(), PlacementSpecialization::StorageClass);
        assert_eq!(input.layout(), LayoutSpecialization::Class);
        assert_eq!(input.alignment_log2(), Some(1));

        let requirements = SpecializationRequirements::new(vec![input]);
        assert_eq!(requirements.inputs().len(), 1);
        let signature = one_input_signature(&[2], Placement::default(), Some(3));
        let projection = requirements.project(&signature).unwrap();
        assert_eq!(projection.requirements(), &requirements);
        assert_eq!(projection.inputs().len(), 1);
        assert_eq!(projection.inputs()[0].dtype(), Some(DType::F64));
        assert_eq!(projection.inputs()[0].rank(), Some(1));
        assert_eq!(projection.inputs()[0].concrete_dimensions(), &[(0, 2)]);
        assert!(matches!(
            projection.inputs()[0].placement(),
            Some(PlacementProjection::StorageClass(_))
        ));
        assert!(matches!(
            projection.inputs()[0].layout(),
            Some(LayoutProjection::Class(_))
        ));
        assert_eq!(projection.inputs()[0].alignment_log2(), Some(1));

        let error = requirements
            .project(&InputSignature::new(Vec::new()))
            .unwrap_err();
        assert!(error
            .source()
            .and_then(|source| source.downcast_ref::<SpecializationError>())
            .is_some());
    }
}
