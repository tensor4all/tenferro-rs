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
