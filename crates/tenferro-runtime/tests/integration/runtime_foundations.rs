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
