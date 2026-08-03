use std::fmt::Debug;
use std::sync::Arc;

use tenferro_runtime::{
    CoreCapabilityBundle, EngineId, EngineRegistrationMetadata, ExecutableEngineRegistrationConfig,
    ExecutionContextIdentity, HardwareClassId, ImmediateEventDomainDriver, InputIngressContract,
    InputPlacementContract, InputSignatureContract, PreparationOnlyEngineRegistrationConfig,
    ProviderDeviceIdentity, ProviderId, ResidentOutputContract, RuntimeInputContract, StorageClass,
};

struct NonDebugBackend;

fn debug_contract_metadata() -> EngineRegistrationMetadata {
    let engine_id = EngineId::new("debug.contract.engine").expect("valid test engine id");
    let provider = ProviderDeviceIdentity::new(
        ProviderId::new("debug.contract.provider").expect("valid test provider id"),
        "device:0",
    )
    .expect("valid test provider target");
    let hardware = HardwareClassId::new("debug.contract.hardware").expect("valid test hardware");
    let storage = StorageClass::new("debug.contract.storage").expect("valid test storage");
    EngineRegistrationMetadata::new(
        engine_id,
        provider,
        hardware,
        Arc::from([storage.clone()]),
        storage,
        CoreCapabilityBundle::default(),
    )
}

fn debug_contract_ingress() -> InputIngressContract {
    InputIngressContract::new(
        InputPlacementContract::new(|_, _| true),
        InputSignatureContract::new(|_, _, _, _| true),
        RuntimeInputContract::new(|_, _| true),
        ResidentOutputContract::new(|_, _| true),
    )
}

#[test]
fn engine_registration_descriptors_have_public_debug_bounds() {
    fn assert_debug<T: Debug>() {}

    assert_debug::<EngineRegistrationMetadata>();
    assert_debug::<PreparationOnlyEngineRegistrationConfig>();
    assert_debug::<ExecutableEngineRegistrationConfig<NonDebugBackend>>();
}

#[test]
fn engine_registration_descriptor_debug_output_is_bounded_and_useful() {
    let metadata = debug_contract_metadata();
    let metadata_debug = format!("{metadata:?}");
    assert!(metadata_debug.contains("EngineRegistrationMetadata"));
    assert!(metadata_debug.contains("debug.contract.engine"));
    assert!(metadata_debug.contains("storage_classes"));
    assert!(metadata_debug.contains("capabilities"));

    let preparation = PreparationOnlyEngineRegistrationConfig::new(
        debug_contract_metadata(),
        ExecutionContextIdentity::of::<()>(),
    );
    let preparation_debug = format!("{preparation:?}");
    assert!(preparation_debug.contains("PreparationOnlyEngineRegistrationConfig"));
    assert!(preparation_debug.contains("metadata"));
    assert!(preparation_debug.contains("context_identity"));

    let executable = ExecutableEngineRegistrationConfig::new(
        metadata,
        NonDebugBackend,
        Arc::new(ImmediateEventDomainDriver::new()),
        debug_contract_ingress(),
        None,
    );
    let executable_debug = format!("{executable:?}");
    assert!(executable_debug.contains("ExecutableEngineRegistrationConfig"));
    assert!(executable_debug.contains("metadata"));
    assert!(executable_debug.contains("backend_type"));
    assert!(executable_debug.contains(std::any::type_name::<NonDebugBackend>()));
    assert!(executable_debug.contains("event_domain_driver_present: true"));
    assert!(executable_debug.contains("ingress_present: true"));
    assert!(executable_debug.contains("cache_owner_present: false"));
    assert!(!executable_debug.contains("backend:"));
    assert!(!executable_debug.contains("cache_owner:"));
}
