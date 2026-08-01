use std::sync::Arc;

use super::super::{
    CoreCapabilityBundle, EngineId, EngineRegistration, EngineRegistrationState,
    ExecutableEngineContract, ExecutionContextIdentity, HardwareClassId,
    ImmediateEventDomainDriver, InputIngressContract, InputPlacementContract,
    InputSignatureContract, ProviderDeviceIdentity, ProviderId, ResidentOutputContract,
    RuntimeInputContract, RuntimeConfigError, StorageClass,
};

fn registration_metadata(
    engine_name: &str,
) -> Result<
    (
        EngineId,
        ProviderDeviceIdentity,
        ExecutionContextIdentity,
        HardwareClassId,
        Arc<[StorageClass]>,
        StorageClass,
    ),
    RuntimeConfigError,
> {
    let storage = StorageClass::new("tenferro.test.remediation.host")
        .map_err(RuntimeConfigError::from)?;
    Ok((
        EngineId::new(engine_name).map_err(RuntimeConfigError::from)?,
        ProviderDeviceIdentity::new(
            ProviderId::new("tenferro.test.remediation.provider")
                .map_err(RuntimeConfigError::from)?,
            "device:0",
        )
        .map_err(RuntimeConfigError::from)?,
        ExecutionContextIdentity::of::<tenferro_cpu::CpuBackend>(),
        HardwareClassId::new("tenferro.test.remediation.host")
            .map_err(RuntimeConfigError::from)?,
        Arc::from(vec![storage.clone()]),
        storage,
    ))
}

fn ingress_contract() -> InputIngressContract {
    InputIngressContract::new(
        InputPlacementContract::new(|_, _| true),
        InputSignatureContract::new(|_, _, _, _| true),
        RuntimeInputContract::new(|_, _| true),
        ResidentOutputContract::new(|_, _| false),
    )
}

#[test]
fn registration_state_is_a_single_preparation_or_executable_witness() {
    let (engine_id, provider_device_identity, context_identity, hardware_class, storage_classes, default_storage_class) =
        registration_metadata("tenferro.test.remediation.preparation-only").unwrap();
    let preparation_only = EngineRegistration::preparation_only(
        engine_id,
        provider_device_identity,
        context_identity,
        hardware_class,
        storage_classes,
        default_storage_class,
        CoreCapabilityBundle::default(),
    )
    .unwrap();
    assert!(matches!(
        preparation_only.execution_state(),
        EngineRegistrationState::PreparationOnly { .. }
    ));

    let (engine_id, provider_device_identity, context_identity, hardware_class, storage_classes, default_storage_class) =
        registration_metadata("tenferro.test.remediation.executable").unwrap();
    let executable = EngineRegistration::executable(
        engine_id,
        provider_device_identity,
        context_identity,
        hardware_class,
        storage_classes,
        default_storage_class,
        ExecutableEngineContract::new(
            CoreCapabilityBundle::default(),
            tenferro_cpu::CpuBackend::new(),
            Arc::new(ImmediateEventDomainDriver::new()),
            ingress_contract(),
            None,
        ),
    )
    .unwrap();
    assert!(matches!(
        executable.execution_state(),
        EngineRegistrationState::Executable(_)
    ));
}

#[test]
fn production_reconfiguration_assigns_identity_before_freeze(
) -> Result<(), Box<dyn std::error::Error>> {
    let (engine_id, provider_device_identity, context_identity, hardware_class, storage_classes, default_storage_class) =
        registration_metadata("tenferro.test.remediation.identity").unwrap();
    let registration = EngineRegistration::preparation_only(
        engine_id.clone(),
        provider_device_identity,
        context_identity,
        hardware_class,
        storage_classes,
        default_storage_class,
        CoreCapabilityBundle::default(),
    )?;
    let mut builder = super::super::Runtime::builder();
    builder.register_engine(registration)?;
    let runtime = builder.build()?;
    let before = runtime
        .snapshot()?
        .engine(&engine_id)
        .expect("engine in initial snapshot")
        .registration_identity();

    let (engine_id, provider_device_identity, context_identity, hardware_class, storage_classes, default_storage_class) =
        registration_metadata("tenferro.test.remediation.identity").unwrap();
    let replacement = EngineRegistration::preparation_only(
        engine_id,
        provider_device_identity,
        context_identity,
        hardware_class,
        storage_classes,
        default_storage_class,
        CoreCapabilityBundle::default(),
    )?;
    runtime.reconfigure(|edit| {
        edit.replace_engine(replacement)?;
        Ok(())
    })?;
    let after = runtime
        .snapshot()?
        .engine(&engine_id)
        .expect("engine in replacement snapshot")
        .registration_identity();
    assert_ne!(before, after);
    Ok(())
}
