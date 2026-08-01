use std::sync::Arc;

use super::super::engine_registration::EngineRegistrationState;
use super::super::{
    CoreCapabilityBundle, EngineId, EngineRegistration, ExecutableEngineContract,
    ExecutionContextIdentity, HardwareClassId, ImmediateEventDomainDriver, InputIngressContract,
    InputPlacementContract, InputSignatureContract, ProviderDeviceIdentity,
    ProviderExecutableBinding, ProviderId, ProviderPreparationBinding, ResidentOutputContract,
    RuntimeConfigError, RuntimeInputContract, StorageClass,
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
    let storage =
        StorageClass::new("tenferro.test.remediation.host").map_err(RuntimeConfigError::from)?;
    Ok((
        EngineId::new(engine_name).map_err(RuntimeConfigError::from)?,
        ProviderDeviceIdentity::new(
            ProviderId::new("tenferro.test.remediation.provider")
                .map_err(RuntimeConfigError::from)?,
            "device:0",
        )
        .map_err(RuntimeConfigError::from)?,
        ExecutionContextIdentity::of::<tenferro_cpu::CpuBackend>(),
        HardwareClassId::new("tenferro.test.remediation.host").map_err(RuntimeConfigError::from)?,
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
    let (
        engine_id,
        provider_device_identity,
        context_identity,
        hardware_class,
        storage_classes,
        default_storage_class,
    ) = registration_metadata("tenferro.test.remediation.preparation-only").unwrap();
    let preparation_only = EngineRegistration::preparation_only(
        ProviderPreparationBinding::new(
            engine_id,
            provider_device_identity,
            context_identity,
            hardware_class,
            storage_classes,
            default_storage_class,
            CoreCapabilityBundle::default(),
        )
        .unwrap(),
    );
    assert!(matches!(
        preparation_only.execution_state(),
        EngineRegistrationState::PreparationOnly { .. }
    ));

    let (
        engine_id,
        provider_device_identity,
        _context_identity,
        hardware_class,
        storage_classes,
        default_storage_class,
    ) = registration_metadata("tenferro.test.remediation.executable").unwrap();
    let executable = EngineRegistration::executable(
        ProviderExecutableBinding::new(
            engine_id,
            hardware_class,
            storage_classes,
            default_storage_class,
            ExecutableEngineContract::new(
                provider_device_identity,
                CoreCapabilityBundle::default(),
                tenferro_cpu::CpuBackend::new(),
                Arc::new(ImmediateEventDomainDriver::new()),
                ingress_contract(),
                None,
            ),
        )
        .unwrap(),
    );
    assert!(matches!(
        executable.execution_state(),
        EngineRegistrationState::Executable(_)
    ));
}

#[test]
fn frozen_executable_selection_returns_one_complete_witness(
) -> Result<(), Box<dyn std::error::Error>> {
    let (
        engine_id,
        provider_device_identity,
        context_identity,
        hardware_class,
        storage_classes,
        default_storage_class,
    ) = registration_metadata("tenferro.test.remediation.frozen-witness")?;
    let registration = EngineRegistration::executable(ProviderExecutableBinding::new(
        engine_id.clone(),
        hardware_class,
        storage_classes,
        default_storage_class,
        ExecutableEngineContract::new(
            provider_device_identity.clone(),
            CoreCapabilityBundle::default(),
            tenferro_cpu::CpuBackend::new(),
            Arc::new(ImmediateEventDomainDriver::new()),
            ingress_contract(),
            None,
        ),
    )?);
    let mut builder = super::super::Runtime::builder();
    builder.register_engine(registration)?;
    let runtime = builder.build()?;
    let snapshot = runtime.snapshot()?;
    let engine = snapshot
        .engine(&engine_id)
        .expect("frozen executable engine");
    let witness = engine
        .executable_witness()
        .expect("executable registration must freeze as one witness");

    assert_eq!(
        witness.provider_device_identity(),
        &provider_device_identity
    );
    assert_eq!(witness.context_identity(), context_identity);
    assert_eq!(
        engine.provider_device_identity(),
        witness.provider_device_identity()
    );
    assert_eq!(engine.context_identity(), witness.context_identity());
    assert!(witness.has_executor());
    assert!(witness.has_event_domain_driver());
    Ok(())
}

#[test]
fn production_reconfiguration_assigns_identity_before_freeze(
) -> Result<(), Box<dyn std::error::Error>> {
    let (
        engine_id,
        provider_device_identity,
        context_identity,
        hardware_class,
        storage_classes,
        default_storage_class,
    ) = registration_metadata("tenferro.test.remediation.identity").unwrap();
    let registration = EngineRegistration::preparation_only(ProviderPreparationBinding::new(
        engine_id.clone(),
        provider_device_identity,
        context_identity,
        hardware_class,
        storage_classes,
        default_storage_class,
        CoreCapabilityBundle::default(),
    )?);
    let mut builder = super::super::Runtime::builder();
    builder.register_engine(registration)?;
    let runtime = builder.build()?;
    let before = runtime
        .snapshot()?
        .engine(&engine_id)
        .expect("engine in initial snapshot")
        .registration_identity();

    let (
        engine_id,
        provider_device_identity,
        context_identity,
        hardware_class,
        storage_classes,
        default_storage_class,
    ) = registration_metadata("tenferro.test.remediation.identity").unwrap();
    let replacement = EngineRegistration::preparation_only(ProviderPreparationBinding::new(
        engine_id.clone(),
        provider_device_identity,
        context_identity,
        hardware_class,
        storage_classes,
        default_storage_class,
        CoreCapabilityBundle::default(),
    )?);
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
