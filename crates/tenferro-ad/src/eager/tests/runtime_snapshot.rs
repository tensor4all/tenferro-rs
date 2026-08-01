use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

use tenferro_cpu::{CpuBackend, CpuPlacement};
use tenferro_runtime::{
    CoreCapabilityBundle, DotGeneralPreparation, ElementwiseRuntime, EngineId, EngineRegistration,
    ExecutableEngineContract, ExecutionContextIdentity, HardwareClassId,
    ImmediateEventDomainDriver, IndexingRuntime, InputIngressContract, InputPlacementContract,
    InputSignatureContract, LayoutRuntime, ProviderDeviceIdentity, ProviderExecutableBinding,
    ProviderId, ProviderPreparationBinding, ReductionRuntime, ResidentOutputContract,
    RuntimeCacheOwner, RuntimeInputContract, StorageClass,
};
use tenferro_tensor::{BackendSession, ErrorKind};

use super::super::{CpuPlacementBoundEager, EagerRuntime, CPU_RUNTIME_SELECTION_REFRESHES};

const CPU_ENGINE_ID: &str = "tenferro-cpu.default.v1";
const CPU_HARDWARE_CLASS_ID: &str = "tenferro-cpu.host.v1";
const CPU_STORAGE_CLASS_ID: &str = "tenferro-cpu.host.v1";
static REFRESH_PROBE_TEST_LOCK: Mutex<()> = Mutex::new(());
type RuntimeBreak = Box<dyn FnOnce(&EagerRuntime, &mut CpuPlacementBoundEager)>;
type RuntimeConfigurationFailureRow = (&'static str, RuntimeBreak, ErrorKind, &'static str);

fn reset_refreshes() {
    CPU_RUNTIME_SELECTION_REFRESHES.store(0, Ordering::SeqCst);
}

fn refreshes() -> usize {
    CPU_RUNTIME_SELECTION_REFRESHES.load(Ordering::SeqCst)
}

fn cpu_engine_id() -> EngineId {
    EngineId::new(CPU_ENGINE_ID).unwrap()
}

fn cpu_hardware_class() -> HardwareClassId {
    HardwareClassId::new(CPU_HARDWARE_CLASS_ID).unwrap()
}

fn cpu_storage_class() -> StorageClass {
    StorageClass::new(CPU_STORAGE_CLASS_ID).unwrap()
}

fn full_cpu_capabilities(backend: &CpuBackend) -> CoreCapabilityBundle {
    let backend = Arc::new(backend.clone());
    let elementwise: Arc<dyn ElementwiseRuntime> = backend.clone();
    let reduction: Arc<dyn ReductionRuntime> = backend.clone();
    let indexing: Arc<dyn IndexingRuntime> = backend.clone();
    let dot_general: Arc<dyn DotGeneralPreparation> = backend.clone();
    let layout: Arc<dyn LayoutRuntime> = backend;
    let mut builder = CoreCapabilityBundle::builder();
    builder
        .elementwise(elementwise)
        .reduction(reduction)
        .indexing(indexing)
        .dot_general(dot_general)
        .layout(layout);
    builder.build()
}

fn cpu_registration_with(
    backend: &CpuBackend,
    context_identity: ExecutionContextIdentity,
    capabilities: CoreCapabilityBundle,
) -> EngineRegistration {
    let storage = cpu_storage_class();
    let execution_info = backend.execution_info();
    let provider_id = match execution_info.backend_kind() {
        tenferro_cpu::CpuBackendKind::Faer => "tenferro.cpu.faer",
        tenferro_cpu::CpuBackendKind::Blas => "tenferro.cpu.blas",
    };
    let provider_device_identity = ProviderDeviceIdentity::new(
        ProviderId::new(provider_id).unwrap(),
        format!("domain:{}", execution_info.domain_id().as_u64()),
    )
    .unwrap();
    if context_identity != ExecutionContextIdentity::of::<CpuBackend>() {
        return EngineRegistration::preparation_only(
            ProviderPreparationBinding::new(
                cpu_engine_id(),
                provider_device_identity,
                context_identity,
                cpu_hardware_class(),
                Arc::from([storage.clone()]),
                storage,
                capabilities,
            )
            .expect("preparation binding"),
        );
    }
    EngineRegistration::executable(
        ProviderExecutableBinding::new(
            cpu_engine_id(),
            cpu_hardware_class(),
            Arc::from([storage.clone()]),
            storage,
            ExecutableEngineContract::new(
                provider_device_identity,
                capabilities,
                backend.clone(),
                Arc::new(ImmediateEventDomainDriver::new()),
                InputIngressContract::new(
                    InputPlacementContract::new(|_, _| true),
                    InputSignatureContract::new(|_, _, _, _| true),
                    RuntimeInputContract::new(|_, _| true),
                    ResidentOutputContract::new(|_, _| true),
                ),
                Some(Arc::new(backend.clone()) as Arc<dyn RuntimeCacheOwner>),
            ),
        )
        .expect("executable binding"),
    )
}

fn assert_bound_matches_current_runtime(cpu: &CpuPlacementBoundEager) {
    assert_eq!(cpu.epoch, cpu.runtime.runtime.epoch().unwrap());
    assert_eq!(cpu.snapshot.epoch(), cpu.epoch);
    assert_eq!(cpu.runtime_id(), cpu.runtime.id());
    assert_eq!(cpu.engine_id.as_str(), CPU_ENGINE_ID);
    assert_eq!(
        cpu.registration_identity,
        cpu.snapshot
            .engine(&cpu.engine_id)
            .expect("CPU engine registered")
            .registration_identity()
    );
    assert!(cpu.capabilities.elementwise().is_some());
    assert!(cpu.capabilities.reduction().is_some());
    assert!(cpu.capabilities.indexing().is_some());
    assert!(cpu.capabilities.dot_general().is_some());
    assert!(cpu.capabilities.layout().is_some());
}

#[test]
fn placement_bound_view_reuses_cached_snapshot_until_runtime_epoch_changes() {
    let _guard = REFRESH_PROBE_TEST_LOCK.lock().unwrap();
    reset_refreshes();
    let runtime = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let mut cpu = runtime.on_cpu(CpuPlacement::Auto).unwrap();
    assert_bound_matches_current_runtime(&cpu);
    let original_epoch = cpu.epoch;

    cpu.with_eager_session(|_: &mut dyn BackendSession| Ok(()))
        .unwrap();
    cpu.with_eager_session(|_: &mut dyn BackendSession| Ok(()))
        .unwrap();
    assert_eq!(refreshes(), 0);
    assert_eq!(cpu.epoch, original_epoch);

    let replacement = CpuBackend::new();
    runtime
        .runtime
        .reconfigure(|edit| {
            edit.replace_engine(cpu_registration_with(
                &replacement,
                ExecutionContextIdentity::of::<CpuBackend>(),
                full_cpu_capabilities(&replacement),
            ))?;
            Ok(())
        })
        .unwrap();
    let reconfigured_epoch = runtime.runtime.epoch().unwrap();
    assert_ne!(reconfigured_epoch, original_epoch);

    cpu.with_eager_session(|_: &mut dyn BackendSession| Ok(()))
        .unwrap();
    assert_eq!(refreshes(), 1);
    assert_bound_matches_current_runtime(&cpu);
    assert_eq!(cpu.epoch, reconfigured_epoch);
}

#[test]
fn runtime_snapshot_refresh_reports_typed_configuration_failures() {
    let _guard = REFRESH_PROBE_TEST_LOCK.lock().unwrap();
    let rows: [RuntimeConfigurationFailureRow; 3] = [
        (
            "missing CPU engine",
            Box::new(|runtime, _cpu| {
                runtime
                    .runtime
                    .reconfigure(|edit| {
                        edit.remove_engine(&cpu_engine_id())?;
                        Ok(())
                    })
                    .unwrap();
            }),
            ErrorKind::Unsupported,
            "missing CPU runtime engine",
        ),
        (
            "missing direct CPU capability",
            Box::new(|runtime, _cpu| {
                let backend = CpuBackend::new();
                runtime
                    .runtime
                    .reconfigure(|edit| {
                        edit.replace_engine(cpu_registration_with(
                            &backend,
                            ExecutionContextIdentity::of::<CpuBackend>(),
                            CoreCapabilityBundle::builder().build(),
                        ))?;
                        Ok(())
                    })
                    .unwrap();
            }),
            ErrorKind::Unsupported,
            "missing CPU runtime capability",
        ),
        (
            "context mismatch",
            Box::new(|runtime, _cpu| {
                let backend = CpuBackend::new();
                runtime
                    .runtime
                    .reconfigure(|edit| {
                        edit.replace_engine(cpu_registration_with(
                            &backend,
                            ExecutionContextIdentity::of::<()>(),
                            full_cpu_capabilities(&backend),
                        ))?;
                        Ok(())
                    })
                    .unwrap();
            }),
            ErrorKind::Unsupported,
            "CPU runtime context mismatch",
        ),
    ];

    for (label, break_runtime, expected_kind, expected_message) in rows {
        let runtime = EagerRuntime::with_cpu_backend(CpuBackend::new());
        let mut cpu = runtime.on_cpu(CpuPlacement::Auto).unwrap();
        break_runtime(&runtime, &mut cpu);

        let error = cpu
            .with_eager_session(|_: &mut dyn BackendSession| Ok(()))
            .unwrap_err();

        assert_eq!(error.kind(), expected_kind, "{label}: {error}");
        assert!(
            error.to_string().contains(expected_message),
            "{label}: {error}"
        );
    }
}
