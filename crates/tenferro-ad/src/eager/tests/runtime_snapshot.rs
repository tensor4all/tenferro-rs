use std::error::Error as StdError;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use tenferro_cpu::{CpuBackend, CpuPlacement, CpuProviderBundle};
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_runtime::{
    assemble_executable_engine_registration, assemble_preparation_only_engine_registration,
    CoreCapabilityBundle, DotGeneralPreparation, ElementwiseRuntime, EngineId, EngineRegistration,
    Error as RuntimeError, ErrorPhase, ExecutionContextIdentity, HardwareClassId, IndexingRuntime,
    InputIngressContract, InputPlacementContract, InputSignatureContract, LayoutRuntime,
    ProviderDeviceIdentity, ProviderId, ReductionRuntime, ResidentOutputContract,
    RuntimeCacheOwner, RuntimeInputContract, StorageClass,
};
use tenferro_tensor::{BackendSession, ErrorKind, Tensor};

use super::super::{
    CpuPlacementBoundEager, CpuRuntimeRegistrationState, EagerBackend, EagerRuntime,
    InjectedRegistrationFailure, CPU_RUNTIME_SELECTION_REFRESHES,
};

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

fn reset_registration_snapshot_reads(runtime: &EagerRuntime) {
    runtime
        .registration_snapshot_reads
        .store(0, Ordering::SeqCst);
}

fn registration_snapshot_reads(runtime: &EagerRuntime) -> usize {
    runtime.registration_snapshot_reads.load(Ordering::SeqCst)
}

fn assert_reconciliation_required(runtime: &EagerRuntime) {
    assert!(matches!(
        runtime
            .cpu_runtime_registration_state_for_test()
            .expect("CPU runtime registration state"),
        CpuRuntimeRegistrationState::ReconciliationRequired
    ));
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
        return assemble_preparation_only_engine_registration(
            cpu_engine_id(),
            provider_device_identity,
            context_identity,
            cpu_hardware_class(),
            Arc::from([storage.clone()]),
            storage,
            capabilities,
        )
        .expect("preparation binding");
    }
    assemble_executable_engine_registration(
        cpu_engine_id(),
        cpu_hardware_class(),
        Arc::from([storage.clone()]),
        storage,
        provider_device_identity,
        capabilities,
        backend.clone(),
        Arc::new(tenferro_runtime::ImmediateEventDomainDriver::new()),
        InputIngressContract::new(
            InputPlacementContract::new(|_, _| true),
            InputSignatureContract::new(|_, _, _, _| true),
            RuntimeInputContract::new(|_, _| true),
            ResidentOutputContract::new(|_, _| true),
        ),
        Some(Arc::new(backend.clone()) as Arc<dyn RuntimeCacheOwner>),
    )
    .expect("executable binding")
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
fn backend_sync_does_not_advance_epoch_for_the_same_backend() {
    let runtime = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let before = runtime.runtime.snapshot().unwrap();
    let before_epoch = before.epoch();
    let before_identity = before
        .engine(&cpu_engine_id())
        .expect("CPU engine registered")
        .registration_identity();

    runtime.with_backend_mut(|_| ()).unwrap();

    let after = runtime.runtime.snapshot().unwrap();
    assert_eq!(after.epoch(), before_epoch);
    assert_eq!(
        after
            .engine(&cpu_engine_id())
            .expect("CPU engine registered")
            .registration_identity(),
        before_identity
    );
}

#[test]
fn unchanged_cpu_runtime_identity_avoids_registration_snapshot_and_reconfigure() {
    let _guard = REFRESH_PROBE_TEST_LOCK.lock().unwrap();
    let runtime = EagerRuntime::with_cpu_backend(CpuBackend::new());
    reset_registration_snapshot_reads(&runtime);
    let before = runtime.runtime.snapshot().unwrap();
    let before_registration = before
        .engine(&cpu_engine_id())
        .expect("CPU engine registered")
        .registration_identity();

    runtime.with_backend_mut(|_| ()).unwrap();

    let after = runtime.runtime.snapshot().unwrap();
    assert_eq!(registration_snapshot_reads(&runtime), 0);
    assert_eq!(after.epoch(), before.epoch());
    assert_eq!(
        after
            .engine(&cpu_engine_id())
            .expect("CPU engine registered")
            .registration_identity(),
        before_registration
    );
}

#[test]
fn extension_only_epoch_change_keeps_cpu_registration_fast_path() {
    let _guard = REFRESH_PROBE_TEST_LOCK.lock().unwrap();
    let runtime = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let before_epoch = runtime.runtime.snapshot().unwrap().epoch();

    runtime
        .install_extension_module(super::ReadPathFallbackModule::module())
        .unwrap();
    let after_extension_epoch = runtime.runtime.snapshot().unwrap().epoch();
    assert_ne!(after_extension_epoch, before_epoch);

    reset_registration_snapshot_reads(&runtime);
    runtime.with_backend_mut(|_| ()).unwrap();

    assert_eq!(registration_snapshot_reads(&runtime), 0);
    assert_eq!(
        runtime.runtime.snapshot().unwrap().epoch(),
        after_extension_epoch
    );
}

#[test]
fn backend_sync_refreshes_a_new_witness_with_the_same_provider_device_target() {
    let backend = CpuBackend::new();
    let replacement = backend
        .clone()
        .with_provider_bundle(CpuProviderBundle::builder(backend.kind()).build().unwrap())
        .unwrap();
    let runtime = EagerRuntime::with_cpu_backend(backend);
    let engine_id = cpu_engine_id();
    let before = runtime.runtime.snapshot().unwrap();
    let before_engine = before.engine(&engine_id).expect("CPU engine registered");
    let before_provider_device = before_engine.provider_device_identity().clone();
    let before_registration = before_engine.registration_identity();
    let before_epoch = before.epoch();

    runtime
        .with_backend_mut(|backend| *backend = EagerBackend::Cpu(replacement))
        .unwrap();

    let after = runtime.runtime.snapshot().unwrap();
    let after_engine = after.engine(&engine_id).expect("CPU engine registered");
    assert_eq!(
        after_engine.provider_device_identity(),
        &before_provider_device
    );
    assert_ne!(after_engine.registration_identity(), before_registration);
    assert_ne!(after.epoch(), before_epoch);
}

#[test]
fn backend_cache_closure_error_still_synchronizes_mutated_cpu_runtime() {
    let backend = CpuBackend::new();
    let replacement = backend
        .clone()
        .with_provider_bundle(CpuProviderBundle::builder(backend.kind()).build().unwrap())
        .unwrap();
    let runtime = EagerRuntime::with_cpu_backend(backend);
    let before = runtime.runtime.snapshot().unwrap();
    let before_engine = before
        .engine(&cpu_engine_id())
        .expect("CPU engine registered");
    let before_provider_device = before_engine.provider_device_identity().clone();
    let before_registration = before_engine.registration_identity();
    let before_epoch = before.epoch();

    let error = runtime
        .with_backend_and_extension_caches_mut::<()>(|backend, _| {
            *backend = EagerBackend::Cpu(replacement);
            Err(tenferro_tensor::Error::BackendFailure {
                op: "test_backend_cache_closure",
                message: "primary closure failure".to_owned(),
            })
        })
        .unwrap_err();
    assert!(error.to_string().contains("primary closure failure"));

    let after = runtime.runtime.snapshot().unwrap();
    let after_engine = after
        .engine(&cpu_engine_id())
        .expect("CPU engine registered");
    assert_eq!(
        after_engine.provider_device_identity(),
        &before_provider_device
    );
    assert_ne!(after_engine.registration_identity(), before_registration);
    assert_ne!(after.epoch(), before_epoch);
}

#[test]
fn failed_backend_mutation_preserves_primary_and_quarantines_registration() {
    let backend = CpuBackend::new();
    let replacement = backend
        .clone()
        .with_provider_bundle(CpuProviderBundle::builder(backend.kind()).build().unwrap())
        .unwrap();
    let runtime = EagerRuntime::with_cpu_backend(backend);
    let before = runtime.runtime.snapshot().unwrap();
    let before_registration = before
        .engine(&cpu_engine_id())
        .expect("CPU engine registered")
        .registration_identity();
    runtime.inject_next_registration_failure_for_test().unwrap();

    let error = runtime
        .with_backend_and_extension_caches_mut::<()>(|backend, _| {
            *backend = EagerBackend::Cpu(replacement);
            Err(tenferro_tensor::Error::BackendSource {
                op: "test_backend_cache_closure",
                source: Box::new(std::io::Error::other("primary closure source")),
            })
        })
        .unwrap_err();

    assert_reconciliation_required(&runtime);
    assert_eq!(error.kind(), ErrorKind::BackendFailure);
    assert_eq!(error.phase(), Some(ErrorPhase::Execution));
    assert!(matches!(error, RuntimeError::WithSuppressed { .. }));

    let primary = error.primary().expect("primary closure error");
    assert_eq!(primary.kind(), ErrorKind::BackendFailure);
    assert_eq!(primary.phase(), Some(ErrorPhase::Execution));
    let primary_source = StdError::source(primary).expect("typed primary source");
    assert!(primary_source
        .to_string()
        .contains("primary closure source"));
    assert_eq!(
        StdError::source(&error)
            .expect("aggregate delegates primary source")
            .to_string(),
        primary.to_string()
    );

    let suppressed = error
        .suppressed()
        .expect("suppressed synchronization error");
    assert_eq!(suppressed.kind(), ErrorKind::RuntimeState);
    assert_eq!(suppressed.phase(), Some(ErrorPhase::Execution));
    assert!(matches!(
        suppressed,
        RuntimeError::RuntimeStateSource { .. }
    ));
    let suppressed_source = StdError::source(suppressed).expect("typed sync source");
    assert!(suppressed_source
        .downcast_ref::<InjectedRegistrationFailure>()
        .is_some());

    let after = runtime.runtime.snapshot().unwrap();
    assert_eq!(
        after
            .engine(&cpu_engine_id())
            .expect("CPU engine remains registered")
            .registration_identity(),
        before_registration
    );
}

#[test]
fn registration_quarantine_blocks_use_until_retry_then_clears() {
    let _guard = REFRESH_PROBE_TEST_LOCK.lock().unwrap();
    let backend = CpuBackend::new();
    let replacement = backend
        .clone()
        .with_provider_bundle(CpuProviderBundle::builder(backend.kind()).build().unwrap())
        .unwrap();
    let replacement_identity = replacement.runtime_identity();
    let runtime = EagerRuntime::with_cpu_backend(backend);
    let mut cpu = runtime.on_cpu(CpuPlacement::Auto).unwrap();
    runtime.inject_next_registration_failure_for_test().unwrap();
    runtime
        .with_backend_and_extension_caches_mut::<()>(|backend, _| {
            *backend = EagerBackend::Cpu(replacement);
            Ok(())
        })
        .unwrap_err();

    runtime.inject_next_registration_failure_for_test().unwrap();
    let bound_session_calls = AtomicUsize::new(0);
    let error = cpu
        .with_eager_session(|_: &mut dyn BackendSession| {
            bound_session_calls.fetch_add(1, Ordering::SeqCst);
            Ok(())
        })
        .unwrap_err();
    assert_eq!(bound_session_calls.load(Ordering::SeqCst), 0);
    assert!(matches!(error, RuntimeError::RuntimeStateSource { .. }));
    assert_reconciliation_required(&runtime);

    runtime.inject_next_registration_failure_for_test().unwrap();
    let closure_calls = AtomicUsize::new(0);
    let error = runtime
        .with_backend_mut(|_| {
            closure_calls.fetch_add(1, Ordering::SeqCst);
        })
        .unwrap_err();
    assert_eq!(closure_calls.load(Ordering::SeqCst), 0);
    assert!(matches!(error, RuntimeError::RuntimeStateSource { .. }));
    assert_reconciliation_required(&runtime);

    runtime.inject_next_registration_failure_for_test().unwrap();
    let execution_calls = AtomicUsize::new(0);
    let error = runtime
        .exec_outputs_with_runtime(
            "test_execution.lock_backend",
            "test_execution.execute",
            &StdTensorOp::Neg,
            |_backend, _runtime| {
                execution_calls.fetch_add(1, Ordering::SeqCst);
                Ok::<(), crate::Error>(())
            },
        )
        .unwrap_err();
    assert_eq!(execution_calls.load(Ordering::SeqCst), 0);
    assert!(matches!(error, RuntimeError::RuntimeStateSource { .. }));
    assert_reconciliation_required(&runtime);

    let retry_calls = AtomicUsize::new(0);
    runtime
        .with_backend_mut(|_| {
            retry_calls.fetch_add(1, Ordering::SeqCst);
        })
        .unwrap();
    assert_eq!(retry_calls.load(Ordering::SeqCst), 1);
    assert!(matches!(
        runtime
            .cpu_runtime_registration_state_for_test()
            .unwrap(),
        CpuRuntimeRegistrationState::Present { identity } if identity == replacement_identity
    ));

    let bound_session_calls = AtomicUsize::new(0);
    cpu.with_eager_session(|_: &mut dyn BackendSession| {
        bound_session_calls.fetch_add(1, Ordering::SeqCst);
        Ok(())
    })
    .unwrap();
    assert_eq!(bound_session_calls.load(Ordering::SeqCst), 1);

    let input = Tensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap();
    let output = runtime
        .exec_outputs(&StdTensorOp::Neg, &[&input])
        .unwrap()
        .pop()
        .expect("neg output");
    assert_eq!(output.as_slice::<f64>().unwrap(), &[-2.0]);
}

#[test]
fn cpu_runtime_identity_is_shared_only_by_exact_backend_clones() {
    let backend = CpuBackend::new();
    let clone = backend.clone();
    let replacement = backend
        .clone()
        .with_provider_bundle(CpuProviderBundle::builder(backend.kind()).build().unwrap())
        .unwrap();

    assert_eq!(backend.runtime_identity(), clone.runtime_identity());
    assert_ne!(backend.runtime_identity(), replacement.runtime_identity());
    assert_ne!(
        CpuBackend::new().runtime_identity(),
        CpuBackend::new().runtime_identity()
    );
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
