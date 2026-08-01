use std::error::Error as StdError;
use std::num::NonZeroU64;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Barrier};
use std::time::{SystemTime, UNIX_EPOCH};

use super::super::*;
use crate::runtime::engine_registration::EngineRegistrationState;

#[derive(Debug)]
struct TestContext;

#[derive(Debug)]
struct NoopElementwise;

impl ElementwiseRuntime for NoopElementwise {
    fn prepare(
        &self,
        _request: ElementwisePrepareRequest<'_>,
    ) -> Result<PrepareCapability, PrepareError> {
        Ok(PrepareCapability::Unsupported(
            UnsupportedReason::MissingCapability {
                capability: CoreCapabilityKind::Elementwise,
            },
        ))
    }
}

fn engine_id(name: &str) -> EngineId {
    EngineId::new(name).expect("valid test engine id")
}

fn hardware() -> HardwareClassId {
    HardwareClassId::new("tenferro.cpu.host").expect("valid test hardware class")
}

fn storage(name: &str) -> StorageClass {
    StorageClass::new(name).expect("valid test storage class")
}

fn provider_target(target: &str) -> ProviderDeviceIdentity {
    ProviderDeviceIdentity::new(
        ProviderId::new("tenferro.test.provider").expect("valid provider id"),
        target,
    )
    .expect("valid provider target")
}

fn registration_with_provider_target(
    engine_name: &str,
    target: &str,
) -> Result<EngineRegistration, RuntimeConfigError> {
    let host = storage("tenferro.storage.host");
    ProviderPreparationBinding::new(
        engine_id(engine_name),
        provider_target(target),
        ExecutionContextIdentity::of::<TestContext>(),
        hardware(),
        Arc::from(vec![host.clone()]),
        host,
        capabilities(),
    )
    .map(EngineRegistration::preparation_only)
}

fn capabilities() -> CoreCapabilityBundle {
    let mut builder = CoreCapabilityBundle::builder();
    builder.elementwise(Arc::new(NoopElementwise));
    builder.build()
}

fn registration(
    engine_name: &str,
    _candidate_nonce: u64,
) -> Result<EngineRegistration, RuntimeConfigError> {
    let host = storage("tenferro.storage.host");
    ProviderPreparationBinding::new(
        engine_id(engine_name),
        provider_target(engine_name),
        ExecutionContextIdentity::of::<TestContext>(),
        hardware(),
        Arc::from(vec![host.clone()]),
        host,
        capabilities(),
    )
    .map(EngineRegistration::preparation_only)
}

fn executable_registration(
    engine_name: &str,
    driver: Arc<dyn EventDomainDriver>,
) -> Result<EngineRegistration, RuntimeConfigError> {
    let host = storage("tenferro.storage.host");
    let provider = provider_target(engine_name);
    let ingress = InputIngressContract::new(
        InputPlacementContract::new(|_, _| true),
        InputSignatureContract::new(|_, _, _, _| true),
        RuntimeInputContract::new(|_, _| true),
        ResidentOutputContract::new(|_, _| true),
    );
    let contract = ExecutableEngineContract::new(
        provider.clone(),
        capabilities(),
        tenferro_cpu::CpuBackend::new(),
        driver,
        ingress,
        None,
    );
    ProviderExecutableBinding::new(
        engine_id(engine_name),
        hardware(),
        Arc::from(vec![host.clone()]),
        host,
        contract,
    )
    .map(EngineRegistration::executable)
}

fn runtime_with(engine_name: &str, candidate_nonce: u64) -> Result<Runtime, RuntimeConfigError> {
    let mut builder = Runtime::builder();
    builder.register_engine(registration(engine_name, candidate_nonce)?)?;
    builder.build()
}

#[derive(Debug)]
struct UnusedTransferProvider;

impl TransferProvider for UnusedTransferProvider {
    fn transfer_blocking(
        &self,
        _request: TransferRequest<'_>,
    ) -> crate::Result<tenferro_tensor::Tensor> {
        Err(crate::Error::Internal("unused test transfer".into()))
    }
}

fn single_identity(runtime: &Runtime, engine_name: &str) -> RegistrationIdentity {
    runtime
        .snapshot()
        .expect("snapshot")
        .engine(&engine_id(engine_name))
        .expect("engine slot")
        .registration_identity()
}

#[test]
fn engine_registration_has_no_implicit_event_domain_driver() {
    let registration = registration("tenferro.engine.no-event-driver", 1).unwrap();
    assert!(matches!(
        registration.execution_state(),
        EngineRegistrationState::PreparationOnly { .. }
    ));
}

#[test]
fn duplicate_provider_device_target_is_structured_and_does_not_insert_second_engine(
) -> Result<(), Box<dyn StdError>> {
    let mut builder = Runtime::builder();
    builder.register_engine(registration_with_provider_target(
        "tenferro.engine.target.first",
        "device-0",
    )?)?;
    let error = builder
        .register_engine(registration_with_provider_target(
            "tenferro.engine.target.second",
            "device-0",
        )?)
        .expect_err("duplicate physical target must be rejected");

    assert!(matches!(
        error,
        RuntimeConfigError::DuplicateProviderDeviceTarget {
            provider_device_identity,
            first_engine_id,
            duplicate_engine_id,
        } if provider_device_identity == provider_target("device-0")
            && first_engine_id == engine_id("tenferro.engine.target.first")
            && duplicate_engine_id == engine_id("tenferro.engine.target.second")
    ));
    let runtime = builder.build()?;
    assert_eq!(runtime.snapshot()?.engine_count(), 1);
    Ok(())
}

#[test]
fn same_target_engine_replacement_is_allowed_and_exposes_binding() -> Result<(), Box<dyn StdError>>
{
    let runtime = {
        let mut builder = Runtime::builder();
        builder.register_engine(registration_with_provider_target(
            "tenferro.engine.target.same",
            "device-0",
        )?)?;
        builder.build()?
    };
    let before_epoch = runtime.epoch()?;

    runtime.reconfigure(|edit| {
        edit.replace_engine(registration_with_provider_target(
            "tenferro.engine.target.same",
            "device-0",
        )?)?;
        Ok(())
    })?;

    let snapshot = runtime.snapshot()?;
    assert!(snapshot.epoch() > before_epoch);
    assert_eq!(
        snapshot
            .engine(&engine_id("tenferro.engine.target.same"))
            .unwrap()
            .provider_device_identity(),
        &provider_target("device-0")
    );
    Ok(())
}

#[test]
fn direct_target_replacement_is_rejected_before_mutation() -> Result<(), Box<dyn StdError>> {
    let runtime = {
        let mut builder = Runtime::builder();
        builder.register_engine(registration_with_provider_target(
            "tenferro.engine.target.rebind",
            "device-0",
        )?)?;
        builder.build()?
    };
    let before = runtime.snapshot()?;
    let before_epoch = before.epoch();

    let error = runtime
        .reconfigure(|edit| {
            edit.replace_engine(registration_with_provider_target(
                "tenferro.engine.target.rebind",
                "device-1",
            )?)?;
            Ok(())
        })
        .expect_err("direct target rebind must require explicit route transaction");
    let diagnostic = match &error {
        RuntimeReconfigureError::Edit { source } => source.to_string(),
        other => panic!("unexpected reconfiguration error: {other:?}"),
    };
    assert!(diagnostic.contains("remove affected transfer routes"));
    assert!(diagnostic.contains("remove the old engine"));
    assert!(diagnostic.contains("register the replacement under the engine ID"));
    assert!(diagnostic.contains("re-register the routes"));

    assert!(matches!(
        error,
        RuntimeReconfigureError::Edit {
            source: RuntimeConfigError::EngineTargetRebind {
                engine_id,
                current,
                replacement,
            }
        } if engine_id.as_str() == "tenferro.engine.target.rebind"
            && current == provider_target("device-0")
            && replacement == provider_target("device-1")
    ));
    let after = runtime.snapshot()?;
    assert!(Arc::ptr_eq(&before, &after));
    assert_eq!(after.epoch(), before_epoch);
    assert_eq!(
        after
            .engine(&engine_id("tenferro.engine.target.rebind"))
            .unwrap()
            .provider_device_identity(),
        &provider_target("device-0")
    );
    Ok(())
}

#[test]
fn stale_route_rejects_target_rebind_and_explicit_route_rebind_succeeds(
) -> Result<(), Box<dyn StdError>> {
    let source_id = engine_id("tenferro.engine.route.source");
    let destination_id = engine_id("tenferro.engine.route.destination");
    let storage = storage("tenferro.storage.host");
    let source_endpoint = TransferEndpoint::new(source_id.clone(), storage.clone());
    let destination_endpoint = TransferEndpoint::new(destination_id.clone(), storage.clone());
    let provider = Arc::new(UnusedTransferProvider);
    let runtime = {
        let mut builder = Runtime::builder();
        builder.register_engine(registration_with_provider_target(
            source_id.as_str(),
            "device-0",
        )?)?;
        builder.register_engine(registration_with_provider_target(
            destination_id.as_str(),
            "host-0",
        )?)?;
        builder.register_transfer_provider(
            source_endpoint.clone(),
            destination_endpoint.clone(),
            Arc::clone(&provider) as Arc<dyn TransferProvider>,
        )?;
        builder.build()?
    };
    let before = runtime.snapshot()?;

    let stale_error = runtime
        .reconfigure(|edit| {
            edit.register_transfer_provider(
                source_endpoint.clone(),
                destination_endpoint.clone(),
                Arc::clone(&provider) as Arc<dyn TransferProvider>,
            )?;
            edit.remove_engine(&source_id)?;
            edit.register_engine(registration_with_provider_target(
                source_id.as_str(),
                "device-1",
            )?)?;
            Ok(())
        })
        .expect_err("retaining a route across a target change must be stale");
    assert!(matches!(
        stale_error,
        RuntimeReconfigureError::Edit {
            source: RuntimeConfigError::StaleTransferRoute {
                source_endpoint: actual_source,
                destination,
                endpoint,
                registered,
                current,
            }
        } if actual_source == source_endpoint
            && destination == destination_endpoint
            && endpoint == source_endpoint
            && registered.as_ref() == &provider_target("device-0")
            && current.as_ref() == &provider_target("device-1")
    ));
    let after_failure = runtime.snapshot()?;
    assert!(Arc::ptr_eq(&before, &after_failure));
    assert_eq!(after_failure.transfer_provider_count(), 1);

    runtime.reconfigure(|edit| {
        edit.remove_transfer_provider(source_endpoint.clone(), destination_endpoint.clone())?;
        edit.remove_engine(&source_id)?;
        edit.register_engine(registration_with_provider_target(
            source_id.as_str(),
            "device-1",
        )?)?;
        edit.register_transfer_provider(
            source_endpoint,
            destination_endpoint,
            Arc::clone(&provider) as Arc<dyn TransferProvider>,
        )?;
        Ok(())
    })?;
    let rebound = runtime.snapshot()?;
    assert_eq!(rebound.transfer_provider_count(), 1);
    assert_eq!(
        rebound
            .engine(&engine_id("tenferro.engine.route.source"))
            .unwrap()
            .provider_device_identity(),
        &provider_target("device-1")
    );
    Ok(())
}

#[test]
fn route_registration_is_independent_of_engine_registration_order() -> Result<(), Box<dyn StdError>>
{
    let source_id = engine_id("tenferro.engine.order.source");
    let destination_id = engine_id("tenferro.engine.order.destination");
    let storage = storage("tenferro.storage.host");
    let provider: Arc<dyn TransferProvider> = Arc::new(UnusedTransferProvider);
    let mut builder = Runtime::builder();
    builder.register_transfer_provider(
        TransferEndpoint::new(source_id.clone(), storage.clone()),
        TransferEndpoint::new(destination_id.clone(), storage.clone()),
        provider,
    )?;
    builder.register_engine(registration_with_provider_target(
        destination_id.as_str(),
        "host-0",
    )?)?;
    builder.register_engine(registration_with_provider_target(
        source_id.as_str(),
        "device-0",
    )?)?;

    assert_eq!(builder.build()?.snapshot()?.transfer_provider_count(), 1);
    Ok(())
}

#[test]
fn new_route_binding_ignores_pre_freeze_engine_edits() -> Result<(), Box<dyn StdError>> {
    let source_id = engine_id("tenferro.engine.pre-freeze.source");
    let destination_id = engine_id("tenferro.engine.pre-freeze.destination");
    let storage = storage("tenferro.storage.host");
    let source_endpoint = TransferEndpoint::new(source_id.clone(), storage.clone());
    let destination_endpoint = TransferEndpoint::new(destination_id.clone(), storage.clone());
    let provider: Arc<dyn TransferProvider> = Arc::new(UnusedTransferProvider);
    let mut builder = Runtime::builder();

    builder.register_engine(registration_with_provider_target(
        source_id.as_str(),
        "device-before",
    )?)?;
    builder.register_engine(registration_with_provider_target(
        destination_id.as_str(),
        "host-before",
    )?)?;
    builder.register_transfer_provider(
        source_endpoint.clone(),
        destination_endpoint.clone(),
        provider,
    )?;
    builder.remove_engine(&source_id)?;
    builder.remove_engine(&destination_id)?;
    builder.register_engine(registration_with_provider_target(
        source_id.as_str(),
        "device-after",
    )?)?;
    builder.register_engine(registration_with_provider_target(
        destination_id.as_str(),
        "host-after",
    )?)?;

    let snapshot = builder.build()?.snapshot()?;
    let routes: Vec<_> = snapshot.transfer_routes_for_test().collect();
    assert_eq!(routes.len(), 1);
    assert_eq!(routes[0].source().logical(), &source_endpoint);
    assert_eq!(routes[0].destination().logical(), &destination_endpoint);
    assert_eq!(
        routes[0].source().provider_device_identity(),
        &provider_target("device-after")
    );
    assert_eq!(
        routes[0].destination().provider_device_identity(),
        &provider_target("host-after")
    );
    Ok(())
}

#[test]
fn custom_event_domain_driver_survives_snapshot_freeze() -> Result<(), Box<dyn StdError>> {
    let driver: Arc<dyn EventDomainDriver> = Arc::new(ImmediateEventDomainDriver::new());
    let weak_driver = Arc::downgrade(&driver);
    let registration =
        executable_registration("tenferro.engine.event-driver", Arc::clone(&driver))?;
    let mut builder = Runtime::builder();
    builder.register_engine(registration)?;
    let runtime = builder.build()?;
    drop(driver);

    let snapshot = runtime.snapshot()?;
    let frozen = snapshot
        .engine(&engine_id("tenferro.engine.event-driver"))
        .expect("engine slot")
        .executable_witness()
        .expect("explicit event-domain driver");
    assert!(Arc::ptr_eq(
        frozen.event_domain_driver(),
        &weak_driver.upgrade().expect("snapshot retains driver")
    ));
    Ok(())
}

#[test]
fn fresh_builds_have_distinct_runtime_and_registration_identities() -> Result<(), Box<dyn StdError>>
{
    let first = runtime_with("tenferro.engine.same", 1)?;
    let second = runtime_with("tenferro.engine.same", 2)?;

    assert_ne!(first.id(), second.id());
    let first_snapshot = first.snapshot()?;
    let second_snapshot = second.snapshot()?;
    assert_ne!(first_snapshot.runtime_id(), second_snapshot.runtime_id());

    let first_identity = first_snapshot
        .engine(&engine_id("tenferro.engine.same"))
        .expect("first engine")
        .registration_identity();
    let second_identity = second_snapshot
        .engine(&engine_id("tenferro.engine.same"))
        .expect("second engine")
        .registration_identity();
    assert_eq!(first_identity.ordinal().get(), 1);
    assert_eq!(second_identity.ordinal().get(), 1);
    assert_ne!(first_identity, second_identity);

    Ok(())
}

#[test]
fn unchanged_engine_gets_a_distinct_event_domain_in_the_next_epoch() -> Result<(), Box<dyn StdError>>
{
    let runtime = runtime_with("tenferro.engine.event-epoch", 1)?;
    let before = runtime.snapshot()?;
    let engine = engine_id("tenferro.engine.event-epoch");
    let before_engine = before.engine(&engine).expect("initial event engine");
    let before_domain = before_engine.event_domain_id();
    let before_identity = before_engine.registration_identity();

    runtime.reconfigure(|edit| {
        edit.register_engine(registration("tenferro.engine.event-epoch-other", 2)?)?;
        Ok(())
    })?;

    let after = runtime.snapshot()?;
    let after_engine = after.engine(&engine).expect("next-epoch event engine");
    let after_domain = after_engine.event_domain_id();
    assert_eq!(before_identity, after_engine.registration_identity());
    assert_ne!(before.epoch(), after.epoch());
    assert_ne!(before_domain, after_domain);
    assert_eq!(before_domain.runtime_id(), after_domain.runtime_id());
    assert_eq!(after_domain.epoch(), after.epoch());
    assert_eq!(
        after_domain.registration_identity(),
        after_engine.registration_identity()
    );
    Ok(())
}

#[test]
fn fresh_build_epoch_matches_snapshot_epoch() -> Result<(), Box<dyn StdError>> {
    let runtime = runtime_with("tenferro.engine.epoch", 1)?;
    let snapshot = runtime.snapshot()?;

    assert_eq!(runtime.id(), snapshot.runtime_id());
    assert_eq!(runtime.epoch()?, snapshot.epoch());
    assert_eq!(snapshot.engine_count(), 1);
    assert_eq!(snapshot.extension_module_count(), 0);

    Ok(())
}

#[test]
fn epoch_acquire_returns_published_epoch_repeatedly() -> Result<(), Box<dyn StdError>> {
    let runtime = runtime_with("tenferro.engine.epoch-fast", 1)?;
    let epoch = runtime.snapshot()?.epoch();

    for _ in 0..100 {
        assert_eq!(runtime.epoch()?, epoch);
    }

    Ok(())
}

#[test]
fn engine_slots_are_sorted_by_engine_id() -> Result<(), Box<dyn StdError>> {
    let mut builder = Runtime::builder();
    builder.register_engine(registration("tenferro.engine.zed", 1)?)?;
    builder.register_engine(registration("tenferro.engine.alpha", 2)?)?;

    let runtime = builder.build()?;
    let snapshot = runtime.snapshot()?;
    let ids: Vec<_> = snapshot
        .engine_ids_for_test()
        .map(EngineId::as_str)
        .collect();

    assert_eq!(ids, vec!["tenferro.engine.alpha", "tenferro.engine.zed"]);

    Ok(())
}

#[test]
fn engine_registration_validates_storage_classes_before_candidate_token() {
    let engine = engine_id("tenferro.engine.storage");
    let default = storage("tenferro.storage.host");

    let empty = ProviderPreparationBinding::new(
        engine.clone(),
        provider_target("validation-empty"),
        ExecutionContextIdentity::of::<TestContext>(),
        hardware(),
        Arc::from(Vec::<StorageClass>::new()),
        default.clone(),
        capabilities(),
    )
    .expect_err("empty storage class list");
    assert!(matches!(
        empty,
        RuntimeConfigError::EmptyStorageClasses { engine_id } if engine_id == engine
    ));

    let duplicate = ProviderPreparationBinding::new(
        engine.clone(),
        provider_target("validation-duplicate"),
        ExecutionContextIdentity::of::<TestContext>(),
        hardware(),
        Arc::from(vec![default.clone(), default.clone()]),
        default.clone(),
        capabilities(),
    )
    .expect_err("duplicate storage class");
    assert!(matches!(
        duplicate,
        RuntimeConfigError::DuplicateStorageClass {
            engine_id,
            storage_class,
            first_index: 0,
            duplicate_index: 1,
        } if engine_id == engine && storage_class == default
    ));

    let missing_default = ProviderPreparationBinding::new(
        engine.clone(),
        provider_target("validation-default"),
        ExecutionContextIdentity::of::<TestContext>(),
        hardware(),
        Arc::from(vec![storage("tenferro.storage.device")]),
        default.clone(),
        capabilities(),
    )
    .expect_err("missing default storage class");
    assert!(matches!(
        missing_default,
        RuntimeConfigError::DefaultStorageClassNotListed {
            engine_id,
            default_storage_class,
        } if engine_id == engine && default_storage_class == default
    ));
}

#[test]
fn engine_registration_records_tensor_backend_execution_bridge() -> Result<(), Box<dyn StdError>> {
    let registration = executable_registration(
        "tenferro.engine.bridge",
        Arc::new(ImmediateEventDomainDriver::new()),
    )?;
    assert!(matches!(
        registration.execution_state(),
        EngineRegistrationState::Executable(_)
    ));

    let mut builder = Runtime::builder();
    builder.register_engine(registration)?;
    let runtime = builder.build()?;
    let snapshot = runtime.snapshot()?;
    let engine = snapshot
        .engine(&engine_id("tenferro.engine.bridge"))
        .expect("engine with bridge");

    assert!(engine.has_execution_engine_for_test());

    Ok(())
}

#[test]
fn identical_registration_is_noop_and_preserves_epoch_and_identity() -> Result<(), Box<dyn StdError>>
{
    let registration = registration("tenferro.engine.noop", 1)?;
    let mut builder = Runtime::builder();
    builder.register_engine(registration.clone())?;
    let runtime = builder.build()?;
    let before_epoch = runtime.epoch()?;
    let before_identity = single_identity(&runtime, "tenferro.engine.noop");

    let returned_epoch = runtime.reconfigure(|edit| {
        edit.register_engine(registration)?;
        Ok(())
    })?;

    assert_eq!(returned_epoch, before_epoch);
    assert_eq!(runtime.epoch()?, before_epoch);
    assert_eq!(
        single_identity(&runtime, "tenferro.engine.noop"),
        before_identity
    );

    Ok(())
}

#[test]
fn conflicting_registration_is_typed_and_publishes_nothing() -> Result<(), Box<dyn StdError>> {
    let runtime = runtime_with("tenferro.engine.conflict", 1)?;
    let before_epoch = runtime.epoch()?;
    let before_identity = single_identity(&runtime, "tenferro.engine.conflict");

    let error = runtime
        .reconfigure(|edit| {
            edit.register_engine(registration("tenferro.engine.conflict", 2)?)?;
            Ok(())
        })
        .expect_err("conflicting registration should fail");

    assert!(matches!(
        error,
        RuntimeReconfigureError::Edit {
            source: RuntimeConfigError::DuplicateEngine { engine_id: duplicate },
        } if duplicate == engine_id("tenferro.engine.conflict")
    ));
    assert_eq!(runtime.epoch()?, before_epoch);
    assert_eq!(
        single_identity(&runtime, "tenferro.engine.conflict"),
        before_identity
    );

    Ok(())
}

#[test]
fn replacement_advances_epoch_and_changes_registration_identity() -> Result<(), Box<dyn StdError>> {
    let runtime = runtime_with("tenferro.engine.replace", 1)?;
    let before_epoch = runtime.epoch()?;
    let before_identity = single_identity(&runtime, "tenferro.engine.replace");

    let returned_epoch = runtime.reconfigure(|edit| {
        edit.replace_engine(registration("tenferro.engine.replace", 2)?)?;
        Ok(())
    })?;

    assert!(returned_epoch > before_epoch);
    assert_eq!(runtime.epoch()?, returned_epoch);
    let after_identity = single_identity(&runtime, "tenferro.engine.replace");
    assert_ne!(after_identity, before_identity);
    assert_eq!(after_identity.ordinal().get(), 2);

    Ok(())
}

#[test]
fn old_snapshot_remains_readable_after_replacement() -> Result<(), Box<dyn StdError>> {
    let runtime = runtime_with("tenferro.engine.snapshot", 1)?;
    let old_snapshot = runtime.snapshot()?;
    let old_identity = old_snapshot
        .engine(&engine_id("tenferro.engine.snapshot"))
        .expect("old engine")
        .registration_identity();

    runtime.reconfigure(|edit| {
        edit.replace_engine(registration("tenferro.engine.snapshot", 2)?)?;
        Ok(())
    })?;

    assert_eq!(
        old_snapshot
            .engine(&engine_id("tenferro.engine.snapshot"))
            .expect("old engine remains readable")
            .registration_identity(),
        old_identity
    );
    assert_ne!(
        runtime
            .snapshot()?
            .engine(&engine_id("tenferro.engine.snapshot"))
            .expect("new engine")
            .registration_identity(),
        old_identity
    );

    Ok(())
}

#[test]
fn failed_edit_allocates_no_registration_ordinal() -> Result<(), Box<dyn StdError>> {
    let runtime = runtime_with("tenferro.engine.base", 1)?;
    let failed_new_registration = registration("tenferro.engine.failed", 2)?;

    let error = runtime
        .reconfigure(|edit| {
            edit.register_engine(failed_new_registration)?;
            Err(RuntimeConfigError::DuplicateEngine {
                engine_id: engine_id("tenferro.engine.base"),
            })
        })
        .expect_err("edit failure should abort");
    assert!(matches!(error, RuntimeReconfigureError::Edit { .. }));
    assert!(runtime
        .snapshot()?
        .engine(&engine_id("tenferro.engine.failed"))
        .is_none());

    runtime.reconfigure(|edit| {
        edit.register_engine(registration("tenferro.engine.next", 3)?)?;
        Ok(())
    })?;

    let new_identity = single_identity(&runtime, "tenferro.engine.next");
    assert_eq!(new_identity.ordinal().get(), 2);

    Ok(())
}

#[test]
fn unchanged_records_retain_identity_while_new_records_get_next_ordinal(
) -> Result<(), Box<dyn StdError>> {
    let runtime = runtime_with("tenferro.engine.stable", 1)?;
    let stable_identity = single_identity(&runtime, "tenferro.engine.stable");

    runtime.reconfigure(|edit| {
        edit.register_engine(registration("tenferro.engine.added", 2)?)?;
        Ok(())
    })?;

    assert_eq!(
        single_identity(&runtime, "tenferro.engine.stable"),
        stable_identity
    );
    assert_eq!(
        single_identity(&runtime, "tenferro.engine.added")
            .ordinal()
            .get(),
        2
    );

    Ok(())
}

#[test]
fn simultaneous_writers_yield_one_publication_and_one_concurrent_error(
) -> Result<(), Box<dyn StdError>> {
    let runtime = runtime_with("tenferro.engine.concurrent", 1)?;
    let base_epoch = runtime.epoch()?;
    let barrier = Arc::new(Barrier::new(2));

    let first_runtime = runtime.clone();
    let first_barrier = Arc::clone(&barrier);
    let first = std::thread::spawn(move || {
        first_runtime.reconfigure(|edit| {
            edit.replace_engine(registration("tenferro.engine.concurrent", 2)?)?;
            first_barrier.wait();
            Ok(())
        })
    });

    let second_runtime = runtime.clone();
    let second_barrier = Arc::clone(&barrier);
    let second = std::thread::spawn(move || {
        second_runtime.reconfigure(|edit| {
            edit.replace_engine(registration("tenferro.engine.concurrent", 3)?)?;
            second_barrier.wait();
            Ok(())
        })
    });

    let first = first.join().expect("first writer should not panic");
    let second = second.join().expect("second writer should not panic");

    let (published, rejected) = match (first, second) {
        (Ok(epoch), Err(error)) | (Err(error), Ok(epoch)) => (epoch, error),
        other => panic!("expected one success and one concurrent error, got {other:?}"),
    };

    assert!(published > base_epoch);
    assert_eq!(runtime.epoch()?, published);
    assert!(matches!(
        rejected,
        RuntimeReconfigureError::ConcurrentReconfiguration { base, current }
            if base == base_epoch && current == published
    ));

    Ok(())
}

#[test]
fn epoch_overflow_publishes_nothing() -> Result<(), Box<dyn StdError>> {
    let runtime = runtime_with("tenferro.engine.epoch-overflow", 1)?;
    let max_epoch = RuntimeEpoch::from_nonzero(NonZeroU64::new(u64::MAX).unwrap());
    runtime.force_epoch_for_test(max_epoch);

    let error = runtime
        .reconfigure(|edit| {
            edit.execution_policy(ExecutionPolicy::new(Determinism::Reproducible, None, 9));
            Ok(())
        })
        .expect_err("epoch overflow should fail");

    assert!(matches!(
        error,
        RuntimeReconfigureError::EpochExhausted { current } if current == max_epoch
    ));
    assert_eq!(runtime.epoch()?, max_epoch);

    Ok(())
}

#[test]
fn registration_ordinal_overflow_publishes_nothing() -> Result<(), Box<dyn StdError>> {
    let runtime = runtime_with("tenferro.engine.ordinal-overflow", 1)?;
    let before_epoch = runtime.epoch()?;
    runtime.force_next_registration_ordinal_for_test(NonZeroU64::new(u64::MAX).unwrap());

    let error = runtime
        .reconfigure(|edit| {
            edit.register_engine(registration("tenferro.engine.ordinal-new", 2)?)?;
            Ok(())
        })
        .expect_err("ordinal overflow should fail before publication");

    assert!(matches!(
        error,
        RuntimeReconfigureError::Edit {
            source: RuntimeConfigError::IdentityExhausted,
        }
    ));
    assert_eq!(runtime.epoch()?, before_epoch);
    assert!(runtime
        .snapshot()?
        .engine(&engine_id("tenferro.engine.ordinal-new"))
        .is_none());

    Ok(())
}

#[test]
fn poisoned_active_lock_returns_runtime_state_error() -> Result<(), Box<dyn StdError>> {
    let runtime = runtime_with("tenferro.engine.poison", 1)?;

    runtime.poison_active_lock_for_test();

    let snapshot_error = runtime
        .snapshot()
        .expect_err("poisoned active snapshot lock should error");
    assert!(matches!(
        snapshot_error,
        RuntimeStateError::Poisoned {
            lock: "runtime.active"
        }
    ));

    let reconfigure_error = runtime
        .reconfigure(|edit| {
            edit.execution_policy(ExecutionPolicy::new(Determinism::Reproducible, None, 1));
            Ok(())
        })
        .expect_err("poisoned active lock should abort reconfigure");
    assert!(matches!(
        reconfigure_error,
        RuntimeReconfigureError::State {
            source: RuntimeStateError::Poisoned {
                lock: "runtime.active"
            },
        }
    ));

    Ok(())
}

#[test]
fn runtime_config_snapshot_has_no_downstream_construction_path() {
    let temp = temp_compile_crate_dir();
    write_compile_fail_crate(&temp);

    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_owned());
    let output = Command::new(cargo)
        .arg("check")
        .arg("--quiet")
        .current_dir(&temp)
        .env("CARGO_TARGET_DIR", temp.join("target"))
        .output()
        .expect("cargo check should run");

    let _ = std::fs::remove_dir_all(&temp);
    assert!(
        !output.status.success(),
        "downstream crate unexpectedly compiled:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    for forbidden in ["builder", "build", "issuer", "registration_identity"] {
        assert!(
            stderr.contains(forbidden),
            "compile-fail output should mention forbidden path {forbidden:?}:\n{stderr}"
        );
    }
}

fn temp_compile_crate_dir() -> PathBuf {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "tenferro-runtime-snapshot-compile-fail-{}-{unique}",
        std::process::id()
    ))
}

fn write_compile_fail_crate(dir: &Path) {
    std::fs::create_dir_all(dir.join("src")).expect("create compile-fail crate");
    std::fs::write(
        dir.join("Cargo.toml"),
        format!(
            "[package]\nname = \"snapshot_compile_fail\"\nversion = \"0.0.0\"\nedition = \"2021\"\n\n[dependencies]\ntenferro-runtime = {{ path = {:?} }}\n",
            env!("CARGO_MANIFEST_DIR")
        ),
    )
    .expect("write compile-fail manifest");
    std::fs::write(
        dir.join("src/lib.rs"),
        r#"
use tenferro_runtime::RuntimeConfigSnapshot;

pub fn forbidden_snapshot_construction(snapshot: &RuntimeConfigSnapshot) {
    let _ = RuntimeConfigSnapshot::builder();
    let _ = RuntimeConfigSnapshot::build();
    let _ = RuntimeConfigSnapshot {
        issuer: (),
        registration_identity: (),
    };
    let _ = snapshot.issuer();
    let _ = snapshot.registration_identity();
}
"#,
    )
    .expect("write compile-fail source");
}
