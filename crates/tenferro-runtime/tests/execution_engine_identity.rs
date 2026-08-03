use std::any::Any;
use std::error::Error as StdError;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use tenferro_runtime::{
    assemble_preparation_only_engine_registration, CoreCapabilityBundle, EngineId,
    EngineRegistration, EngineRegistrationMetadata, Error, EventDomainDriver, EventDomainError,
    EventDomainId, EventDomainOperation, EventToken, ExecutionContextIdentity, HardwareClassId,
    ImmediateEventDomainDriver, PreparationOnlyEngineRegistrationConfig, ProviderDeviceIdentity,
    ProviderId, RegistrationKey, Runtime, RuntimeConfigBuilder, RuntimeConfigError, StorageClass,
    TransferEndpoint, TransferProvider, TransferRequest,
};

#[derive(Debug)]
struct TestProviderContext;

fn registration(
    engine_id: EngineId,
    target: &str,
    storage: StorageClass,
) -> Result<EngineRegistration, RuntimeConfigError> {
    let metadata = EngineRegistrationMetadata::new(
        engine_id,
        ProviderDeviceIdentity::new(ProviderId::new("tenferro.test.phase0")?, target)?,
        HardwareClassId::new("tenferro.test.phase0")?,
        Arc::from(vec![storage.clone()]),
        storage,
        CoreCapabilityBundle::default(),
    );
    assemble_preparation_only_engine_registration(PreparationOnlyEngineRegistrationConfig::new(
        metadata,
        ExecutionContextIdentity::of::<TestProviderContext>(),
    ))
}

struct TwoEngineFixture {
    builder: RuntimeConfigBuilder,
    first_id: EngineId,
    second_id: EngineId,
    third_id: EngineId,
    storage: StorageClass,
}

fn two_engine_fixture() -> Result<TwoEngineFixture, RuntimeConfigError> {
    let first_id = EngineId::new("tenferro.test.phase0.device0")?;
    let second_id = EngineId::new("tenferro.test.phase0.device1")?;
    let third_id = EngineId::new("tenferro.test.phase0.device2")?;
    let storage = StorageClass::new("tenferro.test.phase0.storage")?;
    let mut builder = Runtime::builder();
    builder.register_engine(registration(first_id.clone(), "device:0", storage.clone())?)?;
    builder.register_engine(registration(
        second_id.clone(),
        "device:1",
        storage.clone(),
    )?)?;
    builder.register_engine(registration(third_id.clone(), "device:2", storage.clone())?)?;
    Ok(TwoEngineFixture {
        builder,
        first_id,
        second_id,
        third_id,
        storage,
    })
}

#[derive(Debug)]
struct UnusedTransferProvider;

impl TransferProvider for UnusedTransferProvider {
    fn transfer_blocking(
        &self,
        _request: TransferRequest<'_>,
    ) -> tenferro_runtime::Result<tenferro_tensor::Tensor> {
        Err(Error::Internal(
            "the phase-0 routing contract does not execute transfers".to_owned(),
        ))
    }
}

#[derive(Debug)]
struct ForeignToken {
    origin: EventDomainId,
    waits: Arc<AtomicUsize>,
}

impl EventToken for ForeignToken {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn origin(&self) -> EventDomainId {
        self.origin
    }

    fn wait(&self) -> tenferro_runtime::Result<()> {
        self.waits.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

#[test]
fn caller_selected_engines_have_distinct_physical_and_event_identity(
) -> Result<(), Box<dyn StdError>> {
    let mut fixture = two_engine_fixture()?;
    let error = fixture
        .builder
        .register_engine(registration(
            fixture.first_id.clone(),
            "replacement-device",
            fixture.storage.clone(),
        )?)
        .expect_err("a conflicting engine ID must be rejected");
    assert!(matches!(
        error,
        RuntimeConfigError::DuplicateEngine { engine_id } if engine_id == fixture.first_id
    ));

    let runtime = fixture.builder.build()?;
    let snapshot = runtime.snapshot()?;
    let first = snapshot
        .engine(&fixture.first_id)
        .expect("first caller-selected engine");
    let second = snapshot
        .engine(&fixture.second_id)
        .expect("second caller-selected engine");
    let third = snapshot
        .engine(&fixture.third_id)
        .expect("third caller-selected engine");

    assert_eq!(first.engine_id(), &fixture.first_id);
    assert_eq!(second.engine_id(), &fixture.second_id);
    assert_eq!(third.engine_id(), &fixture.third_id);
    assert_ne!(
        first.provider_device_identity(),
        second.provider_device_identity()
    );
    assert_ne!(first.event_domain_id(), second.event_domain_id());
    assert_ne!(first.event_domain_id(), third.event_domain_id());
    assert_ne!(second.event_domain_id(), third.event_domain_id());
    Ok(())
}

#[test]
fn transfer_routes_are_keyed_by_the_complete_endpoint_pair() -> Result<(), Box<dyn StdError>> {
    let mut fixture = two_engine_fixture()?;
    let first = TransferEndpoint::new(fixture.first_id.clone(), fixture.storage.clone());
    let second = TransferEndpoint::new(fixture.second_id.clone(), fixture.storage.clone());
    let third = TransferEndpoint::new(fixture.third_id.clone(), fixture.storage.clone());
    fixture.builder.register_transfer_provider(
        first.clone(),
        second.clone(),
        Arc::new(UnusedTransferProvider),
    )?;
    fixture.builder.register_transfer_provider(
        first.clone(),
        third.clone(),
        Arc::new(UnusedTransferProvider),
    )?;
    fixture.builder.register_transfer_provider(
        second.clone(),
        first.clone(),
        Arc::new(UnusedTransferProvider),
    )?;
    fixture.builder.register_transfer_provider(
        third,
        first.clone(),
        Arc::new(UnusedTransferProvider),
    )?;

    let error = fixture
        .builder
        .register_transfer_provider(
            first.clone(),
            second.clone(),
            Arc::new(UnusedTransferProvider),
        )
        .expect_err("only an exact duplicate endpoint pair must conflict");
    assert!(matches!(
        error,
        RuntimeConfigError::ConflictingRegistration {
            key: RegistrationKey::TransferProvider {
                source,
                destination,
            }
        } if source == first && destination == second
    ));

    let runtime = fixture.builder.build()?;
    assert_eq!(runtime.snapshot()?.transfer_provider_count(), 4);
    Ok(())
}

#[test]
fn event_domain_rejects_a_foreign_dependency_before_launch() -> Result<(), Box<dyn StdError>> {
    let fixture = two_engine_fixture()?;
    let runtime = fixture.builder.build()?;
    let snapshot = runtime.snapshot()?;
    let destination = snapshot
        .engine(&fixture.first_id)
        .expect("destination engine")
        .event_domain_id();
    let foreign = snapshot
        .engine(&fixture.second_id)
        .expect("foreign engine")
        .event_domain_id();
    let waits = Arc::new(AtomicUsize::new(0));
    let dependency: Arc<dyn EventToken> = Arc::new(ForeignToken {
        origin: foreign,
        waits: Arc::clone(&waits),
    });
    let launches = AtomicUsize::new(0);
    let mut launch = || {
        launches.fetch_add(1, Ordering::SeqCst);
        Ok(())
    };
    let mut run = ImmediateEventDomainDriver::new().begin_run(destination)?;

    let error = run
        .enqueue(&[dependency], &mut launch)
        .expect_err("foreign event dependency must be rejected");
    assert!(matches!(
        error,
        Error::EventDomain {
            source: EventDomainError::DependencyDomainMismatch {
                operation: EventDomainOperation::Enqueue,
                expected,
                actual,
                ..
            }
        } if expected == destination && actual == foreign
    ));
    assert_eq!(launches.load(Ordering::SeqCst), 0);
    assert_eq!(waits.load(Ordering::SeqCst), 0);
    Ok(())
}
