use std::any::Any;
use std::error::Error as StdError;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use tenferro_runtime::runtime::{EventDomainDriver, EventToken, ImmediateEventDomainDriver};
use tenferro_runtime::{
    assemble_preparation_only_engine_registration, CoreCapabilityBundle, EngineId,
    EngineRegistrationMetadata, EventDomainId, ExecutionContextIdentity, HardwareClassId,
    PreparationOnlyEngineRegistrationConfig, ProviderDeviceIdentity, ProviderId, Runtime,
    StorageClass,
};

fn test_domain(suffix: &str) -> Result<EventDomainId, Box<dyn StdError>> {
    let engine_id = EngineId::new(format!("tenferro.test.event.engine.{suffix}"))?;
    let storage = StorageClass::new(format!("tenferro.test.event.storage.{suffix}"))?;
    let metadata = EngineRegistrationMetadata::new(
        engine_id.clone(),
        ProviderDeviceIdentity::new(
            ProviderId::new("tenferro.test.event")?,
            format!("target:{suffix}"),
        )?,
        HardwareClassId::new("tenferro.test.event")?,
        Arc::from(vec![storage.clone()]),
        storage,
        CoreCapabilityBundle::default(),
    );
    let registration = assemble_preparation_only_engine_registration(
        PreparationOnlyEngineRegistrationConfig::new(
            metadata,
            ExecutionContextIdentity::of::<()>(),
        ),
    )?;
    let mut builder = Runtime::builder();
    builder.register_engine(registration)?;
    let runtime = builder.build()?;
    let snapshot = runtime.snapshot()?;
    snapshot
        .engine(&engine_id)
        .map(|engine| engine.event_domain_id())
        .ok_or_else(|| "event test engine was not published".into())
}

#[derive(Debug)]
struct CountingToken {
    origin: EventDomainId,
    waits: Arc<AtomicUsize>,
}

impl EventToken for CountingToken {
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

#[derive(Debug)]
struct FailingToken(EventDomainId);

impl EventToken for FailingToken {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn origin(&self) -> EventDomainId {
        self.0
    }

    fn wait(&self) -> tenferro_runtime::Result<()> {
        Err(tenferro_runtime::Error::Internal(
            "dependency failed".to_owned(),
        ))
    }
}

#[test]
fn immediate_event_domain_launches_once_and_drains() -> Result<(), Box<dyn std::error::Error>> {
    let domain = test_domain("launch")?;
    let driver = ImmediateEventDomainDriver::new();
    let mut run = driver.begin_run(domain)?;
    let launches = AtomicUsize::new(0);
    let mut launch = || {
        launches.fetch_add(1, Ordering::SeqCst);
        Ok(())
    };

    let completion = run.enqueue(&[], &mut launch)?;
    completion.wait()?;
    run.drain()?;

    assert_eq!(launches.load(Ordering::SeqCst), 1);
    Ok(())
}

#[test]
fn immediate_event_domain_waits_for_same_domain_dependencies_before_launch(
) -> Result<(), Box<dyn std::error::Error>> {
    let domain = test_domain("same-domain")?;
    let driver = ImmediateEventDomainDriver::new();
    let mut run = driver.begin_run(domain)?;
    let waits = Arc::new(AtomicUsize::new(0));
    let dependency: Arc<dyn EventToken> = Arc::new(CountingToken {
        origin: domain,
        waits: Arc::clone(&waits),
    });
    let mut launch = || {
        assert_eq!(waits.load(Ordering::SeqCst), 1);
        Ok(())
    };

    run.enqueue(&[dependency], &mut launch)?;

    assert_eq!(waits.load(Ordering::SeqCst), 1);
    Ok(())
}

#[test]
fn immediate_completion_supports_concurrent_repeated_waits(
) -> Result<(), Box<dyn std::error::Error>> {
    let domain = test_domain("repeated-wait")?;
    let driver = ImmediateEventDomainDriver::new();
    let mut run = driver.begin_run(domain)?;
    let mut launch = || Ok(());
    let completion = run.enqueue(&[], &mut launch)?;

    let first = Arc::clone(&completion);
    let second = Arc::clone(&completion);
    let first_wait = std::thread::spawn(move || first.wait());
    let second_wait = std::thread::spawn(move || second.wait());

    first_wait.join().unwrap()?;
    second_wait.join().unwrap()?;
    completion.wait()?;
    Ok(())
}

#[test]
fn immediate_event_domain_does_not_launch_after_dependency_failure() {
    let domain = test_domain("failure").expect("event-domain test runtime");
    let driver = ImmediateEventDomainDriver::new();
    let mut run = driver.begin_run(domain).unwrap();
    let launches = AtomicUsize::new(0);
    let dependency: Arc<dyn EventToken> = Arc::new(FailingToken(domain));
    let mut launch = || {
        launches.fetch_add(1, Ordering::SeqCst);
        Ok(())
    };

    assert!(run.enqueue(&[dependency], &mut launch).is_err());
    assert_eq!(launches.load(Ordering::SeqCst), 0);
}
