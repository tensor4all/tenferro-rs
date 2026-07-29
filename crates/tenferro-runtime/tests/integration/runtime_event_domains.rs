use std::any::Any;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use tenferro_runtime::runtime::{EventDomainDriver, EventToken, ImmediateEventDomainDriver};

#[derive(Debug)]
struct CountingToken(Arc<AtomicUsize>);

impl EventToken for CountingToken {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn wait(&self) -> tenferro_runtime::Result<()> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

#[derive(Debug)]
struct FailingToken;

impl EventToken for FailingToken {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn wait(&self) -> tenferro_runtime::Result<()> {
        Err(tenferro_runtime::Error::Internal(
            "dependency failed".to_owned(),
        ))
    }
}

#[test]
fn immediate_event_domain_launches_once_and_drains() -> Result<(), Box<dyn std::error::Error>> {
    let driver = ImmediateEventDomainDriver::new();
    let mut run = driver.begin_run()?;
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
fn immediate_event_domain_waits_for_foreign_dependencies_before_launch(
) -> Result<(), Box<dyn std::error::Error>> {
    let driver = ImmediateEventDomainDriver::new();
    let mut run = driver.begin_run()?;
    let waits = Arc::new(AtomicUsize::new(0));
    let dependency: Arc<dyn EventToken> = Arc::new(CountingToken(Arc::clone(&waits)));
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
    let driver = ImmediateEventDomainDriver::new();
    let mut run = driver.begin_run()?;
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
    let driver = ImmediateEventDomainDriver::new();
    let mut run = driver.begin_run().unwrap();
    let launches = AtomicUsize::new(0);
    let dependency: Arc<dyn EventToken> = Arc::new(FailingToken);
    let mut launch = || {
        launches.fetch_add(1, Ordering::SeqCst);
        Ok(())
    };

    assert!(run.enqueue(&[dependency], &mut launch).is_err());
    assert_eq!(launches.load(Ordering::SeqCst), 0);
}
