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
