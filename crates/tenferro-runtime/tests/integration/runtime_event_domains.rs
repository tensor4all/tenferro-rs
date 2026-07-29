use std::sync::atomic::{AtomicUsize, Ordering};

use tenferro_runtime::runtime::{EventDomainDriver, ImmediateEventDomainDriver};

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
    assert!(completion
        .as_any()
        .is::<tenferro_runtime::runtime::ReadyEventToken>());
    run.drain()?;

    assert_eq!(launches.load(Ordering::SeqCst), 1);
    Ok(())
}
