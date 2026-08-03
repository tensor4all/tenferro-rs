use std::any::Any;
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use super::super::execution::{
    EventDomainRunLifecycleError, MissingScheduledDependencyCompletionError, ScheduledEventDomains,
};
use super::super::schedule::{
    EventCompletion, EventDependency, EventDomainId, EventSlotId, ExecutionLocation, ScheduledNode,
    ScheduledOperation, ScheduledTransfer,
};
use super::super::{
    EngineId, EventDomainDriver, EventDomainError, EventDomainOperation, EventDomainRun,
    EventToken, ImmediateEventDomainDriver, ProviderDeviceIdentity, ProviderId, StorageClass,
};
use crate::{Error, RegistrationIdentity, Result, RuntimeEpoch, RuntimeId};

fn qualified_domain(
    runtime_value: u64,
    epoch_value: u64,
    issuer_value: u64,
    ordinal_value: u64,
) -> EventDomainId {
    EventDomainId::runtime_created_for_test(
        RuntimeId::from_nonzero(NonZeroU64::new(runtime_value).expect("runtime id")),
        RuntimeEpoch::from_nonzero(NonZeroU64::new(epoch_value).expect("runtime epoch")),
        RegistrationIdentity::new(
            NonZeroU64::new(issuer_value).expect("registration issuer"),
            NonZeroU64::new(ordinal_value).expect("registration ordinal"),
        ),
    )
}

fn location(domain: EventDomainId, name: &str) -> ExecutionLocation {
    ExecutionLocation::new(
        EngineId::new(format!("tenferro.test.event.{name}")).expect("event test engine id"),
        ProviderDeviceIdentity::new(
            ProviderId::new("tenferro.test.event").expect("event test provider id"),
            format!("target:{name}"),
        )
        .expect("event test provider target"),
        domain,
        StorageClass::new(format!("tenferro.test.event.{name}")).expect("event test storage class"),
    )
}

#[derive(Debug)]
struct TestEventToken {
    origin: EventDomainId,
    waits: Arc<AtomicUsize>,
    fail_wait: bool,
}

impl EventToken for TestEventToken {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn origin(&self) -> EventDomainId {
        self.origin
    }

    fn wait(&self) -> Result<()> {
        self.waits.fetch_add(1, Ordering::SeqCst);
        if self.fail_wait {
            Err(Error::runtime_state(
                "event-domain-test-token",
                crate::ErrorPhase::Execution,
                "injected host wait failure",
            ))
        } else {
            Ok(())
        }
    }
}

#[derive(Clone, Debug)]
struct TestEventDomainDriver {
    run_domain: EventDomainId,
    completion_origin: EventDomainId,
    waits: Arc<AtomicUsize>,
    fail_wait: bool,
    observed_dependencies: Arc<Mutex<Vec<Vec<EventDomainId>>>>,
}

impl EventDomainDriver for TestEventDomainDriver {
    fn begin_run(&self, _requested_domain: EventDomainId) -> Result<Box<dyn EventDomainRun>> {
        Ok(Box::new(TestEventDomainRun {
            domain: self.run_domain,
            completion_origin: self.completion_origin,
            waits: Arc::clone(&self.waits),
            fail_wait: self.fail_wait,
            observed_dependencies: Arc::clone(&self.observed_dependencies),
        }))
    }
}

#[derive(Debug)]
struct TestEventDomainRun {
    domain: EventDomainId,
    completion_origin: EventDomainId,
    waits: Arc<AtomicUsize>,
    fail_wait: bool,
    observed_dependencies: Arc<Mutex<Vec<Vec<EventDomainId>>>>,
}

impl EventDomainRun for TestEventDomainRun {
    fn domain(&self) -> EventDomainId {
        self.domain
    }

    fn enqueue(
        &mut self,
        dependencies: &[Arc<dyn EventToken>],
        launch: &mut dyn FnMut() -> Result<()>,
    ) -> Result<Arc<dyn EventToken>> {
        self.observed_dependencies
            .lock()
            .expect("event dependency log lock")
            .push(dependencies.iter().map(|token| token.origin()).collect());
        for dependency in dependencies {
            if dependency.origin() != self.domain {
                return Err(Error::from(EventDomainError::DependencyDomainMismatch {
                    operation: EventDomainOperation::Enqueue,
                    node_index: None,
                    expected: self.domain,
                    actual: dependency.origin(),
                }));
            }
        }
        launch()?;
        Ok(Arc::new(TestEventToken {
            origin: self.completion_origin,
            waits: Arc::clone(&self.waits),
            fail_wait: self.fail_wait,
        }))
    }

    fn drain(&mut self) -> Result<()> {
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct ChangingEventDomainDriver {
    requested: EventDomainId,
    changed: EventDomainId,
    change_after: usize,
    domain_calls: Arc<AtomicUsize>,
    enqueue_calls: Arc<AtomicUsize>,
}

impl EventDomainDriver for ChangingEventDomainDriver {
    fn begin_run(&self, _requested_domain: EventDomainId) -> Result<Box<dyn EventDomainRun>> {
        Ok(Box::new(ChangingEventDomainRun {
            requested: self.requested,
            changed: self.changed,
            change_after: self.change_after,
            domain_calls: Arc::clone(&self.domain_calls),
            enqueue_calls: Arc::clone(&self.enqueue_calls),
        }))
    }
}

#[derive(Debug)]
struct ChangingEventDomainRun {
    requested: EventDomainId,
    changed: EventDomainId,
    change_after: usize,
    domain_calls: Arc<AtomicUsize>,
    enqueue_calls: Arc<AtomicUsize>,
}

impl EventDomainRun for ChangingEventDomainRun {
    fn domain(&self) -> EventDomainId {
        if self.domain_calls.fetch_add(1, Ordering::SeqCst) < self.change_after {
            self.requested
        } else {
            self.changed
        }
    }

    fn enqueue(
        &mut self,
        _dependencies: &[Arc<dyn EventToken>],
        launch: &mut dyn FnMut() -> Result<()>,
    ) -> Result<Arc<dyn EventToken>> {
        self.enqueue_calls.fetch_add(1, Ordering::SeqCst);
        launch()?;
        Ok(Arc::new(TestEventToken {
            origin: self.requested,
            waits: Arc::new(AtomicUsize::new(0)),
            fail_wait: false,
        }))
    }

    fn drain(&mut self) -> Result<()> {
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
enum ProbeDrainBehavior {
    Return,
    ReturnError,
    PanicStatic,
    PanicNonString,
}

#[derive(Clone, Copy, Debug)]
enum ProbeDropBehavior {
    Return,
    PanicStatic,
}

#[derive(Clone, Debug)]
struct DrainProbeDriver {
    label: &'static str,
    drain_behavior: ProbeDrainBehavior,
    drop_behavior: ProbeDropBehavior,
    events: Arc<Mutex<Vec<String>>>,
}

impl EventDomainDriver for DrainProbeDriver {
    fn begin_run(&self, domain: EventDomainId) -> Result<Box<dyn EventDomainRun>> {
        Ok(Box::new(DrainProbeRun {
            label: self.label,
            domain,
            drain_behavior: self.drain_behavior,
            drop_behavior: self.drop_behavior,
            events: Arc::clone(&self.events),
        }))
    }
}

#[derive(Debug)]
struct DrainProbeRun {
    label: &'static str,
    domain: EventDomainId,
    drain_behavior: ProbeDrainBehavior,
    drop_behavior: ProbeDropBehavior,
    events: Arc<Mutex<Vec<String>>>,
}

impl EventDomainRun for DrainProbeRun {
    fn domain(&self) -> EventDomainId {
        self.domain
    }

    fn enqueue(
        &mut self,
        _dependencies: &[Arc<dyn EventToken>],
        launch: &mut dyn FnMut() -> Result<()>,
    ) -> Result<Arc<dyn EventToken>> {
        launch()?;
        Ok(Arc::new(TestEventToken {
            origin: self.domain,
            waits: Arc::new(AtomicUsize::new(0)),
            fail_wait: false,
        }))
    }

    fn drain(&mut self) -> Result<()> {
        self.events
            .lock()
            .expect("drain probe event log lock")
            .push(format!("{}:drain", self.label));
        match self.drain_behavior {
            ProbeDrainBehavior::Return => Ok(()),
            ProbeDrainBehavior::ReturnError => Err(Error::runtime_state(
                "event-domain-test",
                crate::ErrorPhase::Execution,
                format!("{} drain failure", self.label),
            )),
            ProbeDrainBehavior::PanicStatic => panic!("{} drain panic", self.label),
            ProbeDrainBehavior::PanicNonString => std::panic::panic_any(42_u8),
        }
    }
}

impl Drop for DrainProbeRun {
    fn drop(&mut self) {
        self.events
            .lock()
            .expect("drain probe event log lock")
            .push(format!("{}:drop", self.label));
        match self.drop_behavior {
            ProbeDropBehavior::Return => {}
            ProbeDropBehavior::PanicStatic => panic!("{} drop panic", self.label),
        }
    }
}

type TestDriver = (
    Arc<dyn EventDomainDriver>,
    Arc<AtomicUsize>,
    Arc<Mutex<Vec<Vec<EventDomainId>>>>,
);

fn driver(run_domain: EventDomainId, completion_origin: EventDomainId) -> TestDriver {
    let waits = Arc::new(AtomicUsize::new(0));
    let observed_dependencies = Arc::new(Mutex::new(Vec::new()));
    (
        Arc::new(TestEventDomainDriver {
            run_domain,
            completion_origin,
            waits: Arc::clone(&waits),
            fail_wait: false,
            observed_dependencies: Arc::clone(&observed_dependencies),
        }),
        waits,
        observed_dependencies,
    )
}

fn operation(
    domain: EventDomainId,
    slot: u32,
    dependencies: impl Into<Box<[EventDependency]>>,
) -> ScheduledNode {
    ScheduledNode::Operation(ScheduledOperation::new(
        0,
        location(domain, "operation"),
        [],
        [],
        dependencies,
        EventCompletion::new(domain, EventSlotId::new(slot), 0),
    ))
}

fn transfer(
    source: EventDomainId,
    destination: EventDomainId,
    completion_slot: u32,
    dependencies: impl Into<Box<[EventDependency]>>,
) -> ScheduledNode {
    ScheduledNode::Transfer(ScheduledTransfer::new(
        0,
        location(source, "source"),
        location(destination, "destination"),
        dependencies,
        EventCompletion::new(destination, EventSlotId::new(completion_slot), 0),
    ))
}

#[test]
fn event_domain_is_unique_across_runtime_epoch_and_registration_provenance() {
    let first = qualified_domain(1, 1, 1, 1);
    let next_epoch = qualified_domain(1, 2, 1, 1);
    let next_runtime = qualified_domain(2, 1, 1, 1);
    let next_registration = qualified_domain(1, 1, 1, 2);

    assert_ne!(first, next_epoch);
    assert_ne!(first, next_runtime);
    assert_ne!(first, next_registration);
    assert!(first < next_epoch || next_epoch < first);
    assert_eq!(
        first.runtime_id(),
        RuntimeId::from_nonzero(NonZeroU64::new(1).expect("runtime"))
    );
    assert_eq!(
        first.epoch(),
        RuntimeEpoch::from_nonzero(NonZeroU64::new(1).expect("epoch"))
    );
}

#[test]
fn event_domain_operation_display_and_immediate_token_access_are_complete() -> Result<()> {
    for (operation, expected) in [
        (EventDomainOperation::BeginRun, "begin run"),
        (EventDomainOperation::Enqueue, "enqueue"),
        (EventDomainOperation::Drain, "drain"),
        (EventDomainOperation::TransferBridge, "transfer bridge"),
        (
            EventDomainOperation::ValidateCompletion,
            "validate completion",
        ),
    ] {
        assert_eq!(operation.to_string(), expected);
    }

    let domain = qualified_domain(1, 1, 1, 1);
    let mut run = ImmediateEventDomainDriver::new().begin_run(domain)?;
    let mut launch = || Ok(());
    let completion = run.enqueue(&[], &mut launch)?;
    assert_eq!(completion.origin(), domain);
    let token_data = Arc::as_ptr(&completion).cast::<()>();
    let any_data = std::ptr::from_ref(completion.as_any()).cast::<()>();
    assert_eq!(
        token_data, any_data,
        "as_any must expose the completion token itself"
    );
    Ok(())
}

#[test]
fn immediate_driver_rejects_foreign_tokens_without_launching() -> Result<()> {
    let destination = qualified_domain(1, 1, 1, 1);
    let foreign = qualified_domain(2, 1, 1, 1);
    let driver = ImmediateEventDomainDriver::new();
    let mut run = driver.begin_run(destination)?;
    assert_eq!(run.domain(), destination);

    let launches = Arc::new(AtomicUsize::new(0));
    let waits = Arc::new(AtomicUsize::new(0));
    let dependency: Arc<dyn EventToken> = Arc::new(TestEventToken {
        origin: foreign,
        waits: Arc::clone(&waits),
        fail_wait: false,
    });
    let launches_for_closure = Arc::clone(&launches);
    let mut launch = move || {
        launches_for_closure.fetch_add(1, Ordering::SeqCst);
        Ok(())
    };

    let error = run
        .enqueue(&[dependency], &mut launch)
        .expect_err("foreign token");
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

#[test]
fn scheduler_rejects_foreign_operation_dependencies_before_launch() -> Result<()> {
    let foreign = qualified_domain(1, 1, 1, 1);
    let destination = qualified_domain(1, 1, 1, 2);
    let (foreign_driver, _, _) = driver(foreign, foreign);
    let (destination_driver, _, destination_dependencies) = driver(destination, destination);
    let mut scheduler = ScheduledEventDomains::for_test(vec![
        (foreign, foreign_driver),
        (destination, destination_driver),
    ])?;

    let mut noop = || Ok(());
    scheduler.enqueue(0, &operation(foreign, 0, []), &mut noop)?;
    let foreign_dependency = EventDependency::new(foreign, EventSlotId::new(0), 0);
    let launches = Arc::new(AtomicUsize::new(0));
    let launches_for_closure = Arc::clone(&launches);
    let mut launch = move || {
        launches_for_closure.fetch_add(1, Ordering::SeqCst);
        Ok(())
    };

    let error = scheduler
        .enqueue(
            1,
            &operation(destination, 1, [foreign_dependency]),
            &mut launch,
        )
        .expect_err("foreign operation dependency");
    assert!(matches!(
        error,
        Error::EventDomain {
            source: EventDomainError::DependencyDomainMismatch {
                operation: EventDomainOperation::Enqueue,
                expected,
                actual,
                node_index: Some(1),
                ..
            }
        } if expected == destination && actual == foreign
    ));
    assert_eq!(launches.load(Ordering::SeqCst), 0);
    assert!(destination_dependencies
        .lock()
        .expect("destination dependency log lock")
        .is_empty());
    Ok(())
}

#[test]
fn scheduler_rejects_a_run_that_reports_another_domain() -> Result<()> {
    let requested = qualified_domain(1, 1, 1, 1);
    let actual = qualified_domain(1, 1, 1, 2);
    let (driver, _, _) = driver(requested, requested);
    let mismatching = Arc::new(TestEventDomainDriver {
        run_domain: actual,
        completion_origin: actual,
        waits: Arc::new(AtomicUsize::new(0)),
        fail_wait: false,
        observed_dependencies: Arc::new(Mutex::new(Vec::new())),
    });

    let error = ScheduledEventDomains::for_test(vec![(requested, mismatching)])
        .expect_err("run-domain mismatch");
    assert!(matches!(
        error,
        Error::EventDomain {
            source: EventDomainError::RunDomainMismatch {
                operation: EventDomainOperation::BeginRun,
                expected,
                actual: reported,
                ..
            }
        } if expected == requested && reported == actual
    ));
    drop(driver);
    Ok(())
}

#[test]
fn scheduler_reports_missing_event_domain_driver_with_domain() -> Result<()> {
    let registered = qualified_domain(1, 1, 1, 1);
    let missing = qualified_domain(1, 1, 1, 2);
    let (driver, _, _) = driver(registered, registered);
    let mut scheduler = ScheduledEventDomains::for_test(vec![(registered, driver)])?;
    let mut launches = 0;
    let mut launch = || {
        launches += 1;
        Ok(())
    };

    let error = scheduler
        .enqueue(0, &operation(missing, 0, []), &mut launch)
        .expect_err("missing event-domain driver");
    assert!(matches!(
        error,
        Error::EventDomain {
            source: EventDomainError::MissingDriver { domain }
        } if domain == missing
    ));
    assert_eq!(launches, 0);
    Ok(())
}

#[test]
fn scheduler_rejects_forged_completion_after_enqueue_before_recording_or_downstream_launch(
) -> Result<()> {
    let domain = qualified_domain(1, 1, 1, 1);
    let forged = qualified_domain(2, 1, 1, 1);
    let (driver, _, _) = driver(domain, forged);
    let mut scheduler = ScheduledEventDomains::for_test(vec![(domain, driver)])?;
    let launches = Arc::new(AtomicUsize::new(0));
    let launches_for_closure = Arc::clone(&launches);
    let mut launch = move || {
        launches_for_closure.fetch_add(1, Ordering::SeqCst);
        Ok(())
    };

    let error = scheduler
        .enqueue(0, &operation(domain, 0, []), &mut launch)
        .expect_err("forged completion origin");
    assert!(matches!(
        error,
        Error::EventDomain {
            source: EventDomainError::CompletionTokenDomainMismatch {
                operation: EventDomainOperation::ValidateCompletion,
                expected,
                actual,
                node_index: Some(0),
                ..
            }
        } if expected == domain && actual == forged
    ));
    assert_eq!(launches.load(Ordering::SeqCst), 1);
    let downstream_launches = Arc::new(AtomicUsize::new(0));
    let downstream_launches_for_closure = Arc::clone(&downstream_launches);
    let mut downstream_launch = move || {
        downstream_launches_for_closure.fetch_add(1, Ordering::SeqCst);
        Ok(())
    };
    let downstream_error = scheduler
        .enqueue(
            1,
            &operation(
                domain,
                1,
                [EventDependency::new(domain, EventSlotId::new(0), 0)],
            ),
            &mut downstream_launch,
        )
        .expect_err("forged completion must not be available downstream");
    let Error::RuntimeStateSource { source, .. } = downstream_error else {
        panic!("missing scheduled completion must retain a typed source");
    };
    let missing = source
        .downcast_ref::<MissingScheduledDependencyCompletionError>()
        .expect("typed missing scheduled completion source");
    assert_eq!(
        missing.dependency,
        EventDependency::new(domain, EventSlotId::new(0), 0)
    );
    assert_eq!(missing.node_index, 1);
    assert_eq!(downstream_launches.load(Ordering::SeqCst), 0);
    Ok(())
}

#[test]
fn scheduler_contains_drain_and_box_drop_panics_and_drains_later_runs() -> Result<()> {
    let first = qualified_domain(1, 1, 1, 1);
    let second = qualified_domain(1, 1, 1, 2);
    let events = Arc::new(Mutex::new(Vec::new()));
    let first_driver: Arc<dyn EventDomainDriver> = Arc::new(DrainProbeDriver {
        label: "first",
        drain_behavior: ProbeDrainBehavior::PanicStatic,
        drop_behavior: ProbeDropBehavior::PanicStatic,
        events: Arc::clone(&events),
    });
    let second_driver: Arc<dyn EventDomainDriver> = Arc::new(DrainProbeDriver {
        label: "second",
        drain_behavior: ProbeDrainBehavior::Return,
        drop_behavior: ProbeDropBehavior::PanicStatic,
        events: Arc::clone(&events),
    });

    let error;
    {
        let mut scheduler =
            ScheduledEventDomains::for_test(vec![(first, first_driver), (second, second_driver)])?;
        error = scheduler
            .drain()
            .expect_err("the first drain panic must become a typed cleanup error");
    }

    assert!(matches!(
        error,
        Error::EventDomain {
            source: EventDomainError::DrainPanicked {
                operation: EventDomainOperation::Drain,
                domain,
                message,
            }
        } if domain == first && message == "first drain panic"
    ));
    let events = events.lock().expect("drain probe event log lock");
    assert!(events.iter().any(|event| event == "first:drain"));
    assert!(events.iter().any(|event| event == "second:drain"));
    assert!(events.iter().any(|event| event == "first:drop"));
    assert!(events.iter().any(|event| event == "second:drop"));
    Ok(())
}

#[test]
fn scheduler_run_drain_is_terminal_and_does_not_call_provider_again() -> Result<()> {
    for (label, drain_behavior, expected_first_error) in [
        ("retired", ProbeDrainBehavior::Return, false),
        ("failed", ProbeDrainBehavior::ReturnError, true),
        ("panicked", ProbeDrainBehavior::PanicStatic, true),
    ] {
        let domain = qualified_domain(1, 1, 1, label.len() as u64);
        let events = Arc::new(Mutex::new(Vec::new()));
        let driver: Arc<dyn EventDomainDriver> = Arc::new(DrainProbeDriver {
            label,
            drain_behavior,
            drop_behavior: ProbeDropBehavior::Return,
            events: Arc::clone(&events),
        });
        let mut scheduler = ScheduledEventDomains::for_test(vec![(domain, driver)])?;

        let first = scheduler.drain();
        assert_eq!(first.is_err(), expected_first_error);

        let second = scheduler
            .drain()
            .expect_err("a terminal run must reject a second drain");
        assert!(matches!(
            &second,
            Error::RuntimeStateSource { source, .. }
                if source
                    .downcast_ref::<EventDomainRunLifecycleError>()
                    .is_some()
        ));

        let events = events.lock().expect("drain probe event log lock");
        assert_eq!(
            events
                .iter()
                .filter(|event| *event == &format!("{label}:drain"))
                .count(),
            1,
            "provider drain must run once for {label}"
        );
    }
    Ok(())
}

#[test]
fn scheduler_drain_returns_all_failures_in_run_order() -> Result<()> {
    let first = qualified_domain(1, 1, 1, 1);
    let second = qualified_domain(1, 1, 1, 2);
    let third = qualified_domain(1, 1, 1, 3);
    let events = Arc::new(Mutex::new(Vec::new()));
    let drivers: Vec<(EventDomainId, Arc<dyn EventDomainDriver>)> = [
        ("first", first, ProbeDrainBehavior::ReturnError),
        ("second", second, ProbeDrainBehavior::PanicStatic),
        ("third", third, ProbeDrainBehavior::ReturnError),
    ]
    .into_iter()
    .map(|(label, domain, drain_behavior)| {
        (
            domain,
            Arc::new(DrainProbeDriver {
                label,
                drain_behavior,
                drop_behavior: ProbeDropBehavior::Return,
                events: Arc::clone(&events),
            }) as Arc<dyn EventDomainDriver>,
        )
    })
    .collect();
    let mut scheduler = ScheduledEventDomains::for_test(drivers)?;

    let error = scheduler
        .drain()
        .expect_err("all injected cleanup failures must be returned");
    let message = error.to_string();
    let first_failure = message.find("first drain failure").expect("first failure");
    let second_failure = message.find("second drain panic").expect("second failure");
    let third_failure = message.find("third drain failure").expect("third failure");
    assert!(first_failure < second_failure && second_failure < third_failure);

    let events = events.lock().expect("drain probe event log lock");
    let drain_events: Vec<_> = events
        .iter()
        .filter(|event| event.ends_with(":drain"))
        .cloned()
        .collect();
    assert_eq!(
        drain_events,
        [
            "first:drain".to_owned(),
            "second:drain".to_owned(),
            "third:drain".to_owned(),
        ]
    );
    Ok(())
}

#[test]
fn scheduler_reports_non_string_drain_panic_with_safe_message() -> Result<()> {
    let domain = qualified_domain(1, 1, 1, 1);
    let events = Arc::new(Mutex::new(Vec::new()));
    let driver: Arc<dyn EventDomainDriver> = Arc::new(DrainProbeDriver {
        label: "non-string-payload",
        drain_behavior: ProbeDrainBehavior::PanicNonString,
        drop_behavior: ProbeDropBehavior::Return,
        events,
    });
    let mut scheduler = ScheduledEventDomains::for_test(vec![(domain, driver)])?;

    let error = scheduler
        .drain()
        .expect_err("non-string drain panic must become a typed cleanup error");
    assert!(matches!(
        error,
        Error::EventDomain {
            source: EventDomainError::DrainPanicked { message, .. }
        } if message == "non-string panic payload"
    ));
    Ok(())
}

#[test]
fn scheduler_waits_and_filters_cross_domain_transfer_dependencies() -> Result<()> {
    let source = qualified_domain(1, 1, 1, 1);
    let destination = qualified_domain(1, 1, 1, 2);
    let (source_driver, source_waits, _) = driver(source, source);
    let (destination_driver, _, destination_dependencies) = driver(destination, destination);
    let mut scheduler = ScheduledEventDomains::for_test(vec![
        (source, source_driver),
        (destination, destination_driver),
    ])?;

    let mut noop = || Ok(());
    scheduler.enqueue(0, &operation(source, 0, []), &mut noop)?;
    let source_dependency = EventDependency::new(source, EventSlotId::new(0), 0);
    let mut transfer_launches = 0;
    let mut transfer_launch = || {
        transfer_launches += 1;
        Ok(())
    };
    scheduler.enqueue(
        1,
        &transfer(source, destination, 1, [source_dependency]),
        &mut transfer_launch,
    )?;

    assert_eq!(source_waits.load(Ordering::SeqCst), 1);
    assert_eq!(transfer_launches, 1);
    assert_eq!(
        destination_dependencies
            .lock()
            .expect("destination dependency log lock")
            .as_slice(),
        &[Vec::new()]
    );
    Ok(())
}

#[test]
fn scheduler_keeps_same_destination_dependencies_and_waits_for_each_fanout() -> Result<()> {
    let source = qualified_domain(1, 1, 1, 1);
    let destination = qualified_domain(1, 1, 1, 2);
    let (source_driver, source_waits, _) = driver(source, source);
    let (destination_driver, _, destination_dependencies) = driver(destination, destination);
    let mut scheduler = ScheduledEventDomains::for_test(vec![
        (source, source_driver),
        (destination, destination_driver),
    ])?;

    let mut noop = || Ok(());
    scheduler.enqueue(0, &operation(source, 0, []), &mut noop)?;
    scheduler.enqueue(1, &operation(destination, 1, []), &mut noop)?;
    let source_dependency = EventDependency::new(source, EventSlotId::new(0), 0);
    let destination_dependency = EventDependency::new(destination, EventSlotId::new(1), 0);
    scheduler.enqueue(
        2,
        &transfer(
            source,
            destination,
            2,
            [source_dependency, destination_dependency],
        ),
        &mut noop,
    )?;
    scheduler.enqueue(
        3,
        &transfer(source, destination, 3, [source_dependency]),
        &mut noop,
    )?;

    assert_eq!(source_waits.load(Ordering::SeqCst), 2);
    assert_eq!(
        destination_dependencies
            .lock()
            .expect("destination dependency log lock")
            .as_slice(),
        &[Vec::new(), vec![destination], Vec::new()]
    );
    Ok(())
}

#[test]
fn scheduler_rejects_third_domain_and_wait_failures_before_destination_launch() -> Result<()> {
    let source = qualified_domain(1, 1, 1, 1);
    let destination = qualified_domain(1, 1, 1, 2);
    let third = qualified_domain(1, 1, 1, 3);
    let (third_driver, _, _) = driver(third, third);
    let (destination_driver, _, destination_dependencies) = driver(destination, destination);
    let mut scheduler = ScheduledEventDomains::for_test(vec![
        (third, third_driver),
        (destination, destination_driver),
    ])?;
    let mut noop = || Ok(());
    scheduler.enqueue(0, &operation(third, 0, []), &mut noop)?;
    let third_dependency = EventDependency::new(third, EventSlotId::new(0), 0);
    let error = scheduler
        .enqueue(
            1,
            &transfer(source, destination, 1, [third_dependency]),
            &mut noop,
        )
        .expect_err("third-domain token");
    assert!(matches!(
        error,
        Error::EventDomain {
            source: EventDomainError::DependencyDomainMismatch {
                operation: EventDomainOperation::TransferBridge,
                expected,
                actual,
                node_index: Some(1),
                ..
            }
        } if expected == source && actual == third
    ));
    assert!(destination_dependencies
        .lock()
        .expect("destination dependency log lock")
        .is_empty());

    let waits = Arc::new(AtomicUsize::new(0));
    let failing_source_driver: Arc<dyn EventDomainDriver> = Arc::new(TestEventDomainDriver {
        run_domain: source,
        completion_origin: source,
        waits: Arc::clone(&waits),
        fail_wait: true,
        observed_dependencies: Arc::new(Mutex::new(Vec::new())),
    });
    let (destination_driver, _, destination_dependencies) = driver(destination, destination);
    let mut scheduler = ScheduledEventDomains::for_test(vec![
        (source, failing_source_driver),
        (destination, destination_driver),
    ])?;
    scheduler.enqueue(0, &operation(source, 0, []), &mut noop)?;
    let source_dependency = EventDependency::new(source, EventSlotId::new(0), 0);
    let error = scheduler
        .enqueue(
            1,
            &transfer(source, destination, 1, [source_dependency]),
            &mut noop,
        )
        .expect_err("host wait failure");
    assert!(matches!(
        error,
        Error::EventDomain {
            source: EventDomainError::DependencyWaitFailed {
                operation: EventDomainOperation::TransferBridge,
                expected,
                actual,
                node_index: Some(1),
                ..
            }
        } if expected == destination && actual == source
    ));
    assert_eq!(waits.load(Ordering::SeqCst), 1);
    assert!(destination_dependencies
        .lock()
        .expect("destination dependency log lock")
        .is_empty());
    Ok(())
}

#[test]
fn scheduler_rechecks_run_domain_before_forwarding_dependencies_or_launch() -> Result<()> {
    let requested = qualified_domain(1, 1, 1, 1);
    let changed = qualified_domain(1, 1, 1, 2);
    let domain_calls = Arc::new(AtomicUsize::new(0));
    let enqueue_calls = Arc::new(AtomicUsize::new(0));
    let driver: Arc<dyn EventDomainDriver> = Arc::new(ChangingEventDomainDriver {
        requested,
        changed,
        change_after: 2,
        domain_calls: Arc::clone(&domain_calls),
        enqueue_calls: Arc::clone(&enqueue_calls),
    });
    let mut scheduler = ScheduledEventDomains::for_test(vec![(requested, driver)])?;
    let launches = Arc::new(AtomicUsize::new(0));
    let launches_for_closure = Arc::clone(&launches);
    let mut launch = move || {
        launches_for_closure.fetch_add(1, Ordering::SeqCst);
        Ok(())
    };

    let error = scheduler
        .enqueue(0, &operation(requested, 0, []), &mut launch)
        .expect_err("run changed domain after preflight");
    assert!(matches!(
        error,
        Error::EventDomain {
            source: EventDomainError::RunDomainMismatch {
                operation: EventDomainOperation::Enqueue,
                node_index: Some(0),
                expected,
                actual,
            }
        } if expected == requested && actual == changed
    ));
    assert_eq!(domain_calls.load(Ordering::SeqCst), 3);
    assert_eq!(enqueue_calls.load(Ordering::SeqCst), 0);
    assert_eq!(launches.load(Ordering::SeqCst), 0);
    Ok(())
}

#[test]
fn scheduler_rejects_changed_run_before_transfer_host_wait() -> Result<()> {
    let source = qualified_domain(1, 1, 1, 1);
    let destination = qualified_domain(1, 1, 1, 2);
    let changed = qualified_domain(1, 1, 1, 3);
    let (source_driver, source_waits, _) = driver(source, source);
    let destination_domain_calls = Arc::new(AtomicUsize::new(0));
    let destination_enqueue_calls = Arc::new(AtomicUsize::new(0));
    let destination_driver: Arc<dyn EventDomainDriver> = Arc::new(ChangingEventDomainDriver {
        requested: destination,
        changed,
        change_after: 1,
        domain_calls: Arc::clone(&destination_domain_calls),
        enqueue_calls: Arc::clone(&destination_enqueue_calls),
    });
    let mut scheduler = ScheduledEventDomains::for_test(vec![
        (source, source_driver),
        (destination, destination_driver),
    ])?;
    let mut noop = || Ok(());
    scheduler.enqueue(0, &operation(source, 0, []), &mut noop)?;
    let source_dependency = EventDependency::new(source, EventSlotId::new(0), 0);
    let mut launches = 0;
    let mut transfer_launch = || {
        launches += 1;
        Ok(())
    };

    let error = scheduler
        .enqueue(
            1,
            &transfer(source, destination, 1, [source_dependency]),
            &mut transfer_launch,
        )
        .expect_err("run changed before transfer bridge");
    assert!(matches!(
        error,
        Error::EventDomain {
            source: EventDomainError::RunDomainMismatch {
                operation: EventDomainOperation::Enqueue,
                node_index: Some(1),
                expected,
                actual,
            }
        } if expected == destination && actual == changed
    ));
    assert_eq!(destination_domain_calls.load(Ordering::SeqCst), 2);
    assert_eq!(destination_enqueue_calls.load(Ordering::SeqCst), 0);
    assert_eq!(source_waits.load(Ordering::SeqCst), 0);
    assert_eq!(launches, 0);
    Ok(())
}
