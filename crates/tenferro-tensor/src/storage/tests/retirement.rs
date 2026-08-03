use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crate::{AllocationDomainId, AllocationId, BackendId};

use super::super::retirement::{
    AdmissionDecision, AdmissionError, EventCompletion, PreparedPackage, ProviderContext,
    ProviderEvent, ProviderRetirementBinding, RetirementError, RetirementOutcome,
};
use super::super::{
    import_unique_root, AllocationKey, BackendAllocation, ProviderCapabilities, ProviderKind,
    RootResourceExtent,
};

#[derive(Debug)]
struct DropAllocation {
    extent: RootResourceExtent,
    drops: Arc<AtomicUsize>,
}

impl Drop for DropAllocation {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::Relaxed);
    }
}

unsafe impl BackendAllocation for DropAllocation {
    fn root_extent(&self) -> RootResourceExtent {
        self.extent
    }

    fn provider_kind(&self) -> ProviderKind {
        BackendId::Cpu
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::none()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

struct DropBinding(Arc<AtomicUsize>);

impl Drop for DropBinding {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
}

impl ProviderRetirementBinding for DropBinding {}

struct DropContext(Arc<AtomicUsize>);

impl Drop for DropContext {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
}

impl ProviderContext for DropContext {}

#[derive(Clone, Copy)]
enum EventKind {
    Proven,
    Failed,
    Unproven,
}

struct DropEvent {
    kind: EventKind,
    drops: Arc<AtomicUsize>,
}

impl Drop for DropEvent {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::Relaxed);
    }
}

impl ProviderEvent for DropEvent {
    fn completion(&self) -> EventCompletion {
        match self.kind {
            EventKind::Proven => EventCompletion::Proven,
            EventKind::Failed => EventCompletion::Failed(RetirementError::Provider {
                message: "fake provider failure".to_owned(),
            }),
            EventKind::Unproven => EventCompletion::Unproven(RetirementError::Unproven {
                message: "fake provider did not prove completion".to_owned(),
            }),
        }
    }
}

fn allocation_key(local: u64) -> AllocationKey {
    AllocationKey::new(
        AllocationDomainId::fresh(),
        AllocationId::from_backend_id(local),
    )
}

fn package() -> (PreparedPackage, [Arc<AtomicUsize>; 4], Arc<AtomicUsize>) {
    let binding_drops = Arc::new(AtomicUsize::new(0));
    let root_drops = Arc::new(AtomicUsize::new(0));
    let provider_drops = Arc::new(AtomicUsize::new(0));
    let event_drops = Arc::new(AtomicUsize::new(0));
    let extent = RootResourceExtent::try_new(allocation_key(90), 0, 1, 1).expect("extent");
    let owner = import_unique_root(Box::new(DropAllocation {
        extent,
        drops: Arc::clone(&root_drops),
    }))
    .expect("root import");
    let package = PreparedPackage::new(
        vec![
            Box::new(DropBinding(Arc::clone(&binding_drops))) as Box<dyn ProviderRetirementBinding>
        ]
        .into_boxed_slice(),
        vec![owner.into_root_pin()].into_boxed_slice(),
        Box::new(DropContext(Arc::clone(&provider_drops))) as Box<dyn ProviderContext>,
    );
    (
        package,
        [
            binding_drops,
            root_drops,
            provider_drops,
            Arc::clone(&event_drops),
        ],
        event_drops,
    )
}

fn admit(
    package: PreparedPackage,
    kind: EventKind,
    event_drops: Arc<AtomicUsize>,
) -> super::super::retirement::RetirementRecord {
    match package.admit(AdmissionDecision::Enqueued(Box::new(DropEvent {
        kind,
        drops: event_drops,
    }))) {
        Ok(record) => record,
        Err(_) => panic!("enqueue admission unexpectedly rejected"),
    }
}

#[test]
fn proven_retirement_releases_binding_root_and_context_once() {
    let (package, drops, event_drops) = package();
    let record = admit(package, EventKind::Proven, event_drops);
    let outcome = record.finish();
    assert!(matches!(outcome, RetirementOutcome::Completed));
    assert_eq!(format!("{outcome:?}"), "Completed");
    for counter in &drops {
        assert_eq!(counter.load(Ordering::Relaxed), 1);
    }
}

#[test]
fn unproven_retirement_keeps_binding_root_and_context_alive() {
    let (package, drops, event_drops) = package();
    let record = admit(package, EventKind::Unproven, event_drops);
    let outcome = record.finish();
    assert!(matches!(outcome, RetirementOutcome::CompletionUnproven(_)));
    assert!(format!("{outcome:?}").contains("CompletionUnproven"));
    for counter in &drops {
        assert_eq!(counter.load(Ordering::Relaxed), 0);
    }
}

#[test]
fn proven_provider_failure_still_releases_owned_resources() {
    let (package, drops, event_drops) = package();
    let record = admit(package, EventKind::Failed, event_drops);
    let outcome = record.finish();
    assert!(matches!(outcome, RetirementOutcome::Failed(_)));
    assert!(format!("{outcome:?}").contains("Failed"));
    for counter in &drops {
        assert_eq!(counter.load(Ordering::Relaxed), 1);
    }
}

#[test]
fn pre_admission_rejection_returns_the_unchanged_prepared_package() {
    let (package, drops, event_drops) = package();
    let (package, error) =
        match package.admit(AdmissionDecision::Rejected(AdmissionError::Rejected {
            message: "fake admission rejection".to_owned(),
        })) {
            Err(value) => value,
            Ok(_) => panic!("rejection must not create a retirement record"),
        };
    assert!(matches!(error, AdmissionError::Rejected { .. }));
    assert_eq!(event_drops.load(Ordering::Relaxed), 0);
    drop(package);
    for counter in &drops[..3] {
        assert_eq!(counter.load(Ordering::Relaxed), 1);
    }
}
