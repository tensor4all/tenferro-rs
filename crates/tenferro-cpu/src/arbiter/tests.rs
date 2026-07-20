use super::*;
use crate::{CpuId, CpuSet};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;
use std::time::Duration;

#[test]
fn disjoint_domains_run_together_but_all_allowed_waits() {
    let arbiter = ResourceArbiter::new();
    let node0 = arbiter.acquire(cpu_set([0, 1])).unwrap();
    let node1 = try_cpu_on_other_thread(&arbiter, cpu_set([2, 3])).unwrap();
    assert!(try_cpu_on_other_thread(&arbiter, cpu_set([0, 1, 2, 3])).is_none());
    drop((node0, node1));
    assert!(arbiter
        .try_acquire(cpu_set([0, 1, 2, 3]))
        .unwrap()
        .is_some());
}

#[test]
fn provider_exclusive_conflicts_with_every_cpu_domain() {
    let arbiter = ResourceArbiter::new();
    let _node = arbiter.acquire(cpu_set([4, 5])).unwrap();
    let other = arbiter.clone();
    assert!(
        std::thread::spawn(move || other.try_acquire_provider_exclusive().unwrap())
            .join()
            .unwrap()
            .is_none()
    );
}

#[test]
fn same_thread_reentrant_request_does_not_wait_on_its_own_permit() {
    let arbiter = ResourceArbiter::new();
    let _outer = arbiter.acquire_provider_exclusive().unwrap();

    assert!(arbiter.try_acquire(cpu_set([0])).unwrap().is_some());
}

#[test]
fn blocking_same_thread_reentrant_request_completes() {
    let (completed_tx, completed_rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let arbiter = ResourceArbiter::new();
        let _outer = arbiter.acquire_provider_exclusive().unwrap();
        let _inner = arbiter.acquire(cpu_set([0])).unwrap();
        completed_tx.send(()).unwrap();
    });

    assert!(completed_rx
        .recv_timeout(std::time::Duration::from_secs(2))
        .is_ok());
}

#[test]
fn panic_releases_provider_exclusive_reservation() {
    let arbiter = ResourceArbiter::new();
    let _ = catch_unwind(AssertUnwindSafe(|| {
        let _permit = arbiter.acquire_provider_exclusive().unwrap();
        panic!("forced");
    }));
    assert!(arbiter.try_acquire_provider_exclusive().unwrap().is_some());
}

#[test]
fn released_active_request_retains_reusable_storage() {
    let arbiter = ResourceArbiter::new();
    drop(arbiter.acquire(cpu_set([0, 1])).unwrap());
    let warm_capacity = arbiter.inner.state.lock().unwrap().active.capacity();

    for _ in 0..64 {
        drop(arbiter.acquire(cpu_set([0, 1])).unwrap());
    }

    let state = arbiter.inner.state.lock().unwrap();
    assert!(state.active.is_empty());
    assert_eq!(state.active.capacity(), warm_capacity);
}

#[test]
fn older_all_allowed_waiter_blocks_younger_disjoint_admission() {
    let arbiter = Arc::new(ResourceArbiter::new());
    let active = arbiter.acquire(cpu_set([0, 1])).unwrap();
    let waiter_arbiter = Arc::clone(&arbiter);
    let waiter = std::thread::spawn(move || waiter_arbiter.acquire(cpu_set([0, 1, 2, 3])).unwrap());
    assert!(arbiter.wait_for_waiter_count_for_test(1, Duration::from_secs(2)));

    assert!(try_cpu_on_other_thread(&arbiter, cpu_set([2, 3])).is_none());
    drop(active);
    drop(waiter.join().unwrap());
}

#[test]
fn older_blocked_node_waiter_does_not_serialize_a_disjoint_node() {
    let arbiter = Arc::new(ResourceArbiter::new());
    let active = arbiter.acquire(cpu_set([0, 1])).unwrap();
    let waiter_arbiter = Arc::clone(&arbiter);
    let waiter = std::thread::spawn(move || waiter_arbiter.acquire(cpu_set([0, 1])).unwrap());
    assert!(arbiter.wait_for_waiter_count_for_test(1, Duration::from_secs(2)));

    let disjoint = try_cpu_on_other_thread(&arbiter, cpu_set([2, 3]));
    assert!(disjoint.is_some());
    drop(disjoint);
    drop(active);
    drop(waiter.join().unwrap());
}

#[test]
fn poisoned_state_returns_a_typed_error() {
    let arbiter = ResourceArbiter::new();
    arbiter.poison_for_test();

    assert!(matches!(
        arbiter.try_acquire(cpu_set([0])),
        Err(ResourceArbiterError::StatePoisoned)
    ));
}

#[test]
fn uncontended_blocking_acquire_uses_direct_admission_without_notification() {
    let arbiter = ResourceArbiter::new();
    assert_eq!(
        arbiter.admission_metrics_for_test(),
        AdmissionMetrics::default()
    );

    let permit = arbiter.acquire(cpu_set([0, 1])).unwrap();
    assert_eq!(
        arbiter.admission_metrics_for_test(),
        AdmissionMetrics {
            direct: 1,
            queued: 0,
            acquire_notifications: 0,
            release_notifications: 0,
        }
    );

    drop(permit);
    assert_eq!(
        arbiter.admission_metrics_for_test(),
        AdmissionMetrics {
            direct: 1,
            queued: 0,
            acquire_notifications: 0,
            release_notifications: 0,
        }
    );
}

#[test]
fn conflicting_blocking_acquire_queues_and_release_notifies() {
    let arbiter = Arc::new(ResourceArbiter::new());
    let active = arbiter.acquire(cpu_set([0, 1])).unwrap();
    let waiter_arbiter = Arc::clone(&arbiter);
    let (waiter_tx, waiter_rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let permit = waiter_arbiter.acquire(cpu_set([0, 1])).unwrap();
        waiter_tx.send(permit).unwrap();
    });
    assert!(arbiter.wait_for_waiter_count_for_test(1, Duration::from_secs(2)));
    assert_eq!(
        arbiter.admission_metrics_for_test(),
        AdmissionMetrics {
            direct: 1,
            queued: 1,
            acquire_notifications: 1,
            release_notifications: 0,
        }
    );

    drop(active);
    let waiter = waiter_rx.recv_timeout(Duration::from_secs(2)).unwrap();
    drop(waiter);
    assert_eq!(
        arbiter.admission_metrics_for_test(),
        AdmissionMetrics {
            direct: 1,
            queued: 1,
            acquire_notifications: 1,
            release_notifications: 1,
        }
    );
}

#[test]
fn recovering_acquire_resets_exhausted_request_ids_when_idle() {
    let arbiter = Arc::new(ResourceArbiter::new());
    arbiter.inner.state.lock().unwrap().next_request_id = u64::MAX;
    let recovering_arbiter = Arc::clone(&arbiter);
    let (id_tx, id_rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let permit = recovering_arbiter.acquire_recovering(cpu_set([0]), ResourceOwner::fresh());
        let id = permit.id;
        drop(permit);
        id_tx.send(id).unwrap();
    });

    assert_eq!(id_rx.recv_timeout(Duration::from_secs(2)).unwrap(), 0);
}

#[test]
fn recovering_acquire_waits_for_active_permit_and_wakes_on_release() {
    let arbiter = Arc::new(ResourceArbiter::new());
    let active = arbiter.acquire(cpu_set([0])).unwrap();
    arbiter.inner.state.lock().unwrap().next_request_id = u64::MAX;
    let recovering_arbiter = Arc::clone(&arbiter);
    let (permit_tx, permit_rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let permit = recovering_arbiter.acquire_recovering(cpu_set([1]), ResourceOwner::fresh());
        permit_tx.send((permit.id, permit)).unwrap();
    });
    assert!(arbiter.wait_for_recovery_waiter_count_for_test(1, Duration::from_secs(2)));

    drop(active);
    let (id, recovered) = permit_rx.recv_timeout(Duration::from_secs(2)).unwrap();
    assert_eq!(id, 0);
    drop(recovered);
}

#[test]
fn multiple_recovering_acquires_wake_without_losing_waiter_accounting() {
    let arbiter = Arc::new(ResourceArbiter::new());
    let active = arbiter.acquire(cpu_set([0])).unwrap();
    arbiter.inner.state.lock().unwrap().next_request_id = u64::MAX;
    let (id_tx, id_rx) = std::sync::mpsc::channel();
    for cpu in [1, 2] {
        let recovering_arbiter = Arc::clone(&arbiter);
        let id_tx = id_tx.clone();
        std::thread::spawn(move || {
            let permit =
                recovering_arbiter.acquire_recovering(cpu_set([cpu]), ResourceOwner::fresh());
            let id = permit.id;
            drop(permit);
            id_tx.send(id).unwrap();
        });
    }
    drop(id_tx);
    assert!(arbiter.wait_for_recovery_waiter_count_for_test(2, Duration::from_secs(2)));

    drop(active);
    let mut ids = [
        id_rx.recv_timeout(Duration::from_secs(2)).unwrap(),
        id_rx.recv_timeout(Duration::from_secs(2)).unwrap(),
    ];
    ids.sort_unstable();
    assert!(ids == [0, 0] || ids == [0, 1]);
    assert_eq!(arbiter.inner.state.lock().unwrap().recovery_waiters, 0);
}

#[test]
fn recovering_acquire_clears_poison_and_admits() {
    let arbiter = Arc::new(ResourceArbiter::new());
    arbiter.poison_for_test();
    let recovering_arbiter = Arc::clone(&arbiter);
    let (completed_tx, completed_rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let permit = recovering_arbiter.acquire_recovering(cpu_set([0]), ResourceOwner::fresh());
        drop(permit);
        completed_tx.send(()).unwrap();
    });

    completed_rx.recv_timeout(Duration::from_secs(2)).unwrap();

    assert!(arbiter.try_acquire(cpu_set([0])).unwrap().is_some());
}

fn cpu_set<const N: usize>(cpus: [usize; N]) -> CpuSet {
    CpuSet::new(cpus.map(CpuId::new)).unwrap()
}

fn try_cpu_on_other_thread(arbiter: &ResourceArbiter, cpus: CpuSet) -> Option<ResourcePermit> {
    let other = arbiter.clone();
    std::thread::spawn(move || other.try_acquire(cpus).unwrap())
        .join()
        .unwrap()
}
