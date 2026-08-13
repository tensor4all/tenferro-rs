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
fn exhaustion_recovery_waiter_wakes_when_last_active_permit_drops() {
    // The request-id-exhaustion recovery loop parks on the condvar WITHOUT a
    // waiter-list entry until active and waiters are both empty. Dropping the
    // last active permit must still wake it (issue #1667 drop fast path);
    // otherwise it sleeps forever.
    let arbiter = Arc::new(ResourceArbiter::new());
    let active = arbiter.acquire(cpu_set([0])).unwrap();
    {
        let mut state = arbiter.inner.state.lock().unwrap();
        state.next_request_id = u64::MAX;
    }
    let (tx, rx) = std::sync::mpsc::channel();
    let arbiter2 = Arc::clone(&arbiter);
    let handle = std::thread::spawn(move || {
        // Use the recovering path (as the backend does); the plain test
        // helper returns RequestIdExhausted directly without recovery.
        let permit = arbiter2.acquire_recovering(cpu_set([1]), request_owner());
        tx.send(()).unwrap();
        permit
    });
    // Wait until the recovery thread has entered the exhaustion-recovery park
    // (it parks without a waiter-list entry), then drop the last active permit.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    while arbiter
        .inner
        .recovery_waiters
        .load(std::sync::atomic::Ordering::Relaxed)
        == 0
    {
        assert!(
            std::time::Instant::now() < deadline,
            "recovery thread did not enter the exhaustion-recovery park"
        );
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    drop(active);
    assert!(
        rx.recv_timeout(std::time::Duration::from_secs(2)).is_ok(),
        "exhaustion-recovery waiter was not woken by the last active drop"
    );
    drop(handle.join().unwrap());
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

fn cpu_set<const N: usize>(cpus: [usize; N]) -> CpuSet {
    CpuSet::new(cpus.map(CpuId::new)).unwrap()
}

fn try_cpu_on_other_thread(arbiter: &ResourceArbiter, cpus: CpuSet) -> Option<ResourcePermit> {
    let other = arbiter.clone();
    std::thread::spawn(move || other.try_acquire(cpus).unwrap())
        .join()
        .unwrap()
}
