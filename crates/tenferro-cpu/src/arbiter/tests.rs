use super::*;
use crate::{CpuId, CpuSet};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;
use std::time::Duration;

#[test]
fn disjoint_domains_run_together_but_all_allowed_waits() {
    let arbiter = ResourceArbiter::new();
    let node0 = arbiter.acquire(cpu_set([0, 1])).unwrap();
    let node1 = arbiter.try_acquire(cpu_set([2, 3])).unwrap().unwrap();
    assert!(arbiter
        .try_acquire(cpu_set([0, 1, 2, 3]))
        .unwrap()
        .is_none());
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
    assert!(arbiter.try_acquire_provider_exclusive().unwrap().is_none());
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
fn older_all_allowed_waiter_blocks_younger_disjoint_admission() {
    let arbiter = Arc::new(ResourceArbiter::new());
    let active = arbiter.acquire(cpu_set([0, 1])).unwrap();
    let waiter_arbiter = Arc::clone(&arbiter);
    let waiter = std::thread::spawn(move || waiter_arbiter.acquire(cpu_set([0, 1, 2, 3])).unwrap());
    assert!(arbiter.wait_for_waiter_count_for_test(1, Duration::from_secs(2)));

    assert!(arbiter.try_acquire(cpu_set([2, 3])).unwrap().is_none());
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

    let disjoint = arbiter.try_acquire(cpu_set([2, 3])).unwrap();
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

fn cpu_set<const N: usize>(cpus: [usize; N]) -> CpuSet {
    CpuSet::new(cpus.map(CpuId::new)).unwrap()
}
