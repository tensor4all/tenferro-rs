use std::num::NonZeroUsize;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Mutex, Weak};
use std::thread;
use std::time::Duration;

use tenferro_cpu::CpuBackend;

use super::super::cache::{
    CacheLookup, CacheProduced, PreparedCacheKey, PreparedPlanCache, PreparedValue,
    RuntimeCacheSet, SharedRetention,
};
use super::super::*;

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct TestKey(u64);

#[derive(Debug)]
struct TestShared {
    bytes: usize,
    sentinel: Option<Arc<()>>,
}

impl TestShared {
    fn new(bytes: usize) -> Arc<Self> {
        Arc::new(Self {
            bytes,
            sentinel: None,
        })
    }
}

impl PreparedCacheKey for TestKey {
    type Shared = TestShared;

    fn compact_digest(&self) -> u128 {
        0x1451
    }

    fn exact_eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }

    fn retained_bytes(&self) -> Option<usize> {
        Some(0)
    }

    fn summary(&self) -> PreparationKeySummary {
        PreparationKeySummary::for_test(self.0)
    }

    fn shared_retention(&self) -> Option<SharedRetention<Self::Shared>> {
        None
    }
}

#[derive(Debug)]
struct CloneCountingKey {
    value: u64,
    clones: Arc<AtomicUsize>,
}

impl Clone for CloneCountingKey {
    fn clone(&self) -> Self {
        self.clones.fetch_add(1, Ordering::SeqCst);
        Self {
            value: self.value,
            clones: Arc::clone(&self.clones),
        }
    }
}

impl PreparedCacheKey for CloneCountingKey {
    type Shared = TestShared;

    fn compact_digest(&self) -> u128 {
        0x1451
    }

    fn exact_eq(&self, other: &Self) -> bool {
        self.value == other.value
    }

    fn retained_bytes(&self) -> Option<usize> {
        Some(0)
    }

    fn summary(&self) -> PreparationKeySummary {
        PreparationKeySummary::for_test(self.value)
    }

    fn shared_retention(&self) -> Option<SharedRetention<Self::Shared>> {
        None
    }
}

#[derive(Debug)]
struct TestValue {
    bytes: usize,
    sentinel: Option<Arc<()>>,
}

impl TestValue {
    fn new(bytes: usize) -> Arc<Self> {
        Arc::new(Self {
            bytes,
            sentinel: None,
        })
    }
}

impl PreparedValue for TestValue {
    fn retained_bytes(&self) -> Option<usize> {
        Some(self.bytes)
    }
}

#[derive(Clone, Debug)]
struct BlockingKey {
    value: u64,
    entered: Arc<Barrier>,
    release: Arc<Barrier>,
}

impl PreparedCacheKey for BlockingKey {
    type Shared = TestShared;

    fn compact_digest(&self) -> u128 {
        0x1451
    }

    fn exact_eq(&self, other: &Self) -> bool {
        self.entered.wait();
        self.release.wait();
        self.value == other.value
    }

    fn retained_bytes(&self) -> Option<usize> {
        Some(0)
    }

    fn summary(&self) -> PreparationKeySummary {
        PreparationKeySummary::for_test(self.value)
    }

    fn shared_retention(&self) -> Option<SharedRetention<Self::Shared>> {
        None
    }
}

#[derive(Clone, Debug)]
struct DropKey {
    value: u64,
    sentinel: Arc<()>,
    shared: Option<SharedRetention<TestShared>>,
}

impl PreparedCacheKey for DropKey {
    type Shared = TestShared;

    fn compact_digest(&self) -> u128 {
        u128::from(self.value)
    }

    fn exact_eq(&self, other: &Self) -> bool {
        self.value == other.value
    }

    fn retained_bytes(&self) -> Option<usize> {
        let _ = Arc::strong_count(&self.sentinel);
        Some(0)
    }

    fn summary(&self) -> PreparationKeySummary {
        PreparationKeySummary::for_test(self.value)
    }

    fn shared_retention(&self) -> Option<SharedRetention<Self::Shared>> {
        self.shared.clone()
    }
}

fn limits(
    entries: usize,
    bytes: usize,
    in_flight: usize,
    queued: usize,
) -> PreparedPlanCacheLimits {
    PreparedPlanCacheLimits {
        max_entries: NonZeroUsize::new(entries).unwrap(),
        max_retained_bytes: NonZeroUsize::new(bytes).unwrap(),
        max_in_flight_entries: NonZeroUsize::new(in_flight).unwrap(),
        max_queued_distinct_keys: NonZeroUsize::new(queued).unwrap(),
    }
}

fn prepare_ready(
    cache: &PreparedPlanCache<TestKey, TestValue>,
    key: u64,
    bytes: usize,
    calls: &AtomicUsize,
) -> Arc<TestValue> {
    match cache
        .get_or_prepare(TestKey(key), CacheInFlightBehavior::Wait, 0, || {
            calls.fetch_add(1, Ordering::SeqCst);
            CacheProduced::Ready {
                value: TestValue::new(bytes),
                shared: None,
            }
        })
        .unwrap()
    {
        CacheLookup::Ready(value) => value,
        other => panic!("expected ready lookup, got {other:?}"),
    }
}

fn eventually(condition: impl Fn() -> bool) {
    for _ in 0..30_000 {
        if condition() {
            return;
        }
        thread::sleep(Duration::from_micros(100));
    }
    panic!("condition did not become true");
}

fn assert_capacity_error(error: Arc<PrepareError>, in_flight: usize, queued: usize) {
    assert!(matches!(
        error.as_ref(),
        PrepareError::CacheInFlightCapacityExceeded {
            in_flight: actual_in_flight,
            queued_distinct_keys: actual_queued,
        } if (*actual_in_flight, *actual_queued) == (in_flight, queued)
    ));
}

#[test]
fn retained_hit_path_does_not_clone_candidate_keys() {
    let cache = PreparedPlanCache::<CloneCountingKey, TestValue>::new(limits(8, 4096, 1, 1));
    let clones = Arc::new(AtomicUsize::new(0));
    let calls = AtomicUsize::new(0);

    let first = cache
        .get_or_prepare(
            CloneCountingKey {
                value: 7,
                clones: Arc::clone(&clones),
            },
            CacheInFlightBehavior::Wait,
            0,
            || {
                calls.fetch_add(1, Ordering::SeqCst);
                CacheProduced::Ready {
                    value: TestValue::new(16),
                    shared: None,
                }
            },
        )
        .unwrap();
    assert!(matches!(first, CacheLookup::Ready(_)));
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    clones.store(0, Ordering::SeqCst);
    let second = cache
        .get_or_prepare(
            CloneCountingKey {
                value: 7,
                clones: Arc::clone(&clones),
            },
            CacheInFlightBehavior::Wait,
            0,
            || panic!("retained hit must not call producer"),
        )
        .unwrap();

    assert!(matches!(second, CacheLookup::Ready(_)));
    assert_eq!(
        clones.load(Ordering::SeqCst),
        0,
        "retained cache hits must not clone stored candidate keys"
    );
}

#[test]
fn probe_path_does_not_collect_arc_cloned_candidate_keys() {
    let source = include_str!("../cache.rs");
    let probe_start = source
        .find("    fn probe(")
        .expect("prepared cache probe should exist");
    let handle_entry_start = source[probe_start..]
        .find("    fn handle_entry(")
        .expect("handle_entry should follow probe")
        + probe_start;
    let probe_source = &source[probe_start..handle_entry_start];

    assert!(
        !probe_source.contains("ProbeCandidate"),
        "probe must not build an Arc-cloned candidate list on the cache hit path"
    );
}

#[test]
fn same_key_has_one_producer_and_shared_result() {
    let cache = Arc::new(PreparedPlanCache::<TestKey, TestValue>::new(limits(
        8, 4096, 2, 2,
    )));
    assert_eq!(cache.limits().unwrap(), limits(8, 4096, 2, 2));
    let calls = Arc::new(AtomicUsize::new(0));
    let producer_barrier = Arc::new(Barrier::new(2));
    let producer_release = Arc::new(Barrier::new(2));

    thread::scope(|scope| {
        let mut handles = Vec::new();
        for _ in 0..8 {
            let cache = Arc::clone(&cache);
            let calls = Arc::clone(&calls);
            let barrier = Arc::clone(&producer_barrier);
            let release = Arc::clone(&producer_release);
            handles.push(scope.spawn(move || {
                cache
                    .get_or_prepare(TestKey(7), CacheInFlightBehavior::Wait, 0, || {
                        calls.fetch_add(1, Ordering::SeqCst);
                        barrier.wait();
                        release.wait();
                        CacheProduced::Ready {
                            value: TestValue::new(16),
                            shared: None,
                        }
                    })
                    .unwrap()
            }));
        }

        producer_barrier.wait();
        let stats = cache.stats().unwrap();
        assert_eq!(stats.in_flight, 1);
        assert_eq!(stats.queued_distinct_keys, 0);
        producer_release.wait();

        let results = handles
            .into_iter()
            .map(|handle| match handle.join().unwrap() {
                CacheLookup::Ready(value) => value,
                other => panic!("expected ready lookup, got {other:?}"),
            })
            .collect::<Vec<_>>();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        for value in &results[1..] {
            assert!(Arc::ptr_eq(&results[0], value));
        }
    });
}

#[test]
fn same_key_waiter_does_not_miss_publication_before_first_wait() {
    let cache = Arc::new(PreparedPlanCache::<TestKey, TestValue>::new(limits(
        8, 4096, 2, 2,
    )));
    let producer_entered = Arc::new(Barrier::new(2));
    let producer_release = Arc::new(Barrier::new(2));
    let waiter_before_wait = Arc::new(Barrier::new(2));
    let waiter_release = Arc::new(Barrier::new(2));

    let producer = {
        let cache = Arc::clone(&cache);
        let producer_entered = Arc::clone(&producer_entered);
        let producer_release = Arc::clone(&producer_release);
        thread::spawn(move || {
            cache
                .get_or_prepare(TestKey(8), CacheInFlightBehavior::Wait, 0, || {
                    producer_entered.wait();
                    producer_release.wait();
                    CacheProduced::Ready {
                        value: TestValue::new(16),
                        shared: None,
                    }
                })
                .unwrap()
        })
    };
    producer_entered.wait();

    let waiter = {
        let cache = Arc::clone(&cache);
        let waiter_before_wait = Arc::clone(&waiter_before_wait);
        let waiter_release = Arc::clone(&waiter_release);
        thread::spawn(move || {
            let lookup = cache
                .get_or_prepare_with_entry_wait_hooks_for_test(
                    TestKey(8),
                    CacheInFlightBehavior::Wait,
                    0,
                    || panic!("same-key waiter must not become the producer"),
                    move || {
                        waiter_before_wait.wait();
                        waiter_release.wait();
                    },
                    || panic!("waiter must not sleep after producer publication"),
                )
                .unwrap();
            lookup
        })
    };
    waiter_before_wait.wait();

    producer_release.wait();
    let produced = match producer.join().unwrap() {
        CacheLookup::Ready(value) => value,
        other => panic!("expected producer ready lookup, got {other:?}"),
    };
    waiter_release.wait();

    let waited = match waiter.join().unwrap() {
        CacheLookup::Ready(value) => value,
        other => panic!("expected waiter ready lookup, got {other:?}"),
    };
    assert!(Arc::ptr_eq(&produced, &waited));
}

#[test]
fn digest_collision_uses_exact_equality_outside_lock() {
    let cache = Arc::new(PreparedPlanCache::<BlockingKey, TestValue>::new(limits(
        4, 4096, 1, 4,
    )));
    let seed_entered = Arc::new(Barrier::new(1));
    let seed_release = Arc::new(Barrier::new(1));
    cache
        .get_or_prepare(
            BlockingKey {
                value: 1,
                entered: Arc::clone(&seed_entered),
                release: Arc::clone(&seed_release),
            },
            CacheInFlightBehavior::Wait,
            0,
            || CacheProduced::Ready {
                value: TestValue::new(8),
                shared: None,
            },
        )
        .unwrap();

    let entered = Arc::new(Barrier::new(2));
    let release = Arc::new(Barrier::new(2));
    thread::scope(|scope| {
        let handle = {
            let cache = Arc::clone(&cache);
            let entered = Arc::clone(&entered);
            let release = Arc::clone(&release);
            scope.spawn(move || {
                cache
                    .get_or_prepare(
                        BlockingKey {
                            value: 2,
                            entered,
                            release,
                        },
                        CacheInFlightBehavior::Wait,
                        0,
                        || CacheProduced::Ready {
                            value: TestValue::new(8),
                            shared: None,
                        },
                    )
                    .unwrap()
            })
        };

        entered.wait();
        assert_eq!(cache.stats().unwrap().entries, 1);
        release.wait();
        assert!(matches!(handle.join().unwrap(), CacheLookup::Ready(_)));
    });
}

#[test]
fn distinct_keys_can_prepare_concurrently_and_do_not_share_values() {
    let cache = Arc::new(PreparedPlanCache::<TestKey, TestValue>::new(limits(
        8, 4096, 2, 2,
    )));
    let calls = Arc::new(AtomicUsize::new(0));
    let barrier = Arc::new(Barrier::new(3));

    thread::scope(|scope| {
        let left = {
            let cache = Arc::clone(&cache);
            let calls = Arc::clone(&calls);
            let barrier = Arc::clone(&barrier);
            scope.spawn(move || {
                cache
                    .get_or_prepare(TestKey(7), CacheInFlightBehavior::Wait, 0, || {
                        calls.fetch_add(1, Ordering::SeqCst);
                        barrier.wait();
                        CacheProduced::Ready {
                            value: TestValue::new(16),
                            shared: None,
                        }
                    })
                    .unwrap()
            })
        };
        let right = {
            let cache = Arc::clone(&cache);
            let calls = Arc::clone(&calls);
            let barrier = Arc::clone(&barrier);
            scope.spawn(move || {
                cache
                    .get_or_prepare(TestKey(8), CacheInFlightBehavior::Wait, 0, || {
                        calls.fetch_add(1, Ordering::SeqCst);
                        barrier.wait();
                        CacheProduced::Ready {
                            value: TestValue::new(16),
                            shared: None,
                        }
                    })
                    .unwrap()
            })
        };

        eventually(|| calls.load(Ordering::SeqCst) == 2);
        assert_eq!(cache.stats().unwrap().in_flight, 2);
        barrier.wait();
        let left = match left.join().unwrap() {
            CacheLookup::Ready(value) => value,
            other => panic!("expected ready lookup, got {other:?}"),
        };
        let right = match right.join().unwrap() {
            CacheLookup::Ready(value) => value,
            other => panic!("expected ready lookup, got {other:?}"),
        };
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert!(!Arc::ptr_eq(&left, &right));
    });
}

#[test]
fn recursive_and_nested_preparation_fail_before_capacity_changes() {
    let cache = PreparedPlanCache::<TestKey, TestValue>::new(limits(8, 4096, 1, 1));
    let calls = AtomicUsize::new(0);

    let outer = cache
        .get_or_prepare(TestKey(1), CacheInFlightBehavior::Wait, 0, || {
            calls.fetch_add(1, Ordering::SeqCst);
            let cycle = cache
                .get_or_prepare(TestKey(1), CacheInFlightBehavior::Wait, 0, || {
                    panic!("cycle must not call nested producer")
                })
                .unwrap_err();
            assert!(matches!(
                cycle.as_ref(),
                PrepareError::PreparationCycle { key } if *key == PreparationKeySummary::for_test(1)
            ));

            let nested = cache
                .get_or_prepare(TestKey(2), CacheInFlightBehavior::Wait, 0, || {
                    panic!("nested request must not call nested producer")
                })
                .unwrap_err();
            assert!(matches!(
                nested.as_ref(),
                PrepareError::NestedPreparationUnsupported {
                    parent,
                    requested,
                } if (*parent, *requested)
                    == (PreparationKeySummary::for_test(1), PreparationKeySummary::for_test(2))
            ));
            let stats = cache.stats().unwrap();
            assert_eq!(stats.in_flight, 1);
            assert_eq!(stats.queued_distinct_keys, 0);
            CacheProduced::Ready {
                value: TestValue::new(8),
                shared: None,
            }
        })
        .unwrap();
    assert!(matches!(outer, CacheLookup::Ready(_)));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let stats = cache.stats().unwrap();
    assert_eq!(stats.in_flight, 0);
    assert_eq!(stats.queued_distinct_keys, 0);
}

#[test]
fn distinct_wait_queue_is_fifo_and_capacity_errors_are_typed() {
    let cache = Arc::new(PreparedPlanCache::<TestKey, TestValue>::new(limits(
        8, 4096, 1, 3,
    )));
    let producer_order = Arc::new(Mutex::new(Vec::new()));
    let first_entered = Arc::new(Barrier::new(2));
    let first_release = Arc::new(Barrier::new(2));

    thread::scope(|scope| {
        let first = {
            let cache = Arc::clone(&cache);
            let producer_order = Arc::clone(&producer_order);
            let first_entered = Arc::clone(&first_entered);
            let first_release = Arc::clone(&first_release);
            scope.spawn(move || {
                cache
                    .get_or_prepare(TestKey(1), CacheInFlightBehavior::Wait, 0, || {
                        producer_order.lock().unwrap().push(1);
                        first_entered.wait();
                        first_release.wait();
                        CacheProduced::Ready {
                            value: TestValue::new(8),
                            shared: None,
                        }
                    })
                    .unwrap()
            })
        };
        first_entered.wait();

        let second = {
            let cache = Arc::clone(&cache);
            let producer_order = Arc::clone(&producer_order);
            scope.spawn(move || {
                cache
                    .get_or_prepare(TestKey(2), CacheInFlightBehavior::Wait, 0, || {
                        producer_order.lock().unwrap().push(2);
                        CacheProduced::Ready {
                            value: TestValue::new(8),
                            shared: None,
                        }
                    })
                    .unwrap()
            })
        };
        eventually(|| cache.stats().unwrap().queued_distinct_keys == 1);
        let third = {
            let cache = Arc::clone(&cache);
            let producer_order = Arc::clone(&producer_order);
            scope.spawn(move || {
                cache
                    .get_or_prepare(TestKey(3), CacheInFlightBehavior::Wait, 0, || {
                        producer_order.lock().unwrap().push(3);
                        CacheProduced::Ready {
                            value: TestValue::new(8),
                            shared: None,
                        }
                    })
                    .unwrap()
            })
        };
        eventually(|| cache.stats().unwrap().queued_distinct_keys == 2);

        let refused = cache
            .get_or_prepare(TestKey(4), CacheInFlightBehavior::Refuse, 0, || {
                panic!("refused request must not call producer")
            })
            .unwrap_err();
        assert_capacity_error(refused, 1, 2);
        assert_eq!(cache.stats().unwrap().queued_distinct_keys, 2);

        first_release.wait();
        assert!(matches!(first.join().unwrap(), CacheLookup::Ready(_)));
        assert!(matches!(second.join().unwrap(), CacheLookup::Ready(_)));
        assert!(matches!(third.join().unwrap(), CacheLookup::Ready(_)));
    });

    assert_eq!(*producer_order.lock().unwrap(), vec![1, 2, 3]);
}

#[test]
fn wait_queue_limit_rejects_next_distinct_waiter() {
    let cache = Arc::new(PreparedPlanCache::<TestKey, TestValue>::new(limits(
        8, 4096, 1, 1,
    )));
    let first_entered = Arc::new(Barrier::new(2));
    let first_release = Arc::new(Barrier::new(2));

    thread::scope(|scope| {
        let first = {
            let cache = Arc::clone(&cache);
            let first_entered = Arc::clone(&first_entered);
            let first_release = Arc::clone(&first_release);
            scope.spawn(move || {
                cache
                    .get_or_prepare(TestKey(1), CacheInFlightBehavior::Wait, 0, || {
                        first_entered.wait();
                        first_release.wait();
                        CacheProduced::Ready {
                            value: TestValue::new(8),
                            shared: None,
                        }
                    })
                    .unwrap()
            })
        };
        first_entered.wait();

        let second = {
            let cache = Arc::clone(&cache);
            scope.spawn(move || {
                cache
                    .get_or_prepare(TestKey(2), CacheInFlightBehavior::Wait, 0, || {
                        CacheProduced::Ready {
                            value: TestValue::new(8),
                            shared: None,
                        }
                    })
                    .unwrap()
            })
        };
        eventually(|| cache.stats().unwrap().queued_distinct_keys == 1);

        let error = cache
            .get_or_prepare(TestKey(3), CacheInFlightBehavior::Wait, 0, || {
                panic!("full queue must not call producer")
            })
            .unwrap_err();
        assert_capacity_error(error, 1, 1);

        first_release.wait();
        assert!(matches!(first.join().unwrap(), CacheLookup::Ready(_)));
        assert!(matches!(second.join().unwrap(), CacheLookup::Ready(_)));
    });
}

#[test]
fn producer_panic_releases_gauges_and_next_prepare_completes() {
    let cache = PreparedPlanCache::<TestKey, TestValue>::new(limits(8, 4096, 1, 1));
    let calls = AtomicUsize::new(0);
    let panic_result = panic::catch_unwind(AssertUnwindSafe(|| {
        let _ = cache.get_or_prepare(TestKey(1), CacheInFlightBehavior::Wait, 0, || {
            calls.fetch_add(1, Ordering::SeqCst);
            panic!("producer panic")
        });
    }));
    assert!(panic_result.is_err());
    let stats = cache.stats().unwrap();
    assert_eq!(stats.in_flight, 0);
    assert_eq!(stats.queued_distinct_keys, 0);
    assert_eq!(stats.entries, 0);

    let value = prepare_ready(&cache, 2, 8, &calls);
    assert_eq!(value.bytes, 8);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[test]
fn deterministic_and_transient_failures_have_distinct_retention_behavior() {
    let cache = Arc::new(PreparedPlanCache::<TestKey, TestValue>::new(limits(
        8, 4096, 1, 8,
    )));
    let deterministic_calls = AtomicUsize::new(0);
    let deterministic_error = Arc::new(PrepareError::Unsupported {
        reason: UnsupportedReason::Operation {
            operation: "deterministic-test",
        },
    });
    let first = cache
        .get_or_prepare(TestKey(10), CacheInFlightBehavior::Wait, 0, || {
            deterministic_calls.fetch_add(1, Ordering::SeqCst);
            CacheProduced::FailedDeterministic {
                error: Arc::clone(&deterministic_error),
                shared: None,
            }
        })
        .unwrap();
    let second = cache
        .get_or_prepare(TestKey(10), CacheInFlightBehavior::Wait, 0, || {
            panic!("deterministic failure should be retained")
        })
        .unwrap();
    let (CacheLookup::FailedDeterministic(first), CacheLookup::FailedDeterministic(second)) =
        (first, second)
    else {
        panic!("expected deterministic failures");
    };
    assert!(Arc::ptr_eq(&first, &second));
    assert_eq!(deterministic_calls.load(Ordering::SeqCst), 1);
    assert_eq!(cache.stats().unwrap().negative_hits, 1);

    let transient_calls = Arc::new(AtomicUsize::new(0));
    let transient_error = Arc::new(PrepareError::Unsupported {
        reason: UnsupportedReason::Operation {
            operation: "transient-test",
        },
    });
    let barrier = Arc::new(Barrier::new(2));
    thread::scope(|scope| {
        let mut handles = Vec::new();
        for _ in 0..4 {
            let cache = Arc::clone(&cache);
            let transient_calls = Arc::clone(&transient_calls);
            let transient_error = Arc::clone(&transient_error);
            let barrier = Arc::clone(&barrier);
            handles.push(scope.spawn(move || {
                cache
                    .get_or_prepare(TestKey(20), CacheInFlightBehavior::Wait, 0, || {
                        transient_calls.fetch_add(1, Ordering::SeqCst);
                        barrier.wait();
                        CacheProduced::FailedTransient(Arc::clone(&transient_error))
                    })
                    .unwrap()
            }));
        }
        eventually(|| cache.stats().unwrap().waits >= 3);
        barrier.wait();
        for handle in handles {
            match handle.join().unwrap() {
                CacheLookup::FailedTransient(error) => {
                    assert!(Arc::ptr_eq(&error, &transient_error));
                }
                other => panic!("expected transient failure, got {other:?}"),
            }
        }
    });
    assert_eq!(transient_calls.load(Ordering::SeqCst), 1);

    let value = cache
        .get_or_prepare(TestKey(20), CacheInFlightBehavior::Wait, 0, || {
            transient_calls.fetch_add(1, Ordering::SeqCst);
            CacheProduced::Ready {
                value: TestValue::new(8),
                shared: None,
            }
        })
        .unwrap();
    assert!(matches!(value, CacheLookup::Ready(_)));
    assert_eq!(transient_calls.load(Ordering::SeqCst), 2);
}

#[test]
fn redirects_retain_pointer_identical_shared_root() {
    let cache = PreparedPlanCache::<TestKey, TestValue>::new(limits(8, 4096, 1, 1));
    let shared = TestShared::new(64);
    let first = cache
        .get_or_prepare(TestKey(30), CacheInFlightBehavior::Wait, 0, || {
            CacheProduced::Redirect {
                requirements: SpecializationRequirements::polymorphic(0),
                shared: Some(SharedRetention {
                    value: Arc::clone(&shared),
                    retained_bytes: Some(shared.bytes),
                }),
            }
        })
        .unwrap();
    let second = cache
        .get_or_prepare(TestKey(30), CacheInFlightBehavior::Wait, 0, || {
            panic!("redirect should be retained")
        })
        .unwrap();
    let (CacheLookup::Redirect { shared: first, .. }, CacheLookup::Redirect { shared: second, .. }) =
        (first, second)
    else {
        panic!("expected redirects");
    };
    assert!(Arc::ptr_eq(
        first.as_ref().expect("first shared"),
        second.as_ref().expect("second shared")
    ));
}

#[test]
fn lru_and_byte_limits_control_retention_without_affecting_returned_values() {
    let cache = PreparedPlanCache::<TestKey, TestValue>::new(limits(2, 1 << 20, 1, 4));
    let calls = AtomicUsize::new(0);
    let _a = prepare_ready(&cache, 1, 8, &calls);
    let _b = prepare_ready(&cache, 2, 8, &calls);
    let _a_hit = prepare_ready(&cache, 1, 8, &calls);
    let _c = prepare_ready(&cache, 3, 8, &calls);
    assert_eq!(cache.stats().unwrap().entries, 2);
    let _b_again = prepare_ready(&cache, 2, 8, &calls);
    assert_eq!(calls.load(Ordering::SeqCst), 4);

    let before = cache.stats().unwrap();
    cache
        .set_limits(limits(2, before.retained_bytes - 1, 1, 4))
        .unwrap();
    let after = cache.stats().unwrap();
    assert!(after.entries < before.entries);
    assert!(after.retained_bytes < before.retained_bytes);

    let oversize = PreparedPlanCache::<TestKey, TestValue>::new(limits(4, 64, 1, 4));
    let oversize_calls = AtomicUsize::new(0);
    assert_eq!(
        prepare_ready(&oversize, 1, 4096, &oversize_calls).bytes,
        4096
    );
    assert_eq!(oversize.stats().unwrap().entries, 0);
    prepare_ready(&oversize, 1, 4096, &oversize_calls);
    assert_eq!(oversize_calls.load(Ordering::SeqCst), 2);
}

#[test]
fn clear_during_active_attempt_advances_generation_and_prevents_reinsert() {
    let cache = Arc::new(PreparedPlanCache::<TestKey, TestValue>::new(limits(
        8, 4096, 1, 4,
    )));
    let calls = Arc::new(AtomicUsize::new(0));
    let entered = Arc::new(Barrier::new(2));
    let release = Arc::new(Barrier::new(2));

    thread::scope(|scope| {
        let handle = {
            let cache = Arc::clone(&cache);
            let calls = Arc::clone(&calls);
            let entered = Arc::clone(&entered);
            let release = Arc::clone(&release);
            scope.spawn(move || {
                cache
                    .get_or_prepare(TestKey(40), CacheInFlightBehavior::Wait, 32, || {
                        calls.fetch_add(1, Ordering::SeqCst);
                        entered.wait();
                        release.wait();
                        CacheProduced::Ready {
                            value: TestValue::new(8),
                            shared: None,
                        }
                    })
                    .unwrap()
            })
        };
        entered.wait();
        let active = cache.stats().unwrap();
        assert_eq!(active.in_flight, 1);
        assert!(active.retained_bytes >= 32);
        cache.clear().unwrap();
        let cleared = cache.stats().unwrap();
        assert_eq!(cleared.entries, 0);
        assert_eq!(cleared.queued_distinct_keys, 0);
        assert_eq!(cleared.in_flight, 1);
        release.wait();
        assert!(matches!(handle.join().unwrap(), CacheLookup::Ready(_)));
    });

    let after = cache.stats().unwrap();
    assert_eq!(after.entries, 0);
    assert_eq!(after.in_flight, 0);
    prepare_ready(&cache, 40, 8, &calls);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[test]
fn poisoned_state_stays_visible_after_producer_and_waiter_cleanup() {
    let cache = Arc::new(PreparedPlanCache::<TestKey, TestValue>::new(limits(
        8, 4096, 1, 4,
    )));
    let producer_entered = Arc::new(Barrier::new(2));
    let producer_release = Arc::new(Barrier::new(2));

    thread::scope(|scope| {
        let producer = {
            let cache = Arc::clone(&cache);
            let producer_entered = Arc::clone(&producer_entered);
            let producer_release = Arc::clone(&producer_release);
            scope.spawn(move || {
                let result = panic::catch_unwind(AssertUnwindSafe(|| {
                    cache.get_or_prepare(TestKey(41), CacheInFlightBehavior::Wait, 0, || {
                        producer_entered.wait();
                        producer_release.wait();
                        panic!("abort producer for poisoned-cache cleanup test");
                    })
                }));
                assert!(result.is_err());
            })
        };
        producer_entered.wait();

        let waiter = {
            let cache = Arc::clone(&cache);
            scope.spawn(move || {
                cache.get_or_prepare(TestKey(41), CacheInFlightBehavior::Wait, 0, || {
                    panic!("waiter must not become the producer");
                })
            })
        };
        eventually(|| {
            cache
                .stats()
                .is_ok_and(|stats| stats.waits >= 1 && stats.in_flight == 1)
        });

        cache.poison_state_for_test();
        producer_release.wait();

        producer.join().unwrap();
        assert!(waiter.join().unwrap().is_err());
    });

    assert!(matches!(
        cache.stats(),
        Err(RuntimeStateError::Poisoned {
            lock: "prepared-cache.state"
        })
    ));
}

#[test]
fn poisoned_state_stays_visible_after_queue_ticket_cleanup() {
    let cache = Arc::new(PreparedPlanCache::<TestKey, TestValue>::new(limits(
        8, 4096, 1, 4,
    )));
    let producer_entered = Arc::new(Barrier::new(2));
    let producer_release = Arc::new(Barrier::new(2));

    thread::scope(|scope| {
        let producer = {
            let cache = Arc::clone(&cache);
            let producer_entered = Arc::clone(&producer_entered);
            let producer_release = Arc::clone(&producer_release);
            scope.spawn(move || {
                cache.get_or_prepare(TestKey(42), CacheInFlightBehavior::Wait, 0, || {
                    producer_entered.wait();
                    producer_release.wait();
                    CacheProduced::Ready {
                        value: TestValue::new(8),
                        shared: None,
                    }
                })
            })
        };
        producer_entered.wait();

        let queued = {
            let cache = Arc::clone(&cache);
            scope.spawn(move || {
                cache.get_or_prepare(TestKey(43), CacheInFlightBehavior::Wait, 0, || {
                    panic!("queued request must not become the producer");
                })
            })
        };
        eventually(|| {
            cache
                .stats()
                .is_ok_and(|stats| stats.queued_distinct_keys == 1 && stats.in_flight == 1)
        });

        cache.poison_state_for_test();
        producer_release.wait();

        assert!(producer.join().unwrap().is_err());
        assert!(queued.join().unwrap().is_err());
    });

    assert!(matches!(
        cache.stats(),
        Err(RuntimeStateError::Poisoned {
            lock: "prepared-cache.state"
        })
    ));
}

#[test]
fn shared_roots_are_charged_once_and_drop_with_cache() {
    let key_sentinel = Arc::new(());
    let value_sentinel = Arc::new(());
    let shared_sentinel = Arc::new(());
    let key_weak = Arc::downgrade(&key_sentinel);
    let value_weak = Arc::downgrade(&value_sentinel);
    let shared_weak = Arc::downgrade(&shared_sentinel);
    let shared = Arc::new(TestShared {
        bytes: 0,
        sentinel: Some(shared_sentinel),
    });
    assert!(shared.sentinel.is_some());
    let shared_retention = SharedRetention {
        value: Arc::clone(&shared),
        retained_bytes: Some(shared.bytes),
    };
    let cache = PreparedPlanCache::<DropKey, TestValue>::new(limits(4, 4096, 1, 4));
    let value = TestValue {
        bytes: 8,
        sentinel: Some(value_sentinel),
    };
    let expected_without_shared = cache
        .ready_charge_breakdown_for_test(
            &DropKey {
                value: 1,
                sentinel: Arc::clone(&key_sentinel),
                shared: None,
            },
            &value,
            false,
        )
        .unwrap();
    assert_eq!(
        expected_without_shared.total,
        expected_without_shared.key_payload
            + expected_without_shared.value_payload
            + expected_without_shared.entry_record
            + expected_without_shared.compact_bucket_record
            + expected_without_shared.lru_record
            + expected_without_shared.shared_record
    );
    let shared_charge = cache
        .ready_charge_breakdown_for_test(
            &DropKey {
                value: 1,
                sentinel: Arc::clone(&key_sentinel),
                shared: Some(shared_retention.clone()),
            },
            &value,
            true,
        )
        .unwrap()
        .shared_record;

    for key in 1..=2 {
        let key_sentinel = Arc::clone(&key_sentinel);
        let shared_retention = shared_retention.clone();
        cache
            .get_or_prepare(
                DropKey {
                    value: key,
                    sentinel: key_sentinel,
                    shared: Some(shared_retention.clone()),
                },
                CacheInFlightBehavior::Wait,
                0,
                || CacheProduced::Ready {
                    value: Arc::new(TestValue {
                        bytes: 8,
                        sentinel: value.sentinel.clone(),
                    }),
                    shared: Some(shared_retention),
                },
            )
            .unwrap();
    }
    assert_eq!(
        cache.stats().unwrap().retained_bytes,
        expected_without_shared.total * 2 + shared_charge
    );

    drop(shared);
    drop(shared_retention);
    drop(key_sentinel);
    drop(value);
    drop(cache);
    assert!(Weak::upgrade(&key_weak).is_none());
    assert!(Weak::upgrade(&value_weak).is_none());
    assert!(Weak::upgrade(&shared_weak).is_none());
}

#[derive(Debug, thiserror::Error)]
#[error("owner {name} failed")]
struct OwnerFailure {
    name: &'static str,
}

#[derive(Debug)]
struct RecordingOwner {
    name: &'static str,
    stats: CacheStats,
    fail_stats: bool,
    fail_clear: bool,
    stats_calls: Arc<AtomicUsize>,
    clear_calls: Arc<AtomicUsize>,
    order: Arc<Mutex<Vec<String>>>,
}

impl RuntimeCacheOwner for RecordingOwner {
    fn cache_stats(&self) -> Result<CacheStats, CacheOwnerError> {
        self.stats_calls.fetch_add(1, Ordering::SeqCst);
        self.order
            .lock()
            .unwrap()
            .push(format!("stats:{}", self.name));
        if self.fail_stats {
            Err(CacheOwnerError::new(Arc::new(OwnerFailure {
                name: self.name,
            })))
        } else {
            Ok(self.stats)
        }
    }

    fn clear_caches(&self) -> Result<(), CacheOwnerError> {
        self.clear_calls.fetch_add(1, Ordering::SeqCst);
        self.order
            .lock()
            .unwrap()
            .push(format!("clear:{}", self.name));
        if self.fail_clear {
            Err(CacheOwnerError::new(Arc::new(OwnerFailure {
                name: self.name,
            })))
        } else {
            Ok(())
        }
    }
}

#[derive(Debug)]
struct OwnerOnlyModule {
    id: ExtensionModuleId,
    owners: Vec<(CacheOwnerId, Arc<dyn RuntimeCacheOwner>)>,
}

impl ExtensionModule for OwnerOnlyModule {
    fn module_id(&self) -> &ExtensionModuleId {
        &self.id
    }

    fn configure(
        &self,
        registrar: &mut ExtensionModuleRegistrar<'_>,
    ) -> Result<(), ExtensionModuleError> {
        for (id, owner) in &self.owners {
            registrar.register_cache_owner(id.clone(), Arc::clone(owner))?;
        }
        Ok(())
    }
}

fn owner(
    name: &'static str,
    stats: CacheStats,
    fail_stats: bool,
    fail_clear: bool,
    order: &Arc<Mutex<Vec<String>>>,
) -> (Arc<RecordingOwner>, Arc<dyn RuntimeCacheOwner>) {
    let owner = Arc::new(RecordingOwner {
        name,
        stats,
        fail_stats,
        fail_clear,
        stats_calls: Arc::new(AtomicUsize::new(0)),
        clear_calls: Arc::new(AtomicUsize::new(0)),
        order: Arc::clone(order),
    });
    (Arc::clone(&owner), owner)
}

fn engine_with_owner(id: &str, owner: Arc<dyn RuntimeCacheOwner>) -> EngineRegistration {
    let storage = StorageClass::new("tenferro.storage.host").unwrap();
    let provider_device_identity = ProviderDeviceIdentity::new(
        ProviderId::new("tenferro.test.cache").unwrap(),
        format!("engine:{id}"),
    )
    .unwrap();
    EngineRegistration::executable(
        ProviderExecutableBinding::new(
            EngineId::new(id).unwrap(),
            HardwareClassId::new("tenferro.hardware.host").unwrap(),
            Arc::from(vec![storage.clone()]),
            storage.clone(),
            ExecutableEngineContract::new(
                provider_device_identity,
                CoreCapabilityBundle::builder().build(),
                CpuBackend::new(),
                Arc::new(ImmediateEventDomainDriver::new()),
                InputIngressContract::new(
                    InputPlacementContract::new(|_, _| true),
                    InputSignatureContract::new(|_, _, _, _| true),
                    RuntimeInputContract::new(|_, _| true),
                    ResidentOutputContract::new(|_, _| true),
                ),
                Some(owner),
            ),
        )
        .unwrap(),
    )
}

fn cache_stats(
    entries: usize,
    retained_bytes: usize,
    hits: u64,
    misses: u64,
    evictions: u64,
    clears: u64,
) -> CacheStats {
    CacheStats {
        entries,
        retained_bytes,
        hits,
        misses,
        evictions,
        clears,
    }
}

fn canonical_engine_owner_id(engine_id: &str) -> String {
    format!("engine[{}]:{engine_id}", engine_id.len())
}

fn canonical_extension_owner_id(module_id: &str, local_id: &str) -> String {
    format!(
        "extension[{}]:{module_id}[{}]:{local_id}",
        module_id.len(),
        local_id.len()
    )
}

#[test]
fn runtime_cache_set_aggregates_owner_stats_and_clear_failures_in_snapshot_order() {
    let order = Arc::new(Mutex::new(Vec::new()));
    let (engine_b_record, engine_b) = owner(
        "engine_b",
        cache_stats(1, 10, 100, 1, 2, 3),
        false,
        false,
        &order,
    );
    let (engine_a_record, engine_a) = owner(
        "engine_a",
        cache_stats(2, 20, u64::MAX - 1, 4, 5, 6),
        false,
        true,
        &order,
    );
    let (ext_b_record, ext_b) = owner(
        "ext_b",
        cache_stats(3, 30, 7, 8, 9, 10),
        true,
        false,
        &order,
    );
    let (ext_a_record, ext_a) = owner(
        "ext_a",
        cache_stats(4, 40, 11, 12, 13, 14),
        false,
        true,
        &order,
    );

    let mut builder = RuntimeConfigBuilder::new();
    builder
        .register_engine(engine_with_owner("tenferro.engine.b", engine_b))
        .unwrap()
        .register_engine(engine_with_owner("tenferro.engine.a", engine_a))
        .unwrap()
        .install_extension_module(Arc::new(OwnerOnlyModule {
            id: ExtensionModuleId::new("tenferro.module.b").unwrap(),
            owners: vec![(
                CacheOwnerId::new("tenferro.owner.b").unwrap(),
                Arc::clone(&ext_b),
            )],
        }))
        .unwrap()
        .install_extension_module(Arc::new(OwnerOnlyModule {
            id: ExtensionModuleId::new("tenferro.module.a").unwrap(),
            owners: vec![(
                CacheOwnerId::new("tenferro.owner.a").unwrap(),
                Arc::clone(&ext_a),
            )],
        }))
        .unwrap();
    let runtime = builder.build().unwrap();
    let snapshot = runtime.snapshot().unwrap();
    let cache_set = RuntimeCacheSet::<TestKey, TestValue>::new(limits(8, 4096, 1, 4));

    let error = cache_set
        .cache_stats(snapshot.cache_owners_for_test())
        .unwrap_err();
    let RuntimeCacheError::Aggregate { runtime, owners } = error;
    assert!(runtime.is_none());
    assert_eq!(owners.len(), 1);
    assert_eq!(
        owners[0].owner.as_str(),
        canonical_extension_owner_id("tenferro.module.b", "tenferro.owner.b")
    );
    assert_eq!(owners[0].source.to_string(), "owner ext_b failed");
    assert_eq!(
        *order.lock().unwrap(),
        vec![
            "stats:engine_a",
            "stats:engine_b",
            "stats:ext_a",
            "stats:ext_b",
        ]
    );
    assert_eq!(engine_a_record.stats_calls.load(Ordering::SeqCst), 1);
    assert_eq!(engine_b_record.stats_calls.load(Ordering::SeqCst), 1);
    assert_eq!(ext_a_record.stats_calls.load(Ordering::SeqCst), 1);
    assert_eq!(ext_b_record.stats_calls.load(Ordering::SeqCst), 1);

    order.lock().unwrap().clear();
    let clear_error = cache_set
        .clear_caches(snapshot.cache_owners_for_test())
        .unwrap_err();
    let RuntimeCacheError::Aggregate { runtime, owners } = clear_error;
    assert!(runtime.is_none());
    assert_eq!(owners.len(), 2);
    assert_eq!(
        owners[0].owner.as_str(),
        canonical_engine_owner_id("tenferro.engine.a")
    );
    assert_eq!(
        owners[1].owner.as_str(),
        canonical_extension_owner_id("tenferro.module.a", "tenferro.owner.a")
    );
    assert_eq!(
        *order.lock().unwrap(),
        vec![
            "clear:engine_a",
            "clear:engine_b",
            "clear:ext_a",
            "clear:ext_b",
        ]
    );
    assert_eq!(engine_a_record.clear_calls.load(Ordering::SeqCst), 1);
    assert_eq!(engine_b_record.clear_calls.load(Ordering::SeqCst), 1);
    assert_eq!(ext_a_record.clear_calls.load(Ordering::SeqCst), 1);
    assert_eq!(ext_b_record.clear_calls.load(Ordering::SeqCst), 1);
}
