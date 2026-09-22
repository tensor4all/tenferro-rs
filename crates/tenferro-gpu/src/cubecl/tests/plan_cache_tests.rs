//! Pure-logic tests for the hash-keyed LRU plan-cache core. No CUDA hardware
//! is required: the entry type is a plain value standing in for a cached
//! cuTENSOR plan.

use std::num::NonZeroUsize;

use super::super::plan_cache::LruPlanCache;

type TestCache = LruPlanCache<Vec<i64>, &'static str>;

fn cache_with(max_entries: usize) -> TestCache {
    LruPlanCache::new(NonZeroUsize::new(max_entries).unwrap())
}

fn insert(cache: &mut TestCache, hash: u64, key: Vec<i64>, value: &'static str, bytes: usize) {
    cache
        .ensure(
            hash,
            |stored| *stored == key,
            || Ok((key.clone(), value, bytes)),
        )
        .unwrap();
}

#[test]
fn plan_cache_hits_and_misses_are_counted() {
    let mut cache = cache_with(4);
    insert(&mut cache, 1, vec![1], "a", 10);
    assert!(!cache
        .ensure(
            1,
            |stored| stored == &[1],
            || panic!("hit must not rebuild")
        )
        .unwrap());
    let stats = cache.stats();
    assert_eq!(stats.entries, 1);
    assert_eq!(stats.hits, 1);
    assert_eq!(stats.misses, 1);
    assert_eq!(stats.evictions, 0);
    assert_eq!(cache.get(1, |stored| stored == &[1]), Some(&"a"));
    assert_eq!(cache.get(2, |_| true), None);
}

#[test]
fn plan_cache_build_error_leaves_cache_unchanged() {
    let mut cache = cache_with(4);
    let result = cache.ensure(
        9,
        |_| false,
        || {
            Err(crate::Error::runtime_state(
                "test_plan_cache",
                "build failed",
            ))
        },
    );
    assert!(result.is_err());
    let stats = cache.stats();
    assert_eq!(stats.entries, 0);
    assert_eq!(stats.misses, 1);
    assert_eq!(cache.retained_bytes(), std::mem::size_of::<TestCache>());
}

#[test]
fn plan_cache_evicts_least_recently_used_entry() {
    let mut cache = cache_with(2);
    insert(&mut cache, 1, vec![1], "a", 10);
    insert(&mut cache, 2, vec![2], "b", 20);
    // Touch entry 1 so entry 2 becomes least recently used.
    assert!(cache.get(1, |stored| stored == &[1]).is_some());
    insert(&mut cache, 3, vec![3], "c", 30);
    let stats = cache.stats();
    assert_eq!(stats.entries, 2);
    assert_eq!(stats.evictions, 1);
    assert!(cache.get(2, |stored| stored == &[2]).is_none());
    assert!(cache.get(1, |stored| stored == &[1]).is_some());
    assert!(cache.get(3, |stored| stored == &[3]).is_some());
}

#[test]
fn plan_cache_retained_bytes_track_insert_and_evict() {
    let base = std::mem::size_of::<TestCache>();
    let mut cache = cache_with(2);
    assert_eq!(cache.retained_bytes(), base);
    insert(&mut cache, 1, vec![1], "a", 10);
    assert_eq!(cache.retained_bytes(), base + 10);
    insert(&mut cache, 2, vec![2], "b", 20);
    assert_eq!(cache.retained_bytes(), base + 30);
    // Evicts entry 1 (least recently used): 10 bytes leave, 40 arrive.
    insert(&mut cache, 3, vec![3], "c", 40);
    assert_eq!(cache.retained_bytes(), base + 60);
    assert_eq!(cache.stats().retained_bytes, base + 60);
}

#[test]
fn plan_cache_hash_collision_replaces_entry() {
    let mut cache = cache_with(4);
    insert(&mut cache, 7, vec![1], "a", 10);
    // Same hash, different materialized key: the lookup must reject the
    // stored entry and the rebuild must replace it.
    insert(&mut cache, 7, vec![2], "b", 20);
    let stats = cache.stats();
    assert_eq!(stats.entries, 1);
    assert_eq!(stats.misses, 2);
    assert_eq!(stats.evictions, 1);
    assert_eq!(cache.get(7, |stored| stored == &[2]), Some(&"b"));
    assert_eq!(
        cache.retained_bytes(),
        std::mem::size_of::<TestCache>() + 20
    );
}

#[test]
fn plan_cache_shrinking_max_entries_evicts_and_reaccounts() {
    let base = std::mem::size_of::<TestCache>();
    let mut cache = cache_with(3);
    insert(&mut cache, 1, vec![1], "a", 10);
    insert(&mut cache, 2, vec![2], "b", 20);
    insert(&mut cache, 3, vec![3], "c", 40);
    assert_eq!(cache.max_entries().get(), 3);

    cache.set_max_entries(NonZeroUsize::new(1).unwrap());
    assert_eq!(cache.max_entries().get(), 1);
    let stats = cache.stats();
    assert_eq!(stats.entries, 1);
    assert_eq!(stats.evictions, 2);
    assert_eq!(cache.retained_bytes(), base + 40);
    // Only the most recently used entry survives.
    assert!(cache.get(3, |stored| stored == &[3]).is_some());

    // Growing the bound back keeps the retained entry and evicts nothing.
    cache.set_max_entries(NonZeroUsize::new(4).unwrap());
    assert_eq!(cache.stats().entries, 1);
    assert_eq!(cache.stats().evictions, 2);
}

#[test]
fn plan_cache_values_iterates_retained_entries() {
    let mut cache = cache_with(3);
    insert(&mut cache, 1, vec![1], "a", 1);
    insert(&mut cache, 2, vec![2], "b", 1);
    let mut values: Vec<&&str> = cache.values().collect();
    values.sort();
    assert_eq!(values, [&"a", &"b"]);
}
