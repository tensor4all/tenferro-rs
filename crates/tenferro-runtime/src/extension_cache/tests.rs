use super::*;

const FAMILY_A: &str = "example.a.v1";
const FAMILY_B: &str = "example.b.v1";
const PLANS: &str = "plans";
const BUFFERS: &str = "buffers";

fn key(family_id: &'static str, cache_name: &'static str, discriminator: u64) -> ExtensionCacheKey {
    ExtensionCacheKey::new(family_id, cache_name, discriminator)
}

fn limits(max_entries: usize) -> ExtensionCacheLimits {
    ExtensionCacheLimits::new(NonZeroUsize::new(max_entries).unwrap())
}

#[test]
fn selector_matches_all_family_and_cache_scopes() {
    let plans = key(FAMILY_A, PLANS, 1);
    let buffers = key(FAMILY_A, BUFFERS, 1);
    let other_family = key(FAMILY_B, PLANS, 1);

    assert!(ExtensionCacheSelector::All.matches(&plans));
    assert!(ExtensionCacheSelector::Family {
        family_id: FAMILY_A,
    }
    .matches(&buffers));
    assert!(!ExtensionCacheSelector::Family {
        family_id: FAMILY_A,
    }
    .matches(&other_family));
    assert!(ExtensionCacheSelector::Cache {
        family_id: FAMILY_A,
        cache_name: PLANS,
    }
    .matches(&plans));
    assert!(!ExtensionCacheSelector::Cache {
        family_id: FAMILY_A,
        cache_name: PLANS,
    }
    .matches(&buffers));
}

#[test]
fn store_put_get_get_mut_and_stats_are_typed() {
    let mut store = ExtensionCacheStore::new();
    let plan = key(FAMILY_A, PLANS, 7);
    let buffer = key(FAMILY_A, BUFFERS, 3);

    assert!(store.is_empty());
    assert_eq!(store.limits(), ExtensionCacheLimits::default());
    store.put(plan, String::from("plan-a"), 32);
    store.put(buffer, vec![1_usize, 2, 3], 24);

    assert_eq!(store.len(), 2);
    assert_eq!(
        store.get::<String>(&plan).map(String::as_str),
        Some("plan-a")
    );
    assert!(store.get::<Vec<usize>>(&plan).is_none());
    store.get_mut::<Vec<usize>>(&buffer).unwrap().push(4);
    assert_eq!(store.get::<Vec<usize>>(&buffer).unwrap(), &[1, 2, 3, 4]);

    let family_stats = store.stats(ExtensionCacheSelector::Family {
        family_id: FAMILY_A,
    });
    assert_eq!(family_stats.entries, 2);
    assert_eq!(family_stats.retained_bytes, 56);

    let cache_stats = store.stats(ExtensionCacheSelector::Cache {
        family_id: FAMILY_A,
        cache_name: PLANS,
    });
    assert_eq!(cache_stats.entries, 1);
    assert_eq!(cache_stats.retained_bytes, 32);
}

#[test]
fn dynamic_retained_bytes_follow_mutated_cache_entries() {
    let mut store = ExtensionCacheStore::new();
    let buffers = key(FAMILY_A, BUFFERS, 9);

    store.put_with_retained_bytes(buffers, Vec::<usize>::with_capacity(2), |values| {
        values.capacity() * std::mem::size_of::<usize>()
    });
    assert_eq!(
        store.stats(ExtensionCacheSelector::All).retained_bytes,
        2 * std::mem::size_of::<usize>()
    );

    let values = store.get_mut::<Vec<usize>>(&buffers).unwrap();
    values.reserve_exact(8);
    let retained_capacity = values.capacity();

    assert_eq!(
        store.stats(ExtensionCacheSelector::All).retained_bytes,
        retained_capacity * std::mem::size_of::<usize>()
    );
}

#[test]
fn store_stats_saturate_retained_bytes() {
    let mut store = ExtensionCacheStore::new();
    store.put(key(FAMILY_A, PLANS, 1), 1_u64, usize::MAX);
    store.put(key(FAMILY_A, PLANS, 2), 2_u64, usize::MAX);

    assert_eq!(
        store.stats(ExtensionCacheSelector::All).retained_bytes,
        usize::MAX
    );
}

#[test]
fn store_stats_track_hits_misses_evictions_and_clears() {
    let mut store = ExtensionCacheStore::with_limits(limits(2));
    let first = key(FAMILY_A, PLANS, 1);
    let second = key(FAMILY_A, PLANS, 2);
    let third = key(FAMILY_A, PLANS, 3);

    assert!(store.get::<u64>(&first).is_none());
    store.put(first, 1_u64, 8);
    store.put(second, 2_u64, 8);
    assert_eq!(store.get::<u64>(&first), Some(&1));
    assert!(store.get::<String>(&first).is_none());
    store.put(third, 3_u64, 8);

    let stats = store.stats(ExtensionCacheSelector::All);
    assert_eq!(stats.entries, 2);
    assert_eq!(stats.hits, 1);
    assert_eq!(stats.misses, 2);
    assert_eq!(stats.evictions, 1);
    assert_eq!(stats.clears, 0);

    store.clear();
    let stats = store.stats(ExtensionCacheSelector::All);
    assert_eq!(stats.entries, 0);
    assert_eq!(stats.hits, 1);
    assert_eq!(stats.misses, 2);
    assert_eq!(stats.evictions, 1);
    assert_eq!(stats.clears, 1);
}

#[test]
fn store_stats_scope_events_to_selected_family_and_cache() {
    let mut store = ExtensionCacheStore::with_limits(limits(2));
    let a_plan = key(FAMILY_A, PLANS, 1);
    let a_buffer = key(FAMILY_A, BUFFERS, 1);
    let b_plan = key(FAMILY_B, PLANS, 1);

    store.put(a_plan, 1_u64, 8);
    store.put(a_buffer, 2_u64, 16);
    assert_eq!(store.get::<u64>(&a_plan), Some(&1));
    assert!(store.get::<u64>(&b_plan).is_none());
    assert!(store.get::<String>(&a_buffer).is_none());
    store.put(b_plan, 3_u64, 32);

    let all_stats = store.stats(ExtensionCacheSelector::All);
    assert_eq!(all_stats.entries, 2);
    assert_eq!(all_stats.hits, 1);
    assert_eq!(all_stats.misses, 2);
    assert_eq!(all_stats.evictions, 1);

    let family_a_stats = store.stats(ExtensionCacheSelector::Family {
        family_id: FAMILY_A,
    });
    assert_eq!(family_a_stats.entries, 1);
    assert_eq!(family_a_stats.retained_bytes, 16);
    assert_eq!(family_a_stats.hits, 1);
    assert_eq!(family_a_stats.misses, 1);
    assert_eq!(family_a_stats.evictions, 1);

    let family_b_stats = store.stats(ExtensionCacheSelector::Family {
        family_id: FAMILY_B,
    });
    assert_eq!(family_b_stats.entries, 1);
    assert_eq!(family_b_stats.retained_bytes, 32);
    assert_eq!(family_b_stats.hits, 0);
    assert_eq!(family_b_stats.misses, 1);
    assert_eq!(family_b_stats.evictions, 0);

    let a_plan_stats = store.stats(ExtensionCacheSelector::Cache {
        family_id: FAMILY_A,
        cache_name: PLANS,
    });
    assert_eq!(a_plan_stats.entries, 0);
    assert_eq!(a_plan_stats.hits, 1);
    assert_eq!(a_plan_stats.misses, 0);
    assert_eq!(a_plan_stats.evictions, 1);

    let a_buffer_stats = store.stats(ExtensionCacheSelector::Cache {
        family_id: FAMILY_A,
        cache_name: BUFFERS,
    });
    assert_eq!(a_buffer_stats.entries, 1);
    assert_eq!(a_buffer_stats.hits, 0);
    assert_eq!(a_buffer_stats.misses, 1);
    assert_eq!(a_buffer_stats.evictions, 0);

    store.clear_selected(ExtensionCacheSelector::Family {
        family_id: FAMILY_A,
    });
    assert_eq!(
        store
            .stats(ExtensionCacheSelector::Family {
                family_id: FAMILY_A,
            })
            .clears,
        1
    );
    assert_eq!(
        store
            .stats(ExtensionCacheSelector::Family {
                family_id: FAMILY_B,
            })
            .clears,
        0
    );

    let fresh_cache = key(FAMILY_A, BUFFERS, 99);
    store.put(fresh_cache, 99_u64, 8);
    store.clear_selected(ExtensionCacheSelector::Family {
        family_id: FAMILY_A,
    });
    assert_eq!(
        store
            .stats(ExtensionCacheSelector::Cache {
                family_id: FAMILY_A,
                cache_name: BUFFERS,
            })
            .clears,
        2
    );
}

#[test]
fn clear_selected_removes_only_matching_entries() {
    let mut store = ExtensionCacheStore::new();
    let a_plan = key(FAMILY_A, PLANS, 1);
    let a_buffer = key(FAMILY_A, BUFFERS, 1);
    let b_plan = key(FAMILY_B, PLANS, 1);
    store.put(a_plan, 1_u64, 8);
    store.put(a_buffer, 2_u64, 8);
    store.put(b_plan, 3_u64, 8);

    store.clear_selected(ExtensionCacheSelector::Cache {
        family_id: FAMILY_A,
        cache_name: PLANS,
    });
    assert!(store.get::<u64>(&a_plan).is_none());
    assert_eq!(store.get::<u64>(&a_buffer), Some(&2));
    assert_eq!(store.get::<u64>(&b_plan), Some(&3));

    store.clear_selected(ExtensionCacheSelector::Family {
        family_id: FAMILY_A,
    });
    assert!(store.get::<u64>(&a_buffer).is_none());
    assert_eq!(store.get::<u64>(&b_plan), Some(&3));

    store.clear_selected(ExtensionCacheSelector::All);
    assert!(store.is_empty());
}

#[test]
fn store_limits_evict_least_recently_used_entries() {
    let mut store = ExtensionCacheStore::with_limits(limits(2));
    let first = key(FAMILY_A, PLANS, 1);
    let second = key(FAMILY_A, PLANS, 2);
    let third = key(FAMILY_A, PLANS, 3);

    store.put(first, 1_u64, 8);
    store.put(second, 2_u64, 8);
    assert_eq!(store.get::<u64>(&first), Some(&1));
    store.put(third, 3_u64, 8);

    assert_eq!(store.len(), 2);
    assert_eq!(store.get::<u64>(&first), Some(&1));
    assert!(store.get::<u64>(&second).is_none());
    assert_eq!(store.get::<u64>(&third), Some(&3));

    store.set_limits(limits(1));
    assert_eq!(store.limits().max_entries().get(), 1);
    assert_eq!(store.len(), 1);
    store.clear();
    assert!(store.is_empty());
}
