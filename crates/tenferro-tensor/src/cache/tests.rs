use super::{CacheStats, RuntimeCacheControl};

#[test]
fn cache_stats_empty_reports_zeroes() {
    assert_eq!(
        CacheStats::empty(),
        CacheStats {
            entries: 0,
            retained_bytes: 0
        }
    );
}

#[test]
fn unit_runtime_cache_control_is_empty_and_clearable() {
    let mut cache = ();
    assert_eq!(cache.stats(), CacheStats::empty());
    cache.clear();
    assert_eq!(cache.stats().entries, 0);
}
