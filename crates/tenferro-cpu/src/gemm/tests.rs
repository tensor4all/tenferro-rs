use tenferro_tensor::RuntimeCacheControl;

use super::*;

#[test]
fn capacity_and_clear_are_reported_through_the_cache_stats() {
    let mut cache = GemmAnalysisCache::default();
    assert_eq!(cache.capacity(), DEFAULT_GEMM_ANALYSIS_CACHE_CAPACITY);
    assert!(format!("{cache:?}").contains("GemmAnalysisCache"));

    cache.set_capacity(3);
    assert_eq!(cache.capacity(), 3);
    assert_eq!(cache.stats().clears, 0);

    cache.clear();
    let stats = cache.stats();
    assert_eq!(stats.clears, 1);
    assert_eq!(stats.entries, 0);
    assert!(format!("{cache:?}").contains("max_slots"));
}

#[test]
fn a_zero_capacity_cache_retains_no_plans() {
    let mut cache = GemmAnalysisCache::with_capacity(4);
    cache.set_capacity(0);
    assert_eq!(cache.capacity(), 0);
    assert_eq!(cache.stats().entries, 0);
}
