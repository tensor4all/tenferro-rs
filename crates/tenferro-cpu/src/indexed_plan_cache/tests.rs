use super::*;
use strided_kernel::ErasedDynamicSlicePlan;

fn key(size: usize) -> IndexedPlanKey {
    IndexedPlanKey::from_slices(
        IndexedPlanFamily::DynamicSlice,
        KernelDType::F64,
        KernelDType::I64,
        &[&[size], &[1], &[size]],
        &[&[1], &[1], &[1]],
        &[&[size]],
    )
}

fn plan(size: usize) -> ErasedDynamicSlicePlan {
    ErasedDynamicSlicePlan::compile(
        KernelDType::F64,
        KernelDType::I64,
        &[size],
        &[1],
        &[1],
        &[1],
        &[size],
        &[1],
        &[size],
    )
    .expect("valid test plan")
}

#[test]
fn key_distinguishes_family_layout_and_config_boundaries() {
    let base = key(8);
    let different_layout = key(16);
    let different_family = IndexedPlanKey::from_slices(
        IndexedPlanFamily::Gather,
        KernelDType::F64,
        KernelDType::I64,
        &[&[8], &[1], &[8]],
        &[&[1], &[1], &[1]],
        &[&[8]],
    );
    let different_config_boundary = IndexedPlanKey::from_slices(
        IndexedPlanFamily::DynamicSlice,
        KernelDType::F64,
        KernelDType::I64,
        &[&[8], &[1], &[8]],
        &[&[1], &[1], &[1]],
        &[&[], &[8]],
    );

    assert_ne!(base, different_layout);
    assert_ne!(base, different_family);
    assert_ne!(base, different_config_boundary);
}

#[test]
fn repeated_key_compiles_once_and_reports_hit() {
    let mut cache = IndexedPlanCache::new(IndexedPlanCacheLimits::new(4, usize::MAX));
    let mut compiles = 0;

    for _ in 0..2 {
        cache
            .dynamic_slice(key(8), || {
                compiles += 1;
                Ok::<_, std::convert::Infallible>(plan(8))
            })
            .expect("cache lookup");
    }

    assert_eq!(compiles, 1);
    assert_eq!(cache.stats().entries, 1);
    assert_eq!(cache.stats().hits, 1);
    assert_eq!(cache.stats().misses, 1);
    assert!(cache.stats().retained_bytes > 0);
}

#[test]
fn entry_limit_evicts_least_recently_used_plan() {
    let mut cache = IndexedPlanCache::new(IndexedPlanCacheLimits::new(2, usize::MAX));
    for size in [4, 8] {
        cache
            .dynamic_slice(key(size), || Ok::<_, std::convert::Infallible>(plan(size)))
            .expect("insert plan");
    }
    cache
        .dynamic_slice(key(4), || Ok::<_, std::convert::Infallible>(plan(4)))
        .expect("touch first plan");
    cache
        .dynamic_slice(key(16), || Ok::<_, std::convert::Infallible>(plan(16)))
        .expect("insert third plan");

    let misses_before = cache.stats().misses;
    cache
        .dynamic_slice(key(8), || Ok::<_, std::convert::Infallible>(plan(8)))
        .expect("recompile evicted plan");

    assert_eq!(cache.stats().entries, 2);
    assert_eq!(cache.stats().evictions, 2);
    assert_eq!(cache.stats().misses, misses_before + 1);
}

#[test]
fn byte_limit_and_clear_are_accounted() {
    let mut cache = IndexedPlanCache::new(IndexedPlanCacheLimits::new(8, usize::MAX));
    cache
        .dynamic_slice(key(8), || Ok::<_, std::convert::Infallible>(plan(8)))
        .expect("insert plan");
    let one_entry_bytes = cache.stats().retained_bytes;
    cache.set_limits(IndexedPlanCacheLimits::new(8, one_entry_bytes - 1));

    assert_eq!(cache.stats().entries, 0);
    assert_eq!(cache.stats().retained_bytes, 0);
    assert_eq!(cache.stats().evictions, 1);

    cache.clear();
    assert_eq!(cache.stats().clears, 1);
}

#[test]
fn retained_bytes_charge_inline_and_spilled_plan_payloads() {
    let inline = key(8);
    assert_eq!(inline.retained_bytes(), 0);
    assert!(inline.logical_payload_bytes() > 0);

    let rank = 12;
    let dims = vec![2usize; rank];
    let strides = vec![1isize; rank];
    let spilled = IndexedPlanKey::from_slices(
        IndexedPlanFamily::DynamicSlice,
        KernelDType::F64,
        KernelDType::I64,
        &[&dims, &[1], &dims],
        &[&strides, &[1], &strides],
        &[&dims],
    );
    assert!(spilled.retained_bytes() > 0);
    assert!(
        spilled.logical_payload_bytes() > inline.logical_payload_bytes(),
        "higher-rank plans must carry a larger logical retention charge"
    );
}
