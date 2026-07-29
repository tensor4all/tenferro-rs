use std::mem::{size_of, ManuallyDrop};

use super::{
    parse_default_max_retained_capacity_bytes, BufferPool, PoolScalar,
    DEFAULT_MAX_RETAINED_CAPACITY_BYTES,
};

#[test]
fn default_retention_limit_parser_covers_missing_invalid_zero_and_valid_values() {
    assert_eq!(
        parse_default_max_retained_capacity_bytes(None),
        DEFAULT_MAX_RETAINED_CAPACITY_BYTES
    );
    assert_eq!(
        parse_default_max_retained_capacity_bytes(Some("invalid".into())),
        DEFAULT_MAX_RETAINED_CAPACITY_BYTES
    );
    assert_eq!(
        parse_default_max_retained_capacity_bytes(Some("0".into())),
        0
    );
    assert_eq!(
        parse_default_max_retained_capacity_bytes(Some("4096".into())),
        4096
    );
}

#[cfg(unix)]
#[test]
fn default_retention_limit_parser_rejects_non_unicode_values() {
    use std::os::unix::ffi::OsStringExt;

    assert_eq!(
        parse_default_max_retained_capacity_bytes(Some(std::ffi::OsString::from_vec(vec![0xff]))),
        DEFAULT_MAX_RETAINED_CAPACITY_BYTES
    );
}

#[test]
fn acquire_release_reuse() {
    let mut pool = BufferPool::new();

    let buf = unsafe { <f64 as PoolScalar>::pool_acquire(&mut pool, 64) };
    let ptr = buf.as_ptr();
    let cap = buf.capacity();
    <f64 as PoolScalar>::pool_release(&mut pool, buf);

    let reused = unsafe { <f64 as PoolScalar>::pool_acquire(&mut pool, 64) };
    assert_eq!(reused.as_ptr(), ptr);
    assert_eq!(reused.capacity(), cap);
    assert!(pool.is_empty());
}

#[test]
fn best_fit() {
    let mut pool = BufferPool::new();
    <f64 as PoolScalar>::pool_release(&mut pool, Vec::with_capacity(100));
    <f64 as PoolScalar>::pool_release(&mut pool, Vec::with_capacity(200));
    <f64 as PoolScalar>::pool_release(&mut pool, Vec::with_capacity(300));

    let reused = unsafe { <f64 as PoolScalar>::pool_acquire(&mut pool, 150) };
    assert_eq!(reused.capacity(), 200);
    assert_eq!(pool.len(), 2);
}

#[test]
fn type_separation() {
    let mut pool = BufferPool::new();
    <f64 as PoolScalar>::pool_release(&mut pool, Vec::with_capacity(16));
    assert_eq!(pool.len(), 1);

    let f32_buf = unsafe { <f32 as PoolScalar>::pool_acquire(&mut pool, 16) };
    assert_eq!(f32_buf.capacity(), 16);
    assert_eq!(pool.len(), 1);

    let f64_buf = unsafe { <f64 as PoolScalar>::pool_acquire(&mut pool, 16) };
    assert_eq!(f64_buf.capacity(), 16);
    assert!(pool.is_empty());
}

#[test]
fn fresh_alloc_fallback() {
    let mut pool = BufferPool::new();
    let buf = unsafe { <f64 as PoolScalar>::pool_acquire(&mut pool, 32) };
    assert_eq!(buf.len(), 32);
    assert!(buf.capacity() >= 32);
    assert!(pool.is_empty());
}

#[test]
fn zeroed_acquire_initializes_fresh_and_reused_buffers() {
    let mut pool = BufferPool::new();

    let fresh = <f64 as PoolScalar>::pool_acquire_zeroed(&mut pool, 4);
    assert_eq!(fresh, vec![0.0; 4]);

    <f64 as PoolScalar>::pool_release(&mut pool, vec![7.0, 8.0, 9.0, 10.0]);
    let reused = <f64 as PoolScalar>::pool_acquire_zeroed(&mut pool, 4);
    assert_eq!(reused, vec![0.0; 4]);
}

#[test]
fn raw_acquire_does_not_zero_initialized_reused_buffers() {
    let mut pool = BufferPool::new();

    <f64 as PoolScalar>::pool_release(&mut pool, vec![7.0, 8.0, 9.0, 10.0]);
    let reused = unsafe { <f64 as PoolScalar>::pool_acquire(&mut pool, 4) };

    assert_eq!(reused, vec![7.0, 8.0, 9.0, 10.0]);
}

#[test]
fn uninit_acquire_reuses_storage_and_can_be_initialized_before_release() {
    let mut pool = BufferPool::new();
    let original = vec![true; 4];
    let ptr = original.as_ptr();
    <bool as PoolScalar>::pool_release(&mut pool, original);

    let mut reused = <bool as PoolScalar>::pool_acquire_uninit(&mut pool, 4);
    assert_eq!(reused.as_ptr().cast::<bool>(), ptr);
    reused.iter_mut().for_each(|value| {
        value.write(false);
    });
    let mut reused = ManuallyDrop::new(reused);
    let initialized = unsafe {
        Vec::from_raw_parts(
            reused.as_mut_ptr().cast::<bool>(),
            reused.len(),
            reused.capacity(),
        )
    };
    assert_eq!(initialized, vec![false; 4]);
    <bool as PoolScalar>::pool_release(&mut pool, initialized);
    assert_eq!(pool.len(), 1);
}

#[test]
fn uninit_acquire_error_cleanup_drops_owner_and_clears_in_flight_accounting() {
    let mut pool = BufferPool::new();
    <bool as PoolScalar>::pool_release(&mut pool, Vec::with_capacity(4));

    let in_flight = <bool as PoolScalar>::pool_acquire_uninit(&mut pool, 4);
    drop(in_flight);
    pool.clear_in_flight_retained();

    assert_eq!(pool.stats().buffers, 0);
    assert_eq!(pool.stats().capacity_bytes, 0);
    assert!(pool.bool_in_flight.is_empty());
}

#[test]
fn uninit_acquire_panic_cleanup_replenishes_only_an_empty_replacement() {
    let mut pool = BufferPool::new();
    <bool as PoolScalar>::pool_release(&mut pool, Vec::with_capacity(4));

    let mut in_flight = <bool as PoolScalar>::pool_acquire_uninit(&mut pool, 4);
    in_flight[0].write(true);
    drop(in_flight);
    pool.replenish_in_flight_retained();

    let replacements = pool.bool_pool.get(&4).unwrap();
    assert_eq!(replacements.len(), 1);
    assert!(replacements[0].is_empty());
    assert_eq!(pool.retained_capacity_bytes(), 4 * size_of::<bool>());
    assert!(pool.bool_in_flight.is_empty());
}

#[test]
fn uninit_acquire_success_release_is_accounted_once() {
    let mut pool = BufferPool::new();
    <bool as PoolScalar>::pool_release(&mut pool, Vec::with_capacity(4));
    let expected = pool.stats();

    let mut reused = <bool as PoolScalar>::pool_acquire_uninit(&mut pool, 4);
    reused.iter_mut().for_each(|value| {
        value.write(true);
    });
    let mut reused = ManuallyDrop::new(reused);
    let initialized = unsafe {
        Vec::from_raw_parts(
            reused.as_mut_ptr().cast::<bool>(),
            reused.len(),
            reused.capacity(),
        )
    };
    <bool as PoolScalar>::pool_release(&mut pool, initialized);
    pool.replenish_in_flight_retained();

    assert_eq!(pool.stats(), expected);
    assert!(pool.bool_in_flight.is_empty());
}

#[test]
fn zero_len_not_pooled() {
    let mut pool = BufferPool::new();
    <f64 as PoolScalar>::pool_release(&mut pool, Vec::new());
    assert!(pool.is_empty());
}

#[test]
fn acquire_with_capacity_reuses_buffer_as_empty_vec() {
    let mut pool = BufferPool::new();

    let buf = vec![1.0_f64; 8];
    let ptr = buf.as_ptr();
    let cap = buf.capacity();
    <f64 as PoolScalar>::pool_release(&mut pool, buf);

    let reused = pool.acquire_with_capacity::<f64>(8);
    assert_eq!(reused.as_ptr(), ptr);
    assert_eq!(reused.len(), 0);
    assert_eq!(reused.capacity(), cap);
    assert!(pool.is_empty());
}

#[test]
fn acquire_updates_retained_capacity_stats() {
    let mut pool = BufferPool::new();
    <f64 as PoolScalar>::pool_release(&mut pool, Vec::with_capacity(8));
    assert_eq!(pool.retained_capacity_bytes(), 8 * size_of::<f64>());

    let _reused = unsafe { <f64 as PoolScalar>::pool_acquire(&mut pool, 4) };

    assert_eq!(pool.retained_capacity_bytes(), 0);
    assert!(pool.is_empty());
}

#[test]
fn replenish_in_flight_retained_restores_lost_capacity() {
    let mut pool = BufferPool::new();
    <f64 as PoolScalar>::pool_release(&mut pool, Vec::with_capacity(8));

    let _in_flight = unsafe { <f64 as PoolScalar>::pool_acquire(&mut pool, 8) };
    assert_eq!(pool.retained_capacity_bytes(), 0);
    assert!(pool.is_empty());

    pool.replenish_in_flight_retained();

    assert_eq!(pool.len(), 1);
    assert_eq!(pool.retained_capacity_bytes(), 8 * size_of::<f64>());
    assert!(!pool.is_empty());
}

#[test]
fn replenish_in_flight_retained_skips_successfully_released_buffers() {
    let mut pool = BufferPool::new();
    <f64 as PoolScalar>::pool_release(&mut pool, Vec::with_capacity(8));

    let buf = unsafe { <f64 as PoolScalar>::pool_acquire(&mut pool, 8) };
    <f64 as PoolScalar>::pool_release(&mut pool, buf);

    let stats_before = pool.stats();
    pool.replenish_in_flight_retained();

    assert_eq!(pool.stats(), stats_before);
    assert_eq!(pool.len(), 1);
    assert_eq!(pool.retained_capacity_bytes(), 8 * size_of::<f64>());
}

#[test]
fn stats_counts_typed_capacity_bytes() {
    let mut pool = BufferPool::new();
    <f64 as PoolScalar>::pool_release(&mut pool, Vec::with_capacity(3));
    <f32 as PoolScalar>::pool_release(&mut pool, Vec::with_capacity(5));
    <num_complex::Complex64 as PoolScalar>::pool_release(&mut pool, Vec::with_capacity(7));

    let stats = pool.stats();
    assert_eq!(stats.buffers, 3);
    assert_eq!(
        stats.capacity_bytes,
        3 * size_of::<f64>() + 5 * size_of::<f32>() + 7 * size_of::<num_complex::Complex64>()
    );
    assert_eq!(pool.retained_capacity_bytes(), stats.capacity_bytes);
}

#[test]
fn clear_drops_retained_buffers() {
    let mut pool = BufferPool::new();
    <f64 as PoolScalar>::pool_release(&mut pool, Vec::with_capacity(11));
    <f32 as PoolScalar>::pool_release(&mut pool, Vec::with_capacity(13));

    assert!(!pool.is_empty());
    assert!(pool.retained_capacity_bytes() > 0);

    pool.clear();

    assert!(pool.is_empty());
    assert_eq!(pool.stats(), Default::default());
}

#[test]
fn retention_limit_evicts_smallest_obsolete_buffers() {
    let mut pool = BufferPool::with_max_retained_capacity_bytes(200);
    <f64 as PoolScalar>::pool_release(&mut pool, Vec::with_capacity(10));
    <f64 as PoolScalar>::pool_release(&mut pool, Vec::with_capacity(20));

    assert_eq!(pool.stats().buffers, 1);
    assert_eq!(pool.retained_capacity_bytes(), 20 * size_of::<f64>());

    let reused = unsafe { <f64 as PoolScalar>::pool_acquire(&mut pool, 10) };
    assert_eq!(reused.capacity(), 20);
    assert!(pool.is_empty());
}

#[test]
fn zero_retention_limit_drops_released_buffers() {
    let mut pool = BufferPool::with_max_retained_capacity_bytes(0);
    <f64 as PoolScalar>::pool_release(&mut pool, Vec::with_capacity(10));

    assert!(pool.is_empty());
    assert_eq!(pool.retained_capacity_bytes(), 0);
}

#[test]
fn retention_limit_documents_zero_byte_eviction_progress() {
    let source = include_str!("../buffer_pool.rs");
    assert!(
        source.contains("evicted_bytes == 0"),
        "retention-limit eviction must explicitly handle zero-byte retained entries"
    );
}
