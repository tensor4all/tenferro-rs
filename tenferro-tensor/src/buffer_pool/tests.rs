use std::mem::size_of;

use super::{BufferPool, PoolScalar};

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
