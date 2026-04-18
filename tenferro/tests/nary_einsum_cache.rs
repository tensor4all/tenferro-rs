//! Tests for the LRU-backed Engine::einsum_cache.

use std::num::NonZeroUsize;

use tenferro::einsum::einsum;
use tenferro::{CpuBackend, Engine, TracedTensor};

fn run_matmul(engine: &mut Engine<CpuBackend>, rows: usize, cols: usize, mid: usize) {
    let a = TracedTensor::from_vec(
        vec![rows, mid],
        (0..rows * mid).map(|i| i as f64).collect::<Vec<_>>(),
    );
    let b = TracedTensor::from_vec(
        vec![mid, cols],
        (0..mid * cols).map(|i| i as f64).collect::<Vec<_>>(),
    );
    let mut c = einsum(engine, &[&a, &b], "ij,jk->ik").expect("einsum");
    c.eval(engine).expect("eval");
}

#[test]
fn default_capacity_is_nonzero() {
    let engine = Engine::new(CpuBackend::new());
    assert_eq!(
        engine.einsum_cache_capacity(),
        NonZeroUsize::new(tenferro::engine::DEFAULT_EINSUM_CACHE_CAPACITY).unwrap(),
    );
}

#[test]
fn with_einsum_cache_capacity_sets_capacity() {
    let cap = NonZeroUsize::new(4).unwrap();
    let engine = Engine::with_einsum_cache_capacity(CpuBackend::new(), cap);
    assert_eq!(engine.einsum_cache_capacity(), cap);
}

#[test]
fn set_einsum_cache_capacity_shrinks_len() {
    let mut engine = Engine::with_einsum_cache_capacity(
        CpuBackend::new(),
        NonZeroUsize::new(10).unwrap(),
    );
    // Populate with 5 distinct einsum shapes (same subscripts, different shapes).
    for k in 1..=5 {
        run_matmul(&mut engine, 2, 2, k);
    }
    assert_eq!(engine.einsum_cache_len(), 5);
    engine.set_einsum_cache_capacity(NonZeroUsize::new(3).unwrap());
    assert_eq!(engine.einsum_cache_len(), 3);
    assert_eq!(engine.einsum_cache_capacity(), NonZeroUsize::new(3).unwrap());
}
