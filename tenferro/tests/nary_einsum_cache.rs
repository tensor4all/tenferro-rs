//! Tests for the LRU-backed Engine::einsum_cache.

use std::num::NonZeroUsize;

use tenferro::traced_tensor::einsum;
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
    let mut engine =
        Engine::with_einsum_cache_capacity(CpuBackend::new(), NonZeroUsize::new(10).unwrap());
    // Populate with 5 distinct einsum shapes (same subscripts, different shapes).
    for k in 1..=5 {
        run_matmul(&mut engine, 2, 2, k);
    }
    assert_eq!(engine.einsum_cache_len(), 5);
    engine.set_einsum_cache_capacity(NonZeroUsize::new(3).unwrap());
    assert_eq!(engine.einsum_cache_len(), 3);
    assert_eq!(
        engine.einsum_cache_capacity(),
        NonZeroUsize::new(3).unwrap()
    );
}

#[test]
fn lru_eviction_preserves_recently_used() {
    let mut engine =
        Engine::with_einsum_cache_capacity(CpuBackend::new(), NonZeroUsize::new(2).unwrap());

    // Three distinct cache keys via shapes A, B, C.
    // Sequence: A (miss), B (miss), A (hit — now MRU), C (miss — evicts B).
    // Expected final state: A and C present, B evicted.

    let key_a = ("ij,jk->ik".to_string(), vec![vec![2, 3], vec![3, 2]]);
    let key_b = ("ij,jk->ik".to_string(), vec![vec![2, 4], vec![4, 2]]);
    let key_c = ("ij,jk->ik".to_string(), vec![vec![2, 5], vec![5, 2]]);

    run_matmul(&mut engine, 2, 2, 3); // A
    run_matmul(&mut engine, 2, 2, 4); // B
    run_matmul(&mut engine, 2, 2, 3); // A again — should be a hit, moves A to MRU
    run_matmul(&mut engine, 2, 2, 5); // C — cache full, evicts LRU (which is B)

    assert_eq!(engine.einsum_cache_len(), 2);
    assert!(
        engine.einsum_cache_contains(&key_a),
        "A should be retained (MRU)"
    );
    assert!(!engine.einsum_cache_contains(&key_b), "B should be evicted");
    assert!(
        engine.einsum_cache_contains(&key_c),
        "C should be present (just inserted)"
    );
}

/// When an ExecProgram containing a NaryEinsum instruction is evaluated twice
/// through Engine::eval_exec_ir with identical inputs, the second call must hit
/// the cache — `einsum_cache_len()` stays at 1 after both runs.
#[test]
fn nary_einsum_on_exec_path_hits_cache() {
    use tenferro::{Tensor, TracedTensor};
    use tenferro_tensor::DType;

    let mut engine = Engine::new(CpuBackend::new());

    // Build an einsum with at least one symbolic-shape input so the graph keeps
    // a NaryEinsum op (not decomposed at build time).
    let a = TracedTensor::input_symbolic_shape(DType::F64, 2);
    let b = TracedTensor::from_vec(vec![3, 4], vec![1.0_f64; 12]);
    let mut c = einsum(&mut engine, &[&a, &b], "ij,jk->ik").expect("einsum");

    // Concrete input for the symbolic leg.
    let a_concrete = Tensor::from_vec(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);

    // First eval: miss -> inserts one entry.
    c.eval_with_inputs(&mut engine, &[(&a, &a_concrete)])
        .expect("eval 1");
    let len_after_first = engine.einsum_cache_len();
    assert_eq!(
        len_after_first, 1,
        "expected one cache entry after first eval"
    );

    // Second eval with the same concrete input: must hit the cache.
    c.eval_with_inputs(&mut engine, &[(&a, &a_concrete)])
        .expect("eval 2");
    let len_after_second = engine.einsum_cache_len();
    assert_eq!(
        len_after_second, 1,
        "cache len must stay at 1 on repeated identical (subscripts, shapes)"
    );
}
