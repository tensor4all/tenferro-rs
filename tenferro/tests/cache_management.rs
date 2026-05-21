use std::num::NonZeroUsize;

use tenferro::traced_tensor::einsum;
use tenferro::{CpuBackend, Engine, TracedTensor};

fn eval_add(engine: &mut Engine<CpuBackend>) {
    let x = TracedTensor::from_vec(vec![2], vec![1.0_f64, 2.0]);
    let mut y = &x + &x;
    y.eval(engine).expect("add eval");
}

fn eval_neg(engine: &mut Engine<CpuBackend>) {
    let x = TracedTensor::from_vec(vec![2], vec![1.0_f64, 2.0]);
    let mut y = x.neg();
    y.eval(engine).expect("neg eval");
}

fn eval_nary_einsum(engine: &mut Engine<CpuBackend>, mid: usize) {
    let a = TracedTensor::from_vec(vec![2, mid], vec![1.0_f64; 2 * mid]);
    let b = TracedTensor::from_vec(vec![mid, 3], vec![1.0_f64; mid * 3]);
    let c = TracedTensor::from_vec(vec![3, 2], vec![1.0_f64; 6]);
    let mut out = einsum(engine, &[&a, &b, &c], "ij,jk,kl->il").expect("einsum");
    out.eval(engine).expect("einsum eval");
}

#[test]
fn compile_cache_is_bounded_and_reports_stats() {
    let mut engine = Engine::new(CpuBackend::new());
    engine.set_compile_cache_capacity(NonZeroUsize::new(1).unwrap());

    eval_add(&mut engine);
    eval_neg(&mut engine);

    assert_eq!(engine.compile_cache_capacity().get(), 1);
    let stats = engine.cache_stats();
    assert_eq!(stats.compile.entries, 1);
    assert!(stats.compile.retained_bytes > 0);
}

#[test]
fn clear_caches_clears_engine_owned_cache_entries() {
    let mut engine = Engine::new(CpuBackend::new());

    eval_add(&mut engine);
    eval_nary_einsum(&mut engine, 3);

    let before = engine.cache_stats();
    assert!(before.compile.entries > 0);
    assert!(before.einsum_plans.entries > 0);
    assert!(before.einsum_parse.entries > 0);

    engine.clear_caches();

    let after = engine.cache_stats();
    assert_eq!(after.compile.entries, 0);
    assert_eq!(after.einsum_plans.entries, 0);
    assert_eq!(after.einsum_parse.entries, 0);
    assert_eq!(after.backend.entries, 0);
}

#[test]
fn cpu_cache_stats_include_buffer_pool_and_clear_all_caches() {
    let mut engine = Engine::new(CpuBackend::new());

    eval_add(&mut engine);
    let before = engine.cpu_cache_stats();
    assert!(before.engine.compile.entries > 0);
    assert_eq!(
        before.buffer_pool.retained_bytes,
        engine.buffer_pool_stats().capacity_bytes
    );

    engine.clear_all_caches();

    let after = engine.cpu_cache_stats();
    assert_eq!(after.engine.compile.entries, 0);
    assert_eq!(after.engine.einsum_plans.entries, 0);
    assert_eq!(after.engine.einsum_parse.entries, 0);
    assert_eq!(after.engine.backend.entries, 0);
    assert_eq!(after.buffer_pool.entries, 0);
    assert_eq!(engine.buffer_pool_len(), 0);
}
