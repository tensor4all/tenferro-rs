use tenferro::{CpuBackend, GraphCompiler, GraphExecutor, TracedTensor};

#[test]
fn compiler_clear_caches_clears_compile_entries() {
    let mut compiler = GraphCompiler::new();

    let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
    let y = &x + &x;
    let _ = compiler.compile(&y).expect("compile");

    let before = compiler.cache_stats();
    assert!(before.compile.entries > 0);

    compiler.clear_caches();

    let after = compiler.cache_stats();
    assert_eq!(after.compile.entries, 0);
    assert_eq!(after.extensions.entries, 0);
}

#[test]
fn executor_clear_caches_leaves_no_extension_entries_without_extensions() {
    let mut compiler = GraphCompiler::new();
    let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
    let y = &x + &x;
    let program = compiler.compile(&y).expect("compile");
    let mut executor = GraphExecutor::new(CpuBackend::new());

    let _ = executor.run(&program).expect("run");

    let before = executor.cache_stats();
    assert_eq!(before.extensions.entries, 0);

    executor.clear_caches();

    let after = executor.cache_stats();
    assert_eq!(after.extensions.entries, 0);
}

#[test]
fn cpu_executor_clear_all_caches_clears_buffer_pool() {
    let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
    let y = &x + &x;
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(&y).expect("compile");
    let mut executor = GraphExecutor::new(CpuBackend::new());

    let _ = executor.run(&program).expect("run");
    let before = executor.cpu_cache_stats();
    assert_eq!(
        before.buffer_pool.retained_bytes,
        executor.buffer_pool_stats().capacity_bytes
    );

    executor.clear_all_caches();

    let after = executor.cpu_cache_stats();
    assert_eq!(after.executor.extensions.entries, 0);
    assert_eq!(after.executor.backend.entries, 0);
    assert_eq!(after.buffer_pool.entries, 0);
    assert_eq!(executor.buffer_pool_len(), 0);
}
