use std::num::NonZeroUsize;

use tenferro::{CpuBackend, DType, GraphCompiler, GraphExecutor, Tensor, TracedTensor};

#[test]
fn graph_executor_runs_compiled_single_output_program() {
    let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
    let y = &x + &x;

    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(&y).unwrap();
    let mut executor = GraphExecutor::new(CpuBackend::new());

    let out = executor.run(&program).unwrap();

    assert_eq!(out.as_slice::<f64>().unwrap(), &[2.0, 4.0]);
}

#[test]
fn graph_executor_runs_compiled_multi_output_program() {
    let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
    let sum = &x + &x;
    let product = &x * &x;

    let mut compiler = GraphCompiler::new();
    let program = compiler.compile_many(&[&sum, &product]).unwrap();
    let mut executor = GraphExecutor::new(CpuBackend::new());

    let outputs = executor.run_many(&program).unwrap();

    assert_eq!(outputs.len(), 2);
    assert_eq!(outputs[0].as_slice::<f64>().unwrap(), &[2.0, 4.0]);
    assert_eq!(outputs[1].as_slice::<f64>().unwrap(), &[1.0, 4.0]);
}

#[test]
fn graph_executor_validates_runtime_bindings() {
    let x = TracedTensor::input_symbolic_shape(DType::F64, 1);
    let y = &x + &x;

    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_with_input_specs(&y, &[(&x, DType::F64, &[3])])
        .unwrap();
    let mut executor = GraphExecutor::new(CpuBackend::new());

    let ok = Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]);
    let out = executor.run_with_inputs(&program, &[(&x, &ok)]).unwrap();
    assert_eq!(out.as_slice::<f64>().unwrap(), &[2.0, 4.0, 6.0]);

    let wrong_shape = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
    let err = executor
        .run_with_inputs(&program, &[(&x, &wrong_shape)])
        .unwrap_err();
    assert!(format!("{err}").contains("shape"));
}

#[test]
fn graph_executor_cache_stats_are_separate_from_compiler_stats() {
    let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
    let y = &x + &x;

    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(&y).unwrap();
    let mut executor = GraphExecutor::new(CpuBackend::new());
    let _ = executor.run(&program).unwrap();

    assert!(compiler.cache_stats().compile.entries > 0);
    assert_eq!(executor.cache_stats().runtime_einsum_plans.entries, 0);
}

#[test]
fn graph_executor_cpu_cache_controls_are_available() {
    let mut executor =
        GraphExecutor::with_einsum_cache_capacity(CpuBackend::new(), NonZeroUsize::new(2).unwrap());

    assert_eq!(executor.einsum_cache_len(), 0);
    assert_eq!(executor.einsum_cache_capacity().get(), 2);
    executor.set_einsum_cache_capacity(NonZeroUsize::new(3).unwrap());
    assert_eq!(executor.einsum_cache_capacity().get(), 3);

    let original_gemm_capacity = executor.gemm_analysis_cache_capacity();
    executor.set_gemm_analysis_cache_capacity(0);
    assert_eq!(executor.gemm_analysis_cache_capacity(), 0);
    executor.set_gemm_analysis_cache_capacity(original_gemm_capacity);

    let original_pool_limit = executor.buffer_pool_limit_bytes();
    executor.set_buffer_pool_limit_bytes(0);
    assert_eq!(executor.buffer_pool_limit_bytes(), 0);
    executor.set_buffer_pool_limit_bytes(original_pool_limit);

    let stats = executor.cpu_cache_stats();
    assert_eq!(stats.executor.runtime_einsum_plans.entries, 0);
    assert_eq!(stats.buffer_pool.entries, executor.buffer_pool_len());

    executor.clear_all_caches();
    assert_eq!(executor.cpu_cache_stats().executor.backend.entries, 0);
    assert_eq!(executor.buffer_pool_len(), 0);
}

#[test]
fn graph_executor_synthesizes_deferred_zero_tangents_from_primal_binding() {
    let x = TracedTensor::input_symbolic_shape(DType::F64, 1);
    let loss = (&x * &x).reduce_sum(&[0]);
    let grad = loss.grad(&x).unwrap();

    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_with_input_specs(&grad, &[(&x, DType::F64, &[4])])
        .unwrap();
    let mut executor = GraphExecutor::new(CpuBackend::new());

    let bound = Tensor::from_vec_col_major(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0]);
    let out = executor.run_with_inputs(&program, &[(&x, &bound)]).unwrap();

    assert_eq!(out.shape(), &[4]);
    assert_eq!(out.as_slice::<f64>().unwrap(), &[2.0, 4.0, 6.0, 8.0]);
}
