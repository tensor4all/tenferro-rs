use tenferro::traced_tensor::einsum;
use tenferro::{CpuBackend, DType, GraphCompiler, GraphExecutor, Tensor, TracedTensor};

fn compile_static_einsum(compiler: &mut GraphCompiler) -> tenferro::GraphProgram {
    let a = TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]);
    let b = TracedTensor::from_vec_col_major(vec![3, 2], vec![1.0_f64; 6]);
    let out = einsum(compiler, &[&a, &b], "ij,jk->ik").expect("einsum");
    compiler.compile(&out).expect("compile")
}

fn compile_symbolic_einsum(
    compiler: &mut GraphCompiler,
) -> (TracedTensor, TracedTensor, tenferro::GraphProgram) {
    let a = TracedTensor::input_symbolic_shape(DType::F64, 2);
    let b = TracedTensor::input_symbolic_shape(DType::F64, 2);
    let out = einsum(compiler, &[&a, &b], "ij,jk->ik").expect("einsum");
    let program = compiler
        .compile_with_input_specs(
            &out,
            &[(&a, DType::F64, &[2, 3]), (&b, DType::F64, &[3, 2])],
        )
        .expect("compile");
    (a, b, program)
}

fn run_symbolic_einsum(
    executor: &mut GraphExecutor<CpuBackend>,
    program: &tenferro::GraphProgram,
    a: &TracedTensor,
    b: &TracedTensor,
) {
    let lhs = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]);
    let rhs = Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64; 6]);
    executor
        .run_with_inputs(program, &[(a, &lhs), (b, &rhs)])
        .expect("run");
}

#[test]
fn compiler_clear_caches_clears_compile_static_einsum_and_parse_entries() {
    let mut compiler = GraphCompiler::new();

    let _ = compile_static_einsum(&mut compiler);

    let before = compiler.cache_stats();
    assert!(before.compile.entries > 0);
    assert!(before.static_einsum_plans.entries > 0);
    assert!(before.einsum_parse.entries > 0);

    compiler.clear_caches();

    let after = compiler.cache_stats();
    assert_eq!(after.compile.entries, 0);
    assert_eq!(after.static_einsum_plans.entries, 0);
    assert_eq!(after.einsum_parse.entries, 0);
}

#[test]
fn executor_clear_caches_clears_backend_and_runtime_einsum_entries() {
    let mut compiler = GraphCompiler::new();
    let (a, b, program) = compile_symbolic_einsum(&mut compiler);
    let static_program = compile_static_einsum(&mut compiler);
    let mut executor = GraphExecutor::new(CpuBackend::new());

    run_symbolic_einsum(&mut executor, &program, &a, &b);
    let _ = executor.run(&static_program).expect("static run");

    let before = executor.cache_stats();
    assert!(before.runtime_einsum_plans.entries > 0);
    assert!(before.backend.entries > 0);

    executor.clear_caches();

    let after = executor.cache_stats();
    assert_eq!(after.runtime_einsum_plans.entries, 0);
    assert_eq!(after.backend.entries, 0);
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
    assert_eq!(after.executor.runtime_einsum_plans.entries, 0);
    assert_eq!(after.executor.backend.entries, 0);
    assert_eq!(after.buffer_pool.entries, 0);
    assert_eq!(executor.buffer_pool_len(), 0);
}
