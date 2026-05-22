use tenferro::traced_tensor::einsum;
use tenferro::{CpuBackend, DType, GraphCompiler, GraphExecutor, Tensor, TracedTensor};

#[test]
fn traced_einsum_uses_compiler_for_static_graph_build_and_executor_for_run() {
    let a = TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]);
    let b = TracedTensor::from_vec_col_major(vec![3, 2], vec![1.0_f64; 6]);

    let mut compiler = GraphCompiler::new();
    let out = einsum(&mut compiler, &[&a, &b], "ij,jk->ik").unwrap();
    assert!(compiler.cache_stats().static_einsum_plans.entries > 0);
    assert!(compiler.cache_stats().einsum_parse.entries > 0);

    let program = compiler.compile(&out).unwrap();
    let mut executor = GraphExecutor::new(CpuBackend::new());
    let tensor = executor.run(&program).unwrap();

    assert_eq!(tensor.shape(), &[2, 2]);
}

#[test]
fn symbolic_einsum_reuses_executor_runtime_plan_cache() {
    let a = TracedTensor::input_symbolic_shape(DType::F64, 2);
    let b = TracedTensor::input_symbolic_shape(DType::F64, 2);

    let mut compiler = GraphCompiler::new();
    let out = einsum(&mut compiler, &[&a, &b], "ij,jk->ik").unwrap();
    let program = compiler
        .compile_with_input_specs(
            &out,
            &[(&a, DType::F64, &[2, 3]), (&b, DType::F64, &[3, 2])],
        )
        .unwrap();

    let lhs = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]);
    let rhs = Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64; 6]);
    let mut executor = GraphExecutor::new(CpuBackend::new());
    let _ = executor
        .run_with_inputs(&program, &[(&a, &lhs), (&b, &rhs)])
        .unwrap();
    let before = executor.cache_stats().runtime_einsum_plans.entries;
    let _ = executor
        .run_with_inputs(&program, &[(&a, &lhs), (&b, &rhs)])
        .unwrap();
    let after = executor.cache_stats().runtime_einsum_plans.entries;

    assert_eq!(before, after);
    assert!(after > 0);
}
