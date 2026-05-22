use tenferro::traced_tensor::{einsum, einsum_with, EinsumOptimize};
use tenferro::{CpuBackend, DType, GraphCompiler, GraphExecutor, Tensor, TracedTensor};
use tenferro_einsum::ContractionOptimizerOptions;

#[test]
fn traced_einsum_uses_compiler_for_static_graph_build_and_executor_for_run() {
    let a = TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = TracedTensor::from_vec_col_major(vec![3, 2], vec![7.0_f64, 8.0, 9.0, 10.0, 11.0, 12.0]);

    let mut compiler = GraphCompiler::new();
    let out = einsum(&mut compiler, &[&a, &b], "ij,jk->ik").unwrap();
    assert!(compiler.cache_stats().static_einsum_plans.entries > 0);
    assert!(compiler.cache_stats().einsum_parse.entries > 0);

    let program = compiler.compile(&out).unwrap();
    let mut executor = GraphExecutor::new(CpuBackend::new());
    let tensor = executor.run(&program).unwrap();

    assert_eq!(tensor.shape(), &[2, 2]);
    assert_eq!(
        tensor.as_slice::<f64>().unwrap(),
        &[76.0, 100.0, 103.0, 136.0]
    );
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

#[test]
fn custom_static_auto_options_do_not_reuse_default_auto_cache_entry() {
    let a = TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]);
    let b = TracedTensor::from_vec_col_major(vec![3, 2], vec![1.0_f64; 6]);
    let mut compiler = GraphCompiler::new();

    let _ = einsum(&mut compiler, &[&a, &b], "ij,jk->ik").unwrap();
    assert_eq!(compiler.cache_stats().static_einsum_plans.entries, 1);

    let result = einsum_with(
        &mut compiler,
        &[&a, &b],
        "ij,jk->ik",
        EinsumOptimize::Auto(ContractionOptimizerOptions {
            ntrials: 0,
            ..Default::default()
        }),
    );
    let err = match result {
        Ok(_) => panic!("expected custom Auto options to bypass the default cache and fail"),
        Err(err) => err,
    };

    assert!(format!("{err}").contains("ntrials"), "got {err}");
}

#[test]
fn symbolic_einsum_rejects_non_default_optimizer_strategy() {
    let a = TracedTensor::input_symbolic_shape(DType::F64, 2);
    let b = TracedTensor::input_symbolic_shape(DType::F64, 2);
    let mut compiler = GraphCompiler::new();

    let result = einsum_with(&mut compiler, &[&a, &b], "ij,jk->ik", EinsumOptimize::False);
    let err = match result {
        Ok(_) => panic!("expected symbolic einsum to reject non-default optimizer strategy"),
        Err(err) => err,
    };

    assert!(
        format!("{err}").contains("symbolic einsum supports only default automatic optimization"),
        "got {err}"
    );
}

#[test]
fn symbolic_einsum_rejects_custom_auto_options() {
    let a = TracedTensor::input_symbolic_shape(DType::F64, 2);
    let b = TracedTensor::input_symbolic_shape(DType::F64, 2);
    let mut compiler = GraphCompiler::new();

    let result = einsum_with(
        &mut compiler,
        &[&a, &b],
        "ij,jk->ik",
        EinsumOptimize::Auto(ContractionOptimizerOptions {
            ntrials: 0,
            ..Default::default()
        }),
    );
    let err = match result {
        Ok(_) => panic!("expected symbolic einsum to reject custom Auto options"),
        Err(err) => err,
    };

    assert!(
        format!("{err}").contains("symbolic einsum supports only default automatic optimization"),
        "got {err}"
    );
}
