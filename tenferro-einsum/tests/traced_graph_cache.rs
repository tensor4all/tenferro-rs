#![cfg(feature = "autodiff")]

use std::num::NonZeroUsize;

use tenferro_ad::EagerRuntime;
use tenferro_cpu::CpuBackend;
use tenferro_einsum::{
    eager_tensor::einsum as eager_einsum, einsum, einsum_with, parse_einsum_subscripts,
    ContractionOptimizerOptions, EinsumOptimize,
};
use tenferro_runtime::extension::ExtensionCacheLimits;
use tenferro_runtime::{DType, GraphCompiler, GraphExecutor, GraphProgram, Tensor, TracedTensor};

fn register_runtime(executor: &mut GraphExecutor<CpuBackend>) {
    executor
        .register_extension(tenferro_einsum::register_runtime)
        .unwrap();
}

fn run_static_matmul(compiler: &mut GraphCompiler, rows: usize, cols: usize, mid: usize) {
    let a = TracedTensor::from_vec_col_major(
        vec![rows, mid],
        (0..rows * mid).map(|i| i as f64).collect::<Vec<_>>(),
    );
    let b = TracedTensor::from_vec_col_major(
        vec![mid, cols],
        (0..mid * cols).map(|i| i as f64).collect::<Vec<_>>(),
    );
    let _ = einsum(compiler, &[&a, &b], "ij,jk->ik").expect("einsum");
}

struct RuntimePlannedMatmul {
    a: TracedTensor,
    b: TracedTensor,
    a_value: Tensor,
    b_value: Tensor,
    program: GraphProgram,
}

fn runtime_matmul_values(rows: usize, cols: usize, mid: usize) -> (Tensor, Tensor) {
    let a = Tensor::from_vec_col_major(
        vec![rows, mid],
        (0..rows * mid).map(|i| i as f64).collect::<Vec<_>>(),
    );
    let b = Tensor::from_vec_col_major(
        vec![mid, cols],
        (0..mid * cols).map(|i| i as f64).collect::<Vec<_>>(),
    );
    (a, b)
}

fn compile_runtime_planned_matmul(rows: usize, cols: usize, mid: usize) -> RuntimePlannedMatmul {
    let a = TracedTensor::input_symbolic_shape(DType::F64, 2);
    let b = TracedTensor::input_symbolic_shape(DType::F64, 2);
    let mut compiler = GraphCompiler::new();
    let out = einsum(&mut compiler, &[&a, &b], "ij,jk->ik").expect("einsum");
    let program = compiler
        .compile_with_input_specs(
            &out,
            &[
                (&a, DType::F64, &[rows, mid]),
                (&b, DType::F64, &[mid, cols]),
            ],
        )
        .expect("compile");
    let (a_value, b_value) = runtime_matmul_values(rows, cols, mid);
    RuntimePlannedMatmul {
        a,
        b,
        a_value,
        b_value,
        program,
    }
}

fn run_runtime_planned_matmul(
    executor: &mut GraphExecutor<CpuBackend>,
    case: &RuntimePlannedMatmul,
) -> Tensor {
    executor
        .run_with_inputs(
            &case.program,
            &[(&case.a, &case.a_value), (&case.b, &case.b_value)],
        )
        .expect("run")
}

fn run_eager_matmul(
    ctx: &std::sync::Arc<EagerRuntime>,
    rows: usize,
    cols: usize,
    mid: usize,
) -> Tensor {
    let a = ctx.constant_from(Tensor::from_vec_col_major(
        vec![rows, mid],
        (0..rows * mid).map(|i| i as f64).collect::<Vec<_>>(),
    ));
    let b = ctx.constant_from(Tensor::from_vec_col_major(
        vec![mid, cols],
        (0..mid * cols).map(|i| i as f64).collect::<Vec<_>>(),
    ));
    eager_einsum(&[&a, &b], "ij,jk->ik")
        .expect("eager einsum")
        .data()
        .clone()
}

fn extension_cache_entries(compiler: &GraphCompiler) -> usize {
    compiler.cache_stats().extensions.entries
}

fn runtime_cache_entries(executor: &GraphExecutor<CpuBackend>) -> usize {
    executor.cache_stats().extensions.entries
}

#[test]
fn traced_einsum_uses_extension_compile_caches_and_runtime() {
    let a = TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = TracedTensor::from_vec_col_major(vec![3, 2], vec![7.0_f64, 8.0, 9.0, 10.0, 11.0, 12.0]);
    let mut compiler = GraphCompiler::new();

    let out = einsum(&mut compiler, &[&a, &b], "ij,jk->ik").unwrap();

    assert!(extension_cache_entries(&compiler) >= 2);

    let program = compiler.compile(&out).unwrap();
    let mut executor = GraphExecutor::new(CpuBackend::new());
    register_runtime(&mut executor);
    let tensor = executor.run(&program).unwrap();

    assert_eq!(tensor.shape(), &[2, 2]);
    assert_eq!(
        tensor.as_slice::<f64>().unwrap(),
        &[76.0, 100.0, 103.0, 136.0]
    );
}

#[test]
fn concrete_traced_einsum_reuses_extension_static_and_parse_caches() {
    let mut compiler = GraphCompiler::new();

    run_static_matmul(&mut compiler, 2, 2, 3);
    let entries_after_first = extension_cache_entries(&compiler);
    assert!(entries_after_first >= 2);

    run_static_matmul(&mut compiler, 2, 2, 3);
    assert_eq!(extension_cache_entries(&compiler), entries_after_first);

    run_static_matmul(&mut compiler, 2, 2, 4);
    assert!(extension_cache_entries(&compiler) > entries_after_first);
}

#[test]
fn extension_compile_cache_limits_bound_static_einsum_entries() {
    let mut compiler = GraphCompiler::new();
    compiler
        .extension_caches_mut()
        .set_limits(ExtensionCacheLimits::new(NonZeroUsize::new(3).unwrap()));

    for mid in 1..=5 {
        run_static_matmul(&mut compiler, 2, 2, mid);
    }

    assert_eq!(compiler.cache_stats().extensions.entries, 3);
    assert_eq!(
        compiler.extension_caches().limits().max_entries(),
        NonZeroUsize::new(3).unwrap()
    );
}

#[test]
fn eager_einsum_runtime_plan_cache_is_owned_by_context() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());

    let out = run_eager_matmul(&ctx, 2, 4, 3);
    assert_eq!(out.shape(), &[2, 4]);
    assert_eq!(ctx.cache_stats().extensions.entries, 1);

    let _ = run_eager_matmul(&ctx, 2, 4, 3);
    assert_eq!(ctx.cache_stats().extensions.entries, 1);

    let other_ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let _ = run_eager_matmul(&other_ctx, 2, 4, 3);
    assert_eq!(ctx.cache_stats().extensions.entries, 1);
    assert_eq!(other_ctx.cache_stats().extensions.entries, 1);
}

#[test]
fn eager_extension_cache_limits_bound_runtime_planned_einsum_entries() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    ctx.set_extension_cache_limits(ExtensionCacheLimits::new(NonZeroUsize::new(3).unwrap()));

    for mid in 1..=5 {
        let _ = run_eager_matmul(&ctx, 2, 2, mid);
    }

    let stats = ctx.cache_stats();
    assert_eq!(stats.extensions.entries, 3);
    assert!(stats.extensions.retained_bytes > 0);
    assert_eq!(
        ctx.extension_cache_limits().max_entries(),
        NonZeroUsize::new(3).unwrap()
    );

    ctx.clear_caches();
    assert_eq!(ctx.cache_stats().extensions.entries, 0);
}

#[test]
fn runtime_planned_einsum_reuses_extension_runtime_plan_cache() {
    let case = compile_runtime_planned_matmul(2, 2, 3);
    let mut executor = GraphExecutor::new(CpuBackend::new());
    register_runtime(&mut executor);

    let out = run_runtime_planned_matmul(&mut executor, &case);
    assert_eq!(out.as_slice::<f64>().unwrap(), &[10.0, 13.0, 28.0, 40.0]);
    let before = runtime_cache_entries(&executor);

    let _ = run_runtime_planned_matmul(&mut executor, &case);
    let after = runtime_cache_entries(&executor);

    assert_eq!(before, 1);
    assert_eq!(after, before);
}

#[test]
fn extension_runtime_cache_limits_bound_runtime_planned_einsum_entries() {
    let mut executor = GraphExecutor::new(CpuBackend::new());
    register_runtime(&mut executor);
    executor
        .extension_executor_mut()
        .set_cache_limits(ExtensionCacheLimits::new(NonZeroUsize::new(3).unwrap()));

    for mid in 1..=5 {
        let case = compile_runtime_planned_matmul(2, 2, mid);
        let _ = run_runtime_planned_matmul(&mut executor, &case);
    }

    assert_eq!(executor.cache_stats().extensions.entries, 3);
    assert_eq!(
        executor.extension_executor().cache_limits().max_entries(),
        NonZeroUsize::new(3).unwrap()
    );
}

#[test]
fn compiler_and_executor_clear_caches_clear_extension_einsum_entries() {
    let mut compiler = GraphCompiler::new();
    run_static_matmul(&mut compiler, 2, 2, 3);
    let out = TracedTensor::from_vec_col_major(vec![1], vec![1.0_f64]).neg();
    let _ = compiler.compile(&out).unwrap();
    assert!(compiler.cache_stats().compile.entries > 0);
    assert!(compiler.cache_stats().extensions.entries > 0);

    compiler.clear_caches();

    assert_eq!(compiler.cache_stats().compile.entries, 0);
    assert_eq!(compiler.cache_stats().extensions.entries, 0);

    let program = compile_runtime_planned_matmul(2, 2, 3);
    let mut executor = GraphExecutor::new(CpuBackend::new());
    register_runtime(&mut executor);
    let _ = run_runtime_planned_matmul(&mut executor, &program);
    assert!(executor.cache_stats().extensions.entries > 0);

    executor.clear_caches();

    assert_eq!(executor.cache_stats().extensions.entries, 0);
    assert_eq!(executor.cache_stats().backend.entries, 0);
}

#[test]
fn custom_static_auto_options_do_not_reuse_default_extension_cache_entry() {
    let a = TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]);
    let b = TracedTensor::from_vec_col_major(vec![3, 2], vec![1.0_f64; 6]);
    let mut compiler = GraphCompiler::new();

    let _ = einsum(&mut compiler, &[&a, &b], "ij,jk->ik").unwrap();
    let before = extension_cache_entries(&compiler);

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
    assert_eq!(extension_cache_entries(&compiler), before);
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

#[test]
fn parsed_integer_subscripts_trace_through_extension() {
    let a = TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = TracedTensor::from_vec_col_major(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let subscripts = parse_einsum_subscripts("ij,jk->ik").unwrap();
    let mut compiler = GraphCompiler::new();

    let out = tenferro_einsum::einsum_subscripts(&mut compiler, &[&a, &b], &subscripts).unwrap();
    let program = compiler.compile(&out).unwrap();
    let mut executor = GraphExecutor::new(CpuBackend::new());
    register_runtime(&mut executor);
    let tensor = executor.run(&program).unwrap();

    assert_eq!(tensor.shape(), &[2, 2]);
    assert_eq!(tensor.as_slice::<f64>().unwrap(), &[22.0, 28.0, 49.0, 64.0]);
}
