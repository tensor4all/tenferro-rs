#![cfg(feature = "autodiff")]

use std::num::NonZeroUsize;

use tenferro_ad::EagerRuntime;
use tenferro_cpu::CpuBackend;
use tenferro_einsum::EagerEinsumExt;
use tenferro_einsum::GraphCompilerEinsumExt;
use tenferro_einsum::{
    parse_einsum_subscripts, ContractionOptimizerOptions, ContractionTree, EinsumOptimize,
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
        vec![rows, rows, mid],
        (0..rows * rows * mid).map(|i| i as f64).collect::<Vec<_>>(),
    )
    .unwrap();
    let b = TracedTensor::from_vec_col_major(
        vec![mid, cols],
        (0..mid * cols).map(|i| i as f64).collect::<Vec<_>>(),
    )
    .unwrap();
    let _ = compiler.einsum(&[&a, &b], "iij,jk->ik").expect("einsum");
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
        vec![rows, rows, mid],
        (0..rows * rows * mid).map(|i| i as f64).collect::<Vec<_>>(),
    )
    .unwrap();
    let b = Tensor::from_vec_col_major(
        vec![mid, cols],
        (0..mid * cols).map(|i| i as f64).collect::<Vec<_>>(),
    )
    .unwrap();
    (a, b)
}

fn compile_runtime_planned_matmul(rows: usize, cols: usize, mid: usize) -> RuntimePlannedMatmul {
    let a = TracedTensor::input_symbolic_shape(DType::F64, 3).unwrap();
    let b = TracedTensor::input_symbolic_shape(DType::F64, 2).unwrap();
    let mut compiler = GraphCompiler::new();
    let out = compiler.einsum(&[&a, &b], "iij,jk->ik").expect("einsum");
    let program = compiler
        .compile_with_input_specs(
            &out,
            &[
                (&a, DType::F64, &[rows, rows, mid]),
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

fn run_eager_extension_matmul(
    ctx: &std::sync::Arc<EagerRuntime>,
    rows: usize,
    cols: usize,
    mid: usize,
) -> Tensor {
    let a = ctx
        .constant_from(
            Tensor::from_vec_col_major(
                vec![rows, rows, mid],
                (0..rows * rows * mid).map(|i| i as f64).collect::<Vec<_>>(),
            )
            .unwrap(),
        )
        .unwrap();
    let b = ctx
        .constant_from(
            Tensor::from_vec_col_major(
                vec![mid, cols],
                (0..mid * cols).map(|i| i as f64).collect::<Vec<_>>(),
            )
            .unwrap(),
        )
        .unwrap();
    [&a, &b]
        .einsum("iij,jk->ik")
        .expect("eager einsum")
        .to_tensor()
        .unwrap()
}

fn extension_cache_entries(compiler: &GraphCompiler) -> usize {
    compiler.cache_stats().extensions.entries
}

fn runtime_cache_entries(executor: &GraphExecutor<CpuBackend>) -> usize {
    executor.cache_stats().extensions.entries
}

#[test]
fn traced_einsum_uses_extension_compile_caches_and_runtime() {
    let a = TracedTensor::from_vec_col_major(vec![2, 2, 3], vec![1.0_f64; 12]).unwrap();
    let b = TracedTensor::from_vec_col_major(vec![3, 2], vec![1.0_f64; 6]).unwrap();
    let mut compiler = GraphCompiler::new();

    let out = compiler.einsum(&[&a, &b], "iij,jk->ik").unwrap();

    assert!(extension_cache_entries(&compiler) >= 2);

    let program = compiler.compile(&out).unwrap();
    let mut executor = GraphExecutor::new(CpuBackend::new());
    register_runtime(&mut executor);
    let tensor = executor.run(&program).unwrap();

    assert_eq!(tensor.shape(), &[2, 2]);
    assert_eq!(tensor.as_slice::<f64>().unwrap(), &[3.0_f64; 4]);
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
fn eager_einsum_expansion_reuses_runtime_owned_expanded_program_cache() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());

    let out = run_eager_extension_matmul(&ctx, 2, 4, 3);
    assert_eq!(out.shape(), &[2, 4]);
    let entries_after_first = ctx.cache_stats().unwrap().extensions.entries;
    assert_eq!(entries_after_first, 1);

    let _ = run_eager_extension_matmul(&ctx, 2, 4, 3);
    assert_eq!(
        ctx.cache_stats().unwrap().extensions.entries,
        entries_after_first
    );

    let other_ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let _ = run_eager_extension_matmul(&other_ctx, 2, 4, 3);
    assert_eq!(
        ctx.cache_stats().unwrap().extensions.entries,
        entries_after_first
    );
    assert_eq!(other_ctx.cache_stats().unwrap().extensions.entries, 1);
}

#[test]
fn eager_einsum_expansion_respects_runtime_cache_limits() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    ctx.set_extension_cache_limits(ExtensionCacheLimits::new(NonZeroUsize::new(3).unwrap()))
        .unwrap();

    for mid in 1..=5 {
        let _ = run_eager_extension_matmul(&ctx, 2, 2, mid);
    }

    let stats = ctx.cache_stats().unwrap();
    assert_eq!(stats.extensions.entries, 3);
    assert!(stats.extensions.retained_bytes > 0);
    assert_eq!(
        ctx.extension_cache_limits().unwrap().max_entries(),
        NonZeroUsize::new(3).unwrap()
    );

    ctx.clear_caches().unwrap();
    assert_eq!(ctx.cache_stats().unwrap().extensions.entries, 0);
}

#[test]
fn runtime_planned_einsum_reuses_extension_runtime_caches() {
    let case = compile_runtime_planned_matmul(2, 2, 3);
    let mut executor = GraphExecutor::new(CpuBackend::new());
    register_runtime(&mut executor);

    let out = run_runtime_planned_matmul(&mut executor, &case);
    assert_eq!(out.as_slice::<f64>().unwrap(), &[20.0, 29.0, 56.0, 92.0]);
    let before = runtime_cache_entries(&executor);

    let _ = run_runtime_planned_matmul(&mut executor, &case);
    let after = runtime_cache_entries(&executor);

    assert_eq!(before, 2);
    assert_eq!(after, before);
}

#[test]
fn traced_static_tree_einsum_expands_without_runtime_exec_program_cache() {
    let subs = tenferro_einsum::Subscripts::parse("ij,jk,kl->il").unwrap();
    let tree = ContractionTree::from_pairs(
        &subs,
        &[&[2, 3][..], &[3, 4][..], &[4, 2][..]],
        &[(1, 2), (0, 3)],
    )
    .unwrap();
    let a = TracedTensor::input_concrete_shape(DType::F64, &[2, 3]).unwrap();
    let b = TracedTensor::input_concrete_shape(DType::F64, &[3, 4]).unwrap();
    let c = TracedTensor::input_concrete_shape(DType::F64, &[4, 2]).unwrap();
    let mut compiler = GraphCompiler::new();
    let out = compiler
        .einsum_with(&[&a, &b, &c], "ij,jk,kl->il", EinsumOptimize::Tree(tree))
        .unwrap();
    let program = compiler
        .compile_with_input_specs(
            &out,
            &[
                (&a, DType::F64, &[2, 3]),
                (&b, DType::F64, &[3, 4]),
                (&c, DType::F64, &[4, 2]),
            ],
        )
        .unwrap();

    let mut executor = GraphExecutor::new(CpuBackend::new());
    register_runtime(&mut executor);
    let a_value = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap();
    let b_value = Tensor::from_vec_col_major(vec![3, 4], vec![1.0_f64; 12]).unwrap();
    let c_value = Tensor::from_vec_col_major(vec![4, 2], vec![1.0_f64; 8]).unwrap();

    let first = executor
        .run_with_inputs(&program, &[(&a, &a_value), (&b, &b_value), (&c, &c_value)])
        .unwrap();
    assert_eq!(first.as_slice::<f64>().unwrap(), &[12.0; 4]);
    let after_first = runtime_cache_entries(&executor);

    let _ = executor
        .run_with_inputs(&program, &[(&a, &a_value), (&b, &b_value), (&c, &c_value)])
        .unwrap();
    let after_second = runtime_cache_entries(&executor);

    assert_eq!(after_first, 0);
    assert_eq!(after_second, after_first);
}

#[test]
fn runtime_planned_einsum_cache_distinguishes_plan_spec() {
    let a = TracedTensor::input_symbolic_shape(DType::F64, 2).unwrap();
    let b = TracedTensor::input_symbolic_shape(DType::F64, 2).unwrap();
    let c = TracedTensor::input_symbolic_shape(DType::F64, 2).unwrap();
    let mut compiler = GraphCompiler::new();

    let default_auto = compiler
        .einsum_with(
            &[&a, &b, &c],
            "ij,jk,kl->il",
            EinsumOptimize::Auto(Default::default()),
        )
        .unwrap();
    let custom_auto = compiler
        .einsum_with(
            &[&a, &b, &c],
            "ij,jk,kl->il",
            EinsumOptimize::Auto(ContractionOptimizerOptions {
                ntrials: 2,
                ..Default::default()
            }),
        )
        .unwrap();
    let left_program = compiler
        .compile_with_input_specs(
            &default_auto,
            &[
                (&a, DType::F64, &[2, 3]),
                (&b, DType::F64, &[3, 4]),
                (&c, DType::F64, &[4, 5]),
            ],
        )
        .unwrap();
    let path_program = compiler
        .compile_with_input_specs(
            &custom_auto,
            &[
                (&a, DType::F64, &[2, 3]),
                (&b, DType::F64, &[3, 4]),
                (&c, DType::F64, &[4, 5]),
            ],
        )
        .unwrap();

    let mut executor = GraphExecutor::new(CpuBackend::new());
    register_runtime(&mut executor);
    let a_value = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap();
    let b_value = Tensor::from_vec_col_major(vec![3, 4], vec![1.0_f64; 12]).unwrap();
    let c_value = Tensor::from_vec_col_major(vec![4, 5], vec![1.0_f64; 20]).unwrap();

    let _ = executor
        .run_with_inputs(
            &left_program,
            &[(&a, &a_value), (&b, &b_value), (&c, &c_value)],
        )
        .unwrap();
    let after_left = executor.cache_stats().extensions.entries;
    let _ = executor
        .run_with_inputs(
            &path_program,
            &[(&a, &a_value), (&b, &b_value), (&c, &c_value)],
        )
        .unwrap();
    let after_path = executor.cache_stats().extensions.entries;

    assert!(
        after_path > after_left,
        "runtime plan cache should keep separate entries for different plan specs"
    );
}

#[test]
fn runtime_planned_einsum_honors_explicit_path_execution_order() {
    let a = TracedTensor::input_symbolic_shape(DType::F64, 2).unwrap();
    let b = TracedTensor::input_symbolic_shape(DType::F64, 2).unwrap();
    let c = TracedTensor::input_symbolic_shape(DType::F64, 2).unwrap();
    let mut compiler = GraphCompiler::new();

    let left_to_right = compiler
        .einsum_with(&[&a, &b, &c], "ij,jk,kl->il", EinsumOptimize::False)
        .unwrap();
    let explicit_path = compiler
        .einsum_with(
            &[&a, &b, &c],
            "ij,jk,kl->il",
            EinsumOptimize::Path(vec![(1, 2), (0, 1)]),
        )
        .unwrap();
    let left_program = compiler
        .compile_with_input_specs(
            &left_to_right,
            &[
                (&a, DType::F64, &[1, 1]),
                (&b, DType::F64, &[1, 1]),
                (&c, DType::F64, &[1, 1]),
            ],
        )
        .unwrap();
    let path_program = compiler
        .compile_with_input_specs(
            &explicit_path,
            &[
                (&a, DType::F64, &[1, 1]),
                (&b, DType::F64, &[1, 1]),
                (&c, DType::F64, &[1, 1]),
            ],
        )
        .unwrap();

    let mut executor = GraphExecutor::new(CpuBackend::new());
    register_runtime(&mut executor);
    let a_value = Tensor::from_vec_col_major(vec![1, 1], vec![1.0e308_f64]).unwrap();
    let b_value = Tensor::from_vec_col_major(vec![1, 1], vec![1.0e308_f64]).unwrap();
    let c_value = Tensor::from_vec_col_major(vec![1, 1], vec![1.0e-308_f64]).unwrap();

    let left_result = executor
        .run_with_inputs(
            &left_program,
            &[(&a, &a_value), (&b, &b_value), (&c, &c_value)],
        )
        .unwrap();
    let path_result = executor
        .run_with_inputs(
            &path_program,
            &[(&a, &a_value), (&b, &b_value), (&c, &c_value)],
        )
        .unwrap();

    let left_value = left_result.as_slice::<f64>().unwrap()[0];
    let path_value = path_result.as_slice::<f64>().unwrap()[0];
    assert!(
        left_value.is_infinite(),
        "left-to-right path should overflow, got {left_value}"
    );
    assert!(
        path_value.is_finite(),
        "explicit B*C-first path should avoid overflow, got {path_value}"
    );
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
    let out = TracedTensor::from_vec_col_major(vec![1], vec![1.0_f64])
        .unwrap()
        .neg()
        .unwrap();
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
    let a = TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap();
    let b = TracedTensor::from_vec_col_major(vec![3, 2], vec![1.0_f64; 6]).unwrap();
    let mut compiler = GraphCompiler::new();

    let _ = compiler.einsum(&[&a, &b], "ij,jk->ik").unwrap();
    let before = extension_cache_entries(&compiler);

    let result = compiler.einsum_with(
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
fn concrete_traced_einsum_static_cache_distinguishes_plan_spec() {
    let a = TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap();
    let b = TracedTensor::from_vec_col_major(vec![3, 4], vec![1.0_f64; 12]).unwrap();
    let c = TracedTensor::from_vec_col_major(vec![4, 5], vec![1.0_f64; 20]).unwrap();
    let mut compiler = GraphCompiler::new();

    let _ = compiler
        .einsum_with(&[&a, &b, &c], "ij,jk,kl->il", EinsumOptimize::False)
        .unwrap();
    let after_false = extension_cache_entries(&compiler);

    let _ = compiler
        .einsum_with(
            &[&a, &b, &c],
            "ij,jk,kl->il",
            EinsumOptimize::Path(vec![(1, 2), (0, 1)]),
        )
        .unwrap();
    let after_path = extension_cache_entries(&compiler);

    assert!(
        after_path > after_false,
        "static plan cache should keep separate entries for different plan specs"
    );
}

#[test]
fn symbolic_einsum_accepts_non_default_optimizer_strategy() {
    let a = TracedTensor::input_symbolic_shape(DType::F64, 2).unwrap();
    let b = TracedTensor::input_symbolic_shape(DType::F64, 2).unwrap();
    let mut compiler = GraphCompiler::new();

    let result = compiler.einsum_with(&[&a, &b], "ij,jk->ik", EinsumOptimize::False);

    assert!(
        result.is_ok(),
        "symbolic einsum should carry per-op plan spec"
    );
}

#[test]
fn symbolic_einsum_accepts_custom_auto_options() {
    let a = TracedTensor::input_symbolic_shape(DType::F64, 2).unwrap();
    let b = TracedTensor::input_symbolic_shape(DType::F64, 2).unwrap();
    let mut compiler = GraphCompiler::new();

    let result = compiler.einsum_with(
        &[&a, &b],
        "ij,jk->ik",
        EinsumOptimize::Auto(ContractionOptimizerOptions {
            ntrials: 2,
            ..Default::default()
        }),
    );

    assert!(
        result.is_ok(),
        "symbolic einsum should retain custom Auto options"
    );
}

#[test]
fn symbolic_einsum_rejects_precomputed_tree() {
    let concrete_subs = tenferro_einsum::Subscripts::parse("ij,jk,kl->il").unwrap();
    let tree = ContractionTree::optimize(&concrete_subs, &[&[2, 3][..], &[3, 4][..], &[4, 5][..]])
        .unwrap();
    let a = TracedTensor::input_symbolic_shape(DType::F64, 2).unwrap();
    let b = TracedTensor::input_symbolic_shape(DType::F64, 2).unwrap();
    let c = TracedTensor::input_symbolic_shape(DType::F64, 2).unwrap();
    let mut compiler = GraphCompiler::new();

    let result = compiler.einsum_with(&[&a, &b, &c], "ij,jk,kl->il", EinsumOptimize::Tree(tree));
    let err = match result {
        Ok(_) => panic!("symbolic Tree should be rejected"),
        Err(err) => err,
    };

    assert!(
        format!("{err}").contains("precomputed contraction tree requires concrete input shapes"),
        "got {err}"
    );
}

#[test]
fn symbolic_einsum_rejects_nan_auto_options() {
    let a = TracedTensor::input_symbolic_shape(DType::F64, 2).unwrap();
    let b = TracedTensor::input_symbolic_shape(DType::F64, 2).unwrap();
    let mut compiler = GraphCompiler::new();

    let result = compiler.einsum_with(
        &[&a, &b],
        "ij,jk->ik",
        EinsumOptimize::Auto(ContractionOptimizerOptions {
            betas: vec![f64::NAN],
            ..Default::default()
        }),
    );
    let err = match result {
        Ok(_) => panic!("NaN options should be rejected"),
        Err(err) => err,
    };

    assert!(format!("{err}").contains("NaN"), "got {err}");
}

#[test]
fn parsed_integer_subscripts_trace_through_extension() {
    let a = TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap();
    let b = TracedTensor::from_vec_col_major(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap();
    let subscripts = parse_einsum_subscripts("ij,jk->ik").unwrap();
    let mut compiler = GraphCompiler::new();

    let out = compiler.einsum_subscripts(&[&a, &b], &subscripts).unwrap();
    let program = compiler.compile(&out).unwrap();
    let mut executor = GraphExecutor::new(CpuBackend::new());
    register_runtime(&mut executor);
    let tensor = executor.run(&program).unwrap();

    assert_eq!(tensor.shape(), &[2, 2]);
    assert_eq!(tensor.as_slice::<f64>().unwrap(), &[22.0, 28.0, 49.0, 64.0]);
}
