use tenferro_ad::TracedTensorAdExt;
use tenferro_runtime::error::Error;
use tenferro_runtime::{DType, GraphCompiler, Tensor, TraceContext};

use crate::support::{cpu_runtime, run_compiled_one};

#[test]
fn runtime_runs_compiled_single_output_program() {
    let trace = TraceContext::new();
    let x = trace
        .from_vec_col_major(vec![2], vec![1.0_f64, 2.0])
        .unwrap();
    let y = (&x + &x).unwrap();

    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(&y).unwrap();
    let executor = cpu_runtime();

    let out = run_compiled_one(&executor, &program, &[]).unwrap();

    assert_eq!(out.as_slice::<f64>().unwrap(), &[2.0, 4.0]);
}

#[test]
fn runtime_runs_compiled_multi_output_program() {
    let trace = TraceContext::new();
    let x = trace
        .from_vec_col_major(vec![2], vec![1.0_f64, 2.0])
        .unwrap();
    let sum = (&x + &x).unwrap();
    let product = (&x * &x).unwrap();

    let mut compiler = GraphCompiler::new();
    let program = compiler.compile_many(&[&sum, &product]).unwrap();
    let executor = cpu_runtime();

    let outputs = crate::support::run_on_cpu(&executor, &program, &[]).unwrap();

    assert_eq!(outputs.len(), 2);
    assert_eq!(outputs[0].as_slice::<f64>().unwrap(), &[2.0, 4.0]);
    assert_eq!(outputs[1].as_slice::<f64>().unwrap(), &[1.0, 4.0]);
}

#[test]
fn checkpoint_uses_explicit_compiler_and_runtime() {
    let trace = TraceContext::new();
    let x = trace.from_vec_col_major(vec![], vec![3.0_f64]).unwrap();
    let mut y = (&x * &x).unwrap();

    let mut compiler = GraphCompiler::new();
    let executor = cpu_runtime();
    y.checkpoint(
        &mut compiler,
        &executor,
        &mut tenferro_cpu::CpuEngine::serial(),
    )
    .unwrap();

    let program = compiler.compile(&y).unwrap();
    let out = run_compiled_one(&executor, &program, &[]).unwrap();
    assert_eq!(out.as_slice::<f64>().unwrap(), &[9.0]);
}

#[test]
fn checkpoint_reuses_existing_cached_data_without_recompiling() {
    let trace = TraceContext::new();
    let x = trace
        .from_vec_col_major(vec![2], vec![1.0_f64, 2.0])
        .unwrap();
    let mut y = (&x + &x).unwrap();

    let mut compiler = GraphCompiler::new();
    let executor = cpu_runtime();
    y.checkpoint(
        &mut compiler,
        &executor,
        &mut tenferro_cpu::CpuEngine::serial(),
    )
    .unwrap();

    y.checkpoint(
        &mut compiler,
        &executor,
        &mut tenferro_cpu::CpuEngine::serial(),
    )
    .unwrap();

    let program = compiler.compile(&y).unwrap();
    let out = run_compiled_one(&executor, &program, &[]).unwrap();
    assert_eq!(out.as_slice::<f64>().unwrap(), &[2.0, 4.0]);
}

#[test]
fn checkpoint_gradient_runs_through_runtime() {
    let trace = TraceContext::new();
    let x = trace.from_vec_col_major(vec![], vec![2.0_f64]).unwrap();
    let mut y = (&x * &x).unwrap();

    let mut compiler = GraphCompiler::new();
    let executor = cpu_runtime();
    y.checkpoint(
        &mut compiler,
        &executor,
        &mut tenferro_cpu::CpuEngine::serial(),
    )
    .unwrap();

    let z = (&y * &y).unwrap();
    let grad = z.grad(&x).unwrap();
    let program = compiler.compile(&grad).unwrap();
    let out = run_compiled_one(&executor, &program, &[]).unwrap();

    assert_eq!(out.as_slice::<f64>().unwrap(), &[32.0]);
}

#[test]
fn runtime_validates_runtime_bindings() {
    let trace = TraceContext::new();
    let x = trace.input_symbolic_shape(DType::F64, 1).unwrap();
    let y = (&x + &x).unwrap();

    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_with_input_specs(&y, &[(&x, DType::F64, &[3])])
        .unwrap();
    let executor = cpu_runtime();

    let ok = Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]).unwrap();
    let out = run_compiled_one(&executor, &program, &[&ok]).unwrap();
    assert_eq!(out.as_slice::<f64>().unwrap(), &[2.0, 4.0, 6.0]);

    let wrong_shape = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let err = run_compiled_one(&executor, &program, &[&wrong_shape]).unwrap_err();
    assert!(format!("{err}").contains("shape"));
}

#[test]
fn runtime_rejects_invalid_ordered_inputs() {
    let trace = TraceContext::new();
    let x = trace.input_symbolic_shape(DType::F64, 1).unwrap();
    let y = (&x + &x).unwrap();

    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_with_input_specs(&y, &[(&x, DType::F64, &[2])])
        .unwrap();
    let executor = cpu_runtime();

    let bound = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let err = run_compiled_one(&executor, &program, &[&bound, &bound]).unwrap_err();
    assert!(
        matches!(
            err,
            Error::GraphInputCountMismatch {
                expected: 1,
                actual: 2
            }
        ),
        "got {err:?}"
    );

    let wrong_dtype = Tensor::from_vec_col_major(vec![2], vec![1.0_f32, 2.0]).unwrap();
    let err = run_compiled_one(&executor, &program, &[&wrong_dtype]).unwrap_err();
    assert!(
        matches!(
            err,
            Error::PlaceholderDtypeMismatch {
                expected: DType::F64,
                actual: DType::F32
            }
        ),
        "got {err:?}"
    );

    let err = run_compiled_one(&executor, &program, &[]).unwrap_err();
    assert!(
        matches!(err, Error::UnboundPlaceholder { .. }),
        "got {err:?}"
    );
}

#[test]
fn runtime_cache_stats_are_separate_from_compiler_stats() {
    let trace = TraceContext::new();
    let x = trace
        .from_vec_col_major(vec![2], vec![1.0_f64, 2.0])
        .unwrap();
    let y = (&x + &x).unwrap();

    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(&y).unwrap();
    let executor = cpu_runtime();
    let _ = run_compiled_one(&executor, &program, &[]).unwrap();

    assert_eq!(compiler.cache_stats().entries, 0);
    assert_eq!(executor.cache_stats().unwrap().extensions.entries, 0);
    assert!(
        executor.cache_stats().unwrap().prepared_plans.entries > 0,
        "runtime execution should populate the prepared-plan cache"
    );
}

#[test]
fn runtime_cache_controls_are_available() {
    let executor = cpu_runtime();

    let stats = executor.cache_stats().unwrap();
    assert_eq!(stats.extensions.entries, 0);

    executor.clear_caches().unwrap();

    let stats = executor.cache_stats().unwrap();
    assert_eq!(stats.extensions.entries, 0);
    assert_eq!(stats.prepared_plans.entries, 0);
}
