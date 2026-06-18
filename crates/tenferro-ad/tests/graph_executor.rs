use tenferro_ad::TracedTensorAdExt;
use tenferro_cpu::CpuBackend;
use tenferro_runtime::error::Error;
use tenferro_runtime::extension::{ExecInstruction, ExecOp, ExecProgram};
use tenferro_runtime::{DType, GraphCompiler, GraphExecutor, Tensor, TracedTensor};

#[test]
fn graph_executor_runs_compiled_single_output_program() {
    let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let y = (&x + &x).unwrap();

    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(&y).unwrap();
    let mut executor = GraphExecutor::new(CpuBackend::new());

    let out = executor.run(&program).unwrap();

    assert_eq!(out.as_slice::<f64>().unwrap(), &[2.0, 4.0]);
}

#[test]
fn graph_executor_runs_compiled_multi_output_program() {
    let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let sum = (&x + &x).unwrap();
    let product = (&x * &x).unwrap();

    let mut compiler = GraphCompiler::new();
    let program = compiler.compile_many(&[&sum, &product]).unwrap();
    let mut executor = GraphExecutor::new(CpuBackend::new());

    let outputs = executor.run_many(&program).unwrap();

    assert_eq!(outputs.len(), 2);
    assert_eq!(outputs[0].as_slice::<f64>().unwrap(), &[2.0, 4.0]);
    assert_eq!(outputs[1].as_slice::<f64>().unwrap(), &[1.0, 4.0]);
}

#[test]
fn checkpoint_uses_explicit_compiler_and_executor() {
    let x = TracedTensor::from_vec_col_major(vec![], vec![3.0_f64]).unwrap();
    let mut y = (&x * &x).unwrap();

    let mut compiler = GraphCompiler::new();
    let mut executor = GraphExecutor::new(CpuBackend::new());
    y.checkpoint(&mut compiler, &mut executor).unwrap();

    let program = compiler.compile(&y).unwrap();
    let out = executor.run(&program).unwrap();
    assert_eq!(out.as_slice::<f64>().unwrap(), &[9.0]);
}

#[test]
fn checkpoint_reuses_existing_cached_data_without_recompiling() {
    let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let mut y = (&x + &x).unwrap();

    let mut compiler = GraphCompiler::new();
    let mut executor = GraphExecutor::new(CpuBackend::new());
    y.checkpoint(&mut compiler, &mut executor).unwrap();

    y.checkpoint(&mut compiler, &mut executor).unwrap();

    let program = compiler.compile(&y).unwrap();
    let out = executor.run(&program).unwrap();
    assert_eq!(out.as_slice::<f64>().unwrap(), &[2.0, 4.0]);
}

#[test]
fn checkpoint_gradient_runs_through_graph_executor() {
    let x = TracedTensor::from_vec_col_major(vec![], vec![2.0_f64]).unwrap();
    let mut y = (&x * &x).unwrap();

    let mut compiler = GraphCompiler::new();
    let mut executor = GraphExecutor::new(CpuBackend::new());
    y.checkpoint(&mut compiler, &mut executor).unwrap();

    let z = (&y * &y).unwrap();
    let grad = z.grad(&x).unwrap();
    let program = compiler.compile(&grad).unwrap();
    let out = executor.run(&program).unwrap();

    assert_eq!(out.as_slice::<f64>().unwrap(), &[32.0]);
}

#[test]
fn graph_executor_validates_runtime_bindings() {
    let x = TracedTensor::input_symbolic_shape(DType::F64, 1);
    let y = (&x + &x).unwrap();

    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_with_input_specs(&y, &[(&x, DType::F64, &[3])])
        .unwrap();
    let mut executor = GraphExecutor::new(CpuBackend::new());

    let ok = Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]).unwrap();
    let out = executor.run_with_inputs(&program, &[(&x, &ok)]).unwrap();
    assert_eq!(out.as_slice::<f64>().unwrap(), &[2.0, 4.0, 6.0]);

    let wrong_shape = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let err = executor
        .run_with_inputs(&program, &[(&x, &wrong_shape)])
        .unwrap_err();
    assert!(format!("{err}").contains("shape"));
}

#[test]
fn graph_executor_rejects_invalid_runtime_bindings() {
    let x = TracedTensor::input_symbolic_shape(DType::F64, 1);
    let y = (&x + &x).unwrap();

    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_with_input_specs(&y, &[(&x, DType::F64, &[2])])
        .unwrap();
    let mut executor = GraphExecutor::new(CpuBackend::new());

    let bound = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let err = executor
        .run_with_inputs(&program, &[(&x, &bound), (&x, &bound)])
        .unwrap_err();
    assert!(matches!(err, Error::DuplicateBinding { .. }), "got {err:?}");

    let data_leaf = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let err = executor
        .run_with_inputs(&program, &[(&data_leaf, &bound)])
        .unwrap_err();
    assert!(
        matches!(err, Error::UnexpectedBinding { binding_index: 0 }),
        "got {err:?}"
    );

    let other = TracedTensor::input_symbolic_shape(DType::F64, 1);
    let err = executor
        .run_with_inputs(&program, &[(&other, &bound)])
        .unwrap_err();
    assert!(
        matches!(err, Error::UnexpectedBinding { binding_index: 0 }),
        "got {err:?}"
    );

    let wrong_dtype = Tensor::from_vec_col_major(vec![2], vec![1.0_f32, 2.0]).unwrap();
    let err = executor
        .run_with_inputs(&program, &[(&x, &wrong_dtype)])
        .unwrap_err();
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

    let err = executor.run_with_inputs(&program, &[]).unwrap_err();
    assert!(
        matches!(err, Error::UnboundPlaceholder { .. }),
        "got {err:?}"
    );
}

#[test]
fn graph_executor_eval_exec_ir_rejects_wrong_input_count() {
    let program = ExecProgram {
        instructions: vec![ExecInstruction {
            op: ExecOp::Add,
            input_slots: vec![0, 1],
            output_slots: vec![2],
            dtype: DType::F64,
            output_shapes: vec![vec![]].into(),
            output_extents: vec![vec![]].into(),
            last_use: vec![true, true],
        }],
        input_slots: vec![0, 1],
        output_slots: vec![2],
        n_slots: 3,
    };
    let inputs = vec![
        Tensor::from_vec_col_major(vec![], vec![2.0_f64]).unwrap(),
        Tensor::from_vec_col_major(vec![], vec![3.0_f64]).unwrap(),
        Tensor::from_vec_col_major(vec![], vec![5.0_f64]).unwrap(),
    ];

    let mut executor = GraphExecutor::new(CpuBackend::new());
    let err = executor.eval_exec_ir(&program, inputs).unwrap_err();

    assert!(
        format!("{err}").contains("expected 2 inputs"),
        "got {err:?}"
    );
}

#[test]
fn graph_executor_cache_stats_are_separate_from_compiler_stats() {
    let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let y = (&x + &x).unwrap();

    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(&y).unwrap();
    let mut executor = GraphExecutor::new(CpuBackend::new());
    let _ = executor.run(&program).unwrap();

    assert!(compiler.cache_stats().compile.entries > 0);
    assert_eq!(executor.cache_stats().extensions.entries, 0);
}

#[test]
fn graph_executor_runtime_cache_controls_are_available() {
    let mut executor = GraphExecutor::new(CpuBackend::new());

    let stats = executor.cache_stats();
    assert_eq!(stats.extensions.entries, 0);
    assert_eq!(stats.backend.entries, 0);

    executor.clear_backend_cache();
    assert_eq!(executor.cache_stats().backend.entries, 0);

    executor.clear_caches();
    assert_eq!(executor.cache_stats().extensions.entries, 0);
    assert_eq!(executor.cache_stats().backend.entries, 0);
}

#[test]
fn graph_executor_synthesizes_deferred_zero_tangents_from_primal_binding() {
    let x = TracedTensor::input_symbolic_shape(DType::F64, 1);
    let loss = (&x * &x).unwrap().reduce_sum(&[0]).unwrap();
    let grad = loss.grad(&x).unwrap();

    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_with_input_specs(&grad, &[(&x, DType::F64, &[4])])
        .unwrap();
    let mut executor = GraphExecutor::new(CpuBackend::new());

    let bound = Tensor::from_vec_col_major(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let out = executor.run_with_inputs(&program, &[(&x, &bound)]).unwrap();

    assert_eq!(out.shape(), &[4]);
    assert_eq!(out.as_slice::<f64>().unwrap(), &[2.0, 4.0, 6.0, 8.0]);
}
