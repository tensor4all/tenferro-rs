use tenferro_cpu::CpuBackend;
use tenferro_runtime::{
    tensor, DType, DotGeneralConfig, Error, GraphCompiler, GraphExecutor, Tensor, TensorRead,
    TracedTensor,
};

#[test]
fn runtime_crate_exposes_traced_graph_execution_api() {
    let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
    let y = &x + &x;

    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(&y).unwrap();
    let out = GraphExecutor::new(CpuBackend::default())
        .run(&program)
        .unwrap();

    assert_eq!(out.as_slice::<f64>().unwrap(), &[2.0, 4.0]);
}

#[test]
fn tensor_module_free_functions_cover_eager_runtime_paths() {
    let mut backend = CpuBackend::new();
    let input = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]);

    let converted = tensor::convert(&input, DType::F32, &mut backend).unwrap();
    assert_eq!(converted.dtype(), DType::F32);
    assert_eq!(converted.as_slice::<f32>().unwrap(), &[1.0, 2.0, 3.0, 4.0]);

    let reshaped = tensor::reshape(&input, &[4], &mut backend).unwrap();
    assert_eq!(reshaped.shape(), &[4]);
    assert_eq!(reshaped.as_slice::<f64>().unwrap(), &[1.0, 2.0, 3.0, 4.0]);

    let transposed = tensor::transpose(&input, &[1, 0], &mut backend).unwrap();
    assert_eq!(transposed.shape(), &[2, 2]);
    assert_eq!(transposed.as_slice::<f64>().unwrap(), &[1.0, 3.0, 2.0, 4.0]);

    let summed = tensor::reduce_sum(&input, &[0], &mut backend).unwrap();
    assert_eq!(summed.shape(), &[2]);
    assert_eq!(summed.as_slice::<f64>().unwrap(), &[3.0, 7.0]);
}

#[test]
fn graph_executor_runs_elementwise_and_reduction_with_borrowed_inputs() {
    let x = TracedTensor::input_symbolic_shape(DType::F64, 1);
    let y = (&x + &x).reduce_sum(&[0]);
    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_with_input_specs(&y, &[(&x, DType::F64, &[2])])
        .unwrap();
    let input = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
    let mut executor = GraphExecutor::new(CpuBackend::new());

    let out = executor
        .run_many_with_input_reads(&program, &[(&x, TensorRead::from_tensor(&input))])
        .unwrap();

    assert_eq!(out.len(), 1);
    assert_eq!(out[0].as_slice::<f64>().unwrap(), &[6.0]);
    assert_eq!(input.as_slice::<f64>().unwrap(), &[1.0, 2.0]);
}

#[test]
fn graph_executor_runs_dot_general_with_borrowed_inputs() {
    let lhs = TracedTensor::input_symbolic_shape(DType::F64, 2);
    let rhs = TracedTensor::input_symbolic_shape(DType::F64, 2);
    let product = lhs.dot_general(
        &rhs,
        DotGeneralConfig {
            lhs_contracting_dims: vec![1],
            rhs_contracting_dims: vec![0],
            lhs_batch_dims: vec![],
            rhs_batch_dims: vec![],
        },
    );
    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_with_input_specs(
            &product,
            &[(&lhs, DType::F64, &[2, 3]), (&rhs, DType::F64, &[3, 2])],
        )
        .unwrap();
    let lhs_data = Tensor::from_vec_col_major(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let rhs_data = Tensor::from_vec_col_major(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let mut executor = GraphExecutor::new(CpuBackend::new());

    let out = executor
        .run_many_with_input_reads(
            &program,
            &[
                (&lhs, TensorRead::from_tensor(&lhs_data)),
                (&rhs, TensorRead::from_tensor(&rhs_data)),
            ],
        )
        .unwrap();

    assert_eq!(out.len(), 1);
    assert_eq!(out[0].shape(), &[2, 2]);
    assert_eq!(out[0].as_slice::<f64>().unwrap(), &[22.0, 28.0, 49.0, 64.0]);
}

#[test]
fn graph_executor_public_helpers_and_borrowed_input_errors_are_covered() {
    let mut executor = GraphExecutor::<CpuBackend>::default();
    executor.extension_executor_mut().clear_caches();
    assert_eq!(executor.cache_stats().extensions.entries, 0);
    executor.reclaim_outputs(vec![Tensor::from_vec_col_major(vec![1], vec![0.0_f64])]);

    let x = TracedTensor::input_symbolic_shape(DType::F64, 1);
    let y = &x + &x;
    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_with_input_specs(&y, &[(&x, DType::F64, &[2])])
        .unwrap();
    let input = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);

    let out = executor
        .run_with_input_reads(&program, &[(&x, TensorRead::from_tensor(&input))])
        .unwrap();
    assert_eq!(out.as_slice::<f64>().unwrap(), &[2.0, 4.0]);

    let unbound = executor
        .run_many_with_input_reads(&program, &[])
        .unwrap_err();
    assert!(matches!(unbound, Error::UnboundPlaceholder { .. }));

    let concrete = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
    let unexpected = executor
        .run_many_with_input_reads(&program, &[(&concrete, TensorRead::from_tensor(&input))])
        .unwrap_err();
    assert!(matches!(unexpected, Error::UnexpectedBinding { .. }));

    let duplicate = executor
        .run_many_with_input_reads(
            &program,
            &[
                (&x, TensorRead::from_tensor(&input)),
                (&x, TensorRead::from_tensor(&input)),
            ],
        )
        .unwrap_err();
    assert!(matches!(duplicate, Error::DuplicateBinding { .. }));

    let f32_input = Tensor::from_vec_col_major(vec![2], vec![1.0_f32, 2.0]);
    let dtype = executor
        .run_many_with_input_reads(&program, &[(&x, TensorRead::from_tensor(&f32_input))])
        .unwrap_err();
    assert!(matches!(dtype, Error::PlaceholderDtypeMismatch { .. }));

    let rank_input = Tensor::from_vec_col_major(vec![1, 2], vec![1.0_f64, 2.0]);
    let rank = executor
        .run_many_with_input_reads(&program, &[(&x, TensorRead::from_tensor(&rank_input))])
        .unwrap_err();
    assert!(matches!(rank, Error::PlaceholderRankMismatch { .. }));

    let shape_input = Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]);
    let shape = executor
        .run_many_with_input_reads(&program, &[(&x, TensorRead::from_tensor(&shape_input))])
        .unwrap_err();
    assert!(matches!(shape, Error::PlaceholderShapeMismatch { .. }));

    let multi = compiler
        .compile_many(&[&concrete, &concrete.neg()])
        .unwrap();
    let output_count = executor.run(&multi).unwrap_err();
    assert!(output_count.to_string().contains("expected 1 output"));
}
