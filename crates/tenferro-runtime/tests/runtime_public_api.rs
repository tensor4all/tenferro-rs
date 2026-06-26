use tenferro_cpu::CpuBackend;
use tenferro_runtime::{
    DType, DotGeneralConfig, Error, GatherConfig, GraphCompiler, GraphExecutor, PadConfig,
    ScatterConfig, SliceConfig, Tensor, TensorOpsExt, TensorRead, TensorValue, TracedTensor,
};
use tenferro_tensor::Error as TensorError;

#[test]
fn runtime_crate_exposes_traced_graph_execution_api() {
    let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let y = (&x + &x).unwrap();

    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(&y).unwrap();
    let out = GraphExecutor::new(CpuBackend::default())
        .run(&program)
        .unwrap();

    assert_eq!(out.as_slice::<f64>().unwrap(), &[2.0, 4.0]);
}

#[test]
fn tensor_extension_trait_covers_eager_runtime_paths() {
    let mut backend = CpuBackend::new();
    let input = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let f32_input = Tensor::from_vec_col_major(vec![2], vec![1.0_f32, 2.0]).unwrap();

    let converted = f32_input.convert(DType::F64, &mut backend).unwrap();
    assert_eq!(converted.dtype(), DType::F64);
    assert_eq!(converted.as_slice::<f64>().unwrap(), &[1.0, 2.0]);

    let casted = input.cast(DType::F32, &mut backend).unwrap();
    assert_eq!(casted.dtype(), DType::F32);
    assert_eq!(casted.as_slice::<f32>().unwrap(), &[1.0, 2.0, 3.0, 4.0]);

    let reshaped = input.reshape(&[4], &mut backend).unwrap();
    assert_eq!(reshaped.shape(), &[4]);
    assert_eq!(reshaped.as_slice::<f64>().unwrap(), &[1.0, 2.0, 3.0, 4.0]);

    let transposed = input.transpose(&[1, 0], &mut backend).unwrap();
    assert_eq!(transposed.shape(), &[2, 2]);
    assert_eq!(transposed.as_slice::<f64>().unwrap(), &[1.0, 3.0, 2.0, 4.0]);

    let summed = input.reduce_sum(&[0], &mut backend).unwrap();
    assert_eq!(summed.shape(), &[2]);
    assert_eq!(summed.as_slice::<f64>().unwrap(), &[3.0, 7.0]);
}

#[test]
fn concrete_tensor_matmul_rejects_non_matrix_inputs_without_rank_underflow() {
    let mut backend = CpuBackend::new();
    let scalar = Tensor::from_vec_col_major(vec![], vec![1.0_f64]).unwrap();
    let vector = Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap();

    let err = scalar.matmul(&vector, &mut backend).unwrap_err();

    assert!(matches!(
        err,
        TensorError::RankMismatch {
            op: "matmul",
            expected: 2,
            actual: 0,
        }
    ));
}

#[test]
fn traced_tensor_methods_cover_conversion_and_rank_errors() {
    let scalar = TracedTensor::from_vec_col_major(vec![], vec![1.0_f64]).unwrap();
    let vector = TracedTensor::from_vec_col_major(vec![2], vec![1.25_f64, -2.75]).unwrap();

    let converted = vector.convert(DType::C64).unwrap();
    assert_eq!(converted.dtype, DType::C64);

    let casted = vector.cast(DType::I32);
    assert_eq!(casted.dtype, DType::I32);

    let err = scalar.matmul(&vector).unwrap_err();
    assert!(matches!(
        err,
        Error::InvalidGraphBuild {
            op: "TracedTensor::matmul",
            ..
        }
    ));
}

#[test]
fn traced_tensor_methods_cover_structural_surface() {
    fn run(output: &TracedTensor) -> Tensor {
        let mut compiler = GraphCompiler::new();
        let program = compiler.compile(output).unwrap();
        GraphExecutor::new(CpuBackend::new()).run(&program).unwrap()
    }

    let vector = TracedTensor::from_vec_col_major(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let sliced = vector
        .slice(SliceConfig {
            starts: vec![1],
            limits: vec![3],
            strides: vec![1],
        })
        .unwrap();
    assert_eq!(run(&sliced).as_slice::<f64>().unwrap(), &[2.0, 3.0]);

    let padded = sliced
        .pad(PadConfig {
            edge_padding_low: vec![1],
            edge_padding_high: vec![1],
            interior_padding: vec![0],
        })
        .unwrap();
    let reversed = padded.reverse(&[0]).unwrap();
    assert_eq!(
        run(&reversed).as_slice::<f64>().unwrap(),
        &[0.0, 3.0, 2.0, 0.0]
    );

    let starts = TracedTensor::from_vec_col_major(vec![1], vec![1_i64]).unwrap();
    let dynamic = vector.dynamic_slice(&starts, &[2]).unwrap();
    assert_eq!(run(&dynamic).as_slice::<f64>().unwrap(), &[2.0, 3.0]);

    let indices = TracedTensor::from_vec_col_major(vec![3], vec![3_i64, 1, 0]).unwrap();
    let gathered = vector
        .gather(
            &indices,
            GatherConfig {
                offset_dims: vec![],
                collapsed_slice_dims: vec![0],
                start_index_map: vec![0],
                index_vector_dim: 1,
                slice_sizes: vec![1],
            },
        )
        .unwrap();
    assert_eq!(run(&gathered).as_slice::<f64>().unwrap(), &[4.0, 2.0, 1.0]);

    let operand = TracedTensor::from_vec_col_major(vec![4], vec![0.0_f64, 0.0, 0.0, 0.0]).unwrap();
    let scatter_indices = TracedTensor::from_vec_col_major(vec![2, 1], vec![1_i64, 3]).unwrap();
    let updates = TracedTensor::from_vec_col_major(vec![2], vec![5.0_f64, 7.0]).unwrap();
    let scattered = operand
        .scatter(
            &scatter_indices,
            &updates,
            ScatterConfig {
                update_window_dims: vec![],
                inserted_window_dims: vec![0],
                scatter_dims_to_operand_dims: vec![0],
                index_vector_dim: 1,
            },
        )
        .unwrap();
    assert_eq!(
        run(&scattered).as_slice::<f64>().unwrap(),
        &[0.0, 5.0, 0.0, 7.0]
    );

    let concatenated = TracedTensor::concatenate(&[&vector, &vector], 0).unwrap();
    assert_eq!(run(&concatenated).shape(), &[8]);

    let matrix =
        TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    assert_eq!(
        run(&matrix.tril(0)).as_slice::<f64>().unwrap(),
        &[1.0, 2.0, 0.0, 4.0]
    );
    assert_eq!(
        run(&matrix.triu(0)).as_slice::<f64>().unwrap(),
        &[1.0, 0.0, 3.0, 4.0]
    );
    let rectangular =
        TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
    let diag = rectangular.extract_diag(1, 0).unwrap();
    assert_eq!(diag.try_concrete_shape(), Some(vec![2]));
    assert_eq!(run(&diag).shape(), &[2]);

    let lhs = TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap();
    let rhs = TracedTensor::from_vec_col_major(vec![3, 2], vec![1.0_f64; 6]).unwrap();
    assert_eq!(run(&lhs.matmul(&rhs).unwrap()).shape(), &[2, 2]);
}

#[test]
fn traced_shape_packing_rejects_symbolic_shapes_as_graph_build_errors() {
    let x = TracedTensor::input_symbolic_shape(DType::F64, 1).unwrap();

    let err = x.index_select(0, &[0]).unwrap_err();
    assert!(matches!(
        err,
        Error::InvalidGraphBuild {
            op: "index_select",
            ..
        }
    ));

    let err = TracedTensor::stack(&[&x], 0).unwrap_err();
    assert!(matches!(err, Error::InvalidGraphBuild { op: "stack", .. }));
}

#[test]
fn graph_executor_runs_elementwise_and_reduction_with_borrowed_inputs() {
    let x = TracedTensor::input_symbolic_shape(DType::F64, 1).unwrap();
    let y = (&x + &x).unwrap().reduce_sum(&[0]).unwrap();
    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_with_input_specs(&y, &[(&x, DType::F64, &[2])])
        .unwrap();
    let input = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let mut executor = GraphExecutor::new(CpuBackend::new());

    let out = executor
        .run_many_with_input_reads(&program, &[(&x, TensorRead::from_tensor(&input))])
        .unwrap();

    assert_eq!(out.len(), 1);
    assert_eq!(out[0].as_slice::<f64>().unwrap(), &[6.0]);
    assert_eq!(input.as_slice::<f64>().unwrap(), &[1.0, 2.0]);
}

#[test]
fn traced_broadcast_binary_accepts_symbolic_same_rank_input() {
    let x = TracedTensor::input_symbolic_shape(DType::F64, 1).unwrap();
    let y = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();

    let z = (&x + &y).unwrap();

    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_with_input_specs(&z, &[(&x, DType::F64, &[2])])
        .unwrap();
    let input = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).unwrap();
    let out = GraphExecutor::new(CpuBackend::new())
        .run_with_input_reads(&program, &[(&x, TensorRead::from_tensor(&input))])
        .unwrap();

    assert_eq!(out.as_slice::<f64>().unwrap(), &[4.0, 6.0]);
}

#[test]
fn traced_reduction_with_too_many_axes_returns_error_without_rank_underflow() {
    let x = TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64; 4]).unwrap();

    let err = x.reduce_max(&[0, 1, 2]).unwrap_err().to_string();

    assert!(err.contains("axis 2 out of bounds for rank 2"), "{err}");
}

#[test]
fn graph_executor_runs_dot_general_with_borrowed_inputs() {
    let lhs = TracedTensor::input_symbolic_shape(DType::F64, 2).unwrap();
    let rhs = TracedTensor::input_symbolic_shape(DType::F64, 2).unwrap();
    let product = lhs
        .dot_general(
            &rhs,
            DotGeneralConfig {
                lhs_contracting_dims: vec![1],
                rhs_contracting_dims: vec![0],
                lhs_batch_dims: vec![],
                rhs_batch_dims: vec![],
            },
        )
        .unwrap();
    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_with_input_specs(
            &product,
            &[(&lhs, DType::F64, &[2, 3]), (&rhs, DType::F64, &[3, 2])],
        )
        .unwrap();
    let lhs_data =
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let rhs_data =
        Tensor::from_vec_col_major(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
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
fn graph_executor_can_return_final_transpose_as_lazy_value() {
    let x = TracedTensor::input_symbolic_shape(DType::F64, 2).unwrap();
    let y = (&x + &x).unwrap().transpose(&[1, 0]).unwrap();
    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_with_input_specs(&y, &[(&x, DType::F64, &[2, 3])])
        .unwrap();
    let input =
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let mut executor = GraphExecutor::new(CpuBackend::new());

    let compact = executor
        .run_many_with_input_reads(&program, &[(&x, TensorRead::from_tensor(&input))])
        .unwrap();
    assert_eq!(
        compact[0].as_slice::<f64>().unwrap(),
        &[2.0, 6.0, 10.0, 4.0, 8.0, 12.0]
    );

    let values = executor
        .run_many_values_with_input_reads(&program, &[(&x, TensorRead::from_tensor(&input))])
        .unwrap();

    assert_eq!(values.len(), 1);
    assert_eq!(values[0].shape(), &[3, 2]);
    match &values[0] {
        TensorValue::View(view) => {
            assert_eq!(view.shape(), &[3, 2]);
            assert_eq!(view.strides(), &[2, 1]);
        }
        TensorValue::Tensor(_) => panic!("final transpose should stay as a lazy owned view"),
    }
    assert_eq!(
        values[0].to_tensor().unwrap().as_slice::<f64>().unwrap(),
        &[2.0, 6.0, 10.0, 4.0, 8.0, 12.0]
    );
}

#[test]
fn graph_executor_public_helpers_and_borrowed_input_errors_are_covered() {
    let mut executor = GraphExecutor::<CpuBackend>::default();
    executor.extension_executor_mut().clear_caches();
    assert_eq!(executor.cache_stats().extensions.entries, 0);
    executor.reclaim_outputs(vec![
        Tensor::from_vec_col_major(vec![1], vec![0.0_f64]).unwrap()
    ]);

    let x = TracedTensor::input_symbolic_shape(DType::F64, 1).unwrap();
    let y = (&x + &x).unwrap();
    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_with_input_specs(&y, &[(&x, DType::F64, &[2])])
        .unwrap();
    let input = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();

    let out = executor
        .run_with_input_reads(&program, &[(&x, TensorRead::from_tensor(&input))])
        .unwrap();
    assert_eq!(out.as_slice::<f64>().unwrap(), &[2.0, 4.0]);

    let unbound = executor
        .run_many_with_input_reads(&program, &[])
        .unwrap_err();
    assert!(matches!(unbound, Error::UnboundPlaceholder { .. }));

    let concrete = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
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

    let f32_input = Tensor::from_vec_col_major(vec![2], vec![1.0_f32, 2.0]).unwrap();
    let dtype = executor
        .run_many_with_input_reads(&program, &[(&x, TensorRead::from_tensor(&f32_input))])
        .unwrap_err();
    assert!(matches!(dtype, Error::PlaceholderDtypeMismatch { .. }));

    let rank_input = Tensor::from_vec_col_major(vec![1, 2], vec![1.0_f64, 2.0]).unwrap();
    let rank = executor
        .run_many_with_input_reads(&program, &[(&x, TensorRead::from_tensor(&rank_input))])
        .unwrap_err();
    assert!(matches!(rank, Error::PlaceholderRankMismatch { .. }));

    let shape_input = Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]).unwrap();
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
