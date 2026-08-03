use tenferro_cpu::CpuBackend;
use tenferro_ops::dim_expr::DimExpr;
use tenferro_runtime::{
    CompiledGraph, DType, DotGeneralConfig, Error, ErrorPhase, GatherConfig, GraphCompiler,
    PadConfig, Runtime, ScatterConfig, SliceConfig, Tensor, TensorOpsExt, TensorValue,
    TracedTensor,
};
use tenferro_tensor::{Error as TensorError, ValidationError};

fn cpu_runtime() -> Runtime {
    let backend = CpuBackend::new();
    let mut builder = Runtime::builder();
    builder
        .register_engine(tenferro_cpu::runtime_engine_registration(&backend).unwrap())
        .unwrap();
    builder.build().unwrap()
}

fn run_compiled_one(program: &CompiledGraph, inputs: &[&Tensor]) -> Tensor {
    let mut outputs = cpu_runtime().run_compiled(program, inputs).unwrap();
    assert_eq!(outputs.len(), 1);
    outputs.pop().unwrap()
}

#[test]
fn runtime_prepared_compiled_graph_runs_repeated_inputs() {
    let runtime = cpu_runtime();
    let x = TracedTensor::input_concrete_shape(DType::F64, &[2]).unwrap();
    let y = (&x + &x).unwrap();
    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_with_input_specs(&y, &[(&x, DType::F64, &[2])])
        .unwrap();
    let first = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let second = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).unwrap();

    let prepared = runtime.prepare_compiled(&program, &[&first]).unwrap();

    let mut outputs = runtime.run_prepared(&prepared, &[&first]).unwrap();
    assert_eq!(
        outputs.pop().unwrap().as_slice::<f64>().unwrap(),
        &[2.0, 4.0]
    );
    let mut outputs = runtime.run_prepared(&prepared, &[&second]).unwrap();
    assert_eq!(
        outputs.pop().unwrap().as_slice::<f64>().unwrap(),
        &[6.0, 8.0]
    );

    let other_runtime = cpu_runtime();
    assert!(other_runtime.run_prepared(&prepared, &[&first]).is_err());
}

#[test]
fn runtime_prepared_execution_hot_path_keeps_input_metadata_inline() {
    let source = include_str!("../../src/runtime/execution.rs");

    assert!(
        source.contains("type RuntimeInputRefs<'a> = SmallVec"),
        "Runtime::run_prepared should keep short input reference lists inline"
    );
    assert!(
        source.contains("type RuntimeInputShapes<'a> = SmallVec"),
        "Runtime::run_prepared should keep short input shape lists inline"
    );
    assert!(
        !source.contains("inputs.to_vec()"),
        "Runtime::run_prepared should not allocate a heap Vec just to copy input refs"
    );
}

#[test]
fn runtime_crate_exposes_traced_graph_execution_api() {
    let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let y = (&x + &x).unwrap();

    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(&y).unwrap();
    let out = run_compiled_one(&program, &[]);

    assert_eq!(out.as_slice::<f64>().unwrap(), &[2.0, 4.0]);
}

#[test]
fn runtime_program_api_is_public_read_only_and_bounded_debug() {
    use tenferro_runtime::program::{
        CoreSemanticOp, ProgramInputSpec, SemanticProgramBuilder, SemanticTransform,
    };

    fn accepts_transform_object(_: &dyn SemanticTransform) {}

    let mut builder = SemanticProgramBuilder::new();
    let input = builder
        .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(2)]))
        .unwrap();
    let output = builder.add_op(CoreSemanticOp::Neg, &[input]).unwrap()[0];
    let frozen = builder.finish(&[output]).unwrap();
    let operation = frozen.program.operations().next().unwrap();

    assert_eq!(frozen.program.inputs(), &[input]);
    assert_eq!(frozen.program.outputs(), &[output]);
    assert!(format!("{operation:?}").len() < 256);
    assert!(format!("{:?}", frozen.program).len() < 256);
    let _ = accepts_transform_object;
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
        TensorError::Validation {
            op: "matmul",
            source: ValidationError::RankMismatch {
                expected: 2,
                actual: 0,
            },
        }
    ));
}

#[test]
fn traced_tensor_methods_cover_conversion_and_rank_errors() {
    let scalar = TracedTensor::from_vec_col_major(vec![], vec![1.0_f64]).unwrap();
    let vector = TracedTensor::from_vec_col_major(vec![2], vec![1.25_f64, -2.75]).unwrap();

    let converted = vector.convert(DType::C64).unwrap();
    assert_eq!(converted.dtype, DType::C64);

    let casted = vector.cast(DType::I32).unwrap();
    assert_eq!(casted.dtype, DType::I32);

    let err = scalar.matmul(&vector).unwrap_err();
    assert!(matches!(
        &err,
        Error::Validation {
            op: "TracedTensor::matmul",
            phase: ErrorPhase::GraphBuild,
            source: ValidationError::RankMismatch {
                expected: 2,
                actual: 0,
            },
        }
    ));
}

#[test]
fn traced_tensor_methods_cover_structural_surface() {
    fn run(output: &TracedTensor) -> Tensor {
        let mut compiler = GraphCompiler::new();
        let program = compiler.compile(output).unwrap();
        run_compiled_one(&program, &[])
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

    let matrix_3x4 = TracedTensor::from_vec_col_major(
        vec![3, 4],
        vec![
            1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    )
    .unwrap();
    let row_slice = matrix_3x4.slice_axis(0, 1..3).unwrap();
    assert_eq!(row_slice.try_concrete_shape(), Some(vec![2, 4]));
    assert_eq!(
        run(&row_slice).as_slice::<f64>().unwrap(),
        &[2.0, 3.0, 5.0, 6.0, 8.0, 9.0, 11.0, 12.0]
    );

    let builder_slice = matrix_3x4
        .slice_builder()
        .axis(0, 1..3)
        .axis_step(1, 0..4, 2)
        .apply()
        .unwrap();
    assert_eq!(builder_slice.try_concrete_shape(), Some(vec![2, 2]));
    assert_eq!(
        run(&builder_slice).as_slice::<f64>().unwrap(),
        &[2.0, 3.0, 8.0, 9.0]
    );

    let selected = matrix_3x4.take_axis(1, &[3, 1, 3]).unwrap();
    assert_eq!(selected.try_concrete_shape(), Some(vec![3, 3]));
    assert_eq!(
        run(&selected).as_slice::<f64>().unwrap(),
        &[10.0, 11.0, 12.0, 4.0, 5.0, 6.0, 10.0, 11.0, 12.0]
    );

    let mixed = matrix_3x4
        .slice_builder()
        .axis(0, 0..2)
        .take_axis(1, &[3, 1, 3])
        .apply()
        .unwrap();
    assert_eq!(mixed.try_concrete_shape(), Some(vec![2, 3]));
    assert_eq!(
        run(&mixed).as_slice::<f64>().unwrap(),
        &[10.0, 11.0, 4.0, 5.0, 10.0, 11.0]
    );

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
        run(&matrix.tril(0).unwrap()).as_slice::<f64>().unwrap(),
        &[1.0, 2.0, 0.0, 4.0]
    );
    assert_eq!(
        run(&matrix.triu(0).unwrap()).as_slice::<f64>().unwrap(),
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
        &err,
        Error::Validation {
            phase: ErrorPhase::GraphBuild,
            source: ValidationError::InvalidArgument { .. },
            ..
        }
    ));

    let err = TracedTensor::stack(&[&x], 0).unwrap_err();
    assert!(matches!(
        &err,
        Error::Validation {
            phase: ErrorPhase::GraphBuild,
            source: ValidationError::InvalidArgument { .. },
            ..
        }
    ));
}

#[test]
fn runtime_runs_elementwise_and_reduction_with_ordered_inputs() {
    let x = TracedTensor::input_symbolic_shape(DType::F64, 1).unwrap();
    let y = (&x + &x).unwrap().reduce_sum(Some(&[0])).unwrap();
    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_with_input_specs(&y, &[(&x, DType::F64, &[2])])
        .unwrap();
    let input = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();

    let out = cpu_runtime().run_compiled(&program, &[&input]).unwrap();

    assert_eq!(out.len(), 1);
    assert_eq!(out[0].as_slice::<f64>().unwrap(), &[6.0]);
    assert_eq!(input.as_slice::<f64>().unwrap(), &[1.0, 2.0]);
}

#[test]
fn traced_broadcast_binary_accepts_symbolic_same_rank_input() {
    let x = TracedTensor::input_symbolic_shape(DType::F64, 1).unwrap();
    let y_data = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let y = TracedTensor::from_tensor_concrete_shape(y_data.duplicate().unwrap()).unwrap();

    let z = (&x + &y).unwrap();

    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_with_input_specs(&z, &[(&x, DType::F64, &[2])])
        .unwrap();
    let input = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).unwrap();
    let out = run_compiled_one(&program, &[&input, &y_data]);

    assert_eq!(out.as_slice::<f64>().unwrap(), &[4.0, 6.0]);
}

#[test]
fn traced_reduction_with_too_many_axes_returns_error_without_rank_underflow() {
    let x = TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64; 4]).unwrap();

    let err = x.reduce_max(Some(&[0, 1, 2])).unwrap_err().to_string();

    assert!(err.contains("axis 2 out of bounds for rank 2"), "{err}");
}

#[test]
fn runtime_runs_dot_general_with_ordered_inputs() {
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

    let out = cpu_runtime()
        .run_compiled(&program, &[&lhs_data, &rhs_data])
        .unwrap();

    assert_eq!(out.len(), 1);
    assert_eq!(out[0].shape(), &[2, 2]);
    assert_eq!(out[0].as_slice::<f64>().unwrap(), &[22.0, 28.0, 49.0, 64.0]);
}

#[test]
fn runtime_can_return_final_transpose_as_lazy_value() {
    let x = TracedTensor::input_symbolic_shape(DType::F64, 2).unwrap();
    let y = (&x + &x).unwrap().transpose(&[1, 0]).unwrap();
    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_with_input_specs(&y, &[(&x, DType::F64, &[2, 3])])
        .unwrap();
    let input =
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

    let compact = cpu_runtime().run_compiled(&program, &[&input]).unwrap();
    assert_eq!(
        compact[0].as_slice::<f64>().unwrap(),
        &[2.0, 6.0, 10.0, 4.0, 8.0, 12.0]
    );

    let values = cpu_runtime()
        .run_compiled_values(&program, &[&input])
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
}

#[test]
fn runtime_ordered_input_errors_are_covered() {
    let x = TracedTensor::input_symbolic_shape(DType::F64, 1).unwrap();
    let y = (&x + &x).unwrap();
    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_with_input_specs(&y, &[(&x, DType::F64, &[2])])
        .unwrap();
    let input = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let runtime = cpu_runtime();

    let out = run_compiled_one(&program, &[&input]);
    assert_eq!(out.as_slice::<f64>().unwrap(), &[2.0, 4.0]);

    let unbound = runtime.run_compiled(&program, &[]).unwrap_err();
    assert!(matches!(unbound, Error::UnboundPlaceholder { .. }));

    let extra = runtime
        .run_compiled(&program, &[&input, &input])
        .unwrap_err();
    assert!(matches!(extra, Error::GraphInputCountMismatch { .. }));

    let f32_input = Tensor::from_vec_col_major(vec![2], vec![1.0_f32, 2.0]).unwrap();
    let dtype = runtime.run_compiled(&program, &[&f32_input]).unwrap_err();
    assert!(matches!(dtype, Error::PlaceholderDtypeMismatch { .. }));

    let rank_input = Tensor::from_vec_col_major(vec![1, 2], vec![1.0_f64, 2.0]).unwrap();
    let rank = runtime.run_compiled(&program, &[&rank_input]).unwrap_err();
    assert!(matches!(rank, Error::PlaceholderRankMismatch { .. }));

    let shape_input = Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]).unwrap();
    let shape = runtime.run_compiled(&program, &[&shape_input]).unwrap_err();
    assert!(matches!(shape, Error::PlaceholderShapeMismatch { .. }));
}
