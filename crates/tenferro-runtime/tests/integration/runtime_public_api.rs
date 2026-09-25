use tenferro_cpu::CpuBackend;
use tenferro_ops::dim_expr::DimExpr;
use tenferro_runtime::{
    CompiledGraph, DType, DotGeneralConfig, Error, ErrorPhase, GatherConfig, GraphCompiler,
    PadConfig, Runtime, ScatterConfig, SliceConfig, Tensor, TensorSessionOpsExt, TracedTensor,
};
use tenferro_tensor::{BackendSessionHost, Error as TensorError, ValidationError};

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
        source.contains("type RuntimeInputReads<'a> = SmallVec"),
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

    let (converted, casted, reshaped, transposed, summed) = backend
        .with_backend_session(
            |session| -> tenferro_tensor::Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
                let converted = f32_input.convert(DType::F64, session)?;
                let casted = input.cast(DType::F32, session)?;
                let reshaped = input.reshape(&[4], session)?;
                let transposed = input.transpose(&[1, 0], session)?;
                let summed = input.reduce_sum(&[0], session)?;
                Ok((converted, casted, reshaped, transposed, summed))
            },
        )
        .unwrap();
    assert_eq!(converted.dtype(), DType::F64);
    assert_eq!(converted.as_slice::<f64>().unwrap(), &[1.0, 2.0]);
    assert_eq!(casted.dtype(), DType::F32);
    assert_eq!(casted.as_slice::<f32>().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(reshaped.shape(), &[4]);
    assert_eq!(reshaped.as_slice::<f64>().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(transposed.shape(), &[2, 2]);
    assert_eq!(transposed.as_slice::<f64>().unwrap(), &[1.0, 3.0, 2.0, 4.0]);
    assert_eq!(summed.shape(), &[2]);
    assert_eq!(summed.as_slice::<f64>().unwrap(), &[3.0, 7.0]);
}

#[test]
fn concrete_tensor_matmul_rejects_non_matrix_inputs_without_rank_underflow() {
    let mut backend = CpuBackend::new();
    let scalar = Tensor::from_vec_col_major(vec![], vec![1.0_f64]).unwrap();
    let vector = Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap();

    let err = backend
        .with_backend_session(|session| scalar.matmul(&vector, session))
        .unwrap_err();

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
                lhs_contracting_dims: [1].as_slice().into(),
                rhs_contracting_dims: [0].as_slice().into(),
                lhs_batch_dims: [].as_slice().into(),
                rhs_batch_dims: [].as_slice().into(),
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
    assert!(values[0].is_view());
    assert_eq!(values[0].strides(), &[2, 1]);
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

/// Execution-path parity (Stage 0): the prepared and unprepared paths must agree
/// numerically on an elementwise chain, even though they submit a different
/// number of commands today.
#[test]
fn runtime_prepared_matches_compiled_for_elementwise_chain() {
    let runtime = cpu_runtime();
    let x = TracedTensor::input_concrete_shape(DType::F64, &[4]).unwrap();
    let doubled = (&x + &x).unwrap();
    let y = doubled
        .mul(&doubled)
        .unwrap()
        .exp()
        .unwrap()
        .tanh()
        .unwrap();
    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_with_input_specs(&y, &[(&x, DType::F64, &[4])])
        .unwrap();
    let input = Tensor::from_vec_col_major(vec![4], vec![0.5_f64, 1.0, 1.5, 2.0]).unwrap();

    let mut compiled = runtime.run_compiled(&program, &[&input]).unwrap();
    let prepared = runtime.prepare_compiled(&program, &[&input]).unwrap();
    let mut prepared_out = runtime.run_prepared(&prepared, &[&input]).unwrap();

    let compiled = compiled.pop().unwrap();
    let prepared_out = prepared_out.pop().unwrap();
    assert_eq!(compiled.shape(), prepared_out.shape());
    let compiled = compiled.as_slice::<f64>().unwrap();
    let prepared_out = prepared_out.as_slice::<f64>().unwrap();
    for (left, right) in compiled.iter().zip(prepared_out) {
        assert!(
            (left - right).abs() <= 1e-12,
            "prepared and unprepared paths diverged: {left} != {right}"
        );
    }
}

/// Plan census: a pure elementwise chain is one command, and a run containing a
/// reduction stays one command per instruction.
///
/// The scheduled executor is the only production executor, so this census is the
/// production command count for the prepared path; the legacy segmented executor
/// is test-only.
#[test]
fn prepared_execution_command_count_fuses_elementwise_chains() {
    let runtime = cpu_runtime();
    let n = 4usize;

    let x = TracedTensor::input_concrete_shape(DType::F64, &[n]).unwrap();
    let doubled = (&x + &x).unwrap();
    let chain = doubled
        .mul(&doubled)
        .unwrap()
        .exp()
        .unwrap()
        .tanh()
        .unwrap();
    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_with_input_specs(&chain, &[(&x, DType::F64, &[n])])
        .unwrap();
    let input = Tensor::from_vec_col_major(vec![n], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let prepared = runtime.prepare_compiled(&program, &[&input]).unwrap();
    assert_eq!(
        prepared.execution_command_count(),
        1,
        "the chain should be one fused command"
    );

    let x2 = TracedTensor::input_concrete_shape(DType::F64, &[n]).unwrap();
    let with_reduce = (&x2 + &x2)
        .unwrap()
        .exp()
        .unwrap()
        .reduce_sum(None)
        .unwrap();
    let mut compiler2 = GraphCompiler::new();
    let program2 = compiler2
        .compile_with_input_specs(&with_reduce, &[(&x2, DType::F64, &[n])])
        .unwrap();
    let input2 = Tensor::from_vec_col_major(vec![n], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let prepared2 = runtime.prepare_compiled(&program2, &[&input2]).unwrap();
    assert_eq!(
        prepared2.execution_command_count(),
        3,
        "a run containing a reduction is fusion-ineligible and keeps one command per instruction"
    );
}

/// Runtime evidence: the compiled and prepared entry points each submit a pure
/// elementwise chain once through the scheduled production executor.
#[test]
fn compiled_and_prepared_submission_counts_match() {
    let runtime = cpu_runtime();
    let n = 4usize;

    let x = TracedTensor::input_concrete_shape(DType::F64, &[n]).unwrap();
    let chain = (&x + &x)
        .unwrap()
        .mul(&x)
        .unwrap()
        .exp()
        .unwrap()
        .tanh()
        .unwrap();
    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_with_input_specs(&chain, &[(&x, DType::F64, &[n])])
        .unwrap();
    let input = Tensor::from_vec_col_major(vec![n], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let prepared = runtime.prepare_compiled(&program, &[&input]).unwrap();
    assert_eq!(prepared.execution_submission_count(), 0);

    let _ = runtime.run_compiled(&program, &[&input]).unwrap();
    assert_eq!(
        prepared.execution_submission_count(),
        1,
        "compiled execution should submit the region once"
    );

    let _ = runtime.run_prepared(&prepared, &[&input]).unwrap();
    assert_eq!(
        prepared.execution_submission_count(),
        2,
        "prepared execution should submit the same region once"
    );
}

/// Runtime parity matrix: shapes in the normal fused range submit once through
/// each production entry point.
#[test]
fn compiled_and_prepared_submission_counts_match_shape_matrix() {
    for n in [4usize, 64, 1024] {
        let runtime = cpu_runtime();
        let x = TracedTensor::input_concrete_shape(DType::F64, &[n]).unwrap();
        let chain = (&x + &x)
            .unwrap()
            .mul(&x)
            .unwrap()
            .exp()
            .unwrap()
            .tanh()
            .unwrap();
        let mut compiler = GraphCompiler::new();
        let program = compiler
            .compile_with_input_specs(&chain, &[(&x, DType::F64, &[n])])
            .unwrap();
        let input = Tensor::from_vec_col_major(vec![n], vec![1.0_f64; n]).unwrap();
        let prepared = runtime.prepare_compiled(&program, &[&input]).unwrap();
        let baseline = prepared.execution_submission_count();

        runtime.run_compiled(&program, &[&input]).unwrap();
        let after_compiled = prepared.execution_submission_count();
        runtime.run_prepared(&prepared, &[&input]).unwrap();
        let after_prepared = prepared.execution_submission_count();

        assert_eq!(
            after_compiled - baseline,
            1,
            "compiled path submission count for shape {n}"
        );
        assert_eq!(
            after_prepared - after_compiled,
            1,
            "prepared path submission count for shape {n}"
        );
    }
}

/// Stage 1 planning: the elementwise chain becomes one planned region covering
/// all four instructions, while a run containing a reduction plans none.
#[test]
fn prepared_elementwise_regions_are_planned_at_prepare_time() {
    let runtime = cpu_runtime();
    let n = 4usize;

    let x = TracedTensor::input_concrete_shape(DType::F64, &[n]).unwrap();
    let doubled = (&x + &x).unwrap();
    let chain = doubled
        .mul(&doubled)
        .unwrap()
        .exp()
        .unwrap()
        .tanh()
        .unwrap();
    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_with_input_specs(&chain, &[(&x, DType::F64, &[n])])
        .unwrap();
    let input = Tensor::from_vec_col_major(vec![n], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let prepared = runtime.prepare_compiled(&program, &[&input]).unwrap();
    assert_eq!(
        prepared.elementwise_region_summary(),
        (1, 4),
        "the pure elementwise chain should plan one region over four instructions"
    );

    let x2 = TracedTensor::input_concrete_shape(DType::F64, &[n]).unwrap();
    let with_reduce = (&x2 + &x2)
        .unwrap()
        .exp()
        .unwrap()
        .reduce_sum(None)
        .unwrap();
    let mut compiler2 = GraphCompiler::new();
    let program2 = compiler2
        .compile_with_input_specs(&with_reduce, &[(&x2, DType::F64, &[n])])
        .unwrap();
    let input2 = Tensor::from_vec_col_major(vec![n], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let prepared2 = runtime.prepare_compiled(&program2, &[&input2]).unwrap();
    assert_eq!(
        prepared2.elementwise_region_summary(),
        (0, 0),
        "a run containing a reduction is not an elementwise region"
    );
}

/// Stage 1 evidence: the planned region executes as one fused command (the
/// runtime counter records it), the results match the unprepared path, and a
/// run containing a reduction plans and executes no region.
#[test]
fn prepared_elementwise_region_executes_as_one_fused_command() {
    let runtime = cpu_runtime();
    // The CPU fused kernel only engages above its element-count floor, so use a
    // size that can actually fuse; the small-size fallback is covered by
    // prepared_elementwise_region_falls_back_when_fusion_is_declined.
    let n = 16 * 1024usize;

    let x = TracedTensor::input_concrete_shape(DType::F64, &[n]).unwrap();
    let doubled = (&x + &x).unwrap();
    let chain = doubled
        .mul(&doubled)
        .unwrap()
        .exp()
        .unwrap()
        .tanh()
        .unwrap();
    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_with_input_specs(&chain, &[(&x, DType::F64, &[n])])
        .unwrap();
    let data: Vec<f64> = (0..n).map(|i| 0.25 + (i % 17) as f64 * 0.125).collect();
    let input = Tensor::from_vec_col_major(vec![n], data).unwrap();
    let prepared = runtime.prepare_compiled(&program, &[&input]).unwrap();

    // `run_compiled` shares the runtime's prepared-entry cache, so the reference
    // run uses a second runtime to keep this runtime's counters meaningful.
    let reference_runtime = cpu_runtime();
    let mut compiled = reference_runtime.run_compiled(&program, &[&input]).unwrap();
    let mut fused = runtime.run_prepared(&prepared, &[&input]).unwrap();
    assert_eq!(
        prepared.elementwise_region_execution_counts(),
        (1, 0),
        "the chain region should execute fused, with no fallback"
    );

    let compiled = compiled.pop().unwrap();
    let fused = fused.pop().unwrap();
    let compiled = compiled.as_slice::<f64>().unwrap();
    let fused = fused.as_slice::<f64>().unwrap();
    for (left, right) in compiled.iter().zip(fused) {
        assert!(
            (left - right).abs() <= 1e-12,
            "fused region diverged from the unprepared path: {left} != {right}"
        );
    }

    let x2 = TracedTensor::input_concrete_shape(DType::F64, &[n]).unwrap();
    let with_reduce = (&x2 + &x2)
        .unwrap()
        .exp()
        .unwrap()
        .reduce_sum(None)
        .unwrap();
    let mut compiler2 = GraphCompiler::new();
    let program2 = compiler2
        .compile_with_input_specs(&with_reduce, &[(&x2, DType::F64, &[n])])
        .unwrap();
    let data2: Vec<f64> = (0..n).map(|i| 0.5 + (i % 11) as f64 * 0.25).collect();
    let input2 = Tensor::from_vec_col_major(vec![n], data2).unwrap();
    let prepared2 = runtime.prepare_compiled(&program2, &[&input2]).unwrap();
    let _ = runtime.run_prepared(&prepared2, &[&input2]).unwrap();
    assert_eq!(
        prepared2.elementwise_region_execution_counts(),
        (0, 0),
        "a run containing a reduction plans no region"
    );
}

/// Stage 1 evidence: when the backend declines the fusion the region falls back
/// to its instructions and still matches the unprepared path. A tiny element
/// count is below the CPU fused kernel's floor, so this exercises the fallback.
#[test]
fn prepared_elementwise_region_falls_back_when_fusion_is_declined() {
    let runtime = cpu_runtime();
    let n = 4usize;

    let x = TracedTensor::input_concrete_shape(DType::F64, &[n]).unwrap();
    let doubled = (&x + &x).unwrap();
    let chain = doubled
        .mul(&doubled)
        .unwrap()
        .exp()
        .unwrap()
        .tanh()
        .unwrap();
    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_with_input_specs(&chain, &[(&x, DType::F64, &[n])])
        .unwrap();
    let input = Tensor::from_vec_col_major(vec![n], vec![0.5_f64, 1.0, 1.5, 2.0]).unwrap();
    let prepared = runtime.prepare_compiled(&program, &[&input]).unwrap();

    let reference_runtime = cpu_runtime();
    let mut compiled = reference_runtime.run_compiled(&program, &[&input]).unwrap();
    let mut prepared_out = runtime.run_prepared(&prepared, &[&input]).unwrap();
    let (fused, fallbacks) = prepared.elementwise_region_execution_counts();
    assert_eq!(fused, 0, "the CPU backend declines fusion below its floor");
    assert_eq!(fallbacks, 1, "the region should fall back exactly once");

    let compiled = compiled.pop().unwrap();
    let prepared_out = prepared_out.pop().unwrap();
    let compiled = compiled.as_slice::<f64>().unwrap();
    let prepared_out = prepared_out.as_slice::<f64>().unwrap();
    for (left, right) in compiled.iter().zip(prepared_out) {
        assert!(
            (left - right).abs() <= 1e-12,
            "fallback diverged from the unprepared path: {left} != {right}"
        );
    }
}

/// Stage 1: a region with several live-outs publishes every one of them.
///
/// Both `y` and `z` are program outputs of one elementwise region.
#[test]
fn prepared_elementwise_region_publishes_multiple_live_outs() {
    let runtime = cpu_runtime();
    let n = 16 * 1024usize;

    let data: Vec<f64> = (0..n).map(|i| 0.25 + (i % 13) as f64 * 0.125).collect();
    let x = TracedTensor::from_tensor_concrete_shape(
        Tensor::from_vec_col_major(vec![n], data).unwrap(),
    )
    .unwrap();
    let shared = (&x + &x).unwrap();
    let y = shared.exp().unwrap().tanh().unwrap();
    let z = (&shared * &shared).unwrap();
    let program = GraphCompiler::new()
        .compile_many(&[&y, &z])
        .expect("two-output constant graph should compile");

    let prepared = runtime.prepare_compiled(&program, &[]).unwrap();
    let (regions, covered) = prepared.elementwise_region_summary();
    assert_eq!((regions, covered), (1, 4), "one region covers the chain");

    let mut prepared_out = runtime.run_prepared(&prepared, &[]).unwrap();
    let (fused, fallbacks) = prepared.elementwise_region_execution_counts();
    assert_eq!((fused, fallbacks), (1, 0), "the region should fuse once");

    let reference_runtime = cpu_runtime();
    let mut compiled = reference_runtime.run_compiled(&program, &[]).unwrap();
    assert_eq!(prepared_out.len(), 2);
    assert_eq!(compiled.len(), 2);
    while let Some(expected) = compiled.pop() {
        let actual = prepared_out.pop().unwrap();
        assert_eq!(actual.shape(), expected.shape());
        let actual = actual.as_slice::<f64>().unwrap();
        let expected = expected.as_slice::<f64>().unwrap();
        for (left, right) in actual.iter().zip(expected) {
            assert!(
                (left - right).abs() <= 1e-12,
                "live-out diverged: {left} != {right}"
            );
        }
    }
}

/// Stage 1: the value output mode keeps its per-instruction path and results.
#[test]
fn runtime_compiled_values_matches_prepared_for_elementwise_chain() {
    let runtime = cpu_runtime();
    let n = 1024usize;

    let x = TracedTensor::input_concrete_shape(DType::F64, &[n]).unwrap();
    let doubled = (&x + &x).unwrap();
    let chain = doubled
        .mul(&doubled)
        .unwrap()
        .exp()
        .unwrap()
        .tanh()
        .unwrap();
    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_with_input_specs(&chain, &[(&x, DType::F64, &[n])])
        .unwrap();
    let data: Vec<f64> = (0..n).map(|i| 0.5 + (i % 7) as f64 * 0.25).collect();
    let input = Tensor::from_vec_col_major(vec![n], data).unwrap();

    let prepared = runtime.prepare_compiled(&program, &[&input]).unwrap();
    let mut prepared_out = runtime.run_prepared(&prepared, &[&input]).unwrap();
    let mut value_out = runtime.run_compiled_values(&program, &[&input]).unwrap();
    assert_eq!(value_out.len(), 1);
    let value = value_out.pop().unwrap();
    let value = value.as_tensor().expect("value output should own a tensor");
    let prepared_out = prepared_out.pop().unwrap();
    assert_eq!(value.shape(), prepared_out.shape());
    let value = value.as_slice::<f64>().unwrap();
    let prepared_out = prepared_out.as_slice::<f64>().unwrap();
    for (left, right) in value.iter().zip(prepared_out) {
        assert!(
            (left - right).abs() <= 1e-12,
            "value mode diverged: {left} != {right}"
        );
    }
}

/// Stage 1: an elementwise region next to an FFI operation.
///
/// The region covers only the elementwise chain; the matrix multiply stays its
/// own command, and both outputs match the unprepared path.
#[test]
fn prepared_elementwise_region_stays_separate_from_ffi_op() {
    let runtime = cpu_runtime();
    // The CPU fused kernel only engages above its element floor, so the chain
    // runs over a vector large enough to fuse while the multiply stays small.
    let n = 8usize;
    let chain_len = 16 * 1024usize;

    let matrix = |seed: f64| {
        Tensor::from_vec_col_major(
            vec![n, n],
            (0..n * n).map(|i| seed + (i % 7) as f64 * 0.125).collect(),
        )
        .unwrap()
    };
    let a = TracedTensor::from_tensor_concrete_shape(matrix(0.5)).unwrap();
    let b = TracedTensor::from_tensor_concrete_shape(matrix(0.25)).unwrap();
    let x = TracedTensor::from_tensor_concrete_shape(
        Tensor::from_vec_col_major(
            vec![chain_len],
            (0..chain_len)
                .map(|i| 0.5 + (i % 5) as f64 * 0.25)
                .collect(),
        )
        .unwrap(),
    )
    .unwrap();

    let m = a.matmul(&b).unwrap();
    let chain = (&x * &x).unwrap().exp().unwrap().tanh().unwrap();
    let program = GraphCompiler::new()
        .compile_many(&[&m, &chain])
        .expect("mixed graph should compile");

    let prepared = runtime.prepare_compiled(&program, &[]).unwrap();
    let (regions, covered) = prepared.elementwise_region_summary();
    assert_eq!(
        (regions, covered),
        (1, 3),
        "only the elementwise chain should form a region"
    );

    let mut prepared_out = runtime.run_prepared(&prepared, &[]).unwrap();
    let (fused, fallbacks) = prepared.elementwise_region_execution_counts();
    assert_eq!((fused, fallbacks), (1, 0), "the chain region should fuse");

    let reference_runtime = cpu_runtime();
    let mut compiled = reference_runtime.run_compiled(&program, &[]).unwrap();
    assert_eq!(prepared_out.len(), 2);
    assert_eq!(compiled.len(), 2);
    while let Some(expected) = compiled.pop() {
        let actual = prepared_out.pop().unwrap();
        assert_eq!(actual.shape(), expected.shape());
        let actual = actual.as_slice::<f64>().unwrap();
        let expected = expected.as_slice::<f64>().unwrap();
        for (left, right) in actual.iter().zip(expected) {
            assert!(
                (left - right).abs() <= 1e-12,
                "mixed-graph output diverged: {left} != {right}"
            );
        }
    }
}

/// Stage 1: repeated prepared runs neither fall back nor accumulate state.
#[test]
fn prepared_elementwise_region_is_stable_across_repeated_runs() {
    let runtime = cpu_runtime();
    let n = 16 * 1024usize;

    let x = TracedTensor::input_concrete_shape(DType::F64, &[n]).unwrap();
    let doubled = (&x + &x).unwrap();
    let chain = doubled
        .mul(&doubled)
        .unwrap()
        .exp()
        .unwrap()
        .tanh()
        .unwrap();
    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_with_input_specs(&chain, &[(&x, DType::F64, &[n])])
        .unwrap();
    let data: Vec<f64> = (0..n).map(|i| 0.25 + (i % 17) as f64 * 0.125).collect();
    let input = Tensor::from_vec_col_major(vec![n], data).unwrap();
    let prepared = runtime.prepare_compiled(&program, &[&input]).unwrap();

    let mut first: Option<Vec<f64>> = None;
    for run in 1..=16 {
        let mut out = runtime.run_prepared(&prepared, &[&input]).unwrap();
        let out = out.pop().unwrap();
        let values = out.as_slice::<f64>().unwrap().to_vec();
        match &first {
            None => first = Some(values),
            Some(expected) => {
                for (left, right) in values.iter().zip(expected) {
                    assert!(
                        (left - right).abs() <= 1e-15,
                        "run {run} diverged from the first run: {left} != {right}"
                    );
                }
            }
        }
        let (fused, fallbacks) = prepared.elementwise_region_execution_counts();
        assert_eq!((fused, fallbacks), (run, 0), "run {run} should fuse once");
    }
}

/// Stage 1: a chain, an FFI operation, and another chain in one program.
///
/// The large chain fuses, the small chain after the matrix multiply falls back
/// below the CPU fused kernel's element floor, and both outputs match the
/// unprepared path, so interleaving regions with FFI work keeps its contract.
#[test]
fn prepared_regions_interleave_with_ffi_work() {
    let runtime = cpu_runtime();
    let chain_len = 16 * 1024usize;
    let n = 8usize;

    let vector = |len: usize, seed: f64| {
        Tensor::from_vec_col_major(
            vec![len],
            (0..len).map(|i| seed + (i % 11) as f64 * 0.125).collect(),
        )
        .unwrap()
    };
    let matrix = |seed: f64| {
        Tensor::from_vec_col_major(
            vec![n, n],
            (0..n * n).map(|i| seed + (i % 7) as f64 * 0.125).collect(),
        )
        .unwrap()
    };

    let x = TracedTensor::from_tensor_concrete_shape(vector(chain_len, 0.5)).unwrap();
    let a = TracedTensor::from_tensor_concrete_shape(matrix(0.5)).unwrap();
    let b = TracedTensor::from_tensor_concrete_shape(matrix(0.25)).unwrap();

    let before = (&x * &x).unwrap().exp().unwrap().tanh().unwrap();
    let product = a.matmul(&b).unwrap();
    let after = (&product * &product)
        .unwrap()
        .exp()
        .unwrap()
        .tanh()
        .unwrap();
    let program = GraphCompiler::new()
        .compile_many(&[&before, &after])
        .expect("interleaved graph should compile");

    let prepared = runtime.prepare_compiled(&program, &[]).unwrap();
    let (regions, _) = prepared.elementwise_region_summary();
    assert!(regions >= 1, "the chains should plan at least one region");

    let mut prepared_out = runtime.run_prepared(&prepared, &[]).unwrap();
    let (fused, fallbacks) = prepared.elementwise_region_execution_counts();
    assert!(fused >= 1, "the large chain should fuse: {fused} fused");
    assert!(
        fallbacks >= 1,
        "the chain below the element floor should fall back: {fallbacks} fallbacks"
    );

    let reference_runtime = cpu_runtime();
    let mut compiled = reference_runtime.run_compiled(&program, &[]).unwrap();
    assert_eq!(prepared_out.len(), 2);
    assert_eq!(compiled.len(), 2);
    while let Some(expected) = compiled.pop() {
        let actual = prepared_out.pop().unwrap();
        assert_eq!(actual.shape(), expected.shape());
        let actual = actual.as_slice::<f64>().unwrap();
        let expected = expected.as_slice::<f64>().unwrap();
        for (left, right) in actual.iter().zip(expected) {
            assert!(
                (left - right).abs() <= 1e-12,
                "interleaved output diverged: {left} != {right}"
            );
        }
    }
}

/// Stage 1: the fallback path stays stable across repeated runs.
#[test]
fn prepared_elementwise_fallback_is_stable_across_repeated_runs() {
    let runtime = cpu_runtime();
    let n = 4usize;

    let x = TracedTensor::input_concrete_shape(DType::F64, &[n]).unwrap();
    let doubled = (&x + &x).unwrap();
    let chain = doubled
        .mul(&doubled)
        .unwrap()
        .exp()
        .unwrap()
        .tanh()
        .unwrap();
    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_with_input_specs(&chain, &[(&x, DType::F64, &[n])])
        .unwrap();
    let input = Tensor::from_vec_col_major(vec![n], vec![0.5_f64, 1.0, 1.5, 2.0]).unwrap();
    let prepared = runtime.prepare_compiled(&program, &[&input]).unwrap();

    let mut first: Option<Vec<f64>> = None;
    for run in 1..=8 {
        let mut out = runtime.run_prepared(&prepared, &[&input]).unwrap();
        let out = out.pop().unwrap();
        let values = out.as_slice::<f64>().unwrap().to_vec();
        match &first {
            None => first = Some(values),
            Some(expected) => {
                for (left, right) in values.iter().zip(expected) {
                    assert!(
                        (left - right).abs() <= 1e-15,
                        "fallback run {run} diverged: {left} != {right}"
                    );
                }
            }
        }
        let (fused, fallbacks) = prepared.elementwise_region_execution_counts();
        assert_eq!(
            (fused, fallbacks),
            (0, run),
            "run {run} should fall back once"
        );
    }
}
