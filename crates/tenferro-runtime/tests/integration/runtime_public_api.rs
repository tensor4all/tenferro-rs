use tenferro_cpu::CpuEngine;
use tenferro_ops::dim_expr::DimExpr;
use tenferro_runtime::{
    CompiledGraph, DType, DotGeneralConfig, Error, ErrorPhase, ExecutionEngine, GatherConfig,
    GraphCompiler, PadConfig, Prepared, Runtime, ScatterConfig, SliceConfig, Tensor,
    TensorSessionOpsExt, TracedTensor,
};
use tenferro_tensor::{Error as TensorError, ShapeMismatch, TensorRead, ValidationError};

fn cpu_runtime(engine: &CpuEngine) -> Runtime {
    let mut builder = Runtime::builder();
    builder
        .register_engine(tenferro_cpu::runtime_engine_registration(engine).unwrap())
        .unwrap();
    builder.build().unwrap()
}

/// A serial CPU engine with a runtime that registers it.
fn cpu_engine_and_runtime() -> (CpuEngine, Runtime) {
    let engine = CpuEngine::serial();
    let runtime = cpu_runtime(&engine);
    (engine, runtime)
}

fn inputs_of<'a>(inputs: &[&'a Tensor]) -> Vec<TensorRead<'a>> {
    inputs
        .iter()
        .map(|input| TensorRead::from_tensor(input))
        .collect()
}

fn run_compiled(
    engine: &mut CpuEngine,
    runtime: &Runtime,
    program: &CompiledGraph,
    inputs: &[&Tensor],
) -> tenferro_runtime::Result<Vec<Tensor>> {
    let reads = inputs_of(inputs);
    let prepared = runtime.prepare(program, &reads, &[engine.target()])?;
    let outputs = runtime.run(&mut [engine], &prepared, &reads)?;
    let mut owned = Vec::with_capacity(outputs.len());
    for output in outputs {
        owned.push(output.into_tensor(engine).unwrap());
    }
    Ok(owned)
}

fn prepare(
    engine: &CpuEngine,
    runtime: &Runtime,
    program: &CompiledGraph,
    inputs: &[&Tensor],
) -> tenferro_runtime::Result<Prepared> {
    let reads = inputs_of(inputs);
    runtime.prepare(program, &reads, &[engine.target()])
}

fn run_prepared(
    engine: &mut CpuEngine,
    runtime: &Runtime,
    prepared: &Prepared,
    inputs: &[&Tensor],
) -> tenferro_runtime::Result<Vec<Tensor>> {
    let reads = inputs_of(inputs);
    let outputs = {
        let mut engines: [&mut dyn ExecutionEngine; 1] = [engine];
        runtime.run(&mut engines, prepared, &reads)?
    };
    let mut owned = Vec::with_capacity(outputs.len());
    for output in outputs {
        owned.push(output.into_tensor(engine).unwrap());
    }
    Ok(owned)
}

fn run_compiled_one(program: &CompiledGraph, inputs: &[&Tensor]) -> Tensor {
    let (mut engine, runtime) = cpu_engine_and_runtime();
    let mut outputs = run_compiled(&mut engine, &runtime, program, inputs).unwrap();
    assert_eq!(outputs.len(), 1);
    outputs.pop().unwrap()
}

#[test]
fn runtime_prepared_compiled_graph_runs_repeated_inputs() {
    let mut engine = CpuEngine::serial();
    let runtime = cpu_runtime(&engine);
    let x = TracedTensor::input_concrete_shape(DType::F64, &[2]).unwrap();
    let y = (&x + &x).unwrap();
    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_with_input_specs(&y, &[(&x, DType::F64, &[2])])
        .unwrap();
    let first = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let second = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).unwrap();

    let prepared = prepare(&engine, &runtime, &program, &[&first]).unwrap();

    let mut outputs = run_prepared(&mut engine, &runtime, &prepared, &[&first]).unwrap();
    assert_eq!(
        outputs.pop().unwrap().as_slice::<f64>().unwrap(),
        &[2.0, 4.0]
    );
    let mut outputs = run_prepared(&mut engine, &runtime, &prepared, &[&second]).unwrap();
    assert_eq!(
        outputs.pop().unwrap().as_slice::<f64>().unwrap(),
        &[6.0, 8.0]
    );

    // Another runtime cannot run a handle prepared by this one.
    let other_runtime = cpu_runtime(&engine);
    assert!(run_prepared(&mut engine, &other_runtime, &prepared, &[&first]).is_err());
}

#[test]
fn one_prepared_program_runs_concurrently_on_two_engines() {
    let (engine, runtime) = cpu_engine_and_runtime();
    let x = TracedTensor::input_concrete_shape(DType::F64, &[4]).unwrap();
    let y = (&x + &x).unwrap();
    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_with_input_specs(&y, &[(&x, DType::F64, &[4])])
        .unwrap();
    let input = Tensor::from_vec_col_major(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let prepared = prepare(&engine, &runtime, &program, &[&input]).unwrap();

    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..2)
            .map(|_| {
                let runtime = &runtime;
                let prepared = &prepared;
                let input = &input;
                scope.spawn(move || {
                    let mut engine = CpuEngine::serial();
                    run_prepared(&mut engine, runtime, prepared, &[input])
                        .unwrap()
                        .into_iter()
                        .map(|output| output.as_slice::<f64>().unwrap().to_vec())
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        for handle in handles {
            let outputs = handle.join().expect("concurrent prepared run");
            assert_eq!(outputs, vec![vec![2.0, 4.0, 6.0, 8.0]]);
        }
    });
}

#[test]
fn prepared_execution_checks_retained_defaults_preparation_did_not_validate() {
    use tenferro_runtime::program::{ProgramInputSpec, SemanticProgramBuilder};

    // Both inputs declare `[n]` with `n` = input 1's axis 0, so the pair must
    // agree: the retained defaults below are 2 and 3, which do not.
    let shared_extent = || DimExpr::InputDim {
        input_idx: 1,
        axis: 0,
    };
    let mut builder = SemanticProgramBuilder::new();
    let first = builder
        .input(ProgramInputSpec::new(DType::F64, [shared_extent()]))
        .unwrap();
    let second = builder
        .input(ProgramInputSpec::new(DType::F64, [shared_extent()]))
        .unwrap();
    // Binding checks constant extents only, so an inconsistent default pair is
    // accepted here; the mismatch surfaces when a run resolves the defaults.
    builder
        .bind_input(
            first,
            Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap(),
        )
        .unwrap();
    builder
        .bind_input(
            second,
            Tensor::from_vec_col_major(vec![3], vec![3.0_f64, 4.0, 5.0]).unwrap(),
        )
        .unwrap();
    let frozen = builder.finish(&[first, second]).unwrap();
    let program = GraphCompiler::new()
        .compile_frozen_program(&frozen)
        .unwrap();

    let mut engine = CpuEngine::serial();
    let runtime = cpu_runtime(&engine);
    // Preparation replaces every input, so it validates the replacements and
    // never the retained defaults.
    let replacement_first = Tensor::from_vec_col_major(vec![3], vec![7.0_f64; 3]).unwrap();
    let replacement_second = Tensor::from_vec_col_major(vec![3], vec![8.0_f64; 3]).unwrap();
    let prepared = prepare(
        &engine,
        &runtime,
        &program,
        &[&replacement_first, &replacement_second],
    )
    .unwrap();

    let outputs = run_prepared(
        &mut engine,
        &runtime,
        &prepared,
        &[&replacement_first, &replacement_second],
    )
    .expect("the validated replacements satisfy the declared shape relation");
    assert_eq!(outputs.len(), 2);

    // Resolving the retained defaults must still apply the metadata check that
    // preparation never ran on them.
    let error = run_prepared(&mut engine, &runtime, &prepared, &[])
        .expect_err("the retained defaults violate the declared shape relation");
    assert!(
        matches!(error, Error::PlaceholderShapeMismatch { .. }),
        "got {error:?}"
    );
}

#[test]
fn runtime_prepared_execution_hot_path_keeps_input_metadata_inline() {
    let source = include_str!("../../src/runtime/execution.rs");

    assert!(
        source.contains("type RuntimeInputReads<'a> = SmallVec"),
        "Runtime::run_prepared should keep short input reference lists inline"
    );
    // `TensorRead` has no `Default`, so `tinyvec::Array` cannot hold the input
    // read list; the shape list has no such constraint and moved to `TinyVec`.
    assert!(
        source.contains("type RuntimeInputShapes<'a> = TinyVec"),
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
    let mut engine = CpuEngine::serial();
    let input = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let f32_input = Tensor::from_vec_col_major(vec![2], vec![1.0_f32, 2.0]).unwrap();

    let converted = f32_input.convert(DType::F64, &mut engine).unwrap();
    let casted = input.cast(DType::F32, &mut engine).unwrap();
    let reshaped = input.reshape(&[4], &mut engine).unwrap();
    let transposed = input.transpose(&[1, 0], &mut engine).unwrap();
    let summed = input.reduce_sum(Some(&[0]), &mut engine).unwrap();
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
    let mut engine = CpuEngine::serial();
    let scalar = Tensor::from_vec_col_major(vec![], vec![1.0_f64]).unwrap();
    let vector = Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap();

    let err = scalar.matmul(&vector, &mut engine).unwrap_err();

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
    let mut engine = CpuEngine::serial();
    let runtime = cpu_runtime(&engine);
    let x = TracedTensor::input_symbolic_shape(DType::F64, 1).unwrap();
    let y = (&x + &x).unwrap().reduce_sum(Some(&[0])).unwrap();
    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_with_input_specs(&y, &[(&x, DType::F64, &[2])])
        .unwrap();
    let input = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();

    let out = run_compiled(&mut engine, &runtime, &program, &[&input]).unwrap();

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
    let mut engine = CpuEngine::serial();
    let runtime = cpu_runtime(&engine);
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

    let out = run_compiled(&mut engine, &runtime, &program, &[&lhs_data, &rhs_data]).unwrap();

    assert_eq!(out.len(), 1);
    assert_eq!(out[0].shape(), &[2, 2]);
    assert_eq!(out[0].as_slice::<f64>().unwrap(), &[22.0, 28.0, 49.0, 64.0]);
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
    let mut engine = CpuEngine::serial();
    let runtime = cpu_runtime(&engine);

    let out = run_compiled_one(&program, &[&input]);
    assert_eq!(out.as_slice::<f64>().unwrap(), &[2.0, 4.0]);

    let unbound = run_compiled(&mut engine, &runtime, &program, &[]).unwrap_err();
    assert!(matches!(unbound, Error::UnboundPlaceholder { .. }));

    let extra = run_compiled(&mut engine, &runtime, &program, &[&input, &input]).unwrap_err();
    assert!(matches!(extra, Error::GraphInputCountMismatch { .. }));

    let f32_input = Tensor::from_vec_col_major(vec![2], vec![1.0_f32, 2.0]).unwrap();
    let dtype = run_compiled(&mut engine, &runtime, &program, &[&f32_input]).unwrap_err();
    assert!(matches!(dtype, Error::PlaceholderDtypeMismatch { .. }));

    let rank_input = Tensor::from_vec_col_major(vec![1, 2], vec![1.0_f64, 2.0]).unwrap();
    let rank = run_compiled(&mut engine, &runtime, &program, &[&rank_input]).unwrap_err();
    assert!(matches!(rank, Error::PlaceholderRankMismatch { .. }));

    let shape_input = Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]).unwrap();
    let shape = run_compiled(&mut engine, &runtime, &program, &[&shape_input]).unwrap_err();
    assert!(matches!(shape, Error::PlaceholderShapeMismatch { .. }));
}

/// Negative companion to
/// `traced::tests::concatenate_of_distinct_inputs_compiles_without_input_specs`:
/// distinct rank-2 symbolic inputs still compile without concrete input specs,
/// and a concrete non-axis mismatch is rejected on the public compiled path.
#[test]
fn runtime_concatenate_rejects_incompatible_concrete_shapes_without_input_specs() {
    let mut engine = CpuEngine::serial();
    let runtime = cpu_runtime(&engine);
    // Data-attached but shape-symbolic inputs: the compiled program keeps their
    // extents free, so `run_compiled` reaches the concatenate kernel.
    let a = TracedTensor::from_tensor_symbolic_shape(
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap(),
    )
    .unwrap();
    let b = TracedTensor::from_tensor_symbolic_shape(
        Tensor::from_vec_col_major(vec![2, 4], vec![2.0_f64; 8]).unwrap(),
    )
    .unwrap();

    let joined = TracedTensor::concatenate(&[&a, &b], 1).unwrap();
    let program = GraphCompiler::new().compile(&joined).unwrap();
    assert_eq!(program.input_count(), 2);

    // Incompatible concrete non-axis extents ([2, 3] axis 0 versus [4, 4] axis 0),
    // supplied in compiled input-key order.
    let a_value = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap();
    let b_value = Tensor::from_vec_col_major(vec![4, 4], vec![2.0_f64; 16]).unwrap();
    let mut ordered: [Option<&Tensor>; 2] = [None, None];
    ordered[program.input_key_index(&a.input_key().unwrap()).unwrap()] = Some(&a_value);
    ordered[program.input_key_index(&b.input_key().unwrap()).unwrap()] = Some(&b_value);

    let err = run_compiled(
        &mut engine,
        &runtime,
        &program,
        &ordered.map(Option::unwrap),
    )
    .unwrap_err();

    assert!(
        matches!(
            &err,
            Error::TensorRuntime(TensorError::Validation {
                op: "concatenate",
                source: ValidationError::ShapeMismatch(source),
                ..
            }) if matches!(
                source.as_ref(),
                ShapeMismatch::IncompatibleShapes { lhs, rhs }
                    if lhs.as_slice() == [2, 3] && rhs.as_slice() == [4, 4]
            )
        ),
        "expected a concrete concatenate shape mismatch, got: {err}"
    );
}

/// Execution-path parity (Stage 0): the prepared and unprepared paths must agree
/// numerically on an elementwise chain, even though they submit a different
/// number of commands today.
#[test]
fn runtime_prepared_matches_compiled_for_elementwise_chain() {
    let mut engine = CpuEngine::serial();
    let runtime = cpu_runtime(&engine);
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

    let mut compiled = run_compiled(&mut engine, &runtime, &program, &[&input]).unwrap();
    let prepared = prepare(&engine, &runtime, &program, &[&input]).unwrap();
    let mut prepared_out = run_prepared(&mut engine, &runtime, &prepared, &[&input]).unwrap();

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
    let engine = CpuEngine::serial();
    let runtime = cpu_runtime(&engine);
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
    let prepared = prepare(&engine, &runtime, &program, &[&input]).unwrap();
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
    let prepared2 = prepare(&engine, &runtime, &program2, &[&input2]).unwrap();
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
    let mut engine = CpuEngine::serial();
    let runtime = cpu_runtime(&engine);
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
    let prepared = prepare(&engine, &runtime, &program, &[&input]).unwrap();
    assert_eq!(prepared.execution_submission_count(), 0);

    let _ = run_compiled(&mut engine, &runtime, &program, &[&input]).unwrap();
    assert_eq!(
        prepared.execution_submission_count(),
        1,
        "compiled execution should submit the region once"
    );

    let _ = run_prepared(&mut engine, &runtime, &prepared, &[&input]).unwrap();
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
        let mut engine = CpuEngine::serial();
        let runtime = cpu_runtime(&engine);
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
        let prepared = prepare(&engine, &runtime, &program, &[&input]).unwrap();
        let baseline = prepared.execution_submission_count();

        run_compiled(&mut engine, &runtime, &program, &[&input]).unwrap();
        let after_compiled = prepared.execution_submission_count();
        run_prepared(&mut engine, &runtime, &prepared, &[&input]).unwrap();
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
    let engine = CpuEngine::serial();
    let runtime = cpu_runtime(&engine);
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
    let prepared = prepare(&engine, &runtime, &program, &[&input]).unwrap();
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
    let prepared2 = prepare(&engine, &runtime, &program2, &[&input2]).unwrap();
    assert_eq!(
        prepared2.elementwise_region_summary(),
        (0, 0),
        "a run containing a reduction is not an elementwise region"
    );
}

/// Stage 1 evidence: the planned region executes as one fused command (the
/// runtime counter records it), the results match a host reference, and a run
/// containing a reduction plans and executes no region at all.
#[test]
fn prepared_elementwise_region_executes_as_one_fused_command() {
    let mut engine = CpuEngine::serial();
    let runtime = cpu_runtime(&engine);
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
    let input = Tensor::from_vec_col_major(vec![n], data.clone()).unwrap();
    let prepared = prepare(&engine, &runtime, &program, &[&input]).unwrap();

    let mut fused = run_prepared(&mut engine, &runtime, &prepared, &[&input]).unwrap();
    assert_eq!(
        prepared.elementwise_region_execution_counts(),
        (1, 0),
        "the CPU engine should execute the chain region fused"
    );

    let fused = fused.pop().unwrap();
    let fused = fused.as_slice::<f64>().unwrap();
    for (index, value) in fused.iter().enumerate() {
        let doubled = data[index] + data[index];
        let expected = (doubled * doubled).exp().tanh();
        assert!(
            (value - expected).abs() <= 1e-12,
            "fused region diverged from the host reference at {index}: {value} != {expected}"
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
    let prepared2 = prepare(&engine, &runtime, &program2, &[&input2]).unwrap();
    let _ = run_prepared(&mut engine, &runtime, &prepared2, &[&input2]).unwrap();
    assert_eq!(
        prepared2.elementwise_region_execution_counts(),
        (0, 0),
        "a run containing a reduction plans no region"
    );
}

/// An ordered op (`maximum`/`minimum`/`clamp`) above the floor fuses and keeps
/// tenferro's NaN propagation.
///
/// `strided-fused 0.6.0` propagates NaN through its ordered kernels with the same
/// `strided_basic` select the unfused kernels use (TF5, strided-rs#315), so the
/// region executes through the fused route and must still return NaN for a NaN
/// operand (`R-numerics-01`).
#[test]
fn prepared_ordered_region_fuses_and_propagates_nan() {
    for dtype in [DType::F32, DType::F64] {
        let mut engine = CpuEngine::serial();
        let runtime = cpu_runtime(&engine);
        let n = 16 * 1024usize;

        let x = TracedTensor::input_concrete_shape(dtype, &[n]).unwrap();
        let y = TracedTensor::input_concrete_shape(dtype, &[n]).unwrap();
        let z = TracedTensor::input_concrete_shape(dtype, &[n]).unwrap();
        let out = (&x.maximum(&y).unwrap() + &z).unwrap();
        let program = GraphCompiler::new()
            .compile_with_input_specs(
                &out,
                &[(&x, dtype, &[n]), (&y, dtype, &[n]), (&z, dtype, &[n])],
            )
            .unwrap();

        let nan_input = match dtype {
            DType::F32 => Tensor::from_vec_col_major(vec![n], vec![f32::NAN; n]).unwrap(),
            DType::F64 => Tensor::from_vec_col_major(vec![n], vec![f64::NAN; n]).unwrap(),
            _ => unreachable!(),
        };
        let finite_input = match dtype {
            DType::F32 => Tensor::from_vec_col_major(vec![n], vec![1.0_f32; n]).unwrap(),
            DType::F64 => Tensor::from_vec_col_major(vec![n], vec![1.0_f64; n]).unwrap(),
            _ => unreachable!(),
        };
        let one = match dtype {
            DType::F32 => Tensor::from_vec_col_major(vec![n], vec![1.0_f32; n]).unwrap(),
            DType::F64 => Tensor::from_vec_col_major(vec![n], vec![1.0_f64; n]).unwrap(),
            _ => unreachable!(),
        };

        let prepared = prepare(
            &engine,
            &runtime,
            &program,
            &[&nan_input, &finite_input, &one],
        )
        .unwrap();
        assert!(
            prepared.elementwise_region_summary().0 >= 1,
            "{dtype:?}: the ordered chain should still plan a region"
        );

        let mut out = run_prepared(
            &mut engine,
            &runtime,
            &prepared,
            &[&nan_input, &finite_input, &one],
        )
        .unwrap();
        assert_eq!(
            prepared.elementwise_region_execution_counts(),
            (1, 0),
            "{dtype:?}: the NaN-propagating ordered region must run through the fused route"
        );
        let out = out.pop().unwrap();
        match dtype {
            DType::F32 => assert!(
                out.as_slice::<f32>()
                    .unwrap()
                    .iter()
                    .all(|value| value.is_nan()),
                "f32 NaN must propagate through the fused route"
            ),
            DType::F64 => assert!(
                out.as_slice::<f64>()
                    .unwrap()
                    .iter()
                    .all(|value| value.is_nan()),
                "f64 NaN must propagate through the fused route"
            ),
            _ => unreachable!(),
        }
    }
}

/// A region whose dtype is not F32/F64 stays unfused above the floor.
///
/// `strided-fused 0.6.0`'s vectorized chunk replay covers only F32/F64; every
/// other dtype would run the unmeasured per-element interpreter, so the region
/// is declined and the per-instruction path is compared against a host
/// reference.
#[test]
fn prepared_non_f32_f64_region_is_declined() {
    let mut engine = CpuEngine::serial();
    let runtime = cpu_runtime(&engine);
    let n = 16 * 1024usize;

    let x = TracedTensor::input_concrete_shape(DType::I64, &[n]).unwrap();
    let doubled = (&x + &x).unwrap();
    let chain = (&doubled * &doubled).unwrap();
    let program = GraphCompiler::new()
        .compile_with_input_specs(&chain, &[(&x, DType::I64, &[n])])
        .unwrap();
    let data: Vec<i64> = (0..n).map(|i| (i % 100) as i64 + 1).collect();
    let input = Tensor::from_vec_col_major(vec![n], data.clone()).unwrap();
    let prepared = prepare(&engine, &runtime, &program, &[&input]).unwrap();
    assert!(
        prepared.elementwise_region_summary().0 >= 1,
        "the I64 chain should still plan a region"
    );

    let mut out = run_prepared(&mut engine, &runtime, &prepared, &[&input]).unwrap();
    assert_eq!(
        prepared.elementwise_region_execution_counts(),
        (0, 1),
        "I64 has no measured vectorized replay and must run per instruction"
    );
    let out = out.pop().unwrap();
    let out = out.as_slice::<i64>().unwrap();
    for (index, &value) in out.iter().enumerate() {
        let doubled = data[index] + data[index];
        assert_eq!(
            value,
            doubled * doubled,
            "I64 fallback diverged from the host reference at {index}"
        );
    }
}

/// Stage 1 evidence: when the backend declines the fusion the region falls back
/// to its instructions and still matches the unprepared path. A tiny element
/// count is below the CPU fused kernel's floor, so this exercises the fallback.
#[test]
fn prepared_elementwise_region_falls_back_when_fusion_is_declined() {
    let mut engine = CpuEngine::serial();
    let runtime = cpu_runtime(&engine);
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
    let prepared = prepare(&engine, &runtime, &program, &[&input]).unwrap();

    let mut compiled = run_compiled(&mut engine, &runtime, &program, &[&input]).unwrap();
    let mut prepared_out = run_prepared(&mut engine, &runtime, &prepared, &[&input]).unwrap();
    // The two entry points share the prepared-plan cache, so the counters on the
    // prepared handle count both runs: one fallback each.
    let (fused, fallbacks) = prepared.elementwise_region_execution_counts();
    assert_eq!(fused, 0, "the CPU backend declines fusion below its floor");
    assert_eq!(
        fallbacks, 2,
        "each run's region should fall back exactly once"
    );

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

/// Regression for #1968: the CPU engine reports the fused kernel's input
/// limit, so a region reading more inputs is not planned at all. It runs
/// instruction by instruction, without a fusion attempt the backend would
/// decline on every execution, and still matches the unprepared path.
#[test]
fn prepared_elementwise_region_above_cpu_input_limit_is_not_planned() {
    let mut engine = CpuEngine::serial();
    let runtime = cpu_runtime(&engine);
    let n = 16 * 1024usize;
    let inputs = (0..5)
        .map(|_| TracedTensor::input_concrete_shape(DType::F64, &[n]).unwrap())
        .collect::<Vec<_>>();
    let mut sum = (&inputs[0] + &inputs[1]).unwrap();
    for input in &inputs[2..] {
        sum = (&sum + input).unwrap();
    }
    let shape = [n];
    let specs = inputs
        .iter()
        .map(|input| (input, DType::F64, &shape[..]))
        .collect::<Vec<_>>();
    let program = GraphCompiler::new()
        .compile_with_input_specs(&sum, &specs)
        .unwrap();
    let data = (0..5)
        .map(|k| {
            Tensor::from_vec_col_major(
                vec![n],
                (0..n).map(|i| (k + 1) as f64 + (i % 7) as f64).collect(),
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let refs = data.iter().collect::<Vec<_>>();
    let prepared = prepare(&engine, &runtime, &program, &refs).unwrap();
    assert_eq!(
        prepared.elementwise_region_summary(),
        (0, 0),
        "a five-input region exceeds the CPU fused kernel's input limit"
    );

    let mut prepared_out = run_prepared(&mut engine, &runtime, &prepared, &refs).unwrap();
    assert_eq!(prepared.elementwise_region_execution_counts(), (0, 0));
    let actual = prepared_out.pop().unwrap();
    let actual = actual.as_slice::<f64>().unwrap();
    for (i, value) in actual.iter().enumerate() {
        let expected = 15.0 + 5.0 * (i % 7) as f64;
        assert_eq!(*value, expected);
    }
}

/// Stage 1: a region with several live-outs publishes every one of them.
///
/// Both `y` and `z` are program outputs of one elementwise region.
#[test]
fn prepared_elementwise_region_publishes_multiple_live_outs() {
    let mut engine = CpuEngine::serial();
    let runtime = cpu_runtime(&engine);
    let n = 16 * 1024usize;

    let data: Vec<f64> = (0..n).map(|i| 0.25 + (i % 13) as f64 * 0.125).collect();
    let x = TracedTensor::from_tensor_concrete_shape(
        Tensor::from_vec_col_major(vec![n], data.clone()).unwrap(),
    )
    .unwrap();
    let shared = (&x + &x).unwrap();
    let y = shared.exp().unwrap().tanh().unwrap();
    let z = (&shared * &shared).unwrap();
    let program = GraphCompiler::new()
        .compile_many(&[&y, &z])
        .expect("two-output constant graph should compile");

    let prepared = prepare(&engine, &runtime, &program, &[]).unwrap();
    let (regions, covered) = prepared.elementwise_region_summary();
    assert_eq!((regions, covered), (1, 4), "one region covers the chain");

    let mut prepared_out = run_prepared(&mut engine, &runtime, &prepared, &[]).unwrap();
    let (fused, fallbacks) = prepared.elementwise_region_execution_counts();
    assert_eq!(
        (fused, fallbacks),
        (1, 0),
        "the two-live-output region should fuse once"
    );

    assert_eq!(prepared_out.len(), 2);
    let z_out = prepared_out.pop().unwrap();
    let y_out = prepared_out.pop().unwrap();
    let y_out = y_out.as_slice::<f64>().unwrap();
    let z_out = z_out.as_slice::<f64>().unwrap();
    for (index, &value) in data.iter().enumerate() {
        let doubled = value + value;
        let expected_y = doubled.exp().tanh();
        let expected_z = doubled * doubled;
        assert!(
            (y_out[index] - expected_y).abs() <= 1e-12,
            "live-out y diverged from the host reference at {index}: {} != {expected_y}",
            y_out[index]
        );
        assert!(
            (z_out[index] - expected_z).abs() <= 1e-12,
            "live-out z diverged from the host reference at {index}: {} != {expected_z}",
            z_out[index]
        );
    }
}

/// Stage 1: an elementwise region next to an FFI operation.
///
/// The region covers only the elementwise chain; the matrix multiply stays its
/// own command, and both outputs match a host reference.
#[test]
fn prepared_elementwise_region_stays_separate_from_ffi_op() {
    let mut engine = CpuEngine::serial();
    let runtime = cpu_runtime(&engine);
    // The CPU fused kernel only engages above its element floor, so the chain
    // runs over a vector large enough to fuse while the multiply stays small.
    let n = 8usize;
    let chain_len = 16 * 1024usize;

    let matrix = |data: &[f64]| Tensor::from_vec_col_major(vec![n, n], data.to_vec()).unwrap();
    let a_data: Vec<f64> = (0..n * n).map(|i| 0.5 + (i % 7) as f64 * 0.125).collect();
    let b_data: Vec<f64> = (0..n * n).map(|i| 0.25 + (i % 7) as f64 * 0.125).collect();
    let a = TracedTensor::from_tensor_concrete_shape(matrix(&a_data)).unwrap();
    let b = TracedTensor::from_tensor_concrete_shape(matrix(&b_data)).unwrap();
    let chain_data: Vec<f64> = (0..chain_len)
        .map(|i| 0.5 + (i % 5) as f64 * 0.25)
        .collect();
    let x = TracedTensor::from_tensor_concrete_shape(
        Tensor::from_vec_col_major(vec![chain_len], chain_data.clone()).unwrap(),
    )
    .unwrap();

    let m = a.matmul(&b).unwrap();
    let chain = (&x * &x).unwrap().exp().unwrap().tanh().unwrap();
    let program = GraphCompiler::new()
        .compile_many(&[&m, &chain])
        .expect("mixed graph should compile");

    let prepared = prepare(&engine, &runtime, &program, &[]).unwrap();
    let (regions, covered) = prepared.elementwise_region_summary();
    assert_eq!(
        (regions, covered),
        (1, 3),
        "only the elementwise chain should form a region"
    );

    let mut prepared_out = run_prepared(&mut engine, &runtime, &prepared, &[]).unwrap();
    let (fused, fallbacks) = prepared.elementwise_region_execution_counts();
    assert_eq!(
        (fused, fallbacks),
        (1, 0),
        "the CPU engine should fuse the chain region"
    );

    assert_eq!(prepared_out.len(), 2);
    let chain_out = prepared_out.pop().unwrap();
    let m_out = prepared_out.pop().unwrap();
    let chain_out = chain_out.as_slice::<f64>().unwrap();
    let m_out = m_out.as_slice::<f64>().unwrap();
    for (index, &value) in chain_data.iter().enumerate() {
        let expected = (value * value).exp().tanh();
        assert!(
            (chain_out[index] - expected).abs() <= 1e-12,
            "chain output diverged from the host reference at {index}: {} != {expected}",
            chain_out[index]
        );
    }
    for j in 0..n {
        for i in 0..n {
            let expected: f64 = (0..n).map(|k| a_data[i + k * n] * b_data[k + j * n]).sum();
            assert!(
                (m_out[i + j * n] - expected).abs() <= 1e-12,
                "matmul output diverged from the host reference at ({i}, {j}): {} != {expected}",
                m_out[i + j * n]
            );
        }
    }
}

/// Stage 1: repeated prepared runs neither fall back nor accumulate state.
#[test]
fn prepared_elementwise_region_is_stable_across_repeated_runs() {
    let mut engine = CpuEngine::serial();
    let runtime = cpu_runtime(&engine);
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
    let prepared = prepare(&engine, &runtime, &program, &[&input]).unwrap();

    let mut first: Option<Vec<f64>> = None;
    for run in 1..=16 {
        let mut out = run_prepared(&mut engine, &runtime, &prepared, &[&input]).unwrap();
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
/// below the CPU fused kernel's element floor, and both outputs match a host
/// reference, so interleaving regions with FFI work keeps its contract.
#[test]
fn prepared_regions_interleave_with_ffi_work() {
    let mut engine = CpuEngine::serial();
    let runtime = cpu_runtime(&engine);
    let chain_len = 16 * 1024usize;
    let n = 8usize;

    let x_data: Vec<f64> = (0..chain_len)
        .map(|i| 0.5 + (i % 11) as f64 * 0.125)
        .collect();
    let a_data: Vec<f64> = (0..n * n).map(|i| 0.5 + (i % 7) as f64 * 0.125).collect();
    let b_data: Vec<f64> = (0..n * n).map(|i| 0.25 + (i % 7) as f64 * 0.125).collect();

    let x = TracedTensor::from_tensor_concrete_shape(
        Tensor::from_vec_col_major(vec![chain_len], x_data.clone()).unwrap(),
    )
    .unwrap();
    let a = TracedTensor::from_tensor_concrete_shape(
        Tensor::from_vec_col_major(vec![n, n], a_data.clone()).unwrap(),
    )
    .unwrap();
    let b = TracedTensor::from_tensor_concrete_shape(
        Tensor::from_vec_col_major(vec![n, n], b_data.clone()).unwrap(),
    )
    .unwrap();

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

    let prepared = prepare(&engine, &runtime, &program, &[]).unwrap();
    let (regions, _) = prepared.elementwise_region_summary();
    assert!(regions >= 1, "the chains should plan at least one region");

    let mut prepared_out = run_prepared(&mut engine, &runtime, &prepared, &[]).unwrap();
    let (fused, fallbacks) = prepared.elementwise_region_execution_counts();
    assert!(fused >= 1, "the large chain should fuse: {fused} fused");
    assert!(
        fallbacks >= 1,
        "the chain below the element floor should fall back: {fallbacks} fallbacks"
    );

    assert_eq!(prepared_out.len(), 2);
    let after_out = prepared_out.pop().unwrap();
    let before_out = prepared_out.pop().unwrap();
    let after_out = after_out.as_slice::<f64>().unwrap();
    let before_out = before_out.as_slice::<f64>().unwrap();
    for (index, &value) in x_data.iter().enumerate() {
        let expected = (value * value).exp().tanh();
        assert!(
            (before_out[index] - expected).abs() <= 1e-12,
            "fused chain output diverged from the host reference at {index}: {} != {expected}",
            before_out[index]
        );
    }
    let mut product_data = vec![0.0_f64; n * n];
    for j in 0..n {
        for i in 0..n {
            product_data[i + j * n] = (0..n).map(|k| a_data[i + k * n] * b_data[k + j * n]).sum();
        }
    }
    for (index, &value) in product_data.iter().enumerate() {
        let expected = (value * value).exp().tanh();
        assert!(
            (after_out[index] - expected).abs() <= 1e-12,
            "fallback chain output diverged from the host reference at {index}: {} != {expected}",
            after_out[index]
        );
    }
}

/// Stage 1: the fallback path stays stable across repeated runs.
#[test]
fn prepared_elementwise_fallback_is_stable_across_repeated_runs() {
    let mut engine = CpuEngine::serial();
    let runtime = cpu_runtime(&engine);
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
    let prepared = prepare(&engine, &runtime, &program, &[&input]).unwrap();

    let mut first: Option<Vec<f64>> = None;
    for run in 1..=8 {
        let mut out = run_prepared(&mut engine, &runtime, &prepared, &[&input]).unwrap();
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

#[test]
fn traced_slice_rejects_out_of_range_limit_at_graph_build() {
    // The sibling `slice_axis` builder, the CPU backend, and the CUDA backend
    // all reject a limit that exceeds a concrete input extent; raw
    // `TracedTensor::slice` used to build successfully and register a phantom
    // output shape. See issue #2042.
    let x = TracedTensor::from_vec_col_major(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let err = x
        .slice(SliceConfig {
            starts: vec![0],
            limits: vec![100],
            strides: vec![1],
        })
        .unwrap_err();
    assert!(matches!(
        err,
        Error::Validation {
            op: "TracedTensor::slice",
            phase: ErrorPhase::GraphBuild,
            source: ValidationError::InvalidSliceBounds {
                start: 0,
                end: 100,
                axis_len: 4,
            },
        }
    ));
}

#[test]
fn prepared_identity_output_is_borrowed_and_a_computed_output_is_compact() {
    let (engine, runtime) = cpu_engine_and_runtime();
    let x = TracedTensor::input_concrete_shape(DType::F64, &[2, 2]).unwrap();
    let identity = GraphCompiler::new()
        .compile_with_input_specs(&x, &[(&x, DType::F64, &[2, 2])])
        .unwrap();
    let input = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let prepared = prepare(&engine, &runtime, &identity, &[&input]).unwrap();
    let reads = inputs_of(&[&input]);
    let mut engine = engine;
    let outputs = runtime.run(&mut [&mut engine], &prepared, &reads).unwrap();
    assert!(matches!(outputs[0], tenferro_runtime::Output::Borrowed(_)));

    // A computed value with a non-compact program layout is materialized.
    let y = x.transpose(&[1, 0]).unwrap().neg().unwrap();
    let program = GraphCompiler::new()
        .compile_with_input_specs(&y, &[(&x, DType::F64, &[2, 2])])
        .unwrap();
    let prepared = prepare(&engine, &runtime, &program, &[&input]).unwrap();
    let outputs = runtime.run(&mut [&mut engine], &prepared, &reads).unwrap();
    match &outputs[0] {
        tenferro_runtime::Output::Owned(tensor) => {
            assert!(tensor.is_col_major_contiguous().unwrap());
            assert_eq!(tensor.shape(), &[2, 2]);
        }
        tenferro_runtime::Output::Borrowed(_) => panic!("a computed output must be owned"),
    }
}
