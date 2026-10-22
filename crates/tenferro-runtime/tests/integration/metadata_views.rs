//! Folded metadata views (#2226): a prepared program executes a `Transpose`,
//! `BroadcastInDim`, `Slice` or `Reshape` with one reading consumer as that
//! consumer's input view, so the schedule loses the command and the result is
//! the one the materializing command produced.

use tenferro_cpu::CpuEngine;
use tenferro_runtime::{
    DType, ExecutionEngine, GraphCompiler, Output, Prepared, Runtime, TracedTensor,
};
use tenferro_tensor::{SliceConfig, Tensor, TensorRead};

fn f64_tensor(shape: &[usize], data: Vec<f64>) -> Tensor {
    Tensor::from_vec_col_major(shape.to_vec(), data).expect("test tensor")
}

fn constant(shape: &[usize], data: Vec<f64>) -> TracedTensor {
    TracedTensor::from_tensor_concrete_shape(f64_tensor(shape, data)).expect("traced constant")
}

fn runtime_for(engine: &CpuEngine) -> Runtime {
    let mut builder = Runtime::builder();
    builder
        .register_engine(tenferro_cpu::runtime_engine_registration(engine).expect("registration"))
        .expect("register engine");
    builder.build().expect("runtime")
}

fn prepare(
    runtime: &Runtime,
    engine: &CpuEngine,
    outputs: &[&TracedTensor],
    inputs: &[TensorRead<'_>],
) -> Prepared {
    let program = GraphCompiler::new().compile_many(outputs).expect("compile");
    runtime
        .prepare(&program, inputs, &[engine.target()])
        .expect("prepare")
}

fn run_owned(
    runtime: &Runtime,
    engine: &mut CpuEngine,
    prepared: &Prepared,
    inputs: &[TensorRead<'_>],
) -> Vec<Tensor> {
    let outputs = runtime
        .run(&mut [&mut *engine], prepared, inputs)
        .expect("run");
    outputs
        .into_iter()
        .map(|output| output.into_tensor(engine).expect("output"))
        .collect()
}

fn values(tensor: &Tensor) -> Vec<f64> {
    tensor.as_slice::<f64>().expect("f64 output").to_vec()
}

#[test]
fn transpose_of_an_intermediate_folds_into_add() {
    let mut engine = CpuEngine::serial();
    let runtime = runtime_for(&engine);
    // a = [[1, 3, 5], [2, 4, 6]] (column-major 2x3); -a^T + b.
    let a = constant(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = constant(&[3, 2], vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
    let out = (&a.neg().unwrap().transpose(&[1, 0]).unwrap() + &b).unwrap();

    let prepared = prepare(&runtime, &engine, &[&out], &[]);
    assert_eq!(prepared.execution_command_count(), 2, "Negate, Add");
    let outputs = run_owned(&runtime, &mut engine, &prepared, &[]);
    assert_eq!(outputs[0].shape(), &[3, 2]);
    // -a^T column-major: [-1, -3, -5, -2, -4, -6].
    assert_eq!(values(&outputs[0]), vec![9.0, 17.0, 25.0, 38.0, 46.0, 54.0]);
}

#[test]
fn transpose_folds_into_a_reduction() {
    let mut engine = CpuEngine::serial();
    let runtime = runtime_for(&engine);
    let a = constant(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let negated = a.neg().unwrap();
    let transposed = negated.transpose(&[1, 0]).unwrap();
    let column_sums = transposed.reduce_sum(Some(&[0])).unwrap();

    let prepared = prepare(&runtime, &engine, &[&column_sums], &[]);
    assert_eq!(prepared.execution_command_count(), 2, "Negate, ReduceSum");
    let outputs = run_owned(&runtime, &mut engine, &prepared, &[]);
    // Rows of a summed: -(1+3+5), -(2+4+6).
    assert_eq!(values(&outputs[0]), vec![-9.0, -12.0]);
}

#[test]
fn broadcast_of_an_intermediate_folds_into_multiply() {
    let mut engine = CpuEngine::serial();
    let runtime = runtime_for(&engine);
    let v = constant(&[3], vec![1.0, 2.0, 3.0]);
    let w = constant(&[3, 2], vec![1.0, 1.0, 1.0, 10.0, 10.0, 10.0]);
    let out = (&v.neg().unwrap().broadcast_in_dim(&[3, 2], &[0]).unwrap() * &w).unwrap();

    let prepared = prepare(&runtime, &engine, &[&out], &[]);
    assert_eq!(prepared.execution_command_count(), 2, "Negate, Multiply");
    let outputs = run_owned(&runtime, &mut engine, &prepared, &[]);
    assert_eq!(
        values(&outputs[0]),
        vec![-1.0, -2.0, -3.0, -10.0, -20.0, -30.0]
    );
}

#[test]
fn slice_and_reshape_of_intermediates_fold() {
    let mut engine = CpuEngine::serial();
    let runtime = runtime_for(&engine);
    let a = constant(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let negated = a.neg().unwrap();
    let sliced = negated
        .slice(SliceConfig {
            starts: vec![1, 0],
            limits: vec![2, 3],
            strides: vec![1, 1],
        })
        .unwrap()
        .exp()
        .unwrap();
    let reshaped = negated.reshape([3, 2]).unwrap().exp().unwrap();

    let prepared = prepare(&runtime, &engine, &[&sliced, &reshaped], &[]);
    assert_eq!(prepared.execution_command_count(), 3, "Negate, Exp, Exp");
    let outputs = run_owned(&runtime, &mut engine, &prepared, &[]);
    let expected: Vec<f64> = [-2.0_f64, -4.0, -6.0].iter().map(|x| x.exp()).collect();
    assert_eq!(outputs[0].shape(), &[1, 3]);
    assert_eq!(values(&outputs[0]), expected);
    assert_eq!(outputs[1].shape(), &[3, 2]);
    let expected: Vec<f64> = [-1.0_f64, -2.0, -3.0, -4.0, -5.0, -6.0]
        .iter()
        .map(|x| x.exp())
        .collect();
    assert_eq!(values(&outputs[1]), expected);
}

#[test]
fn a_transposed_program_output_keeps_its_command() {
    let mut engine = CpuEngine::serial();
    let runtime = runtime_for(&engine);
    let a = constant(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let out = a.neg().unwrap().transpose(&[1, 0]).unwrap();

    let prepared = prepare(&runtime, &engine, &[&out], &[]);
    assert_eq!(prepared.execution_command_count(), 2, "Negate, Transpose");
    let outputs = run_owned(&runtime, &mut engine, &prepared, &[]);
    assert_eq!(outputs[0].shape(), &[3, 2]);
    assert_eq!(
        values(&outputs[0]),
        vec![-1.0, -3.0, -5.0, -2.0, -4.0, -6.0]
    );
}

#[test]
fn a_view_of_a_run_input_folds_into_its_consumer() {
    let mut engine = CpuEngine::serial();
    let runtime = runtime_for(&engine);
    let x = TracedTensor::input_concrete_shape(DType::F64, &[2, 3]).unwrap();
    let c = constant(&[3, 2], vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
    let sum = (&x.transpose(&[1, 0]).unwrap() + &c).unwrap();
    let input = f64_tensor(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let reads = [TensorRead::from_tensor(&input)];

    let program = GraphCompiler::new()
        .compile_with_input_specs(&sum, &[(&x, DType::F64, &[2, 3])])
        .expect("compile");
    let prepared = runtime
        .prepare(&program, &reads, &[engine.target()])
        .expect("prepare");
    assert_eq!(prepared.execution_command_count(), 1, "Add");
    let outputs = run_owned(&runtime, &mut engine, &prepared, &reads);
    assert_eq!(
        values(&outputs[0]),
        vec![11.0, 23.0, 35.0, 42.0, 54.0, 66.0]
    );
}

#[test]
fn an_input_output_stays_borrowed_and_a_repeated_output_is_duplicated() {
    let mut engine = CpuEngine::serial();
    let runtime = runtime_for(&engine);

    // The identity program returns the run's input without a copy.
    let x = TracedTensor::input_concrete_shape(DType::F64, &[2]).unwrap();
    let input = f64_tensor(&[2], vec![1.0, 2.0]);
    let reads = [TensorRead::from_tensor(&input)];
    let program = GraphCompiler::new()
        .compile_with_input_specs(&x, &[(&x, DType::F64, &[2])])
        .expect("compile");
    let prepared = runtime
        .prepare(&program, &reads, &[engine.target()])
        .expect("prepare");
    let outputs = runtime
        .run(&mut [&mut engine], &prepared, &reads)
        .expect("run");
    assert!(matches!(outputs[0], Output::Borrowed(_)));
    let owned = outputs
        .into_iter()
        .next()
        .unwrap()
        .into_tensor(&mut engine)
        .unwrap();
    assert_eq!(values(&owned), vec![1.0, 2.0]);

    // One owned result named three times: the first position takes it, the
    // others are engine copies with the same values.
    let negated = constant(&[2], vec![1.0, 2.0]).neg().unwrap();
    let prepared = prepare(&runtime, &engine, &[&negated, &negated, &negated], &[]);
    let outputs = runtime
        .run(&mut [&mut engine], &prepared, &[])
        .expect("run");
    assert_eq!(outputs.len(), 3);
    assert!(outputs
        .iter()
        .all(|output| matches!(output, Output::Owned(_))));
    for output in outputs {
        let tensor = output.into_tensor(&mut engine).unwrap();
        assert_eq!(values(&tensor), vec![-1.0, -2.0]);
    }
}
