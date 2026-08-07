//! Executable examples for the bundled `tenferro-compute` downstream skill.

#![allow(dead_code, unused_imports, unused_variables, unused_mut)]

fn assert_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let error = (actual - expected).abs();
        assert!(
            error < 1.0e-12,
            "value {index}: actual={actual}, expected={expected}, error={error}"
        );
    }
}

fn concrete_operation_and_column_major() -> Result<(), Box<dyn std::error::Error>> {
    // snippet-start:concrete-operation
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{TypedTensor, TypedTensorOpsExt};

let mut backend = CpuBackend::new();
// The leftmost dimension varies fastest: this is a 2 x 3 column-major tensor.
let x = TypedTensor::<f64>::from_vec_col_major(
    vec![2, 3],
    vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
)?;
let weights = TypedTensor::<f64>::from_vec_col_major(
    vec![3, 2],
    vec![0.5, -1.0, 1.5, 1.0, 2.0, -0.5],
)?;
let projected = x.matmul(&weights, &mut backend)?;
assert_eq!(projected.shape(), &[2, 2]);
assert_eq!(projected.host_data()?, &[3.0, 6.0, 3.5, 11.0]);
    // snippet-end:concrete-operation
    Ok(())
}

fn eager_operation() -> Result<(), Box<dyn std::error::Error>> {
    // snippet-start:eager-operation
use tenferro_ad::{EagerRuntime, Tensor};

let runtime = EagerRuntime::new()?;
let x = runtime.variable_from(Tensor::from_vec_col_major(
    vec![3],
    vec![1.0_f64, 2.0, 3.0],
)?)?;
let prediction = x.mul(&x)?;
let loss = prediction.reduce_sum(Some(&[0]))?;
loss.backward()?;
assert_eq!(
    x.grad()?.expect("tracked variable should receive a gradient").as_slice::<f64>()?,
    &[2.0, 4.0, 6.0],
);
    // snippet-end:eager-operation
    Ok(())
}

fn traced_operation_with_extension_registration() -> Result<(), Box<dyn std::error::Error>> {
    // snippet-start:traced-extension-operation
use tenferro_cpu::{runtime_engine_id, runtime_engine_registration, CpuBackend};
use tenferro_einsum::TraceContextEinsumExt;
use tenferro_runtime::program::ProgramInputSpec;
use tenferro_runtime::{GraphCompiler, Runtime, Tensor, TraceContext};

let a = Tensor::from_vec_col_major(
    vec![2, 3],
    vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
)?;
let b = Tensor::from_vec_col_major(
    vec![3, 2],
    vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
)?;
let mut trace = TraceContext::new();
let a_value = trace.input(ProgramInputSpec::new(a.dtype(), [2.into(), 3.into()]))?;
let b_value = trace.input(ProgramInputSpec::new(b.dtype(), [3.into(), 2.into()]))?;
let product = trace.einsum(&[a_value, b_value], "ij,jk->ik")?;
let graph = trace.finish(&[product])?;
let program = GraphCompiler::new().compile_traced_graph(&graph)?;

let backend = CpuBackend::new();
let mut builder = Runtime::builder();
builder.register_engine(runtime_engine_registration(&backend)?)?;
builder.install_extension_module(tenferro_einsum::extension_module::<CpuBackend>(
    runtime_engine_id()?,
)?)?;
let runtime = builder.build()?;
let mut outputs = runtime.run_compiled(&program, &[&a, &b])?;
let result = outputs.remove(0);
assert_eq!(result.as_slice::<f64>()?, &[22.0, 28.0, 49.0, 64.0]);
    // snippet-end:traced-extension-operation
    Ok(())
}

fn compile_once_run_many() -> Result<(), Box<dyn std::error::Error>> {
    // snippet-start:compile-once-run-many
use tenferro_cpu::{runtime_engine_id, runtime_engine_registration, CpuBackend};
use tenferro_einsum::TraceContextEinsumExt;
use tenferro_runtime::program::ProgramInputSpec;
use tenferro_runtime::{GraphCompiler, Runtime, Tensor, TraceContext};

let mut trace = TraceContext::new();
let input = trace.input(ProgramInputSpec::new(tenferro_runtime::DType::F64, [2.into()]))?;
let output = trace.einsum(&[input], "i->i")?;
let graph = trace.finish(&[output])?;
// Compile the shape-specialized program once.
let program = GraphCompiler::new().compile_traced_graph(&graph)?;

// Reuse one backend, extension registration, and runtime for repeated inputs.
let backend = CpuBackend::new();
let mut builder = Runtime::builder();
builder.register_engine(runtime_engine_registration(&backend)?)?;
builder.install_extension_module(tenferro_einsum::extension_module::<CpuBackend>(
    runtime_engine_id()?,
)?)?;
let runtime = builder.build()?;
for (input, expected) in [
    (vec![1.0_f64, 2.0], vec![1.0, 2.0]),
    (vec![3.0_f64, 4.0], vec![3.0, 4.0]),
] {
    let value = Tensor::from_vec_col_major(vec![2], input)?;
    let mut outputs = runtime.run_compiled(&program, &[&value])?;
    assert_eq!(outputs.remove(0).as_slice::<f64>()?, &expected);
}
    // snippet-end:compile-once-run-many
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    concrete_operation_and_column_major()?;
    eager_operation()?;
    traced_operation_with_extension_registration()?;
    compile_once_run_many()?;
    Ok(())
}
