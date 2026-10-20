//! Executable examples for the bundled `tenferro-compute` downstream skill.

#[rustfmt::skip]
fn concrete_operation_and_column_major() -> Result<(), Box<dyn std::error::Error>> {
    // snippet-start:concrete-operation
use tenferro_cpu::CpuEngine;
use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};

let mut engine = CpuEngine::serial();
// The leftmost dimension varies fastest: this is a 2 x 3 column-major tensor.
let x = TypedTensor::<f64>::from_vec_col_major(
    vec![2, 3],
    vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
)?;
let weights = TypedTensor::<f64>::from_vec_col_major(
    vec![3, 2],
    vec![0.5, -1.0, 1.5, 1.0, 2.0, -0.5],
)?;
let projected = x.matmul(&weights, &mut engine)?;
assert_eq!(projected.shape(), &[2, 2]);
assert_eq!(projected.host_data()?, &[3.0, 6.0, 3.5, 11.0]);
    // snippet-end:concrete-operation
    Ok(())
}

#[rustfmt::skip]
fn eager_operation() -> Result<(), Box<dyn std::error::Error>> {
    // snippet-start:eager-operation
use tenferro_ad::{EagerRuntime, Tensor};
use tenferro_cpu::CpuEngine;

let runtime = EagerRuntime::new()?;
let mut engine = CpuEngine::serial();
let x = runtime.variable_from(Tensor::from_vec_col_major(
    vec![3],
    vec![1.0_f64, 2.0, 3.0],
)?)?;
let loss = runtime.with_eager_session(&mut engine, |s| {
    let prediction = s.mul(&x, &x)?;
    s.reduce_sum(&prediction, Some(&[0]))
})?;
loss.backward(&mut engine)?;
assert_eq!(
    x.grad()?.expect("tracked variable should receive a gradient").as_slice::<f64>()?,
    &[2.0, 4.0, 6.0],
);
    // snippet-end:eager-operation
    Ok(())
}

#[rustfmt::skip]
fn traced_operation_with_extension_registration() -> Result<(), Box<dyn std::error::Error>> {
    // snippet-start:traced-extension-operation
use tenferro_cpu::{runtime_engine_id, runtime_engine_registration, CpuEngine};
use tenferro_einsum::TraceContextEinsumExt;
use tenferro_runtime::program::ProgramInputSpec;
use tenferro_runtime::{ExecutionEngine, GraphCompiler, Runtime, Tensor, TraceContext};

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

let mut engine = CpuEngine::serial();
let mut builder = Runtime::builder();
builder.register_engine(runtime_engine_registration(&engine)?)?;
builder.install_extension_module(tenferro_einsum::extension_module::<CpuEngine>(
    runtime_engine_id()?,
)?)?;
let runtime = builder.build()?;
let reads = [
    tenferro_runtime::TensorRead::from_tensor(&a),
    tenferro_runtime::TensorRead::from_tensor(&b),
];
let prepared = runtime.prepare(&program, &reads, &[engine.target()])?;
let mut outputs = runtime.run(&mut [&mut engine], &prepared, &reads)?;
let result = outputs.remove(0).into_tensor(&mut engine)?;
assert_eq!(result.as_slice::<f64>()?, &[22.0, 28.0, 49.0, 64.0]);
    // snippet-end:traced-extension-operation
    Ok(())
}

#[rustfmt::skip]
fn compile_once_run_many() -> Result<(), Box<dyn std::error::Error>> {
    // snippet-start:compile-once-run-many
use tenferro_cpu::{runtime_engine_id, runtime_engine_registration, CpuEngine};
use tenferro_einsum::TraceContextEinsumExt;
use tenferro_runtime::program::ProgramInputSpec;
use tenferro_runtime::{ExecutionEngine, GraphCompiler, Runtime, Tensor, TraceContext};

let mut trace = TraceContext::new();
let input = trace.input(ProgramInputSpec::new(tenferro_runtime::DType::F64, [2.into()]))?;
let output = trace.einsum(&[input], "i->i")?;
let graph = trace.finish(&[output])?;
// Compile the shape-specialized program once.
let program = GraphCompiler::new().compile_traced_graph(&graph)?;

// Reuse one engine, extension registration, runtime and prepared program for
// repeated inputs: `prepare` runs once, `run` replays it.
let mut engine = CpuEngine::serial();
let mut builder = Runtime::builder();
builder.register_engine(runtime_engine_registration(&engine)?)?;
builder.install_extension_module(tenferro_einsum::extension_module::<CpuEngine>(
    runtime_engine_id()?,
)?)?;
let runtime = builder.build()?;
let target = engine.target();
let first = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
let prepared = runtime.prepare(
    &program,
    &[tenferro_runtime::TensorRead::from_tensor(&first)],
    &[target],
)?;
for (input, expected) in [
    (vec![1.0_f64, 2.0], vec![1.0, 2.0]),
    (vec![3.0_f64, 4.0], vec![3.0, 4.0]),
] {
    let value = Tensor::from_vec_col_major(vec![2], input)?;
    let read = tenferro_runtime::TensorRead::from_tensor(&value);
    let outputs = runtime.run(&mut [&mut engine], &prepared, &[read])?;
    let result = outputs.into_iter().next().expect("one output");
    assert_eq!(result.into_tensor(&mut engine)?.as_slice::<f64>()?, &expected);
}
    // snippet-end:compile-once-run-many
    Ok(())
}

#[rustfmt::skip]
fn borrowing_external_memory() -> Result<(), Box<dyn std::error::Error>> {
    // snippet-start:borrowing-external-memory
use tenferro_runtime::TypedTensorView;
use tenferro_tensor::TypedTensorViewMut;

// Wrap a column-major faer::Mat without copying. faer pads columns to
// alignment, so the borrowed slice spans `col_stride * ncols` elements and the
// column stride is passed explicitly; the padding is never read logically.
let mat = faer::Mat::from_fn(2, 3, |r, c| (r * 3 + c) as f64);
let data = unsafe { std::slice::from_raw_parts(mat.as_ref().as_ptr(), (mat.col_stride() as usize) * mat.ncols()) };
let view = TypedTensorView::from_slice(vec![2, 3], vec![1_isize, mat.col_stride()], 0, data)?;
assert_eq!(view.get(&[1, 2]), Some(&5.0));

// Wrap an ndarray row-major view the same way. Strides are arbitrary, so a
// row-major buffer is not transposed; it is only wrapped.
let arr = ndarray::Array2::from_shape_vec((2, 3), (0..6).map(|i| i as f64).collect())?;
let nview = TypedTensorView::from_slice(arr.shape(), arr.strides(), 0, arr.as_slice().expect("row-major array is contiguous"))?;
assert_eq!(nview.get(&[0, 2]), Some(&2.0));

// The mutable variant writes through the borrowed buffer.
let mut buffer = [0.0_f64, 0.0, 0.0, 0.0];
let mut mview = TypedTensorViewMut::from_slice(vec![2, 2], vec![1, 2], 0, &mut buffer)?;
*mview.get_mut(&[0, 0]).expect("in-bounds index") = 7.0;
assert_eq!(buffer[0], 7.0);
    // snippet-end:borrowing-external-memory
    Ok(())
}

#[rustfmt::skip]
fn ordinary_and_prepared_einsum() -> Result<(), Box<dyn std::error::Error>> {
    // snippet-start:ordinary-and-prepared-einsum
use tenferro_cpu::CpuEngine;
use tenferro_einsum::{ConcreteEinsumPlan, EinsumSubscripts, TensorEinsumExt};
use tenferro_tensor::Tensor;

let lhs = Tensor::from_vec_col_major([2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
let rhs = Tensor::from_vec_col_major([2, 2], vec![2.0_f64, 0.0, 1.0, 2.0])?;
let mut engine = CpuEngine::serial();
// Ordinary execution: no explicit preparation needed.
let ordinary = [&lhs, &rhs].einsum("ij,jk->ik", &mut engine)?;
assert_eq!(ordinary.as_slice::<f64>()?, &[2.0, 4.0, 7.0, 10.0]);

// Strings are fine for one-time preparation. The plan does not retain inputs.
let plan = ConcreteEinsumPlan::prepare([&lhs, &rhs], "ij,jk->ik")?;
// Borrow the same engine for the whole batch. The engine holds the plan cache
// and buffer pool, so reusing it is what makes the per-call cost the numerical
// operation alone.
for (data, expected) in [
    (vec![1.0_f64, 2.0, 3.0, 4.0], [2.0, 4.0, 7.0, 10.0]),
    (vec![2.0_f64, 4.0, 6.0, 8.0], [4.0, 8.0, 14.0, 20.0]),
] {
    let next_lhs = Tensor::from_vec_col_major([2, 2], data)?;
    let result = plan.execute([&next_lhs, &rhs], &mut engine)?;
    assert_eq!(result.as_slice::<f64>()?, &expected);
}

// Integer labels describe the equation; this ordinary call still plans.
let equation = EinsumSubscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
let structured = [&lhs, &rhs].einsum_subscripts(&equation, &mut engine)?;
assert_eq!(structured.as_slice::<f64>()?, &[2.0, 4.0, 7.0, 10.0]);
    // snippet-end:ordinary-and-prepared-einsum
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    ordinary_and_prepared_einsum()?;
    concrete_operation_and_column_major()?;
    eager_operation()?;
    traced_operation_with_extension_registration()?;
    compile_once_run_many()?;
    borrowing_external_memory()?;
    Ok(())
}
