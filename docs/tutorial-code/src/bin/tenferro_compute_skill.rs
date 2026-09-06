//! Executable examples for the bundled `tenferro-compute` downstream skill.

#[rustfmt::skip]
fn concrete_operation_and_column_major() -> Result<(), Box<dyn std::error::Error>> {
    // snippet-start:concrete-operation
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
use tenferro_tensor::BackendSessionHost;

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
let projected = backend.with_backend_session(|session| x.matmul(&weights, session))?;
assert_eq!(projected.shape(), &[2, 2]);
assert_eq!(projected.host_data()?, &[3.0, 6.0, 3.5, 11.0]);
    // snippet-end:concrete-operation
    Ok(())
}

#[rustfmt::skip]
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

#[rustfmt::skip]
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

#[rustfmt::skip]
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
use tenferro_cpu::CpuBackend;
use tenferro_einsum::{ConcreteEinsumPlan, EinsumSubscripts, TensorEinsumExt};
use tenferro_tensor::{BackendSessionHost, Tensor};

let lhs = Tensor::from_vec_col_major([2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
let rhs = Tensor::from_vec_col_major([2, 2], vec![2.0_f64, 0.0, 1.0, 2.0])?;
let mut backend = CpuBackend::new();
// Ordinary execution: no explicit preparation needed.
let ordinary = backend.with_backend_session(|session| {
    [&lhs, &rhs].einsum("ij,jk->ik", session)
})?;
assert_eq!(ordinary.as_slice::<f64>()?, &[2.0, 4.0, 7.0, 10.0]);

// Strings are fine for one-time preparation. The plan does not retain inputs.
let plan = ConcreteEinsumPlan::prepare([&lhs, &rhs], "ij,jk->ik")?;
for (data, expected) in [
    (vec![1.0_f64, 2.0, 3.0, 4.0], [2.0, 4.0, 7.0, 10.0]),
    (vec![2.0_f64, 4.0, 6.0, 8.0], [4.0, 8.0, 14.0, 20.0]),
] {
    let next_lhs = Tensor::from_vec_col_major([2, 2], data)?;
    let result = backend.with_backend_session(|session| {
        plan.execute([&next_lhs, &rhs], session)
    })?;
    assert_eq!(result.as_slice::<f64>()?, &expected);
}

// Integer labels describe the equation; this ordinary call still plans.
let equation = EinsumSubscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
let structured = backend.with_backend_session(|session| {
    [&lhs, &rhs].einsum_subscripts(&equation, session)
})?;
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
