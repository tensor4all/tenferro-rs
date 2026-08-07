# API cheatsheet

Choose the tier before choosing the method. Direct operations receive an
explicit mutable backend; eager operations run through an `EagerRuntime`; traced
operations build a graph and execute it later.

## Direct concrete tensors

Import the backend and the extension trait that owns the method:

<!-- snippet-source: docs/tutorial-code/src/bin/tenferro_compute_skill.rs#concrete-operation -->
```rust
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
```
<!-- end-snippet-source -->

The important shape is `x.matmul(&weights, &mut backend)`. For dynamic dtypes
use `TensorOpsExt`; for static scalar types use `TypedTensorOpsExt`.

## Eager tensors

`EagerTensor` methods omit the backend argument because the runtime owns it.
Tracked values support `backward()` and functional eager transforms:

<!-- snippet-source: docs/tutorial-code/src/bin/tenferro_compute_skill.rs#eager-operation -->
```rust
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
```
<!-- end-snippet-source -->

Common import recipes:

- `tenferro_einsum::EagerEinsumExt` for eager einsum.
- `tenferro_linalg::EagerTensorLinalgExt` for eager linear algebra.
- `tenferro_ad::{EagerRuntime, Tensor}` for eager values and runtime AD.

## Traced tensors and extensions

Traced core operations compile into a graph. Standard operation families are
extensions: import their traced trait and install the same family's extension
module into the runtime that owns the registered backend engine.

<!-- snippet-source: docs/tutorial-code/src/bin/tenferro_compute_skill.rs#traced-extension-operation -->
```rust
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
```
<!-- end-snippet-source -->

Import recipes:

- `tenferro_einsum::TraceContextEinsumExt` for `TraceContext::einsum` graph
  construction; use `tenferro_einsum::TracedTensorEinsumExt` when the receiver
  is an already traced tensor.
- `tenferro_linalg::TracedTensorLinalgExt` for traced linear algebra.
- `tenferro_ad::TracedTensorAdExt` for `grad`, `vjp`, and `jvp` transforms.

If a method is present in the guide but Rust reports E0599, check the owning
crate's root re-exports and bring its `*Ext` trait into the local module.
