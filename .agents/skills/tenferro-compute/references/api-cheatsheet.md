# API cheatsheet

Choose the tier before choosing the method. Direct operations run inside an explicit
`with_backend_session` entry; eager operations run through an `EagerRuntime`;
traced operations build a graph and execute it later. If an older example refers
to a removed module, free function, or fallible-constructor change, start with
the [API migration guide](../../../../docs/getting-started/api-migration.md).

## Direct concrete tensors

Import the backend, `BackendSessionHost`, and the extension trait that owns
the session method:

<!-- snippet-source: docs/tutorial-code/src/bin/tenferro_compute_skill.rs#concrete-operation -->
```rust
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
```
<!-- end-snippet-source -->

The important shape is
`backend.with_backend_session(|session| x.matmul(&weights, session))`. For
dynamic dtypes use `TensorSessionOpsExt`; for static scalar types use
`TypedTensorSessionOpsExt`.

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

## Borrowing external memory

Wrap an existing column-major buffer (a `faer::Mat`, an ndarray view, or any
slice plus strides) zero-copy with `TypedTensorView::from_slice`; use
`TypedTensorViewMut::from_slice` to write through the borrowed buffer. This is
how a tenferro kernel consumes memory owned by another library without a
migration: no tensor ownership is transferred.

<!-- snippet-source: docs/tutorial-code/src/bin/tenferro_compute_skill.rs#borrowing-external-memory -->
```rust
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
```
<!-- end-snippet-source -->

Because strides are arbitrary, row-major data wraps without transposition —
but kernels are tuned for column-major contiguity. Materialize a copy when the
wrapped buffer is row-major or its lifetime ends before the operation;
`TypedTensorView::duplicate()` requires a column-major-contiguous view, so for
a row-major wrap copy into a fresh `TypedTensor` (or an owned `Vec` in
column-major order) instead. If you search for the retired
`TypedStridedTensorView` (tenferro-rs#886), stop: `TypedTensorView::from_slice`
is its successor.
