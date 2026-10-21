# API cheatsheet

Choose the tier before choosing the method. Direct operations take the caller's
`&mut dyn Engine`; eager operations run through an `EagerRuntime` that borrows
that same engine; traced operations build a graph and execute it later. If an older example refers
to a removed module, free function, or fallible-constructor change, start with
the [API migration guide](../../../../docs/getting-started/api-migration.md).

## Direct concrete tensors

Import the engine type and the extension trait that owns the operation method:

<!-- snippet-source: docs/tutorial-code/src/bin/tenferro_compute_skill.rs#concrete-operation -->
```rust
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
```
<!-- end-snippet-source -->

The important shape is `x.matmul(&weights, &mut engine)`. For
dynamic dtypes use `TensorSessionOpsExt`; for static scalar types use
`TypedTensorSessionOpsExt`. The receiver methods mirror the eager session
methods with the engine last (`x.gather(&idx, config, &mut engine)`,
`x.reduce_max(None, &mut engine)`, `x.dot_general(&y, config, &mut engine)`),
including indexing, padding, concatenation and the triangular/diagonal ops.
Only the dtype-changing `convert` / `cast` are `Tensor`-only
(`Tensor::from_typed` moves a typed tensor there without a copy).

## Eager tensors

`EagerRuntime::with_eager_session` borrows the caller's engine for eager
operations. Tracked values support `backward(&mut engine)` and functional eager
transforms (`grad`, `vjp`, `jvp`, each taking `&mut engine`):

<!-- snippet-source: docs/tutorial-code/src/bin/tenferro_compute_skill.rs#eager-operation -->
```rust
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
```
<!-- end-snippet-source -->

Common import recipes:

- `tenferro_einsum::EagerSessionEinsumExt` for borrowed eager einsum and `tensordot` (`runtime.with_eager_session(&mut engine, |s| s.einsum(&[&a, &b], "ij,jk->ik"))?`).
- `tenferro_linalg::EagerSessionLinalgExt` for borrowed eager linear algebra (`EagerTensorLinalgExt::solve` remains a calling-thread `no_grad` exception).
- `tenferro_ad::{EagerRuntime, Tensor}` for eager values and runtime AD.

## Activations, softmax and normalization

`erf`, `sigmoid`, `silu`, `softplus`, `gelu` (exact) / `gelu_tanh`,
`reduce_mean`, `softmax` / `log_softmax`, `masked_softmax` /
`masked_log_softmax` (Bool mask, `true` participates), `layer_norm` /
`rms_norm` (optional rank-1 `weight` / `bias`, `eps`) and NumPy-style
`take_along_axis` have one name on every tier: `s.softmax(&x, axis)` on an
`EagerSession`, `x.softmax(axis)` on a `TracedTensor`, and
`x.softmax(axis, &mut engine)` through `TensorSessionOpsExt`. They are stable
compositions (no overflow, max-subtracted softmax, all-masked slices give 0 /
`-inf` with a zero gradient) and differentiate through eager and traced AD; do
not hand-roll them. The edge-case policy is in
[`docs/spec/tensor-semantics.md`](../../../../docs/spec/tensor-semantics.md),
Section VIII. `TypedTensorSessionOpsExt` has `take_along_axis` too, with the
index tensor kept dtype-erased; `erf` is real-only and does not lower to
StableHLO.

## Traced tensors and extensions

Traced core operations compile into a graph. Standard operation families are
extensions: import their traced trait and install the same family's extension
module into the runtime that owns the registered backend engine.

<!-- snippet-source: docs/tutorial-code/src/bin/tenferro_compute_skill.rs#traced-extension-operation -->
```rust
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

## Repeated CPU eager/AD and prepared trace calls

Hold one engine and one runtime and reuse both: register the engine once
(`runtime_engine_registration(&engine)`), prepare the traced program once
(`Runtime::prepare`), and replay it with `Runtime::run` while passing
`&mut engine` to ordinary and eager calls. There is no scope to enter and no
admission to amortize; the engine owns the pool, plan cache and buffer pool.
Compile and prepare, and warm up, outside steady-state timing;
allocation-returning calls still allocate outputs (the `_into` forms do not).
See the
[engine and prepared-execution contract](../../../../docs/guides/ordinary-and-prepared-execution.md#one-engine-across-ordinary-eager-and-prepared-work).

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
