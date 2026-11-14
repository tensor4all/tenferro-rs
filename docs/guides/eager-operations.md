# Eager Operations

This guide covers immediate execution: direct tensor computation without autodiff and
`EagerTensor` forward execution with optional eager autodiff. Start with
`TypedTensor<T, R>` or `Tensor` for work without autodiff. Use `EagerTensor`
when you want operations to run immediately inside an `EagerRuntime`, create
tracked variables when the workflow needs stateful reverse-mode and functional
`grad`/`vjp`/`jvp`, and use `backward()` when gradients should accumulate in
leaf gradient slots.

## Setup

For a published build, depend on the crates you use:

```toml
[dependencies]
tenferro-runtime = "..."
tenferro-cpu = "..."
tenferro-tensor = "..."
tenferro-ad = "..."
tenferro-linalg = "..."
tenferro-einsum = { version = "...", features = ["autodiff"] }
```

When working from a local checkout, replace the versions with `path = "..."`
entries that match your project layout. For a scratch crate created directly
inside the `tenferro-rs` checkout, include an empty `[workspace]` table so Cargo
does not try to enroll the scratch crate in the parent workspace:

```toml
[workspace]

[dependencies]
tenferro-runtime = { path = "../crates/tenferro-runtime" }
tenferro-cpu = { path = "../crates/tenferro-cpu" }
tenferro-tensor = { path = "../crates/tenferro-tensor" }
tenferro-ad = { path = "../crates/tenferro-ad" }
tenferro-linalg = { path = "../crates/tenferro-linalg" }
tenferro-einsum = { path = "../crates/tenferro-einsum", features = ["autodiff"] }
```

The first local build can spend several minutes compiling the faer CPU stack.
That is expected on a fresh machine.

Most direct tensor examples start by building a CPU engine and importing the
concrete tensor types:

<!-- snippet-source: docs/tutorial-code/src/bin/core_tensor_snippets.rs#eager_operations_1 -->
```rust
use tenferro_cpu::CpuEngine;
use tenferro_runtime::{TraceContext, Tensor, TypedTensor};

let mut engine = CpuEngine::serial();
```
<!-- end-snippet-source -->

Every direct tensor operation borrows an engine from the caller.
`tenferro_cpu::CpuEngine` is the standard CPU engine over the faer linear
algebra library. With the `cuda` feature, the same concrete and eager APIs can
execute supported operations on a `tenferro_gpu::cuda::CudaEngine` when tensors
are explicitly uploaded to that device.

`EagerRuntime` owns the optional gradient slots for tracked eager tensors; it
stores no engine, so every eager session borrows the one you pass in. Untracked eager tensors are forward-only. If you share
one context across multiple tracked tensors, their gradients accumulate into
the same state and you can reset them together with `clear_grads()`.

Most broad non-AD concrete operations are available as `CoreOps` methods
(`tenferro_runtime::CoreOps`) on any `Tensor`, `TensorRead`, `TypedTensor<T>`,
or `TypedTensorView<T>`, with the caller's `&mut dyn Engine` as the last
argument: `x.exp(exec)`, `a.matmul(&b, exec)`. AD workflows use the eager
session surface instead. `TypedTensor<T, R>` is the first layer to consider
when you want compile-time dtype safety, optional rank typing, or typed data
that may live on the host or in backend-owned storage. Einsum is provided by
the separate `tenferro-einsum` standard extension.

Eager core operations are inherent methods of the `EagerSession` borrowed with
`runtime.with_eager_session(&mut engine, |session| { ... })`; the session is
the receiver and the tensors are arguments, for example
`session.add(&a, &b)` or `session.reduce_sum(&x, Some(&[-1]))`. Tracked
inputs record the operation for `backward()` and functional transforms. The
differentiable surface most loss functions need:

| Category | `EagerSession` methods |
| --- | --- |
| Elementwise | `add`, `mul`, `neg`, `sign`, `exp`, `log` |
| Reduction | `reduce_sum`, `reduce_max`, `reduce_min`, `reduce_prod` |
| Matrix products | `dot_general`, `dot_general_with` (per-operand conjugation through `DotGeneralOptions`), `matmul` |
| Shape/layout | `reshape`, `transpose`, `broadcast_in_dim`, `slice`, `dynamic_slice`, `pad`, `reverse`, `tril`, `triu`, `concatenate`, `stack` |
| Indexing | `gather`, `scatter`, index-selection and diagonal routes |
| DType | checked `convert`, explicit lossy `cast` |

Axis arguments are `isize`; a negative axis counts from the end. The generated
[operation registry](../spec/generated/api-ops.md#core) lists which core
operations exist on each layer.

`with_eager_session` returns the callback's own `Result<T, E>`, so one `?`
propagates both a failed engine registration and a failed operation. `E` is any
error type with `From<tenferro_ad::Error>`, including your own application
error.
A callback that ends in `Ok(..)` after `?`-chains may need the error type named,
for example `Ok::<_, tenferro_ad::Error>(value)`, and a tensor-level error from
`session.engine()` converts with
`.map_err(tenferro_ad::Error::from)`.

Operation-family crates add eager traits on the same session. For example,
`tenferro_linalg::EagerLinalgOps` provides `session.svd(&x)` and the other
linalg operations, `tenferro_fft::EagerFftOps` provides `session.fft(&x)`, and
`tenferro_einsum::EagerEinsumOps` provides eager einsum and `tensordot`, for
example
`ctx.with_eager_session(&mut engine, |s| s.einsum(&[&a, &b], "ij,jk->ik"))?`.

For CUDA, eager means the operation is submitted immediately. It does not mean
the host waits after every GPU kernel. Host synchronization happens at
download/read boundaries, through `EagerRuntime::synchronize()`, or inside
operations that must inspect device-side status. See
[Execution Models](execution-models.md) and
[Devices and GPU](devices-and-gpu.md).

## Creating tensors

<!-- snippet-source: docs/tutorial-code/src/bin/core_tensor_snippets.rs#eager_operations_2 -->
```rust
use tenferro_runtime::{Tensor, TypedTensor};
use tenferro_tensor::Rank;

// Dynamic dtype (`Tensor`)
let a = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])?;

// Static dtype (`TypedTensor`)
let b = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
let ranked: TypedTensor<f64, Rank<2>> = match b.try_into_rank::<2>() {
    Ok(ranked) => ranked,
    Err(err) => panic!("unexpected rank mismatch: {err}"),
};
assert_eq!(ranked.shape(), &[2, 3]);
let b_bad = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
assert!(b_bad.try_into_rank::<3>().is_err());

// Convert between layers for a specific dtype.
let b_for_tensor = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
let c = Tensor::from_typed(b_for_tensor);
assert_eq!(c.shape(), &[2, 3]);
```
<!-- end-snippet-source -->

The flat buffers above are in column-major order, so a `[2, 3]` tensor stores
its columns as `[1, 2]`, `[3, 4]`, and `[5, 6]`.
Owned tensors stay compact column-major. Metadata-only strided views live on
`TypedTensorView` and `TypedTensorViewMut`; operations that require compact
storage may copy a view into compact storage on the same device, but they do
not silently upload CPU tensors or download CUDA tensors.

To bring a `Tensor` into an eager runtime as an untracked constant, pick by
where the data lives:

| Source | Call | Transfer |
|---|---|---|
| Already on the runtime's backend (any tensor on a CPU runtime; a device tensor on CUDA/WebGPU) | `session.constant_from(tensor)` or `runtime.constant_from(tensor)` | none |
| Host data, runtime possibly on a device | `session.constant_from_host(tensor)` | uploads to the backend (a host copy on CPU) |

On a CPU runtime both give the same result. Trainable leaves use
`variable_from` with the same residency rule as `constant_from`.

`Tensor` and the dynamic `TypedTensor` do not implement `Clone`. When two
branches need the same constant, call `tensor.duplicate()?`: it returns an
independent copy, and fails for device-only storage instead of silently
downloading it. To share one tensor without copying, wrap it in `Arc<Tensor>`
or pass views.

## Mixing eager and concrete work

There are two ways to run eager code, and both take the engine you hold:

| Entry | Use for |
|---|---|
| `EagerRuntime::with_eager_session(&mut engine, \|s\| ...)` | The only entry: eager operations on `s`, including AD (`s.backward`, `s.grad`, `s.vjp`, `s.jvp`) and value extraction (`s.duplicate_value`, `s.detach`), plus AD-free `Tensor` operations through `s.engine()`. |
| `EagerRuntime::{constant_from, variable_from}` | Engine-free import of an owned host tensor as a leaf. A device-resident or view input needs the session constructors `s.constant_from`, `s.constant_from_host`, `s.variable_from`. |

The callback returns one `Result`; see `with_eager_session` above. Do all work
for one region inside one callback: the engine is borrowed for the callback, so
a second eager session on the same engine is a borrow error at compile time, not
a runtime reentry error. Nothing is locked and nothing deadlocks.

<!-- snippet-source: docs/tutorial-code/src/bin/core_tensor_snippets.rs#eager_operations_mixing -->
```rust
use tenferro_ad::{EagerRuntime, EagerTensor};
use tenferro_cpu::CpuEngine;
use tenferro_runtime::{Tensor, CoreOps};

let runtime = EagerRuntime::new()?;
let mut engine = CpuEngine::serial();
let x = runtime.constant_from(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, -2.0])?)?;

// Canonical entry: eager ops on a borrowed `EagerSession`.
let y = runtime.with_eager_session(&mut engine, |s| s.mul(&x, &x))?;
let values = runtime.with_eager_session(&mut engine, |s| s.duplicate_value(&y))?;

// AD-free work in the same session: the eager session lends its engine.
let total = runtime.with_eager_session(&mut engine, |s| {
    values
        .reduce_sum(None, s.engine())
        .map_err(tenferro_ad::Error::from)
})?;
assert_eq!(total.as_slice::<f64>()?, &[5.0]);

// The same engine carries concrete work outside an eager session.
let doubled = values.scale_real(2.0, &mut engine)?;
assert_eq!(doubled.as_slice::<f64>()?, &[2.0, 8.0]);

// Nested entry is not needed and not offered: the engine is already borrowed,
// so hold the outer session and issue every operation inside it.
let both = runtime.with_eager_session(&mut engine, |s| {
    let negated = s.neg(&x)?;
    let summed = s.reduce_sum(&negated, None)?;
    Ok::<_, tenferro_ad::Error>((negated, summed))
})?;
assert_eq!(both.0.value()?.as_slice::<f64>()?, &[-1.0, 2.0]);
```
<!-- end-snippet-source -->

## Materializing metadata-only views

View transforms such as transpose and slice only change layout metadata. When
an owned compact tensor is required, materialize on the active engine so
the copy uses that engine's memory pool and thread policy:

<!-- snippet-source: docs/tutorial-code/src/bin/core_tensor_snippets.rs#eager_operations_3 -->
```rust
use tenferro_cpu::CpuEngine;
use tenferro_tensor::{TensorStructural, TypedTensor};

let tensor = TypedTensor::<f64>::from_vec_col_major(
    vec![2, 3],
    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
).unwrap();
let view = tensor.as_view().transpose_view([1, 0]).unwrap();
let read = view.into_tensor_read().unwrap();
let mut engine = CpuEngine::serial();
let compact = engine.to_contiguous_read(read).unwrap();

assert_eq!(compact.shape(), &[3, 2]);
assert_eq!(compact.as_slice::<f64>().unwrap(), &[1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
```
<!-- end-snippet-source -->

The same rule applies to caller-owned destinations through the engine's
canonicalization capability. Backend-neutral tensor and view types own layout
metadata, not CPU memory reuse or Rayon configuration. Same-placement
canonicalization never performs a hidden CPU/GPU transfer.

## Arithmetic

<!-- snippet-source: docs/tutorial-code/src/bin/core_tensor_snippets.rs#eager_operations_4 -->
```rust
use tenferro_cpu::CpuEngine;
use tenferro_runtime::{Tensor, CoreOps};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0])?;
let b = Tensor::from_vec_col_major(vec![3], vec![4.0_f64, 5.0, 6.0])?;

let sum = a.add(&b, &mut engine)?;
let product = a.mul(&b, &mut engine)?;
let negated = a.neg(&mut engine)?;

assert_eq!(sum.as_slice::<f64>().unwrap(), &[5.0, 7.0, 9.0]);
assert_eq!(product.as_slice::<f64>().unwrap(), &[4.0, 10.0, 18.0]);
assert_eq!(negated.as_slice::<f64>().unwrap(), &[-1.0, -2.0, -3.0]);
```
<!-- end-snippet-source -->

## Linear algebra

<!-- snippet-source: docs/tutorial-code/src/bin/core_tensor_snippets.rs#eager_operations_5 -->
```rust
use tenferro_cpu::CpuEngine;
use tenferro_linalg::LinalgOps;
use tenferro_runtime::Tensor;

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![3, 3], vec![
    2.0_f64, 1.0, 0.0,
    1.0, 3.0, 1.0,
    0.0, 1.0, 2.0,
])?;
let b = Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0])?;
let svd = a.svd(&mut engine)?;
let qr = a.qr(&mut engine)?;
let chol = a.cholesky(&mut engine)?;
let eigh = a.eigh(&mut engine)?;
let x = a.solve(&b, &mut engine)?;
assert_eq!(svd.s.shape(), &[3]);
assert_eq!(qr.q.shape(), &[3, 3]);
assert_eq!(chol.shape(), &[3, 3]);
assert_eq!(eigh.values.shape(), &[3]);
assert_eq!(eigh.vectors.shape(), &[3, 3]);
assert_eq!(x.shape(), &[3]);
```
<!-- end-snippet-source -->

## Shape operations

<!-- snippet-source: docs/tutorial-code/src/bin/core_tensor_snippets.rs#eager_operations_6 -->
```rust
use tenferro_cpu::CpuEngine;
use tenferro_runtime::{Tensor, CoreOps};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])?;

// Transpose / Reshape / Reduce on one borrowed engine.
let at = a.transpose(&[1, 0], &mut engine)?;
let flat = a.reshape(&[6], &mut engine)?;
let col_sum = a.reduce_sum(Some(&[0]), &mut engine)?;
assert_eq!(at.shape(), &[3, 2]);
assert_eq!(flat.shape(), &[6]);
assert_eq!(col_sum.shape(), &[3]);
```
<!-- end-snippet-source -->

The `reduce_sum(Some(&[0]), session)` call removes axis `0`. For this `[2, 3]` tensor, that
means summing down each column and keeping one value per column.

## Einsum

Use `tenferro_einsum::EagerEinsumOps` on a borrowed session when working
with `EagerTensor`:
`ctx.with_eager_session(&mut engine, |s| s.einsum(&[&a, &b], "ij,jk->ik"))?`.
For traced graph execution, use `tenferro_einsum::TracedEinsumOps` on the
`TraceContext` (`trace.einsum(&[&a, &b], "ij,jk->ik")`) and install
`tenferro_einsum::extension_module` on the `Runtime`.

## Extracting data

<!-- snippet-source: docs/tutorial-code/src/bin/core_tensor_snippets.rs#eager_operations_7 -->
```rust
use tenferro_runtime::Tensor;

let t = Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0])?;
let data: &[f64] = t.as_slice::<f64>().unwrap();
assert_eq!(data, &[1.0, 2.0, 3.0]);
```
<!-- end-snippet-source -->

## Column-major storage

tenferro stores tensors in column-major (Fortran) order. For a `[2, 3]` tensor
with data `[1, 2, 3, 4, 5, 6]`, the layout is:

```text
Column 0: [1, 2]
Column 1: [3, 4]
Column 2: [5, 6]
```

This matches Fortran, Julia, and MATLAB conventions but differs from C/NumPy
row-major order.

## Eager Forward And Reverse-Mode Gradients

Eager tensors always compute the forward value immediately. Tracked eager
tensors support two AD styles:

- `s.backward(&loss)` and `s.backward_with(&loss, &seed)` are stateful
  reverse-mode APIs. They return the cotangent map and accumulate reachable
  tracked leaves into `grad()` slots.
- `s.grad`, `s.vjp` and `s.jvp` (and their `_optional` forms, which return
  `None` for an inactive input) are functional APIs. They return ordinary eager
  tensors and do not mutate `grad()` slots, so their results can feed later
  eager transforms.

All of them are `EagerSession` methods: they reject an operand or seed from
another eager runtime with `ContextMismatch` before anything is materialized or
written. Repeated `s.backward(..)` calls add to the existing gradients, and you
clear them explicitly when you want a fresh pass.

<!-- snippet-source: docs/tutorial-code/src/bin/core_tensor_snippets.rs#eager_operations_8 -->
```rust
use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
use tenferro_cpu::CpuEngine;

fn main() -> Result<(), Box<dyn std::error::Error>> {
let ctx = EagerRuntime::new()?;
let mut engine = CpuEngine::serial();
let x = ctx.variable_from(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap()).unwrap();
let y = ctx.variable_from(Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).unwrap()).unwrap();

// Each forward pass borrows the engine for its own eager session.
let loss = ctx.with_eager_session(&mut engine, |s| {
    let product = s.mul(&x, &y)?;
    s.reduce_sum(&product, Some(&[0]))
})?;
ctx.with_eager_session(&mut engine, |s| s.backward(&loss))?;
assert_eq!(x.grad().unwrap().unwrap().as_slice::<f64>().unwrap(), &[3.0, 4.0]);

let loss = ctx.with_eager_session(&mut engine, |s| {
    let product = s.mul(&x, &y)?;
    s.reduce_sum(&product, Some(&[0]))
})?;
ctx.with_eager_session(&mut engine, |s| s.backward(&loss))?;
assert_eq!(x.grad().unwrap().unwrap().as_slice::<f64>().unwrap(), &[6.0, 8.0]);

x.clear_grad().unwrap();
assert!(x.grad().unwrap().is_none());

let loss = ctx.with_eager_session(&mut engine, |s| {
    let product = s.mul(&x, &y)?;
    s.reduce_sum(&product, Some(&[0]))
})?;
ctx.with_eager_session(&mut engine, |s| s.backward(&loss))?;
assert_eq!(x.grad().unwrap().unwrap().as_slice::<f64>().unwrap(), &[3.0, 4.0]);

ctx.clear_grads().unwrap();
assert!(x.grad().unwrap().is_none());
assert!(y.grad().unwrap().is_none());
Ok(())
}
```
<!-- end-snippet-source -->

Use `backward_with` when the output is not scalar or when reverse mode should
start from an explicit cotangent seed:

<!-- snippet-source: docs/tutorial-code/src/bin/core_tensor_snippets.rs#eager_operations_9 -->
```rust
use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
use tenferro_cpu::CpuEngine;

fn main() -> Result<(), Box<dyn std::error::Error>> {
let ctx = EagerRuntime::new()?;
let mut engine = CpuEngine::serial();
let x = ctx.variable_from(Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 3.0]).unwrap()).unwrap();
let seed = ctx.constant_from(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap()).unwrap();

let y = ctx.with_eager_session(&mut engine, |s| s.mul(&x, &x))?;
ctx.with_eager_session(&mut engine, |s| s.backward_with(&y, &seed)).unwrap();
assert_eq!(x.grad().unwrap().unwrap().as_slice::<f64>().unwrap(), &[4.0, 12.0]);
Ok(())
}
```
<!-- end-snippet-source -->

Functional eager transforms return tensors instead of updating gradient slots:

<!-- snippet-source: docs/tutorial-code/src/bin/core_tensor_snippets.rs#eager_operations_10 -->
```rust
use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
use tenferro_cpu::CpuEngine;

fn main() -> Result<(), Box<dyn std::error::Error>> {
let ctx = EagerRuntime::new()?;
let mut engine = CpuEngine::serial();
let x = ctx.variable_from(Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 3.0]).unwrap()).unwrap();
let y = ctx.with_eager_session(&mut engine, |s| s.mul(&x, &x))?;
let seed = ctx.constant_from(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 1.0]).unwrap()).unwrap();
let tangent = ctx.constant_from(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 1.0]).unwrap()).unwrap();

let vjp = ctx.with_eager_session(&mut engine, |s| s.vjp(&y, &x, &seed)).unwrap();
let jvp = ctx.with_eager_session(&mut engine, |s| s.jvp(&y, &x, &tangent)).unwrap();
let vjp_tensor = ctx.with_eager_session(&mut engine, |s| s.duplicate_value(&vjp)).unwrap();
assert_eq!(vjp_tensor.as_slice::<f64>().unwrap(), &[4.0, 6.0]);
let jvp_tensor = ctx.with_eager_session(&mut engine, |s| s.duplicate_value(&jvp)).unwrap();
assert_eq!(jvp_tensor.as_slice::<f64>().unwrap(), &[4.0, 6.0]);
assert!(x.grad().unwrap().is_none());
Ok(())
}
```
<!-- end-snippet-source -->

Because functional derivatives are eager tensors, Hessian-vector products can
be written as `jvp(grad(f))`:

<!-- snippet-source: docs/tutorial-code/src/bin/core_tensor_snippets.rs#eager_operations_11 -->
```rust
use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
use tenferro_cpu::CpuEngine;

fn main() -> Result<(), Box<dyn std::error::Error>> {
let ctx = EagerRuntime::new()?;
let mut engine = CpuEngine::serial();
let x = ctx.variable_from(Tensor::from_vec_col_major(vec![], vec![3.0_f64]).unwrap()).unwrap();
let tangent = ctx.constant_from(Tensor::from_vec_col_major(vec![], vec![1.0_f64]).unwrap()).unwrap();

let loss = ctx.with_eager_session(&mut engine, |s| {
    let square = s.mul(&x, &x)?;
    s.mul(&square, &x)
})?;
let grad = ctx.with_eager_session(&mut engine, |s| s.grad(&loss, &x)).unwrap();
let hvp = ctx.with_eager_session(&mut engine, |s| s.jvp(&grad, &x, &tangent)).unwrap();

let grad_tensor = ctx.with_eager_session(&mut engine, |s| s.duplicate_value(&grad)).unwrap();
assert_eq!(grad_tensor.as_slice::<f64>().unwrap(), &[27.0]);
let hvp_tensor = ctx.with_eager_session(&mut engine, |s| s.duplicate_value(&hvp)).unwrap();
assert_eq!(hvp_tensor.as_slice::<f64>().unwrap(), &[18.0]);
Ok(())
}
```
<!-- end-snippet-source -->

Wrap updates, metric computations, and other non-differentiated eager work in
`no_grad` when they should not be recorded:

<!-- snippet-source: docs/tutorial-code/src/bin/core_tensor_snippets.rs#eager_operations_12 -->
```rust
use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
use tenferro_cpu::CpuEngine;

fn main() -> Result<(), Box<dyn std::error::Error>> {
let ctx = EagerRuntime::new()?;
let mut engine = CpuEngine::serial();
let x = ctx.variable_from(Tensor::from_vec_col_major(vec![1], vec![3.0_f64]).unwrap()).unwrap();
let y = ctx.with_eager_session(&mut engine, |s| {
    let _guard = ctx.no_grad();
    s.mul(&x, &x)
})?;
assert!(!y.tracks_grad());
Ok(())
}
```
<!-- end-snippet-source -->

`matmul` participates in the same eager reverse-mode workflow:

<!-- snippet-source: docs/tutorial-code/src/bin/core_tensor_snippets.rs#eager_operations_13 -->
```rust
use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
use tenferro_cpu::CpuEngine;

fn main() -> Result<(), Box<dyn std::error::Error>> {
let ctx = EagerRuntime::new()?;
let mut engine = CpuEngine::serial();
let a = ctx.variable_from(Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap()).unwrap();
let x = ctx.variable_from(Tensor::from_vec_col_major(vec![2, 1], vec![5.0_f64, 6.0]).unwrap()).unwrap();

let y = ctx.with_eager_session(&mut engine, |session| session.matmul(&a, &x))?;
let y_tensor = ctx.with_eager_session(&mut engine, |s| s.duplicate_value(&y)).unwrap();
assert_eq!(y_tensor.as_slice::<f64>().unwrap(), &[23.0, 34.0]);

let loss = ctx.with_eager_session(&mut engine, |s| {
    let squared = s.mul(&y, &y)?;
    s.reduce_sum(&squared, Some(&[0, 1]))
})?;
let loss_tensor = ctx.with_eager_session(&mut engine, |s| s.duplicate_value(&loss)).unwrap();
assert_eq!(loss_tensor.as_slice::<f64>().unwrap(), &[1685.0]);

ctx.with_eager_session(&mut engine, |s| s.backward(&loss)).unwrap();
assert_eq!(x.grad().unwrap().unwrap().as_slice::<f64>().unwrap(), &[182.0, 410.0]);
Ok(())
}
```
<!-- end-snippet-source -->

## When To Use Each Immediate Layer

| Scenario | Recommended |
|----------|-------------|
| Fixed scalar type and no autodiff | `TypedTensor<T, R>` |
| Dynamic dtype and no autodiff | `Tensor` + an engine |
| Data preprocessing | `Tensor` + an engine |
| Tight inner loops | Direct/eager execution |
| Exploratory computation | Direct/eager execution |
| Immediate forward execution through one runtime | `EagerTensor` |
| Need reverse-mode gradients with gradient slots | tracked `EagerTensor` variables + `backward()` / `backward_with(seed)` |
| Need functional eager `grad` / `vjp` / `jvp` / HVP composition | `EagerSession` functional APIs |
| Need compiled `grad` / `vjp` / `jvp` / HVP composition | Lazy traced (`TracedTensor` + `GraphCompiler` + `Runtime`) |
| CUDA execution for supported operations | Eager (`Tensor` / `EagerTensor`) or lazy traced (`TracedTensor` + `Runtime`) with explicit upload/download |
