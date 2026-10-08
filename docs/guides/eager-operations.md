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

The first local build can spend several minutes compiling the default
`native` stack. That is expected on a fresh machine.

Most direct tensor examples start by importing the CPU backend and concrete
tensor types:

<!-- snippet-source: docs/tutorial-code/src/bin/core_tensor_snippets.rs#eager_operations_1 -->
```rust
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{Tensor, TypedTensor};

let mut backend = CpuBackend::new();
```
<!-- end-snippet-source -->

Every direct tensor operation requires a backend context. `CpuBackend` is the
standard CPU backend using the faer linear algebra library. With the `cuda`
feature, the same concrete and eager APIs can execute supported operations
on the CUDA backend when tensors are explicitly placed on the GPU.

`EagerRuntime` owns the eager backend and the optional gradient slots for
tracked eager tensors. Untracked eager tensors are forward-only. If you share
one context across multiple tracked tensors, their gradients accumulate into
the same state and you can reset them together with `clear_grads()`.

Most broad non-AD concrete operations are available as `TensorSessionOpsExt` /
`TypedTensorSessionOpsExt` methods inside an explicit backend session. AD workflows use the
`EagerTensor` method surface instead. `TypedTensor<T, R>` is the first layer to
consider when you want compile-time dtype safety, optional rank typing, or typed
data that may live on the host or in backend-owned storage. Einsum is provided
by the separate `tenferro-einsum` standard extension.

Tracked `EagerTensor` values support the differentiable method surface most
loss functions need:

| Category | `EagerTensor` methods |
| --- | --- |
| Elementwise | `add`, `mul`, `neg` |
| Reduction | `reduce_sum`, `reduce_max`, `reduce_min` |
| Matrix products | `dot_general` |
| Shape/layout | `reshape`, `transpose`, `broadcast_in_dim` |
| DType | checked `convert`, explicit lossy `cast` |

For operations that have moved to an explicit boundary, borrow a runtime-bound
session with `runtime.with_eager_session(|session| { ... })`. This includes
`matmul`, `dot_general_with_conj`, `scatter`, `sign`, `exp`, `log`, `reduce_prod`,
`slice`, `dynamic_slice`, `pad`, `reverse`, `tril`, `triu`, `concatenate`,
`stack`, `gather`, and the borrowed index-selection/diagonal routes. Do not call a remaining implicit
`EagerTensor` operation while holding a borrowed session.

`with_eager_session` returns the callback's own `Result<T, E>`, so one `?`
propagates both a failed session entry and a failed operation. `E` is any error
type with `From<tenferro_ad::Error>`, including your own application error.
A callback that ends in `Ok(..)` after `?`-chains may need the error type named,
for example `Ok::<_, tenferro_ad::Error>(value)`, and a tensor-level error from
`session.backend_session()` converts with `.map_err(tenferro_ad::Error::from)`.

Operation-family crates add eager extension traits. For example,
`tenferro_linalg::EagerTensorLinalgExt` owns linalg eager methods and
`tenferro_einsum::EagerSessionEinsumExt` owns eager einsum and `tensordot` on a
borrowed session, for example
`ctx.with_eager_session(|s| s.einsum(&[&a, &b], "ij,jk->ik"))?`.

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

There are three ways to reach a backend session from eager code:

| Entry | Use for |
|---|---|
| `EagerRuntime::with_eager_session(\|s\| ...)` | The canonical entry: eager operations on `s`, plus AD-free `Tensor` operations through `s.backend_session()`. |
| `EagerRuntime::on_cpu(placement)?.with_eager_session(\|session\| ...)` | Core `Tensor` operations on the runtime's CPU backend with an explicit CPU placement. |
| `EagerRuntime::with_execution_session(\|session\| ...)` | Raw backend access for extension code; no eager operations. |

The callback must be `Send` (the CPU backend may run it on a pool thread) and
returns one `Result`; see `with_eager_session` above. Do all work for one region
inside one callback: entering the same runtime again from inside the callback,
or calling an operation that opens its own session, returns a typed
`SessionEntry` error (`Reentered`) instead of running, and never deadlocks.

<!-- snippet-source: docs/tutorial-code/src/bin/core_tensor_snippets.rs#eager_operations_mixing -->
```rust
use tenferro_ad::{EagerRuntime, EagerTensor};
use tenferro_cpu::{CpuBackend, CpuPlacement};
use tenferro_runtime::{Tensor, TensorSessionOpsExt};

let runtime = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
let x = EagerTensor::from_tensor_in(
    Tensor::from_vec_col_major(vec![2], vec![1.0_f64, -2.0])?,
    runtime.clone(),
)?;

// Canonical entry: eager ops on a borrowed `EagerSession`.
let y = runtime.with_eager_session(|s| s.mul(&x, &x))?;
let values = y.to_tensor()?;

// AD-free work in the same session: the eager session lends its backend session.
let total = runtime.with_eager_session(|s| {
    values
        .reduce_sum(None, s.backend_session())
        .map_err(tenferro_ad::Error::from)
})?;
assert_eq!(total.as_slice::<f64>()?, &[5.0]);

// CPU placement bridge: core ops on the runtime's CPU backend session.
let mut cpu = runtime.on_cpu(CpuPlacement::Auto)?;
let doubled = cpu.with_eager_session(|session| {
    values
        .scale_real(2.0, session)
        .map_err(tenferro_ad::Error::from)
})?;
assert_eq!(doubled.as_slice::<f64>()?, &[2.0, 8.0]);

// A nested entry into the same runtime is rejected, not deadlocked.
let nested = runtime.with_eager_session(|_| {
    Ok::<_, tenferro_ad::Error>(runtime.with_eager_session(|s| s.neg(&x)).is_err())
})?;
assert!(nested);
```
<!-- end-snippet-source -->

## Materializing metadata-only views

View transforms such as transpose and slice only change layout metadata. When
an owned compact tensor is required, materialize through the active backend so
the copy uses that backend's memory pool and thread policy:

<!-- snippet-source: docs/tutorial-code/src/bin/core_tensor_snippets.rs#eager_operations_3 -->
```rust
use tenferro_cpu::CpuBackend;
use tenferro_tensor::{BackendSessionHost, TypedTensor};

let tensor = TypedTensor::<f64>::from_vec_col_major(
    vec![2, 3],
    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
).unwrap();
let view = tensor.as_view().transpose_view([1, 0]).unwrap();
let read = view.into_tensor_read().unwrap();
let mut backend = CpuBackend::new();
let compact = backend
    .with_backend_session(|session| session.to_contiguous_read(read))
    .unwrap()
    .unwrap();

assert_eq!(compact.shape(), &[3, 2]);
assert_eq!(compact.as_slice::<f64>().unwrap(), &[1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
```
<!-- end-snippet-source -->

The same rule applies to caller-owned destinations through the backend
canonicalization capability. Backend-neutral tensor and view types own layout
metadata, not CPU memory reuse or Rayon configuration. Same-placement
canonicalization never performs a hidden CPU/GPU transfer.

## Arithmetic

<!-- snippet-source: docs/tutorial-code/src/bin/core_tensor_snippets.rs#eager_operations_4 -->
```rust
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{Tensor, TensorSessionOpsExt};
use tenferro_tensor::BackendSessionHost;

let mut backend = CpuBackend::new();
let a = Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0])?;
let b = Tensor::from_vec_col_major(vec![3], vec![4.0_f64, 5.0, 6.0])?;

let (sum, product, negated) = backend.with_backend_session(|session| {
    let sum = a.add(&b, session).unwrap();
    let product = a.mul(&b, session).unwrap();
    let negated = a.neg(session).unwrap();
    (sum, product, negated)
})?;

assert_eq!(sum.as_slice::<f64>().unwrap(), &[5.0, 7.0, 9.0]);
assert_eq!(product.as_slice::<f64>().unwrap(), &[4.0, 10.0, 18.0]);
assert_eq!(negated.as_slice::<f64>().unwrap(), &[-1.0, -2.0, -3.0]);
```
<!-- end-snippet-source -->

## Linear algebra

<!-- snippet-source: docs/tutorial-code/src/bin/core_tensor_snippets.rs#eager_operations_5 -->
```rust
use tenferro_cpu::CpuBackend;
use tenferro_linalg::TensorLinalgExt;
use tenferro_runtime::{BackendSessionHost, Tensor};

let mut backend = CpuBackend::new();
let a = Tensor::from_vec_col_major(vec![3, 3], vec![
    2.0_f64, 1.0, 0.0,
    1.0, 3.0, 1.0,
    0.0, 1.0, 2.0,
])?;
let b = Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0])?;
backend.with_backend_session(|session| {
    let svd = a.svd(session).unwrap();
    let qr = a.qr(session).unwrap();
    let chol = a.cholesky(session).unwrap();
    let eigh = a.eigh(session).unwrap();
    let x = a.solve(&b, session).unwrap();
    assert_eq!(svd.1.shape(), &[3]);
    assert_eq!(qr.0.shape(), &[3, 3]);
    assert_eq!(chol.shape(), &[3, 3]);
    assert_eq!(eigh.0.shape(), &[3]);
    assert_eq!(eigh.1.shape(), &[3, 3]);
    assert_eq!(x.shape(), &[3]);
})?;
```
<!-- end-snippet-source -->

## Shape operations

<!-- snippet-source: docs/tutorial-code/src/bin/core_tensor_snippets.rs#eager_operations_6 -->
```rust
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{Tensor, TensorSessionOpsExt};
use tenferro_tensor::BackendSessionHost;

let mut backend = CpuBackend::new();
let a = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])?;

// Transpose / Reshape / Reduce in one backend session.
let (at, flat, col_sum) = backend.with_backend_session(|session| {
    let at = a.transpose(&[1, 0], session).unwrap();
    let flat = a.reshape(&[6], session).unwrap();
    let col_sum = a.reduce_sum(Some(&[0]), session).unwrap();
    (at, flat, col_sum)
})?;
assert_eq!(at.shape(), &[3, 2]);
assert_eq!(flat.shape(), &[6]);
assert_eq!(col_sum.shape(), &[3]);
```
<!-- end-snippet-source -->

The `reduce_sum(Some(&[0]), session)` call removes axis `0`. For this `[2, 3]` tensor, that
means summing down each column and keeping one value per column.

## Einsum

Use `tenferro_einsum::EagerSessionEinsumExt` on a borrowed session when working
with `EagerTensor`: `ctx.with_eager_session(|s| s.einsum(&[&a, &b], "ij,jk->ik"))?`.
For traced graph execution, use `tenferro_einsum::TraceContextEinsumExt` and
install `tenferro_einsum::extension_module` on the `Runtime`.

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

- `backward()` and `backward_with(seed)` are stateful reverse-mode APIs. They
  return the cotangent map and accumulate reachable tracked leaves into
  `grad()` slots.
- `EagerRuntime::grad`, `EagerRuntime::vjp`, and `EagerRuntime::jvp` are
  functional APIs. They return ordinary eager tensors and do not mutate
  `grad()` slots, so their results can feed later eager transforms.

Repeated `backward()` calls add to the existing gradients, and you clear them
explicitly when you want a fresh pass.

<!-- snippet-source: docs/tutorial-code/src/bin/core_tensor_snippets.rs#eager_operations_8 -->
```rust
use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
use tenferro_cpu::CpuBackend;

fn main() -> Result<(), Box<dyn std::error::Error>> {
let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
let x = EagerTensor::requires_grad_in(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap(), ctx.clone()).unwrap();
let y = EagerTensor::requires_grad_in(Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).unwrap(), ctx.clone()).unwrap();

let make_loss = || ctx.with_eager_session(|s| {
    let product = s.mul(&x, &y)?;
    s.reduce_sum(&product, Some(&[0]))
}).unwrap();
let loss = make_loss();
loss.backward().unwrap();
assert_eq!(x.grad().unwrap().unwrap().as_slice::<f64>().unwrap(), &[3.0, 4.0]);

let loss = make_loss();
loss.backward().unwrap();
assert_eq!(x.grad().unwrap().unwrap().as_slice::<f64>().unwrap(), &[6.0, 8.0]);

x.clear_grad().unwrap();
assert!(x.grad().unwrap().is_none());

let loss = make_loss();
loss.backward().unwrap();
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
use tenferro_cpu::CpuBackend;

fn main() -> Result<(), Box<dyn std::error::Error>> {
let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
let x = EagerTensor::requires_grad_in(
    Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 3.0]).unwrap(),
    ctx.clone(),
).unwrap();
let seed = EagerTensor::from_tensor_in(
    Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap(),
    ctx,
).unwrap();

let y = x.runtime().with_eager_session(|s| s.mul(&x, &x))?;
y.backward_with(&seed).unwrap();
assert_eq!(x.grad().unwrap().unwrap().as_slice::<f64>().unwrap(), &[4.0, 12.0]);
Ok(())
}
```
<!-- end-snippet-source -->

Functional eager transforms return tensors instead of updating gradient slots:

<!-- snippet-source: docs/tutorial-code/src/bin/core_tensor_snippets.rs#eager_operations_10 -->
```rust
use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
use tenferro_cpu::CpuBackend;

fn main() -> Result<(), Box<dyn std::error::Error>> {
let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
let x = EagerTensor::requires_grad_in(
    Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 3.0]).unwrap(),
    ctx.clone(),
).unwrap();
let y = ctx.with_eager_session(|s| s.mul(&x, &x))?;
let seed = EagerTensor::from_tensor_in(
    Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 1.0]).unwrap(),
    ctx.clone(),
).unwrap();
let tangent = EagerTensor::from_tensor_in(
    Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 1.0]).unwrap(),
    ctx.clone(),
).unwrap();

let vjp = ctx.vjp(&y, &x, &seed).unwrap();
let jvp = ctx.jvp(&y, &x, &tangent).unwrap();
let vjp_tensor = vjp.to_tensor().unwrap();
assert_eq!(vjp_tensor.as_slice::<f64>().unwrap(), &[4.0, 6.0]);
let jvp_tensor = jvp.to_tensor().unwrap();
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
use tenferro_cpu::CpuBackend;

fn main() -> Result<(), Box<dyn std::error::Error>> {
let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
let x = EagerTensor::requires_grad_in(
    Tensor::from_vec_col_major(vec![], vec![3.0_f64]).unwrap(),
    ctx.clone(),
).unwrap();
let tangent = EagerTensor::from_tensor_in(
    Tensor::from_vec_col_major(vec![], vec![1.0_f64]).unwrap(),
    ctx.clone(),
).unwrap();

let loss = ctx.with_eager_session(|s| {
    let square = s.mul(&x, &x)?;
    s.mul(&square, &x)
})?;
let grad = ctx.grad(&loss, &x).unwrap();
let hvp = ctx.jvp(&grad, &x, &tangent).unwrap();

let grad_tensor = grad.to_tensor().unwrap();
assert_eq!(grad_tensor.as_slice::<f64>().unwrap(), &[27.0]);
let hvp_tensor = hvp.to_tensor().unwrap();
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
use tenferro_cpu::CpuBackend;

fn main() -> Result<(), Box<dyn std::error::Error>> {
let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
let x = EagerTensor::requires_grad_in(
    Tensor::from_vec_col_major(vec![1], vec![3.0_f64]).unwrap(),
    ctx.clone(),
).unwrap();
let y = ctx.with_eager_session(|s| {
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
use tenferro_cpu::CpuBackend;

fn main() -> Result<(), Box<dyn std::error::Error>> {
let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
let a = EagerTensor::requires_grad_in(
    Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap(),
    ctx.clone(),
).unwrap();
let x = EagerTensor::requires_grad_in(
    Tensor::from_vec_col_major(vec![2, 1], vec![5.0_f64, 6.0]).unwrap(),
    ctx.clone(),
).unwrap();

let y = ctx.with_eager_session(|session| session.matmul(&a, &x))?;
let y_tensor = y.to_tensor().unwrap();
assert_eq!(y_tensor.as_slice::<f64>().unwrap(), &[23.0, 34.0]);

let loss = ctx.with_eager_session(|s| {
    let squared = s.mul(&y, &y)?;
    s.reduce_sum(&squared, Some(&[0, 1]))
})?;
let loss_tensor = loss.to_tensor().unwrap();
assert_eq!(loss_tensor.as_slice::<f64>().unwrap(), &[1685.0]);

loss.backward().unwrap();
assert_eq!(x.grad().unwrap().unwrap().as_slice::<f64>().unwrap(), &[182.0, 410.0]);
Ok(())
}
```
<!-- end-snippet-source -->

## When To Use Each Immediate Layer

| Scenario | Recommended |
|----------|-------------|
| Fixed scalar type and no autodiff | `TypedTensor<T, R>` |
| Dynamic dtype and no autodiff | `Tensor` + a backend |
| Data preprocessing | `Tensor` + a backend |
| Tight inner loops | Direct/eager execution |
| Exploratory computation | Direct/eager execution |
| Immediate forward execution through one runtime | `EagerTensor` |
| Need reverse-mode gradients with gradient slots | tracked `EagerTensor` variables + `backward()` / `backward_with(seed)` |
| Need functional eager `grad` / `vjp` / `jvp` / HVP composition | `EagerRuntime` functional APIs |
| Need compiled `grad` / `vjp` / `jvp` / HVP composition | Lazy traced (`TracedTensor` + `GraphCompiler` + `Runtime`) |
| CUDA execution for supported operations | Eager (`Tensor` / `EagerTensor`) or lazy traced (`TracedTensor` + `Runtime`) with explicit upload/download |
