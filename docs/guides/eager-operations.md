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
`cpu-faer` stack. That is expected on a fresh machine.

Most direct tensor examples start by importing the CPU backend and concrete
tensor types:

```rust
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{Tensor, TypedTensor};

let mut backend = CpuBackend::new();
```

Every direct tensor operation requires a backend context. `CpuBackend` is the
standard CPU backend using the faer linear algebra library. With the `cuda`
feature, the same concrete and eager APIs can execute supported operations
on the CUDA backend when tensors are explicitly placed on the GPU.

`EagerRuntime` owns the eager backend and the optional gradient slots for
tracked eager tensors. Untracked eager tensors are forward-only. If you share
one context across multiple tracked tensors, their gradients accumulate into
the same state and you can reset them together with `clear_grads()`.

Most broad non-AD concrete operations are available as `TensorOpsExt` /
`TypedTensorOpsExt` methods with an explicit backend. AD workflows use the
`EagerTensor` method surface instead. `TypedTensor<T, R>` is the first layer to
consider when you want compile-time dtype safety, optional rank typing, or typed
data that may live on the host or in backend-owned storage. Einsum is provided
by the separate `tenferro-einsum` standard extension.

Tracked `EagerTensor` values support the differentiable method surface most
loss functions need:

| Category | `EagerTensor` methods |
| --- | --- |
| Elementwise | `add`, `mul`, `neg`, `exp` |
| Reduction | `reduce_sum`, `reduce_prod`, `reduce_max`, `reduce_min` |
| Matrix products | `matmul`, `dot_general`, `dot_general_with_conj` |
| Shape/layout | `reshape`, `transpose`, `broadcast_in_dim`, `slice`, `pad`, `reverse`, `concatenate` |
| Indexing/diagonal | `gather`, `scatter`, `dynamic_slice`, `extract_diag`, `embed_diag`, `tril`, `triu` |
| DType | checked `convert`, explicit lossy `cast` |

Operation-family crates add eager extension traits. For example,
`tenferro_linalg::EagerTensorLinalgExt` owns linalg eager methods and
`tenferro_einsum::EagerEinsumExt` owns eager einsum on input slices/arrays.

For CUDA, eager means the operation is submitted immediately. It does not mean
the host waits after every GPU kernel. Host synchronization happens at
download/read boundaries, through `EagerRuntime::synchronize()`, or inside
operations that must inspect device-side status. See
[Execution Models](execution-models.md) and
[Devices and GPU](devices-and-gpu.md).

## Creating tensors

```rust
use tenferro_runtime::{Tensor, TypedTensor};
use tenferro_tensor::Rank;

// Dynamic dtype (`Tensor`)
let a = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);

// Static dtype (`TypedTensor`)
let b = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
let ranked: TypedTensor<f64, Rank<2>> = match b.clone().try_into_rank::<2>() {
    Ok(ranked) => ranked,
    Err(err) => panic!("unexpected rank mismatch: {err}"),
};
assert_eq!(ranked.shape(), &[2, 3]);
assert!(b.clone().try_into_rank::<3>().is_err());

// Convert between layers for a specific dtype.
let c = Tensor::F64(b.clone());
assert_eq!(c.shape(), &[2, 3]);
```

The flat buffers above are in column-major order, so a `[2, 3]` tensor stores
its columns as `[1, 2]`, `[3, 4]`, and `[5, 6]`.
Owned tensors stay compact column-major. Metadata-only strided views live on
`TypedTensorView` and `TypedTensorViewMut`; operations that require compact
storage may copy a view into compact storage on the same device, but they do
not silently upload CPU tensors or download CUDA tensors.

## Materializing metadata-only views

View transforms such as transpose and slice only change layout metadata. When
an owned compact tensor is required, materialize through the active backend so
the copy uses that backend's memory pool and thread policy:

```rust
use tenferro_cpu::CpuBackend;
use tenferro_tensor::{TensorViewCanonicalization, TypedTensor};

let tensor = TypedTensor::<f64>::from_vec_col_major(
    vec![2, 3],
    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
).unwrap();
let view = tensor.as_view().transpose_view([1, 0]).unwrap();
let mut backend = CpuBackend::new();
let compact = backend.to_contiguous(&view).unwrap();

assert_eq!(compact.shape(), &[3, 2]);
assert_eq!(compact.as_slice().unwrap(), &[1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
```

The same rule applies to caller-owned destinations through the backend
canonicalization capability. Backend-neutral tensor and view types own layout
metadata, not CPU memory reuse or Rayon configuration. Same-placement
canonicalization never performs a hidden CPU/GPU transfer.

## Arithmetic

```rust
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{Tensor, TensorOpsExt};

let mut backend = CpuBackend::new();
let a = Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]);
let b = Tensor::from_vec_col_major(vec![3], vec![4.0_f64, 5.0, 6.0]);

let sum = a.add(&b, &mut backend).unwrap();
let product = a.mul(&b, &mut backend).unwrap();
let negated = a.neg(&mut backend).unwrap();

assert_eq!(sum.as_slice::<f64>().unwrap(), &[5.0, 7.0, 9.0]);
assert_eq!(product.as_slice::<f64>().unwrap(), &[4.0, 10.0, 18.0]);
assert_eq!(negated.as_slice::<f64>().unwrap(), &[-1.0, -2.0, -3.0]);
```

## Linear algebra

```rust
use tenferro_linalg::LinalgBackend;
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{Tensor, TensorOpsExt};

let mut backend = CpuBackend::new();
let a = Tensor::from_vec_col_major(vec![3, 3], vec![
    2.0_f64, 1.0, 0.0,
    1.0, 3.0, 1.0,
    0.0, 1.0, 2.0,
]);

// SVD
let svd = LinalgBackend::svd(&mut backend, &a).unwrap();

// QR
let qr = LinalgBackend::qr(&mut backend, &a).unwrap();

// Cholesky (for positive definite matrices)
let chol = LinalgBackend::cholesky(&mut backend, &a).unwrap();

// Eigendecomposition (symmetric)
let eigh = LinalgBackend::eigh(&mut backend, &a).unwrap();

// Solve Ax = b
let b = Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]);
let x = LinalgBackend::solve(&mut backend, &a, &b).unwrap();

let s = &svd[1];
assert_eq!(s.shape(), &[3]);
assert_eq!(qr[0].shape(), &[3, 3]);
assert_eq!(chol.shape(), &[3, 3]);
let eigenvalues = &eigh[0];
let eigenvectors = &eigh[1];
assert_eq!(eigenvalues.shape(), &[3]);
assert_eq!(eigenvectors.shape(), &[3, 3]);
assert_eq!(x.shape(), &[3]);
```

## Shape operations

```rust
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{Tensor, TensorOpsExt};

let mut backend = CpuBackend::new();
let a = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);

// Transpose
let at = a.transpose(&[1, 0], &mut backend).unwrap();
assert_eq!(at.shape(), &[3, 2]);

// Reshape
let flat = a.reshape(&[6], &mut backend).unwrap();
assert_eq!(flat.shape(), &[6]);

// Reduce
let col_sum = a.reduce_sum(&[0], &mut backend).unwrap();
assert_eq!(col_sum.shape(), &[3]);
```

The `reduce_sum(&[0], &mut backend)` call removes axis `0`. For this `[2, 3]` tensor, that
means summing down each column and keeping one value per column.

## Einsum

Use `tenferro_einsum::EagerEinsumExt` when working with `EagerTensor`.
For traced graph execution, use `tenferro_einsum::TraceContextEinsumExt` and
install `tenferro_einsum::extension_module` on the `Runtime`.

## Extracting data

```rust
use tenferro_runtime::Tensor;

let t = Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]);
let data: &[f64] = t.as_slice::<f64>().unwrap();
assert_eq!(data, &[1.0, 2.0, 3.0]);
```

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

```rust
use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
use tenferro_cpu::CpuBackend;

let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
let x = EagerTensor::requires_grad_in(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap(), ctx.clone()).unwrap();
let y = EagerTensor::requires_grad_in(Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).unwrap(), ctx.clone()).unwrap();

let loss = x.mul(&y).unwrap().reduce_sum(Some(&[0])).unwrap();
loss.backward().unwrap();
assert_eq!(x.grad().unwrap().unwrap().as_slice::<f64>().unwrap(), &[3.0, 4.0]);

let loss = x.mul(&y).unwrap().reduce_sum(Some(&[0])).unwrap();
loss.backward().unwrap();
assert_eq!(x.grad().unwrap().unwrap().as_slice::<f64>().unwrap(), &[6.0, 8.0]);

x.clear_grad().unwrap();
assert!(x.grad().unwrap().is_none());

let loss = x.mul(&y).unwrap().reduce_sum(Some(&[0])).unwrap();
loss.backward().unwrap();
assert_eq!(x.grad().unwrap().unwrap().as_slice::<f64>().unwrap(), &[3.0, 4.0]);

ctx.clear_grads().unwrap();
assert!(x.grad().unwrap().is_none());
assert!(y.grad().unwrap().is_none());
```

Use `backward_with` when the output is not scalar or when reverse mode should
start from an explicit cotangent seed:

```rust
use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
use tenferro_cpu::CpuBackend;

let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
let x = EagerTensor::requires_grad_in(
    Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 3.0]).unwrap(),
    ctx.clone(),
).unwrap();
let seed = EagerTensor::from_tensor_in(
    Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap(),
    ctx,
).unwrap();

let y = x.mul(&x).unwrap();
y.backward_with(&seed).unwrap();
assert_eq!(x.grad().unwrap().unwrap().as_slice::<f64>().unwrap(), &[4.0, 12.0]);
```

Functional eager transforms return tensors instead of updating gradient slots:

```rust
use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
use tenferro_cpu::CpuBackend;

let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
let x = EagerTensor::requires_grad_in(
    Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 3.0]).unwrap(),
    ctx.clone(),
).unwrap();
let y = x.mul(&x).unwrap();
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
assert_eq!(vjp.materialized().unwrap().as_slice::<f64>().unwrap(), &[4.0, 6.0]);
assert_eq!(jvp.materialized().unwrap().as_slice::<f64>().unwrap(), &[4.0, 6.0]);
assert!(x.grad().unwrap().is_none());
```

Because functional derivatives are eager tensors, Hessian-vector products can
be written as `jvp(grad(f))`:

```rust
use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
use tenferro_cpu::CpuBackend;

let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
let x = EagerTensor::requires_grad_in(
    Tensor::from_vec_col_major(vec![], vec![3.0_f64]).unwrap(),
    ctx.clone(),
).unwrap();
let tangent = EagerTensor::from_tensor_in(
    Tensor::from_vec_col_major(vec![], vec![1.0_f64]).unwrap(),
    ctx,
).unwrap();

let loss = x.mul(&x).unwrap().mul(&x).unwrap();
let grad = ctx.grad(&loss, &x).unwrap();
let hvp = ctx.jvp(&grad, &x, &tangent).unwrap();

assert_eq!(grad.materialized().unwrap().as_slice::<f64>().unwrap(), &[27.0]);
assert_eq!(hvp.materialized().unwrap().as_slice::<f64>().unwrap(), &[18.0]);
```

Wrap updates, metric computations, and other non-differentiated eager work in
`no_grad` when they should not be recorded:

```rust
use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
use tenferro_cpu::CpuBackend;

let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
let x = EagerTensor::requires_grad_in(
    Tensor::from_vec_col_major(vec![1], vec![3.0_f64]).unwrap(),
    ctx.clone(),
).unwrap();
let y = {
    let _guard = ctx.no_grad();
    x.mul(&x).unwrap()
};
assert!(!y.tracks_grad());
```

`matmul` participates in the same eager reverse-mode workflow:

```rust
use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
use tenferro_cpu::CpuBackend;

let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
let a = EagerTensor::requires_grad_in(
    Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap(),
    ctx.clone(),
).unwrap();
let x = EagerTensor::requires_grad_in(
    Tensor::from_vec_col_major(vec![2, 1], vec![5.0_f64, 6.0]).unwrap(),
    ctx.clone(),
).unwrap();

let y = a.matmul(&x).unwrap();
assert_eq!(y.materialized().unwrap().as_slice::<f64>().unwrap(), &[23.0, 34.0]);

let loss = y.mul(&y).unwrap().reduce_sum(Some(&[0, 1])).unwrap();
assert_eq!(loss.materialized().unwrap().as_slice::<f64>().unwrap(), &[1685.0]);

loss.backward().unwrap();
assert_eq!(x.grad().unwrap().unwrap().as_slice::<f64>().unwrap(), &[182.0, 410.0]);
```

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
