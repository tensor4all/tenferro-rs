# Linear Algebra

tenferro exposes linear algebra through the `tenferro-linalg` operation crate.
Use `TensorLinalgExt`, `TensorReadLinalgExt`, or `TypedTensorLinalgExt` for direct
execution without autodiff, `EagerTensorLinalgExt`
for immediate forward execution and eager `backward()` / functional transform
workflows under an `EagerRuntime`, and `TracedTensorLinalgExt` when the
operation should be part of a graph, `grad`/`vjp`/`jvp`, or repeated compile/run
workflow.

## Setup

When working from a local checkout, use paths that match your project layout.
For a scratch crate created directly inside the `tenferro-rs` checkout, include
an empty `[workspace]` table:

```toml
[workspace]
```

Then add the dependencies:

```toml
[dependencies]
tenferro-runtime = { path = "../crates/tenferro-runtime" }
tenferro-cpu = { path = "../crates/tenferro-cpu" }
tenferro-ad = { path = "../crates/tenferro-ad" }
tenferro-linalg = { path = "../crates/tenferro-linalg", features = ["autodiff"] }
```

For published crates, use the same crate set with version requirements:

```toml
[dependencies]
tenferro-runtime = "..."
tenferro-cpu = "..."
tenferro-ad = "..."
tenferro-linalg = { version = "...", features = ["autodiff"] }
```

Concrete graph/runtime users can omit `tenferro-ad` and the `autodiff` feature
when they do not need eager linalg helpers or linalg AD rules.
The examples below are Rust fragments; copy them into `fn main() -> Result<(),
Box<dyn std::error::Error>>` for a standalone binary.

## Layer Coverage

| Layer | Linear algebra style |
| --- | --- |
| Concrete `Tensor` / `TensorRead` / `TypedTensor<T>` | crate-root linalg extension traits; methods take `&mut dyn BackendSession` obtained via `BackendSessionHost::with_backend_session` |
| `EagerTensor` | `EagerTensorLinalgExt` methods behind `autodiff`; tracked variables support `backward()` and `EagerRuntime` functional transforms where AD rules support the operation |
| `TracedTensor` | `TracedTensorLinalgExt` methods for graph execution and `grad`/`vjp`/`jvp` workflows |

CUDA is a backend/device choice for supported `Tensor`, `EagerTensor`, and
`TracedTensor` paths. It is not a separate linear algebra layer. See
[Devices and GPU](devices-and-gpu.md) for the CUDA support table.

## Operation Surface

| Operation family | Concrete backend | Eager helper | Traced helper |
| --- | --- | --- | --- |
| Dense solve | `solve` | `solve` | `solve` |
| Triangular solve | `triangular_solve` | `triangular_solve` | `triangular_solve` |
| Cholesky | `cholesky` | `cholesky` | `cholesky` |
| SVD | `svd`, `svdvals`, `svd_with_options` | `svd`, `svd_with_options` | `svd`, `svd_with_options` |
| QR | `qr`, `qr_with_options` | `qr`, `qr_with_options` | `qr`, `qr_with_options` |
| Hermitian eigen | `eigh`, `eigh_with_options` | `eigh`, `eigh_with_options`, `eigvalsh` | `eigh`, `eigh_with_options`, `eigvalsh` |
| General eigen | `eig` | `eig`, `eigvals` | `eig`, `eigvals` |
| LU | `lu` | `lu` | `lu` |
| Complete-pivot LU | `full_piv_lu`, `full_piv_lu_solve` | `full_piv_lu`, `full_piv_lu_solve` | `full_piv_lu`, `full_piv_lu_solve` |
| Pseudoinverse | `pinv` | `pinv`, `pinv_with_rtol` | `pinv`, `pinv_with_rtol` |
| Determinants | `det`, `slogdet` | `det`, `slogdet` | `det`, `slogdet` |
| Matrix inverse | `inv` | `inv` | `inv` |
| Norms | `norm` | `norm` | `norm` |

Concrete, read, typed, eager, and traced tensor APIs are crate-root extension
traits. Concrete methods take `&mut dyn BackendSession` obtained through
`BackendSessionHost::with_backend_session`. `TensorReadLinalgExt::svdvals_read`
and `eigvalsh_read` accept borrowed inputs; eligible faer host views avoid a
full input copy, while providers that need owned compact storage materialize at
the provider boundary. A values-only backend capability returns only the
values tensor and does not allocate singular/eigenvectors; unsupported
providers return a typed error rather than computing a full decomposition and
discarding its factors. `LinalgBackend` is the SPI provider contract used by
backend implementations and lower-level session tests, not by ordinary
callers.

## Batch And Inner Parallelism

Batched decompositions have two independent kinds of parallel work:

- **Outer batch parallelism** runs independent matrices from the batch on the
  selected CPU domain or device execution context. The engine owns this
  scheduling and its resource budget.
- **Inner decomposition parallelism** partitions the factorization of one
  matrix. The selected provider owns this algorithmic decomposition, subject
  to the execution context passed by the engine.

For SVD, QR, Hermitian eigenvalue, and other batched linalg calls, the engine
must choose the outer fan-out and the provider must respect the selected inner
policy. A faer-backed CPU context can pass `Par::Seq` or `Par::rayon(n)` while
preserving one domain budget. A BLAS/LAPACK provider may use its own worker
pool, so its environment-controlled inner threads can oversubscribe an outer
batch; see [Choosing A Backend](choosing-a-backend.md) before combining the
two. Providers must report unsupported decomposition or dtype requests rather
than silently changing the algorithm or execution backend.

## Concrete Solve

<!-- snippet-source: docs/tutorial-code/src/bin/math_snippets.rs#linear_algebra_1 -->
```rust
use tenferro_cpu::CpuBackend;
use tenferro_runtime::BackendSessionHost;
use tenferro_linalg::TensorLinalgExt;
use tenferro_runtime::Tensor;

let a = Tensor::from_vec_col_major(vec![2, 2], vec![4.0_f64, 0.0, 0.0, 9.0])?;
let b = Tensor::from_vec_col_major(vec![2, 1], vec![8.0_f64, 27.0])?;
let mut backend = CpuBackend::new();
let x = backend.with_backend_session(|session| a.solve(&b, session))?;

assert_eq!(x.shape(), &[2, 1]);
assert_eq!(x.as_slice::<f64>()?, &[2.0, 3.0]);
```
<!-- end-snippet-source -->

## Concrete Cholesky

<!-- snippet-source: docs/tutorial-code/src/bin/math_snippets.rs#linear_algebra_2 -->
```rust
use tenferro_cpu::CpuBackend;
use tenferro_runtime::BackendSessionHost;
use tenferro_linalg::TensorLinalgExt;
use tenferro_runtime::{Tensor, TensorSessionOpsExt};

fn max_abs_diff(lhs: &Tensor, rhs: &Tensor) -> Result<f64, tenferro_tensor::Error> {
    let lhs = lhs.as_slice::<f64>()?;
    let rhs = rhs.as_slice::<f64>()?;
    Ok(lhs
        .iter()
        .zip(rhs.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0, f64::max))
}
let a = Tensor::from_vec_col_major(vec![2, 2], vec![4.0_f64, 1.0, 1.0, 3.0])?;
let mut backend = CpuBackend::new();
let (factor, reconstructed) = backend.with_backend_session(|session| -> tenferro_tensor::Result<(Tensor, Tensor)> {
    let factor = a.cholesky(session)?;
    let factor_t = factor.transpose(&[1, 0], session)?;
    let reconstructed = factor.matmul(&factor_t, session)?;
    Ok((factor, reconstructed))
})?;

assert_eq!(factor.shape(), &[2, 2]);
assert_eq!(a.shape(), &[2, 2]);
assert!(max_abs_diff(&reconstructed, &a)? < 1.0e-12);
```
<!-- end-snippet-source -->

## Direct Decompositions

The same operation families are available outside traced graphs. Use concrete
or typed tensors for direct execution without autodiff, eager tensors when the result
should be produced immediately under an `EagerRuntime`, and traced helpers when
the operation belongs in a reusable graph. Use tracked eager tensors when the
result should remain connected to a scalar loss `backward()` pass or to
functional eager `grad`/`vjp`/`jvp` transforms.
For linalg eager helpers or linalg AD rules, add the `tenferro-ad` dependency
and enable `tenferro-linalg`'s `autodiff` feature.

When traced graph AD must linearize through linalg extension ops, include the
owned rule set in an explicit context:

<!-- snippet-source: docs/tutorial-code/src/bin/math_snippets.rs#linear_algebra_3 -->
```rust
use tenferro_ad::AdContext;

let ad = AdContext::builder()
    .with_semantic_extension_rules(tenferro_linalg::semantic_ad_rules()?)?
    .build()?;
```
<!-- end-snippet-source -->

## Singular value decomposition

<!-- snippet-source: docs/tutorial-code/src/bin/math_snippets.rs#linear_algebra_4 -->
```rust
use tenferro_cpu::CpuBackend;
use tenferro_runtime::BackendSessionHost;
use tenferro_linalg::TensorLinalgExt;
use tenferro_runtime::{Tensor, TensorSessionOpsExt};

fn max_abs_diff(lhs: &Tensor, rhs: &Tensor) -> Result<f64, tenferro_tensor::Error> {
    let lhs = lhs.as_slice::<f64>()?;
    let rhs = rhs.as_slice::<f64>()?;
    Ok(lhs
        .iter()
        .zip(rhs.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0, f64::max))
}
let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 3.0, 2.0, 4.0])?;
let mut backend = CpuBackend::new();
let (u, s, vt) = backend.with_backend_session(|session| a.svd(session))?;

assert_eq!(u.shape(), &[2, 2]);
assert_eq!(vt.shape(), &[2, 2]);

let s_values = s.as_slice::<f64>()?;
let sigma = Tensor::from_vec_col_major(
    vec![2, 2],
    vec![s_values[0], 0.0, 0.0, s_values[1]],
)?;
let (reconstructed,) = backend.with_backend_session(|session| -> tenferro_tensor::Result<(Tensor,)> {
    let us = u.matmul(&sigma, session)?;
    let reconstructed = us.matmul(&vt, session)?;
    Ok((reconstructed,))
})?;

assert_eq!(a.shape(), &[2, 2]);
assert!(max_abs_diff(&reconstructed, &a)? < 1.0e-12);
```
<!-- end-snippet-source -->

## Decomposition Options And SVD Truncation

SVD, QR, and Hermitian eigen decomposition expose options when you need an
opt-in deterministic sign or phase convention. The default remains the
backend's raw gauge. SVD and Hermitian eigen options also expose
`derivative_eps`, which regularizes AD formulas for repeated or nearly repeated
singular values or eigenvalues. It is not a backend solver tolerance and does
not change the forward decomposition algorithm.

<!-- snippet-source: docs/tutorial-code/src/bin/math_snippets.rs#linear_algebra_5 -->
```rust
use tenferro_linalg::{SvdGauge, SvdOptions, TracedTensorLinalgExt};
use tenferro_runtime::TracedTensor;

let a = TracedTensor::from_vec_col_major(
    vec![3, 3],
    vec![
        3.0_f64, 0.0, 0.0,
        0.0, 2.0, 0.0,
        0.0, 0.0, 1.0,
    ],
)?;
let (u, s, vt) = a
    .svd_with_options(
        SvdOptions::default()
            .gauge(SvdGauge::CanonicalPivot)
            .derivative_eps(1.0e-10),
    )?;

let rank = 2;
let u_rank2 = u.slice_axis(1, 0..rank)?;
let s_rank2 = s.slice_axis(0, 0..rank)?;
let vt_rank2 = vt.slice_axis(0, 0..rank)?;

assert_eq!(u_rank2.concrete_shape()?, vec![3, 2]);
assert_eq!(s_rank2.concrete_shape()?, vec![2]);
assert_eq!(vt_rank2.concrete_shape()?, vec![2, 3]);
```
<!-- end-snippet-source -->

Use `slice_axis` for rank-preserving contiguous ranges and `take_axis` when the
selected axis needs repeated or reordered indices:

<!-- snippet-source: docs/tutorial-code/src/bin/math_snippets.rs#linear_algebra_6 -->
```rust
use tenferro_linalg::TracedTensorLinalgExt;
use tenferro_runtime::TracedTensor;

let a = TracedTensor::from_vec_col_major(
    vec![3, 3],
    vec![
        3.0_f64, 0.0, 0.0,
        0.0, 2.0, 0.0,
        0.0, 0.0, 1.0,
    ],
)?;
let (_u, s, _vt) = a.svd()?;
let repeated = s.take_axis(0, &[0, 1, 0])?;

assert_eq!(repeated.concrete_shape()?, vec![3]);
```
<!-- end-snippet-source -->

## QR decomposition

<!-- snippet-source: docs/tutorial-code/src/bin/math_snippets.rs#linear_algebra_7 -->
```rust
use tenferro_cpu::CpuBackend;
use tenferro_runtime::BackendSessionHost;
use tenferro_linalg::{QrGauge, QrOptions, TensorLinalgExt};
use tenferro_runtime::{Tensor, TensorSessionOpsExt};

fn max_abs_diff(lhs: &Tensor, rhs: &Tensor) -> Result<f64, tenferro_tensor::Error> {
    let lhs = lhs.as_slice::<f64>()?;
    let rhs = rhs.as_slice::<f64>()?;
    Ok(lhs
        .iter()
        .zip(rhs.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0, f64::max))
}
let a = Tensor::from_vec_col_major(
    vec![4, 3],
    vec![
        1.0_f64, 4.0, 7.0, 2.0,
        2.0, 5.0, 8.0, 3.0,
        3.0, 6.0, 10.0, 5.0,
    ],
)?;
let mut backend = CpuBackend::new();
let (q, r) = backend.with_backend_session(|session| {
    a.qr_with_options(
        QrOptions::default().gauge(QrGauge::PositiveDiagonal),
        session,
    )
})?;

let identity = Tensor::from_vec_col_major(
    vec![3, 3],
    vec![1.0_f64, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
)?;
let (reconstructed, qtq) = backend.with_backend_session(|session| -> tenferro_tensor::Result<(Tensor, Tensor)> {
    let reconstructed = q.matmul(&r, session)?;
    let qt = q.transpose(&[1, 0], session)?;
    let qtq = qt.matmul(&q, session)?;
    Ok((reconstructed, qtq))
})?;

assert_eq!(q.shape(), &[4, 3]);
assert_eq!(r.shape(), &[3, 3]);
assert_eq!(identity.shape(), &[3, 3]);
assert!(max_abs_diff(&reconstructed, &a)? < 1.0e-12);
assert!(max_abs_diff(&qtq, &identity)? < 1.0e-12);
```
<!-- end-snippet-source -->

## Hermitian eigenvalue decomposition

<!-- snippet-source: docs/tutorial-code/src/bin/math_snippets.rs#linear_algebra_8 -->
```rust
use tenferro_cpu::CpuBackend;
use tenferro_runtime::BackendSessionHost;
use tenferro_linalg::TensorLinalgExt;
use tenferro_runtime::{Tensor, TensorSessionOpsExt};

fn max_abs_diff(lhs: &Tensor, rhs: &Tensor) -> Result<f64, tenferro_tensor::Error> {
    let lhs = lhs.as_slice::<f64>()?;
    let rhs = rhs.as_slice::<f64>()?;
    Ok(lhs
        .iter()
        .zip(rhs.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0, f64::max))
}
let a = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 1.0, 1.0, 2.0])?;
let mut backend = CpuBackend::new();
let (values, vectors) = backend.with_backend_session(|session| a.eigh(session))?;

assert_eq!(values.shape(), &[2]);
assert_eq!(vectors.shape(), &[2, 2]);

let value_slice = values.as_slice::<f64>()?;
let diagonal = Tensor::from_vec_col_major(
    vec![2, 2],
    vec![value_slice[0], 0.0, 0.0, value_slice[1]],
)?;
let (reconstructed,) = backend.with_backend_session(|session| -> tenferro_tensor::Result<(Tensor,)> {
    let vd = vectors.matmul(&diagonal, session)?;
    let vt = vectors.transpose(&[1, 0], session)?;
    let reconstructed = vd.matmul(&vt, session)?;
    Ok((reconstructed,))
})?;

assert_eq!(a.shape(), &[2, 2]);
assert!(max_abs_diff(&reconstructed, &a)? < 1.0e-12);
```
<!-- end-snippet-source -->

## Traced Cholesky Factorization

<!-- snippet-source: docs/tutorial-code/src/bin/math_snippets.rs#linear_algebra_9 -->
```rust
use tenferro_cpu::{runtime_engine_id, runtime_engine_registration, CpuBackend};
use tenferro_runtime::{GraphCompiler, Runtime, TracedTensor};
use tenferro_linalg::TracedTensorLinalgExt;

let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![4.0_f64, 0.0, 0.0, 9.0])?;
let factor = a.cholesky()?;

let mut compiler = GraphCompiler::new();
let program = compiler.compile(&factor)?;
let backend = CpuBackend::new();
let mut builder = Runtime::builder();
builder
    .register_engine(runtime_engine_registration(&backend)?)?;
builder
    .install_extension_module(
        tenferro_linalg::extension_module::<CpuBackend>(runtime_engine_id()?)?,
    )?;
let runtime = builder.build()?;
let mut outputs = runtime.run_compiled(&program, &[])?;
let result = outputs.remove(0);

assert_eq!(result.shape(), &[2, 2]);
assert_eq!(result.as_slice::<f64>()?, &[2.0, 0.0, 0.0, 3.0]);
```
<!-- end-snippet-source -->

## Traced Solve In A Graph

<!-- snippet-source: docs/tutorial-code/src/bin/math_snippets.rs#linear_algebra_10 -->
```rust
use tenferro_cpu::{runtime_engine_id, runtime_engine_registration, CpuBackend};
use tenferro_runtime::{GraphCompiler, Runtime, TracedTensor};
use tenferro_linalg::TracedTensorLinalgExt;

let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![4.0_f64, 0.0, 0.0, 9.0])?;
let b = TracedTensor::from_vec_col_major(vec![2, 1], vec![8.0_f64, 27.0])?;
let x = a.solve(&b)?;

let mut compiler = GraphCompiler::new();
let program = compiler.compile(&x)?;
let backend = CpuBackend::new();
let mut builder = Runtime::builder();
builder
    .register_engine(runtime_engine_registration(&backend)?)?;
builder
    .install_extension_module(
        tenferro_linalg::extension_module::<CpuBackend>(runtime_engine_id()?)?,
    )?;
let runtime = builder.build()?;
let mut outputs = runtime.run_compiled(&program, &[])?;
let result = outputs.remove(0);

assert_eq!(result.shape(), &[2, 1]);
assert_eq!(result.as_slice::<f64>()?, &[2.0, 3.0]);
```
<!-- end-snippet-source -->

## Complete-Pivot LU Solve

`full_piv_lu` returns `(P, L, U, Q, parity)` with the reconstruction convention
`A = P^T * L * U * Q`, equivalently `P * A * Q^T = L * U`. The `parity` output
is a scalar real tensor containing `+1` or `-1`: `F32` for `F32`/`C32` inputs
and `F64` for `F64`/`C64` inputs. The tensor extension
`full_piv_lu_solve` solves `A * x = b`; the lower-level backend contract also
exposes an explicit transpose flag.

<!-- snippet-source: docs/tutorial-code/src/bin/math_snippets.rs#linear_algebra_11 -->
```rust
use tenferro_cpu::CpuBackend;
use tenferro_runtime::BackendSessionHost;
use tenferro_linalg::TensorLinalgExt;
use tenferro_runtime::{Tensor, TensorSessionOpsExt};

fn max_abs_diff(lhs: &Tensor, rhs: &Tensor) -> Result<f64, tenferro_tensor::Error> {
    let lhs = lhs.as_slice::<f64>()?;
    let rhs = rhs.as_slice::<f64>()?;
    Ok(lhs
        .iter()
        .zip(rhs.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0, f64::max))
}
let mut backend = CpuBackend::new();
let a = Tensor::from_vec_col_major(
    vec![4, 4],
    vec![
        0.0_f64, 4.0, 7.0, 1.0,
        2.0, 5.0, 8.0, 0.0,
        3.0, 6.0, 10.0, 2.0,
        1.0, 2.0, 3.0, 4.0,
    ],
)?;
let b = Tensor::from_vec_col_major(vec![4, 1], vec![1.0_f64, 2.0, 3.0, 4.0])?;

let (p, l, u, q, parity) = backend.with_backend_session(|session| a.full_piv_lu(session))?;
let (reconstructed,) = backend.with_backend_session(|session| -> tenferro_tensor::Result<(Tensor,)> {
    let pt = p.transpose(&[1, 0], session)?;
    let pt_l = pt.matmul(&l, session)?;
    let pt_lu = pt_l.matmul(&u, session)?;
    let reconstructed = pt_lu.matmul(&q, session)?;
    Ok((reconstructed,))
})?;
let x = backend.with_backend_session(|session| a.full_piv_lu_solve(&b, session))?;

assert_eq!(p.shape(), &[4, 4]);
assert_eq!(a.shape(), &[4, 4]);
assert!(max_abs_diff(&reconstructed, &a)? < 1.0e-12);
assert_eq!(parity.shape(), &[] as &[usize]);
let parity_value = parity
    .as_slice::<f64>()?
    .first()
    .copied()
    .ok_or_else(|| tenferro_tensor::Error::Internal("missing LU parity value".to_owned()))?;
assert!(parity_value == 1.0 || parity_value == -1.0);
assert_eq!(x.shape(), &[4, 1]);
```
<!-- end-snippet-source -->
