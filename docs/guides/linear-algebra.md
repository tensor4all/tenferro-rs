# Linear Algebra

tenferro exposes linear algebra through the `tenferro-linalg` operation crate.
Use `LinalgBackend` for direct execution without autodiff, `EagerTensor` helpers for
immediate forward execution under an `EagerRuntime`, and `TracedTensor` helpers
when the operation should be part of a graph, `grad`/`vjp`/`jvp`, or repeated
compile/run workflow.

## Setup

```toml
[dependencies]
tenferro-runtime = { path = "../crates/tenferro-runtime" }
tenferro-cpu = { path = "../crates/tenferro-cpu" }
tenferro-ad = { path = "../crates/tenferro-ad" }
tenferro-linalg = { path = "../crates/tenferro-linalg", features = ["autodiff"] }
```

Concrete graph/runtime users can omit `tenferro-ad` and the `autodiff` feature
when they do not need eager linalg helpers or linalg AD rules.

## Layer Coverage

| Layer | Linear algebra style |
| --- | --- |
| Concrete `Tensor` | `tenferro_linalg::LinalgBackend` methods on a backend |
| `EagerTensor` | `tenferro_linalg::eager_tensor` helpers behind `autodiff`; tracked variables record gradients for scalar losses where AD rules support the operation |
| `TracedTensor` | `tenferro_linalg::traced_tensor` helpers for graph execution and `grad`/`vjp`/`jvp` workflows |

CUDA is a backend/device choice for supported `Tensor`, `EagerTensor`, and
`TracedTensor` paths. It is not a separate linear algebra layer. See
[Devices and GPU](devices-and-gpu.md) for the CUDA support table.

## Operation Surface

| Operation family | Concrete backend | Eager helper | Traced helper |
| --- | --- | --- | --- |
| Dense solve | `solve` | `solve` | `solve` |
| Triangular solve | `triangular_solve` | `triangular_solve` | `triangular_solve` |
| Cholesky | `cholesky` | `cholesky` | `cholesky` |
| SVD | `svd` | `svd` | `svd` |
| QR | `qr` | `qr` | `qr` |
| Hermitian eigen | `eigh` | `eigh` | `eigh`, `eigvalsh` |
| General eigen | `eig` | `eig` | `eig`, `eigvals` |
| LU | `lu` | `lu` | `lu` |
| Complete-pivot LU | `full_piv_lu`, `full_piv_lu_solve` | `full_piv_lu`, `full_piv_lu_solve` | `full_piv_lu`, `full_piv_lu_solve` |
| Pseudoinverse | `pinv` | - | `pinv`, `pinv_with_rtol` |
| Determinants | `det`, `slogdet` | - | `det`, `slogdet` |
| Norms | `norm` | - | `norm` |

Concrete methods are exposed by `LinalgBackend`; eager and traced helpers live
under `tenferro_linalg::eager_tensor` and `tenferro_linalg::traced_tensor`.

## Concrete Solve

```rust
use tenferro_linalg::LinalgBackend;
use tenferro_cpu::CpuBackend;
use tenferro_runtime::Tensor;

let mut backend = CpuBackend::new();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![4.0_f64, 0.0, 0.0, 9.0]);
let b = Tensor::from_vec_col_major(vec![2, 1], vec![8.0_f64, 27.0]);

let x = LinalgBackend::solve(&mut backend, &a, &b).unwrap();

assert_eq!(x.shape(), &[2, 1]);
assert_eq!(x.as_slice::<f64>().unwrap(), &[2.0, 3.0]);
```

## Concrete Cholesky

```rust
use tenferro_linalg::LinalgBackend;
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{tensor, Tensor};

fn max_abs_diff(lhs: &Tensor, rhs: &Tensor) -> f64 {
    lhs.as_slice::<f64>()
        .unwrap()
        .iter()
        .zip(rhs.as_slice::<f64>().unwrap())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0, f64::max)
}

let mut backend = CpuBackend::new();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![4.0_f64, 1.0, 1.0, 3.0]);

let factor = LinalgBackend::cholesky(&mut backend, &a).unwrap();
let factor_t = tensor::transpose(&factor, &[1, 0], &mut backend).unwrap();
let reconstructed = tensor::matmul(&factor, &factor_t, &mut backend).unwrap();

assert_eq!(factor.shape(), &[2, 2]);
assert!(max_abs_diff(&reconstructed, &a) < 1.0e-12);
```

## Direct Decompositions

The same operation families are available outside traced graphs. Use concrete
or typed tensors for direct execution without autodiff, eager tensors when the result
should be produced immediately under an `EagerRuntime`, and traced helpers when
the operation belongs in a reusable graph. Use tracked eager tensors only when
the result should remain connected to a scalar loss `backward()` pass.
For linalg eager helpers or linalg AD rules, add the `tenferro-ad` dependency
and enable `tenferro-linalg`'s `autodiff` feature.

When traced graph AD must linearize through linalg extension ops, include the
owned rule set in an explicit context:

```rust
use tenferro_ad::AdContext;

let ad = AdContext::builder()
    .with_extension_rules(tenferro_linalg::ad_rules().unwrap())
    .build()
    .unwrap();
```

## Singular value decomposition

```rust
use tenferro_linalg::LinalgBackend;
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{tensor, Tensor};

fn max_abs_diff(lhs: &Tensor, rhs: &Tensor) -> f64 {
    lhs.as_slice::<f64>()
        .unwrap()
        .iter()
        .zip(rhs.as_slice::<f64>().unwrap())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0, f64::max)
}

let mut backend = CpuBackend::new();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 3.0, 2.0, 4.0]);
let outputs = LinalgBackend::svd(&mut backend, &a).unwrap();
let u = &outputs[0];
let s = &outputs[1];
let vt = &outputs[2];

assert_eq!(u.shape(), &[2, 2]);
assert_eq!(vt.shape(), &[2, 2]);

let s_values = s.as_slice::<f64>().unwrap();
let sigma = Tensor::from_vec_col_major(
    vec![2, 2],
    vec![s_values[0], 0.0, 0.0, s_values[1]],
);
let us = tensor::matmul(u, &sigma, &mut backend).unwrap();
let reconstructed = tensor::matmul(&us, vt, &mut backend).unwrap();

assert!(max_abs_diff(&reconstructed, &a) < 1.0e-12);
```

## QR decomposition

```rust
use tenferro_linalg::LinalgBackend;
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{tensor, Tensor};

fn max_abs_diff(lhs: &Tensor, rhs: &Tensor) -> f64 {
    lhs.as_slice::<f64>()
        .unwrap()
        .iter()
        .zip(rhs.as_slice::<f64>().unwrap())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0, f64::max)
}

let mut backend = CpuBackend::new();
let a = Tensor::from_vec_col_major(
    vec![4, 3],
    vec![
        1.0_f64, 4.0, 7.0, 2.0,
        2.0, 5.0, 8.0, 3.0,
        3.0, 6.0, 10.0, 5.0,
    ],
);
let outputs = LinalgBackend::qr(&mut backend, &a).unwrap();
let q = &outputs[0];
let r = &outputs[1];

assert_eq!(q.shape(), &[4, 3]);
assert_eq!(r.shape(), &[3, 3]);

let reconstructed = tensor::matmul(q, r, &mut backend).unwrap();
let qt = tensor::transpose(q, &[1, 0], &mut backend).unwrap();
let qtq = tensor::matmul(&qt, q, &mut backend).unwrap();
let identity = Tensor::from_vec_col_major(
    vec![3, 3],
    vec![1.0_f64, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
);

assert!(max_abs_diff(&reconstructed, &a) < 1.0e-12);
assert!(max_abs_diff(&qtq, &identity) < 1.0e-12);
```

## Hermitian eigenvalue decomposition

```rust
use tenferro_linalg::LinalgBackend;
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{tensor, Tensor};

fn max_abs_diff(lhs: &Tensor, rhs: &Tensor) -> f64 {
    lhs.as_slice::<f64>()
        .unwrap()
        .iter()
        .zip(rhs.as_slice::<f64>().unwrap())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0, f64::max)
}

let mut backend = CpuBackend::new();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 1.0, 1.0, 2.0]);
let outputs = LinalgBackend::eigh(&mut backend, &a).unwrap();
let values = &outputs[0];
let vectors = &outputs[1];

assert_eq!(values.shape(), &[2]);
assert_eq!(vectors.shape(), &[2, 2]);

let value_slice = values.as_slice::<f64>().unwrap();
let diagonal = Tensor::from_vec_col_major(
    vec![2, 2],
    vec![value_slice[0], 0.0, 0.0, value_slice[1]],
);
let vd = tensor::matmul(vectors, &diagonal, &mut backend).unwrap();
let vt = tensor::transpose(vectors, &[1, 0], &mut backend).unwrap();
let reconstructed = tensor::matmul(&vd, &vt, &mut backend).unwrap();

assert!(max_abs_diff(&reconstructed, &a) < 1.0e-12);
```

## Traced Cholesky Factorization

```rust
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{GraphCompiler, GraphExecutor, TracedTensor};
use tenferro_linalg::traced_tensor::cholesky;

let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![4.0_f64, 0.0, 0.0, 9.0]);
let factor = cholesky(&a).unwrap();

let mut compiler = GraphCompiler::new();
let program = compiler.compile(&factor).unwrap();
let mut executor = GraphExecutor::new(CpuBackend::new());
executor.register_extension(tenferro_linalg::register_runtime).unwrap();
let result = executor.run(&program).unwrap();

assert_eq!(result.shape(), &[2, 2]);
assert_eq!(result.as_slice::<f64>().unwrap(), &[2.0, 0.0, 0.0, 3.0]);
```

## Traced Solve In A Graph

```rust
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{GraphCompiler, GraphExecutor, TracedTensor};
use tenferro_linalg::traced_tensor::solve;

let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![4.0_f64, 0.0, 0.0, 9.0]);
let b = TracedTensor::from_vec_col_major(vec![2, 1], vec![8.0_f64, 27.0]);
let x = solve(&a, &b).unwrap();

let mut compiler = GraphCompiler::new();
let program = compiler.compile(&x).unwrap();
let mut executor = GraphExecutor::new(CpuBackend::new());
executor.register_extension(tenferro_linalg::register_runtime).unwrap();
let result = executor.run(&program).unwrap();

assert_eq!(result.shape(), &[2, 1]);
assert_eq!(result.as_slice::<f64>().unwrap(), &[2.0, 3.0]);
```

## Complete-Pivot LU Solve

`full_piv_lu` returns `(P, L, U, Q, parity)` with the reconstruction convention
`A = P^T * L * U * Q`, equivalently `P * A * Q^T = L * U`. The `parity` output
is a scalar real tensor containing `+1` or `-1`: `F32` for `F32`/`C32` inputs
and `F64` for `F64`/`C64` inputs. `full_piv_lu_solve(..., false)` solves
`A * x = b`; passing `true` solves `A^T * x = b`.

```rust
use tenferro_linalg::LinalgBackend;
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{tensor, Tensor};

fn max_abs_diff(lhs: &Tensor, rhs: &Tensor) -> f64 {
    lhs.as_slice::<f64>()
        .unwrap()
        .iter()
        .zip(rhs.as_slice::<f64>().unwrap())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0, f64::max)
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
);
let b = Tensor::from_vec_col_major(vec![4, 1], vec![1.0_f64, 2.0, 3.0, 4.0]);

let outputs = LinalgBackend::full_piv_lu(&mut backend, &a).unwrap();
let p = &outputs[0];
let l = &outputs[1];
let u = &outputs[2];
let q = &outputs[3];
let parity = &outputs[4];
let pt = tensor::transpose(p, &[1, 0], &mut backend).unwrap();
let pt_l = tensor::matmul(&pt, l, &mut backend).unwrap();
let pt_lu = tensor::matmul(&pt_l, u, &mut backend).unwrap();
let reconstructed = tensor::matmul(&pt_lu, q, &mut backend).unwrap();
let x = LinalgBackend::full_piv_lu_solve(&mut backend, &a, &b, false).unwrap();

assert_eq!(p.shape(), &[4, 4]);
assert!(max_abs_diff(&reconstructed, &a) < 1.0e-12);
assert_eq!(parity.shape(), &[] as &[usize]);
let parity_value = parity.as_slice::<f64>().unwrap()[0];
assert!(parity_value == 1.0 || parity_value == -1.0);
assert_eq!(x.shape(), &[4, 1]);
```
