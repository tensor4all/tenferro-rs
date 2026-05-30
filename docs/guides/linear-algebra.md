# Linear Algebra

tenferro exposes linear algebra through the `tenferro-linalg` operation crate.
Use `LinalgBackend` for direct execution without autodiff, `EagerTensor` helpers for
immediate forward execution under an `EagerRuntime`, and `TracedTensor` helpers
when the operation should be part of a graph, `grad`/`vjp`/`jvp`, or repeated
compile/run workflow.

## Layer Coverage

| Layer | Linear algebra style |
| --- | --- |
| Concrete `Tensor` | `tenferro_linalg::LinalgBackend` methods on a backend |
| `EagerTensor` | `tenferro_linalg::eager_tensor` helpers behind `autodiff`; tracked variables record gradients for scalar losses where AD rules support the operation |
| `TracedTensor` | `tenferro_linalg::traced_tensor` helpers for graph execution and `grad`/`vjp`/`jvp` workflows |

CUDA is a backend/device choice for supported `Tensor`, `EagerTensor`, and
`TracedTensor` paths. It is not a separate linear algebra layer. See
[Devices and GPU](devices-and-gpu.md) for the CUDA support table.

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
use tenferro_runtime::Tensor;

let mut backend = CpuBackend::new();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![4.0_f64, 0.0, 0.0, 9.0]);

let factor = LinalgBackend::cholesky(&mut backend, &a).unwrap();

assert_eq!(factor.shape(), &[2, 2]);
assert_eq!(factor.as_slice::<f64>().unwrap(), &[2.0, 0.0, 0.0, 3.0]);
```

## Direct Decompositions

The same operation families are available outside traced graphs. Use concrete
or typed tensors for direct execution without autodiff, eager tensors when the result
should be produced immediately under an `EagerRuntime`, and traced helpers when
the operation belongs in a reusable graph. Use tracked eager tensors only when
the result should remain connected to a scalar loss `backward()` pass.
For linalg eager helpers or linalg AD rules, enable `tenferro-linalg`'s
`autodiff` feature.

When traced graph AD must differentiate through linalg extension ops, include the
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
use tenferro_runtime::Tensor;

let mut backend = CpuBackend::new();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 2.0]);
let outputs = LinalgBackend::svd(&mut backend, &a).unwrap();
let u = &outputs[0];
let s = &outputs[1];
let vt = &outputs[2];

assert_eq!(u.shape(), &[2, 2]);
assert_eq!(vt.shape(), &[2, 2]);

let mut singular_values = s.as_slice::<f64>().unwrap().to_vec();
singular_values.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap());
assert_eq!(singular_values, vec![1.0, 2.0]);
```

## QR decomposition

```rust
use tenferro_linalg::LinalgBackend;
use tenferro_cpu::CpuBackend;
use tenferro_runtime::Tensor;

let mut backend = CpuBackend::new();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 1.0]);
let outputs = LinalgBackend::qr(&mut backend, &a).unwrap();
let q = &outputs[0];
let r = &outputs[1];

assert_eq!(q.shape(), &[2, 2]);
assert_eq!(r.shape(), &[2, 2]);
assert_eq!(q.as_slice::<f64>().unwrap(), &[1.0, 0.0, 0.0, 1.0]);
assert_eq!(r.as_slice::<f64>().unwrap(), &[1.0, 0.0, 0.0, 1.0]);
```

## Hermitian eigenvalue decomposition

```rust
use tenferro_ad::{EagerRuntime, Tensor};

let ctx = EagerRuntime::new();
let a = ctx.variable_from(Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 3.0]));
let (values, vectors) = tenferro_linalg::eager_tensor::eigh(&a).unwrap();

assert_eq!(values.data().shape(), &[2]);
assert_eq!(vectors.data().shape(), &[2, 2]);

let mut eigenvalues = values.data().as_slice::<f64>().unwrap().to_vec();
eigenvalues.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap());
assert_eq!(eigenvalues, vec![1.0, 3.0]);
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

```rust
use tenferro_linalg::LinalgBackend;
use tenferro_cpu::CpuBackend;
use tenferro_runtime::Tensor;

let mut backend = CpuBackend::new();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![0.0_f64, 2.0, 1.0, 3.0]);
let b = Tensor::from_vec_col_major(vec![2, 1], vec![-1.0_f64, 5.0]);

let outputs = LinalgBackend::full_piv_lu(&mut backend, &a).unwrap();
let p = &outputs[0];
let parity = &outputs[4];
let x = LinalgBackend::full_piv_lu_solve(&mut backend, &a, &b, false).unwrap();

assert_eq!(p.shape(), &[2, 2]);
assert_eq!(parity.shape(), &[] as &[usize]);
assert_eq!(x.shape(), &[2, 1]);
assert_eq!(x.as_slice::<f64>().unwrap(), &[4.0, -1.0]);
```
