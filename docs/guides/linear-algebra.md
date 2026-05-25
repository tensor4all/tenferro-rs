# Linear Algebra

tenferro exposes linear algebra across the concrete, eager, and traced tensor
layers. Use `Tensor` or `TypedTensor<T>` for no-AD direct execution. Use
`EagerTensor` for immediate forward execution under an `EagerRuntime`, with
optional scalar-loss `backward()` where AD rules support the operation. Use
`TracedTensor` when the linear algebra operation should be part of a graph,
transform AD pass, or repeated compile/run workflow.

## Layer Coverage

| Layer | Linear algebra style |
| --- | --- |
| `TypedTensor<T>` | Selected typed methods such as `svd`, `qr`, `cholesky`, and `eigh` |
| `Tensor` | Dynamic dtype methods such as `svd`, `qr`, `cholesky`, `eigh`, and `solve` |
| `EagerTensor` | Immediate forward execution; tracked variables record scalar-loss gradients where AD rules support the operation |
| `TracedTensor` | `tenferro_linalg::traced_tensor` helpers for graph execution and transform AD |

CUDA is a backend/device choice for supported `Tensor`, `EagerTensor`, and
`TracedTensor` paths. It is not a separate linear algebra layer. See
[Devices and GPU](devices-and-gpu.md) for the CUDA support table.

## Concrete Solve

```rust
use tenferro::{CpuBackend, Tensor};

let mut backend = CpuBackend::new();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![4.0_f64, 0.0, 0.0, 9.0]);
let b = Tensor::from_vec_col_major(vec![2, 1], vec![8.0_f64, 27.0]);

let x = a.solve(&b, &mut backend).unwrap();

assert_eq!(x.shape(), &[2, 1]);
assert_eq!(x.as_slice::<f64>().unwrap(), &[2.0, 3.0]);
```

## Typed Cholesky

```rust
use tenferro::{CpuBackend, TypedTensor};

let mut backend = CpuBackend::new();
let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![4.0, 0.0, 0.0, 9.0]);

let factor = a.cholesky(&mut backend).unwrap();

assert_eq!(factor.shape.as_slice(), &[2, 2]);
assert_eq!(factor.host_data(), &[2.0, 0.0, 0.0, 3.0]);
```

## Direct Decompositions

The same operation families are available outside traced graphs. Use concrete
or typed tensors for direct no-AD execution, eager tensors when the result
should be produced immediately under an `EagerRuntime`, and traced helpers when
the operation belongs in a reusable graph. Use tracked eager tensors only when
the result should remain connected to a scalar-loss `backward()` pass.

## Singular value decomposition

```rust
use tenferro::{CpuBackend, Tensor};

let mut backend = CpuBackend::new();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 2.0]);
let (u, s, vt) = a.svd(&mut backend).unwrap();

assert_eq!(u.shape(), &[2, 2]);
assert_eq!(vt.shape(), &[2, 2]);

let mut singular_values = s.as_slice::<f64>().unwrap().to_vec();
singular_values.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap());
assert_eq!(singular_values, vec![1.0, 2.0]);
```

## QR decomposition

```rust
use tenferro::{CpuBackend, TypedTensor};

let mut backend = CpuBackend::new();
let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]);
let (q, r) = a.qr(&mut backend).unwrap();

assert_eq!(q.shape.as_slice(), &[2, 2]);
assert_eq!(r.shape.as_slice(), &[2, 2]);
assert_eq!(q.host_data(), &[1.0, 0.0, 0.0, 1.0]);
assert_eq!(r.host_data(), &[1.0, 0.0, 0.0, 1.0]);
```

## Hermitian eigenvalue decomposition

```rust
use tenferro::{EagerRuntime, Tensor};

let ctx = EagerRuntime::new();
let a = ctx.variable_from(Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 3.0]));
let (values, vectors) = a.eigh().unwrap();

assert_eq!(values.data().shape(), &[2]);
assert_eq!(vectors.data().shape(), &[2, 2]);

let mut eigenvalues = values.data().as_slice::<f64>().unwrap().to_vec();
eigenvalues.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap());
assert_eq!(eigenvalues, vec![1.0, 3.0]);
```

## Traced Cholesky Factorization

```rust
use tenferro::{CpuBackend, GraphCompiler, GraphExecutor, TracedTensor};
use tenferro_linalg::traced_tensor::cholesky;

let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![4.0_f64, 0.0, 0.0, 9.0]);
let factor = cholesky(&a);

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
use tenferro::{CpuBackend, GraphCompiler, GraphExecutor, TracedTensor};
use tenferro_linalg::traced_tensor::solve;

let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![4.0_f64, 0.0, 0.0, 9.0]);
let b = TracedTensor::from_vec_col_major(vec![2, 1], vec![8.0_f64, 27.0]);
let x = solve(&a, &b);

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
use tenferro::{CpuBackend, Tensor};

let mut backend = CpuBackend::new();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![0.0_f64, 2.0, 1.0, 3.0]);
let b = Tensor::from_vec_col_major(vec![2, 1], vec![-1.0_f64, 5.0]);

let (p, _l, _u, _q, parity) = a.full_piv_lu(&mut backend).unwrap();
let x = a.full_piv_lu_solve(&b, &mut backend).unwrap();

assert_eq!(p.shape(), &[2, 2]);
assert_eq!(parity.shape(), &[] as &[usize]);
assert_eq!(x.shape(), &[2, 1]);
assert_eq!(x.as_slice::<f64>().unwrap(), &[4.0, -1.0]);
```
