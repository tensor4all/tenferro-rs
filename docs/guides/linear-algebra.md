# Linear Algebra

tenferro exposes traced linear algebra under `tenferro::traced_tensor`. The
helpers return `TracedTensor` values, so multi-output decompositions are best
compiled with `GraphCompiler::compile_many`.

## Singular value decomposition

```rust
use tenferro::traced_tensor::svd;
use tenferro::{CpuBackend, GraphCompiler, GraphExecutor, TracedTensor};

let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 2.0]);
let (u, s, vt) = svd(&a);

let mut compiler = GraphCompiler::new();
let program = compiler.compile_many(&[&u, &s, &vt]).unwrap();
let mut executor = GraphExecutor::new(CpuBackend::new());
let outputs = executor.run_many(&program).unwrap();

assert_eq!(outputs[0].shape(), &[2, 2]);
assert_eq!(outputs[2].shape(), &[2, 2]);

let mut singular_values = outputs[1].as_slice::<f64>().unwrap().to_vec();
singular_values.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap());
assert_eq!(singular_values, vec![1.0, 2.0]);
```

## QR decomposition

```rust
use tenferro::traced_tensor::qr;
use tenferro::{CpuBackend, GraphCompiler, GraphExecutor, TracedTensor};

let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 1.0]);
let (q, r) = qr(&a);

let mut compiler = GraphCompiler::new();
let program = compiler.compile_many(&[&q, &r]).unwrap();
let mut executor = GraphExecutor::new(CpuBackend::new());
let outputs = executor.run_many(&program).unwrap();

assert_eq!(outputs[0].shape(), &[2, 2]);
assert_eq!(outputs[1].shape(), &[2, 2]);
assert_eq!(outputs[0].as_slice::<f64>().unwrap(), &[1.0, 0.0, 0.0, 1.0]);
assert_eq!(outputs[1].as_slice::<f64>().unwrap(), &[1.0, 0.0, 0.0, 1.0]);
```

## Hermitian eigenvalue decomposition

```rust
use tenferro::traced_tensor::eigh;
use tenferro::{CpuBackend, GraphCompiler, GraphExecutor, TracedTensor};

let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 3.0]);
let (values, vectors) = eigh(&a);

let mut compiler = GraphCompiler::new();
let program = compiler.compile_many(&[&values, &vectors]).unwrap();
let mut executor = GraphExecutor::new(CpuBackend::new());
let outputs = executor.run_many(&program).unwrap();

assert_eq!(outputs[0].shape(), &[2]);
assert_eq!(outputs[1].shape(), &[2, 2]);

let mut eigenvalues = outputs[0].as_slice::<f64>().unwrap().to_vec();
eigenvalues.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap());
assert_eq!(eigenvalues, vec![1.0, 3.0]);
```

## Cholesky factorization

```rust
use tenferro::traced_tensor::cholesky;
use tenferro::{CpuBackend, GraphCompiler, GraphExecutor, TracedTensor};

let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![4.0_f64, 0.0, 0.0, 9.0]);
let factor = cholesky(&a);

let mut compiler = GraphCompiler::new();
let program = compiler.compile(&factor).unwrap();
let mut executor = GraphExecutor::new(CpuBackend::new());
let result = executor.run(&program).unwrap();

assert_eq!(result.shape(), &[2, 2]);
assert_eq!(result.as_slice::<f64>().unwrap(), &[2.0, 0.0, 0.0, 3.0]);
```

## Solve a linear system

```rust
use tenferro::traced_tensor::solve;
use tenferro::{CpuBackend, GraphCompiler, GraphExecutor, TracedTensor};

let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![4.0_f64, 0.0, 0.0, 9.0]);
let b = TracedTensor::from_vec_col_major(vec![2, 1], vec![8.0_f64, 27.0]);
let x = solve(&a, &b);

let mut compiler = GraphCompiler::new();
let program = compiler.compile(&x).unwrap();
let mut executor = GraphExecutor::new(CpuBackend::new());
let result = executor.run(&program).unwrap();

assert_eq!(result.shape(), &[2, 1]);
assert_eq!(result.as_slice::<f64>().unwrap(), &[2.0, 3.0]);
```

## Complete-pivot LU solve

```rust
use tenferro::traced_tensor::{full_piv_lu, full_piv_lu_solve};
use tenferro::{CpuBackend, GraphCompiler, GraphExecutor, TracedTensor};

let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![0.0_f64, 2.0, 1.0, 3.0]);
let b = TracedTensor::from_vec_col_major(vec![2, 1], vec![-1.0_f64, 5.0]);

let (p, l, u, q, parity) = full_piv_lu(&a);
let x = full_piv_lu_solve(&a, &b);

let mut compiler = GraphCompiler::new();
let program = compiler
    .compile_many(&[&p, &l, &u, &q, &parity, &x])
    .unwrap();
let mut executor = GraphExecutor::new(CpuBackend::new());
let outputs = executor.run_many(&program).unwrap();

assert_eq!(outputs[0].shape(), &[2, 2]);
assert_eq!(outputs[4].shape(), &[] as &[usize]);
assert_eq!(outputs[5].shape(), &[2, 1]);
assert_eq!(outputs[5].as_slice::<f64>().unwrap(), &[4.0, -1.0]);
```
