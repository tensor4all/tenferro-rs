# Autodiff

tenferro exposes transform-oriented autodiff directly on `TracedTensor`.
Eager tensors also support scalar-loss reverse-mode via `backward()`, but this
page focuses on graph transforms that you compile and execute explicitly.

- `grad` for scalar-loss reverse mode
- `vjp` for vector-Jacobian products
- `jvp` for Jacobian-vector products
- Higher-order AD via composition, such as `jvp(grad(f))` for HVPs

## Reverse-mode gradient with `grad`

```rust
use tenferro::{CpuBackend, GraphCompiler, GraphExecutor, TracedTensor};

let x = TracedTensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]);
let loss = (&x * &x).reduce_sum(&[0]);
let grad = loss.grad(&x).unwrap();

let mut compiler = GraphCompiler::new();
let program = compiler.compile(&grad).unwrap();
let mut executor = GraphExecutor::new(CpuBackend::new());
let result = executor.run(&program).unwrap();

assert_eq!(result.shape(), &[3]);
assert_eq!(result.as_slice::<f64>().unwrap(), &[2.0, 4.0, 6.0]);
```

## Gradient through einsum

```rust
use tenferro::traced_tensor::einsum;
use tenferro::{CpuBackend, GraphCompiler, GraphExecutor, TracedTensor};

let a = TracedTensor::from_vec_col_major(
    vec![2, 3],
    vec![1.0_f64, -2.0, 0.5, 3.0, 1.25, -0.75],
);
let b = TracedTensor::from_vec_col_major(
    vec![3, 2],
    vec![2.0_f64, 0.25, -1.5, 4.0, 0.75, -0.5],
);

let mut compiler = GraphCompiler::new();
let y = einsum(&mut compiler, &[&a, &b], "ij,jk->ik").unwrap();
let loss = y.reduce_sum(&[0, 1]);
let grad_a = loss.grad(&a).unwrap();
let program = compiler.compile(&grad_a).unwrap();

let mut executor = GraphExecutor::new(CpuBackend::new());
let result = executor.run(&program).unwrap();
assert_eq!(result.shape(), &[2, 3]);
```

## Vector-Jacobian product with `vjp`

```rust
use tenferro::traced_tensor::einsum;
use tenferro::{CpuBackend, GraphCompiler, GraphExecutor, TracedTensor};

let a = TracedTensor::from_vec_col_major(
    vec![2, 3],
    vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
);
let b = TracedTensor::from_vec_col_major(
    vec![3, 2],
    vec![0.5_f64, -1.0, 2.0, 1.5, -0.25, 3.0],
);
let cotangent = TracedTensor::from_vec_col_major(
    vec![2, 2],
    vec![1.0_f64, -0.5, 0.25, 2.0],
);

let mut compiler = GraphCompiler::new();
let y = einsum(&mut compiler, &[&a, &b], "ij,jk->ik").unwrap();
let ct_a = y.vjp(&a, &cotangent);
let program = compiler.compile(&ct_a).unwrap();

let mut executor = GraphExecutor::new(CpuBackend::new());
let result = executor.run(&program).unwrap();
assert_eq!(result.shape(), &[2, 3]);
```

## Jacobian-vector product with `jvp`

```rust
use tenferro::traced_tensor::einsum;
use tenferro::{CpuBackend, GraphCompiler, GraphExecutor, TracedTensor};

let a = TracedTensor::from_vec_col_major(
    vec![2, 3],
    vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
);
let b = TracedTensor::from_vec_col_major(
    vec![3, 2],
    vec![0.5_f64, -1.0, 2.0, 1.5, -0.25, 3.0],
);
let tangent = TracedTensor::from_vec_col_major(
    vec![2, 3],
    vec![1.0_f64, -0.5, 0.25, 0.0, 2.0, -1.0],
);

let mut compiler = GraphCompiler::new();
let y = einsum(&mut compiler, &[&a, &b], "ij,jk->ik").unwrap();
let dy = y.jvp(&a, &tangent);
let program = compiler.compile(&dy).unwrap();

let mut executor = GraphExecutor::new(CpuBackend::new());
let result = executor.run(&program).unwrap();
assert_eq!(result.shape(), &[2, 2]);
```
