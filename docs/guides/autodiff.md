# Autodiff

tenferro supports two autodiff workflows on top of the same dense tensor stack:

- PyTorch-like eager execution with scalar-loss `EagerTensor::backward()`,
- JAX-like transform AD on `TracedTensor`.

This page focuses on graph transforms that you compile and execute explicitly.
For eager forward execution and scalar-loss accumulation semantics, see
[Eager Operations](eager-operations.md).

- `grad` for scalar-loss reverse mode
- `vjp` for vector-Jacobian products
- `jvp` for Jacobian-vector products
- Higher-order AD via composition, such as `jvp(grad(f))` for HVPs

Use `tenferro_ad::AdContext` to own the AD rule set used by a transform. Core
tensor primitive rules are always available. Extension crates can provide owned
JVP/VJP rule sets for their operations; `tenferro-linalg` exposes these through
its `autodiff` feature.

## Reverse-mode gradient with `grad`

```rust
use tenferro_ad::AdContext;
use tenferro_runtime::{CpuBackend, GraphCompiler, GraphExecutor, TracedTensor};

let x = TracedTensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]);
let loss = (&x * &x).reduce_sum(&[0]);
let ad = AdContext::builder().with_core_rules().build().unwrap();
let grad = ad.grad(&loss, &x).unwrap();

let mut compiler = GraphCompiler::new();
let program = compiler.compile(&grad).unwrap();
let mut executor = GraphExecutor::new(CpuBackend::new());
let result = executor.run(&program).unwrap();

assert_eq!(result.shape(), &[3]);
assert_eq!(result.as_slice::<f64>().unwrap(), &[2.0, 4.0, 6.0]);
```

## Gradient through linalg

```rust
use tenferro_ad::AdContext;
use tenferro_runtime::{CpuBackend, GraphCompiler, GraphExecutor, TracedTensor};

let mut compiler = GraphCompiler::new();
let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![4.0_f64, 0.0, 0.0, 9.0]);
let factor = tenferro_linalg::traced_tensor::cholesky(&a).unwrap();
let ad = AdContext::builder()
    .with_extension_rules(tenferro_linalg::ad_rules().unwrap())
    .build()
    .unwrap();
let loss = factor.reduce_sum(&[0, 1]);
let grad_a = ad.grad(&loss, &a).unwrap();
let program = compiler.compile(&grad_a).unwrap();

let mut executor = GraphExecutor::new(CpuBackend::new());
executor.register_extension(tenferro_linalg::register_runtime).unwrap();
let result = executor.run(&program).unwrap();
assert_eq!(result.shape(), &[2, 2]);
```

## Vector-Jacobian product with `vjp`

```rust
use tenferro_ad::AdContext;
use tenferro_runtime::{CpuBackend, GraphCompiler, GraphExecutor, TracedTensor};

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
let y = tenferro_runtime::traced_tensor::matmul(&a, &b);
let ad = AdContext::builder().with_core_rules().build().unwrap();
let ct_a = ad.vjp(&y, &a, &cotangent).unwrap();
let program = compiler.compile(&ct_a).unwrap();

let mut executor = GraphExecutor::new(CpuBackend::new());
let result = executor.run(&program).unwrap();
assert_eq!(result.shape(), &[2, 3]);
```

## Jacobian-vector product with `jvp`

```rust
use tenferro_ad::AdContext;
use tenferro_runtime::{CpuBackend, GraphCompiler, GraphExecutor, TracedTensor};

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
let y = tenferro_runtime::traced_tensor::matmul(&a, &b);
let ad = AdContext::builder().with_core_rules().build().unwrap();
let dy = ad.jvp(&y, &a, &tangent).unwrap();
let program = compiler.compile(&dy).unwrap();

let mut executor = GraphExecutor::new(CpuBackend::new());
let result = executor.run(&program).unwrap();
assert_eq!(result.shape(), &[2, 2]);
```

## Extension AD Rules

External operations can participate in autodiff when the extension crate
provides the corresponding rules and the caller includes those rules in an
`AdContext`. If an extension does not support a given AD path, tenferro reports
that path as unsupported rather than silently returning an incorrect gradient.

The process-global extension-rule registration API is retained as a
compatibility bridge for older helpers. New code should prefer explicit context
ownership. See [Custom Tensor Operations](custom-operations.md) for the
extension model.
