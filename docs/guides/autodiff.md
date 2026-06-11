# Autodiff

tenferro supports two autodiff workflows on top of the same dense tensor stack:

- PyTorch-like eager execution with `EagerTensor::backward()` on scalar losses,
- JAX-like `grad`, `vjp`, and `jvp` on `TracedTensor` graphs.

This page focuses on traced graph workflows that you compile and execute
explicitly. For eager forward execution and scalar loss accumulation semantics, see
[Eager Operations](eager-operations.md).

## Setup

```toml
[dependencies]
tenferro-ad = { path = "../crates/tenferro-ad" }
tenferro-runtime = { path = "../crates/tenferro-runtime" }
tenferro-cpu = { path = "../crates/tenferro-cpu" }
```

Extension AD examples also need that extension crate's AD feature. For linalg:

```toml
tenferro-linalg = { path = "../crates/tenferro-linalg", features = ["autodiff"] }
```

- `grad` for reverse mode on scalar losses
- `vjp` for vector-Jacobian products
- `jvp` for Jacobian-vector products
- Higher-order AD via composition, such as `jvp(grad(f))` for HVPs

For complex tensors, reverse-mode cotangents use tenferro's Hermitian
real-inner-product convention. See [Complex Autodiff](complex-ad.md) before
comparing `grad` or `vjp` values directly against JAX.

Use `tenferro_ad::AdContext` to own the AD rule set used by a transform. Core
tensor primitive rules are always available. Extension crates can provide owned
JVP/VJP rule sets for their operations; `tenferro-linalg` exposes these through
its `autodiff` feature.

## `AdContext` And `TracedTensorAdExt`

There are two public traced-AD entry points over the same graph transforms:

| Entry point | Use when |
| --- | --- |
| `AdContext` | You want explicit ownership of the rule set, especially when adding extension AD rules such as `tenferro_linalg::ad_rules()`. |
| `TracedTensorAdExt` | You want compact method syntax such as `loss.grad(&x)?`, `y.vjp(&x, &ct)`, or `y.jvp(&x, &dx)` for small core-rule examples. |

Prefer `AdContext` in reusable code and in examples that depend on extension
rules. The extension trait is convenience syntax for the same traced transforms;
the [JAX-style traced tutorial](../tutorials/traced-autodiff-jax-style.md) uses
it to keep the first example short.

## Reverse-mode gradient with `grad`

```rust
use tenferro_ad::AdContext;
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{GraphCompiler, GraphExecutor, TracedTensor};

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
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{GraphCompiler, GraphExecutor, TracedTensor};

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
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{GraphCompiler, GraphExecutor, TracedTensor};

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
// For y = A * B, the cotangent with respect to A is cotangent * B^T.
assert_eq!(
    result.as_slice::<f64>().unwrap(),
    &[0.875, 2.75, -1.0625, 0.0, 2.75, 5.0],
);
```

## Jacobian-vector product with `jvp`

```rust
use tenferro_ad::AdContext;
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{GraphCompiler, GraphExecutor, TracedTensor};

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
// For y = A * B, the directional derivative with respect to A is dA * B.
assert_eq!(
    result.as_slice::<f64>().unwrap(),
    &[4.25, -2.25, 7.4375, -3.75],
);
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
