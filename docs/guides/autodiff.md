# Autodiff

tenferro supports two autodiff workflows on top of the same dense tensor stack:

- PyTorch-like eager execution with `EagerTensor::backward()` on scalar losses
  and `EagerRuntime` functional `grad`, `vjp`, and `jvp`,
- JAX-like `grad`, `vjp`, and `jvp` on `TracedTensor` graphs.

This page focuses on traced graph workflows that you compile and execute
explicitly. For eager forward execution, scalar loss accumulation, and
functional eager transforms, see [Eager Operations](eager-operations.md).

## Setup

For a published build, depend on the AD, runtime, and backend crates:

```toml
[dependencies]
tenferro-ad = "..."
tenferro-runtime = "..."
tenferro-cpu = "..."
```

When working from a local checkout, use paths that match your project layout.
For a scratch crate created directly inside the `tenferro-rs` checkout, include
an empty `[workspace]` table:

```toml
[workspace]

[dependencies]
tenferro-ad = { path = "../crates/tenferro-ad" }
tenferro-runtime = { path = "../crates/tenferro-runtime" }
tenferro-cpu = { path = "../crates/tenferro-cpu" }
```

Extension AD examples also need that extension crate's AD feature. For linalg:

```toml
tenferro-linalg = { path = "../crates/tenferro-linalg", features = ["autodiff"] }
```

In code that uses extension operations during traced execution, also install
the extension module on the `Runtime` and add the extension rule set to
`AdContext` when differentiating through that operation family.

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
| `AdContext` | You want explicit ownership of the rule set, especially when adding extension AD rules such as `tenferro_linalg::semantic_ad_rules()`. |
| `TracedTensorAdExt` | You want compact method syntax such as `loss.grad(&x)?`, `y.vjp(&x, &ct)?`, or `y.jvp(&x, &dx)?` for small core-rule examples. |

Prefer `AdContext` in reusable code and in examples that depend on extension
rules. The extension trait is convenience syntax for the same traced transforms;
the [JAX-style traced tutorial](../tutorials/traced-autodiff-jax-style.md) uses
it to keep the first example short.

## Reverse-mode gradient with `grad`

```rust
use tenferro_ad::AdContext;
use tenferro_cpu::CpuBackend;
use tenferro_linalg::TracedTensorLinalgExt;
use tenferro_runtime::{GraphCompiler, Runtime, TracedTensor};

let x = TracedTensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]);
let loss = (&x * &x).reduce_sum(Some(&[0]));
let ad = AdContext::builder().build().unwrap();
let grad = ad.grad(&loss, &x).unwrap();

let mut compiler = GraphCompiler::new();
let program = compiler.compile(&grad).unwrap();
let mut builder = Runtime::builder();
builder
    .register_engine(tenferro_cpu::runtime_engine_registration(&CpuBackend::new()).unwrap())
    .unwrap();
let runtime = builder.build().unwrap();
let mut outputs = runtime.run_compiled(&program, &[]).unwrap();
let result = outputs.remove(0);

assert_eq!(result.shape(), &[3]);
assert_eq!(result.as_slice::<f64>().unwrap(), &[2.0, 4.0, 6.0]);
```

## Gradient through linalg

```rust
use tenferro_ad::AdContext;
use tenferro_cpu::{runtime_engine_id, runtime_engine_registration, CpuBackend};
use tenferro_runtime::{GraphCompiler, Runtime, TracedTensor};

let mut compiler = GraphCompiler::new();
let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![4.0_f64, 0.0, 0.0, 9.0]);
let factor = a.cholesky().unwrap();
let ad = AdContext::builder()
    .with_semantic_extension_rules(tenferro_linalg::semantic_ad_rules().unwrap())
    .unwrap()
    .build()
    .unwrap();
let loss = factor.reduce_sum(Some(&[0, 1]));
let grad_a = ad.grad(&loss, &a).unwrap();
let program = compiler.compile(&grad_a).unwrap();

let backend = CpuBackend::new();
let mut builder = Runtime::builder();
builder
    .register_engine(runtime_engine_registration(&backend).unwrap())
    .unwrap();
builder
    .install_extension_module(
        tenferro_linalg::extension_module::<CpuBackend>(runtime_engine_id().unwrap()).unwrap(),
    )
    .unwrap();
let runtime = builder.build().unwrap();
let mut outputs = runtime.run_compiled(&program, &[]).unwrap();
let result = outputs.remove(0);
assert_eq!(result.shape(), &[2, 2]);
```

## Vector-Jacobian product with `vjp`

```rust
use tenferro_ad::AdContext;
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{GraphCompiler, Runtime, TracedTensor};

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
let y = a.matmul(&b).unwrap();
let ad = AdContext::builder().build().unwrap();
let ct_a = ad.vjp(&y, &a, &cotangent).unwrap();
let program = compiler.compile(&ct_a).unwrap();

let mut builder = Runtime::builder();
builder
    .register_engine(tenferro_cpu::runtime_engine_registration(&CpuBackend::new()).unwrap())
    .unwrap();
let runtime = builder.build().unwrap();
let mut outputs = runtime.run_compiled(&program, &[]).unwrap();
let result = outputs.remove(0);
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
use tenferro_runtime::{GraphCompiler, Runtime, TracedTensor};

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
let y = a.matmul(&b).unwrap();
let ad = AdContext::builder().build().unwrap();
let dy = ad.jvp(&y, &a, &tangent).unwrap();
let program = compiler.compile(&dy).unwrap();

let mut builder = Runtime::builder();
builder
    .register_engine(tenferro_cpu::runtime_engine_registration(&CpuBackend::new()).unwrap())
    .unwrap();
let runtime = builder.build().unwrap();
let mut outputs = runtime.run_compiled(&program, &[]).unwrap();
let result = outputs.remove(0);
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

There is no process-global extension-rule registry. Extension AD must be owned
by an explicit `AdContext` so tests and applications cannot accidentally depend
on hidden process state. See [Custom Tensor Operations](custom-operations.md)
for the extension model.
