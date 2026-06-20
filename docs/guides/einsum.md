# Einsum

`einsum` is a standard extension, not part of `tenferro` core. Add the
`tenferro-einsum` crate and import its extension traits. Traced graph
construction uses `GraphCompilerEinsumExt`; eager execution uses
`EagerEinsumExt`; `tensordot` contraction sugar uses tensor extension traits.
Compiled execution also requires explicit runtime registration for einsum
extension ops.

When working from a local checkout, use paths that match your project layout.
For a scratch crate created directly inside the `tenferro-rs` checkout, include
an empty `[workspace]` table so Cargo does not try to enroll it in the parent
workspace:

```toml
[workspace]
```

Then add the dependencies:

```toml
[dependencies]
tenferro-runtime = { path = "../crates/tenferro-runtime" }
tenferro-cpu = { path = "../crates/tenferro-cpu" }
tenferro-ad = { path = "../crates/tenferro-ad" }
tenferro-einsum = { path = "../crates/tenferro-einsum", features = ["autodiff"] }
```

For published crates, use the same crate set with version requirements:

```toml
[dependencies]
tenferro-runtime = "..."
tenferro-cpu = "..."
tenferro-ad = "..."
tenferro-einsum = { version = "...", features = ["autodiff"] }
```

Graph-only users can omit `tenferro-ad` and the `autodiff` feature. The traced
examples below are fragments; copy them into `fn main() -> Result<(), Box<dyn
std::error::Error>>` when turning them into a standalone `src/main.rs`.

## Traced Matrix Multiply

Use the traced route when einsum should be part of a graph compiled by
`GraphCompiler` and executed by `GraphExecutor`.

```rust
use tenferro_cpu::CpuBackend;
use tenferro_einsum::GraphCompilerEinsumExt;
use tenferro_runtime::{GraphCompiler, GraphExecutor, TracedTensor};

let a = TracedTensor::from_vec_col_major(
    vec![2, 3],
    vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
)?;
let b = TracedTensor::from_vec_col_major(
    vec![3, 2],
    vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
)?;

let mut compiler = GraphCompiler::new();
let c = compiler.einsum(&[&a, &b], "ij,jk->ik").unwrap();
let program = compiler.compile(&c).unwrap();

let mut executor = GraphExecutor::new(CpuBackend::new());
executor.register_extension(tenferro_einsum::register_runtime).unwrap();
let result = executor.run(&program).unwrap();

assert_eq!(result.shape(), &[2, 2]);
assert_eq!(result.as_slice::<f64>().unwrap(), &[22.0, 28.0, 49.0, 64.0]);
```

## EagerTensor

With the `autodiff` feature, `tenferro-einsum` also exposes immediate
`EagerTensor` execution.
The `"i->ii"` form embeds a vector on a diagonal. This is a tenferro extension
to the common NumPy/PyTorch einsum surface; NumPy rejects repeated output
labels in that form.

```rust
use tenferro_ad::{EagerRuntime, Tensor};
use tenferro_einsum::EagerEinsumExt;

let ctx = EagerRuntime::new();
let u = ctx.variable_from(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap()).unwrap();
let v = ctx.variable_from(Tensor::from_vec_col_major(vec![3], vec![3.0_f64, 4.0, 5.0]).unwrap()).unwrap();

let outer = [&u, &v].einsum("i,j->ij").unwrap();
let diag = [&v].einsum("i->ii").unwrap();

assert_eq!(outer.shape(), &[2, 3]);
assert_eq!(
    outer.materialized().unwrap().as_slice::<f64>().unwrap(),
    &[3.0, 6.0, 4.0, 8.0, 5.0, 10.0],
);
assert_eq!(diag.shape(), &[3, 3]);
assert_eq!(
    diag.materialized().unwrap().as_slice::<f64>().unwrap(),
    &[3.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 5.0],
);
```

## Tensordot Sugar

Use `tensordot` when the operation is naturally described as "contract these
axis pairs" instead of by writing explicit labels. `TensorDotAxes::Count(n)`
contracts the last `n` axes of the left tensor with the first `n` axes of the
right tensor. `TensorDotAxes::Axes` accepts explicit axis pairs, including
negative axes.

```rust
use tenferro_cpu::CpuBackend;
use tenferro_einsum::{TensorDotAxes, TracedTensorEinsumExt};
use tenferro_runtime::{GraphCompiler, GraphExecutor, TracedTensor};

let lhs = TracedTensor::from_vec_col_major(
    vec![2, 3],
    vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
);
let rhs = TracedTensor::from_vec_col_major(
    vec![3, 4],
    vec![
        1.0_f64, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
        10.0, 11.0, 12.0,
    ],
);
let out = lhs.tensordot(&rhs, TensorDotAxes::Count(1)).unwrap();

assert_eq!(out.rank, 2);
let mut compiler = GraphCompiler::new();
let program = compiler.compile(&out).unwrap();
let mut executor = GraphExecutor::new(CpuBackend::new());
let result = executor.run(&program).unwrap();

assert_eq!(result.shape(), &[2, 4]);
assert_eq!(
    result.as_slice::<f64>().unwrap(),
    &[22.0, 28.0, 49.0, 64.0, 76.0, 100.0, 103.0, 136.0],
);
```

## Optimization Controls

The default policy chooses an N-ary contraction order automatically. Advanced
users can pass an explicit strategy through `GraphCompilerEinsumExt::einsum_with`.
The public optimizer API is limited to the types needed to express that
choice: `EinsumOptimize`, `ContractionTree`, `ContractionOptimizerOptions`,
`Subscripts`, `NestedEinsum`, and `EinsumSubscripts`.

`EinsumOptimize::Path` accepts a JAX-style positional path over the current
operand list. After each contraction the referenced operands are removed and
the result is appended, so `[(1, 2), (0, 1)]` for `ij,jk,kl->il` contracts the
last two operands first and then contracts that result with the first operand.
This path is shape-independent and can be used with symbolic traced inputs.
`EinsumOptimize::Tree` is for concrete or precomputed `ContractionTree` values;
it requires concrete shapes and is converted to fixed contraction pairs when a
concrete traced op is built.

```rust
use tenferro_runtime::{GraphCompiler, TracedTensor};
use tenferro_einsum::{EinsumOptimize, GraphCompilerEinsumExt};

let a = TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]);
let b = TracedTensor::from_vec_col_major(vec![3, 2], vec![1.0_f64; 6]);

let mut compiler = GraphCompiler::new();
let c = compiler.einsum_with(
    &[&a, &b],
    "ij,jk->ik",
    EinsumOptimize::False,
).unwrap();

assert_eq!(c.try_concrete_shape(), Some(vec![2, 2]));
```

## Cache Management

Einsum uses the shared extension cache infrastructure from `tenferro-runtime`.
Compile-time extension caches live on `GraphCompiler`; runtime
contraction-plan and inner execution-program caches live on `GraphExecutor`
and `EagerRuntime`.
Einsum plan cache identity includes the planning policy or explicit path, not
only the subscripts and shapes. Traced extension operation identity also includes
those planner options and paths, so two calls with different policies are not
treated as identical extension ops.

Use `tenferro_einsum::EINSUM_EXTENSION_FAMILY_ID` with
`ExtensionCacheSelector` when you need to inspect or clear only einsum cache
entries.

## Autodiff

With the `autodiff` feature, einsum VJP rules preserve the primal planning
policy. Explicit positional paths are remapped to the VJP operand order so the
gradient contraction inherits the caller's intended plan instead of falling
back to an unrelated default.
