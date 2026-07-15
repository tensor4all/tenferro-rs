# Einsum

`einsum` is a standard extension, not part of `tenferro` core. Add the
`tenferro-einsum` crate and import its extension traits. Concrete tensor
execution uses `TensorEinsumExt` and `TypedTensorEinsumExt` for owned inputs,
and `TensorReadEinsumExt` and `TypedTensorReadEinsumExt` for borrowed inputs;
preallocated-output execution uses the matching `*IntoExt` traits;
repeated-shape concrete workloads can use `ConcreteEinsumPlan`. Traced graph
construction uses `GraphCompilerEinsumExt`;
autodiff eager execution uses `EagerEinsumExt`; `tensordot` contraction sugar
uses tensor extension traits. Compiled traced execution also requires explicit
runtime registration for einsum extension ops.

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
tenferro-tensor = { path = "../crates/tenferro-tensor" }
tenferro-cpu = { path = "../crates/tenferro-cpu" }
tenferro-einsum = { path = "../crates/tenferro-einsum", features = ["autodiff"] }

# Only needed for EagerTensor/autodiff examples.
tenferro-ad = { path = "../crates/tenferro-ad" }
num-complex = "0.4"
```

For published crates, use the same crate set with version requirements:

```toml
[dependencies]
tenferro-runtime = "..."
tenferro-tensor = "..."
tenferro-cpu = "..."
tenferro-einsum = { version = "...", features = ["autodiff"] }

# Only needed for EagerTensor/autodiff examples.
tenferro-ad = "..."
num-complex = "0.4"
```

Concrete and graph-only users can omit `tenferro-ad` and the `autodiff`
feature. Enable `tenferro-einsum`'s `autodiff` feature when using
`EagerEinsumExt` or einsum AD rules. The traced examples below are fragments;
copy them into `fn main() -> Result<(), Box<dyn std::error::Error>>` when
turning them into a standalone `src/main.rs`.

## Concrete Tensor And TypedTensor

Use the concrete route when you have `Tensor` or `TypedTensor` values and want
to run the contraction immediately on an explicit backend without autodiff.
Arrays such as `[&lhs, &rhs]` implement the extension traits directly; their
receiver type is the fixed-size array of borrowed tensor references.

```rust
use num_complex::Complex64;
use tenferro_cpu::CpuBackend;
use tenferro_einsum::{
    TensorEinsumExt, TensorEinsumIntoExt, TypedTensorEinsumExt,
    TypedTensorEinsumIntoExt, TypedTensorReadEinsumIntoExt,
};
use tenferro_tensor::{
    Tensor, TensorWrite, TypedTensor, TypedTensorView, TypedTensorViewMut,
    TypedTensorWrite,
};

let lhs = Tensor::from_vec_col_major(
    vec![2, 3],
    vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
)?;
let rhs = Tensor::from_vec_col_major(
    vec![3, 2],
    vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
)?;
let mut backend = CpuBackend::new();
let product = [&lhs, &rhs].einsum("ij,jk->ik", &mut backend)?;
assert_eq!(product.as_slice::<f64>()?, &[22.0, 28.0, 49.0, 64.0]);

let mut product_out = Tensor::from_vec_col_major(vec![2, 2], vec![0.0_f64; 4])?;
[&lhs, &rhs].einsum_into(
    "ij,jk->ik",
    &mut backend,
    TensorWrite::from_tensor(&mut product_out),
)?;
assert_eq!(product_out.as_slice::<f64>()?, &[22.0, 28.0, 49.0, 64.0]);

let complex_lhs = TypedTensor::<Complex64>::from_vec_col_major(
    vec![2, 2],
    vec![
        Complex64::new(1.0, 1.0),
        Complex64::new(2.0, -1.0),
        Complex64::new(3.0, 0.0),
        Complex64::new(4.0, 2.0),
    ],
)?;
let complex_rhs = TypedTensor::<Complex64>::from_vec_col_major(
    vec![2, 1],
    vec![Complex64::new(5.0, 0.0), Complex64::new(6.0, -1.0)],
)?;
let complex = [&complex_lhs, &complex_rhs].einsum("ij,jk->ik", &mut backend)?;
assert_eq!(
    complex.as_slice()?,
    &[Complex64::new(23.0, 2.0), Complex64::new(36.0, 3.0)],
);

let borrowed = TypedTensorView::from_slice([2, 2], [1, 2], 0, complex_lhs.as_slice()?)?;
let borrowed_rhs = complex_rhs.as_view();
let mut borrowed_storage = [Complex64::new(0.0, 0.0); 4];
let borrowed_out =
    TypedTensorViewMut::from_slice([2, 1], [2, 4], 1, &mut borrowed_storage)?;
[borrowed, borrowed_rhs].einsum_read_into(
    "ij,jk->ik",
    &mut backend,
    TypedTensorWrite::from_view(borrowed_out),
)?;
assert_eq!(
    [borrowed_storage[1], borrowed_storage[3]],
    [Complex64::new(23.0, 2.0), Complex64::new(36.0, 3.0)],
);
# Ok::<(), tenferro_tensor::Error>(())
```

## TensorRead And Prepared Plans

Use `TensorReadEinsumExt` for dtype-erased borrowed inputs and
`TypedTensorReadEinsumExt` for typed borrowed views. The `_read` suffix is
reserved for these read-oriented APIs; compact owned tensor inputs use the
unsuffixed `einsum` method. Typed `_into` methods accept `TypedTensorWrite`, so
the destination can be either an owned `TypedTensor` or a
`TypedTensorViewMut`.

Use `ConcreteEinsumPlan` when the same subscripts, dtypes, and shapes are
executed repeatedly. Preparing the plan parses and optimizes the contraction
tree once; each execution validates the input count, dtype, and shape before
running the stored tree.

```rust
use tenferro_cpu::CpuBackend;
use tenferro_einsum::{ConcreteEinsumPlan, TensorReadEinsumExt, TensorReadEinsumIntoExt};
use tenferro_tensor::{Tensor, TensorRead, TensorView, TensorWrite, TypedTensorView};

let matrix_data = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
let matrix = TypedTensorView::from_slice([2, 3], [3, 1], 0, &matrix_data)?;
let vector = Tensor::from_vec_col_major(vec![3], vec![10.0_f64, 20.0, 30.0])?;
let inputs = [
    TensorRead::from_view(TensorView::F64(matrix)),
    TensorRead::from_tensor(&vector),
];

let mut backend = CpuBackend::new();
let result = inputs.einsum_read("ij,j->i", &mut backend)?;
assert_eq!(result.as_slice::<f64>()?, &[140.0, 320.0]);

let plan = ConcreteEinsumPlan::prepare_read(inputs.clone(), "ij,j->i")?;
let planned = plan.execute_read(inputs, &mut backend)?;
assert_eq!(planned.as_slice::<f64>()?, &[140.0, 320.0]);

let mut planned_out = Tensor::from_vec_col_major(vec![2], vec![0.0_f64; 2])?;
let inputs = [
    TensorRead::from_view(TensorView::F64(matrix)),
    TensorRead::from_tensor(&vector),
];
plan.execute_read_into(
    inputs,
    &mut backend,
    TensorWrite::from_tensor(&mut planned_out),
)?;
assert_eq!(planned_out.as_slice::<f64>()?, &[140.0, 320.0]);
# Ok::<(), tenferro_tensor::Error>(())
```

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
