# Einsum

For choosing between ordinary calls, compatible plan reuse and programmatic
labels, start with [Ordinary calls and prepared execution](ordinary-and-prepared-execution.md).

`einsum` is a standard extension, not part of `tenferro` core. Add the
`tenferro-einsum` crate and import its extension traits. Concrete tensor
execution uses `TensorEinsumExt` and `TypedTensorEinsumExt` for owned inputs,
and `TensorReadEinsumExt` and `TypedTensorReadEinsumExt` for borrowed inputs;
preallocated-output execution uses the matching `*IntoExt` traits;
repeated-shape concrete workloads can use `ConcreteEinsumPlan`. Traced graph
construction uses `TraceContextEinsumExt`;
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

<!-- snippet-source: docs/tutorial-code/src/bin/math_snippets.rs#einsum_12 -->
```rust
use num_complex::Complex64;
use tenferro_cpu::CpuBackend;
use tenferro_einsum::{
    TensorEinsumExt, TensorEinsumIntoExt, TypedTensorEinsumExt,
    TypedTensorEinsumIntoExt, TypedTensorReadEinsumIntoExt,
};
use tenferro_tensor::{
    BackendSessionHost, Tensor, TensorWrite, TypedTensor, TypedTensorView,
    TypedTensorViewMut, TypedTensorWrite,
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
let product = backend.with_backend_session(|session| {
    [&lhs, &rhs].einsum("ij,jk->ik", session)
})?;
assert_eq!(product.as_slice::<f64>()?, &[22.0, 28.0, 49.0, 64.0]);

let mut product_out = Tensor::from_vec_col_major(vec![2, 2], vec![0.0_f64; 4])?;
backend.with_backend_session(|session| {
    [&lhs, &rhs].einsum_into(
        "ij,jk->ik",
        session,
        TensorWrite::from_tensor(&mut product_out),
    )
})?;
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
let complex = backend.with_backend_session(|session| {
    [&complex_lhs, &complex_rhs].einsum("ij,jk->ik", session)
})?;
assert_eq!(
    complex.as_slice()?,
    &[Complex64::new(23.0, 2.0), Complex64::new(36.0, 3.0)],
);

let borrowed = TypedTensorView::from_slice([2, 2], [1, 2], 0, complex_lhs.as_slice()?)?;
let borrowed_rhs = complex_rhs.as_view();
let mut borrowed_storage = [Complex64::new(0.0, 0.0); 4];
let borrowed_out =
    TypedTensorViewMut::from_slice([2, 1], [2, 4], 1, &mut borrowed_storage)?;
backend.with_backend_session(|session| {
    [borrowed, borrowed_rhs].einsum_read_into(
        "ij,jk->ik",
        session,
        TypedTensorWrite::from_view(borrowed_out),
    )
})?;
assert_eq!(
    [borrowed_storage[1], borrowed_storage[3]],
    [Complex64::new(23.0, 2.0), Complex64::new(36.0, 3.0)],
);
```
<!-- end-snippet-source -->

### Ellipsis and programmatic notation

Flat string notation supports one NumPy-style ellipsis per term. Ellipsis
axes right-align and broadcast dimensions of size one. The programmatic
`EinsumNotation` form resolves to the same canonical labels before planning.

> **Warning:** string equations require exactly one explicit `->`. Omitting it
> returns `Error::InvalidSubscripts` during parsing. Parenthesized contraction
> order containing ellipsis is also rejected during parsing; use flat `...`
> notation or `EinsumNotation` instead.

<!-- snippet-source: docs/tutorial-code/src/bin/math_snippets.rs#einsum_20 -->
```rust
use tenferro_cpu::CpuBackend;
use tenferro_einsum::{EinsumAxis, EinsumNotation, TensorEinsumExt};
use tenferro_tensor::{BackendSessionHost, Tensor};

let lhs = Tensor::from_vec_col_major(vec![2, 2, 3], vec![1.0_f64; 12])?;
let rhs = Tensor::from_vec_col_major(vec![1, 3, 2], vec![1.0_f64; 6])?;
let notation = EinsumNotation::new(
    &[
        &[EinsumAxis::Ellipsis, EinsumAxis::Label(0), EinsumAxis::Label(1)],
        &[EinsumAxis::Ellipsis, EinsumAxis::Label(1), EinsumAxis::Label(2)],
    ],
    &[EinsumAxis::Ellipsis, EinsumAxis::Label(0), EinsumAxis::Label(2)],
);
let mut backend = CpuBackend::new();
let string_result = backend.with_backend_session(|session| {
    [&lhs, &rhs].einsum("...ij,...jk->...ik", session)
})?;
let programmatic_result = backend.with_backend_session(|session| {
    [&lhs, &rhs].einsum_notation(&notation, session)
})?;
assert_eq!(string_result.shape(), &[2, 2, 2]);
assert_eq!(string_result.as_slice::<f64>()?, &[3.0; 8]);
assert_eq!(programmatic_result.as_slice::<f64>()?, &[3.0; 8]);
```
<!-- end-snippet-source -->

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

<!-- snippet-source: docs/tutorial-code/src/bin/math_snippets.rs#einsum_13 -->
```rust
use tenferro_cpu::CpuBackend;
use tenferro_einsum::{ConcreteEinsumPlan, TensorReadEinsumExt, TensorReadEinsumIntoExt};
use tenferro_tensor::{BackendSessionHost, Tensor, TensorRead, TensorView, TensorWrite, TypedTensorView};

let matrix_data = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
let matrix = TypedTensorView::from_slice([2, 3], [3, 1], 0, &matrix_data)?;
let vector = Tensor::from_vec_col_major(vec![3], vec![10.0_f64, 20.0, 30.0])?;
let inputs = [
    TensorRead::from_view(TensorView::F64(matrix)),
    TensorRead::from_tensor(&vector),
];

let mut backend = CpuBackend::new();
let result = backend.with_backend_session(|session| {
    inputs.einsum_read("ij,j->i", session)
})?;
assert_eq!(result.as_slice::<f64>()?, &[140.0, 320.0]);

let plan = ConcreteEinsumPlan::prepare_read(inputs.clone(), "ij,j->i")?;
let planned = backend
    .with_backend_session(|session| plan.execute_read(inputs, session))?;
assert_eq!(planned.as_slice::<f64>()?, &[140.0, 320.0]);

let mut planned_out = Tensor::from_vec_col_major(vec![2], vec![0.0_f64; 2])?;
let matrix = TypedTensorView::from_slice([2, 3], [3, 1], 0, &matrix_data)?;
let inputs = [
    TensorRead::from_view(TensorView::F64(matrix)),
    TensorRead::from_tensor(&vector),
];
backend.with_backend_session(|session| {
    plan.execute_read_into(
        inputs,
        session,
        TensorWrite::from_tensor(&mut planned_out),
    )
})?;
assert_eq!(planned_out.as_slice::<f64>()?, &[140.0, 320.0]);
```
<!-- end-snippet-source -->

## Traced Matrix Multiply

Use the traced route when einsum should be part of a graph compiled by
`GraphCompiler` and executed by `Runtime::run_compiled`.

<!-- snippet-source: docs/tutorial-code/src/bin/math_snippets.rs#einsum_14 -->
```rust
use tenferro_cpu::{runtime_engine_id, runtime_engine_registration, CpuBackend};
use tenferro_einsum::TraceContextEinsumExt;
use tenferro_runtime::program::ProgramInputSpec;
use tenferro_runtime::{GraphCompiler, Runtime, Tensor, TraceContext};

let a = Tensor::from_vec_col_major(
    vec![2, 3],
    vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
)?;
let b = Tensor::from_vec_col_major(
    vec![3, 2],
    vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
)?;

let mut trace = TraceContext::new();
let a_value = trace.input(ProgramInputSpec::new(a.dtype(), [2.into(), 3.into()]))?;
let b_value = trace.input(ProgramInputSpec::new(b.dtype(), [3.into(), 2.into()]))?;
let c = trace.einsum(&[a_value, b_value], "ij,jk->ik")?;
let graph = trace.finish(&[c])?;
let program = GraphCompiler::new().compile_traced_graph(&graph)?;

let backend = CpuBackend::new();
let mut builder = Runtime::builder();
builder.register_engine(runtime_engine_registration(&backend)?)?;
builder.install_extension_module(tenferro_einsum::extension_module::<CpuBackend>(
    runtime_engine_id()?,
)?)?;
let runtime = builder.build()?;
let mut outputs = runtime.run_compiled(&program, &[&a, &b])?;
let result = outputs.remove(0);

assert_eq!(result.shape(), &[2, 2]);
assert_eq!(result.as_slice::<f64>()?, &[22.0, 28.0, 49.0, 64.0]);
```
<!-- end-snippet-source -->

## EagerTensor

With the `autodiff` feature, `tenferro-einsum` also exposes immediate
`EagerTensor` execution.
The `"i->ii"` form embeds a vector on a diagonal. This is a tenferro extension
to the common NumPy/PyTorch einsum surface; NumPy rejects repeated output
labels in that form.

<!-- snippet-source: docs/tutorial-code/src/bin/math_snippets.rs#einsum_15 -->
```rust
use tenferro_ad::{EagerRuntime, Tensor};
use tenferro_einsum::EagerEinsumExt;

let ctx = EagerRuntime::new()?;
let u = ctx.variable_from(Tensor::from_vec_col_major(
    vec![2],
    vec![1.0_f64, 2.0],
)?)?;
let v = ctx.variable_from(Tensor::from_vec_col_major(
    vec![3],
    vec![3.0_f64, 4.0, 5.0],
)?)?;

let outer = [&u, &v].einsum("i,j->ij")?;
let diag = [&v].einsum("i->ii")?;

assert_eq!(outer.shape(), &[2, 3]);
let outer_tensor = outer.to_tensor()?;
assert_eq!(
    outer_tensor.as_slice::<f64>()?,
    &[3.0, 6.0, 4.0, 8.0, 5.0, 10.0],
);
assert_eq!(diag.shape(), &[3, 3]);
let diag_tensor = diag.to_tensor()?;
assert_eq!(
    diag_tensor.as_slice::<f64>()?,
    &[3.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 5.0],
);
```
<!-- end-snippet-source -->

## Tensordot Sugar

Use `tensordot` when the operation is naturally described as "contract these
axis pairs" instead of by writing explicit labels. `TensorDotAxes::Count(n)`
contracts the last `n` axes of the left tensor with the first `n` axes of the
right tensor. `TensorDotAxes::Axes` accepts explicit axis pairs, including
negative axes.

<!-- snippet-source: docs/tutorial-code/src/bin/math_snippets.rs#einsum_16 -->
```rust
use tenferro_cpu::{runtime_engine_id, runtime_engine_registration, CpuBackend};
use tenferro_einsum::{TensorDotAxes, TracedTensorEinsumExt};
use tenferro_runtime::{GraphCompiler, Runtime, TracedTensor};

let lhs = TracedTensor::from_vec_col_major(
    vec![2, 3],
    vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
)?;
let rhs = TracedTensor::from_vec_col_major(
    vec![3, 4],
    vec![
        1.0_f64, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
        10.0, 11.0, 12.0,
    ],
)?;
let out = lhs.tensordot(&rhs, TensorDotAxes::Count(1))?;

assert_eq!(out.concrete_shape()?, vec![2, 4]);
let mut compiler = GraphCompiler::new();
let program = compiler.compile(&out)?;
let backend = CpuBackend::new();
let mut builder = Runtime::builder();
builder.register_engine(runtime_engine_registration(&backend)?)?;
builder.install_extension_module(tenferro_einsum::extension_module::<CpuBackend>(
    runtime_engine_id()?,
)?)?;
let runtime = builder.build()?;
let mut outputs = runtime.run_compiled(&program, &[])?;
let result = outputs.remove(0);

assert_eq!(result.shape(), &[2, 4]);
assert_eq!(
    result.as_slice::<f64>()?,
    &[22.0, 28.0, 49.0, 64.0, 76.0, 100.0, 103.0, 136.0],
);
```
<!-- end-snippet-source -->

## Optimization Controls

The default policy chooses an N-ary contraction order automatically. Advanced
users can pass an explicit strategy through `TraceContextEinsumExt::einsum_with`.
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

<!-- snippet-source: docs/tutorial-code/src/bin/math_snippets.rs#einsum_17 -->
```rust
use tenferro_einsum::{EinsumOptimize, TraceContextEinsumExt};
use tenferro_ops::dim_expr::DimExpr;
use tenferro_runtime::program::ProgramInputSpec;
use tenferro_runtime::{DType, TraceContext};

let mut trace = TraceContext::new();
let a = trace.input(ProgramInputSpec::new(
    DType::F64,
    DimExpr::from_concrete(&[2, 3]),
))?;
let b = trace.input(ProgramInputSpec::new(
    DType::F64,
    DimExpr::from_concrete(&[3, 2]),
))?;
let c = trace.einsum_with(
    &[a, b],
    "ij,jk->ik",
    EinsumOptimize::False,
)?;

let graph = trace.finish(&[c])?;
let metadata = graph
    .program()
    .value_metadata(graph.program().outputs()[0])?;
assert_eq!(metadata.shape().len(), 2);
```
<!-- end-snippet-source -->

## Cache Management

Einsum uses the shared extension cache infrastructure from `tenferro-runtime`.
Compile-time extension caches live on `GraphCompiler`; runtime
contraction-plan caches live on installed runtime cache owners and
`EagerRuntime`.
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
