# Automatic Differentiation

This document defines the current AD architecture for tenferro-rs.

For math derivations, see [AD Formula Notes](../AD/index.md). For
einsum/frontend integration details, see
[einsum-dyadtensor.md](./einsum-dyadtensor.md).

## Design Goals

1. Keep `chainrules-core` small and stable for downstream custom types.
2. Keep `tenferro-einsum` explicit AD entry points (`tracked_einsum`,
   `dual_einsum`, `einsum_rrule`, `einsum_frule`, `einsum_hvp`).
3. Provide a single reverse-mode tape model that works for tensors and
   downstream custom homogeneous value types.
4. Match PyTorch/LibTorch behavioral expectations for `retain_graph` and
   `create_graph`.
5. Use PyTorch-style tensor scalar semantics: rank-0 tensor means scalar;
   shape `[1]` does not.

## Architecture

### `chainrules-core`

`chainrules-core` defines the value-level AD contracts only:

- `Differentiable`
- `ReverseRule<V>`
- `ForwardRule<V>`
- `AutodiffError`, `AdResult`
- `NodeId`

It does not provide an execution engine.

### `chainrules`

`chainrules` provides engine-independent scalar AD rules and helpers:

- scalar `rrule` / `frule` entrypoints (`add`, `mul`, `sqrt`, `exp`, `sin`, ...)
- `ScalarAd`
- real/complex projection helpers such as `handle_r_to_c_*`

It does not provide a tape or tracked-value runtime.

### `tidu`

`tidu` provides the execution engine:

- `Tape<V>`: reverse-mode tape for homogeneous graphs
- `TrackedValue<V>`: reverse-mode tracked value
- `DualValue<V>`: forward-mode value+tangent wrapper
- `AutogradGraph<V>`

There is no heterogeneous tape surface in the current design. A graph contains
exactly one value type `V`.

Examples:

- `Tape<Tensor<f64>>` is supported
- `Tape<MyType>` is supported if `MyType: Differentiable`
- mixing `Tensor<f64>` and `MyType` in the same tape is unsupported

## Homogeneous Graph Model

The reverse-mode model is intentionally monomorphic.

- Every `Tape<V>` contains only values of type `V`
- Every `TrackedValue<V>` in an `AutogradGraph<V>` shares the same `V`
- Downstream custom-type AD remains supported through `Differentiable`
  implementations and operation-specific rules

This is the only graph model that the current public API supports.

## Public Surface

```rust
pub struct BackwardOptions {
    pub retain_graph: bool,
    pub create_graph: bool,
}

pub struct GradOptions {
    pub retain_graph: bool,
    pub create_graph: bool,
}

pub struct Tape<V: Differentiable> { /* internal */ }
pub struct TrackedValue<V: Differentiable> { /* internal */ }
pub struct DualValue<V: Differentiable> { /* internal */ }

pub fn grad(
    outputs: &[&Tensor],
    inputs: &[&Tensor],
    grad_outputs: Option<&[Tensor]>,
    options: GradOptions,
) -> Result<Vec<Option<Tensor>>>;

pub fn backward(
    outputs: &[&Tensor],
    grad_outputs: Option<&[Tensor]>,
    inputs: &[&Tensor],
    options: BackwardOptions,
) -> Result<()>;
```

## Tensor Scalar Semantics

For tensor-valued APIs, tenferro follows PyTorch conventions.

### Scalar tensors

A tensor scalar is rank-0:

- scalar tensor: `shape=[]`
- one-element vector: `shape=[1]`

`shape=[1]` is not scalar. It participates in normal shape and broadcast rules.

### Elementwise scalar overloads

Tensor expressions such as `tensor + scalar` or `tensor * scalar` are
convenience APIs only. Their semantic model is:

- normalize the scalar to a rank-0 tensor operand, or
- lower it to a backend scalar parameter if the backend has a dedicated fast
  path

The canonical conceptual path is still tensor-tensor elementwise execution.

`tenferro` keeps this boundary explicit in the dynamic API:

- `Tensor` is the canonical dynamic execution payload
- tensor operations apply implicit result-type promotion internally when they
  need a common dtype (`complex` beats `real`, and 64-bit beats 32-bit)
- `Tensor::to_scalar_type(...)` is the explicit cast boundary, analogous
  to PyTorch `tensor.to(dtype)`
- `Tensor::detach()` drops tape metadata while preserving the same dynamic
  tensor object for storage/FFI-style boundaries

Reverse-mode graphs are homogeneous over one runtime-typed tensor payload.
That means mixed-dtype reverse propagation is supported as long as operands
share one reverse graph; gradients are cast back to each input dtype during
pullback.

## Seed Semantics

Reverse-mode implicit seed creation is based on `Differentiable::num_elements()`
rather than tensor rank.

- implicit seed is allowed when `num_elements() == 1`
- otherwise an explicit cotangent seed is required

This applies to:

- `Tape::pullback`
- `Tape::hvp`
- `tenferro::backward`
- `tenferro::grad`

That means:

- a rank-0 tensor usually has `num_elements() == 1`
- a shape-`[1]` tensor also has `num_elements() == 1`
- for tensor ops, only rank-0 is scalar
- for seed omission, any `V` with `num_elements() == 1` qualifies

## `retain_graph` / `create_graph` Contract

Current frontend behavior:

- `retain_graph = false` frees the shared tape after `tenferro::grad` /
  `tenferro::backward`
- `create_graph = true` is currently rejected by the eager `tenferro` frontend
- `tidu::expert::Tape::hvp` remains available for low-level tangent-seeded reverse
  queries

There is no fallback heterogeneous graph path for `V::Tangent != V`.

## DyadTensor Code Layout

The current workspace splits the AD frontend into a thin public facade plus
internal implementation crates:

- `tenferro-dynamic-compute` owns the public dynamic primal surface
- `tenferro` owns the public dynamic AD surface and user-facing helpers such as
  `grad`, `backward`, `pullback`, and the AD-aware `Tensor`
- `tenferro-internal-frontend-core` owns the shared dynamic tensor substrate
  and structured-layout helpers
- `tenferro-internal-ad-core` owns `AdTensor<T>`, homogeneous tape glue, and
  shared AD helper functions that were previously embedded in
  `tenferro/src/ops/common.rs`
- `tenferro-internal-ad-surface` owns the dynamic AD `Tensor` surface,
  `grad`/`backward`/`forward_ad`, and the builder-style linalg wrappers used by
  the public `tenferro` facade
- `tenferro-internal-ad-ops` owns the typed scalar, reduction, and einsum AD
  builders and local pullback helpers used behind `tenferro`
- `tenferro-internal-ad-linalg` owns the typed linalg AD builders, eager
  linalg helpers, and typed linalg AD result structs used by `tenferro`

This keeps the public AD surface operation-first while allowing the heaviest
typed codegen to move out of the `tenferro` facade.

## Context and Mutation Semantics

- `AutogradGraph<V>` is shared through `Arc<Mutex<_>>`
- backward execution is single-threaded per graph in this phase
- `tenferro::backward` accumulates into leaf `.grad()` buffers
- `tenferro::grad` is a query API and does not mutate stored `.grad()` buffers
- `zero_grad()` is explicit per leaf; there is no graph-wide reset helper

## Custom Types

Downstream crates extend the system by implementing `Differentiable` and
providing operation-specific rules.

Minimal example:

```rust
#[derive(Clone, Copy)]
struct MyScalar(f64);

impl Differentiable for MyScalar {
    type Tangent = Self;

    fn zero_tangent(&self) -> Self::Tangent { Self(0.0) }
    fn accumulate_tangent(a: Self::Tangent, b: &Self::Tangent) -> Self::Tangent {
        Self(a.0 + b.0)
    }
    fn num_elements(&self) -> usize { 1 }
    fn seed_cotangent(&self) -> Self::Tangent { Self(1.0) }
}

let tape = tidu::expert::Tape::<MyScalar>::new();
let x = tape.leaf(MyScalar(2.0));
let grads = tape.pullback(&x).unwrap();
assert_eq!(grads.get(x.node_id().unwrap()).unwrap().0, 1.0);
```

## Usage Examples

### Reverse mode on tensors

```rust,ignore
use tidu::expert::Tape;
use std::cell::RefCell;
use std::rc::Rc;
use tenferro_algebra::Standard;
use tenferro_device::LogicalMemorySpace;
use tenferro_einsum::tracked_einsum;
use tenferro_prims::{CpuBackend, CpuContext};
use tenferro_tensor::{MemoryOrder, Tensor};

let tape = Tape::<Tensor<f64>>::new();
let ctx = Rc::new(RefCell::new(CpuContext::new(1)));
let x = tape.leaf(Tensor::ones(
    &[3],
    LogicalMemorySpace::MainMemory,
    MemoryOrder::ColumnMajor,
));
let loss =
    tracked_einsum::<Standard<f64>, CpuBackend>(ctx, "i,i->", &[&x, &x]).unwrap(); // rank-0 tensor
let grads = tape.pullback(&loss).unwrap();
let _gx = grads.get(x.node_id().unwrap()).unwrap();
```

### Forward mode on tensors

```rust,ignore
use tidu::DualValue;
use tenferro_algebra::Standard;
use tenferro_einsum::dual_einsum;
use tenferro_prims::{CpuBackend, CpuContext};

let mut ctx = CpuContext::new(1);
let x = /* Tensor<f64> */;
let dx = /* Tensor<f64> tangent */;
let x_dual = DualValue::with_tangent(x, dx).unwrap();
let y_dual = dual_einsum::<Standard<f64>, CpuBackend>(&mut ctx, "i,i->", &[&x_dual, &x_dual])
    .unwrap();
let _dy = y_dual.tangent();
```

## Testing Requirements

Current AD tests should cover at least:

- homogeneous `Tape<V>` behavior for built-in and downstream custom value types
- non-scalar outputs requiring explicit seeds
- single-element outputs allowing implicit seeds
- `retain_graph` / `create_graph` lifetime behavior
- eager `backward(...)` / `grad(...)` behavior on supported tensor wrappers
- low-level `DualValue` and HVP behavior on supported operation families

## Current Runtime Status

The current eager AD surface uses runtime dispatch and capability-driven
backend contracts in production code.

- `extension/tenferro` no longer routes production paths through
  `with_cpu_runtime(...)`
- linalg and einsum AD entrypoints now dispatch through the relevant family
  traits and runtime slots
- builder `.run()` now relies on an explicit default-runtime holder, while
  reverse-mode bookkeeping stays on one homogeneous runtime-typed reverse
  graph
- many public examples still instantiate `CpuContext` directly because CPU is
  the most complete backend today, not because the API contract is CPU-only

The remaining debt is now mostly breadth:

- unsupported backend capabilities still surface as truthful capability
  failures
- AD formula coverage is broader for some families than others
- GPU custom pointwise/reduction kernels are still incomplete
