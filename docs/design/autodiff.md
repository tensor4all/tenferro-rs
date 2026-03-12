# Automatic Differentiation

This document defines the current AD architecture for tenferro-rs.

For math derivations, see [AD Formula Notes](../AD/index.md). For
einsum/dyadtensor integration details, see
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

`chainrules` provides the execution engine:

- `Tape<V>`: reverse-mode tape for homogeneous graphs
- `TrackedTensor<V>`: reverse-mode tracked value
- `DualTensor<V>`: forward-mode value+tangent wrapper
- `Variable<V>`: torch-like reverse-mode wrapper with `.grad()` / `.hvp()`
- `BackwardOptions<V>`
- `AutogradContext<V>`

There is no heterogeneous tape surface in the current design. A graph contains
exactly one value type `V`.

Examples:

- `Tape<Tensor<f64>>` is supported
- `Tape<MyType>` is supported if `MyType: Differentiable`
- mixing `Tensor<f64>` and `MyType` in the same tape is unsupported

## Homogeneous Graph Model

The reverse-mode model is intentionally monomorphic.

- Every `Tape<V>` contains only values of type `V`
- Every `Variable<V>` in an `AutogradContext<V>` shares the same `V`
- Downstream custom-type AD remains supported through `Differentiable`
  implementations and operation-specific rules

This is the only graph model that the current public API supports.

## Public Surface

```rust
pub struct BackwardOptions<V: Differentiable> {
    pub retain_graph: Option<bool>,
    pub create_graph: bool,
    pub seed_grad: Option<V::Tangent>,
}

pub struct Tape<V: Differentiable> { /* internal */ }
pub struct TrackedTensor<V: Differentiable> { /* internal */ }
pub struct DualTensor<V: Differentiable> { /* internal */ }
pub struct Variable<V: Differentiable> { /* internal */ }

pub mod autograd {
    pub fn add<V>(lhs: &Variable<V>, rhs: &Variable<V>) -> AdResult<Variable<V>>;
    pub fn square<V>(x: &Variable<V>) -> AdResult<Variable<V>>;

    pub fn grad_tangent<V>(
        output: &Variable<V>,
        inputs: &[&Variable<V>],
        options: BackwardOptions<V>,
    ) -> AdResult<Vec<V::Tangent>>
    where
        V: Differentiable,
        V::Tangent: Clone;

    pub fn grad_variable<V>(
        output: &Variable<V>,
        inputs: &[&Variable<V>],
        options: BackwardOptions<V>,
    ) -> AdResult<Vec<Variable<V>>>
    where
        V: Differentiable<Tangent = V> + Clone + Add<Output = V> + Mul<Output = V> + 'static;
}
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

## Seed Semantics

Reverse-mode implicit seed creation is based on `Differentiable::num_elements()`
rather than tensor rank.

- implicit seed is allowed when `num_elements() == 1`
- otherwise `seed_grad` is required

This applies to:

- `Tape::pullback`
- `Tape::hvp`
- `Variable::backward`
- `Variable::backward_hvp`
- `autograd::grad_tangent`
- `autograd::grad_variable`

That means:

- a rank-0 tensor usually has `num_elements() == 1`
- a shape-`[1]` tensor also has `num_elements() == 1`
- for tensor ops, only rank-0 is scalar
- for seed omission, any `V` with `num_elements() == 1` qualifies

## `retain_graph` / `create_graph` Contract

The effective retain policy follows PyTorch:

```text
effective_retain_graph = options.retain_graph.unwrap_or(options.create_graph)
```

### `grad_tangent`

- returns detached `V::Tangent`
- rejects `create_graph = true` with `ModeNotSupported`
- requires `seed_grad` when `output.num_elements() != 1`

### `grad_variable`

- returns `Variable<V>` results
- supports graph-connected outputs only when `V::Tangent = V`
- requires `seed_grad` when `output.num_elements() != 1`
- may reject `create_graph = true` on operations that do not yet provide
  graph-aware pullback wiring

### `backward_hvp`

- updates both `.grad()` and `.hvp()`
- requires tangent-seeded leaves
- currently rejects `create_graph = true`

There is no fallback heterogeneous graph path for `V::Tangent != V`. If
higher-order graph-connected gradients are needed, the operation/type is simply
unsupported in the current phase unless it fits the `V::Tangent = V` path.

## Context and Mutation Semantics

- `AutogradContext<V>` is shared through `Arc<Mutex<_>>`
- backward execution is single-threaded per context in this phase
- `Variable<V>::backward` and `backward_hvp` accumulate into stored buffers
- `autograd::grad_*` query APIs do not mutate stored `.grad()` / `.hvp()`
  buffers
- `zero_grad()` is explicit per leaf; there is no context-wide reset helper

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

let tape = Tape::<MyScalar>::new();
let x = tape.leaf(MyScalar(2.0));
let grads = tape.pullback(&x).unwrap();
assert_eq!(grads.get(x.node_id().unwrap()).unwrap().0, 1.0);
```

## Usage Examples

### Reverse mode on tensors

```rust,ignore
use chainrules::Tape;
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
use chainrules::DualTensor;
use tenferro_algebra::Standard;
use tenferro_einsum::dual_einsum;
use tenferro_prims::{CpuBackend, CpuContext};

let mut ctx = CpuContext::new(1);
let x = /* Tensor<f64> */;
let dx = /* Tensor<f64> tangent */;
let x_dual = DualTensor::with_tangent(x, dx).unwrap();
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
- detached tangent query behavior for `grad_tangent`
- graph-connected higher-order behavior for supported `grad_variable` cases

## Current Runtime Status

The current eager AD surface uses runtime dispatch and capability-driven
backend contracts in production code.

- `extension/tenferro-dyadtensor` no longer routes production paths through
  `with_cpu_runtime(...)`
- linalg and einsum AD entrypoints now dispatch through the relevant family
  traits and runtime slots
- builder `.run()` now relies on an explicit default-runtime holder, while
  reverse-mode bookkeeping uses one tape-local rule store per tape instead of
  a generic global context map
- many public examples still instantiate `CpuContext` directly because CPU is
  the most complete backend today, not because the API contract is CPU-only

The remaining debt is now mostly breadth:

- unsupported backend capabilities still surface as truthful capability
  failures
- AD formula coverage is broader for some families than others
- GPU custom pointwise/reduction kernels are still incomplete
