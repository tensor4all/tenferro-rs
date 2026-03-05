# Automatic Differentiation

This document defines the AD architecture for tenferro-rs.

For math derivations, see [AD Formula Notes](../AD/index.md).

## Design Goals

1. Keep `tenferro-einsum` explicit AD interfaces as-is.
2. Provide torch-like UX in higher layers (`dyadtensor` and similar wrappers).
3. Keep `chainrules-core` minimal and stable for external custom types.
4. Support both homogeneous and heterogeneous AD graphs:
   - `Tape<V>` for monomorphic, type-safe fast path
   - `DynTape` for mixed custom-value graphs
5. Expose explicit `backward` options:
   - `retain_graph`
   - `create_graph`
6. Match LibTorch C++ autograd behavior for `retain_graph` / `create_graph`
   semantics (behavioral compatibility, not API-level compatibility).

## Fixed Decisions

- `chainrules-core` remains math contracts only:
  `Differentiable`, `ReverseRule<V>`, `ForwardRule<V>`, errors.
- Torch-like API lives in `chainrules` (not in `chainrules-core`).
- `Tape<V>` and `DynTape` co-exist and are both public.
- `tenferro-einsum` keeps explicit interfaces (`tracked_einsum`, `dual_einsum`,
  `einsum_rrule`, `einsum_frule`, `einsum_hvp`).
- Autograd context is shared by `Arc<Mutex<_>>`.
- This phase assumes single-threaded backward execution per context; parallel
  backward semantics are out of scope until lock granularity is redesigned.
- Backward executors must not hold the context mutex while invoking rule
  callbacks; use snapshot/queue/apply style updates to avoid non-reentrant
  mutex deadlock under `create_graph=true`.
- Lock protocol (normative): `lock -> snapshot -> unlock -> run callbacks ->
  lock -> apply queued graph updates -> unlock`. Pullback closures must not
  capture/re-lock the same context mutex.
- Graph updates collected during callback execution are staged in
  `Vec<QueuedNode>` (internal), and node IDs are assigned only in the apply
  phase under lock.
- Snapshot node IDs are immutable during callback execution; queued nodes can
  reference snapshot IDs but do not receive stable IDs until apply.
- `Variable<V>` API (`backward`, `backward_hvp`) accumulates into stored
  `.grad()` / `.hvp()` buffers; `zero_grad` clears them.
- This phase does not provide a context-wide reset API such as
  `AutogradContext::zero_grad_all()`; multi-leaf reset is explicit per leaf.
- `DynTape` API is functional for gradients/HVP (`grad_dyn_*`, `DynTape::hvp`);
  it returns result containers and does not persist per-variable `.grad()`/`.hvp()`.
- `backward_hvp` updates both `.grad()` and `.hvp()` in a single pass.
- Gradient query API is split:
  - `autograd::grad_tangent`: returns detached `V::Tangent` values
  - `autograd::grad_variable`: returns `Variable<V>` and supports higher-order
    differentiation when `create_graph = true`
- All `autograd::grad_*` APIs are query-style and do not mutate stored
  `.grad()` / `.hvp()` buffers.
- Heterogeneous graphs support higher-order differentiation via
  `autograd::grad_dyn_variable` with `DynBackwardOptions { create_graph: true, .. }`.
- `DynTape` uses a functional gradient API (`autograd::grad_dyn_*` results);
  `DynVariable` does not expose `.grad()`, `.hvp()`, or `.zero_grad()`.
- `DynVariable` exposes read-only `requires_grad()`; all values created through
  `DynTape::leaf` / `leaf_with_tangent` and op outputs are tracked
  (`requires_grad=true`) in this phase.
- `DynTape` does not provide `constant(...)` in this phase; constants are passed
  as plain values to operations (for example
  `axpy(&tape, alpha_plain, &x_dyn, &y_dyn)` where `alpha_plain` is untracked).
- Heterogeneous op signature convention in this phase:
  tracked operands use `&DynVariable`; untracked constants use plain `T`/`&T`.
  No implicit promotion from plain constants to tracked variables.
- `DynTape::leaf_with_tangent` is HVP-direction seeding only; heterogeneous
  standalone JVP/tangent propagation remains out of scope in this phase.
- `DynTape::leaf_with_tangent` always succeeds as metadata attachment; JVP mode
  errors are raised only when a heterogeneous JVP propagation API is requested.
- `DynVariable::detach()` is out of scope in this phase.
- API placement split is intentional:
  - `DynTape::hvp` stays as a tape method because it consumes tape-local tangent
    seeds and returns both gradient and HVP together.
  - `autograd::grad_dyn_*` stays as free functions for first-/higher-order
    gradient queries over selected `outputs`/`inputs` subsets.
- Standalone forward-mode JVP API for heterogeneous graphs is out of scope in
  this phase; unsupported requests return `ModeNotSupported`.
- `Variable<V>` and `DynVariable` are cheap-clone handles (Arc-backed metadata);
  cloning preserves graph/context linkage.
- `detach()` preserves the primal value and clears reverse/forward AD linkage.
- There is no `autograd::grad_*_hvp` helper in this phase; HVP is exposed via
  `backward_hvp` / `DynTape::hvp`.
- Unsupported modes return typed errors (`ModeNotSupported`).

## Public API Sketch

```rust
pub struct BackwardOptions<V: Differentiable> {
    // None means: infer effective retain from create_graph.
    pub retain_graph: Option<bool>,
    pub create_graph: bool,
    // Seed for single-output grad/backward paths in this phase.
    // For `grad_variable`, this is valid under `V::Tangent = V`.
    pub seed_grad: Option<V::Tangent>,
}

impl<V: Differentiable> Default for BackwardOptions<V> {
    fn default() -> Self {
        Self {
            retain_graph: None,
            create_graph: false,
            seed_grad: None,
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct NodeId(usize);

pub struct Variable<V: Differentiable> {
    // value, node id, and shared context (internal)
}

impl<V: Differentiable> Variable<V> {
    pub fn new(value: V) -> Self;
    pub fn new_in(value: V, ctx: Arc<Mutex<AutogradContext<V>>>) -> Self;
    pub fn value(&self) -> &V;
    pub fn ones_like(&self) -> V::Tangent;
    pub fn node_id(&self) -> Option<NodeId>;
    pub fn context_id(&self) -> Option<u64>;
    pub fn context(&self) -> Option<Arc<Mutex<AutogradContext<V>>>>;
    pub fn is_leaf(&self) -> bool;
    pub fn requires_grad(&self) -> bool;
    pub fn requires_grad_(self, enabled: bool) -> AdResult<Self>;
    pub fn with_tangent_(self, tangent: V::Tangent) -> AdResult<Self>;
    pub fn tangent(&self) -> Option<&V::Tangent>;
    pub fn detach(&self) -> Self;

    pub fn backward(&self, options: BackwardOptions<V>) -> AdResult<()>;
    pub fn backward_hvp(&self, options: BackwardOptions<V>) -> AdResult<()>;
    pub fn grad(&self) -> Option<V::Tangent>;
    pub fn hvp(&self) -> Option<V::Tangent>;
    pub fn zero_grad(&self) -> AdResult<()>;
}

impl<V: Differentiable> Clone for Variable<V> {}

pub struct AutogradContext<V: Differentiable> {
    // graph storage and gradient accumulators (internal)
}

impl<V: Differentiable> AutogradContext<V> {
    pub fn new() -> Arc<Mutex<Self>>;
    pub fn id(&self) -> u64;
}

pub struct DynBackwardOptions {
    pub retain_graph: Option<bool>,
    pub create_graph: bool,
    pub seed_grads: Option<Vec<DynTangent>>,
}

impl Default for DynBackwardOptions {
    fn default() -> Self {
        Self {
            retain_graph: None,
            create_graph: false,
            seed_grads: None,
        }
    }
}

pub struct DynHvpOptions {
    pub retain_graph: Option<bool>,
    pub create_graph: bool,
    pub seed_grad: Option<DynTangent>,
}

impl Default for DynHvpOptions {
    fn default() -> Self {
        Self {
            retain_graph: None,
            create_graph: false,
            seed_grad: None,
        }
    }
}

pub struct DynVariable {
    // erased value, node id, and shared context (internal)
}

impl DynVariable {
    pub fn value_as<T: 'static>(&self) -> AdResult<&T>;
    pub fn node_id(&self) -> DynNodeId;
    pub fn context_id(&self) -> u64;
    pub fn requires_grad(&self) -> bool;
    pub fn is_scalar(&self) -> bool;
}

impl Clone for DynVariable {}

// DynVariable is always produced by DynTape graph construction (`DynTape::leaf`
// or operation outputs), so node_id() and context_id() are non-optional.

pub struct DynTape {
    // heterogeneous graph storage (internal)
}

impl DynTape {
    pub fn new() -> Self;
    pub fn id(&self) -> u64;
    pub fn leaf<T: 'static + Send + Sync>(&self, value: T) -> DynVariable;
    pub fn leaf_with_tangent<T: 'static + Send + Sync, G: 'static + Send + Sync>(
        &self,
        value: T,
        tangent: G,
    ) -> AdResult<DynVariable>;
    pub fn hvp(
        &self,
        // `loss` is the differentiation target for HVP (typically scalar).
        loss: &DynVariable,
        options: DynHvpOptions,
    ) -> AdResult<DynHvpResult>;
}

// First-order heterogeneous backward without HVP is provided by
// autograd::grad_dyn_tangent / autograd::grad_dyn_variable.

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct DynNodeId(usize);

pub struct DynTangent(Arc<dyn Any + Send + Sync>);

impl DynTangent {
    pub fn new<T: 'static + Send + Sync>(value: T) -> Self;
    pub fn downcast_ref<T: 'static>(&self) -> AdResult<&T>;
}

pub struct DynGradients {
    // typed map keyed by DynNodeId (internal)
}

impl DynGradients {
    pub fn get<T: 'static>(&self, node: DynNodeId) -> Option<&T>;
}

pub struct DynHvpResult {
    pub gradients: DynGradients,
    pub hvp: DynGradients,
}

pub mod autograd {
    // Monomorphic APIs are single-output in this phase.
    pub fn grad_tangent<V: Differentiable>(
        output: &Variable<V>,
        inputs: &[&Variable<V>],
        options: BackwardOptions<V>,
    ) -> AdResult<Vec<V::Tangent>>;

    pub fn grad_variable<V: Differentiable<Tangent = V>>(
        output: &Variable<V>,
        inputs: &[&Variable<V>],
        options: BackwardOptions<V>,
    ) -> AdResult<Vec<Variable<V>>>;

    // Heterogeneous APIs support multi-output queries.
    pub fn grad_dyn_tangent(
        outputs: &[&DynVariable],
        inputs: &[&DynVariable],
        options: DynBackwardOptions,
    ) -> AdResult<Vec<DynTangent>>;

    pub fn grad_dyn_variable(
        outputs: &[&DynVariable],
        inputs: &[&DynVariable],
        options: DynBackwardOptions,
    ) -> AdResult<Vec<DynVariable>>;
}
```

`DynTangent` denotes an erased tangent payload for heterogeneous graphs.
Concrete implementation can use an owned type-erased container (for example,
`Arc<dyn Any + Send + Sync>`), as long as typed extraction is deterministic and
error paths remain typed (`InvalidArgument` / `ModeNotSupported`).

## Review Guide

Use this order when reviewing:

1. Read **Terminology Bridge** if you are new to PyTorch-style AD terms.
2. Read **Usage Examples 1-6** to understand same-type flows (`Variable<V>`),
   seed gradients, and minimal end-to-end usage.
3. Read **Usage Examples 7-8** for heterogeneous custom types and
   heterogeneous higher-order differentiation.
4. Read **Error Contract** and **Context Merge Rule** as behavioral constraints.

## Terminology Bridge

| Concept | This design | PyTorch term | Meaning |
|---------|-------------|--------------|---------|
| Differentiable value | `Variable<V>` | `Tensor` (autograd-enabled) | Value carrying graph linkage and AD metadata |
| Leaf variable | Created by `new/new_in` and not produced by an AD op (`is_leaf()==true`) | Leaf tensor | Graph input endpoint (may have `requires_grad` true or false) |
| Backward seed | `BackwardOptions { seed_grad: Some(...) }` | `gradient` argument to `backward` | Cotangent seed for non-scalar outputs |
| Reuse graph | `retain_graph: Some(true)` | `retain_graph=True` | Keep graph after backward |
| Build grad graph | `create_graph = true` | `create_graph=True` | Make first-order gradients differentiable |
| Graph break | `detach()` | `detach()` | Stop tracking reverse and forward paths |
| JVP direction | `with_tangent_(v)` | forward AD tangent | Seed forward-mode direction |
| HVP | `backward_hvp(...)` | composed `jvp(grad(f))` pattern | Hessian-vector product |

## Context Model

`Variable` context ownership is explicit:

1. `Variable::new(value)` creates a value with no attached autograd context and
   `requires_grad=false`.
2. `Variable::new_in(value, ctx)` creates a value attached to `ctx`.
3. `requires_grad_(true)` keeps existing context if present; if absent, it
   attaches the value to a new context. The auto-created context is an isolated
   instance (not implicitly shared with other variables). For multi-leaf
   graphs, using only `requires_grad_(true)` on independently created values is
   not allowed in this design; use explicit shared context + `new_in`.
4. `requires_grad_(false)` clears only the `requires_grad` flag and keeps the
   current context linkage; use `detach()` when graph linkage must be severed.
5. Binary/multi-input AD operations require all attached contexts to be the
   same `context_id`; otherwise they return `InvalidArgument`.
6. Multi-leaf graphs should share one explicit context:
   create one `AutogradContext::new()` and build leaves with
   `Variable::new_in(..., Arc::clone(&ctx))`.
7. A variable's attached context can be reused via `Variable::context()` when
   constructing additional leaves with `new_in`.
8. `DynTape::leaf` / `leaf_with_tangent` values are always tracked
   (`requires_grad=true`).

## Usage Examples

In examples below, `AdError` is a short alias for `chainrules_core::AutodiffError`.

### 1. Same Type for Primal / Forward / Backward / HVP

```rust
type ADMyScalar = Variable<MyScalar>;

// primal only
let x0 = ADMyScalar::new(MyScalar(3.0));
let y0 = square(&x0)?;
assert_eq!(y0.value().0, 9.0);

// forward (JVP)
let x1 = ADMyScalar::new(MyScalar(3.0)).with_tangent_(MyScalar(1.0))?;
let y1 = square(&x1)?;
assert_eq!(y1.value().0, 9.0);
// d(x^2)/dx at x=3 = 6
assert_eq!(y1.tangent().unwrap().0, 6.0);

// backward
let x2 = ADMyScalar::new(MyScalar(3.0)).requires_grad_(true)?;
let y2 = square(&x2)?;
y2.backward(BackwardOptions::default())?;
// d(x^2)/dx at x=3 = 6
assert_eq!(x2.grad().unwrap().0, 6.0);

// hvp
let x3 = ADMyScalar::new(MyScalar(3.0))
    .requires_grad_(true)?
    .with_tangent_(MyScalar(5.0))?;
let y3 = square(&x3)?;
y3.backward_hvp(BackwardOptions::default())?;
// HVP = f''(x) * v = 2 * 5 = 10
assert_eq!(x3.hvp().unwrap().0, 10.0);
```

### 2. `retain_graph = true`

```rust
let x = ADMyScalar::new(MyScalar(2.0)).requires_grad_(true)?;
let loss = square(&x)?; // f(x) = x^2

loss.backward(BackwardOptions {
    retain_graph: Some(true),
    ..Default::default()
})?;

// Same graph reused without re-running forward construction.
loss.backward(BackwardOptions {
    retain_graph: Some(true),
    ..Default::default()
})?;

// grad is accumulated by default
let gx = x.grad().unwrap();
assert_eq!(gx.0, 8.0); // 2*2 + 2*2
```

### 3. `detach` (PyTorch-compatible graph break)

```rust
let x = ADMyScalar::new(MyScalar(3.0))
    .requires_grad_(true)?
    .with_tangent_(MyScalar(1.0))?;

let y = square(&x)?;
let y_detached = y.detach();

assert!(!y_detached.requires_grad());
assert!(y_detached.tangent().is_none()); // forward tangent is also cleared
assert_eq!(y_detached.value().0, y.value().0); // primal value is preserved
```

### 4. `create_graph = true` (higher-order differentiation)

```rust
let x = ADMyScalar::new(MyScalar(3.0)).requires_grad_(true)?;
let loss = cube(&x)?; // f(x) = x^3

// gx = df/dx = 3x^2 and remains graph-connected
let gx = autograd::grad_variable(
    &loss,
    &[&x],
    BackwardOptions {
        create_graph: true,
        ..Default::default()
    },
)?[0].clone();

// grad(gx, x) = d2f/dx2 at x=3
let gxx = autograd::grad_variable(&gx, &[&x], BackwardOptions::default())?[0].clone();
assert_eq!(gxx.value().0, 18.0);
```

If `gx` is non-scalar, supply a seed gradient via
`BackwardOptions { seed_grad: Some(...) }`
in the second `autograd::grad_variable` call.

```rust
let gx = vector_grad(&x)?; // non-scalar gx
let seed = gx.ones_like();
let gxx = autograd::grad_variable(
    &gx,
    &[&x],
    BackwardOptions {
        seed_grad: Some(seed),
        ..Default::default()
    },
)?[0].clone();
# let _ = gxx;
```

### 5. Minimal End-to-End Flow (non-PyTorch readers)

```rust
type ADMyScalar = Variable<MyScalar>;

// Step 1: create value
let x = ADMyScalar::new(MyScalar(3.0)).requires_grad_(true)?;

// Step 2: run forward computation
let loss = cube(&x)?; // scalar output

// Step 3: run backward
loss.backward(BackwardOptions::default())?;

// Step 4: read first-order gradient
let gx = x.grad().unwrap();
assert_eq!(gx.0, 27.0); // d(x^3)/dx at x=3

// Step 5: clear accumulation before next step
x.zero_grad()?;
```

### 5b. Shared Context for Multiple Leaves

```rust
let ctx = AutogradContext::<MyScalar>::new();
let a = ADMyScalar::new_in(MyScalar(1.0), Arc::clone(&ctx)).requires_grad_(true)?;
let b = ADMyScalar::new_in(MyScalar(2.0), Arc::clone(&ctx)).requires_grad_(true)?;
let y = add(&a, &b)?; // succeeds because context_id matches
a.zero_grad()?;
b.zero_grad()?;
# let _ = y;
```

### 6. Non-Scalar Output with Seed Gradient

```rust
let y = vector_output(&x)?; // y is non-scalar
let seed = y.ones_like();

y.backward(BackwardOptions {
    seed_grad: Some(seed),
    ..Default::default()
})?;
```

### 7. Heterogeneous Custom Types with `DynTape`

```rust
let tape = DynTape::new();
let s = tape.leaf_with_tangent(MyScalar(2.0), MyScalar(1.0))?; // HVP direction on scalar leg
let v = tape.leaf(MyVec2 { x: 3.0, y: -1.0 });

let y = mul_scalar_vec(&tape, &s, &v)?;  // MyScalar x MyVec2 -> MyVec2
let loss = squared_norm(&tape, &y)?;     // MyVec2 -> MyScalar

let result = tape.hvp(&loss, DynBackwardOptions::default())?;
let grad_s = result.gradients.get::<MyScalar>(s.node_id()).unwrap();
let hvp_s = result.hvp.get::<MyScalar>(s.node_id()).unwrap();
let _ = (grad_s, hvp_s);
```

### 8. Heterogeneous Second Derivative Without HVP (`create_graph`)

```rust
let tape = DynTape::new();
let s = tape.leaf(MyScalar(2.0));
let v = tape.leaf(MyVec2 { x: 3.0, y: -1.0 });

let y = mul_scalar_vec(&tape, &s, &v)?; // MyScalar x MyVec2 -> MyVec2
let loss = squared_norm(&tape, &y)?;    // MyVec2 -> MyScalar

let gs = autograd::grad_dyn_variable(
    &[&loss],
    &[&s],
    DynBackwardOptions {
        create_graph: true,
        ..Default::default()
    },
)?[0].clone();

let ggs = autograd::grad_dyn_variable(
    &[&gs],
    &[&s],
    DynBackwardOptions::default(),
)?[0].clone();

let vv = v.value_as::<MyVec2>()?;
assert_eq!(ggs.value_as::<MyScalar>()?.0, 2.0 * (vv.x * vv.x + vv.y * vv.y));
```

## Common Failure Cases

### Non-scalar backward without seed gradient

```rust
let y = vector_output(&x)?; // non-scalar
let err = y.backward(BackwardOptions::default()).unwrap_err();
assert!(matches!(err, AdError::InvalidArgument(_)));
```

### Mixed contexts in one operation

```rust
type ADMyScalar = Variable<MyScalar>;
let ctx_a = AutogradContext::<MyScalar>::new();
let ctx_b = AutogradContext::<MyScalar>::new();
let a = ADMyScalar::new_in(MyScalar(1.0), ctx_a).requires_grad_(true)?;
let b = ADMyScalar::new_in(MyScalar(2.0), ctx_b).requires_grad_(true)?; // different context
let err = add(&a, &b).unwrap_err();
assert!(matches!(err, AdError::InvalidArgument(_)));
```

### Independent `requires_grad_(true)` leaves in one operation

```rust
let a = ADMyScalar::new(MyScalar(1.0)).requires_grad_(true)?;
let b = ADMyScalar::new(MyScalar(2.0)).requires_grad_(true)?;
let err = add(&a, &b).unwrap_err();
assert!(matches!(err, AdError::InvalidArgument(_)));
// This pattern is not allowed for multi-leaf graphs in this design.
// Use `new_in(..., same_ctx)` instead.
```

### HVP on rule without tangent support

```rust
let x = ADMyScalar::new(MyScalar(3.0))
    .requires_grad_(true)?
    .with_tangent_(MyScalar(1.0))?;
let y = custom_op_without_hvp_rule(&x)?;
let err = y.backward_hvp(BackwardOptions::default()).unwrap_err();
assert!(matches!(err, AdError::ModeNotSupported { .. }));
```

## Context Merge Rule

For multi-input operations, context selection is deterministic.
Only attached contexts (`ctx != None`) participate in context choice; `requires_grad`
on context-less inputs does not affect context selection.

Evaluation order is normative:

1. If all inputs have `requires_grad=false`, output `ctx=None`.
2. Otherwise collect attached contexts from inputs (`ctx != None`).
3. No attached contexts -> output `ctx=None`.
4. Two or more distinct attached contexts -> `InvalidArgument`.
5. Exactly one attached context:
   - if at least one input on that context has `requires_grad=true`, adopt it
   - otherwise output `ctx=None`.

For heterogeneous (`DynTape`) operations:

1. All participating `DynVariable` operands must share the same
   `context_id()`/tape identity.
2. Mixing operands from different `DynTape` instances returns `InvalidArgument`.

## `create_graph` Contract

Higher-order differentiation behavior is fixed by the following rules:

1. Effective retain policy follows PyTorch semantics:
   `effective_retain_graph = options.retain_graph.unwrap_or(options.create_graph)`.
2. Monomorphic `autograd::grad_tangent` and `autograd::grad_variable` take a
   single output argument in this phase (`output: &Variable<V>`).
3. `create_graph = false` returns detached gradient results.
4. `create_graph = true` + `autograd::grad_variable` returns
   graph-connected gradients in the same `AutogradContext` as the input graph,
   so second and higher derivatives are possible. This surface requires
   `V::Tangent = V`; otherwise use `autograd::grad_tangent`. For
   `V::Tangent != V` with higher-order needs, use `DynTape` +
   `autograd::grad_dyn_variable`.
   Routing summary:
   - `V::Tangent = V` + higher-order required -> `grad_variable`
   - `V::Tangent != V` + first-order only -> `grad_tangent`
   - `V::Tangent != V` + higher-order required -> `DynTape` + `grad_dyn_variable`
5. `create_graph = true` + `autograd::grad_dyn_variable` returns
   graph-connected heterogeneous gradients in the same `DynTape` graph, so
   second and higher derivatives are possible for `DynTape`.
6. `autograd::grad_tangent` always returns detached `V::Tangent` values and
   must reject `create_graph = true` with
   `ModeNotSupported { mode: "create_graph_tangent", ... }`; `seed_grad` and
   `retain_graph` follow the same scalar/retain rules as `backward`.
7. `autograd::grad_dyn_tangent` always returns detached `DynTangent` values and
   must reject `create_graph = true` with
   `ModeNotSupported { mode: "create_graph_tangent_dyntape", ... }`.
8. Non-scalar output requires explicit seed gradient (`options.seed_grad`) for
   `backward`, `backward_hvp`, `grad_tangent`, and `grad_variable`.
9. Non-scalar heterogeneous outputs require explicit seed gradients
   (`DynBackwardOptions.seed_grads`) for `grad_dyn_tangent` and
   `grad_dyn_variable`.
10. Non-scalar `DynTape::hvp` `loss` input requires `DynHvpOptions.seed_grad`
    (seed in the cotangent space of `loss`).
11. `DynBackwardOptions.seed_grads` is positionally aligned with `outputs`:
   `seed_grads[i]` is the seed for `outputs[i]`, and lengths must match exactly.
12. When `DynBackwardOptions.seed_grads` is `None`, all outputs must be scalar;
    otherwise return `InvalidArgument` (`DynVariable::is_scalar()` defines this).
13. For `DynTape::hvp`: scalar `loss` allows `seed_grad=None`; non-scalar
    `loss` requires `seed_grad=Some(...)`.
14. `backward` and `backward_hvp` accumulate by default; for `backward_hvp`,
   both `.grad()` and `.hvp()` are additive accumulators. `retain_graph`
   controls graph lifetime, not accumulation semantics. Mixing
   `backward` then `backward_hvp` also additively accumulates `.grad()`
   contributions unless `zero_grad()` is called between them.
   `backward_hvp` must contribute the same first-order reverse gradient as
   `backward`; rules that cannot satisfy this must return
   `ModeNotSupported { mode: "hvp", ... }`.
15. All `autograd::grad_*` query APIs are side-effect free with respect to
    `.grad()` / `.hvp()` accumulators.
16. `zero_grad()` clears only the called variable's `.grad()` and `.hvp()`
    accumulators (no cross-variable side effects). Stored `.tangent()` is not
    cleared by `zero_grad()`.
17. `zero_grad()` is valid on leaf variables only; calling it on non-leaf
    variables returns `InvalidArgument`.
18. `backward_hvp` with `create_graph = true` is out of scope in this phase and
    returns `ModeNotSupported { mode: "create_graph_hvp", ... }`.
19. `backward_hvp` requires at least one tangent-seeded leaf (e.g., via
    `with_tangent_` / `leaf_with_tangent`); otherwise return `InvalidArgument`.
20. `backward_hvp` follows the same retain/free graph-lifetime policy as
    `backward` (rule 1).
21. `DynTape::hvp` with `create_graph = true` is out of scope in this phase and
    returns `ModeNotSupported { mode: "create_graph_hvp_dyntape", ... }`.

## Error Contract

- Unsupported JVP path -> `ModeNotSupported { mode: "frule", ... }`
- Unsupported HVP path -> `ModeNotSupported { mode: "hvp", ... }`
- Non-scalar backward without `seed_grad` -> `InvalidArgument`
- `grad_tangent` with `create_graph = true` ->
  `ModeNotSupported { mode: "create_graph_tangent", ... }`
- `grad_dyn_tangent` with `create_graph = true` ->
  `ModeNotSupported { mode: "create_graph_tangent_dyntape", ... }`
- `backward_hvp` with `create_graph = true` -> `ModeNotSupported { mode: "create_graph_hvp", ... }`
- `backward_hvp` with no tangent-seeded leaves -> `InvalidArgument`
- `backward_hvp` on rules that cannot provide backward-equivalent `.grad()`
  contribution -> `ModeNotSupported { mode: "hvp", ... }`
- `DynTape::hvp` with `create_graph = true` -> `ModeNotSupported { mode: "create_graph_hvp_dyntape", ... }`
- `DynTape::hvp` with non-scalar `loss` and `seed_grad = None` -> `InvalidArgument`
- `DynTape::hvp` with `seed_grad` type/shape mismatch -> `InvalidArgument`
- Re-running `backward_hvp` on a freed graph -> `GraphFreed`
- Re-running `DynTape::hvp` on a freed graph -> `GraphFreed`
- Re-running `grad_tangent` on a freed graph -> `GraphFreed`
- Re-running `grad_dyn_tangent` on a freed graph -> `GraphFreed`
- `grad_dyn_variable` on an operation that does not provide graph-aware
  pullback wiring -> `ModeNotSupported { mode: "create_graph_dyntape", ... }`
- `DynBackwardOptions.seed_grads` length/type/shape mismatch -> `InvalidArgument`
- `DynBackwardOptions.seed_grads = None` with non-scalar outputs -> `InvalidArgument`
- Standalone heterogeneous JVP request -> `ModeNotSupported { mode: "frule_dyntape", ... }`
- JVP/tangent propagation request from `DynTape::leaf_with_tangent` path ->
  `ModeNotSupported { mode: "frule_dyntape", ... }`
- `DynTangent::downcast_ref` / `DynVariable::value_as` with wrong target type -> `InvalidArgument`
- Re-running `backward` on a freed graph -> `GraphFreed`
- Re-running `grad_variable` / `grad_dyn_variable` on a freed graph -> `GraphFreed`
- `zero_grad()` called on a non-leaf variable -> `InvalidArgument`
- Context mismatch across operands -> `InvalidArgument`
  (diagnostic should suggest `new_in(..., same_ctx)` for multi-leaf graphs)

## Compatibility With Current APIs

- `tenferro-einsum` explicit AD functions remain stable and unchanged.
- `Tape<V>` remains public and first-class.
- `DynTape` is public for heterogeneous user-defined type graphs.
- Torch-like wrappers are additive and do not remove existing explicit APIs.

Intentional API divergence note:
- `retain_graph` is modeled as `Option<bool>` (not plain `bool`) so `None`
  can encode "infer from `create_graph`" in a Rust-explicit way while
  preserving PyTorch-compatible runtime behavior.
- Monomorphic and heterogeneous seed APIs differ by design in this phase:
  `BackwardOptions.seed_grad` (single-output monomorphic API) vs
  `DynBackwardOptions.seed_grads` (multi-output heterogeneous API).

## LibTorch C++ Compatibility Scope

This design targets **behavioral compatibility** with LibTorch C++ autograd for:

- `backward(..., retain_graph, create_graph)`
- `grad(..., retain_graph, create_graph)` semantics
- Non-scalar output seed-gradient requirements
- Gradient accumulation default behavior

API names and return types are intentionally Rust-native (`Variable<V>`,
`V::Tangent`, explicit split between `grad_tangent` and `grad_variable`) and
are **not** a 1:1 source-level API match with LibTorch.

## Test Pattern Contract (MUST)

The following test patterns are **required** for the next AD system.
Merging implementation work without these cases is not allowed.

| ID | Category | Surface | Scenario | Expected |
|----|----------|---------|----------|----------|
| AD-NEXT-001 | Monomorphic | `Variable<V>::backward` | Scalar loss reverse-mode on a single-context monomorphic graph | Leaf gradients are correct and accumulated once per call |
| AD-NEXT-002 | Monomorphic | `Variable<V>::with_tangent_` + frule-backed op | Forward-mode JVP on same operation family | Primal and tangent both match analytic values |
| AD-NEXT-003 | Monomorphic | `Variable<V>::backward_hvp` | HVP with tangent-enabled rule | `.grad()` and `.hvp()` both match analytic values |
| AD-NEXT-004 | Monomorphic | `autograd::grad_variable` | Second derivative **without HVP**: run `grad_variable(create_graph=true)` then `grad_variable` again | Second-order result matches analytic value |
| AD-NEXT-005 | Monomorphic | `autograd::grad_tangent` | Call with `create_graph=true` | Returns `ModeNotSupported { mode: "create_graph_tangent", .. }` |
| AD-NEXT-006 | Detach | `Variable<V>::detach` | Detach value produced from tracked computation | Detached value has no reverse linkage and no forward tangent |
| AD-NEXT-007 | Detach | mixed graph | Upstream tracked value + detached branch in same formula | No gradient/hvp flows through detached branch |
| AD-NEXT-008 | Seed gradient | `backward`/`grad_*` | Non-scalar output without `seed_grad` | Returns `InvalidArgument` |
| AD-NEXT-009 | Seed gradient | `backward`/`grad_*` | Non-scalar output with valid `seed_grad` | Succeeds and matches analytic seeded result |
| AD-NEXT-010 | Retain policy | `backward`/`grad_*` | `retain_graph=None`, `create_graph=true` | Effective retain is `true`; repeated backward/grad on same graph succeeds |
| AD-NEXT-011 | Reset semantics | `zero_grad` | After `backward` + `backward_hvp`, call `zero_grad` | Both stored `.grad()` and `.hvp()` are cleared |
| AD-NEXT-012 | Heterogeneous | `DynTape` | Mixed custom types (e.g., scalar × vec -> vec -> scalar loss) | Reverse/HVP result retrieval by concrete type succeeds |
| AD-NEXT-013 | Context contract | multi-input op | Mix different AD contexts in one op | Returns `InvalidArgument` |
| AD-NEXT-014 | Custom type | `Tape<V>` + custom rule | User-defined `Differentiable` + `ReverseRule` + `ForwardRule` | Reverse and JVP pass with expected values |
| AD-NEXT-015 | Custom type error path | HVP | Custom reverse rule without tangent support | `backward_hvp` returns `ModeNotSupported { mode: "hvp", .. }` |
| AD-NEXT-016 | Heterogeneous higher-order | `DynTape` | Second derivative **without HVP** on mixed-type graph (`create_graph` path) using rule families with graph-aware pullback wiring | Returns correct analytic second-order value |
| AD-NEXT-017 | Retain override | `backward`/`grad_*` | `retain_graph: Some(false), create_graph: true` | Computation graph is freed after call; second call on same output fails with `GraphFreed` |
| AD-NEXT-018 | Retain default free | `backward`/`grad_*` | `retain_graph: None, create_graph: false` (default) | Computation graph is freed after call; second call on same output fails with `GraphFreed` |
| AD-NEXT-019 | Heterogeneous create-graph error path | `DynTape` | `grad_dyn_variable(create_graph=true)` on a rule family without graph-aware pullback wiring | Returns `ModeNotSupported { mode: "create_graph_dyntape", .. }` |
| AD-NEXT-020 | Heterogeneous tangent contract | `autograd::grad_dyn_tangent` | Call with `create_graph=true` | Returns `ModeNotSupported { mode: "create_graph_tangent_dyntape", .. }` |
| AD-NEXT-021 | Heterogeneous seed mismatch | `autograd::grad_dyn_*` | `seed_grads` length/type/shape mismatch vs outputs | Returns `InvalidArgument` |
| AD-NEXT-022 | Heterogeneous retain override | `autograd::grad_dyn_variable` | `retain_graph: Some(false), create_graph: true` | Computation graph is freed after call; second call on same output fails with `GraphFreed` |
| AD-NEXT-023 | Heterogeneous retain default free | `autograd::grad_dyn_variable` | `retain_graph: None, create_graph: false` (default) | Computation graph is freed after call; second call on same output fails with `GraphFreed` |
| AD-NEXT-024 | Non-leaf reset error | `Variable<V>::zero_grad` | Call on non-leaf/intermediate variable | Returns `InvalidArgument` |
| AD-NEXT-025 | Accumulation invariant | `backward`/`backward_hvp` | Example: `f(x)=x^2`, `x=2`, `v=1`, run `backward`, then `backward_hvp`, and repeated calls with retained graph | Explicit transition: after `backward`: `grad=4`; after `backward_hvp`: `grad=8`, `hvp=2`; further calls add same increments |
| AD-NEXT-026 | Retain explicit keep | `backward`/`grad_*` | `retain_graph: Some(true), create_graph: false` | Computation graph is kept; second call on same output succeeds |
| AD-NEXT-027 | Heterogeneous retain implicit keep | `autograd::grad_dyn_variable` | `retain_graph: None, create_graph: true` | Effective retain is `true`; second call on same output succeeds |
| AD-NEXT-028 | Context merge success | multi-input op | Exactly one input has non-`None` context and others are `None` | Output uses that context; no `InvalidArgument` |
| AD-NEXT-029 | HVP create-graph unsupported | `Variable<V>::backward_hvp` | `create_graph: true` | Returns `ModeNotSupported { mode: "create_graph_hvp", .. }` |
| AD-NEXT-030 | Heterogeneous seed-none invalid | `autograd::grad_dyn_variable` | `seed_grads: None` with non-scalar outputs | Returns `InvalidArgument` |
| AD-NEXT-031 | Heterogeneous JVP unsupported | `DynTape` / autograd | Standalone heterogeneous JVP request | Returns `ModeNotSupported { mode: "frule_dyntape", .. }` |
| AD-NEXT-032 | Heterogeneous first-order tangent success | `autograd::grad_dyn_tangent` | Basic mixed-type first-order gradient query | Returns correct analytic first-order tangents |
| AD-NEXT-033 | Heterogeneous first-order variable success | `autograd::grad_dyn_variable` | Basic mixed-type first-order gradient query | Returns correct analytic first-order gradient variables |
| AD-NEXT-034 | HVP missing direction seed | `Variable<V>::backward_hvp` / `DynTape::hvp` | HVP invoked without tangent-seeded leaves | Returns `InvalidArgument` |
| AD-NEXT-035 | Heterogeneous downcast error | `DynVariable::value_as` / `DynTangent::downcast_ref` | Request wrong target type | Returns `InvalidArgument` |
| AD-NEXT-036 | HVP retain default free | `Variable<V>::backward_hvp` | `retain_graph: None, create_graph: false` (default) | Computation graph is freed after call; second call on same output fails with `GraphFreed` |
| AD-NEXT-037 | DynTape HVP seed-none invalid | `DynTape::hvp` | Non-scalar `loss` with `seed_grad: None` | Returns `InvalidArgument` |
| AD-NEXT-038 | DynTape HVP seed mismatch | `DynTape::hvp` | `seed_grad` type/shape mismatch vs non-scalar `loss` | Returns `InvalidArgument` |
| AD-NEXT-039 | DynTape HVP create-graph unsupported | `DynTape::hvp` | `create_graph: true` | Returns `ModeNotSupported { mode: "create_graph_hvp_dyntape", .. }` |
| AD-NEXT-040 | Tangent persistence on reset | `Variable<V>::zero_grad` | Leaf has `.tangent()` + populated `.grad()`/`.hvp()`, then `zero_grad` | `.grad()`/`.hvp()` are cleared while `.tangent()` stays unchanged |
| AD-NEXT-041 | Shared-context multi-leaf success | multi-input op | Two tracked leaves created via `new_in(..., same_ctx)` | Operation succeeds (no context mismatch) and gradients are correct |
| AD-NEXT-042 | Query side-effect free | `autograd::grad_*` | Run `grad_tangent`, `grad_variable`, `grad_dyn_tangent`, `grad_dyn_variable` after prior grad accumulation | Returned query values are correct and `.grad()`/`.hvp()` buffers are unchanged |
| AD-NEXT-043 | `requires_grad_(false)` semantics | `Variable<V>` | Disable grad on a context-attached variable | `requires_grad=false` while context linkage remains intact (not detached) |
| AD-NEXT-044 | DynTape HVP freed-graph error | `DynTape::hvp` | Run with effective retain=false, then call again on same loss | Second call returns `GraphFreed` |
| AD-NEXT-045 | `grad_tangent` freed-graph error | `autograd::grad_tangent` | Free graph (`retain_graph=false`) then query again on same output | Returns `GraphFreed` |
| AD-NEXT-046 | `grad_dyn_tangent` freed-graph error | `autograd::grad_dyn_tangent` | Free graph (`retain_graph=false`) then query again on same output set | Returns `GraphFreed` |
| AD-NEXT-047 | Multi-leaf manual reset | `Variable<V>::zero_grad` | Shared-context multi-leaf graph after accumulation | All leaves require explicit `zero_grad()`; each leaf clears only its own buffers |
| AD-NEXT-048 | `grad_variable` type constraint | compile-time contract | `V::Tangent != V` with `grad_variable` call | API is unavailable by trait bound; use `grad_tangent` |
| AD-NEXT-049 | Cross-tape context mismatch | heterogeneous op | Mix `DynVariable` operands from different `DynTape` instances | Returns `InvalidArgument` |

### Coverage Scope

1. `AD-NEXT-001` through `AD-NEXT-011` must run for at least one scalar-like
   differentiable type and one tensor-like differentiable type.
2. `AD-NEXT-012` must include at least two distinct custom value types in a
   single `DynTape` graph.
3. `AD-NEXT-016` applies only to rule families with graph-aware pullback
   wiring and must assert a concrete analytic second-order value.
4. `AD-NEXT-014` and `AD-NEXT-015` must be implemented in crate-level tests as
   minimal user-defined types (not only internal tensor types).
5. Error-path tests must assert typed variants (`InvalidArgument`,
   `ModeNotSupported`) rather than string matching.
6. `AD-NEXT-017` must assert the explicit override semantics:
   `retain_graph: Some(false)` takes precedence over `create_graph: true`.
   `AD-NEXT-022` must assert the same override semantics for `DynTape`.
7. `AD-NEXT-018` must assert default graph-free behavior when
   `create_graph=false`.
8. `AD-NEXT-019` must cover heterogeneous rule families that do not expose
   graph-aware pullback wiring yet.
9. `AD-NEXT-020` through `AD-NEXT-023` must cover heterogeneous parity for
   create-graph rejection, seed mismatch, and retain/free semantics.
10. `AD-NEXT-024` must validate non-leaf `zero_grad` behavior as a typed error.
11. `AD-NEXT-025` must assert accumulation as an invariant, not an artifact of
    a specific retain_graph setting.
12. `AD-NEXT-026` must validate explicit graph retention without create-graph.
13. `AD-NEXT-027` must validate implicit retain behavior parity for `DynTape`.
14. `AD-NEXT-028` must validate Context Merge Rule case 2 (single-context
    adoption success path).
15. `AD-NEXT-029` through `AD-NEXT-031` must validate typed behavior for
    out-of-scope higher-order/JVP paths in heterogeneous mode.
16. `AD-NEXT-032` and `AD-NEXT-033` must validate the primary first-order
    heterogeneous happy paths (not only error/second-order paths).
17. `AD-NEXT-034` and `AD-NEXT-035` must validate missing-tangent and
    downcast-type error paths with typed errors.
18. `AD-NEXT-036` must validate that `backward_hvp` follows the same default
    graph-free policy as `backward`.
19. `AD-NEXT-037` through `AD-NEXT-039` must validate `DynTape::hvp` option
    contracts for non-scalar seed requirements, seed mismatch, and
    `create_graph` rejection.
20. `AD-NEXT-040` must validate that `zero_grad` does not clear stored
    forward tangents.
21. `AD-NEXT-041` must validate Context Merge Rule case 3 (same shared context
    across multiple tracked leaves).
22. `AD-NEXT-042` must validate query-only semantics for tangent APIs
    (`grad_tangent` / `grad_dyn_tangent`) with no buffer mutation.
23. `AD-NEXT-043` must validate that `requires_grad_(false)` is not equivalent
    to `detach()`.
24. `AD-NEXT-044` must validate `GraphFreed` parity for `DynTape::hvp`.
25. `AD-NEXT-045` and `AD-NEXT-046` must validate freed-graph parity for
    tangent query APIs (`grad_tangent` / `grad_dyn_tangent`).
26. `AD-NEXT-047` must validate explicit per-leaf reset requirements in
    shared-context multi-leaf usage.
27. `AD-NEXT-048` must validate the documented `grad_variable` type constraint
    (`V::Tangent = V`) at compile-time, and document `DynTape` as the
    higher-order fallback for `V::Tangent != V`.
28. `AD-NEXT-049` must validate heterogeneous context/tape mismatch handling.
