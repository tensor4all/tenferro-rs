# Einsum + DyadTensor AD Design

This document defines the target AD design for `tenferro-einsum` and
`tenferro-dyadtensor`, built on top of `chainrules-core`/`chainrules`
contracts.

Current implemented behavior is documented in [autodiff.md](./autodiff.md).

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
- `Differentiable` must provide scalar introspection (`is_scalar`) used by
  seed-gradient validation in monomorphic APIs.
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
- Queued nodes may reference other queued nodes via queue-local indices; apply
  resolves queue-local indices to stable `NodeId` values before committing
  edge lists to graph storage.
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
    // Initial `requires_grad` is false.
    pub fn new_in(value: V, ctx: Arc<Mutex<AutogradContext<V>>>) -> Self;
    pub fn value(&self) -> &V;
    pub fn ones_like(&self) -> V::Tangent;
    pub fn is_scalar(&self) -> bool;
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
        loss: &DynVariable,
        options: DynHvpOptions,
    ) -> AdResult<DynHvpResult>;
}

pub struct DynHvpResult {
    pub gradients: DynTangentMap,
    pub hvp: DynTangentMap,
}

pub mod autograd {
    pub fn grad_tangent<V: Differentiable>(
        output: &Variable<V>,
        inputs: &[&Variable<V>],
        options: BackwardOptions<V>,
    ) -> AdResult<Vec<V::Tangent>>;

    pub fn grad_variable<V: Differentiable>(
        output: &Variable<V>,
        inputs: &[&Variable<V>],
        options: BackwardOptions<V>,
    ) -> AdResult<Vec<Variable<V>>>;

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

## Scope and Ownership

This document focuses on operation-level AD behavior for
`tenferro-einsum`/`tenferro-dyadtensor` integration and how it maps onto
`chainrules` runtime contracts.

`chainrules-core` stays operation-agnostic:

- No tensor-specific dependency
- No torch-like API surface
- No backend-specific branching
- Only differentiability contracts and typed errors

The dyadtensor layer may hold heterogeneous values internally (for example
`Arc<dyn Any + Send + Sync>`), as long as typed extraction is deterministic and
error paths remain typed (`InvalidArgument` / `ModeNotSupported`).

Monomorphic scalar/non-scalar validation requires scalar introspection from the
core trait layer. Relevant `Differentiable` methods (excerpt):

```rust
pub trait Differentiable {
    type Tangent;
    fn ones_like(&self) -> Self::Tangent;
    fn is_scalar(&self) -> bool;
}
```

`Variable::is_scalar()` delegates to this trait contract.

## Review Guide

Use this order when reviewing:

1. Read **Terminology Bridge** if you are new to PyTorch-style AD terms.
2. Read **Usage Examples 1-4** to understand same-type flows (`Variable<V>`).
3. Read **Usage Example 7** only if you need heterogeneous custom types.
4. Read **Error Contract** and **Context Merge Rule** as behavioral constraints.

## Terminology Bridge

| Concept | This design | PyTorch term | Meaning |
|---------|-------------|--------------|---------|
| Differentiable value | `Variable<V>` | `Tensor` (autograd-enabled) | Value carrying graph linkage and AD metadata |
| Leaf variable | `requires_grad_(true)` at creation edge | Leaf tensor | Gradient storage endpoint |
| Backward seed | `BackwardOptions { seed_grad: Some(...) }` | `gradient` argument to `backward` | Cotangent seed for non-scalar outputs |
| Reuse graph | `retain_graph = Some(true)` | `retain_graph=True` | Keep graph after backward |
| Build grad graph | `create_graph = true` | `create_graph=True` | Make first-order gradients differentiable |
| Graph break | `detach()` | `detach()` | Stop tracking reverse and forward paths |
| JVP direction | `with_tangent_(v)` | forward AD tangent | Seed forward-mode direction |
| HVP | `backward_hvp(...)` | composed `jvp(grad(f))` pattern | Hessian-vector product |

## Context Merge Rule

1. `Variable::new(value)` creates a value with no attached autograd context and
   `requires_grad=false`.
2. `Variable::new_in(value, ctx)` creates a value attached to `ctx` with
   initial `requires_grad=false`.
3. `requires_grad_(true)` keeps existing context if present; if absent, it
   attaches the value to a new context. The auto-created context is an isolated
   instance (not implicitly shared with other variables). For multi-leaf
   gradients through one graph, callers must explicitly share context via
   `new_in(..., shared_ctx)`. Implicit context merge on operation edges is
   not allowed in this design; use explicit shared context + `new_in`.
4. `requires_grad_(false)` clears only the `requires_grad` flag and keeps the
   current context linkage; use `detach()` when graph linkage must be severed.
   Backward executors must not accumulate `.grad()` / `.hvp()` into variables
   whose `requires_grad` is `false` at backward invocation time, even when they
   remain context-attached.
5. Binary/multi-input AD operations require all attached contexts to be the
   same `context_id`; otherwise they return `InvalidArgument`.
6. Multi-leaf graphs should share one explicit context:

```rust
let ctx = AutogradContext::<MyValue>::new();
let x = Variable::new_in(x0, Arc::clone(&ctx)).requires_grad_(true)?;
let y = Variable::new_in(y0, Arc::clone(&ctx)).requires_grad_(true)?;
let z = add(&x, &y)?;
```

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
assert_eq!(y1.tangent().unwrap().0, 6.0);

// backward
let x2 = ADMyScalar::new(MyScalar(3.0)).requires_grad_(true)?;
let y2 = square(&x2)?;
y2.backward(BackwardOptions::default())?;
assert_eq!(x2.grad().unwrap().0, 6.0);

// hvp
let x3 = ADMyScalar::new(MyScalar(3.0))
    .requires_grad_(true)?
    .with_tangent_(MyScalar(5.0))?;
let y3 = square(&x3)?;
y3.backward_hvp(BackwardOptions::default())?;
assert_eq!(x3.hvp().unwrap().0, 10.0);
```

### 2. `retain_graph = Some(true)`

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
```

### 4. `create_graph = true` (higher-order differentiation, no HVP helper)

```rust
let x = ADMyScalar::new(MyScalar(3.0)).requires_grad_(true)?;
let loss = cube(&x)?; // f(x) = x^3

// First derivative as Variable, graph-connected for second derivative.
let gx = autograd::grad_variable(
    &loss,
    &[&x],
    BackwardOptions {
        retain_graph: Some(true),
        create_graph: true,
        ..Default::default()
    },
)?[0].clone();

// Second derivative by differentiating gx (no dedicated grad_hvp API).
let gxx = autograd::grad_variable(
    &gx,
    &[&x],
    BackwardOptions::default(),
)?[0].clone();

assert_eq!(gxx.value().0, 18.0); // d/dx(3x^2) at x=3
```

### 5. `zero_grad` clears accumulated buffers

```rust
let x = ADMyScalar::new(MyScalar(2.0))
    .requires_grad_(true)?
    .with_tangent_(MyScalar(1.0))?;
let loss = square(&x)?;
loss.backward_hvp(BackwardOptions::default())?;

assert!(x.grad().is_some());
assert!(x.hvp().is_some());

x.zero_grad()?;
assert!(x.grad().is_none());
assert!(x.hvp().is_none());
```

### 6. Non-scalar output requires explicit seed

```rust
let x = Variable::new(vec2(1.0, 2.0)).requires_grad_(true)?;
let y = square_vec2(&x)?; // output is non-scalar

// Without seed_grad, this must fail.
assert!(matches!(
    y.backward(BackwardOptions::default()),
    Err(AdError::InvalidArgument(_))
));

// Provide seed in output cotangent space.
y.backward(BackwardOptions {
    seed_grad: Some(vec2(1.0, 1.0)),
    ..Default::default()
})?;
```

### 7. Heterogeneous graph (`DynTape`) with mixed custom types

```rust
let tape = DynTape::new();
let s = tape.leaf_with_tangent(MyScalar(3.0), MyScalar(1.0))?;
let v = tape.leaf_with_tangent(MyVec2([1.0, 2.0]), MyVec2([0.5, -1.0]))?;

let y = mul_scalar_vec(&tape, &s, &v)?;  // MyScalar x MyVec2 -> MyVec2
let loss = squared_norm(&tape, &y)?;     // MyVec2 -> MyScalar

let result = tape.hvp(&loss, DynHvpOptions::default())?;
let grad_s = result.gradients.get::<MyScalar>(s.node_id()).unwrap();
let hvp_s = result.hvp.get::<MyScalar>(s.node_id()).unwrap();
let _ = (grad_s, hvp_s);
```

## Retain/Create Contract

Define:

```text
effective_retain_graph = options.retain_graph.unwrap_or(options.create_graph)
```

Required behavior:

- `retain_graph=None, create_graph=false`:
  graph is freed after backward query/execute.
- `retain_graph=None, create_graph=true`:
  graph is retained by default for higher-order use.
- `retain_graph=Some(v)`:
  explicit override takes precedence over `create_graph`.

## HVP Path Contract

`backward_hvp` and `DynTape::hvp` require tangent-seeded leaves.

- Monomorphic:
  at least one participating leaf must carry tangent via `with_tangent_`.
- Heterogeneous:
  at least one participating leaf must be created with
  `leaf_with_tangent`.

If this precondition is not met, return `InvalidArgument`.

`backward_hvp` with `create_graph = true` is out of scope in this phase
and must return:

```rust
AdError::ModeNotSupported {
    mode: "create_graph_hvp".into(),
    reason: "...".into(),
}
```

`DynTape::hvp` with `create_graph = true` is also out of scope:

```rust
AdError::ModeNotSupported {
    mode: "create_graph_hvp_dyntape".into(),
    reason: "...".into(),
}
```

## Grad Query API Contract

- `autograd::grad_tangent` / `autograd::grad_dyn_tangent`
  return detached cotangent values.
- `autograd::grad_variable` / `autograd::grad_dyn_variable`
  return graph-connected variables when `create_graph=true`.
- Query APIs must not mutate stored `.grad()` / `.hvp()` buffers.

`create_graph=true` is rejected on tangent-only APIs:

- `grad_tangent` -> `ModeNotSupported { mode: "create_graph_tangent", ... }`
- `grad_dyn_tangent` -> `ModeNotSupported { mode: "create_graph_tangent_dyntape", ... }`

## Normative Rules

1. Effective retain policy is:
   `effective_retain_graph = options.retain_graph.unwrap_or(options.create_graph)`.
2. `retain_graph=Some(v)` must override `create_graph`.
3. Monomorphic `Variable<V>` may hold non-scalar values.
4. `autograd::grad_tangent` and `autograd::grad_variable` are single-output
   APIs in this phase.
5. Heterogeneous `autograd::grad_dyn_tangent` / `grad_dyn_variable` are
   multi-output APIs in this phase.
6. `autograd::grad_tangent` always returns detached `V::Tangent` values and
   must reject `create_graph = true` with
   `ModeNotSupported { mode: "create_graph_tangent", ... }`.
7. `autograd::grad_dyn_tangent` always returns detached `DynTangent` values and
   must reject `create_graph = true` with
   `ModeNotSupported { mode: "create_graph_tangent_dyntape", ... }`.
   `Variable::backward` accepts `create_graph = true`; in this phase this flag
   affects only effective retain policy (rule 1), and stored `.grad()` values
   remain detached `V::Tangent`. For graph-connected first-order gradients,
   use `autograd::grad_variable` / `autograd::grad_dyn_variable`.
8. Scalarity checks are normative:
   - monomorphic path uses `Variable::is_scalar()`
   - heterogeneous path uses `DynVariable::is_scalar()`.
9. Non-scalar output requires explicit seed gradient (`options.seed_grad`) for
   `backward`, `backward_hvp`, `grad_tangent`, and `grad_variable`.
10. For scalar output, `seed_grad=None` is interpreted as
    `seed_grad=Some(output.ones_like())`.
11. Non-scalar heterogeneous outputs require explicit seed gradients
    (`DynBackwardOptions.seed_grads`) for `grad_dyn_tangent` and
    `grad_dyn_variable`.
12. Non-scalar `DynTape::hvp` `loss` input requires `DynHvpOptions.seed_grad`
    (seed in the cotangent space of `loss`).
13. `DynBackwardOptions.seed_grads` is positionally aligned with `outputs`:
    `seed_grads[i]` is the seed for `outputs[i]`, and lengths must match exactly.
14. When `DynBackwardOptions.seed_grads` is `None`, all outputs must be scalar;
    otherwise return `InvalidArgument` (`DynVariable::is_scalar()` defines this).
    For non-scalar outputs in this phase, seed construction is caller-provided
    with concrete output-type knowledge; fully opaque auto-seeding is out of
    scope.
15. For `DynTape::hvp`: scalar `loss` allows `seed_grad=None`; non-scalar
    `loss` requires `seed_grad=Some(...)`.
16. `backward` and `backward_hvp` accumulate by default; for `backward_hvp`,
    both `.grad()` and `.hvp()` are additive accumulators. `retain_graph`
    controls graph lifetime, not accumulation semantics. Mixing
    `backward` then `backward_hvp` also additively accumulates `.grad()`
    contributions unless `zero_grad()` is called between them.
    `backward_hvp` must contribute the same first-order reverse gradient as
    `backward` for the same `seed_grad`, graph state, and retain settings.
    "Same" means the identical floating-point result produced by the same VJP
    code path (no approximation or finite-difference substitution).
    Rules that cannot satisfy this must return
    `ModeNotSupported { mode: "hvp", ... }`.
17. All `autograd::grad_*` query APIs are side-effect free with respect to
    `.grad()` / `.hvp()` accumulators.
18. `zero_grad()` clears only the called variable's `.grad()` and `.hvp()`
    accumulators (no cross-variable side effects). Stored `.tangent()` is not
    cleared by `zero_grad()`.
19. `zero_grad()` is valid on leaf variables only; calling it on non-leaf
    variables returns `InvalidArgument`.
20. `backward_hvp` with `create_graph = true` is out of scope in this phase and
    returns `ModeNotSupported { mode: "create_graph_hvp", ... }`.
21. `backward_hvp` requires at least one tangent-seeded leaf (e.g., via
    `with_tangent_` / `leaf_with_tangent`); otherwise return `InvalidArgument`.
22. `backward_hvp` follows the same retain/free graph-lifetime policy as
    `backward` (rule 1).
23. `DynTape::hvp` with `create_graph = true` is out of scope in this phase and
    returns `ModeNotSupported { mode: "create_graph_hvp_dyntape", ... }`.

## Error Contract

Error taxonomy:

1. `InvalidArgument`:
   - context mismatch
   - missing seed on non-scalar output
   - tangent/shape mismatch
   - no tangent-seeded leaves for HVP
2. `ModeNotSupported`:
   - unsupported `create_graph` in tangent-only APIs
   - unsupported `create_graph` in HVP APIs for this phase
3. `GraphFreed`:
   - backward/query called after graph free when effective retain is false

## Test Pattern Contract

Every implementation and refactor must preserve the following test patterns.
IDs are stable contract labels used across review and implementation plans.

| ID | Category | API | Scenario | Expected |
|----|----------|-----|----------|----------|
| AD-NEXT-001 | Basic | `Variable<V>` | primal-only value (`requires_grad=false`) | forward value is correct; `.grad()`/`.hvp()` are `None` |
| AD-NEXT-002 | Basic | `with_tangent_` + op | JVP through single op (`square`) | tangent matches analytic JVP |
| AD-NEXT-003 | Basic | `backward` | scalar output, tracked leaf | `.grad()` matches analytic gradient |
| AD-NEXT-004 | Higher-order | `grad_variable` | `create_graph=true` then second grad | second derivative value is correct without HVP helper |
| AD-NEXT-005 | Error | `grad_tangent` | `create_graph=true` | `ModeNotSupported { mode: "create_graph_tangent", ... }` |
| AD-NEXT-006 | Retain | `backward` | `retain_graph=Some(true)` repeated backward | second call succeeds; grad accumulates |
| AD-NEXT-007 | Detach | mixed graph | Upstream tracked value + detached branch in same formula | No gradient/hvp flows through detached branch |
| AD-NEXT-008 | Seed gradient | `backward`/`grad_*` | Non-scalar output without `seed_grad` | Returns `InvalidArgument` |
| AD-NEXT-009 | Seed gradient | `backward`/`grad_*` | Non-scalar output with valid `seed_grad` | Succeeds and matches analytic seeded result |
| AD-NEXT-010 | Retain policy | `backward`/`grad_*` | `retain_graph=None`, `create_graph=true` | Effective retain is `true`; repeated backward/grad on same graph succeeds. For `backward`, stored `.grad()` stays detached `V::Tangent` |
| AD-NEXT-011 | Reset semantics | `zero_grad` | After `backward` + `backward_hvp`, call `zero_grad` | Both stored `.grad()` and `.hvp()` are cleared |
| AD-NEXT-012 | Heterogeneous | `DynTape` | Mixed custom types (e.g., scalar × vec -> vec -> scalar loss) | Reverse/HVP result retrieval by concrete type succeeds |
| AD-NEXT-013 | Context contract | multi-input op | Mix different AD contexts in one op | Returns `InvalidArgument` |
| AD-NEXT-014 | Retain override | `backward`/`grad_*` | `retain_graph=Some(false)`, `create_graph=true` | Explicit retain override wins; graph is freed after call |
| AD-NEXT-015 | HVP mode gate | `backward_hvp` | `create_graph=true` | `ModeNotSupported { mode: "create_graph_hvp", ... }` |
| AD-NEXT-016 | HVP precondition | `backward_hvp` | No tangent-seeded leaves | Returns `InvalidArgument` |
| AD-NEXT-017 | Heterogeneous scalarity | `grad_dyn_*` | Mixed scalar/non-scalar outputs without `seed_grads` | Returns `InvalidArgument` |
| AD-NEXT-018 | Retain default free | `backward`/`grad_*` | `retain_graph=None`, `create_graph=false` | graph freed after call; second call errors with `GraphFreed` |
| AD-NEXT-019 | Heterogeneous mode gate | `grad_dyn_*` | Unsupported heterogeneous op/rule path | `ModeNotSupported` with clear mode tag |
| AD-NEXT-020 | Heterogeneous mode gate | `grad_dyn_tangent` | `create_graph=true` on dyn tangent query | `ModeNotSupported { mode: "create_graph_tangent_dyntape", ... }` |
| AD-NEXT-021 | Heterogeneous retain policy | `grad_dyn_variable` | `retain_graph=None`, `create_graph=true` | effective retain is true; repeated query on same outputs succeeds |
| AD-NEXT-022 | Heterogeneous retain override | `grad_dyn_variable` | `retain_graph=Some(false)`, `create_graph=true` | graph freed; repeated query on same outputs returns `GraphFreed` |
| AD-NEXT-023 | Heterogeneous seed contract | `grad_dyn_*` | non-scalar outputs with explicit `seed_grads` | succeeds and matches analytic seeded result |
| AD-NEXT-024 | Heterogeneous API baseline | `DynTape` | leaf + leaf_with_tangent + value_as + type mismatch | typed extraction works; mismatch returns `InvalidArgument` |
| AD-NEXT-025 | Accumulation invariant | `backward` + `backward_hvp` | same graph, same output, repeated calls | `.grad()` and `.hvp()` accumulate additively |
| AD-NEXT-026 | Query API split | `grad_tangent` vs `grad_variable` | same scalar output | tangent API returns detached cotangent; variable API returns `Variable<V>` (graph-capable under `create_graph=true`) |
| AD-NEXT-027 | Graph lifetime | `grad_variable` | `create_graph=true` with `retain_graph=None` | repeated gradient query succeeds without rebuilding graph |
| AD-NEXT-028 | Context merge | `new_in` + ops | one input tracked in shared ctx + one constant (`requires_grad=false`) | output stays in shared context and backward succeeds |
| AD-NEXT-029 | Detach idempotence | `detach()` | call `detach` twice on tracked value | remains untracked; no context/node/tangent resurrection |
| AD-NEXT-030 | Heterogeneous retention default | `grad_dyn_variable` | `create_graph=false`, `retain_graph=None` | graph freed after query; second query returns `GraphFreed` |
| AD-NEXT-031 | Option equality | monomorphic vs heterogeneous | same logical retain/create settings | effective lifetime behavior matches across APIs |
| AD-NEXT-032 | HVP accumulation split | `backward` then `backward_hvp` | same output | second call increments `.grad()` and populates/increments `.hvp()` |
| AD-NEXT-033 | Leaf-only reset | `zero_grad()` | called on non-leaf variable | returns `InvalidArgument`; leaf buffers unchanged |
| AD-NEXT-034 | Seed contract | `DynTape::hvp` | non-scalar loss with `seed_grad=None` | returns `InvalidArgument` |
| AD-NEXT-035 | Seed contract | `DynTape::hvp` | scalar loss with `seed_grad=None` | succeeds (implicit ones-like seed) |
| AD-NEXT-036 | Seed alignment | `grad_dyn_*` | `seed_grads.len() != outputs.len()` | returns `InvalidArgument` |
| AD-NEXT-037 | Dyn HVP precondition | `DynTape::hvp` | no tangent-seeded leaves | returns `InvalidArgument` |
| AD-NEXT-038 | Dyn HVP mode gate | `DynTape::hvp` | `create_graph=true` | returns `ModeNotSupported { mode: "create_graph_hvp_dyntape", ... }` |
| AD-NEXT-039 | Cross-tape safety | `grad_dyn_*` | outputs/inputs from different `DynTape` ids | returns `InvalidArgument` |
| AD-NEXT-040 | Tangent persistence on reset | `Variable<V>::zero_grad` | Leaf has `.tangent()` + populated `.grad()`/`.hvp()`, then `zero_grad` | `.grad()`/`.hvp()` are cleared while `.tangent()` stays unchanged |
| AD-NEXT-041 | Shared-context multi-leaf success | multi-input op | Two tracked leaves created via `new_in(..., same_ctx)` | Operation succeeds (no context mismatch) and gradients are correct |
| AD-NEXT-042 | Query side-effect free | `autograd::grad_*` | Run `grad_tangent`, `grad_variable`, `grad_dyn_tangent`, `grad_dyn_variable` after prior grad accumulation | Returned query values are correct and `.grad()`/`.hvp()` buffers are unchanged |
| AD-NEXT-043 | `requires_grad_(false)` semantics | `Variable<V>` | Disable grad on a context-attached variable | `requires_grad=false` while context linkage remains intact (not detached), and no `.grad()`/`.hvp()` accumulation occurs for that variable |
| AD-NEXT-044 | DynTape HVP freed-graph error | `DynTape::hvp` | Run with effective retain=false, then call again on same loss | Second call returns `GraphFreed` |
| AD-NEXT-045 | `grad_tangent` freed-graph error | `autograd::grad_tangent` | Free graph (`retain_graph=false`) then query again on same output | Returns `GraphFreed` |
| AD-NEXT-046 | `grad_dyn_tangent` freed-graph error | `autograd::grad_dyn_tangent` | Free graph (`retain_graph=false`) then query again on same output set | Returns `GraphFreed` |
| AD-NEXT-047 | Leaf-only reset no-op on detached leaf | `Variable<V>::zero_grad` | Detached/primal-only leaf (no context/node) | Returns `Ok(())`; no panic; state unchanged |
| AD-NEXT-048 | Seed propagation parity | `backward` vs `grad_tangent` | Same non-scalar output and same explicit `seed_grad` | Per-input cotangents match exactly |
| AD-NEXT-049 | Context mismatch parity | monomorphic vs heterogeneous | Cross-context (or cross-tape) mixed-input op attempt | Both APIs reject with `InvalidArgument` |
| AD-NEXT-050 | Scalar introspection | `Variable::is_scalar` / `DynVariable::is_scalar` | Scalar and non-scalar custom values | Returns correct scalarity and drives seed validation branches |

### Notes

1. `AD-NEXT-004` verifies that higher-order differentiation is available
   without introducing a dedicated `autograd::grad_*_hvp` API.
2. `AD-NEXT-005` and `AD-NEXT-020` enforce "no graph-connected tangent API"
   in this phase; higher-order paths must go through variable-returning APIs.
3. `AD-NEXT-006` and `AD-NEXT-025` together verify additive accumulation
   semantics and guard against accidental overwrite behavior.
4. `AD-NEXT-008`, `AD-NEXT-009`, `AD-NEXT-017`, and `AD-NEXT-023` enforce
   seeded non-scalar behavior parity across monomorphic and heterogeneous APIs.
5. `AD-NEXT-010`, `AD-NEXT-014`, and `AD-NEXT-018` cover retain/create policy
   interactions for monomorphic APIs.
6. `AD-NEXT-021`, `AD-NEXT-022`, and `AD-NEXT-030` must mirror the same
   retain/create policy in heterogeneous APIs.
7. `AD-NEXT-018` must assert default graph-free behavior when
   `create_graph=false`.
   `AD-NEXT-010` must also assert that `backward(create_graph=true)` keeps
   stored `.grad()` detached (graph-connected gradients come from `grad_variable`).
8. `AD-NEXT-019` must cover heterogeneous rule families that do not expose
   graph-aware pullback wiring yet.
9. `AD-NEXT-020` through `AD-NEXT-023` must cover heterogeneous parity for
   mode-gating, retain policy, and seed handling.
10. `AD-NEXT-024` is an API-baseline smoke test for dynamic typing ergonomics.
11. `AD-NEXT-026` ensures semantic separation between detached cotangent
    queries and graph-connected gradient variables.
12. `AD-NEXT-027` and `AD-NEXT-030` together validate create-graph retention
    behavior for `grad_variable` and `grad_dyn_variable`.
13. `AD-NEXT-028` explicitly tests the "tracked + constant" merge contract.
14. `AD-NEXT-029` protects detach semantics against accidental metadata reuse.
15. `AD-NEXT-031` enforces lifetime-policy consistency between monomorphic and
    heterogeneous APIs.
16. `AD-NEXT-032` guards mixed backward/HVP accumulation behavior.
17. `AD-NEXT-033` enforces leaf-only reset contract.
18. `AD-NEXT-034` through `AD-NEXT-039` are heterogeneous seed/mode/context
    safety contracts.
19. `AD-NEXT-040` protects forward-tangent persistence semantics.
20. `AD-NEXT-041` verifies explicit shared-context happy path.
21. `AD-NEXT-042` ensures query APIs do not mutate stateful accumulators.
22. `AD-NEXT-042` must validate query-only semantics for tangent APIs
    (`grad_tangent` / `grad_dyn_tangent`) with no buffer mutation.
23. `AD-NEXT-043` must validate that `requires_grad_(false)` is not equivalent
    to `detach()`, and that accumulation is skipped while disabled.
24. `AD-NEXT-044` must validate `GraphFreed` parity for `DynTape::hvp`.
25. `AD-NEXT-045` and `AD-NEXT-046` must validate freed-graph parity for
    tangent query APIs (`grad_tangent` / `grad_dyn_tangent`).
26. `AD-NEXT-047` distinguishes leaf-API no-op behavior from non-leaf rejection.
27. `AD-NEXT-048` enforces seed-consistency between stateful `backward` and
    query-style `grad_tangent`.
28. `AD-NEXT-049` enforces monomorphic/heterogeneous parity for context-mismatch
    rejection.
29. `AD-NEXT-050` enforces scalar-introspection branch coverage for seed
    validation and default-seed behavior.
