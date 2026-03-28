# Public Tensor HVP Design

Issue: #511

## Goal

Add a public `tenferro::hvp` API that computes the Hessian-vector product
(H·v) for scalar-valued tensor computations, leveraging the forward-over-reverse
approach already supported by the tidu tape engine.

### Naming convention

This spec uses fully qualified type names where ambiguity exists:

- `tenferro::Tensor` — the public type-erased enum (`F32 | F64 | C32 | C64`)
- `tenferro_tensor::Tensor<T>` — the inner typed dense tensor
- `tenferro::StructuredTensor<T>` — typed tensor with structure metadata (dims, axis classes)
- `DynTensor` — `tenferro::DynTensor`, enum of `StructuredTensor<T>` variants

## Background

### Mathematical definition

For a scalar function f: R^n → R with input x and direction vector v:

- Gradient: ∇f(x) ∈ R^n
- Hessian: H(x) = ∇²f(x) ∈ R^(n×n)
- HVP: H(x)·v ∈ R^n

### Forward-over-reverse approach

tenferro uses forward-over-reverse (JVP of VJP), NOT PyTorch's double backward
trick (reverse-over-reverse). This is more efficient — a single forward+reverse
traversal computes both gradient and HVP simultaneously.

For each reverse rule:

```
pullback(c̄)                          → grad       (standard VJP)
pullback_with_tangents(c̄, ċ, tangent_provider) → (grad, ġrad)  (VJP + HVP contribution)
```

Where:
- c̄ = output cotangent (reverse seed)
- ċ = cotangent tangent (propagated from upstream in reverse direction)
- tangent_provider = closure providing primal tangents by NodeId
- grad = standard gradient
- ġrad = gradient tangent = HVP contribution

### Current state

- `EinsumReverseRule` already implements `pullback_with_tangents` (in tenferro-einsum)
- tidu's `Tape::hvp` now accepts deferred leaf tangents via `HashMap<NodeId, V::Tangent>`
  (tidu-rs#6, rev ea504cd)
- chainrules-core's `ReverseRule` now has `forward_tangents` and updated
  `pullback_with_tangents` with `input_tangents` closure parameter
  (chainrules-rs#6, rev 6cc4677)
- tenferro's `tape/registry.rs` still uses the old `PullbackRule<T>` closure-only adapter
- tenferro's `autograd_api.rs` has no `hvp` function

## Upstream API (tidu + chainrules-core)

### ReverseRule trait (chainrules-core rev 6cc4677)

```rust
pub trait ReverseRule<V: Differentiable>: Send + Sync {
    fn pullback(&self, cotangent: &V::Tangent) -> AdResult<Vec<PullbackEntry<V>>>;
    fn inputs(&self) -> Vec<NodeId>;

    // HVP support — default returns HvpNotSupported
    fn forward_tangents<'t>(
        &self,
        input_tangents: &dyn Fn(NodeId) -> Option<&'t V::Tangent>,
    ) -> AdResult<Option<V::Tangent>>
    where V::Tangent: 't;

    fn pullback_with_tangents<'t>(
        &self,
        cotangent: &V::Tangent,
        cotangent_tangent: &V::Tangent,
        input_tangents: &dyn Fn(NodeId) -> Option<&'t V::Tangent>,
    ) -> AdResult<Vec<PullbackWithTangentsEntry<V>>>
    where V::Tangent: 't;
}
```

### Tape::hvp (tidu rev ea504cd)

```rust
pub fn hvp(
    &self,
    loss: &TrackedValue<V>,
    leaf_tangents: &HashMap<NodeId, V::Tangent>,
) -> AdResult<HvpResult<V>>
where V::Tangent: Differentiable<Tangent = V::Tangent>;

pub struct HvpResult<V: Differentiable> {
    pub gradients: Gradients<V>,
    pub hvp: Gradients<V>,
}
```

Two-phase execution:
1. **Forward pass**: walk graph topologically, call `forward_tangents` on each rule
   to propagate primal tangents from leaves to output
2. **Reverse pass**: walk graph in reverse, call `pullback_with_tangents` with both
   cotangent tangent and primal tangent provider

## Design

### 1. Registry: accept full ReverseRule objects

**Current**: `register_rule` takes `PullbackRule<T>` (VJP-only closure).
Rules like `EinsumReverseRule` are decomposed into closures, losing HVP info.

**New**: `register_rule` takes `Box<dyn ReverseRule<Tensor<T>>>`.
The `TensorRuleAdapter` wraps it and forwards all methods
(pullback, forward_tangents, pullback_with_tangents) with
`Tensor<T>` ↔ `DynTensor` type conversion.

```rust
// New type accepted by register_rule
// (replaces PullbackRule<T> = Box<dyn Fn(&StructuredTensor<T>) -> ...>)

struct TensorRuleAdapter<T: DynTensorTyped> {
    rule: Box<dyn ReverseRule<Tensor<T>>>,
}

impl<T: DynTensorTyped> ReverseRule<DynTensor> for TensorRuleAdapter<T> {
    fn pullback(&self, cotangent: &DynTensor) -> AdResult<Vec<(NodeId, DynTensor)>> {
        let cotangent = extract_tensor::<T>(cotangent)?;
        self.rule.pullback(&cotangent)?
            .into_iter()
            .map(|(node, grad)| (node, T::into_dyn(grad)))
            .collect()
    }

    fn forward_tangents<'t>(
        &self,
        input_tangents: &dyn Fn(NodeId) -> Option<&'t DynTensor>,
    ) -> AdResult<Option<DynTensor>> {
        // Adapter: DynTensor → StructuredTensor<T> → Tensor<T> (payload)
        let typed_fn = |node: NodeId| -> Option<&'t Tensor<T>> {
            input_tangents(node)
                .and_then(T::structured_ref)  // &'t StructuredTensor<T>
                .map(|s| s.payload())          // &'t Tensor<T>
        };
        self.rule.forward_tangents(&typed_fn)?
            .map(|t| Ok(T::into_dyn(StructuredTensor::from_dense(t))))
            .transpose()
    }

    fn pullback_with_tangents<'t>(
        &self,
        cotangent: &DynTensor,
        cotangent_tangent: &DynTensor,
        input_tangents: &dyn Fn(NodeId) -> Option<&'t DynTensor>,
    ) -> AdResult<Vec<(NodeId, DynTensor, DynTensor)>> {
        let cotangent = T::structured_ref(cotangent)
            .ok_or(AutodiffError::InvalidArgument(...))?
            .payload();
        let cotangent_tangent = T::structured_ref(cotangent_tangent)
            .ok_or(AutodiffError::InvalidArgument(...))?
            .payload();
        let typed_fn = |node: NodeId| -> Option<&'t Tensor<T>> {
            input_tangents(node)
                .and_then(T::structured_ref)
                .map(|s| s.payload())
        };
        self.rule.pullback_with_tangents(cotangent, cotangent_tangent, &typed_fn)?
            .into_iter()
            .map(|(node, g, gt)| {
                (node,
                 T::into_dyn(StructuredTensor::from_dense(g)),
                 T::into_dyn(StructuredTensor::from_dense(gt)))
            })
            .collect()
    }

    fn inputs(&self) -> Vec<NodeId> {
        self.rule.inputs()
    }
}
```

### 2. ClosureRule helper for migration

Existing call sites register VJP-only closures. To avoid rewriting all ~30+ sites
at once, provide a `ClosureRule<T>` wrapper that implements
`ReverseRule<tenferro_tensor::Tensor<T>>` with `HvpNotSupported` defaults.

The wrapper bridges the type gap: existing closures expect
`&StructuredTensor<T>` but `ReverseRule<tenferro_tensor::Tensor<T>>::pullback`
provides `&tenferro_tensor::Tensor<T>`.

```rust
pub(crate) struct ClosureRule<T> {
    pullback_fn: Box<dyn Fn(&StructuredTensor<T>) -> Result<Vec<(NodeId, StructuredTensor<T>)>>
        + Send + Sync + 'static>,
    /// Input node IDs this rule depends on.
    /// Required for forward tangent propagation (Phase 1 of HVP).
    input_node_ids: Vec<NodeId>,
}

impl<T: Scalar> ReverseRule<tenferro_tensor::Tensor<T>> for ClosureRule<T> {
    fn pullback(
        &self,
        cotangent: &tenferro_tensor::Tensor<T>,
    ) -> AdResult<Vec<(NodeId, tenferro_tensor::Tensor<T>)>> {
        // Wrap Tensor<T> → StructuredTensor for the closure
        let structured = StructuredTensor::from_dense(cotangent.clone());
        let results = (self.pullback_fn)(&structured)
            .map_err(|e| AutodiffError::InvalidArgument(e.to_string()))?;
        // Extract Tensor<T> payloads from StructuredTensor results
        Ok(results
            .into_iter()
            .map(|(node, st)| (node, st.into_payload()))
            .collect())
    }

    fn inputs(&self) -> Vec<NodeId> {
        self.input_node_ids.clone()
    }

    // forward_tangents and pullback_with_tangents use trait defaults (HvpNotSupported)
}
```

**Migration note**: each existing `register_rule` call site must also provide
`input_node_ids`. These are typically available from `collect_reverse_input_nodes`
or `collect_reverse_input_specs` already called at the registration site.

### 3. Einsum HVP: two AD levels

Einsum AD exists at two independent levels:

**Level A — `tenferro-einsum` crate (`tracked_einsum`)**:
Operates on `Tape<tenferro_tensor::Tensor<T>>`. Creates `EinsumReverseRule`
and registers it directly via `tape.record_op()`. This is the typed,
lower-level API.

**Level B — `tenferro/src/ops/einsum/ad.rs` (`EinsumAdBuilder::run`)**:
Operates on `Tape<DynTensor>`. Has two sub-paths:
- **Structured path** (non-parenthesized subscripts): calls
  `structured_einsum_pullback_in_backend` via a closure
- **Dense path** (parenthesized or fallback): calls
  `dense_einsum_pullback_in_backend` via a closure

Both Level B sub-paths register VJP-only closures through `register_rule`.

**Migration strategy**:

1. **Level A**: Update `EinsumReverseRule` to the new chainrules-core API:
   - Remove `input_tangents: Vec<Option<Tensor<T>>>` field
   - Implement `forward_tangents` using `einsum_frule_impl`
   - Update `pullback_with_tangents` to read tangents from the closure
   - Note: `EinsumReverseRule` uses `Arc<Mutex<BackendContext>>`. Both
     `forward_tangents` and `pullback_with_tangents` will acquire this lock.
     This is safe (single-threaded tape traversal) but should be documented.

2. **Level B — Dense path**: Create a new `DenseEinsumRule<T>` struct that
   implements `ReverseRule<tenferro_tensor::Tensor<T>>` with HVP support,
   delegating to `tf_einsum::einsum_rrule` / `tf_einsum::einsum_frule`.
   Register it directly via `register_rule` instead of a closure.

3. **Level B — Structured path**: Initially wrap in `ClosureRule` (no HVP).
   Full HVP for structured einsum is a follow-up. The structured path handles
   delta injection for trace operations, which adds complexity to the HVP
   derivation.

### 4. Public API

`tenferro::Tensor` is the public type-erased enum used in the API.

```rust
/// Options for HVP computation.
#[derive(Debug, Clone, Default)]
pub struct HvpOptions {
    /// If true, do not free the computation graph after HVP.
    pub retain_graph: bool,
}

/// Result of a Hessian-vector product computation.
pub struct HvpResult {
    /// Gradients of the output with respect to each input.
    /// One entry per input; None if the input is disconnected.
    pub gradients: Vec<Option<Tensor>>,
    /// Hessian-vector products for each input.
    /// One entry per input; None if the input is disconnected.
    pub hvps: Vec<Option<Tensor>>,
}

/// Computes gradient and Hessian-vector product for a scalar output.
///
/// `output` must be a scalar (rank-0) `tenferro::Tensor` on a reverse-mode tape.
/// `inputs` are the `tenferro::Tensor`s to differentiate with respect to.
/// `v` provides the tangent direction for each input (same shapes as inputs).
///
/// Returns both gradients and HVPs for each input.
///
/// # Examples
///
/// ```ignore
/// // f(x) = einsum("i,i->", &[&x, &x])  (= sum(x^2))
/// // H = 2I, Hv = 2v
/// let result = hvp(&output, &[&x], &[&v], HvpOptions::default())?;
/// ```
pub fn hvp(
    output: &Tensor,
    inputs: &[&Tensor],
    v: &[&Tensor],
    options: HvpOptions,
) -> Result<HvpResult> {
    ...
}
```

**Implementation steps**:

1. Validate `output` is scalar (`num_elements == 1`)
2. Validate `v.len() == inputs.len()` and each `v[i]` has the same shape as `inputs[i]`
3. Extract reverse tape from `output` (same pattern as `grad()`)
4. Build `HashMap<NodeId, DynTensor>` by iterating `inputs` and extracting each
   input's `NodeId` from its reverse handle, paired with the corresponding `v[i]`
   wrapped as `DynTensor`
5. Extract `TrackedValue<DynTensor>` from `output` (dispatch over F32/F64/C32/C64)
6. Call `tape.hvp(&tracked_output, &leaf_tangents)` → `tidu::HvpResult<DynTensor>`
7. Project `Gradients<DynTensor>` to requested inputs by NodeId lookup
   (same projection pattern as `Tensor::pullback_wrt` in
   `tenferro/src/core/dynamic/dyn_ad_tensor/pullback.rs`)
8. Convert each `DynTensor` back to `tenferro::Tensor` via `dyn_primal_from_snapshot`
9. If `!options.retain_graph`, call `tape.free_graph()`
10. Return `HvpResult`

### 5. MixedTensorRuleAdapter

For mixed-type operations (e.g., SVD returns real singular values from complex input),
the `MixedTensorRuleAdapter` follows the same pattern as `TensorRuleAdapter` but with
type conversion between `TOut` and `TIn`. HVP defaults apply for now.

## Scope

### In scope (this issue)

- Bump chainrules-core dependency to rev 6cc4677
- Bump tidu dependency to rev ea504cd
- Refactor `tape/registry.rs`: `register_rule` accepts
  `Box<dyn ReverseRule<tenferro_tensor::Tensor<T>>>`
- Add `ClosureRule<T>` helper for existing VJP-only closures (with `input_node_ids`)
- Migrate all ~30+ `register_rule` call sites to use `ClosureRule<T>`
- Update `EinsumReverseRule` (Level A) to new chainrules-core API
  (remove stored `input_tangents`, implement `forward_tangents`, use closure)
- Add `DenseEinsumRule<T>` (Level B, dense path) with HVP support
- Wrap Level B structured path in `ClosureRule` (no HVP for now)
- Add `tenferro::hvp` public function in `autograd_api.rs`
- Add `HvpResult` and `HvpOptions` types
- Add integration tests (einsum-only):
  - f(x) = x^T x → Hv = 2v
  - Separable two-input bilinear
  - Non-scalar output error
  - HvpNotSupported propagation
  - Shape mismatch error

### Out of scope (follow-up issues)

- HVP for structured einsum path (delta injection complexity)
- HVP for linalg operations (SVD, QR, etc.) — each needs `forward_tangents`
  and `pullback_with_tangents` implemented
- HVP for scalar ops (exp, sin, etc.)
- HVP for reduction ops
- HVP for layout ops (reshape, permute, etc.)
- `create_graph` support in `grad`/`backward`

## Migration Plan

### Phase 1: Dependency bump + registry refactor

1. Bump chainrules-core to rev 6cc4677, tidu to rev ea504cd
2. Change `register_rule` signature:
   `Box<dyn ReverseRule<tenferro_tensor::Tensor<T>>>` + type conversion adapter
3. Add `ClosureRule<T>` wrapper (with `input_node_ids` field)
4. Update all ~30+ existing `register_rule` call sites to wrap closures
   in `ClosureRule`. Each site must provide `input_node_ids` (extract from
   existing `collect_reverse_input_nodes` / `collect_reverse_input_specs`).
5. Update `TensorRuleAdapter` to forward `forward_tangents` and
   `pullback_with_tangents` with `DynTensor` ↔ `tenferro_tensor::Tensor<T>`
   type conversion
6. Update `MixedTensorRuleAdapter` similarly (HVP defaults for now)

### Phase 2: Einsum HVP (Level A + Level B dense)

1. Update `EinsumReverseRule` in `tenferro-einsum/src/ad/reverse_rule.rs`:
   - Remove `input_tangents` field
   - Implement `forward_tangents` using `einsum_frule_impl`
   - Update `pullback_with_tangents` to read tangents from closure
   - Note: both methods acquire `Arc<Mutex<BackendContext>>` lock
2. Update `tracked_einsum` in `tenferro-einsum/src/ad/tracked.rs`:
   - Stop passing `input_tangents` to `EinsumReverseRule` constructor
3. Create `DenseEinsumRule<T>` for Level B dense path in
   `tenferro/src/ops/einsum/ad.rs`, implementing
   `ReverseRule<tenferro_tensor::Tensor<T>>` with HVP support
4. Level B structured path: wrap in `ClosureRule` (no HVP)

### Phase 3: Public API + Tests

1. Add `HvpOptions` and `HvpResult` types in `autograd_api.rs`
2. Add `hvp()` function following the implementation steps above
3. Add integration tests in `tenferro/tests/` (einsum-only computations)

## Error Handling

- `Error::UnsupportedAdOp { op: "hvp" }` — if output is not on a reverse tape
- `Error::InvalidAdTensor` — shape mismatch between inputs and v, or dtype mismatch
- `AutodiffError::HvpNotSupported` — if any rule in the graph doesn't support HVP
  (e.g., a linalg operation or the structured einsum path). Note: the error
  does not currently identify WHICH operation caused the failure. Improving
  error specificity (e.g., including the operation name) is desirable but
  requires changes to chainrules-core's `AutodiffError::HvpNotSupported`
  variant and is out of scope for this issue.
- `AutodiffError::NonScalarLoss` — if output is not rank-0
- `Error::MixedReverseTape` — if inputs belong to different tapes

## Test Plan

All tests use **einsum-only** computations to avoid dependency on HVP
support for scalar ops, reduction, or layout ops (which are out of scope).
Tests live in `tenferro/tests/` (integration tests per AGENTS.md convention).

### Test 1: f(x) = x^T x (= sum(x^2)) → Hv = 2v

Uses `einsum("i,i->", &[&x, &x])` to express the scalar quadratic form.

```text
x = [1.0, 2.0, 3.0], v = [1.0, 1.0, 1.0]
f(x) = x_1^2 + x_2^2 + x_3^2 = 14
grad = [2, 4, 6]   (from both operand positions)
H = diag(2, 2, 2)
Hv = [2, 2, 2]
```

### Test 2: Separable two-input bilinear

Uses `einsum("i,i->", &[&x, &y])` to express x^T y.

```text
f(x, y) = x^T y = sum(x_i * y_i)
grad_x = y, grad_y = x
H_xx = 0, H_xy = I, H_yx = I, H_yy = 0
Hv: hvp_x = vy, hvp_y = vx
```

### Test 3: Non-scalar output error

```text
f(x) = einsum("i,i->i", &[&x, &x])  (vector output, not summed)
hvp(&f_output, &[&x], &[&v]) → Err(NonScalarLoss)
```

### Test 4: HvpNotSupported propagation

```text
Graph containing an operation registered with ClosureRule (no HVP impl).
hvp() → Err(HvpNotSupported)
```

### Test 5: Shape mismatch error

```text
v has different shape from input.
hvp(&output, &[&x], &[&v_wrong_shape]) → Err(InvalidAdTensor)
```
