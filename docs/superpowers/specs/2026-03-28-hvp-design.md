# Public Tensor HVP Design

Issue: #511

## Goal

Add a public `tenferro::hvp` API that computes the Hessian-vector product
(H·v) for scalar-valued tensor computations, leveraging the forward-over-reverse
approach already supported by the tidu tape engine.

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

Existing call sites register VJP-only closures. To avoid rewriting all ~30 sites
at once, provide a `ClosureRule<T>` wrapper that implements `ReverseRule<Tensor<T>>`
with `HvpNotSupported` defaults:

```rust
pub(crate) struct ClosureRule<T> {
    pullback_fn: Box<dyn Fn(&StructuredTensor<T>) -> Result<Vec<(NodeId, StructuredTensor<T>)>>
        + Send + Sync + 'static>,
}

impl<T: Scalar> ReverseRule<Tensor<T>> for ClosureRule<T> {
    fn pullback(&self, cotangent: &Tensor<T>) -> AdResult<Vec<(NodeId, Tensor<T>)>> {
        // Extract from Tensor<T>, call closure, wrap back
        ...
    }
    fn inputs(&self) -> Vec<NodeId> { Vec::new() }
    // forward_tangents and pullback_with_tangents use defaults (HvpNotSupported)
}
```

### 3. Einsum: register EinsumReverseRule directly

Currently einsum wraps its rule into a closure. Change to register
`EinsumReverseRule` (which already implements `ReverseRule<Tensor<T>>`)
directly, so its `forward_tangents` and `pullback_with_tangents` reach the tape.

The `EinsumReverseRule` must be updated to use the new
`pullback_with_tangents` signature (read input tangents from the closure
instead of stored `input_tangents` field).

### 4. Public API

```rust
/// Result of a Hessian-vector product computation.
pub struct HvpResult {
    /// Gradients of the output with respect to each input.
    pub gradients: Vec<Option<Tensor>>,
    /// Hessian-vector products for each input.
    pub hvps: Vec<Option<Tensor>>,
}

/// Computes gradient and Hessian-vector product for a scalar output.
///
/// `output` must be a scalar (rank-0) tensor on a reverse-mode tape.
/// `inputs` are the tensors to differentiate with respect to.
/// `v` provides the tangent direction for each input (same shapes as inputs).
///
/// Returns both gradients and HVPs for each input.
///
/// # Examples
///
/// ```ignore
/// // f(x) = sum(x^2), H = 2I, Hv = 2v
/// let result = hvp(&output, &[&x], &[&v])?;
/// ```
pub fn hvp(
    output: &Tensor,
    inputs: &[&Tensor],
    v: &[&Tensor],
) -> Result<HvpResult> {
    ...
}
```

Implementation:
1. Validate `output` is scalar (num_elements == 1)
2. Validate `v.len() == inputs.len()` and shapes match
3. Extract reverse tape from output
4. Build `HashMap<NodeId, DynTensor>` from inputs' node IDs and corresponding v tensors
5. Call `tape.hvp(&tracked_output, &leaf_tangents)`
6. Project results to requested inputs (same as `grad` does with `pullback_wrt`)
7. Wrap into `HvpResult`

### 5. MixedTensorRuleAdapter

For mixed-type operations (e.g., SVD returns real singular values from complex input),
the `MixedTensorRuleAdapter` follows the same pattern as `TensorRuleAdapter` but with
type conversion between `TOut` and `TIn`. HVP defaults apply for now.

## Scope

### In scope (this issue)

- Bump chainrules-core dependency to rev 6cc4677
- Bump tidu dependency to rev ea504cd
- Refactor `tape/registry.rs`: `register_rule` accepts `Box<dyn ReverseRule<Tensor<T>>>`
- Add `ClosureRule<T>` helper for existing VJP-only closures
- Migrate all ~30 `register_rule` call sites to use `ClosureRule<T>`
- Update `EinsumReverseRule` to new `pullback_with_tangents` signature
  (remove stored `input_tangents`, use closure)
- Register `EinsumReverseRule` directly (not via closure decomposition)
- Add `tenferro::hvp` public function in `autograd_api.rs`
- Add `HvpResult` type
- Add tests:
  - f(x) = sum(x^2) → Hv = 2v
  - Separable two-input quadratic
  - Non-scalar output error

### Out of scope (follow-up issues)

- HVP for linalg operations (SVD, QR, etc.) — each needs `forward_tangents`
  and `pullback_with_tangents` implemented
- HVP for scalar ops (exp, sin, etc.)
- HVP for reduction ops
- HVP for layout ops (reshape, permute, etc.)
- `create_graph` support in `grad`/`backward`

## Migration Plan

### Phase 1: Registry refactor

1. Bump chainrules-core and tidu dependencies
2. Change `register_rule` to accept `Box<dyn ReverseRule<Tensor<T>>>`
3. Add `ClosureRule<T>` wrapper
4. Update all existing call sites to wrap closures in `ClosureRule`
5. Update `TensorRuleAdapter` and `MixedTensorRuleAdapter` to forward all
   `ReverseRule` methods

### Phase 2: Einsum HVP

1. Update `EinsumReverseRule`:
   - Remove `input_tangents` field
   - Implement `forward_tangents` using einsum frule
   - Update `pullback_with_tangents` to read tangents from closure
2. Change einsum AD registration to pass `EinsumReverseRule` directly
   to `register_rule`

### Phase 3: Public API + Tests

1. Add `HvpResult` struct
2. Add `hvp()` function in `autograd_api.rs`
3. Add integration tests

## Error Handling

- `Error::UnsupportedAdOp { op: "hvp" }` — if output is not on a reverse tape
- `Error::InvalidAdTensor` — shape mismatch between inputs and v
- `AutodiffError::HvpNotSupported` — if any rule in the graph doesn't support HVP
  (e.g., a linalg operation)
- `AutodiffError::NonScalarLoss` — if output is not rank-0

## Test Plan

### Test 1: f(x) = sum(x^2) → Hv = 2v

```text
x = [1.0, 2.0, 3.0], v = [1.0, 1.0, 1.0]
f(x) = x_1^2 + x_2^2 + x_3^2 = 14
grad = [2, 4, 6]
H = diag(2, 2, 2)
Hv = [2, 2, 2]
```

### Test 2: Separable two-input quadratic

```text
f(x, y) = sum(x^2) + sum(x * y)
grad_x = 2x + y, grad_y = x
H_xx = 2I, H_xy = I, H_yx = I, H_yy = 0
Hv: hvp_x = 2*vx + vy, hvp_y = vx
```

### Test 3: Non-scalar output error

```text
f(x) = x^2  (vector output, not summed)
hvp(&f_output, &[&x], &[&v]) → Err(NonScalarLoss)
```

### Test 4: HvpNotSupported propagation

```text
Graph containing an operation that only has ClosureRule (no HVP).
hvp() → Err(HvpNotSupported)
```
