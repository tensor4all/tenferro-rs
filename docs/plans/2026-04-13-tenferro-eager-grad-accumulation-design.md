# Tenferro Eager Gradient Accumulation Design

**Date:** 2026-04-13

**Status:** Proposed and approved

## Goal

Align `tenferro::EagerTensor` reverse-mode behavior with PyTorch accumulation
semantics:

- `backward()` accumulates into existing leaf gradients
- gradient reset is explicit through public API
- public docs describe the real eager AD contract instead of claiming eager has
  no AD

This design assumes the local `Arc<Mutex<...>>` eager context refactor is
already in place on the current branch and builds on top of that shape.

## Scope

This design covers:

- `EagerTensor::backward()` accumulation semantics
- public gradient-reset APIs on `EagerTensor` and `EagerContext`
- a public tracked-state getter on `EagerTensor`
- eager rustdoc and user-facing docs updates
- a small facade cleanup needed for user-facing eager docs

This design does not cover:

- traced AD (`TracedTensor::grad`, `vjp`, `jvp`, `hvp`)
- backward compatibility shims for the old eager overwrite semantics
- higher-order eager AD
- backend-sharing helper APIs beyond the current context model
- eliminating typed eager/linalg host-side copies

## Current State

The current eager AD surface is internally close to usable, but the public
contract is mismatched in two important ways.

### 1. `backward()` overwrites instead of accumulating

Today `EagerTensor::backward()` does this:

1. reject non-scalar outputs
2. call `self.ctx.clear_grads()`
3. run `backward_dag(...)`
4. call `self.ctx.store_grads(&cotangents)`

That means:

- repeated `backward()` calls do **not** accumulate
- untouched leaves are reset to `None`
- the behavior does not match PyTorch users' mental model

### 2. Public docs still claim eager has no AD

User-facing docs currently say or imply:

- eager mode is "without automatic differentiation"
- "Need gradients (AD)" means switching to lazy/traced mode
- PyTorch mapping tables do not describe eager `loss.backward()`

That is stale once eager reverse-mode is treated as supported public surface.

### 3. Eager docs currently need a facade cleanup

Repository rules require user-facing docs to import from `tenferro`, not from
internal crates such as `tenferro_tensor` or `tenferro_einsum`. The current
eager guide still uses internal-crate imports. If eager docs are updated, the
public surface must support that presentation cleanly.

## Design Decision

Adopt explicit PyTorch-style accumulation semantics for eager reverse-mode.

### Public API

`EagerTensor<B>` gains:

```rust
impl<B: TensorBackend> EagerTensor<B> {
    pub fn tracks_grad(&self) -> bool;
    pub fn clear_grad(&self);
    pub fn grad(&self) -> Option<Arc<Tensor>>;
    pub fn backward(&self) -> Result<HashMap<GlobalValKey<StdTensorOp>, Arc<Tensor>>>;
}
```

`EagerContext<B>` exposes:

```rust
impl<B: TensorBackend> EagerContext<B> {
    pub fn with_backend(backend: B) -> Arc<Self>;
    pub fn clear_grads(&self);
}
```

`tenferro::eager_einsum` should expose both eager entry points needed by
user-facing docs:

```rust
pub use tenferro_einsum::eager_einsum;
pub fn eager_einsum_ad<B: TensorBackend>(...) -> Result<EagerTensor<B>>;
```

## Semantics

### `backward()` accumulates

`backward()` must no longer clear stored leaf gradients before the reverse
pass.

Instead:

1. run the reverse pass and produce the fresh `cotangents` map for this call
2. for each tracked live slot that appears in `cotangents`:
   - if the slot is empty, store the fresh cotangent
   - if the slot already contains a gradient, add the new cotangent into it
3. for tracked live slots that do **not** appear in `cotangents`, leave them
   unchanged

This matches the PyTorch rule that gradients persist until the user clears
them explicitly.

### Explicit reset

Users reset state explicitly:

- `tensor.clear_grad()` clears one tracked tensor's stored gradient
- `ctx.clear_grads()` clears all live tracked tensors registered in the shared
  eager context

Both operations are idempotent. Calling them on empty slots is a no-op.

### Tracked-state query

`tracks_grad()` exposes whether the eager tensor participates in reverse-mode
tracking. This is needed by downstream migration code and by documentation.

The name is intentionally not `requires_grad()` because the constructor family
already uses `requires_grad(...)` as an associated function.

### Return value of `backward()`

`backward()` still returns the fresh cotangent map produced by the current
reverse pass.

Important distinction:

- returned map = gradients from **this call**
- `grad()` = accumulated leaf gradient storage after merging this call

This preserves the current debugging value of the return result while making
slot semantics PyTorch-compatible.

## Implementation Notes

### Root fix: replace overwrite logic, not just the clear call

Removing `self.ctx.clear_grads()` is necessary but not sufficient.

The current `store_grads(...)` helper overwrites every registered slot with:

- `Some(cotangent)` when the key is present
- `None` when the key is absent

That still breaks accumulation and incorrectly clears untouched leaves.

The actual fix is to replace overwrite storage with accumulation-aware storage.

### Accumulation helper behavior

The eager context should own an internal helper shaped like:

```rust
fn accumulate_grads(
    &self,
    cotangents: &HashMap<GlobalValKey<StdTensorOp>, Arc<Tensor>>,
    backend: &mut B,
) -> Result<()>;
```

Behavior:

- retain only live weak slots
- skip keys missing from `cotangents`
- set empty slots directly to the incoming cotangent
- add incoming cotangents into existing slot values using backend-backed tensor
  addition

Using backend-backed tensor addition keeps accumulation generic over dtype and
backend rather than hard-coding host-side scalar cases.

### Detached and untracked tensors

No special-case redesign is needed:

- detached tensors still break the reverse graph
- untracked tensors still never get registered slots
- `clear_grad()` on an untracked tensor is a no-op

### Shared-context behavior

Accumulation is scoped to registered tracked tensors in the shared
`EagerContext`. This is the correct eager mental model:

- one context can own many leaves
- repeated backwards across computations in that context accumulate until
  explicitly cleared

This also makes the new public `EagerContext::clear_grads()` meaningful.

## Documentation Changes

The public story should become:

- eager mode supports immediate execution and scalar-loss reverse-mode with
  `backward()`
- traced mode remains the transform-oriented API for `grad`, `vjp`, `jvp`, and
  `hvp`
- eager gradients accumulate, just like PyTorch
- users clear gradients explicitly between optimization steps

The minimum docs set to update is:

- `README.md`
- `tenferro/README.md`
- `docs/index.md`
- `docs/guides/eager-operations.md`
- `docs/guides/autodiff.md`
- `docs/getting-started/pytorch-jax-mapping.md`
- rustdoc in `tenferro/src/eager.rs`
- rustdoc in `tenferro/src/eager_einsum.rs`

User-facing pages must import from `tenferro`, not internal crates.

## Testing Strategy

Add or update tests for all public semantics that change:

### Eager reverse-mode contract

In `tenferro/tests/eager_tensor.rs`:

- repeated `backward()` calls accumulate
- `clear_grad()` resets one leaf without touching others
- `EagerContext::clear_grads()` resets all live leaves in the shared context
- untouched leaves retain prior accumulated gradients across unrelated backward
  calls
- detached and untracked tensors do not receive gradients
- `tracks_grad()` reports the correct state

### Eager einsum path

In `tenferro/tests/eager_einsum_ad.rs`:

- repeated eager einsum backwards accumulate across calls
- context-level clear works for eager einsum leaves too

### Thread-safety guard

Add a small compile-time assertion that `EagerContext<CpuBackend>` and
`EagerTensor<CpuBackend>` are `Send + Sync`. The local branch already moved to
`Arc<Mutex<...>>`; the assertion prevents silent regressions.

### Docs/rustdoc

Doctests must demonstrate:

- `backward()` accumulation
- explicit `clear_grad()` / `clear_grads()`
- eager vs traced AD positioning

## Compatibility Policy

No compatibility shim is needed.

Do **not** preserve overwrite semantics behind alternate methods, flags, or
deprecated entry points. The repository is early-stage and the target behavior
is now explicit.

## Deferred Follow-Ups

These are useful but not required for this migration phase:

- a backend-borrow helper such as `EagerContext::with_backend_mut(...)` if
  downstream integration later proves it necessary
- eliminating host-side clone conversions in
  `tenferro-einsum/src/typed_eager.rs` and
  `tenferro-tensor/src/typed_linalg.rs`

They should be handled as separate follow-up work rather than folded into the
gradient-semantics change.

## Summary

The correct upstream change is not "document the current overwrite behavior."
It is to change eager reverse-mode to PyTorch-compatible accumulation semantics
and make reset explicit.

That requires:

- `backward()` accumulation
- public `clear_grad()` and `clear_grads()`
- a public `tracks_grad()` getter
- user-facing doc updates that describe eager AD honestly
- a small facade cleanup so eager docs can stay on `tenferro` imports
