# HVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add public `tenferro::hvp` API for Hessian-vector products via forward-over-reverse AD.

**Architecture:** Refactor the tape registry to accept full `ReverseRule` objects instead of VJP-only closures, update `EinsumReverseRule` to the new chainrules-core API with deferred tangent injection, and add a public `hvp()` function that delegates to tidu's two-phase HVP engine.

**Tech Stack:** Rust, chainrules-core (rev 6cc4677), tidu (rev ea504cd), tenferro workspace

**Spec:** `docs/superpowers/specs/2026-03-28-hvp-design.md`

---

## Task 1: Bump upstream dependencies

**Files:**
- Modify: `Cargo.toml` (workspace root, lines 37-39)

- [ ] **Step 1: Update dependency revisions**

In the workspace `Cargo.toml`, update:

```toml
chainrules-core = { git = "https://github.com/tensor4all/chainrules-rs", rev = "6cc46775b33653f91df96ca1571ce9905a6224f8" }
chainrules = { git = "https://github.com/tensor4all/chainrules-rs", rev = "6cc46775b33653f91df96ca1571ce9905a6224f8" }
tidu = { git = "https://github.com/tensor4all/tidu-rs", rev = "ea504cd" }
```

- [ ] **Step 2: Verify build compiles**

Run: `cargo check --workspace 2>&1 | head -50`

This will fail with compilation errors because the `ReverseRule` trait signatures have changed (new `forward_tangents` and 3-arg `pullback_with_tangents`). That's expected — the next tasks fix these.

- [ ] **Step 3: Commit**

```bash
git add Cargo.toml Cargo.lock
git commit -m "chore: bump chainrules-core and tidu for HVP support"
```

---

## Task 2: Refactor tape registry to accept ReverseRule objects

**Files:**
- Modify: `tenferro/src/tape/registry.rs`

The registry must change from accepting `PullbackRule<T>` closures to accepting `Box<dyn ReverseRule<tenferro_tensor::Tensor<T>>>`. Add `ClosureRule<T>` as a backwards-compatible wrapper.

- [ ] **Step 1: Replace PullbackRule type and TensorRuleAdapter**

Rewrite `tenferro/src/tape/registry.rs`. Key changes:

1. Remove `PullbackRule<T>` and `MixedPullbackRule<TOut, TIn>` type aliases
2. Add `ClosureRule<T>` struct that wraps the old closure signature + `input_node_ids`
3. Implement `ReverseRule<tenferro_tensor::Tensor<T>>` for `ClosureRule<T>`
4. Change `TensorRuleAdapter<T>` to wrap `Box<dyn ReverseRule<tenferro_tensor::Tensor<T>>>`
5. Forward all 4 `ReverseRule<DynTensor>` methods through the adapter with type conversion
6. Change `register_rule` signature to accept `Box<dyn ReverseRule<tenferro_tensor::Tensor<T>>>`
7. Add `register_closure_rule<T>` convenience function that wraps a closure in `ClosureRule` and calls `register_rule`

For `ClosureRule<T>::pullback`: wrap incoming `&tenferro_tensor::Tensor<T>` in `StructuredTensor::from_dense()`, call the closure, extract payloads via `into_payload()`.

For `TensorRuleAdapter<T>::forward_tangents` and `pullback_with_tangents`: use `T::structured_ref(dyn_tensor).map(|s| s.payload())` to extract `&tenferro_tensor::Tensor<T>` from `&DynTensor`, and `T::into_dyn(StructuredTensor::from_dense(tensor))` to wrap back.

For `MixedTensorRuleAdapter`: same pattern but with `TOut`/`TIn` type parameters. HVP methods use trait defaults (`HvpNotSupported`).

- [ ] **Step 2: Verify registry compiles in isolation**

Run: `cargo check -p tenferro 2>&1 | head -80`

Expect errors at call sites — the next task fixes those.

---

## Task 3: Migrate all register_rule call sites to ClosureRule

**Files:**
- Modify: `tenferro/src/ops/einsum/ad.rs` (lines 194, 247)
- Modify: `tenferro/src/ops/linalg/ad/svd_qr.rs` (lines 152, 188, 224, 353, 386)
- Modify: `tenferro/src/ops/linalg/ad/slogdet.rs` (lines 83, 98)
- Modify: `tenferro/src/ops/linalg/ad/lu_lstsq.rs` (lines 97, 132, 270, 315)
- Modify: `tenferro/src/ops/linalg/ad/spectral.rs` (lines 87, 120)
- Modify: `tenferro/src/ops/linalg/ad/common.rs` (lines 82, 178 — macro sites)
- Modify: `tenferro/src/ops/scalar/ad/common.rs` (lines 246, 347)
- Modify: `tenferro/src/ops/reduction/ad.rs` (line 66)
- Modify: `tenferro/src/core/dynamic/dyn_ad_tensor/layout.rs` (lines 136, 204, 252, 409)
- Modify: `tenferro/src/core/dynamic/dyn_ad_tensor/layout/structured.rs` (lines 120, 181)
- Modify: `tenferro/src/core/dynamic/dyn_ad_tensor/scalar_ops.rs` (line 183)
- Modify: `tenferro/src/core/dynamic/dyn_ad_tensor/merge.rs` (lines 60, 133, 288, 309)
- Modify: `tenferro/src/core/dynamic/dyn_ad_tensor/complex.rs` (line 49)

At each call site, change:
```rust
// OLD
tape::register_rule::<T>(&tape, node, Box::new(move |cotangent| { ... }));

// NEW — use convenience function
tape::register_closure_rule::<T>(&tape, node, input_node_ids, Box::new(move |cotangent| { ... }));
```

Where `input_node_ids` is extracted from the context. Most sites already have this data:
- Sites with `collect_reverse_input_specs`: `specs.iter().filter_map(|s| s.as_ref().map(|s| s.node)).collect()`
- Sites with `collect_reverse_input_nodes`: `nodes.iter().filter_map(|n| *n).collect()`
- Sites with a single `spec.node`: `vec![spec.node]`

For `register_mixed_rule` sites: create a similar `register_mixed_closure_rule` convenience function.

- [ ] **Step 1: Migrate all call sites**

Do a systematic find-and-replace across all files listed above.

- [ ] **Step 2: Build and fix compilation errors**

Run: `cargo check --workspace 2>&1 | head -100`

Fix any remaining type errors. The most common issue will be extracting `input_node_ids` at each site.

- [ ] **Step 3: Run existing tests**

Run: `cargo test --workspace --release 2>&1 | tail -30`

All existing tests must still pass (this is a pure refactor).

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor: migrate tape registry to accept ReverseRule objects

Replace PullbackRule closure-only adapter with full ReverseRule<Tensor<T>>
support. Add ClosureRule<T> wrapper for backwards-compatible migration of
~30+ existing VJP-only call sites."
```

---

## Task 4: Update EinsumReverseRule to new chainrules-core API

**Files:**
- Modify: `tenferro-einsum/src/ad/reverse_rule.rs`
- Modify: `tenferro-einsum/src/ad/tracked.rs`

- [ ] **Step 1: Update EinsumReverseRule struct**

In `tenferro-einsum/src/ad/reverse_rule.rs`:

1. Remove the `input_tangents: Vec<Option<Tensor<Alg::Scalar>>>` field
2. Update `pullback` method: remove forward-mode tangent propagation code (the `if let Ok(grad_tangent)` block that calls `einsum_frule_impl` with stored tangents)
3. Add `forward_tangents` method: compute output tangent from input tangents via the closure, using `einsum_frule_impl`
4. Update `pullback_with_tangents` to read input tangents from the closure parameter instead of `self.input_tangents`

For `forward_tangents`:
```rust
fn forward_tangents<'t>(
    &self,
    input_tangents: &dyn Fn(NodeId) -> Option<&'t Tensor<Alg::Scalar>>,
) -> AdResult<Option<Tensor<Alg::Scalar>>>
where Tensor<Alg::Scalar>: 't
{
    let tangents: Vec<Option<&Tensor<Alg::Scalar>>> = self.input_node_ids
        .iter()
        .map(|maybe_id| maybe_id.and_then(|id| input_tangents(id)))
        .collect();
    if tangents.iter().all(|t| t.is_none()) {
        return Ok(None);
    }
    let primals: Vec<&Tensor<Alg::Scalar>> = self.primals.iter().collect();
    let mut ctx = self.ctx.lock().map_err(|_| ...)?;
    Ok(Some(einsum_frule_impl::<Alg, Backend>(&mut *ctx, &self.subscripts, None, &primals, &tangents)
        .map_err(|e| ...)?))
}
```

For `pullback_with_tangents`: similar pattern but computing both grad and grad_tangent.

- [ ] **Step 2: Update tracked_einsum**

In `tenferro-einsum/src/ad/tracked.rs`:

Remove the `input_tangents` field from the `EinsumReverseRule` constructor (line 110):
```rust
// OLD
input_tangents: operands.iter().map(|op| op.tangent().cloned()).collect(),

// NEW — remove this line entirely
```

- [ ] **Step 3: Build and test**

Run: `cargo test --workspace --release 2>&1 | tail -30`

- [ ] **Step 4: Commit**

```bash
git add tenferro-einsum/
git commit -m "feat: update EinsumReverseRule to deferred tangent API

Remove stored input_tangents field. Implement forward_tangents using
einsum_frule_impl. Update pullback_with_tangents to read tangents from
the closure parameter."
```

---

## Task 5: Register einsum dense path with HVP-capable rule

**Files:**
- Modify: `tenferro/src/ops/einsum/ad.rs`

- [ ] **Step 1: Create DenseEinsumRule and register directly**

In `tenferro/src/ops/einsum/ad.rs`, for the dense einsum path (around line 247):

1. Create a `DenseEinsumRule<T>` struct that stores subscripts, primals, reverse_specs, and implements `ReverseRule<tenferro_tensor::Tensor<T>>`
2. Its `pullback` delegates to `dense_einsum_pullback_in_backend`
3. Its `forward_tangents` computes output tangent using `tf_einsum::einsum_frule`
4. Its `pullback_with_tangents` computes both grad and grad_tangent using `tf_einsum::einsum_rrule` and `tf_einsum::einsum_frule`
5. Register this rule via `tape::register_rule` instead of the closure

For the structured path (around line 194): wrap in `register_closure_rule` (no HVP support yet).

- [ ] **Step 2: Build and test**

Run: `cargo test --workspace --release 2>&1 | tail -30`

- [ ] **Step 3: Commit**

```bash
git add tenferro/src/ops/einsum/
git commit -m "feat: add DenseEinsumRule with HVP support for Level B dense path"
```

---

## Task 6: Add public hvp() API

**Files:**
- Modify: `tenferro/src/autograd_api.rs`
- Modify: `tenferro/src/lib.rs`

- [ ] **Step 1: Add HvpOptions and HvpResult types**

In `tenferro/src/autograd_api.rs`, add:

```rust
/// Options for HVP computation.
#[derive(Debug, Clone, Default)]
pub struct HvpOptions {
    /// If true, do not free the computation graph after HVP.
    pub retain_graph: bool,
}

/// Result of a Hessian-vector product computation.
#[derive(Debug)]
pub struct HvpResult {
    /// Gradients of the output with respect to each input.
    pub gradients: Vec<Option<Tensor>>,
    /// Hessian-vector products for each input.
    pub hvps: Vec<Option<Tensor>>,
}
```

- [ ] **Step 2: Implement hvp() function**

Add the `hvp` function following the 10 implementation steps from the spec. Key patterns to reuse from existing code:

- `reverse_tape(output)` — same as in `grad()`
- `reverse_handle(input)` — from `pullback.rs`
- `dyn_primal_from_snapshot(dyn_tensor)` — from `pullback.rs`
- Match over `output` variants (F32/F64/C32/C64) to extract `TrackedValue<DynTensor>`

The function must:
1. Validate scalar output, matching v/inputs lengths and shapes
2. Build `HashMap<NodeId, DynTensor>` from inputs and v
3. Call `tape.hvp(&tracked, &leaf_tangents)`
4. Project results to requested inputs
5. Handle `retain_graph`

- [ ] **Step 3: Export from lib.rs**

In `tenferro/src/lib.rs`, add to exports:
```rust
pub use autograd_api::{backward, grad, hvp, BackwardOptions, GradOptions, HvpOptions, HvpResult};
```

- [ ] **Step 4: Build**

Run: `cargo check -p tenferro`

- [ ] **Step 5: Commit**

```bash
git add tenferro/src/autograd_api.rs tenferro/src/lib.rs
git commit -m "feat: add public tenferro::hvp API for Hessian-vector products"
```

---

## Task 7: Add integration tests

**Files:**
- Create: `tenferro/tests/hvp_tests.rs`

- [ ] **Step 1: Write all integration tests**

Create `tenferro/tests/hvp_tests.rs` with the following tests:

1. **`test_hvp_quadratic_form`**: `einsum("i,i->", &[&x, &x])` with x=[1,2,3], v=[1,1,1]. Assert grad=[2,4,6], hvp=[2,2,2].
2. **`test_hvp_bilinear_two_inputs`**: `einsum("i,i->", &[&x, &y])`. Assert hvp_x = vy, hvp_y = vx.
3. **`test_hvp_non_scalar_output_error`**: `einsum("i,i->i", ...)` should return `NonScalarLoss`.
4. **`test_hvp_shape_mismatch_error`**: v with wrong shape should return `InvalidAdTensor`.
5. **`test_hvp_not_supported_propagation`**: Graph with a non-HVP operation (e.g., use `Tensor::sum()` which uses `ClosureRule`) should return `HvpNotSupported`.

Each test must:
- Set up runtime: `let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));`
- Create tensors with `set_requires_grad(true)`
- Compute output via einsum
- Call `hvp()`
- Assert results

- [ ] **Step 2: Run tests**

Run: `cargo test --workspace --release -p tenferro --test hvp_tests -- --nocapture 2>&1`

- [ ] **Step 3: Commit**

```bash
git add tenferro/tests/hvp_tests.rs
git commit -m "test: add HVP integration tests for einsum-only computations"
```

---

## Task 8: Final verification

- [ ] **Step 1: Run full test suite**

```bash
cargo fmt --all --check
cargo test --workspace --release
cargo doc --workspace --no-deps
```

- [ ] **Step 2: Fix any failures**

- [ ] **Step 3: Final commit if needed**
