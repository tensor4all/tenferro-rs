# Torch-Style Shape Packing API Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add torch-style `unsqueeze`, `squeeze`, `stack`, and `cat` APIs so rank-0 tensors can be packed back into dense tensors without extracting Rust/Python/C++ scalars, while preserving first-order AD semantics on the public `tenferro::Tensor` surface.

**Architecture:** Implement the substrate first in `tenferro-tensor`, where `unsqueeze` and `squeeze` are pure metadata views and `stack`/`cat` are explicit dense materializations. Then expose dense-only dynamic wrappers in `tenferro`, reusing the existing promotion and reverse-tape machinery so AD matches PyTorch: `unsqueeze` and `squeeze` are inverse view rules, `stack` backward is per-input `select`, and `cat` backward is per-input `narrow`.

**Tech Stack:** Rust workspace crates, `tenferro-tensor`, `tenferro`, `tidu` reverse tape rules, existing dynamic promotion helpers, existing dense shape-op tests, rustdoc examples, targeted `cargo test` verification.

---

## Scope And Assumptions

- Public API target: `tenferro::Tensor`
- Required substrate: `tenferro_tensor::Tensor<T>`
- Phase-1 layout scope: dense tensors only on the dynamic `tenferro::Tensor` surface
- Phase-1 device scope:
  - `unsqueeze` / `squeeze`: all memory spaces, because they are metadata-only
  - `stack` / `cat`: main-memory tensors only unless an existing transfer-safe path is already present
- Phase-1 structured scope:
  - `unsqueeze` / `squeeze` on `tenferro::Tensor`: reject non-dense structured layouts with a typed error
  - `stack` / `cat` on `tenferro::Tensor`: reject non-dense structured layouts with a typed error
- Dtype scope:
  - `tenferro_tensor::Tensor<T>` stays monomorphic by `T`
  - `tenferro::Tensor` uses the existing dynamic promotion join before `stack` / `cat`
- Torch parity rule:
  - `cat` rejects rank-0 inputs
  - `stack` accepts rank-0 inputs and creates a new axis
  - negative dimensions are supported for the new APIs

## Alternatives Considered

### Option 1: Frontend-only materialization in `tenferro`

Implement all four operations only on `tenferro::Tensor`, rebuilding dense payloads there.

- Pros:
  - smallest initial diff in the dynamic frontend
- Cons:
  - duplicates shape logic that belongs in `tenferro-tensor`
  - makes typed downstream users second-class
  - complicates future reuse from linalg/einsum helpers

### Option 2: Dense-first substrate in `tenferro-tensor`, then dynamic wrappers

Add the low-level typed ops first, then layer dense-only dynamic wrappers and AD rules on top.

- Pros:
  - clean crate layering
  - mirrors how existing view/data ops are split today
  - gives a reusable typed substrate for future APIs
  - keeps AD logic small and mechanical in the dynamic wrapper
- Cons:
  - touches both crates instead of one

### Option 3: Full dense + structured + GPU parity in one issue

Land dense, diagonal/structured, and GPU behavior together.

- Pros:
  - maximal one-shot parity story
- Cons:
  - too broad for the current partial GPU state
  - structured `cat` / `stack` semantics are not yet settled
  - much higher review and regression risk

**Recommendation:** Option 2. Land a dense-first typed substrate and dynamic wrapper now, and leave structured/GPU extensions as explicit follow-up work.

## API Contract

### `tenferro_tensor::Tensor<T>`

- Add `unsqueeze(dim: isize) -> Result<Tensor<T>>`
- Add `squeeze() -> Tensor<T>`
- Add `squeeze_dim(dim: isize) -> Result<Tensor<T>>`
- Add `stack(tensors: &[&Tensor<T>], dim: isize) -> Result<Tensor<T>>`
- Add `cat(tensors: &[&Tensor<T>], dim: isize) -> Result<Tensor<T>>`

Semantics:

- `unsqueeze` inserts a size-1 axis and returns a zero-copy view
- `squeeze` removes all size-1 axes and returns a zero-copy view
- `squeeze_dim` removes exactly one size-1 axis and errors if that axis is not size 1
- `stack` inserts a new axis and materializes a dense output
- `cat` joins existing axes and errors on rank-0 inputs

### `tenferro::Tensor`

- Add `unsqueeze(&self, dim: isize) -> Result<Self>`
- Add `squeeze(&self) -> Result<Self>`
- Add `squeeze_dim(&self, dim: isize) -> Result<Self>`
- Add `Tensor::stack(tensors: &[&Self], dim: isize) -> Result<Self>`
- Add `Tensor::cat(tensors: &[&Self], dim: isize) -> Result<Self>`

Semantics:

- dense-only in phase 1
- `stack` and `cat` apply dynamic dtype promotion before execution
- AD rules mirror PyTorch:
  - `unsqueeze` backward = `squeeze_dim`
  - `squeeze` backward = `unsqueeze` of the dropped size-1 axes
  - `stack` backward = `select` along the stacked axis
  - `cat` backward = `narrow` by per-input extents

## Acceptance Criteria

- rank-0 `tenferro::Tensor` values can be packed into rank-1 and higher-rank dense tensors via `Tensor::stack(...)`
- `tenferro::Tensor::try_scalar_value()` is no longer the only route from scalar tensor to array packing
- `tenferro_tensor::Tensor<T>` exposes view-level `unsqueeze` and `squeeze` APIs with negative-dim support
- `tenferro_tensor::Tensor<T>::cat(...)` rejects rank-0 inputs with a typed error that documents the torch-compatible limitation
- `tenferro::Tensor::stack(...)` and `Tensor::cat(...)` preserve forward tangents and register reverse pullbacks
- mixed-dtype dynamic stack/cat promote to the existing common result dtype
- new public APIs have rustdoc examples
- targeted tests cover rank-0 packing, higher-rank packing, negative dims, shape validation, dtype promotion, and AD

### Task 1: Add low-level typed tensor API tests

**Files:**
- Modify: `tenferro-tensor/tests/tensor_tests.rs`

**Step 1: Add failing tests for view semantics**

Cover:

```rust
#[test]
fn unsqueeze_scalar_to_vector_round_trips() { /* [] -> [1] -> [] */ }

#[test]
fn unsqueeze_supports_negative_dims() { /* [2, 3] -> [1, 2, 3], [2, 1, 3], [2, 3, 1] */ }

#[test]
fn squeeze_removes_all_size_one_axes() { /* [2, 1, 3, 1] -> [2, 3] */ }

#[test]
fn squeeze_dim_rejects_non_unit_axis() { /* [2, 3] squeeze_dim(1) fails */ }
```

**Step 2: Add failing tests for dense packing**

Cover:

```rust
#[test]
fn stack_rank0_scalars_builds_rank1_tensor() { /* [] x N -> [N] */ }

#[test]
fn stack_vectors_builds_matrix() { /* [M] x N -> [N, M] at dim 0 */ }

#[test]
fn cat_vectors_extends_existing_axis() { /* [M] + [K] -> [M + K] */ }

#[test]
fn cat_rejects_rank0_inputs() { /* torch-compatible failure */ }
```

Also cover:

- empty input list rejection
- `stack` shape mismatch rejection
- `cat` non-concatenated-axis mismatch rejection
- negative-dim wrapping
- same-memory-space validation

**Step 3: Run the targeted tests to confirm they fail**

Run:

```bash
cargo test -p tenferro-tensor tensor_tests -- --nocapture
```

Expected: the new shape-packing tests fail because the APIs do not exist yet.

**Step 4: Commit**

```bash
git add tenferro-tensor/tests/tensor_tests.rs
git commit -m "test: add torch-style tensor packing api regressions"
```

### Task 2: Implement low-level `unsqueeze` and `squeeze` view ops

**Files:**
- Modify: `tenferro-tensor/src/tensor/views.rs`

**Step 1: Add wrapped-dim helpers for the new signed dimension APIs**

Implement small local validation helpers so:

- `unsqueeze(dim)` accepts `[-rank-1, rank]`
- `squeeze_dim(dim)` accepts `[-rank, rank-1]`

Keep error text explicit about the valid interval.

**Step 2: Implement `unsqueeze` as a metadata-only view**

Requirements:

- insert `1` into dims at the wrapped axis
- insert a stride that preserves contiguity for contiguous inputs
- do not copy data
- preserve existing memory-space metadata

For column-major metadata, choose the inserted stride so contiguous tensors stay contiguous after insertion.

**Step 3: Implement `squeeze` and `squeeze_dim` as metadata-only views**

Requirements:

- `squeeze()` removes every axis with size 1
- `squeeze_dim(dim)` removes only the requested size-1 axis
- `squeeze_dim(dim)` errors if the target axis size is not 1
- rank-0 tensors remain rank-0 under `squeeze()`

**Step 4: Re-run the low-level tests**

Run:

```bash
cargo test -p tenferro-tensor tensor_tests -- --nocapture
```

Expected: the `unsqueeze` / `squeeze` tests pass while `stack` / `cat` tests still fail.

**Step 5: Commit**

```bash
git add tenferro-tensor/src/tensor/views.rs tenferro-tensor/tests/tensor_tests.rs
git commit -m "feat: add tensor unsqueeze and squeeze view ops"
```

### Task 3: Implement low-level `stack` and `cat` dense materialization

**Files:**
- Create: `tenferro-tensor/src/tensor/combine.rs`
- Modify: `tenferro-tensor/src/tensor/mod.rs`
- Modify: `tenferro-tensor/tests/tensor_tests.rs`

**Step 1: Add shared validation helpers**

Validate:

- non-empty input list
- same `dims()` for `stack`
- same rank and same non-concatenated extents for `cat`
- wrapped dimension in the torch-compatible range
- same logical memory space across inputs
- same preferred compute-device override across inputs, or explicitly clear it when preserving a mixed override would be ambiguous

`cat` must additionally reject any rank-0 input with a typed error that states that zero-dimensional tensors cannot be concatenated.

**Step 2: Implement `stack`**

Recommended implementation:

- compute result dims by inserting `tensors.len()` at the wrapped axis
- allocate one contiguous column-major output buffer
- copy each input into a `narrow`-like slot along the new axis

Do not implement `stack` by extracting scalars or by forcing a scalar-only special case.

**Step 3: Implement `cat`**

Recommended implementation:

- compute the concatenated extent on the chosen axis
- allocate one contiguous column-major output buffer
- copy each input into the correct offset window using an internal elementwise/strided copy helper

Phase 1 device behavior:

- main-memory inputs are supported
- non-main-memory inputs return a typed limitation instead of silently transferring to CPU

**Step 4: Re-run the low-level tests**

Run:

```bash
cargo test -p tenferro-tensor tensor_tests -- --nocapture
```

Expected: all new low-level tests pass.

**Step 5: Commit**

```bash
git add tenferro-tensor/src/tensor/combine.rs tenferro-tensor/src/tensor/mod.rs tenferro-tensor/tests/tensor_tests.rs
git commit -m "feat: add tensor stack and cat dense packing ops"
```

### Task 4: Add dynamic frontend tests for torch-style shape packing and AD

**Files:**
- Create: `tenferro/tests/dyn_tensor_combine_ops_tests.rs`
- Modify: `tenferro/tests/dyn_tensor_shape_ops_tests.rs`

**Step 1: Add failing tests for dense shape views on `tenferro::Tensor`**

Cover:

```rust
#[test]
fn dyn_tensor_unsqueeze_preserves_forward_mode() { /* primal + tangent both gain a unit axis */ }

#[test]
fn dyn_tensor_squeeze_pullback_reinserts_unit_axis() { /* reverse cotangent shape is restored */ }
```

**Step 2: Add failing tests for `stack` / `cat`**

Cover:

```rust
#[test]
fn dyn_tensor_stack_rank0_scalars_builds_vector_and_preserves_grad() { /* [] x N -> [N] */ }

#[test]
fn dyn_tensor_cat_splits_pullback_by_input_extent() { /* backward = narrow slices */ }

#[test]
fn dyn_tensor_stack_promotes_mixed_real_complex_inputs() { /* promotion join applies */ }
```

Also cover:

- `cat` rejects rank-0 inputs
- `stack` rejects mismatched shapes
- dense-only rejection for structured tensors
- negative-dim support
- no tangent inputs + one tangent input behavior for forward mode

**Step 3: Run the targeted dynamic tests to confirm they fail**

Run:

```bash
cargo test -p tenferro --test dyn_tensor_shape_ops_tests -- --nocapture
cargo test -p tenferro --test dyn_tensor_combine_ops_tests -- --nocapture
```

Expected: failures for missing APIs and missing AD rules.

**Step 4: Commit**

```bash
git add tenferro/tests/dyn_tensor_shape_ops_tests.rs tenferro/tests/dyn_tensor_combine_ops_tests.rs
git commit -m "test: add dynamic tensor shape packing api coverage"
```

### Task 5: Implement dynamic `unsqueeze` and `squeeze` wrappers with AD

**Files:**
- Modify: `tenferro/src/core/dynamic/dyn_ad_tensor/layout.rs`
- Modify: `tenferro/src/core/dynamic/dyn_ad_tensor/shape.rs`

**Step 1: Add typed AD helpers for `unsqueeze` and `squeeze`**

Match existing layout-op structure:

- primal path: call the typed `tenferro_tensor::Tensor<T>` API
- forward path: apply the same transform to both primal and tangent
- reverse path: register a tape rule on the output node

**Step 2: Register reverse rules**

Rules:

- `unsqueeze(dim)` backward = `squeeze_dim(dim)`
- `squeeze_dim(dim)` backward = `unsqueeze(dim)`
- `squeeze()` backward = restore the dropped unit axes in the original positions

Use the original logical dims captured at rule-registration time so the backward shape restoration is unambiguous.

**Step 3: Add the public dynamic methods**

Expose:

- `Tensor::unsqueeze(&self, dim: isize) -> Result<Self>`
- `Tensor::squeeze(&self) -> Result<Self>`
- `Tensor::squeeze_dim(&self, dim: isize) -> Result<Self>`

Keep phase-1 behavior dense-only using the existing layout guard pattern.

**Step 4: Re-run the shape-op dynamic tests**

Run:

```bash
cargo test -p tenferro --test dyn_tensor_shape_ops_tests -- --nocapture
```

Expected: new `unsqueeze` / `squeeze` tests pass.

**Step 5: Commit**

```bash
git add tenferro/src/core/dynamic/dyn_ad_tensor/layout.rs tenferro/src/core/dynamic/dyn_ad_tensor/shape.rs tenferro/tests/dyn_tensor_shape_ops_tests.rs
git commit -m "feat: add dynamic tensor unsqueeze and squeeze with ad"
```

### Task 6: Implement dynamic `stack` and `cat` wrappers with AD

**Files:**
- Create: `tenferro/src/core/dynamic/dyn_ad_tensor/combine.rs`
- Modify: `tenferro/src/core/dynamic/dyn_ad_tensor/mod.rs`
- Modify: `tenferro/tests/dyn_tensor_combine_ops_tests.rs`

**Step 1: Add typed dispatch helpers**

Follow the same pattern as eager `einsum`:

- validate non-empty input list
- run `promote_many_to_common(...)`
- borrow the promoted `AdTensor<T>` values at the typed execution boundary

**Step 2: Implement `stack`**

Behavior:

- primal path: stack typed payloads
- forward path: stack tangents if any input carries one; otherwise leave tangent absent
- reverse path: register one gradient per input by `select(dim, i)` on the cotangent payload

**Step 3: Implement `cat`**

Behavior:

- primal path: concatenate typed payloads
- forward path: concatenate tangents, using zero tangents for primal-only operands whenever any input has a tangent
- reverse path: split the cotangent payload with per-input `narrow` slices along the concatenated axis

**Step 4: Public dynamic API**

Expose:

- `Tensor::stack(tensors: &[&Self], dim: isize) -> Result<Self>`
- `Tensor::cat(tensors: &[&Self], dim: isize) -> Result<Self>`

Use dense-only validation in phase 1.

**Step 5: Re-run the combine-op tests**

Run:

```bash
cargo test -p tenferro --test dyn_tensor_combine_ops_tests -- --nocapture
```

Expected: stack/cat primal, promotion, forward-mode, and reverse-mode tests pass.

**Step 6: Commit**

```bash
git add tenferro/src/core/dynamic/dyn_ad_tensor/combine.rs tenferro/src/core/dynamic/dyn_ad_tensor/mod.rs tenferro/tests/dyn_tensor_combine_ops_tests.rs
git commit -m "feat: add dynamic tensor stack and cat with ad"
```

### Task 7: Document the new API and verify targeted coverage

**Files:**
- Modify: `tenferro-tensor/src/lib.rs`
- Modify: `tenferro-tensor/src/tensor/views.rs`
- Modify: `tenferro-tensor/src/tensor/combine.rs`
- Modify: `tenferro/src/lib.rs`
- Modify: `tenferro/src/core/dynamic/dyn_ad_tensor/shape.rs`
- Modify: `tenferro/src/core/dynamic/dyn_ad_tensor/combine.rs`
- Modify: `docs/design/reference/libtorch.md`

**Step 1: Add rustdoc examples for every new public API**

Examples must show:

- rank-0 scalar creation
- `unsqueeze` to rank-1
- `Tensor::stack` packing scalar tensors into a vector
- `Tensor::cat` on rank-1 inputs

Use `ignore` only when the example truly requires runtime setup that docs cannot run.

**Step 2: Update the libtorch mapping note**

Document explicitly that:

- `cat` rejects zero-dimensional tensors
- `stack` is the packing API for scalar tensors
- tenferro phase 1 matches dense semantics first

**Step 3: Run targeted verification**

Run:

```bash
cargo fmt --all
cargo test -p tenferro-tensor tensor_tests -- --nocapture
cargo test -p tenferro --test dyn_tensor_shape_ops_tests -- --nocapture
cargo test -p tenferro --test dyn_tensor_combine_ops_tests -- --nocapture
```

If doctests were touched materially, also run:

```bash
cargo test --doc -p tenferro-tensor -p tenferro --release
```

**Step 4: Commit**

```bash
git add tenferro-tensor/src/lib.rs tenferro-tensor/src/tensor/views.rs tenferro-tensor/src/tensor/combine.rs tenferro/src/lib.rs tenferro/src/core/dynamic/dyn_ad_tensor/shape.rs tenferro/src/core/dynamic/dyn_ad_tensor/combine.rs docs/design/reference/libtorch.md tenferro/tests/dyn_tensor_shape_ops_tests.rs tenferro/tests/dyn_tensor_combine_ops_tests.rs tenferro-tensor/tests/tensor_tests.rs
git commit -m "docs: document torch-style tensor packing apis"
```

## Follow-Up Issues Explicitly Out Of Scope

- structured-layout `stack` / `cat` semantics for diagonal tensors
- GPU-native `stack` / `cat` materialization
- top-level free-function aliases beyond the associated-function surface above
- `unbind`, `split`, and `tensor_split` API parity
- broad negative-dim normalization cleanup for older `usize`-typed APIs such as `select` and `narrow`
