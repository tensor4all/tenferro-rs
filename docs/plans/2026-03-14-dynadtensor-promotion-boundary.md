# DynAdTensor Promotion Boundary Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add canonical mixed promotion and primal snapshot boundaries to `DynAdTensor` so issues `#494`, `#495`, and `#496` are resolved through one coherent API.

**Architecture:** Keep `DynAdTensor` as the canonical runtime payload, factor promotion logic into a dedicated internal module, route `scale/axpby/div_scalar` through that single promotion engine, and expose a separate primal-only snapshot enum for storage/FFI boundaries.

**Tech Stack:** Rust workspace crates, `tenferro-dyadtensor`, `tenferro-tensor`, `num-complex`, rustdoc examples, workspace verification commands.

---

### Task 1: Add failing promotion boundary tests

**Files:**
- Modify: `extension/tenferro-dyadtensor/src/core/dynamic/tests/mod.rs`
- Modify: `extension/tenferro-dyadtensor/tests/mixed_complex_real_scalar_tests.rs`
- Modify: `extension/tenferro-dyadtensor/tests/mixed_primitives_forward_tests.rs`
- Modify: `extension/tenferro-dyadtensor/tests/mixed_primitives_reverse_tests.rs`

**Step 1: Write failing tests for `promote_to`**

Add tests for:
- identity promotion
- `F64 -> C64`
- unsupported `F64 -> F32`
- forward/reverse metadata preservation

**Step 2: Run focused tests to verify they fail**

Run:

```bash
env CARGO_TARGET_DIR=/tmp/tenferro-dynadtensor-promotion-boundary-target \
  cargo test -p tenferro-dyadtensor promote_to -- --nocapture
```

Expected: compile or runtime failure because `promote_to` does not exist yet.

**Step 3: Add failing mixed-op behavior tests**

Add tests for:
- `scale` promoting `F64 tensor × C64 rank-0 tensor -> C64 tensor`
- `axpby` promoting real and complex operands of matching precision
- `div_scalar` using the same promotion join

**Step 4: Run focused tests to verify they fail for the new contract**

Run:

```bash
env CARGO_TARGET_DIR=/tmp/tenferro-dynadtensor-promotion-boundary-target \
  cargo test -p tenferro-dyadtensor mixed_complex_real_scalar -- --nocapture
```

Expected: red on at least one new contract test before implementation.

### Task 2: Add failing primal snapshot tests

**Files:**
- Modify: `extension/tenferro-dyadtensor/tests/structured_layout_validation_tests.rs`
- Modify: `extension/tenferro-dyadtensor/tests/structured_tensor_root_tests.rs`
- Create or modify: `extension/tenferro-dyadtensor/tests/support/mod.rs`

**Step 1: Write failing tests for `primal_snapshot()`**

Cover:
- dense snapshot returns typed structured payload
- nontrivial structured tensor preserves `axis_classes`
- snapshot intentionally excludes AD metadata

**Step 2: Run focused tests to verify they fail**

Run:

```bash
env CARGO_TARGET_DIR=/tmp/tenferro-dynadtensor-promotion-boundary-target \
  cargo test -p tenferro-dyadtensor primal_snapshot -- --nocapture
```

Expected: failure because the public snapshot API does not exist yet.

### Task 3: Implement canonical promotion module

**Files:**
- Create: `extension/tenferro-dyadtensor/src/core/dynamic/dyn_ad_tensor/promotion.rs`
- Modify: `extension/tenferro-dyadtensor/src/core/dynamic/dyn_ad_tensor/mod.rs`
- Modify: `extension/tenferro-dyadtensor/src/core/dynamic/dyn_ad_tensor/scalar_ops.rs`

**Step 1: Introduce internal promotion helpers**

Implement:
- dtype join logic
- same-precision real-to-complex lifting
- public `DynAdTensor::promote_to(...)`

**Step 2: Keep promotion code out of `scalar_ops.rs`**

Move promotion-specific helpers from `scalar_ops.rs` into `promotion.rs`.

**Step 3: Run focused tests**

Run:

```bash
env CARGO_TARGET_DIR=/tmp/tenferro-dynadtensor-promotion-boundary-target \
  cargo test -p tenferro-dyadtensor promote_to -- --nocapture
```

Expected: green for promotion tests.

### Task 4: Rewire mixed scalar ops to use the shared promotion engine

**Files:**
- Modify: `extension/tenferro-dyadtensor/src/core/dynamic/dyn_ad_tensor/scalar_ops.rs`

**Step 1: Route `scale`, `axpby`, and `div_scalar` through the same join policy**

Ensure:
- no op-specific promotion tables remain
- mixed success/error behavior is derived only from the shared join logic

**Step 2: Run focused mixed-op tests**

Run:

```bash
env CARGO_TARGET_DIR=/tmp/tenferro-dynadtensor-promotion-boundary-target \
  cargo test -p tenferro-dyadtensor mixed_complex_real_scalar -- --nocapture
env CARGO_TARGET_DIR=/tmp/tenferro-dynadtensor-promotion-boundary-target \
  cargo test -p tenferro-dyadtensor mixed_primitives_forward -- --nocapture
env CARGO_TARGET_DIR=/tmp/tenferro-dynadtensor-promotion-boundary-target \
  cargo test -p tenferro-dyadtensor mixed_primitives_reverse -- --nocapture
```

Expected: green for the new mixed-promotion contract.

### Task 5: Implement the primal snapshot boundary

**Files:**
- Create: `extension/tenferro-dyadtensor/src/core/dynamic/dyn_ad_tensor/snapshot.rs`
- Modify: `extension/tenferro-dyadtensor/src/core/dynamic/dyn_ad_tensor/mod.rs`

**Step 1: Add `DynStructuredPrimal`**

Implement:
- `F32(StructuredTensor<f32>)`
- `F64(StructuredTensor<f64>)`
- `C32(StructuredTensor<Complex32>)`
- `C64(StructuredTensor<Complex64>)`

**Step 2: Add `DynAdTensor::primal_snapshot()`**

The method should:
- clone only the primal structured payload
- explicitly discard AD metadata

**Step 3: Run focused snapshot tests**

Run:

```bash
env CARGO_TARGET_DIR=/tmp/tenferro-dynadtensor-promotion-boundary-target \
  cargo test -p tenferro-dyadtensor primal_snapshot -- --nocapture
```

Expected: green for snapshot tests.

### Task 6: Update docs and rustdoc

**Files:**
- Modify: `extension/tenferro-dyadtensor/src/core/dynamic/dyn_ad_tensor/mod.rs`
- Modify: `extension/tenferro-dyadtensor/src/lib.rs`
- Modify: `docs/api_index.md`
- Modify: `docs/design/autodiff.md`
- Modify: `docs/design/supported-ops.md`

**Step 1: Add rustdoc examples**

Document:
- `promote_to(...)`
- `primal_snapshot()`
- rank-0 tensor scalar semantics

**Step 2: Update design docs**

Describe:
- canonical promotion policy
- primal snapshot boundary for storage/FFI

### Task 7: Run full verification

**Files:**
- No code changes expected unless verification reveals issues

**Step 1: Run formatting**

```bash
cargo fmt --all --check
```

**Step 2: Run workspace tests**

```bash
env CARGO_TARGET_DIR=/tmp/tenferro-dynadtensor-promotion-boundary-release-target \
  cargo test --workspace --release
```

**Step 3: Run coverage**

```bash
env CARGO_TARGET_DIR=/tmp/tenferro-dynadtensor-promotion-boundary-cov-target \
  cargo llvm-cov --workspace --release --json --output-path coverage.json
python3 scripts/check-coverage.py coverage.json
```

**Step 4: Run docs**

```bash
env CARGO_TARGET_DIR=/tmp/tenferro-dynadtensor-promotion-boundary-doc-target \
  cargo doc --workspace --no-deps
python3 scripts/check-docs-site.py --doc-root /tmp/tenferro-dynadtensor-promotion-boundary-doc-target/doc
```

### Task 8: Commit and open PR

**Files:**
- All files touched above

**Step 1: Review for residual ad hoc promotion logic**

Check:
- no duplicated join tables remain
- no downstream-only helper is needed to promote `DynAdTensor`
- snapshot boundary is public and documented

**Step 2: Commit**

```bash
git add .
git commit -m "feat: add canonical dynadtensor promotion boundary"
```

**Step 3: Create PR**

```bash
gh pr create --fill
gh pr merge --auto --squash --delete-branch
```
