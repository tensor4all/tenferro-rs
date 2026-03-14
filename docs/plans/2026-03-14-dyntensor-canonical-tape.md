# DynTensor Canonical Tape Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `tenferro-dyadtensor` use `Tape<DynTensor>` as its canonical AD graph model while keeping rank-0 tensor scalar semantics, preserving `Diag`, and leaving linalg AD dense-only.

**Architecture:** `DynTensor` becomes the structured-aware canonical dynamic primal type. `DynAdTensor` becomes the canonical dynamic AD wrapper over `TrackedValue<DynTensor>` / `DualValue<DynTensor>`. Structured AD remains allowed for einsum/reduction/layout-preserving linear ops, while linalg AD rejects non-dense structured inputs.

**Tech Stack:** Rust workspace, `chainrules`, `tenferro-dyadtensor`, `tenferro-linalg`, `tensor-ad-oracles`, rustdoc, workspace CI/coverage/docs gates.

---

### Task 1: Reshape `DynTensor` Into The Canonical Structured-Aware Dynamic Primal Type

**Files:**
- Modify: `extension/tenferro-dyadtensor/src/core/dynamic/dyn_tensor.rs`
- Modify: `extension/tenferro-dyadtensor/src/core/dynamic/mod.rs`
- Modify: `extension/tenferro-dyadtensor/src/core/mod.rs`
- Test: `extension/tenferro-dyadtensor/src/core/dynamic/tests/mod.rs`

**Step 1: Write the failing tests**

Add tests that require:
- `DynTensor` to preserve `logical_dims()` and `axis_classes()`
- dense `Tensor<T>` inputs to round-trip through `DynTensor`
- `Diag` structured inputs to round-trip through `DynTensor`
- `primal_snapshot()` callers to recover structured payloads through `DynTensor`

**Step 2: Run the focused test**

Run:

```bash
env CARGO_TARGET_DIR=/tmp/tenferro-dyntensor-canonical-tape-target cargo test -p tenferro-dyadtensor core::dynamic::tests:: -- --nocapture
```

Expected: FAIL because `DynTensor` still stores `Tensor<T>`.

**Step 3: Implement the minimal code**

Change `DynTensor` variants to wrap `StructuredTensor<T>`. Add accessors for:
- `structured_ref()`
- `logical_dims()`
- `axis_classes()`
- dense/diag helpers as needed

Keep conversions from `Tensor<T>` by wrapping with `StructuredTensor::from(...)`.

**Step 4: Run the focused test again**

Run the same command and verify it passes.

**Step 5: Commit**

```bash
git add extension/tenferro-dyadtensor/src/core/dynamic/dyn_tensor.rs extension/tenferro-dyadtensor/src/core/dynamic/mod.rs extension/tenferro-dyadtensor/src/core/mod.rs extension/tenferro-dyadtensor/src/core/dynamic/tests/mod.rs
git commit -m "refactor: make dytensor canonical dynamic primal structured-aware"
```

### Task 2: Make `DynTensor` Implement `Differentiable` And Introduce `Tape<DynTensor>` Tests

**Files:**
- Create: `extension/tenferro-dyadtensor/src/core/dynamic/autodiff.rs`
- Modify: `extension/tenferro-dyadtensor/src/core/dynamic/mod.rs`
- Test: `extension/tenferro-dyadtensor/src/core/dynamic/tests/mod.rs`

**Step 1: Write the failing tests**

Add tests that require:
- `Tape::<DynTensor>::new()` to compile and work
- rank-0 `DynTensor` leaves to produce gradients
- `Diag` `DynTensor` leaves to preserve structured tangents/cotangents

**Step 2: Run the focused test**

Run:

```bash
env CARGO_TARGET_DIR=/tmp/tenferro-dyntensor-canonical-tape-target cargo test -p tenferro-dyadtensor core::dynamic::tests::rank0 -- --nocapture
```

Expected: FAIL because `DynTensor` is not yet the differentiable graph payload.

**Step 3: Implement the minimal code**

Implement `Differentiable` for `DynTensor` by delegating tangent/cotangent behavior to its structured payload variants.

**Step 4: Run the focused test again**

Run the same command and verify it passes.

**Step 5: Commit**

```bash
git add extension/tenferro-dyadtensor/src/core/dynamic/autodiff.rs extension/tenferro-dyadtensor/src/core/dynamic/mod.rs extension/tenferro-dyadtensor/src/core/dynamic/tests/mod.rs
git commit -m "refactor: make dytensor differentiable tape payload"
```

### Task 3: Move `AdTensor<T>` Off Typed Tape Ownership

**Files:**
- Modify: `extension/tenferro-dyadtensor/src/core/value/tensor.rs`
- Modify: `extension/tenferro-dyadtensor/src/core/value/core.rs`
- Test: `extension/tenferro-dyadtensor/src/core/value/tests/mod.rs`

**Step 1: Write the failing tests**

Add tests that require:
- reverse-mode `AdTensor<T>` to be reconstructible from `TrackedValue<DynTensor>`
- typed convenience wrappers to reject dtype mismatch cleanly
- rank-0 tensor scalar behavior to remain unchanged

**Step 2: Run the focused test**

Run:

```bash
env CARGO_TARGET_DIR=/tmp/tenferro-dyntensor-canonical-tape-target cargo test -p tenferro-dyadtensor core::value::tests:: -- --nocapture
```

Expected: FAIL because `AdTensor<T>` still owns `TrackedValue<StructuredTensor<T>>`.

**Step 3: Implement the minimal code**

Convert `AdTensor<T>` into a typed facade over dynamic tracked values or minimize it to a compatibility convenience wrapper that no longer dictates tape payload type.

**Step 4: Run the focused test again**

Run the same command and verify it passes.

**Step 5: Commit**

```bash
git add extension/tenferro-dyadtensor/src/core/value/tensor.rs extension/tenferro-dyadtensor/src/core/value/core.rs extension/tenferro-dyadtensor/src/core/value/tests/mod.rs
git commit -m "refactor: decouple typed adtensor from structured tape payload"
```

### Task 4: Rebuild `DynAdTensor` Around `TrackedValue<DynTensor>` / `DualValue<DynTensor>`

**Files:**
- Modify: `extension/tenferro-dyadtensor/src/core/dynamic/dyn_ad_tensor/mod.rs`
- Modify: `extension/tenferro-dyadtensor/src/core/dynamic/dyn_ad_tensor/merge.rs`
- Modify: `extension/tenferro-dyadtensor/src/core/dynamic/dyn_ad_tensor/promotion.rs`
- Modify: `extension/tenferro-dyadtensor/src/core/dynamic/dyn_ad_tensor/snapshot.rs`
- Test: `extension/tenferro-dyadtensor/src/core/dynamic/tests/mod.rs`

**Step 1: Write the failing tests**

Add tests that require:
- `DynAdTensor` reverse values to expose `Tape<DynTensor>`
- `primal_snapshot()` to return `DynTensor`
- `Diag` structured payloads to survive reverse-mode wrapping

**Step 2: Run the focused test**

Run:

```bash
env CARGO_TARGET_DIR=/tmp/tenferro-dyntensor-canonical-tape-target cargo test -p tenferro-dyadtensor core::dynamic::tests::dyn_ad_tensor -- --nocapture
```

Expected: FAIL because `DynAdTensor` is still layered over typed tracked values and `DynStructuredPrimal`.

**Step 3: Implement the minimal code**

Make `DynAdTensor` the canonical public wrapper over dynamic tracked values. Replace `DynStructuredPrimal` snapshot boundaries with `DynTensor` where appropriate.

**Step 4: Run the focused test again**

Run the same command and verify it passes.

**Step 5: Commit**

```bash
git add extension/tenferro-dyadtensor/src/core/dynamic/dyn_ad_tensor/mod.rs extension/tenferro-dyadtensor/src/core/dynamic/dyn_ad_tensor/merge.rs extension/tenferro-dyadtensor/src/core/dynamic/dyn_ad_tensor/promotion.rs extension/tenferro-dyadtensor/src/core/dynamic/dyn_ad_tensor/snapshot.rs extension/tenferro-dyadtensor/src/core/dynamic/tests/mod.rs
git commit -m "refactor: rebuild dynadtensor on dynamic tracked values"
```

### Task 5: Replace Typed Tape Rule Registration With `ReverseRule<DynTensor>`

**Files:**
- Modify: `extension/tenferro-dyadtensor/src/tape/registry.rs`
- Modify: `extension/tenferro-dyadtensor/src/tape/tensor_pullback.rs`
- Test: `extension/tenferro-dyadtensor/src/tape/tests/mod.rs`

**Step 1: Write the failing tests**

Add tests that require:
- `tape::register_rule` to attach rules to `Tape<DynTensor>`
- reverse pullbacks to receive and return `DynTensor`
- dtype mismatch and unsupported HVP paths to report structured errors

**Step 2: Run the focused test**

Run:

```bash
env CARGO_TARGET_DIR=/tmp/tenferro-dyntensor-canonical-tape-target cargo test -p tenferro-dyadtensor tape::tests:: -- --nocapture
```

Expected: FAIL because the tape registry still uses `StructuredTensor<T>`-typed rules.

**Step 3: Implement the minimal code**

Introduce dynamic reverse rule adapters that downcast internally only at the rule boundary. Keep the public/internal contract at the tape layer as `DynTensor`.

**Step 4: Run the focused test again**

Run the same command and verify it passes.

**Step 5: Commit**

```bash
git add extension/tenferro-dyadtensor/src/tape/registry.rs extension/tenferro-dyadtensor/src/tape/tensor_pullback.rs extension/tenferro-dyadtensor/src/tape/tests/mod.rs
git commit -m "refactor: register dyadtensor pullbacks on dytensor tape"
```

### Task 6: Rewire Scalar, Reduction, And Einsum AD To The New Dynamic Tape

**Files:**
- Modify: `extension/tenferro-dyadtensor/src/ops/common.rs`
- Modify: `extension/tenferro-dyadtensor/src/ops/scalar/**`
- Modify: `extension/tenferro-dyadtensor/src/ops/reduction/**`
- Modify: `extension/tenferro-dyadtensor/src/ops/einsum/ad.rs`
- Test: `extension/tenferro-dyadtensor/src/ops/ad/tests/*.rs`
- Test: `extension/tenferro-dyadtensor/src/ops/tests/*.rs`

**Step 1: Write the failing tests**

Add or update tests so that:
- rank-0 tensor scalar ops still work in primal/forward/reverse mode
- `Diag` operands survive structured einsum and reduction AD
- mixed-dtype scalar promotion paths still follow the documented policy

**Step 2: Run the focused tests**

Run:

```bash
env CARGO_TARGET_DIR=/tmp/tenferro-dyntensor-canonical-tape-target cargo test -p tenferro-dyadtensor ops::ad::tests::scalar_generic -- --nocapture
env CARGO_TARGET_DIR=/tmp/tenferro-dyntensor-canonical-tape-target cargo test -p tenferro-dyadtensor ops::ad::tests::structured_pullbacks -- --nocapture
env CARGO_TARGET_DIR=/tmp/tenferro-dyntensor-canonical-tape-target cargo test -p tenferro-dyadtensor ops::ad::tests::einsum_two_stage -- --nocapture
```

Expected: FAIL until op wiring no longer assumes typed structured tapes.

**Step 3: Implement the minimal code**

Make op builders and pullback registration operate on dynamic graph values. Preserve current scalar semantics and `Diag` support.

**Step 4: Run the focused tests again**

Run the same commands and verify they pass.

**Step 5: Commit**

```bash
git add extension/tenferro-dyadtensor/src/ops/common.rs extension/tenferro-dyadtensor/src/ops/scalar extension/tenferro-dyadtensor/src/ops/reduction extension/tenferro-dyadtensor/src/ops/einsum/ad.rs extension/tenferro-dyadtensor/src/ops/ad/tests extension/tenferro-dyadtensor/src/ops/tests
git commit -m "refactor: rewire dyadtensor ops to dynamic tape"
```

### Task 7: Make Linalg AD Explicitly Dense-Only

**Files:**
- Modify: `extension/tenferro-dyadtensor/src/ops/linalg/**`
- Modify: `extension/tenferro-dyadtensor/src/ops/ad/layout.rs`
- Test: `extension/tenferro-dyadtensor/src/ops/ad/tests/builder_pullbacks.rs`
- Test: `extension/tenferro-dyadtensor/src/ops/tests/runtime_surface.rs`

**Step 1: Write the failing tests**

Add tests that require:
- dense linalg AD to keep working
- non-dense structured linalg AD to return explicit unsupported errors
- `Diag` linalg AD to reject rather than silently densify

**Step 2: Run the focused tests**

Run:

```bash
env CARGO_TARGET_DIR=/tmp/tenferro-dyntensor-canonical-tape-target cargo test -p tenferro-dyadtensor ops::ad::tests::builder_pullbacks -- --nocapture
env CARGO_TARGET_DIR=/tmp/tenferro-dyntensor-canonical-tape-target cargo test -p tenferro-dyadtensor ops::tests::runtime_surface -- --nocapture
```

Expected: FAIL until dense-only gates are explicit and tested.

**Step 3: Implement the minimal code**

Gate linalg AD on `is_dense()` and return explicit unsupported errors for structured non-dense layouts.

**Step 4: Run the focused tests again**

Run the same commands and verify they pass.

**Step 5: Commit**

```bash
git add extension/tenferro-dyadtensor/src/ops/linalg extension/tenferro-dyadtensor/src/ops/ad/layout.rs extension/tenferro-dyadtensor/src/ops/ad/tests/builder_pullbacks.rs extension/tenferro-dyadtensor/src/ops/tests/runtime_surface.rs
git commit -m "refactor: make dyadtensor linalg ad dense-only"
```

### Task 8: Separate Explicit Cast API From Implicit Promotion

**Files:**
- Modify: `extension/tenferro-dyadtensor/src/core/dynamic/dyn_ad_tensor/promotion.rs`
- Modify: `extension/tenferro-dyadtensor/src/core/dynamic/dyn_ad_tensor/scalar_ops.rs`
- Modify: `extension/tenferro-dyadtensor/src/core/dynamic/dyn_tensor.rs`
- Test: `extension/tenferro-dyadtensor/src/core/dynamic/tests/mod.rs`
- Doc: `docs/design/autodiff.md`

**Step 1: Write the failing tests**

Add tests that require:
- explicit dtype cast API to be separate from op-local promotion
- implicit promotion to remain narrow and operation-local
- reverse-mode behavior for unsupported cast gradients to be explicit

**Step 2: Run the focused test**

Run:

```bash
env CARGO_TARGET_DIR=/tmp/tenferro-dyntensor-canonical-tape-target cargo test -p tenferro-dyadtensor core::dynamic::tests:: -- --nocapture
```

Expected: FAIL because promotion and cast responsibilities are still mixed.

**Step 3: Implement the minimal code**

Introduce explicit cast API naming and keep algebraic promotion internal to op execution. Document any remaining AD limitations explicitly.

**Step 4: Run the focused test again**

Run the same command and verify it passes.

**Step 5: Commit**

```bash
git add extension/tenferro-dyadtensor/src/core/dynamic/dyn_ad_tensor/promotion.rs extension/tenferro-dyadtensor/src/core/dynamic/dyn_ad_tensor/scalar_ops.rs extension/tenferro-dyadtensor/src/core/dynamic/dyn_tensor.rs extension/tenferro-dyadtensor/src/core/dynamic/tests/mod.rs docs/design/autodiff.md
git commit -m "refactor: separate dyadtensor casts from promotion"
```

### Task 9: Update Public Docs, Crate Docs, And Examples

**Files:**
- Modify: `extension/tenferro-dyadtensor/src/lib.rs`
- Modify: `docs/design/autodiff.md`
- Modify: `docs/api_index.md`
- Modify: `docs/design/supported-ops.md`

**Step 1: Write the docs-first checks**

List all public items whose examples refer to:
- `Tape<StructuredTensor<T>>` as the public dyadtensor model
- old snapshot types
- old typed-tape assumptions

**Step 2: Implement the docs update**

Update docs to state:
- `DynTensor` is the canonical dynamic primal tensor
- `DynAdTensor` is the canonical dynamic AD tensor
- scalar = rank-0 tensor
- `Diag` remains supported
- linalg AD is dense-only

Add or refresh minimal `# Examples` sections where signatures changed.

**Step 3: Run docs checks**

Run:

```bash
env CARGO_TARGET_DIR=/tmp/tenferro-dyntensor-canonical-tape-doc-target cargo doc --workspace --no-deps
python3 scripts/check-docs-site.py --doc-root /tmp/tenferro-dyntensor-canonical-tape-doc-target/doc
```

Expected: PASS

**Step 4: Commit**

```bash
git add extension/tenferro-dyadtensor/src/lib.rs docs/design/autodiff.md docs/api_index.md docs/design/supported-ops.md
git commit -m "docs: update dyadtensor canonical tape model"
```

### Task 10: Full Verification, Ad Hoc Audit, And PR

**Files:**
- Review: `extension/tenferro-dyadtensor/src/**`
- Review: `extern/chainrules/src/**`
- Review: `docs/design/**`

**Step 1: Re-read the diff for ad hoc regressions**

Explicitly search for:
- duplicated promotion/cast logic
- lingering `Tape<StructuredTensor<T>>` assumptions in dyadtensor public docs
- any accidental `DynStructuredPrimal`-style intermediate types that are no longer needed
- any silent structured linalg AD fallback

**Step 2: Run full verification**

Run:

```bash
cargo fmt --all --check
env CARGO_TARGET_DIR=/tmp/tenferro-dyntensor-canonical-tape-release-target cargo test --workspace --release
env CARGO_TARGET_DIR=/tmp/tenferro-dyntensor-canonical-tape-cov-target cargo llvm-cov --workspace --release --json --output-path coverage.json
python3 scripts/check-coverage.py coverage.json
env CARGO_TARGET_DIR=/tmp/tenferro-dyntensor-canonical-tape-doc-target cargo doc --workspace --no-deps
python3 scripts/check-docs-site.py --doc-root /tmp/tenferro-dyntensor-canonical-tape-doc-target/doc
```

Expected: PASS

**Step 3: Run oracle coverage for dense linalg support**

Run:

```bash
env CARGO_TARGET_DIR=/tmp/tenferro-dyntensor-canonical-tape-oracle-target cargo test -p tenferro-linalg --test oracle_db -- --nocapture
```

Expected: PASS for all currently supported records.

**Step 4: Commit final cleanup**

```bash
git add .
git commit -m "refactor: make dyadtensor use canonical dytensor tape"
```

**Step 5: Create PR**

Run:

```bash
gh pr create --base main --head refactor/dyntensor-canonical-tape --title "refactor: make dyadtensor use canonical dytensor tape" --body "## Summary
- make DynTensor the canonical structured-aware dynamic tape payload
- rework DynAdTensor around Tape<DynTensor>
- keep Diag support and make linalg AD dense-only

Generated with [Claude Code](https://claude.com/claude-code)"
gh pr merge --auto --squash --delete-branch
bash scripts/monitor-pr-checks.sh <pr-number-or-url> --interval 30
```
