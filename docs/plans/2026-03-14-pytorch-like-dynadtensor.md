# PyTorch-Like DynAdTensor Public Surface Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `DynAdTensor` the only public dyadtensor tensor type while keeping `DynTensor` as the internal `Tape<DynTensor>` payload and enabling PyTorch-like cast/promotion semantics.

**Architecture:** `tenferro-dyadtensor` becomes a dynamic extension layer with one public tensor object. `DynAdTensor` owns primal/forward/reverse state; `DynTensor` stays internal for graph execution, snapshots, and storage boundaries. Implicit promotion stays internal to ops, while explicit cast becomes a public `to_scalar_type(...)` API.

**Tech Stack:** Rust workspace, `chainrules`, `tenferro-dyadtensor`, `tenferro-linalg`, `tensor-ad-oracles`, rustdoc, workspace coverage/docs gates.

---

### Task 1: Freeze The New Public Contract In Tests And Docs

**Files:**
- Create: `extension/tenferro-dyadtensor/tests/public_surface_tests.rs`
- Modify: `docs/design/autodiff.md`
- Modify: `docs/api_index.md`
- Modify: `extension/tenferro-dyadtensor/src/lib.rs`

**Step 1: Write the failing tests**

Add tests and rustdoc expectations that require:
- `DynAdTensor` to be the documented public tensor type
- `AdTensor`, `AdScalar`, and `AdValue` to be absent from public re-exports
- crate-level examples to use `DynAdTensor`

**Step 2: Run test to verify it fails**

Run:

```bash
env CARGO_TARGET_DIR=/tmp/tenferro-pytorch-like-dyadtensor-target cargo test -p tenferro-dyadtensor public_surface_tests -- --nocapture
```

Expected: FAIL because typed public AD symbols are still exported and documented.

**Step 3: Write minimal implementation**

Remove typed AD re-exports from the crate root and rewrite public docs/examples
around `DynAdTensor`.

**Step 4: Run test to verify it passes**

Run the same command and verify it passes.

**Step 5: Commit**

```bash
git add extension/tenferro-dyadtensor/tests/public_surface_tests.rs extension/tenferro-dyadtensor/src/lib.rs docs/design/autodiff.md docs/api_index.md
git commit -m "refactor: make dynadtensor the public dyadtensor surface"
```

### Task 2: Make `DynAdTensor` Own Public Constructors And Introspection

**Files:**
- Modify: `extension/tenferro-dyadtensor/src/core/dynamic/dyn_ad_tensor/mod.rs`
- Modify: `extension/tenferro-dyadtensor/src/core/dynamic/dyn_ad_tensor/basics.rs`
- Modify: `extension/tenferro-dyadtensor/src/core/dynamic/dyn_ad_tensor/snapshot.rs`
- Test: `extension/tenferro-dyadtensor/src/core/dynamic/tests/mod.rs`

**Step 1: Write the failing tests**

Add tests that require:
- primal/forward/reverse constructors on `DynAdTensor`
- `DynAdTensor::primal_snapshot()` returning internal `DynTensor`
- `DynAdTensor` preserving dense/diag metadata

**Step 2: Run test to verify it fails**

Run:

```bash
env CARGO_TARGET_DIR=/tmp/tenferro-pytorch-like-dyadtensor-target cargo test -p tenferro-dyadtensor core::dynamic::tests:: -- --nocapture
```

Expected: FAIL because construction still routes through typed `AdTensor<T>`.

**Step 3: Write minimal implementation**

Move public construction/introspection APIs onto `DynAdTensor` and keep `DynTensor`
strictly internal.

**Step 4: Run test to verify it passes**

Run the same command and verify it passes.

**Step 5: Commit**

```bash
git add extension/tenferro-dyadtensor/src/core/dynamic/dyn_ad_tensor/mod.rs extension/tenferro-dyadtensor/src/core/dynamic/dyn_ad_tensor/basics.rs extension/tenferro-dyadtensor/src/core/dynamic/dyn_ad_tensor/snapshot.rs extension/tenferro-dyadtensor/src/core/dynamic/tests/mod.rs
git commit -m "refactor: give dynadtensor direct public constructors"
```

### Task 3: Internalize Typed AD Wrappers

**Files:**
- Modify: `extension/tenferro-dyadtensor/src/core/value/mod.rs`
- Modify: `extension/tenferro-dyadtensor/src/core/value/tensor.rs`
- Modify: `extension/tenferro-dyadtensor/src/core/value/scalar/mod.rs`
- Modify: `extension/tenferro-dyadtensor/src/core/value/core.rs`
- Test: `extension/tenferro-dyadtensor/src/core/value/tests/mod.rs`

**Step 1: Write the failing tests**

Add tests that require typed AD wrappers to stay internal-only and verify that
dynamic construction paths still preserve AD mode and tape handles.

**Step 2: Run test to verify it fails**

Run:

```bash
env CARGO_TARGET_DIR=/tmp/tenferro-pytorch-like-dyadtensor-target cargo test -p tenferro-dyadtensor core::value::tests:: -- --nocapture
```

Expected: FAIL because typed wrappers still leak into public assumptions.

**Step 3: Write minimal implementation**

Keep or simplify typed wrappers only as private implementation helpers. Remove
their influence on public API shape.

**Step 4: Run test to verify it passes**

Run the same command and verify it passes.

**Step 5: Commit**

```bash
git add extension/tenferro-dyadtensor/src/core/value/mod.rs extension/tenferro-dyadtensor/src/core/value/tensor.rs extension/tenferro-dyadtensor/src/core/value/scalar/mod.rs extension/tenferro-dyadtensor/src/core/value/core.rs extension/tenferro-dyadtensor/src/core/value/tests/mod.rs
git commit -m "refactor: internalize typed dyadtensor ad wrappers"
```

### Task 4: Dynamic Result Types For Linalg And AD Entry Points

**Files:**
- Modify: `extension/tenferro-dyadtensor/src/ops/linalg/results.rs`
- Modify: `extension/tenferro-dyadtensor/src/ops/linalg/ad/eager.rs`
- Modify: `extension/tenferro-dyadtensor/src/ops/ad/scalar_eager.rs`
- Modify: `extension/tenferro-dyadtensor/src/ops/ad/mod.rs`
- Test: `extension/tenferro-dyadtensor/src/ops/ad/tests/eager_surface.rs`

**Step 1: Write the failing tests**

Add tests that require:
- dynamic linalg result wrappers built from `DynAdTensor`
- eager AD entry points to take and return `DynAdTensor`
- scalar eager APIs to use dynamic tensors only

**Step 2: Run test to verify it fails**

Run:

```bash
env CARGO_TARGET_DIR=/tmp/tenferro-pytorch-like-dyadtensor-target cargo test -p tenferro-dyadtensor ops::ad::tests::eager_surface -- --nocapture
```

Expected: FAIL because eager APIs and result types still depend on `AdTensor<T>`.

**Step 3: Write minimal implementation**

Replace typed result wrappers and eager signatures with dynamic equivalents.

**Step 4: Run test to verify it passes**

Run the same command and verify it passes.

**Step 5: Commit**

```bash
git add extension/tenferro-dyadtensor/src/ops/linalg/results.rs extension/tenferro-dyadtensor/src/ops/linalg/ad/eager.rs extension/tenferro-dyadtensor/src/ops/ad/scalar_eager.rs extension/tenferro-dyadtensor/src/ops/ad/mod.rs extension/tenferro-dyadtensor/src/ops/ad/tests/eager_surface.rs
git commit -m "refactor: make dyadtensor eager ad api fully dynamic"
```

### Task 5: Add PyTorch-Like Explicit Cast And Internal Promotion/Cast-Back

**Files:**
- Modify: `extension/tenferro-dyadtensor/src/core/dynamic/dyn_ad_tensor/promotion.rs`
- Modify: `extension/tenferro-dyadtensor/src/core/dynamic/dyn_ad_tensor/scalar_ops.rs`
- Modify: `extension/tenferro-dyadtensor/src/ops/scalar/ad/common.rs`
- Modify: `extension/tenferro-dyadtensor/src/ops/common.rs`
- Test: `extension/tenferro-dyadtensor/tests/mixed_primitives_forward_tests.rs`
- Test: `extension/tenferro-dyadtensor/tests/mixed_primitives_reverse_tests.rs`
- Test: `extension/tenferro-dyadtensor/tests/mixed_complex_real_scalar_tests.rs`

**Step 1: Write the failing tests**

Add tests that require:
- `DynAdTensor::to_scalar_type(...)`
- mixed-dtype forward execution through implicit promotion
- reverse pullbacks cast back to each input dtype

**Step 2: Run test to verify it fails**

Run:

```bash
env CARGO_TARGET_DIR=/tmp/tenferro-pytorch-like-dyadtensor-target cargo test -p tenferro-dyadtensor mixed_primitives_forward_tests -- --nocapture
env CARGO_TARGET_DIR=/tmp/tenferro-pytorch-like-dyadtensor-target cargo test -p tenferro-dyadtensor mixed_primitives_reverse_tests -- --nocapture
env CARGO_TARGET_DIR=/tmp/tenferro-pytorch-like-dyadtensor-target cargo test -p tenferro-dyadtensor mixed_complex_real_scalar_tests -- --nocapture
```

Expected: FAIL because public cast is still `promote_to(...)`-shaped and reverse
cast-back is incomplete.

**Step 3: Write minimal implementation**

Introduce PyTorch-like explicit cast and keep operation-local promotion internal.
Cast gradients back to each input dtype during reverse execution.

**Step 4: Run test to verify it passes**

Run the same commands and verify they pass.

**Step 5: Commit**

```bash
git add extension/tenferro-dyadtensor/src/core/dynamic/dyn_ad_tensor/promotion.rs extension/tenferro-dyadtensor/src/core/dynamic/dyn_ad_tensor/scalar_ops.rs extension/tenferro-dyadtensor/src/ops/scalar/ad/common.rs extension/tenferro-dyadtensor/src/ops/common.rs extension/tenferro-dyadtensor/tests/mixed_primitives_forward_tests.rs extension/tenferro-dyadtensor/tests/mixed_primitives_reverse_tests.rs extension/tenferro-dyadtensor/tests/mixed_complex_real_scalar_tests.rs
git commit -m "feat: add pytorch-like dynamic casts for dyadtensor ad"
```

### Task 6: Keep Structured AD Narrow And Linalg AD Dense-Only

**Files:**
- Modify: `extension/tenferro-dyadtensor/src/ops/linalg/ad/*.rs`
- Modify: `extension/tenferro-dyadtensor/src/ops/einsum/ad.rs`
- Modify: `extension/tenferro-dyadtensor/src/ops/reduction/ad.rs`
- Test: `extension/tenferro-dyadtensor/tests/structured_linalg_fallback_tests.rs`
- Test: `extension/tenferro-dyadtensor/tests/structured_reverse_tests.rs`

**Step 1: Write the failing tests**

Add tests that require:
- structured einsum/reduction AD to keep working
- structured linalg AD to fail explicitly
- dense linalg AD to remain available through the dynamic public surface

**Step 2: Run test to verify it fails**

Run:

```bash
env CARGO_TARGET_DIR=/tmp/tenferro-pytorch-like-dyadtensor-target cargo test -p tenferro-dyadtensor structured_reverse_tests -- --nocapture
env CARGO_TARGET_DIR=/tmp/tenferro-pytorch-like-dyadtensor-target cargo test -p tenferro-dyadtensor structured_linalg_fallback_tests -- --nocapture
```

Expected: FAIL until structured boundaries are enforced through the new dynamic API.

**Step 3: Write minimal implementation**

Preserve supported structured AD families and enforce dense-only linalg AD.

**Step 4: Run test to verify it passes**

Run the same commands and verify they pass.

**Step 5: Commit**

```bash
git add extension/tenferro-dyadtensor/src/ops/linalg/ad extension/tenferro-dyadtensor/src/ops/einsum/ad.rs extension/tenferro-dyadtensor/src/ops/reduction/ad.rs extension/tenferro-dyadtensor/tests/structured_linalg_fallback_tests.rs extension/tenferro-dyadtensor/tests/structured_reverse_tests.rs
git commit -m "refactor: enforce dynamic structured ad boundaries"
```

### Task 7: Final Docs, Oracle Verification, And Full Gate

**Files:**
- Modify: `docs/design/autodiff.md`
- Modify: `docs/design/supported-ops.md`
- Modify: `docs/api_index.md`
- Modify: public rustdoc under `extension/tenferro-dyadtensor/src/**`

**Step 1: Write the failing test or check**

Add any final focused tests needed for public examples and dynamic result docs.

**Step 2: Run focused oracle and crate tests**

Run:

```bash
env CARGO_TARGET_DIR=/tmp/tenferro-pytorch-like-dyadtensor-target cargo test -p tenferro-linalg --test oracle_db -- --nocapture
env CARGO_TARGET_DIR=/tmp/tenferro-pytorch-like-dyadtensor-target cargo test -p tenferro-dyadtensor --release
```

Expected: PASS once the redesign is complete.

**Step 3: Update docs and examples**

Ensure every public type/trait/function still has minimal usable rustdoc
examples and that active docs describe `DynAdTensor` as the only public tensor
object for the extension crate.

**Step 4: Run full verification**

Run:

```bash
cargo fmt --all --check
env CARGO_TARGET_DIR=/tmp/tenferro-pytorch-like-dyadtensor-target cargo test --workspace --release
env CARGO_TARGET_DIR=/tmp/tenferro-pytorch-like-dyadtensor-target cargo llvm-cov --workspace --release --json --output-path coverage.json
python3 scripts/check-coverage.py coverage.json
env CARGO_TARGET_DIR=/tmp/tenferro-pytorch-like-dyadtensor-target cargo doc --workspace --no-deps
python3 scripts/check-docs-site.py --doc-root /tmp/tenferro-pytorch-like-dyadtensor-target/doc
```

Expected: all commands succeed.

**Step 5: Commit**

```bash
git add docs/design/autodiff.md docs/design/supported-ops.md docs/api_index.md extension/tenferro-dyadtensor/src extension/tenferro-dyadtensor/tests
git commit -m "refactor: make dyadtensor a pytorch-like dynamic tensor api"
```
