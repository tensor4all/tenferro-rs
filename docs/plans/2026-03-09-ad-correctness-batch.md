# AD Correctness Batch Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix `#413`, `#414`, `#419`, and `#420` so the public AD surfaces stop returning wrong results or panicking on valid inputs.

**Architecture:** Keep each fix in the crate that owns the bug. Start with regression tests, apply the minimum implementation change that makes the test pass, then refresh the affected public docs/examples where the bug changed the documented contract.

**Tech Stack:** Rust workspace crates (`chainrules`, `chainrules-scalarops`, `tenferro-einsum`, `tenferro-burn`, `tenferro-linalg`), rustdoc examples, cargo test, cargo llvm-cov.

---

### Task 1: Fix tracked einsum HVP and stale public examples (`#413`)

**Files:**
- Modify: `extern/chainrules/src/lib.rs`
- Modify: `docs/design/autodiff.md`
- Modify: `tenferro-einsum/src/ad.rs`
- Test: `tenferro-einsum/tests/einsum_tests.rs`

**Step 1: Write the failing test**

Add a regression in `tenferro-einsum/tests/einsum_tests.rs` that constructs a `TrackedTensor` via `Tape::leaf_with_tangent`, runs `tracked_einsum::<Standard<f64>, CpuBackend>`, and asserts that `Tape::hvp` returns `[2.0, 2.0, 2.0]` for the documented `x·x` example.

**Step 2: Run test to verify it fails**

Run: `cargo test -p tenferro-einsum hvp_via_leaf_with_tangent_tracks_einsum_direction -- --nocapture`

Expected: FAIL with zero HVP output or a mismatched tangent assertion.

**Step 3: Write minimal implementation**

Update `tenferro-einsum/src/ad.rs` so tracked einsum records/propagates the tangent source used by the public `TrackedTensor` tape path instead of relying only on `Tensor::fw_grad`. Refresh the stale `tracked_einsum` / `dual_einsum` examples in `extern/chainrules/src/lib.rs` and `docs/design/autodiff.md` to include the required backend context arguments.

**Step 4: Run test to verify it passes**

Run: `cargo test -p tenferro-einsum hvp_via_leaf_with_tangent_tracks_einsum_direction -- --nocapture`

Expected: PASS with the HVP matching `[2.0, 2.0, 2.0]`.

**Step 5: Commit**

```bash
git add extern/chainrules/src/lib.rs docs/design/autodiff.md tenferro-einsum/src/ad.rs tenferro-einsum/tests/einsum_tests.rs
git commit -m "fix: preserve tracked einsum hvp tangents"
```

### Task 2: Fix singular `sqrt(0)` derivative handling (`#414`)

**Files:**
- Modify: `extern/chainrules-scalarops/src/lib.rs`
- Test: `extern/chainrules-scalarops/tests/scalarops_tests.rs`

**Step 1: Write the failing test**

Add regression tests that call `sqrt_frule(0.0, 1.0)` and `sqrt_rrule(0.0, 1.0)` and assert the derivative/cotangent is non-finite instead of zero.

**Step 2: Run test to verify it fails**

Run: `cargo test -p chainrules-scalarops sqrt_rules_surface_singularity_at_zero -- --nocapture`

Expected: FAIL because the current implementation returns `0.0`.

**Step 3: Write minimal implementation**

Remove the zero-special-case clamping in `extern/chainrules-scalarops/src/lib.rs` and use the direct derivative formula for the zero branch as well.

**Step 4: Run test to verify it passes**

Run: `cargo test -p chainrules-scalarops sqrt_rules_surface_singularity_at_zero -- --nocapture`

Expected: PASS with a non-finite derivative/cotangent at zero.

**Step 5: Commit**

```bash
git add extern/chainrules-scalarops/src/lib.rs extern/chainrules-scalarops/tests/scalarops_tests.rs
git commit -m "fix: surface sqrt singular derivatives"
```

### Task 3: Support N-ary Burn autodiff einsum (`#419`)

**Files:**
- Modify: `extension/tenferro-burn/src/backward.rs`
- Test: `extension/tenferro-burn/src/tests/mod.rs`

**Step 1: Write the failing test**

Add a regression in `extension/tenferro-burn/src/tests/mod.rs` that runs a three-input autodiff einsum on `Autodiff<NdArray<f64>>` and asserts it does not panic and returns gradients for all operands.

**Step 2: Run test to verify it fails**

Run: `cargo test -p tenferro-burn autodiff_three_input_einsum_propagates_gradients -- --nocapture`

Expected: FAIL with the current `panic!` about only supporting unary and binary autodiff einsum.

**Step 3: Write minimal implementation**

Generalize `extension/tenferro-burn/src/backward.rs` so the autodiff backend builds an N-ary reverse rule and maps all returned cotangents back into Burn tensors instead of matching only arities `1` and `2`.

**Step 4: Run test to verify it passes**

Run: `cargo test -p tenferro-burn autodiff_three_input_einsum_propagates_gradients -- --nocapture`

Expected: PASS with no panic and gradient tensors for all inputs.

**Step 5: Commit**

```bash
git add extension/tenferro-burn/src/backward.rs extension/tenferro-burn/src/tests/mod.rs
git commit -m "fix: support n-ary burn autodiff einsum"
```

### Task 4: Preserve square matrix RHS shape in solve AD (`#420`)

**Files:**
- Modify: `tenferro-linalg/src/lib.rs`
- Test: `tenferro-linalg/tests/linalg_tests.rs`

**Step 1: Write the failing test**

Add square-matrix-RHS regressions for `solve_rrule`, `solve_frule`, `solve_triangular_rrule`, and `solve_triangular_frule` that assert returned tangent/cotangent shapes remain `[n, n]`.

**Step 2: Run test to verify it fails**

Run: `cargo test -p tenferro-linalg solve_ad_square_rhs -- --nocapture`

Expected: FAIL because the current rules return `[n, 1]`.

**Step 3: Write minimal implementation**

Replace the open-coded `nrhs` inference in `tenferro-linalg/src/lib.rs` with the same vector/matrix RHS interpretation used by the forward tensor helpers, then keep the AD math unchanged once the shapes are correct.

**Step 4: Run test to verify it passes**

Run: `cargo test -p tenferro-linalg solve_ad_square_rhs -- --nocapture`

Expected: PASS with the full RHS shape preserved for all four rules.

**Step 5: Commit**

```bash
git add tenferro-linalg/src/lib.rs tenferro-linalg/tests/linalg_tests.rs
git commit -m "fix: preserve multi-rhs solve ad shapes"
```

### Task 5: Final verification and PR prep

**Files:**
- Verify staged implementation changes only

**Step 1: Run focused regression crates**

Run:

```bash
cargo test -p tenferro-einsum
cargo test -p chainrules-scalarops
cargo test -p tenferro-burn
cargo test -p tenferro-linalg
```

Expected: PASS

**Step 2: Run repository verification**

Run:

```bash
cargo fmt --all --check
cargo test --workspace --release
cargo llvm-cov --workspace --json --output-path coverage.json
python3 scripts/check-coverage.py coverage.json
cargo doc --workspace --no-deps
python3 scripts/check-docs-site.py
```

Expected: PASS

**Step 3: Remove plan artifacts from the PR if they are not meant to ship**

Before creating the PR, make sure `docs/plans/2026-03-09-ad-correctness-batch-design.md` and `docs/plans/2026-03-09-ad-correctness-batch.md` are not left in the final diff unless explicitly desired.

**Step 4: Commit final branch state**

```bash
git add -A
git commit -m "fix: restore ad correctness contracts"
```

**Step 5: Create PR**

```bash
git push -u origin fix/ad-correctness-batch
gh pr create
gh pr merge --auto --squash --delete-branch
```
