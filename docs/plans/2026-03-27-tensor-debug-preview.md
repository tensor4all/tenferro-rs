# tenferro::Tensor Debug Preview Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a bounded logical-value preview to `tenferro::Tensor` `Debug` output without forcing surprising large-tensor or non-host materialization.

**Architecture:** Keep `tenferro::Tensor` as the public debug entrypoint, but move preview-policy and formatting details into a focused helper module under `dyn_ad_tensor`. The formatter should preserve the existing semantic metadata and append a bounded `preview` field derived from a logical dense snapshot only when the policy allows it.

**Tech Stack:** Rust, `fmt::Debug`, `tenferro::Tensor`, `snapshot::DynTensor`, `tenferro_tensor::StructuredTensor`, integration tests in `tenferro/tests`

---

### Task 1: Lock the public debug contract with failing tests

**Files:**
- Modify: `tenferro/tests/public_surface_tests.rs`

**Step 1: Write the failing test**

Add or update integration tests that assert:
- small dense `tenferro::Tensor` debug output contains metadata and visible values
- small diagonal `tenferro::Tensor` debug output shows logical diagonal values rather than just compressed payload values
- large tensor debug output stays bounded and does not dump all values

**Step 2: Run test to verify it fails**

Run: `cargo test -p tenferro --test public_surface_tests`

Expected: FAIL because `Debug` is still metadata-only today.

**Step 3: Commit**

Do not commit yet. Continue directly to implementation after the red test is confirmed.

### Task 2: Implement bounded preview formatting

**Files:**
- Modify: `tenferro/src/core/dynamic/dyn_ad_tensor/mod.rs`
- Create: `tenferro/src/core/dynamic/dyn_ad_tensor/debug.rs`

**Step 1: Write minimal implementation**

Create a focused debug helper module that:
- decides whether preview is allowed
- skips preview for non-main-memory tensors
- skips or truncates preview when logical element count is above the chosen cutoff
- materializes logical values through a dense snapshot only for small previews
- formats preview values in logical axis order

Wire `impl fmt::Debug for Tensor` through that helper while preserving the
existing metadata fields.

**Step 2: Run test to verify it passes**

Run: `cargo test -p tenferro --test public_surface_tests`

Expected: PASS

### Task 3: Document the new public behavior

**Files:**
- Modify: `tenferro/src/core/dynamic/dyn_ad_tensor/mod.rs`

**Step 1: Update rustdoc examples**

Add a short `Debug` example to the `tenferro::Tensor` type docs showing that
small tensors now include preview values in `{:?}` output.

**Step 2: Run focused doc verification**

Run: `cargo test -p tenferro --doc`

Expected: PASS

### Task 4: Run repository verification for PR readiness

**Files:**
- No code changes expected

**Step 1: Run formatting**

Run: `cargo fmt --all --check`

Expected: PASS

**Step 2: Run release workspace tests**

Run: `cargo test --workspace --release`

Expected: PASS

**Step 3: Run coverage gate**

Run: `cargo llvm-cov --workspace --json --output-path coverage.json`

Expected: PASS and `coverage.json` written

**Step 4: Check coverage thresholds**

Run: `python3 scripts/check-coverage.py coverage.json`

Expected: PASS

**Step 5: Run rustdoc build**

Run: `cargo doc --workspace --no-deps`

Expected: PASS

**Step 6: Run docs-site check**

Run: `python3 scripts/check-docs-site.py`

Expected: PASS

### Task 5: Publish the branch

**Files:**
- No code changes expected

**Step 1: Commit implementation**

Run:

```bash
git add tenferro/src/core/dynamic/dyn_ad_tensor/mod.rs \
        tenferro/src/core/dynamic/dyn_ad_tensor/debug.rs \
        tenferro/tests/public_surface_tests.rs \
        docs/plans/2026-03-27-tensor-debug-preview-design.md \
        docs/plans/2026-03-27-tensor-debug-preview.md
git commit -m "feat: preview values in tenferro tensor debug output"
```

**Step 2: Push branch**

Run: `git push -u origin codex/issue-577-tensor-debug-preview`

Expected: branch pushed successfully

**Step 3: Open PR**

Run `gh pr create` against `main`, reference `Closes #577`, and include a short
Codex attribution line.

**Step 4: Enable auto-merge**

Run: `gh pr merge --auto --squash --delete-branch <pr-number>`

Expected: auto-merge enabled while required checks run
