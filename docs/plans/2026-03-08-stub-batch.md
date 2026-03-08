# Stub Batch Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the remaining non-GPU-runtime public stubs with working implementations and align top-level docs with the current GPU support reality.

**Architecture:** Reuse existing lower-level implementations wherever possible. The batch is split into three layers: low-risk CPU conversion/dispatch work, C API interop that needs explicit ownership handling, and the Burn bridge where custom autodiff integration is required only for unary/binary cases.

**Tech Stack:** Rust workspace crates, Burn `0.21.0-pre.2`, DLPack C ABI, tenferro tensor/einsum/tropical/linalg layers.

---

### Task 1: Add the documentation warnings and fix the easy CPU-side stubs

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `extension/tenferro-mdarray/src/lib.rs`
- Create: `extension/tenferro-mdarray/src/tests/mod.rs`
- Modify: `tenferro-linalg/src/backend/mod.rs`
- Create: `tenferro-linalg/src/backend/cpu_tensor_impl.rs`
- Modify: `tenferro-linalg/src/backend/cpu_faer.rs`
- Modify: `tenferro-linalg/src/backend/cpu_lapack.rs`
- Modify: `tenferro-linalg/src/lib.rs`
- Test: `tenferro-linalg/tests/inject_tests.rs`

**Steps:**
1. Write failing mdarray and lapack tensor-dispatch tests.
2. Verify the tests fail against the current stubs.
3. Implement the mdarray conversions and the shared tensor-level LAPACK dispatch path.
4. Update README/AGENTS wording so GPU support is explicitly marked as stubbed/planned.
5. Re-run the focused tests until they pass.

### Task 2: Implement DLPack import/export

**Files:**
- Modify: `tenferro-tensor/src/lib.rs`
- Modify: `tenferro-capi/src/lib.rs`
- Modify: `tenferro-capi/tests/capi_tests.rs`

**Steps:**
1. Write failing DLPack export/import tests covering round-trip metadata and deleter ownership.
2. Add a tensor constructor for external buffers with validated dims/strides/offset.
3. Implement `tfe_tensor_f64_to_dlpack` with a manager object that owns the consumed tensor handle.
4. Implement `tfe_tensor_f64_from_dlpack` with CPU `f64` validation and release callback plumbing.
5. Re-run focused C API tests until they pass.

### Task 3: Implement tropical C API primal, rrule, and frule

**Files:**
- Modify: `extension/tenferro-tropical/src/ad.rs`
- Modify: `extension/tenferro-tropical-capi/src/lib.rs`
- Create: `extension/tenferro-tropical-capi/tests/mod.rs`

**Steps:**
1. Write failing tropical C API tests for primal, rrule, and frule on valid binary contractions.
2. Add a public tropical frule helper alongside the existing tropical rrule helper.
3. Implement shared operand parsing/conversion helpers in the tropical C API.
4. Wire the nine C entrypoints to the real primal/rrule/frule implementations.
5. Re-run the focused tropical tests until they pass.

### Task 4: Implement the Burn bridge

**Files:**
- Modify: `extension/tenferro-burn/src/lib.rs`
- Modify: `extension/tenferro-burn/src/convert.rs`
- Modify: `extension/tenferro-burn/src/forward.rs`
- Modify: `extension/tenferro-burn/src/backward.rs`
- Create: `extension/tenferro-burn/src/tests/mod.rs`

**Steps:**
1. Write failing tests for Burn conversion round-trips, forward einsum, and unary/binary autodiff gradients.
2. Implement `burn_to_tenferro` and `tenferro_to_burn` through `TensorData`.
3. Implement the `NdArray<f64>` primitive einsum path and the high-level wrapper.
4. Implement unary/binary autodiff recording using Burn `Backward` ops and tenferro `einsum_rrule`.
5. Make unsupported arities fail clearly instead of reaching `todo!()`.
6. Re-run the focused Burn tests until they pass.

### Task 5: Verify, review, and integrate

**Files:**
- Review: workspace diff

**Steps:**
1. Run `cargo fmt --all`.
2. Run `cargo test --workspace --release`.
3. Run `cargo llvm-cov --workspace --json --output-path coverage.json`.
4. Run `python3 scripts/check-coverage.py coverage.json`.
5. Run `cargo doc --workspace --no-deps`.
6. Run `python3 scripts/check-docs-site.py`.
7. Commit the branch, create a PR with `gh pr create`, enable auto-merge, and monitor CI until merge.
