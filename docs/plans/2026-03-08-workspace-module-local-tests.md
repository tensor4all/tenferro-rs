# Workspace Module-Local Test Directories Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move remaining inline unit tests out of production files across the whole workspace and into module-local test directories, while preserving crate-root integration tests.

**Architecture:** Apply a mechanical refactor crate-by-crate. Each affected source file keeps only `#[cfg(test)] mod tests;`, and the previous inline test body moves into a sibling `tests/mod.rs` file that preserves private-item access through normal module nesting.

**Tech Stack:** Rust 2021, Cargo test harness, workspace `AGENTS.md` conventions

---

### Task 1: Record the workspace rule and current scope

**Files:**
- Modify: `AGENTS.md`
- Create: `docs/plans/2026-03-08-workspace-module-local-tests-design.md`

**Step 1: Confirm the repository rule**

Verify that `AGENTS.md` contains the workspace rule for module-local unit-test
directories and crate-root integration tests.

**Step 2: Save the approved design**

Write the approved workspace design doc covering:

- root crates, `extension/*`, and `extern/*`
- crate-level rollout
- final workspace verification

### Task 2: Migrate `tenferro-tensor`

**Files:**
- Modify: `tenferro-tensor/src/lib.rs`
- Create: `tenferro-tensor/src/tests/mod.rs`

**Step 1: Move the inline test module**

Replace the inline `mod tests { ... }` block in `tenferro-tensor/src/lib.rs`
with:

```rust
#[cfg(test)]
mod tests;
```

Move the original test body into `tenferro-tensor/src/tests/mod.rs`.

**Step 2: Verify the crate**

Run:

```bash
cargo test -p tenferro-tensor
```

Expected: the crate compiles and all tests pass.

### Task 3: Migrate `tenferro-prims` and `extension/tenferro-tropical`

**Files:**
- Modify: `tenferro-prims/src/gpu_stubs.rs`
- Create: `tenferro-prims/src/gpu_stubs/tests/mod.rs`
- Modify: `extension/tenferro-tropical/src/ad.rs`
- Create: `extension/tenferro-tropical/src/ad/tests/mod.rs`

**Step 1: Move the inline test modules**

Apply the same extraction pattern in both crates.

**Step 2: Verify each crate**

Run:

```bash
cargo test -p tenferro-prims
cargo test -p tenferro-tropical
```

Expected: both crates compile and all tests pass.

### Task 4: Migrate `tenferro-einsum`

**Files:**
- Modify: `tenferro-einsum/src/lib.rs`
- Create: `tenferro-einsum/src/tests/mod.rs`
- Modify: `tenferro-einsum/src/binary.rs`
- Create: `tenferro-einsum/src/binary/tests/mod.rs`
- Modify: `tenferro-einsum/src/manual.rs`
- Create: `tenferro-einsum/src/manual/tests/mod.rs`
- Modify: `tenferro-einsum/src/pool.rs`
- Create: `tenferro-einsum/src/pool/tests/mod.rs`
- Modify: `tenferro-einsum/src/prepare.rs`
- Create: `tenferro-einsum/src/prepare/tests/mod.rs`
- Modify: `tenferro-einsum/src/util.rs`
- Create: `tenferro-einsum/src/util/tests/mod.rs`

**Step 1: Move each inline test module**

Extract each inline `mod tests { ... }` block into the sibling `tests/mod.rs`
path for that module.

**Step 2: Verify the crate**

Run:

```bash
cargo test -p tenferro-einsum
```

Expected: the crate compiles and all tests pass.

### Task 5: Migrate `tenferro-linalg`

**Files:**
- Modify: `tenferro-linalg/src/prims_bridge.rs`
- Create: `tenferro-linalg/src/prims_bridge/tests/mod.rs`
- Modify: `tenferro-linalg/src/backend/blas_lapack_backend.rs`
- Create: `tenferro-linalg/src/backend/blas_lapack_backend/tests/mod.rs`
- Modify: `tenferro-linalg/src/backend/cpu.rs`
- Create: `tenferro-linalg/src/backend/cpu/tests/mod.rs`
- Modify: `tenferro-linalg/src/backend/cuda.rs`
- Create: `tenferro-linalg/src/backend/cuda/tests/mod.rs`
- Modify: `tenferro-linalg/src/backend/faer_backend.rs`
- Create: `tenferro-linalg/src/backend/faer_backend/tests/mod.rs`
- Modify: `tenferro-linalg/src/backend/hip.rs`
- Create: `tenferro-linalg/src/backend/hip/tests/mod.rs`
- Modify: `tenferro-linalg/src/backend/tensor_api.rs`
- Create: `tenferro-linalg/src/backend/tensor_api/tests/mod.rs`
- Modify: `tenferro-linalg/src/backend/tensor_helpers.rs`
- Create: `tenferro-linalg/src/backend/tensor_helpers/tests/mod.rs`

**Step 1: Move each inline test module**

Extract each inline `mod tests { ... }` block into the sibling `tests/mod.rs`
path for that module.

**Step 2: Verify the crate**

Run:

```bash
cargo test -p tenferro-linalg
```

Expected: the crate compiles and all tests pass.

### Task 6: Final audit and workspace verification

**Files:**
- Modify: `tenferro-tensor/src/**`
- Modify: `tenferro-prims/src/**`
- Modify: `tenferro-einsum/src/**`
- Modify: `tenferro-linalg/src/**`
- Modify: `extension/tenferro-tropical/src/**`

**Step 1: Re-scan the workspace**

Run:

```bash
rg -n "mod tests \\{" --glob '!**/.worktrees/**' --glob '!**/target/**' --glob '*/src/**/*.rs'
```

Expected: no matches in tracked workspace source files.

**Step 2: Run formatter check**

Run:

```bash
cargo fmt --all --check
```

Expected: formatting passes with exit code 0.

**Step 3: Run full workspace tests**

Run:

```bash
cargo test --workspace
```

Expected: all workspace crates pass.
