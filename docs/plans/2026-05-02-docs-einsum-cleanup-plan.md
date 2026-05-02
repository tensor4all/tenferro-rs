# Docs And Einsum Cleanup Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Bring current GPU and einsum documentation in line with the implemented CubeCL backend and repeated-label einsum behavior, while fixing the adjacent contraction-cost release-mode fallback.

**Architecture:** Documentation changes are limited to current design/user-facing docs and rustdoc; historical `docs/plans/` records remain unchanged. Code changes stay inside the `tenferro-einsum` planner utility path, converting a debug-only invariant into checked `Result` propagation.

**Tech Stack:** Rust, `tenferro-einsum`, `tenferro-tensor` CubeCL backend, Markdown docs, cargo test/doc tooling.

---

### Task 1: Planner Missing-Label Regression

**Files:**
- Modify: `tenferro-einsum/src/util.rs`
- Modify: `tenferro-einsum/src/planning/tree.rs`
- Modify: `tenferro-einsum/src/planning/tree/tests.rs`

**Step 1: Write the failing test**

Add a unit test in `tenferro-einsum/src/planning/tree/tests.rs` that calls the self-greedy planner with a `needed` output label absent from `size_dict`, and asserts `Error::InvalidArgument` rather than silently treating the label size as `1`.

**Step 2: Run test to verify it fails**

Run:

```bash
cargo test -p tenferro-einsum --lib -- self_greedy_pair_optimizer_rejects_missing_needed_label
```

Expected: FAIL because `optimize_self_greedy_pairs` currently returns a `Vec` and `contraction_cost` uses a debug-only assertion plus `unwrap_or(1)`.

**Step 3: Implement checked propagation**

Change `contraction_cost` to return `Result<usize>` and produce `Error::InvalidArgument` when an output/intermediate label lacks a known size. Change `optimize_self_greedy_pairs` to return `Result<Vec<(usize, usize)>>` and propagate the error from `optimize_with_options`.

**Step 4: Run focused tests**

Run:

```bash
cargo test -p tenferro-einsum --lib -- self_greedy_pair_optimizer
cargo test -p tenferro-einsum --lib -- optimize_with_options
```

Expected: PASS.

### Task 2: Repeated-Label Einsum Coverage

**Files:**
- Modify: `tenferro-einsum/src/tests/eager_tests.rs`
- Modify: `tenferro-einsum/src/lib.rs`
- Modify: `tenferro-einsum/src/syntax/subscripts.rs`
- Modify: `docs/guides/einsum.md`
- Modify: `docs/design/einsum.md`

**Step 1: Add higher-rank eager test**

Add an eager regression for `iij->ij` using a small deterministic tensor, asserting diagonal extraction across the first two axes while preserving the remaining axis.

**Step 2: Run test to verify current behavior**

Run:

```bash
cargo test -p tenferro-einsum --lib -- eager_einsum_handles_higher_rank_repeated_labels
```

Expected: PASS if the implementation already supports the behavior; keep it as documentation-backed coverage.

**Step 3: Update rustdoc and guide text**

Document that repeated labels in one input select diagonals, repeated labels absent from output are reduced after diagonal extraction, and repeated output labels embed diagonals. Include examples for `ii->`, `ii->i`, `i->ii`, and `iij->ij`. User-facing guide examples must import from `tenferro`, not internal crates.

**Step 4: Run relevant checks**

Run:

```bash
cargo test -p tenferro-einsum --lib -- eager_einsum_handles_higher_rank_repeated_labels
cargo test -p tenferro --doc
```

Expected: PASS.

### Task 3: Current GPU Design Documentation

**Files:**
- Modify: `docs/design/gpu-backend-design.md`
- Modify: `docs/design/einsum.md`

**Step 1: Replace stale direct-backend design claims**

Rewrite `docs/design/gpu-backend-design.md` around the current `CubeclBackend` implementation in `tenferro-tensor/src/cubecl/`, the `cubecl` feature, explicit upload/download policy, CUDA-only status, and lazy runtime loading of cuTENSOR/cuSOLVER/cuBLAS. Clearly state that ROCm is a stub and GPU benchmarking is out of scope for this batch.

**Step 2: Align einsum design status**

Update `docs/design/einsum.md` to describe the current traced/eager surfaces, repeated-label semantics, strict binary lowering fallback for repeated labels, and CubeCL status without presenting deleted `CudaBackend`/`RocmBackend` architecture as current.

**Step 3: Search for stale current claims**

Run:

```bash
rg -n "CudaBackend|RocmBackend|tenferro-prims|tenferro_prims|tenferro_tensor|tenferro_einsum" docs/design/gpu-backend-design.md docs/design/einsum.md docs/guides/einsum.md
```

Expected: no stale current-backend claims and no internal-crate imports in user-facing guide snippets.

### Task 4: Verification And Commit

**Files:**
- All files modified above

**Step 1: Format/check diffs**

Run:

```bash
cargo fmt --all --check
git diff --check
```

Expected: PASS.

**Step 2: Run focused code/docs checks**

Run:

```bash
cargo test -p tenferro-einsum
cargo test -p tenferro --doc
cargo doc --workspace --no-deps
python3 scripts/check-docs-site.py
```

Expected: PASS. If a full docs command is too slow or fails from pre-existing warnings outside the touched scope, record the exact failure and run the narrower command that covers the touched files.

**Step 3: Commit**

Run:

```bash
git add docs/plans/2026-05-02-docs-einsum-cleanup-plan.md \
  docs/design/gpu-backend-design.md docs/design/einsum.md docs/guides/einsum.md \
  tenferro-einsum/src/lib.rs tenferro-einsum/src/syntax/subscripts.rs \
  tenferro-einsum/src/util.rs tenferro-einsum/src/planning/tree.rs \
  tenferro-einsum/src/planning/tree/tests.rs tenferro-einsum/src/tests/eager_tests.rs
git commit -m "docs: align gpu and repeated-label einsum docs"
```
