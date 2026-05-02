# AD Structural Boundary Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add correct first-order AD boundary rules for simple structural ops that currently panic during VJP/JVP, without expanding linalg AD scope.

**Architecture:** Keep `PrimitiveOp::linearize` and `PrimitiveOp::transpose_rule` as the source of truth. Implement only Slice, Pad, and Reverse structural rules because they are linear maps with static configurations and can be checked by finite differences in the `tenferro` facade tests. Leave TriangularSolve/FullPivLuSolve A-cotangents and broader linalg transpose rules to a dedicated linalg AD plan, matching the current batch boundary.

**Tech Stack:** Rust, `tenferro-ops` AD modules, `tenferro` integration AD tests, finite-difference checks.

---

## Scope For This Step

Implement:

- `StdTensorOp::Slice` forward linearization and transpose rule,
- `StdTensorOp::Pad` transpose rule for statically representable padding/cropping,
- `StdTensorOp::Reverse` forward linearization and transpose rule,
- finite-difference integration tests for every new transpose/linearize rule.

Defer:

- TriangularSolve/FullPivLuSolve A-cotangents, because this is linalg AD and should get a dedicated oracle-backed design,
- DynamicSlice/Concatenate/Select/Clamp/Maximum/Minimum AD rules,
- symbolic-shape Scatter transpose redesign, because the current trait cannot return structured unsupported-shape errors.

## Task 1: Add Failing Integration Tests

**Files:**

- Modify: `tenferro/tests/ad.rs`

**Steps:**

1. Add fragment builders for weighted `Slice`, `Pad`, and `Reverse` reductions.
2. Add tests:
   - `grad_slice_weighted_sum_matches_finite_diff`
   - `grad_pad_weighted_sum_matches_finite_diff`
   - `grad_reverse_weighted_sum_matches_finite_diff`
3. Each test computes `grad_from_fragment_with_inputs(...)` and compares every input element with `finite_diff_scalar(...)`.
4. Run:

```bash
cargo test -p tenferro --test ad -- grad_slice_weighted_sum_matches_finite_diff
cargo test -p tenferro --test ad -- grad_pad_weighted_sum_matches_finite_diff
cargo test -p tenferro --test ad -- grad_reverse_weighted_sum_matches_finite_diff
```

Expected: fail because `Slice`/`Reverse` linearize or `Pad` transpose are not implemented.

## Task 2: Add Focused `tenferro-ops` Rule Tests

**Files:**

- Modify: `tenferro-ops/src/tests/std_tensor_op_tests.rs`

**Steps:**

1. Add structural unit assertions that:
   - `Slice` linearize emits `StdTensorOp::Slice`,
   - `Slice` transpose emits `StdTensorOp::Pad`,
   - `Pad` transpose emits `StdTensorOp::Slice` and optionally `StdTensorOp::Pad`,
   - `Reverse` linearize and transpose emit `StdTensorOp::Reverse`.
2. Seed concrete input shape metadata for transpose tests that need original extents.
3. Run:

```bash
cargo test -p tenferro-ops --lib -- structural_special_cases
```

Expected: fail until implementation is added.

## Task 3: Implement Structural Rules

**Files:**

- Modify: `tenferro-ops/src/ad/mod.rs`
- Modify: `tenferro-ops/src/ad/structural.rs`

**Steps:**

1. Add `linearize_slice(builder, tangent_in, config)` and dispatch `StdTensorOp::Slice`.
2. Add `transpose_slice(emitter, cotangent_out, inputs, config, ctx)`:
   - require concrete input extents,
   - emit `StdTensorOp::Pad` with `edge_padding_low = starts`, `interior_padding = strides - 1`, and high padding to restore the original input extent.
3. Add `transpose_pad(emitter, cotangent_out, inputs, config, ctx)`:
   - require concrete input extents and non-negative interior padding,
   - emit a `StdTensorOp::Slice` over the cotangent using the inverse padded positions,
   - emit a trailing zero-interior `StdTensorOp::Pad` only when forward negative edge padding cropped input elements.
4. Add `linearize_reverse(...)` and `transpose_reverse(...)`; both emit the same reverse op.
5. Keep all new panics explicit and specific for unsupported symbolic extents or invalid static configurations.

## Task 4: Verify Focused Tests

**Commands:**

```bash
cargo test -p tenferro-ops --lib -- structural_special_cases
cargo test -p tenferro --test ad -- grad_slice_weighted_sum_matches_finite_diff
cargo test -p tenferro --test ad -- grad_pad_weighted_sum_matches_finite_diff
cargo test -p tenferro --test ad -- grad_reverse_weighted_sum_matches_finite_diff
```

Expected: all pass.

## Task 5: Broader Verification And Commit

**Commands:**

```bash
cargo test -p tenferro-ops ad
cargo test -p tenferro --test ad
cargo fmt --all --check
cargo check -p tenferro-ops
cargo check -p tenferro
git diff --check
```

Commit:

```bash
git add docs/plans/2026-05-02-ad-structural-boundary-plan.md tenferro-ops/src tenferro/tests/ad.rs
git commit -m "fix: add structural ad boundary rules"
```
