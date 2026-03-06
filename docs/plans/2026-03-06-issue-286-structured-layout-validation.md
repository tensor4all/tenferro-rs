# Issue 286 Structured Layout Validation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reject incompatible `StructuredTensor` layouts in dynamic tensor merge paths so `DynAdTensor::axpby` and `DynAdTensor::compose_complex` fail explicitly instead of silently reinterpreting payloads.

**Architecture:** Keep the fix at the shared merge boundary. Add layout validation inside `merge_add_ad_tensors` before payload addition, then cover the public failure modes through focused regression tests. Do not add implicit alignment or dense fallback.

**Tech Stack:** Rust, `tenferro-dyadtensor`, cargo test, doc-tested public API crate

---

### Task 1: Add the failing regression tests

**Files:**
- Create: `extension/tenferro-dyadtensor/tests/structured_layout_validation_tests.rs`
- Reference: `extension/tenferro-dyadtensor/src/dyn_types.rs`

**Step 1: Write the failing test**

```rust
#[test]
fn axpby_rejects_diag_and_dense_vector_layout_mismatch() {
    // Build a diagonal 2x2 and a dense length-2 vector, then assert Err.
}

#[test]
fn compose_complex_rejects_diag_and_dense_vector_layout_mismatch() {
    // Build diagonal real part and dense imaginary part, then assert Err.
}

#[test]
fn axpby_rejects_same_dims_but_different_axis_classes() {
    // Build two structured tensors with dims [2, 2] but different axis_classes.
}

#[test]
fn axpby_accepts_matching_structured_layouts() {
    // Same diag layout on both sides should still succeed.
}
```

**Step 2: Run test to verify it fails**

Run: `CARGO_BUILD_JOBS=1 cargo test -p tenferro-dyadtensor --test structured_layout_validation_tests`

Expected: FAIL because the mismatch cases currently return `Ok(...)`.

**Step 3: Commit**

```bash
git add extension/tenferro-dyadtensor/tests/structured_layout_validation_tests.rs
git commit -m "test(dyadtensor): cover structured layout mismatch merges"
```

### Task 2: Implement shared layout validation in merge_add_ad_tensors

**Files:**
- Modify: `extension/tenferro-dyadtensor/src/dyn_types.rs`
- Test: `extension/tenferro-dyadtensor/tests/structured_layout_validation_tests.rs`

**Step 1: Write minimal implementation**

Add a small shared helper next to `merge_add_ad_tensors`, for example:

```rust
fn ensure_same_structured_layout<T: Scalar>(
    op_name: &'static str,
    lhs: &StructuredTensor<T>,
    rhs: &StructuredTensor<T>,
) -> Result<()> {
    if lhs.logical_dims() != rhs.logical_dims() {
        return Err(Error::InvalidAdTensor {
            message: format!(
                "{op_name} requires matching logical_dims, got lhs={:?}, rhs={:?}",
                lhs.logical_dims(),
                rhs.logical_dims(),
            ),
        });
    }
    if lhs.axis_classes() != rhs.axis_classes() {
        return Err(Error::InvalidAdTensor {
            message: format!(
                "{op_name} requires matching axis_classes, got lhs={:?}, rhs={:?}",
                lhs.axis_classes(),
                rhs.axis_classes(),
            ),
        });
    }
    Ok(())
}
```

Call it from `merge_add_ad_tensors` before:

- primal payload addition
- tangent payload addition when both tangents exist
- tangent passthrough branches when only one tangent exists, so the output tangent cannot inherit a mismatched layout silently

Use a stable operation label like `"tensor add merge"` in the error message.

**Step 2: Run targeted test to verify it passes**

Run: `CARGO_BUILD_JOBS=1 cargo test -p tenferro-dyadtensor --test structured_layout_validation_tests`

Expected: PASS

**Step 3: Run nearby crate tests**

Run: `CARGO_BUILD_JOBS=1 cargo test -p tenferro-dyadtensor dyn_types::tests::dyn_ad_tensor_compose_complex_roundtrip_forward`

Expected: PASS

**Step 4: Commit**

```bash
git add extension/tenferro-dyadtensor/src/dyn_types.rs extension/tenferro-dyadtensor/tests/structured_layout_validation_tests.rs
git commit -m "fix(dyadtensor): reject mixed structured layouts in merges"
```

### Task 3: Re-run focused validation and sanity-check docs/status

**Files:**
- Modify: none expected
- Verify: `extension/tenferro-dyadtensor/src/dyn_types.rs`

**Step 1: Run crate verification**

Run: `CARGO_BUILD_JOBS=1 cargo test -p tenferro-dyadtensor`

Expected: PASS

**Step 2: Confirm worktree is clean except intended commits**

Run: `git status --short --branch`

Expected: clean working tree on `issue-286-structured-layout-validation`

**Step 3: Commit**

```bash
# No new commit if there are no additional changes.
```
