# Primitive DType And Shape Semantics Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the narrow primitive dtype and scalar-size bugs that can be corrected without a backend trait or shape-expression redesign.

**Architecture:** Keep semantic decisions in the `tenferro` facade layer, because the affected behavior is traced metadata and host-side scalar extraction. Add one shared crate-private scalar helper module so traced scaling, eager DynamicTruncate, and compiled DynamicTruncate use the same rounding and scalar validation policy. Do not expand raw backend Add/Mul broadcasting or conversion semantics in this step.

**Tech Stack:** Rust, `tenferro` traced/eager execution, `tenferro_tensor` CPU backend tests, cargo test.

---

## Scope For This Step

Implement:

- `DynamicTruncate` accepts scalar I64 sizes in eager and compiled execution.
- `DynamicTruncate` rejects non-scalar size tensors with `Err`, not indexing the first element.
- `scale_real` and linalg scalar helpers round real-to-I64 constants instead of truncating.
- `eig` metadata maps I64 input to C64, not I64.
- Add/Mul shape inference stops returning the LHS shape for unsupported non-scalar broadcast shapes.

Defer:

- #775 exact `DynamicTruncate` compiled shape inference. `DimExpr` currently represents dimension expressions only, not scalar tensor values, so the truncated axis cannot be expressed without a scalar-value shape expression or constant propagation.
- #776 full non-scalar Add/Mul raw-op broadcasting. Public traced Add/Mul already inserts `BroadcastInDim`; raw CPU and CubeCL Add/Mul only support equal shapes plus scalar CPU cases.
- #780 conversion overflow/narrowing semantics. This needs a coherent cross-backend conversion policy and test migration.
- #784 SVD/Eigh `eps` in primal execution. `eps` is currently consumed by AD linearization; plumbing it into primal backend execution requires changing `TensorBackend::svd/eigh`.

## Task 1: Add Failing Tests For Narrow Fixes

**Files:**

- Modify: `tenferro/tests/dynamic_truncate.rs`
- Modify: `tenferro/tests/eager_exec.rs`
- Modify: `tenferro/tests/primitive_ops.rs`
- Modify: `tenferro/tests/dtype_propagation.rs`
- Modify: `tenferro/tests/shape_inference.rs`

**Steps:**

1. Add a public traced `DynamicTruncate` test with an I64 scalar size.
2. Add eager execution tests for I64 scalar size and vector-size rejection.
3. Add an I64 `scale_real(2.7)` test expecting multiplication by `3`.
4. Add dtype inference tests showing I64 `Eig` metadata returns C64 and traced `eig` on I64 has C64 output metadata.
5. Add shape inference tests asserting Add/Mul reject non-scalar broadcast metadata until raw runtime support exists.
6. Run the focused tests and verify failures.

## Task 2: Shared Scalar Semantics

**Files:**

- Create: `tenferro/src/scalar_semantics.rs`
- Modify: `tenferro/src/lib.rs`
- Modify: `tenferro/src/exec.rs`
- Modify: `tenferro/src/eager_exec.rs`
- Modify: `tenferro/src/traced.rs`
- Modify: `tenferro/src/linalg_api.rs`

**Steps:**

1. Add `round_real_to_i64(value: f64) -> i64` using the DynamicTruncate policy: finite values round to nearest integer, non-finite values become `0`.
2. Add `dynamic_truncate_size(size_tensor: &Tensor, axis_extent: usize) -> Result<usize>` that accepts F64, F32, and I64 scalar tensors only, then rounds and clamps to `[0, axis_extent]`.
3. Replace duplicated DynamicTruncate scalar extraction in eager and compiled execution with the helper.
4. Replace real-to-I64 casts in `TracedTensor::scale_real` and `linalg_api::scalar_real` with `round_real_to_i64`.

## Task 3: Metadata Fixes

**Files:**

- Modify: `tenferro/src/shape_infer.rs`
- Modify: `tenferro/src/linalg_api.rs`

**Steps:**

1. Change `infer_output_dtype(StdTensorOp::Eig { input_dtype: DType::I64 })` to return `DType::C64`.
2. Change `eig_output_dtype(DType::I64)` to return `DType::C64`.
3. Replace `same_or_scalar_broadcast_shape` with a helper that returns equal shape or scalar-broadcast shape and panics for unsupported non-scalar broadcast metadata instead of silently returning LHS.

## Task 4: Verify And Commit

**Commands:**

```bash
cargo test -p tenferro dynamic_truncate
cargo test -p tenferro primitive_ops
cargo test -p tenferro dtype_propagation
cargo test -p tenferro shape_inference
cargo fmt --all --check
cargo check -p tenferro
```

Commit:

```bash
git add docs/plans/2026-05-02-primitive-dtype-shape-plan.md tenferro/src tenferro/tests
git commit -m "fix: align primitive dtype and scalar semantics"
```
