# Owned Contiguous Memory Order Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add CPU owned-contiguous row/column memory-order metadata and zero-copy owned buffer import/export APIs without adding `ndarray` or strided-view support.

**Architecture:** `MemoryOrder` lives only on concrete `TypedTensor<T>` values. `Tensor`, `EagerTensor`, graph metadata, and operation axis semantics remain structurally unchanged. CPU kernels either read using derived contiguous strides from `shape + order` or explicitly materialize row-major inputs to column-major for column-major-only paths.

**Tech Stack:** Rust 2021, `tenferro-tensor`, `tenferro` facade re-exports, existing `strided-kernel` CPU helpers, Cargo tests and doctests.

---

### Task 1: Memory Order Metadata and Owned API

**Files:**
- Modify: `tenferro-tensor/src/types.rs`
- Modify: `tenferro-tensor/src/tests/types_tests.rs`
- Modify: `tenferro/src/lib.rs`

**Step 1: Write failing tests**

Add tests to `tenferro-tensor/src/tests/types_tests.rs`:

```rust
#[test]
fn typed_tensor_explicit_memory_order_constructors_set_order() {
    let col = TypedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]);
    let row = TypedTensor::from_vec_row_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]);

    assert_eq!(col.order(), MemoryOrder::ColMajor);
    assert_eq!(row.order(), MemoryOrder::RowMajor);
    assert_eq!(TypedTensor::from_vec(vec![1], vec![1.0_f64]).order(), MemoryOrder::ColMajor);
}

#[test]
fn tensor_owned_export_is_zero_copy_only_for_matching_order() {
    let data = vec![1.0_f64, 2.0, 3.0, 4.0];
    let ptr = data.as_ptr();
    let tensor = Tensor::from_vec_row_major(vec![2, 2], data);

    let (shape, out) = tensor.try_into_vec_row_major::<f64>().unwrap();

    assert_eq!(shape, vec![2, 2]);
    assert_eq!(out.as_ptr(), ptr);

    let mismatch = Tensor::from_vec_row_major(vec![2], vec![1.0_f64])
        .try_into_vec_col_major::<f64>()
        .unwrap_err();
    assert!(mismatch.to_string().contains("memory order"));
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p tenferro-tensor types_tests -- typed_tensor_explicit_memory_order_constructors_set_order tensor_owned_export_is_zero_copy_only_for_matching_order`

Expected: FAIL because `MemoryOrder`, row-major constructors, `order()`, and typed export methods are missing.

**Step 3: Implement minimal API**

In `types.rs`, add `MemoryOrder`, add `order: MemoryOrder` to `TypedTensor<T>`, set default order to `ColMajor`, add explicit row/col-major constructors, add `Tensor::from_vec_row_major`, `Tensor::from_vec_col_major`, `Tensor::from_vec_with_order`, and typed `try_into_vec_*` methods.

Update `tenferro/src/lib.rs` to re-export `MemoryOrder`.

**Step 4: Run tests to verify green**

Run: `cargo test -p tenferro-tensor types_tests -- typed_tensor_explicit_memory_order_constructors_set_order tensor_owned_export_is_zero_copy_only_for_matching_order`

Expected: PASS.

**Step 5: Commit**

```bash
git add tenferro-tensor/src/types.rs tenferro-tensor/src/tests/types_tests.rs tenferro/src/lib.rs
git commit -m "Add owned tensor memory order metadata"
```

### Task 2: Explicit Layout Conversion and CPU Read Semantics

**Files:**
- Modify: `tenferro-tensor/src/types.rs`
- Modify: `tenferro-tensor/src/cpu/mod.rs`
- Modify: `tenferro-tensor/src/cpu/structural.rs`
- Modify: `tenferro-tensor/src/cpu/gemm/mod.rs`
- Modify: `tenferro-tensor/src/tests/types_tests.rs`
- Test: `tenferro-tensor/src/tests/cpu_tests.rs`

**Step 1: Write failing tests**

Add tests that prove `to_col_major`, `to_row_major`, `linear_offset`, elementwise CPU ops, and `dot_general` preserve logical semantics for row-major inputs.

Use a 2x3 buffer where row-major and column-major logical indexing differ:

```rust
let row = Tensor::from_vec_row_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
assert_eq!(row.to_col_major().unwrap().as_slice::<f64>().unwrap(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p tenferro-tensor types_tests cpu_tests -- row_major`

Expected: FAIL because conversion and CPU row-major read paths are not implemented.

**Step 3: Implement minimal conversion and stride derivation**

Add `row_major_strides`, `contiguous_strides`, and layout conversion helpers derived from `shape + order`. Make `TypedTensor::linear_offset`, `cpu::typed_view`, and `cpu::structural::host_view` use derived strides. Make GEMM analysis use derived strides for inputs while keeping outputs column-major.

**Step 4: Run tests to verify green**

Run: `cargo test -p tenferro-tensor types_tests cpu_tests -- row_major`

Expected: PASS.

**Step 5: Commit**

```bash
git add tenferro-tensor/src/types.rs tenferro-tensor/src/cpu/mod.rs tenferro-tensor/src/cpu/structural.rs tenferro-tensor/src/cpu/gemm/mod.rs tenferro-tensor/src/tests/types_tests.rs tenferro-tensor/src/tests/cpu_tests.rs
git commit -m "Handle contiguous row-major CPU tensors"
```

### Task 3: Graph and Eager Boundary Coverage

**Files:**
- Test: `tenferro/tests/primitive_ops.rs`
- Test: `tenferro/src/eager.rs` or existing eager integration tests
- Modify only if required: `tenferro/src/metadata.rs`, `tenferro/src/traced.rs`

**Step 1: Write failing tests**

Add a traced `eval_with_inputs` test that binds a row-major `Tensor` and checks logical output. Add an eager test that imports a row-major tensor through `EagerTensor::from_tensor_in` and checks a simple operation. The tests should assert graph shape metadata remains logical and output tensors are usable.

**Step 2: Run tests to verify status**

Run: `cargo test -p tenferro row_major`

Expected: PASS if lower layers are sufficient; otherwise FAIL with the specific boundary gap.

**Step 3: Implement only boundary fixes required by the tests**

Do not add layout to graph metadata. If failures appear, fix concrete tensor handling at import/evaluation boundaries without changing `TensorMeta`, `StdTensorOp`, or axis numbering.

**Step 4: Run tests to verify green**

Run: `cargo test -p tenferro row_major`

Expected: PASS.

**Step 5: Commit**

```bash
git add tenferro/tests tenferro/src
git commit -m "Cover row-major tensors at graph boundaries"
```

### Task 4: Docs, Dependency Cleanup, and Verification

**Files:**
- Modify: `Cargo.toml`
- Modify: rustdoc comments in `tenferro-tensor/src/types.rs`
- Modify if needed: `docs/plans/2026-05-12-owned-contiguous-memory-order-design.md`

**Step 1: Remove unused ndarray workspace dependency**

Delete `ndarray = "0.17.2"` from root `Cargo.toml` unless another workspace crate now uses it.

**Step 2: Add rustdoc examples**

Every new public enum and method gets a compiling `# Examples` block using only `Vec + shape`, not `ndarray`.

**Step 3: Run targeted verification**

Run:

```bash
cargo fmt --all --check
cargo test -p tenferro-tensor types_tests cpu_tests -- row_major
cargo test -p tenferro row_major
cargo test --doc -p tenferro-tensor
cargo test --doc -p tenferro
```

Expected: all commands exit 0.

**Step 4: Commit**

```bash
git add Cargo.toml tenferro-tensor/src/types.rs docs/plans/2026-05-12-owned-contiguous-memory-order-design.md
git commit -m "Document memory order API examples"
```
