# Remove Row-Major Tensor Surface Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove row-major tensor storage from the public and concrete tensor surface, leaving one contiguous column-major tensor layout.

**Architecture:** `Tensor` and `TypedTensor` become column-major-only owned dense tensors. CPU helpers construct column-major `StridedView`s directly, GPU upload no longer canonicalizes row-major host buffers, and docs/tests stop advertising row-major APIs. External row-major data conversion stays outside the public tensor API.

**Tech Stack:** Rust workspace crates `tenferro-tensor` and `tenferro`, strided-kernel views, CubeCL feature-gated tests/docs, repository docs under `docs/guides` and `docs/getting-started`.

---

### Task 1: Collapse Tensor Storage Metadata To Column-Major

**Files:**
- Modify: `tenferro-tensor/src/types.rs`
- Modify: `tenferro-tensor/src/types/accessors.rs`
- Modify: `tenferro-tensor/src/cpu/mod.rs`
- Test: `tenferro-tensor/src/tests/types_tests.rs`

**Step 1: Update tests first**

Replace row-major-specific tests in `tenferro-tensor/src/tests/types_tests.rs` with column-major-only coverage:

```rust
#[test]
fn tensor_owned_export_returns_column_major_buffer() {
    let tensor = Tensor::from_vec(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]);

    let (shape, data) = tensor.try_into_vec::<f64>().unwrap();

    assert_eq!(shape, vec![2, 2]);
    assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);
}
```

Also update accessor tests so they use only `TypedTensor::from_vec` / `Tensor::from_vec` and assert column-major offsets.

**Step 2: Run focused tests and verify failure**

Run:

```bash
cargo test -p tenferro-tensor types_tests
```

Expected: FAIL because `try_into_vec` does not exist yet and old row-major APIs still exist.

**Step 3: Remove runtime memory-order API**

In `tenferro-tensor/src/types.rs`:

- delete `MemoryOrder`;
- remove `order: MemoryOrder` from `TypedTensor<T>`;
- remove `row_major_strides`, `contiguous_strides`, `linear_offset_for_order`, `flat_to_multi_for_order`, and `reorder_contiguous` when no remaining caller needs them;
- simplify `TensorScalar` to expose only `into_tensor(shape, data)`;
- make every `TensorScalar` implementation construct via `TypedTensor::from_vec`;
- keep `TypedTensor::from_vec(shape, data)` as the only typed owned constructor;
- add `TypedTensor::try_into_vec(self) -> crate::Result<(Vec<usize>, Vec<T>)>`;
- add `Tensor::try_into_vec<T: TensorScalar>(self) -> crate::Result<(Vec<usize>, Vec<T>)>`;
- remove `from_vec_row_major`, `from_vec_col_major`, `from_vec_with_order`, `order`, `to_order`, `to_col_major`, `to_row_major`, and all `try_into_vec_*_major` / `try_into_vec_with_order` methods.

Use `op: "try_into_vec"` in new owned-export errors.

**Step 4: Simplify accessors**

In `tenferro-tensor/src/types/accessors.rs`, remove `MemoryOrder` imports and order-dispatch helpers. `linear_offset`, `linear_offset2`, and `linear_offset3` should use column-major formulas only:

```rust
fn linear_offset2(shape: &[usize], i: usize, j: usize) -> usize {
    i + shape[0] * j
}
```

Keep checked rank and bounds behavior unchanged.

**Step 5: Simplify CPU views**

In `tenferro-tensor/src/cpu/mod.rs`, replace `crate::contiguous_strides(&tensor.shape, tensor.order)` with `col_major_strides(&tensor.shape)`.

**Step 6: Run focused tests**

Run:

```bash
cargo test -p tenferro-tensor types_tests
cargo test -p tenferro-tensor cpu_tests
```

Expected: PASS after downstream compile fixes in later tasks.

**Step 7: Commit**

```bash
git add tenferro-tensor/src/types.rs tenferro-tensor/src/types/accessors.rs tenferro-tensor/src/cpu/mod.rs tenferro-tensor/src/tests/types_tests.rs
git commit -m "refactor(tensor): remove runtime memory order"
```

### Task 2: Fix CPU, GPU, And Linalg Call Sites

**Files:**
- Modify: `tenferro-tensor/src/cpu/backend.rs`
- Modify: `tenferro-tensor/src/cpu/indexing.rs`
- Modify: `tenferro-tensor/src/cpu/gemm/mod.rs`
- Modify: `tenferro-tensor/src/cpu/linalg/faer_linalg.rs`
- Modify: `tenferro-tensor/src/cpu/linalg/lapack_linalg/helpers.rs`
- Modify: `tenferro-tensor/src/cubecl/memory.rs`
- Modify: `tenferro-tensor/src/cubecl/dispatch.rs`
- Test: `tenferro-tensor/src/cubecl/tests/metadata_tests.rs`
- Test: `tenferro-tensor/src/tests/cpu_tests.rs`

**Step 1: Update tests first**

Remove row-major CPU comparison tests from `tenferro-tensor/src/tests/cpu_tests.rs`; keep the existing operation correctness tests that use column-major fixtures.

In `tenferro-tensor/src/cubecl/tests/metadata_tests.rs`, delete:

- `typed_tensor_binding_rejects_row_major_gpu_tensor`;
- `upload_canonicalizes_row_major_host_tensor_to_col_major`.

Keep buffer length and shape overflow tests.

**Step 2: Run compile check and verify failure list**

Run:

```bash
cargo test -p tenferro-tensor --no-run
```

Expected: FAIL at old `.to_col_major()`, `MemoryOrder::ColMajor`, and struct `order` fields.

**Step 3: Remove production conversions and fields**

Replace `.to_col_major()?` calls in CPU backend/indexing paths with direct cloned or borrowed tensors as appropriate. Remove `order: crate::MemoryOrder::ColMajor` from direct `TypedTensor` initializers in GEMM/linalg helpers.

In `tenferro-tensor/src/cubecl/memory.rs`:

- remove `Cow` usage and `canonical_host_tensor_for_upload`;
- upload host buffers directly;
- return downloaded tensors through `TypedTensor::from_vec`.

In `tenferro-tensor/src/cubecl/dispatch.rs`:

- remove `MemoryOrder` import;
- remove the `tensor.order != MemoryOrder::ColMajor` rejection from `typed_tensor_binding`;
- keep shape product and buffer length validation.

**Step 4: Run focused CPU verification**

Run:

```bash
cargo test -p tenferro-tensor --no-run
cargo test -p tenferro-tensor cpu_tests
```

Expected: PASS.

**Step 5: Run CubeCL metadata compile/test**

Run:

```bash
cargo test -p tenferro-tensor --features cubecl metadata_tests
```

Expected: PASS on CPU-only machines because these metadata tests do not require ignored GPU execution.

**Step 6: Commit**

```bash
git add tenferro-tensor/src/cpu tenferro-tensor/src/cubecl tenferro-tensor/src/tests/cpu_tests.rs
git commit -m "refactor(tensor): assume column-major backend tensors"
```

### Task 3: Update Public Facade, Integration Tests, And User Docs

**Files:**
- Modify: `tenferro/src/lib.rs`
- Modify: `tenferro/tests/symbolic_input.rs`
- Modify: `docs/guides/memory-order.md`
- Modify: `docs/guides/performance.md`
- Modify: `docs/guides/troubleshooting.md`
- Modify: `docs/getting-started/pytorch-jax-mapping.md`
- Modify: `docs/guides/eager-operations.md`

**Step 1: Update tests first**

In `tenferro/tests/symbolic_input.rs`, replace `row_major_symbolic_input_uses_logical_shape_and_values` with a column-major symbolic input test:

```rust
#[test]
fn matrix_symbolic_input_uses_column_major_values() {
    let x = TracedTensor::input_symbolic_shape(DType::F64, 2);
    let mut y = &x + &x;

    let mut engine = Engine::new(CpuBackend::new());
    let bound = Tensor::from_vec(vec![2, 3], vec![1.0_f64, 4.0, 2.0, 5.0, 3.0, 6.0]);
    let out = y
        .eval_with_inputs(&mut engine, &[(&x, &bound)])
        .expect("eval_with_inputs");

    assert_eq!(out.shape(), &[2, 3]);
    assert_eq!(f64_data(out), &[2.0, 8.0, 4.0, 10.0, 6.0, 12.0]);
}
```

**Step 2: Remove facade export**

In `tenferro/src/lib.rs`, remove `MemoryOrder` from the `pub use tenferro_tensor::{...}` list.

**Step 3: Rewrite user-facing docs**

Update docs to say tenferro flat buffers are column-major. Do not mention removed row-major APIs.

For `docs/guides/memory-order.md`, rewrite examples to use:

```rust
use tenferro::Tensor;

let tensor = Tensor::from_vec(vec![2, 3], vec![1.0_f64, 4.0, 2.0, 5.0, 3.0, 6.0]);
assert_eq!(tensor.shape(), &[2, 3]);
assert_eq!(tensor.as_slice::<f64>().unwrap(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
```

Mention that NumPy/PyTorch row-major flat buffers must be converted before construction. Keep imports as `use tenferro::{...}` in user docs.

**Step 4: Search for stale public claims**

Run:

```bash
rg -n "MemoryOrder|RowMajor|from_vec_row_major|from_vec_col_major|from_vec_with_order|to_row_major|to_col_major|to_order|try_into_vec_row_major|try_into_vec_col_major|try_into_vec_with_order|row-major|row major" README.md tenferro tenferro-tensor docs/guides docs/getting-started
```

Expected: only intentional plain-English statements about external row-major formats remain; no removed API names remain in active docs or source.

**Step 5: Run public-facing focused tests**

Run:

```bash
cargo test -p tenferro symbolic_input
cargo test -p tenferro-tensor --doc
```

Expected: PASS.

**Step 6: Commit**

```bash
git add tenferro/src/lib.rs tenferro/tests/symbolic_input.rs docs/guides docs/getting-started
git commit -m "docs: document column-major-only tensor buffers"
```

### Task 4: Workspace Verification And Cleanup

**Files:**
- Review all changed files from Tasks 1-3.

**Step 1: Format**

Run:

```bash
cargo fmt --all
cargo fmt --all --check
```

Expected: PASS.

**Step 2: Run focused no-run checks**

Run:

```bash
cargo test -p tenferro-tensor --no-run
cargo test -p tenferro --no-run
```

Expected: PASS.

**Step 3: Run focused tests**

Run:

```bash
cargo test -p tenferro-tensor types_tests
cargo test -p tenferro-tensor cpu_tests
cargo test -p tenferro symbolic_input
```

Expected: PASS.

**Step 4: Run docs checks affected by public API removal**

Run:

```bash
cargo test --doc -p tenferro-tensor
cargo test --doc -p tenferro
python3 scripts/check-docs-site.py
```

Expected: PASS.

**Step 5: Final stale API search**

Run:

```bash
rg -n "MemoryOrder|RowMajor|from_vec_row_major|from_vec_col_major|from_vec_with_order|to_row_major|to_col_major|to_order|try_into_vec_row_major|try_into_vec_col_major|try_into_vec_with_order" tenferro tenferro-tensor docs/guides docs/getting-started README.md
```

Expected: no matches.

**Step 6: Commit verification fixes if needed**

If formatting or docs changes are produced:

```bash
git add .
git commit -m "chore: finish row-major removal cleanup"
```
