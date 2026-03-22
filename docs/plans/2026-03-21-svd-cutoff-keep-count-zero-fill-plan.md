# SVD Cutoff Keep-Count Zero-Fill Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a reusable tensor-native trailing zero-fill helper driven by batch-local keep counts, then rewrite `tenferro-linalg::svd(..., cutoff)` to use it without host payload extraction.

**Architecture:** The execution primitive lives in `tenferro-tensor` and delegates to Layer 0 CUDA runtime helpers in `tenferro-device` for GPU tensors. `tenferro-linalg` only adds thin SVD-specific orchestration over that helper. Keep counts are represented as a real-valued tensor with batch shape only, and the helper must work for real or complex payload tensors.

**Tech Stack:** Rust, `tenferro-device`, `tenferro-tensor`, `tenferro-linalg`, `cudarc`, runtime-loaded CUDA kernels, `cargo test`, `cargo fmt`.

---

### Task 1: Add CPU tensor semantics for trailing zero-fill by keep counts

**Files:**
- Modify: `tenferro-tensor/src/tensor/data_ops.rs`
- Modify: `tenferro-tensor/src/tests/mod.rs`
- Create or Modify: `tenferro-tensor/src/tests/structural.rs`

**Step 1: Write the failing tests**

Add tensor-level CPU tests:

```rust
#[test]
fn zero_trailing_by_counts_cpu_zero_fills_after_keep_count_real_payload() {
    // 2x3 matrix batch with keep_counts=[2]
    // axis=1 zero-fills the last column only.
}

#[test]
fn zero_trailing_by_counts_cpu_zero_fills_complex_payload() {
    // complex payload, real keep_counts, same fixed-shape semantics.
}

#[test]
fn zero_trailing_by_counts_cpu_rejects_invalid_keep_counts() {
    // shape mismatch, negative count, non-integer count, count > axis dim.
}
```

**Step 2: Run tests to verify they fail**

Run:

```bash
cargo test -p tenferro-tensor zero_trailing_by_counts_cpu_zero_fills_after_keep_count_real_payload -- --exact
cargo test -p tenferro-tensor zero_trailing_by_counts_cpu_zero_fills_complex_payload -- --exact
cargo test -p tenferro-tensor zero_trailing_by_counts_cpu_rejects_invalid_keep_counts -- --exact
```

Expected: FAIL because `Tensor::zero_trailing_by_counts` does not exist yet.

**Step 3: Write minimal implementation**

Add:

```rust
pub fn zero_trailing_by_counts<R>(&self, keep_counts: &Tensor<R>, axis: usize) -> Result<Tensor<T>>
where
    T: Scalar,
    R: Scalar,
```

CPU path requirements:

- validate batch-shape compatibility
- validate `axis < self.ndim()`
- validate `keep_counts` values are integer-valued and in range
- materialize an out-of-place column-major output
- copy only the kept region, leave the trailing region zero

Do not add any GPU code in this task.

**Step 4: Run tests to verify they pass**

Run:

```bash
cargo test -p tenferro-tensor zero_trailing_by_counts_cpu_zero_fills_after_keep_count_real_payload -- --exact
cargo test -p tenferro-tensor zero_trailing_by_counts_cpu_zero_fills_complex_payload -- --exact
cargo test -p tenferro-tensor zero_trailing_by_counts_cpu_rejects_invalid_keep_counts -- --exact
cargo test -p tenferro-tensor --lib
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tenferro-tensor/src/tensor/data_ops.rs tenferro-tensor/src/tests/mod.rs tenferro-tensor/src/tests/structural.rs
git commit -m "feat: add cpu trailing zero-fill by keep counts"
```

### Task 2: Add Layer 0 CUDA trailing zero-fill kernel

**Files:**
- Modify: `tenferro-device/src/cuda/runtime.rs`
- Modify: `tenferro-device/src/cuda/tests/mod.rs`

**Step 1: Write the failing tests**

Add Layer 0 CUDA tests:

```rust
#[test]
fn cuda_runtime_zero_trailing_by_counts_f64_matches_host_reference() {
    // real payload, real counts, axis=1
}

#[test]
fn cuda_runtime_zero_trailing_by_counts_complex64_matches_host_reference() {
    // complex payload, real counts, axis=0
}
```

**Step 2: Run tests to verify they fail**

Run:

```bash
TENFERRO_TEST_CUDA=1 cargo test -p tenferro-device --features cuda cuda::tests::cuda_runtime_zero_trailing_by_counts_f64_matches_host_reference -- --exact
TENFERRO_TEST_CUDA=1 cargo test -p tenferro-device --features cuda cuda::tests::cuda_runtime_zero_trailing_by_counts_complex64_matches_host_reference -- --exact
```

Expected: FAIL because the Layer 0 runtime has no such kernel or entrypoint.

**Step 3: Write minimal implementation**

Add a new Layer 0 CUDA kernel/entrypoints that:

- accept payload pointer, payload dims/strides/offset
- accept keep-count pointer with batch dims/strides/offset
- accept target axis
- zero elements where `coord_axis >= keep_count`
- preserve existing values otherwise

Support:

- payload `f32`, `f64`, `Complex32`, `Complex64`
- keep-counts `f32` and `f64`

Host-visible validation should reject non-integer or out-of-range keep counts before launch.

**Step 4: Run tests to verify they pass**

Run:

```bash
TENFERRO_TEST_CUDA=1 cargo test -p tenferro-device --features cuda cuda::tests::cuda_runtime_zero_trailing_by_counts_f64_matches_host_reference -- --exact
TENFERRO_TEST_CUDA=1 cargo test -p tenferro-device --features cuda cuda::tests::cuda_runtime_zero_trailing_by_counts_complex64_matches_host_reference -- --exact
TENFERRO_TEST_CUDA=1 cargo test -p tenferro-device --features cuda --lib
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tenferro-device/src/cuda/runtime.rs tenferro-device/src/cuda/tests/mod.rs
git commit -m "feat: add cuda trailing zero-fill kernel"
```

### Task 3: Wire tensor CUDA path through Layer 0

**Files:**
- Modify: `tenferro-tensor/src/cuda_runtime.rs`
- Modify: `tenferro-tensor/src/tensor/data_ops.rs`
- Modify: `tenferro-tensor/src/tests/cuda.rs`

**Step 1: Write the failing tests**

Add tensor-level CUDA tests:

```rust
#[test]
fn gpu_zero_trailing_by_counts_matches_cpu_for_real_payload() {
    // same logical tensor, CPU and CUDA outputs must match.
}

#[test]
fn gpu_zero_trailing_by_counts_matches_cpu_for_complex_payload() {
    // complex payload, real counts, no host payload fallback.
}
```

**Step 2: Run tests to verify they fail**

Run:

```bash
cargo test -p tenferro-tensor --features cuda gpu_zero_trailing_by_counts_matches_cpu_for_real_payload -- --exact
cargo test -p tenferro-tensor --features cuda gpu_zero_trailing_by_counts_matches_cpu_for_complex_payload -- --exact
```

Expected: FAIL because `Tensor::zero_trailing_by_counts` is CPU-only.

**Step 3: Write minimal implementation**

Route GPU tensors through `tenferro-device::cuda::runtime` from
`tenferro-tensor/src/cuda_runtime.rs`.

Requirements:

- no direct raw CUDA ownership in `tenferro-tensor`
- same-device validation
- same out-of-place semantics as CPU path
- preserve conjugation/materialization invariants already used elsewhere in tensor GPU paths

**Step 4: Run tests to verify they pass**

Run:

```bash
cargo test -p tenferro-tensor --features cuda gpu_zero_trailing_by_counts_matches_cpu_for_real_payload -- --exact
cargo test -p tenferro-tensor --features cuda gpu_zero_trailing_by_counts_matches_cpu_for_complex_payload -- --exact
cargo test -p tenferro-tensor --features cuda --lib
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tenferro-tensor/src/cuda_runtime.rs tenferro-tensor/src/tensor/data_ops.rs tenferro-tensor/src/tests/cuda.rs
git commit -m "feat: wire tensor keep-count zero-fill to shared cuda runtime"
```

### Task 4: Rewrite `svd(..., cutoff)` to use keep-count zero-fill

**Files:**
- Modify: `tenferro-linalg/src/backend/tensor_helpers.rs`
- Modify: `tenferro-linalg/src/backend/tensor_helpers/tests/mod.rs`
- Modify: `tenferro-linalg/src/primal/decompositions.rs`
- Modify: `tenferro-linalg/src/tests/mod.rs`
- Modify: `tenferro-linalg/src/tests/runtime_capability.rs`

**Step 1: Write the failing tests**

Add:

```rust
#[test]
fn zero_trailing_by_counts_linalg_wrapper_matches_tensor_helper() {
    // thin wrapper test for axis/batch conventions.
}

#[test]
fn svd_cutoff_fixed_shape_zero_fill_semantics_hold() {
    // singular values below cutoff are zeroed, shapes remain fixed.
}
```

If feasible under local CUDA availability, add a CUDA-gated version that checks
the same public semantics on GPU-resident inputs.

**Step 2: Run tests to verify they fail**

Run:

```bash
cargo test -p tenferro-linalg zero_trailing_by_counts_linalg_wrapper_matches_tensor_helper -- --exact
cargo test -p tenferro-linalg svd_cutoff_fixed_shape_zero_fill_semantics_hold -- --exact
```

Expected: FAIL because `svd(..., cutoff)` still rebuilds tensors through host slices.

**Step 3: Write minimal implementation**

In `tenferro-linalg`:

- add a thin wrapper over `Tensor::zero_trailing_by_counts`
- keep `max_rank` on the existing `narrow` path
- derive `keep_counts` from `s` after `max_rank`
- apply trailing zero-fill to:
  - `u` on axis 1
  - `s` on axis 0
  - `vt` on axis 0

Do not retain the existing host rebuild path for CUDA.

If deriving `keep_counts` still cannot be made generic at this point, stop and
record the exact missing tensor primitive instead of adding a fallback.

**Step 4: Run tests to verify they pass**

Run:

```bash
cargo test -p tenferro-linalg zero_trailing_by_counts_linalg_wrapper_matches_tensor_helper -- --exact
cargo test -p tenferro-linalg svd_cutoff_fixed_shape_zero_fill_semantics_hold -- --exact
cargo test -p tenferro-linalg --lib
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tenferro-linalg/src/backend/tensor_helpers.rs tenferro-linalg/src/backend/tensor_helpers/tests/mod.rs tenferro-linalg/src/primal/decompositions.rs tenferro-linalg/src/tests/mod.rs tenferro-linalg/src/tests/runtime_capability.rs
git commit -m "refactor: make svd cutoff use tensor-native keep-count zero-fill"
```

### Task 5: Re-verify foundation invariants and document the stop point

**Files:**
- Modify: `docs/design/**` only if public behavior docs need updating

**Step 1: Run focused verification**

Run:

```bash
cargo fmt --all --check
TENFERRO_TEST_CUDA=1 cargo test -p tenferro-device --features cuda --lib
cargo test -p tenferro-tensor --features cuda --lib
cargo test -p tenferro-linalg --lib
```

Expected: PASS.

**Step 2: Run broader verification as needed**

Run:

```bash
cargo doc -p tenferro-device --features cuda --no-deps
cargo doc -p tenferro-tensor --features cuda --no-deps
cargo doc -p tenferro-linalg --no-deps
```

Expected: PASS.

**Step 3: Document any remaining blocker**

If `pinv`, `matrix_rank`, or AD-layer users still cannot reuse the helper
without another substrate, stop there and record the exact missing primitive.

**Step 4: Commit**

```bash
git add docs/design
git commit -m "docs: record keep-count zero-fill foundation status"
```
