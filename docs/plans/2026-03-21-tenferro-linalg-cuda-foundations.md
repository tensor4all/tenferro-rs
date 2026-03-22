# tenferro-linalg CUDA Foundation-First Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the reusable GPU-generic tensor and linalg substrate that `tenferro-linalg` needs before adding more CUDA linalg kernels beyond the already-wired `solve` path.

**Architecture:** Follow the PyTorch layering, not ad hoc per-op hacks. Tensor layout materialization, conjugation resolution, status propagation, mask/select logic, and structural matrix cleanup belong in reusable substrate layers (`tenferro-tensor`, `tenferro-prims`, `tenferro-linalg-prims`, and small shared `tenferro-linalg` helpers), not inside individual CUDA kernels. `tenferro-linalg` must stay CPU/GPU generic and must not insert normal-path GPU→CPU tensor-data fallbacks.

**Tech Stack:** Rust, `tenferro-tensor`, `tenferro-prims`, `tenferro-linalg-prims`, `tenferro-linalg`, `cudarc`, runtime-loaded CUDA libraries, PyTorch-aligned linalg semantics, `cargo test`, `cargo fmt`, real-GPU tests gated by `TENFERRO_TEST_CUDA=1`.

---

## Scope and stop point

This plan intentionally stops at the foundation layer. Do **not** start new CUDA implementations of `qr`, `cholesky`, `thin_svd`, `svdvals`, `lu_factor`, or `eigen*` until all eight tasks below are merged.

Assume these two substrate pieces already exist and should be reused, not redesigned:

- `clone_batched_column_major` / `copy_batched_column_major` in `tenferro-linalg-prims/src/backend/linalg_utils.rs`
- GPU `tril` / `triu` in `tenferro-tensor/src/tensor/data_ops.rs` and `tenferro-tensor/src/cuda_runtime.rs`

The hard rules for every task in this plan are:

- no ad hoc GPU→CPU transfer of input/output tensor payloads
- host-visible control flow may inspect only minimal status like `info`
- new helper layers must be reusable by more than one linalg op
- if a path is still not generic after a task, keep CUDA capability false rather than adding a fallback

### Task 1: Make `Tensor::contiguous()` GPU-capable

**Files:**
- Modify: `tenferro-tensor/src/tensor/data_ops.rs`
- Modify: `tenferro-tensor/src/cuda_runtime.rs`
- Modify: `tenferro-tensor/src/tests/cuda.rs`

**Step 1: Write the failing test**

Add a CUDA regression in `tenferro-tensor/src/tests/cuda.rs`:

```rust
#[test]
fn gpu_contiguous_matches_cpu_for_strided_views_when_cuda_is_available() {
    if !cuda_device_zero_is_available() {
        return;
    }

    let data: Vec<f32> = (1..=24).map(|value| value as f32).collect();
    let base = Tensor::<f32>::from_slice(&data, &[2, 3, 4], MemoryOrder::ColumnMajor).unwrap();
    let view = base.permute(&[2, 0, 1]).unwrap();
    let gpu = view
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();

    let got = gpu
        .contiguous(MemoryOrder::ColumnMajor)
        .to_memory_space_async(LogicalMemorySpace::MainMemory)
        .unwrap();
    let expected = view.contiguous(MemoryOrder::ColumnMajor);

    assert_eq!(got.buffer().as_slice(), expected.buffer().as_slice());
}
```

**Step 2: Run test to verify it fails**

Run:

```bash
cargo test -p tenferro-tensor --features cuda gpu_contiguous_matches_cpu_for_strided_views_when_cuda_is_available -- --exact
```

Expected: FAIL because `Tensor::contiguous()` still calls `cpu_backed_slice_or_panic("contiguous")` on GPU-backed tensors.

**Step 3: Write minimal implementation**

In `tenferro-tensor/src/tensor/data_ops.rs`, branch `Tensor::contiguous()` for GPU tensors before the CPU slice path:

```rust
#[cfg(feature = "cuda")]
if matches!(self.logical_memory_space, LogicalMemorySpace::GpuMemory { .. }) {
    return crate::cuda_runtime::contiguous_tensor(self, order)
        .unwrap_or_else(|err| panic!("contiguous: GPU materialization failed: {err}"));
}
```

In `tenferro-tensor/src/cuda_runtime.rs`, add `contiguous_tensor<T: Scalar>(...) -> Result<Tensor<T>>` that:

- preserves dims, logical memory space, preferred device, and non-conjugated state
- allocates a contiguous output buffer on the same GPU
- reuses the existing CUDA-side contiguous copy machinery rather than extracting a host slice
- returns a tensor with contiguous strides and offset zero

**Step 4: Run tests to verify they pass**

Run:

```bash
cargo test -p tenferro-tensor --features cuda gpu_contiguous_matches_cpu_for_strided_views_when_cuda_is_available -- --exact
cargo test -p tenferro-tensor --features cuda gpu_round_trip_preserves_view_layout_and_values_when_cuda_is_available -- --exact
cargo test -p tenferro-prims --features cuda cuda_make_contiguous_smoke_runs_on_device_tensors_when_runtime_is_available -- --exact
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tenferro-tensor/src/tensor/data_ops.rs tenferro-tensor/src/cuda_runtime.rs tenferro-tensor/src/tests/cuda.rs
git commit -m "feat: add gpu tensor contiguous materialization"
```

### Task 2: Replace CUDA `resolve_conj()` host fallback with device-side materialization

**Files:**
- Modify: `tenferro-prims/src/cuda/mod.rs`
- Modify: `tenferro-prims/src/cuda/tests/mod.rs`
- Modify: `tenferro-prims/src/gpu_stubs.rs`

**Step 1: Write the failing test**

Add a CUDA regression in `tenferro-prims/src/cuda/tests/mod.rs`:

```rust
#[test]
fn cuda_resolve_conj_keeps_tensor_on_device_and_matches_cpu() {
    let Some(path) = available_cutensor_library_path() else {
        return;
    };
    if !cuda_device_zero_is_available() {
        return;
    }

    use num_complex::Complex64;

    let (_backend, mut ctx) = CudaBackend::load(path).unwrap();
    let cpu = Tensor::from_slice(
        &[
            Complex64::new(1.0, 2.0),
            Complex64::new(3.0, -4.0),
            Complex64::new(-5.0, 6.0),
            Complex64::new(7.0, 8.0),
        ],
        &[2, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let gpu = cpu
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap()
        .conj();

    let resolved = CudaBackend::resolve_conj(&mut ctx, &gpu);
    assert_eq!(resolved.logical_memory_space(), LogicalMemorySpace::GpuMemory { device_id: 0 });

    let round_trip = resolved
        .to_memory_space_async(LogicalMemorySpace::MainMemory)
        .unwrap();
    let expected = cpu.conj().contiguous(MemoryOrder::ColumnMajor);
    assert_eq!(round_trip.buffer().as_slice(), expected.buffer().as_slice());
}
```

**Step 2: Run test to verify it fails**

Run:

```bash
cargo test -p tenferro-prims --features cuda cuda_resolve_conj_keeps_tensor_on_device_and_matches_cpu -- --exact
```

Expected: FAIL or panic because the current implementation still goes through `contiguous().buffer().as_slice()` on the host.

**Step 3: Write minimal implementation**

In `tenferro-prims/src/cuda/mod.rs`, replace the host fallback with device-side materialization:

```rust
pub fn resolve_conj<T: Scalar + Conjugate>(
    ctx: &mut CudaContext,
    src: &Tensor<T>,
) -> Tensor<T> {
    if !src.is_conjugated() {
        return src.clone();
    }
    super::cuda::resolve_conj_tensor(ctx, src).unwrap_or_else(|_| src.clone())
}
```

Add a private helper that:

- first normalizes to contiguous GPU layout using Task 1
- launches a device-side conjugation path over the contiguous buffer
- returns a non-conjugated GPU tensor with the same values as `src.resolve_conj()` would on CPU

Keep `gpu_stubs.rs` surface aligned so the stub build keeps compiling.

**Step 4: Run tests to verify they pass**

Run:

```bash
cargo test -p tenferro-prims --features cuda cuda_resolve_conj_keeps_tensor_on_device_and_matches_cpu -- --exact
cargo test -p tenferro-prims --features cuda cuda_backend_feature_surface_matches_family_contracts -- --exact
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tenferro-prims/src/cuda/mod.rs tenferro-prims/src/cuda/tests/mod.rs tenferro-prims/src/gpu_stubs.rs
git commit -m "feat: resolve conjugated cuda tensors on device"
```

### Task 3: Make `_ex` result status truthful and per-batch

**Files:**
- Modify: `tenferro-linalg/src/primal/linear_systems.rs`
- Modify: `tenferro-linalg/src/primal/least_squares.rs`
- Modify: `tenferro-linalg/src/ad_helpers/lu.rs`
- Modify: `tenferro-linalg/src/result_types/status.rs`
- Modify: `tenferro-linalg-prims/src/backend/cuda/tests/mod.rs`
- Modify: `tenferro-linalg/src/tests/runtime_capability.rs`

**Step 1: Write the failing tests**

Add CPU regressions that prove per-batch status is no longer synthesized as all-zero/all-one:

```rust
#[test]
fn solve_ex_reports_per_batch_info_for_singular_batches() {
    let mut ctx = tenferro_prims::CpuContext::new(1);
    let a = Tensor::from_slice(
        &[
            3.0_f64, 1.0, 1.0, 2.0,
            1.0, 2.0, 2.0, 4.0,
        ],
        &[2, 2, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let b = Tensor::from_slice(&[9.0_f64, 8.0, 1.0, 2.0], &[2, 2], MemoryOrder::ColumnMajor).unwrap();

    let result = solve_ex(&mut ctx, &a, &b).unwrap();
    assert_eq!(result.info, vec![0, 2]);
}

#[test]
fn cholesky_ex_reports_per_batch_info_for_non_spd_batches() {
    let mut ctx = tenferro_prims::CpuContext::new(1);
    let a = Tensor::from_slice(
        &[
            4.0_f64, 2.0, 2.0, 3.0,
            1.0, 0.0, 0.0, -1.0,
        ],
        &[2, 2, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();

    let result = cholesky_ex(&mut ctx, &a).unwrap();
    assert_eq!(result.info, vec![0, 2]);
}
```

**Step 2: Run tests to verify they fail**

Run:

```bash
cargo test -p tenferro-linalg solve_ex_reports_per_batch_info_for_singular_batches -- --exact
cargo test -p tenferro-linalg cholesky_ex_reports_per_batch_info_for_non_spd_batches -- --exact
```

Expected: FAIL because `solve_ex` and `cholesky_ex` still synthesize coarse `[0; bc]` / `[1; bc]`.

**Step 3: Write minimal implementation**

Implement truthful status propagation without changing the public `Vec<i32>` surface yet:

- `solve_ex` should route through `lu_factor_ex` or a backend-provided status helper and preserve the exact batchwise `info`
- `cholesky_ex` should compute `info` from a status-capable backend path instead of `match Ok/Err`
- `lu_factor_impl` should stay the canonical place that maps factorization output to batchwise LU `info`

Keep the doc comments in `tenferro-linalg/src/result_types/status.rs` aligned with the new semantics: `info` is per-batch and should not collapse multiple failures into `1`.

**Step 4: Run tests to verify they pass**

Run:

```bash
cargo test -p tenferro-linalg solve_ex_reports_per_batch_info_for_singular_batches -- --exact
cargo test -p tenferro-linalg cholesky_ex_reports_per_batch_info_for_non_spd_batches -- --exact
cargo test -p tenferro-linalg --lib
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tenferro-linalg/src/primal/linear_systems.rs tenferro-linalg/src/primal/least_squares.rs tenferro-linalg/src/ad_helpers/lu.rs tenferro-linalg/src/result_types/status.rs tenferro-linalg/src/tests/runtime_capability.rs
git commit -m "fix: make linalg ex status per-batch and truthful"
```

### Task 4: Implement the minimal CUDA scalar family needed by generic linalg cleanup

**Files:**
- Modify: `tenferro-prims/src/families/scalar.rs`
- Create: `tenferro-prims/src/cuda/scalar.rs`
- Modify: `tenferro-prims/src/cuda/mod.rs`
- Modify: `tenferro-prims/src/cuda/tests/mod.rs`

**Step 1: Write the failing tests**

Add CUDA regressions for the exact subset that foundation work needs first:

```rust
#[test]
fn cuda_scalar_add_and_abs_match_cpu() {
    let Some(path) = available_cutensor_library_path() else {
        return;
    };
    if !cuda_device_zero_is_available() {
        return;
    }

    let (_backend, mut ctx) = CudaBackend::load(path).unwrap();
    let a = Tensor::<f32>::from_slice(&[-1.0, 2.0, -3.0, 4.0], &[2, 2], MemoryOrder::ColumnMajor)
        .unwrap()
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();
    let b = Tensor::<f32>::from_slice(&[10.0, 20.0, 30.0, 40.0], &[2, 2], MemoryOrder::ColumnMajor)
        .unwrap()
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();

    let add_desc = ScalarPrimsDescriptor::PointwiseBinary { op: ScalarBinaryOp::Add };
    let abs_desc = ScalarPrimsDescriptor::PointwiseUnary { op: ScalarUnaryOp::Abs };

    assert!(<CudaBackend as TensorScalarPrims<Standard<f32>>>::has_scalar_support(add_desc.clone()));
    assert!(<CudaBackend as TensorScalarPrims<Standard<f32>>>::has_scalar_support(abs_desc.clone()));
}
```

Add a reduction regression:

```rust
#[test]
fn cuda_scalar_sum_reduction_matches_cpu() {
    // Reduce over the leading matrix axes and compare against CPU.
}
```

**Step 2: Run tests to verify they fail**

Run:

```bash
cargo test -p tenferro-prims --features cuda cuda_scalar_add_and_abs_match_cpu -- --exact
cargo test -p tenferro-prims --features cuda cuda_scalar_sum_reduction_matches_cpu -- --exact
```

Expected: FAIL because `CudaBackend as TensorScalarPrims<_>` is still a stub that reports no support.

**Step 3: Write minimal implementation**

Implement only the minimal subset that Tasks 5-8 need:

- unary: `Conj`, `Abs`, `Reciprocal`
- binary: `Add`, `Sub`, `Mul`, `Div`, `Maximum`, `Minimum`
- reductions: `Sum`, `Max`, `Min`

In `tenferro-prims/src/cuda/scalar.rs`, add a small CUDA scalar executor that:

- plans simple pointwise/reduction kernels
- keeps all tensors on the same GPU
- supports real dtypes first (`f32`, `f64`) and complex only where needed (`Conj`, `Abs`)
- leaves unsupported descriptors false rather than lying

In `tenferro-prims/src/cuda/mod.rs`, delegate `TensorScalarPrims` planning/execution to the new module.

**Step 4: Run tests to verify they pass**

Run:

```bash
cargo test -p tenferro-prims --features cuda cuda_scalar_add_and_abs_match_cpu -- --exact
cargo test -p tenferro-prims --features cuda cuda_scalar_sum_reduction_matches_cpu -- --exact
cargo test -p tenferro-prims --features cuda
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tenferro-prims/src/families/scalar.rs tenferro-prims/src/cuda/scalar.rs tenferro-prims/src/cuda/mod.rs tenferro-prims/src/cuda/tests/mod.rs
git commit -m "feat: add minimal cuda scalar family support"
```

### Task 5: Add reusable comparison / select / mask substrate for linalg

**Files:**
- Modify: `tenferro-prims/src/families/scalar.rs`
- Modify: `tenferro-prims/src/cuda/scalar.rs`
- Modify: `tenferro-prims/src/cpu/scalar.rs`
- Modify: `tenferro-prims/src/cuda/tests/mod.rs`
- Modify: `tenferro-linalg/src/primal/decompositions.rs`
- Modify: `tenferro-linalg/src/primal/spectral.rs`
- Modify: `tenferro-linalg/src/primal/norms.rs`

**Step 1: Write the failing tests**

Add one substrate test and one linalg consumer test.

Substrate test in `tenferro-prims/src/cuda/tests/mod.rs`:

```rust
#[test]
fn cuda_scalar_threshold_and_mask_sum_match_cpu() {
    // Compare values against a scalar threshold, produce a numeric mask,
    // then reduce the mask with Sum and compare against CPU.
}
```

Consumer test in `tenferro-linalg`:

```rust
#[test]
fn svd_cutoff_on_cpu_and_gpu_share_fixed_shape_zero_fill_semantics() {
    // The CUDA path may still be capability-gated initially, but the postprocess helper
    // itself should operate on tensor-native masks and preserve fixed output shapes.
}
```

**Step 2: Run tests to verify they fail**

Run:

```bash
cargo test -p tenferro-prims --features cuda cuda_scalar_threshold_and_mask_sum_match_cpu -- --exact
cargo test -p tenferro-linalg svd_cutoff_on_cpu_and_gpu_share_fixed_shape_zero_fill_semantics -- --exact
```

Expected: FAIL because there is still no comparison/select surface.

**Step 3: Write minimal implementation**

Do **not** introduce a full public bool tensor type in this task. Instead, add the smallest reusable internal substrate needed by linalg:

- extend scalar-family vocabulary with comparison descriptors that write numeric masks (`0` or `1`) into same-dtype outputs
- add a `where`-style internal helper in the CUDA scalar executor for `select(mask, on_true, on_false)`
- keep this surface private to the crate or `pub(crate)` if possible

Use this substrate to unlock:

- `S > tol` style counting for future `matrix_rank`
- `s > cutoff ? recip(s) : 0` style thresholding for future `pinv`
- fixed-shape trailing zero-fill for future `svd cutoff`

**Step 4: Run tests to verify they pass**

Run:

```bash
cargo test -p tenferro-prims --features cuda cuda_scalar_threshold_and_mask_sum_match_cpu -- --exact
cargo test -p tenferro-linalg svd_cutoff_on_cpu_and_gpu_share_fixed_shape_zero_fill_semantics -- --exact
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tenferro-prims/src/families/scalar.rs tenferro-prims/src/cpu/scalar.rs tenferro-prims/src/cuda/scalar.rs tenferro-prims/src/cuda/tests/mod.rs tenferro-linalg/src/primal/decompositions.rs tenferro-linalg/src/primal/spectral.rs tenferro-linalg/src/primal/norms.rs
git commit -m "feat: add linalg mask and select substrate"
```

### Task 6: Add tensor-native structural composition helpers

**Files:**
- Modify: `tenferro-linalg/src/backend/tensor_helpers.rs`
- Modify: `tenferro-linalg/src/backend/tensor_helpers/tests/mod.rs`
- Modify: `tenferro-linalg/src/ad_helpers/lu.rs`
- Modify: `tenferro-linalg/src/primal/spectral.rs`
- Modify: `tenferro-linalg/src/primal/least_squares.rs`
- Modify: `tenferro-linalg/src/primal/linear_systems.rs`

**Step 1: Write the failing tests**

Add helper-focused regressions in `tenferro-linalg/src/backend/tensor_helpers/tests/mod.rs`:

```rust
#[test]
fn pack_lu_factors_matches_cpu_slice_reference() {
    // Build L and U tensors, pack them into one dense factors tensor,
    // and compare against the old host-loop reference.
}

#[test]
fn tensor_native_inverse_thresholding_matches_host_reference() {
    // Build singular values and verify reciprocal-with-cutoff zero fill.
}
```

**Step 2: Run tests to verify they fail**

Run:

```bash
cargo test -p tenferro-linalg pack_lu_factors_matches_cpu_slice_reference -- --exact
cargo test -p tenferro-linalg tensor_native_inverse_thresholding_matches_host_reference -- --exact
```

Expected: FAIL because these helpers do not exist yet.

**Step 3: Write minimal implementation**

Extend `tenferro-linalg/src/backend/tensor_helpers.rs` with reusable tensor-native helpers such as:

```rust
pub(crate) fn pack_lu_factors_tensor_native<T: LinalgScalar>(...) -> Result<Tensor<T>>;
pub(crate) fn reciprocal_with_threshold<T: LinalgScalar>(...) -> Result<Tensor<T>>;
pub(crate) fn batched_diag_from_vector<T: LinalgScalar>(...) -> Result<Tensor<T>>;
```

These helpers must be implemented from existing tensor views plus the substrate from Tasks 4-5, not from `extract_slice()` loops.

Then rewrite:

- `ad_helpers::lu::pack_lu_factors`
- `primal::spectral::pinv`
- the residual/shaping pieces of `lstsq`
- the factor-using parts of `det` / `slogdet` that can become tensor-native

Only keep host loops where there is still no reusable tensor-native route.

**Step 4: Run tests to verify they pass**

Run:

```bash
cargo test -p tenferro-linalg pack_lu_factors_matches_cpu_slice_reference -- --exact
cargo test -p tenferro-linalg tensor_native_inverse_thresholding_matches_host_reference -- --exact
cargo test -p tenferro-linalg --lib
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tenferro-linalg/src/backend/tensor_helpers.rs tenferro-linalg/src/backend/tensor_helpers/tests/mod.rs tenferro-linalg/src/ad_helpers/lu.rs tenferro-linalg/src/primal/spectral.rs tenferro-linalg/src/primal/least_squares.rs tenferro-linalg/src/primal/linear_systems.rs
git commit -m "refactor: add tensor-native linalg structure helpers"
```

### Task 7: Rewrite public/composite paths to stay generic or capability-gated

**Files:**
- Modify: `tenferro-linalg/src/primal/decompositions.rs`
- Modify: `tenferro-linalg/src/primal/norms.rs`
- Modify: `tenferro-linalg/src/primal/spectral.rs`
- Modify: `tenferro-linalg/src/primal/matrix_functions.rs`
- Modify: `tenferro-linalg/src/primal/tensor_ops.rs`
- Modify: `tenferro-linalg/src/tests/runtime_capability.rs`

**Step 1: Write the failing tests**

Add source-level and behavior-level regressions:

```rust
#[test]
fn runtime_capability_guard_keeps_gpu_off_host_only_paths() {
    // Scan target functions and assert they do not call extract_slice()/buffer().as_slice()
    // once they claim CUDA capability.
}

#[test]
fn norm_and_pinv_do_not_require_host_extraction_for_gpu_capable_paths() {
    // Use capability checks plus small CPU-path reference comparisons.
}
```

**Step 2: Run tests to verify they fail**

Run:

```bash
cargo test -p tenferro-linalg runtime_capability_guard_keeps_gpu_off_host_only_paths -- --exact
cargo test -p tenferro-linalg norm_and_pinv_do_not_require_host_extraction_for_gpu_capable_paths -- --exact
```

Expected: FAIL because several composite paths still call `extract_slice()` or `buffer().as_slice()`.

**Step 3: Write minimal implementation**

Rewrite in this order:

1. `svd(..., None)`
2. `svdvals`-based `NormKind::Spectral` / `NormKind::Nuclear`
3. `pinv`
4. `det` / `slogdet`
5. `matrix_power`, `matrix_exp`, and tensor-ops helpers

For each path:

- if the path can be expressed with Tasks 4-6 substrate, rewrite it
- if not, keep the CUDA capability false and fail early instead of adding a fallback

At the same time, make one explicit policy decision:

- `solve_triangular` validation should follow backend/info semantics rather than introducing a new host-side explicit zero-diagonal scan

**Step 4: Run tests to verify they pass**

Run:

```bash
cargo test -p tenferro-linalg runtime_capability_guard_keeps_gpu_off_host_only_paths -- --exact
cargo test -p tenferro-linalg norm_and_pinv_do_not_require_host_extraction_for_gpu_capable_paths -- --exact
cargo test -p tenferro-linalg --lib
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tenferro-linalg/src/primal/decompositions.rs tenferro-linalg/src/primal/norms.rs tenferro-linalg/src/primal/spectral.rs tenferro-linalg/src/primal/matrix_functions.rs tenferro-linalg/src/primal/tensor_ops.rs tenferro-linalg/src/tests/runtime_capability.rs
git commit -m "refactor: make composite linalg paths tensor-native or gated"
```

### Task 8: Remove AD-layer host extraction from the linalg generic path

**Files:**
- Modify: `tenferro-linalg/src/ad_helpers/layout.rs`
- Modify: `tenferro-linalg/src/ad_helpers/lu.rs`
- Modify: `tenferro-linalg/src/ad_helpers/backend_ops.rs`
- Modify: `tenferro-linalg/src/ad_helpers/svd.rs`
- Modify: `tenferro-linalg/src/rrules/norms.rs`
- Modify: `tenferro-linalg/src/rrules/linear_systems.rs`
- Modify: `tenferro-linalg/src/frules/linear_systems.rs`
- Modify: `tenferro-linalg/src/tests/mod.rs`

**Step 1: Write the failing tests**

Add focused regressions that prove AD helpers are no longer hard-coded to CPU slices:

```rust
#[test]
fn ad_helpers_do_not_extract_cpu_slices_for_gpu_capable_paths() {
    // Source-level scan similar to runtime_capability.rs, but for ad_helpers and rules.
}

#[test]
fn inv_rrule_and_norm_rrule_keep_cpu_behavior_after_tensor_native_rewrite() {
    // Run existing CPU references before and after refactor.
}
```

**Step 2: Run tests to verify they fail**

Run:

```bash
cargo test -p tenferro-linalg ad_helpers_do_not_extract_cpu_slices_for_gpu_capable_paths -- --exact
cargo test -p tenferro-linalg inv_rrule_and_norm_rrule_keep_cpu_behavior_after_tensor_native_rewrite -- --exact
```

Expected: FAIL because `ad_helpers/layout.rs` and `ad_helpers/lu.rs` still depend on `extract_slice()`.

**Step 3: Write minimal implementation**

Rewrite AD helpers to use the same tensor-native helpers from Tasks 4-7:

- stop using `extract_data()` as the only route for linalg AD internals
- keep host extraction only for paths that remain explicitly CPU-only
- reuse tensor-native matrix helpers for LU packing, triangular projection, thresholding, and batched products

If a rule still cannot be made generic after these helpers exist, keep that rule capability-gated for CUDA instead of slipping in a fallback.

**Step 4: Run tests to verify they pass**

Run:

```bash
cargo test -p tenferro-linalg ad_helpers_do_not_extract_cpu_slices_for_gpu_capable_paths -- --exact
cargo test -p tenferro-linalg inv_rrule_and_norm_rrule_keep_cpu_behavior_after_tensor_native_rewrite -- --exact
cargo test -p tenferro-linalg --lib
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tenferro-linalg/src/ad_helpers/layout.rs tenferro-linalg/src/ad_helpers/lu.rs tenferro-linalg/src/ad_helpers/backend_ops.rs tenferro-linalg/src/ad_helpers/svd.rs tenferro-linalg/src/rrules/norms.rs tenferro-linalg/src/rrules/linear_systems.rs tenferro-linalg/src/frules/linear_systems.rs tenferro-linalg/src/tests/mod.rs
git commit -m "refactor: remove host extraction from linalg ad helpers"
```

## Foundation completion gate

Do not proceed to new CUDA kernel tasks until all of these are true:

- `Tensor::contiguous()` works on GPU tensors without host slices
- CUDA `resolve_conj()` is device-side
- `_ex` results preserve truthful per-batch `info`
- CUDA scalar arithmetic and reductions needed by linalg are available
- comparison/select/mask substrate exists
- public/composite `tenferro-linalg` paths are tensor-native or correctly gated
- AD helpers are tensor-native or correctly gated

Only after that gate is met should a follow-up implementation plan wire:

- `cholesky`
- `qr`
- `lu_factor`
- `solve_triangular`
- `svdvals`
- `thin_svd`
- `eigen_sym` / `eig`

Plan complete and saved to `docs/plans/2026-03-21-tenferro-linalg-cuda-foundations.md`. Two execution options:

1. Subagent-Driven (this session) - I dispatch fresh subagent per task, review between tasks, fast iteration

2. Parallel Session (separate) - Open new session with executing-plans, batch execution with checkpoints

Which approach?
