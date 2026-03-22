# tenferro-linalg CUDA Foundation-First Runtime Layering Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the CUDA foundation for `tenferro-linalg` without breaking layering: `tenferro-linalg` stays CPU/GPU generic, `tenferro-tensor` stays a tensor layer, and shared device runtime ownership moves into `tenferro-device` instead of being duplicated in higher layers.

**Architecture:** The previous foundation-first plan correctly identified the missing substrate, but its first task was layered incorrectly. `Tensor::contiguous()` and related tensor-basic GPU operations do need device context, but that context must come from a shared Layer 0 runtime owned below both `tenferro-tensor` and `tenferro-prims`, not from `tenferro-prims::CudaContext` and not from an ad hoc CUDA runtime embedded directly in `tenferro-tensor`. The clean shape is: `tenferro-device` owns shared GPU runtime state and low-level copy/launch helpers; `tenferro-tensor` uses that substrate for tensor-basic operations; `tenferro-prims` and `tenferro-linalg-prims` build execution-family and linalg-family contexts on top of the same runtime substrate.

**Tech Stack:** Rust, `tenferro-device`, `tenferro-tensor`, `tenferro-prims`, `tenferro-linalg-prims`, `tenferro-linalg`, `cudarc`, runtime-loaded CUDA driver APIs, cuBLAS/cuSOLVER wrappers, `cargo test`, `cargo fmt`, real-GPU tests gated by `TENFERRO_TEST_CUDA=1`.

---

## Relationship to the earlier foundation plan

This plan is a layering-correct successor to:

- `docs/plans/2026-03-21-tenferro-linalg-cuda-foundations.md`

Do not delete or overwrite that earlier note. Treat it as a historical intermediate draft. Execute this runtime-layering plan instead.

## Locked design decisions

- `tenferro-device` is the correct Layer 0 home for shared CUDA/HIP runtime substrate.
- `tenferro-tensor` must not depend on `tenferro-prims`.
- `tenferro-tensor` may depend on `tenferro-device` runtime helpers for GPU tensor-basic operations.
- `tenferro-prims::CudaContext` and future `tenferro-linalg-prims` CUDA runtime wrappers should reuse the same underlying `tenferro-device` runtime family rather than spinning up a second unrelated GPU world.
- Large tensor payloads must not bounce GPU→CPU in normal library paths.
- Host-visible control flow may inspect minimal status such as `info`, but not input matrices or decomposition payloads.

## Target layering after this plan

- **Layer 0: `tenferro-device`**
  - shared CUDA runtime handle(s)
  - device selection / context registry
  - buffer allocation / free
  - H2D / D2H / D2D copy
  - generic strided copy kernel launch
  - future shared low-level helpers like device-side conjugation kernels
- **Layer 1: `tenferro-tensor`**
  - `Tensor<T>` metadata + buffer
  - `contiguous()`, transfer, triangular projection, basic materialization using Layer 0
- **Layer 2: `tenferro-prims`**
  - `CudaContext` / family plans / stream policy / plan caches
  - scalar/analytic/semiring execution on top of Layer 0
- **Layer 3: `tenferro-linalg-prims`**
  - cuBLAS/cuSOLVER linalg kernels and `info` plumbing on top of Layer 0 / Layer 2
- **Layer 4: `tenferro-linalg`**
  - public/composite/AD layer, CPU/GPU generic

## Scope and stop point

This plan still stops at the foundation layer. Do **not** start new CUDA implementations of `qr`, `cholesky`, `thin_svd`, `svdvals`, `lu_factor`, or `eigen*` until all eight tasks below are merged.

The already-existing reusable pieces that remain valid and should be reused are:

- `clone_batched_column_major` / `copy_batched_column_major` in `tenferro-linalg-prims/src/backend/linalg_utils.rs`
- GPU `tril` / `triu` behavior already prototyped in `tenferro-tensor`
- the existing real CUDA `solve` path and `info`-only host sync discipline in `tenferro-linalg-prims`

The hard rules for every task in this plan are:

- no ad hoc GPU→CPU transfer of input/output tensor payloads
- shared runtime ownership belongs in `tenferro-device`, not duplicated above it
- new helper layers must be reusable by more than one linalg op
- if a path is still not generic after a task, keep CUDA capability false rather than adding a fallback

### Task 1: Add shared Layer 0 CUDA runtime substrate to `tenferro-device`

**Files:**
- Modify: `tenferro-device/Cargo.toml`
- Modify: `tenferro-device/src/lib.rs`
- Create: `tenferro-device/src/cuda/mod.rs`
- Create: `tenferro-device/src/cuda/runtime.rs`
- Create: `tenferro-device/src/cuda/tests/mod.rs`

**Step 1: Write the failing tests**

Add small runtime-focused tests in `tenferro-device/src/cuda/tests/mod.rs`:

```rust
#[test]
fn cuda_runtime_can_get_or_create_device_zero_handle() {
    if std::env::var_os("TENFERRO_TEST_CUDA").is_none() {
        return;
    }

    let runtime = tenferro_device::cuda::runtime::get_or_init(0).unwrap();
    assert_eq!(runtime.device_id(), 0);
}

#[test]
fn cuda_runtime_dtod_copy_round_trips_small_buffer() {
    if std::env::var_os("TENFERRO_TEST_CUDA").is_none() {
        return;
    }

    let runtime = tenferro_device::cuda::runtime::get_or_init(0).unwrap();
    let src = runtime.alloc::<f32>(4).unwrap();
    let dst = runtime.alloc::<f32>(4).unwrap();
    runtime.copy_htod(&[1.0_f32, 2.0, 3.0, 4.0], &src).unwrap();
    runtime.copy_dtod(&src, &dst).unwrap();
    let got = runtime.copy_dtoh(&dst).unwrap();
    assert_eq!(got, vec![1.0, 2.0, 3.0, 4.0]);
}
```

**Step 2: Run tests to verify they fail**

Run:

```bash
cargo test -p tenferro-device --features cuda cuda_runtime_can_get_or_create_device_zero_handle -- --exact
cargo test -p tenferro-device --features cuda cuda_runtime_dtod_copy_round_trips_small_buffer -- --exact
```

Expected: FAIL because `tenferro-device` currently has no CUDA runtime module at all.

**Step 3: Write minimal implementation**

Add a small internal CUDA runtime layer in `tenferro-device` that owns:

- per-device shared runtime handles
- CUDA driver context binding
- low-level allocation/free wrappers
- H2D / D2H / D2D copy wrappers
- basic error mapping into `tenferro_device::Error`

Keep this layer intentionally low-level:

- no semiring plans
- no linalg policy
- no cuSOLVER op wrappers

It is acceptable for this substrate to start CUDA-only first as long as the module boundary leaves room for HIP later.

**Step 4: Run tests to verify they pass**

Run:

```bash
cargo test -p tenferro-device --features cuda cuda_runtime_can_get_or_create_device_zero_handle -- --exact
cargo test -p tenferro-device --features cuda cuda_runtime_dtod_copy_round_trips_small_buffer -- --exact
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tenferro-device/Cargo.toml tenferro-device/src/lib.rs tenferro-device/src/cuda/mod.rs tenferro-device/src/cuda/runtime.rs tenferro-device/src/cuda/tests/mod.rs
git commit -m "feat: add shared cuda runtime substrate to tenferro-device"
```

### Task 2: Move generic GPU strided copy into Layer 0 and route tensor materialization through it

**Files:**
- Modify: `tenferro-device/src/cuda/runtime.rs`
- Modify: `tenferro-device/src/cuda/tests/mod.rs`
- Modify: `tenferro-tensor/src/cuda_runtime.rs`
- Modify: `tenferro-tensor/src/tensor/data_ops.rs`
- Modify: `tenferro-tensor/src/tests/cuda.rs`

**Step 1: Write the failing tests**

Add one Layer 0 test and one tensor-level consumer test.

In `tenferro-device/src/cuda/tests/mod.rs`:

```rust
#[test]
fn cuda_runtime_strided_copy_matches_host_reference() {
    if std::env::var_os("TENFERRO_TEST_CUDA").is_none() {
        return;
    }

    // Materialize a 3D permuted view layout into a contiguous column-major output
    // using only Layer 0 dims/strides/offset metadata and compare against a host reference.
}
```

In `tenferro-tensor/src/tests/cuda.rs`:

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

**Step 2: Run tests to verify they fail**

Run:

```bash
cargo test -p tenferro-device --features cuda cuda_runtime_strided_copy_matches_host_reference -- --exact
cargo test -p tenferro-tensor --features cuda gpu_contiguous_matches_cpu_for_strided_views_when_cuda_is_available -- --exact
```

Expected: FAIL because Layer 0 still lacks a generic strided copy launch helper.

**Step 3: Write minimal implementation**

Implement a generic Layer 0 strided-copy kernel launcher in `tenferro-device` that takes:

- source device pointer
- destination device pointer
- dims
- source strides
- source offset
- destination order or destination strides

Then make `tenferro-tensor` consume that substrate:

- `Tensor::contiguous()` must branch GPU tensors before any CPU slice access
- `tenferro-tensor/src/cuda_runtime.rs` should become a thin tensor-facing wrapper over `tenferro-device::cuda::runtime`
- do not let `tenferro-tensor` own a second unrelated CUDA context world

**Step 4: Run tests to verify they pass**

Run:

```bash
cargo test -p tenferro-device --features cuda cuda_runtime_strided_copy_matches_host_reference -- --exact
cargo test -p tenferro-tensor --features cuda gpu_contiguous_matches_cpu_for_strided_views_when_cuda_is_available -- --exact
cargo test -p tenferro-prims --features cuda cuda_make_contiguous_smoke_runs_on_device_tensors_when_runtime_is_available -- --exact
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tenferro-device/src/cuda/runtime.rs tenferro-device/src/cuda/tests/mod.rs tenferro-tensor/src/cuda_runtime.rs tenferro-tensor/src/tensor/data_ops.rs tenferro-tensor/src/tests/cuda.rs
git commit -m "feat: route tensor gpu contiguous through shared device runtime"
```

### Task 3: Rebase `tenferro-prims` CUDA context on the shared runtime and make `resolve_conj()` device-side

**Files:**
- Modify: `tenferro-prims/src/cuda/mod.rs`
- Modify: `tenferro-prims/src/cuda/tests/mod.rs`
- Modify: `tenferro-prims/src/gpu_stubs.rs`
- Modify: `tenferro-prims/src/cuda_ffi.rs`

**Step 1: Write the failing tests**

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
}
```

Also add a context-sharing regression:

```rust
#[test]
fn cuda_context_uses_shared_device_runtime() {
    let Some(path) = available_cutensor_library_path() else {
        return;
    };
    if !cuda_device_zero_is_available() {
        return;
    }

    let (_backend, ctx) = CudaBackend::load(path).unwrap();
    assert_eq!(ctx.device_id(), 0);
    assert!(ctx.shared_runtime().is_some());
}
```

**Step 2: Run tests to verify they fail**

Run:

```bash
cargo test -p tenferro-prims --features cuda cuda_resolve_conj_keeps_tensor_on_device_and_matches_cpu -- --exact
cargo test -p tenferro-prims --features cuda cuda_context_uses_shared_device_runtime -- --exact
```

Expected: FAIL because `resolve_conj()` still uses host slices and `CudaContext` still owns an isolated runtime path.

**Step 3: Write minimal implementation**

Refactor `tenferro-prims::CudaContext` so it holds or references the shared Layer 0 runtime instead of privately reimplementing it.

Then replace the host fallback in `CudaBackend::resolve_conj()` with a device-side path that:

- uses GPU `contiguous()` from Task 2
- launches device-side conjugation against the shared runtime
- keeps result tensors on GPU

Keep `gpu_stubs.rs` aligned so non-CUDA builds still compile.

**Step 4: Run tests to verify they pass**

Run:

```bash
cargo test -p tenferro-prims --features cuda cuda_resolve_conj_keeps_tensor_on_device_and_matches_cpu -- --exact
cargo test -p tenferro-prims --features cuda cuda_context_uses_shared_device_runtime -- --exact
cargo test -p tenferro-prims --features cuda
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tenferro-prims/src/cuda/mod.rs tenferro-prims/src/cuda/tests/mod.rs tenferro-prims/src/gpu_stubs.rs tenferro-prims/src/cuda_ffi.rs
git commit -m "refactor: base cuda context on shared device runtime"
```

### Task 4: Make `_ex` status truthful and per-batch

**Files:**
- Modify: `tenferro-linalg/src/primal/linear_systems.rs`
- Modify: `tenferro-linalg/src/primal/least_squares.rs`
- Modify: `tenferro-linalg/src/ad_helpers/lu.rs`
- Modify: `tenferro-linalg/src/result_types/status.rs`
- Modify: `tenferro-linalg/src/tests/runtime_capability.rs`

**Step 1: Write the failing tests**

Add CPU regressions:

```rust
#[test]
fn solve_ex_reports_per_batch_info_for_singular_batches() {
    // One invertible batch, one singular batch.
}

#[test]
fn cholesky_ex_reports_per_batch_info_for_non_spd_batches() {
    // One SPD batch, one failing batch.
}
```

**Step 2: Run tests to verify they fail**

Run:

```bash
cargo test -p tenferro-linalg solve_ex_reports_per_batch_info_for_singular_batches -- --exact
cargo test -p tenferro-linalg cholesky_ex_reports_per_batch_info_for_non_spd_batches -- --exact
```

Expected: FAIL because `_ex` still synthesizes coarse all-zero/all-one status.

**Step 3: Write minimal implementation**

Make `solve_ex`, `cholesky_ex`, and LU helpers preserve truthful per-batch `info` without changing the public `Vec<i32>` result type yet.

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
git commit -m "fix: make linalg ex status truthful and per-batch"
```

### Task 5: Implement the minimal CUDA scalar family needed by generic linalg cleanup

**Files:**
- Modify: `tenferro-prims/src/families/scalar.rs`
- Create: `tenferro-prims/src/cuda/scalar.rs`
- Modify: `tenferro-prims/src/cuda/mod.rs`
- Modify: `tenferro-prims/src/cuda/tests/mod.rs`

**Step 1: Write the failing tests**

Add CUDA regressions covering only the minimal needed subset:

```rust
#[test]
fn cuda_scalar_add_and_abs_match_cpu() {
    // Pointwise Add and Abs on GPU and compare with CPU.
}

#[test]
fn cuda_scalar_sum_reduction_matches_cpu() {
    // Reduce over matrix axes on GPU and compare with CPU.
}
```

**Step 2: Run tests to verify they fail**

Run:

```bash
cargo test -p tenferro-prims --features cuda cuda_scalar_add_and_abs_match_cpu -- --exact
cargo test -p tenferro-prims --features cuda cuda_scalar_sum_reduction_matches_cpu -- --exact
```

Expected: FAIL because CUDA scalar family is still a stub.

**Step 3: Write minimal implementation**

Implement only:

- unary: `Conj`, `Abs`, `Reciprocal`
- binary: `Add`, `Sub`, `Mul`, `Div`, `Maximum`, `Minimum`
- reductions: `Sum`, `Max`, `Min`

Use the shared Layer 0 runtime; do not create a second low-level CUDA launch stack inside `tenferro-prims`.

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

### Task 6: Add reusable comparison / select / mask substrate for linalg

**Files:**
- Modify: `tenferro-prims/src/families/scalar.rs`
- Modify: `tenferro-prims/src/cpu/scalar.rs`
- Modify: `tenferro-prims/src/cuda/scalar.rs`
- Modify: `tenferro-prims/src/cuda/tests/mod.rs`
- Modify: `tenferro-linalg/src/primal/decompositions.rs`
- Modify: `tenferro-linalg/src/primal/spectral.rs`
- Modify: `tenferro-linalg/src/primal/norms.rs`

**Step 1: Write the failing tests**

Add one substrate test and one consumer test:

```rust
#[test]
fn cuda_scalar_threshold_and_mask_sum_match_cpu() {
    // Build a numeric mask with comparison and reduce it.
}

#[test]
fn svd_cutoff_fixed_shape_zero_fill_semantics_hold() {
    // Verify cutoff preserves fixed shape and zero-fills trailing regions.
}
```

**Step 2: Run tests to verify they fail**

Run:

```bash
cargo test -p tenferro-prims --features cuda cuda_scalar_threshold_and_mask_sum_match_cpu -- --exact
cargo test -p tenferro-linalg svd_cutoff_fixed_shape_zero_fill_semantics_hold -- --exact
```

Expected: FAIL because comparison/select substrate does not exist yet.

**Step 3: Write minimal implementation**

Add the smallest reusable internal comparison/select surface needed by linalg:

- numeric-mask comparisons
- `select(mask, on_true, on_false)` style helper
- reduction of numeric masks with existing scalar reductions

Do not add a full public bool tensor type in this task.

**Step 4: Run tests to verify they pass**

Run:

```bash
cargo test -p tenferro-prims --features cuda cuda_scalar_threshold_and_mask_sum_match_cpu -- --exact
cargo test -p tenferro-linalg svd_cutoff_fixed_shape_zero_fill_semantics_hold -- --exact
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tenferro-prims/src/families/scalar.rs tenferro-prims/src/cpu/scalar.rs tenferro-prims/src/cuda/scalar.rs tenferro-prims/src/cuda/tests/mod.rs tenferro-linalg/src/primal/decompositions.rs tenferro-linalg/src/primal/spectral.rs tenferro-linalg/src/primal/norms.rs
git commit -m "feat: add linalg comparison and mask substrate"
```

### Task 7: Add tensor-native structural helpers and rewrite composite public paths

**Files:**
- Modify: `tenferro-linalg/src/backend/tensor_helpers.rs`
- Modify: `tenferro-linalg/src/backend/tensor_helpers/tests/mod.rs`
- Modify: `tenferro-linalg/src/ad_helpers/lu.rs`
- Modify: `tenferro-linalg/src/primal/spectral.rs`
- Modify: `tenferro-linalg/src/primal/least_squares.rs`
- Modify: `tenferro-linalg/src/primal/linear_systems.rs`
- Modify: `tenferro-linalg/src/primal/decompositions.rs`
- Modify: `tenferro-linalg/src/primal/norms.rs`
- Modify: `tenferro-linalg/src/primal/matrix_functions.rs`
- Modify: `tenferro-linalg/src/primal/tensor_ops.rs`
- Modify: `tenferro-linalg/src/tests/runtime_capability.rs`

**Step 1: Write the failing tests**

Add helper-level and source-level regressions:

```rust
#[test]
fn pack_lu_factors_tensor_native_matches_host_reference() {
    // Pack L and U into factors tensor and compare with reference.
}

#[test]
fn runtime_capability_guard_keeps_gpu_off_host_only_paths() {
    // Source-level assertion that GPU-capable public paths avoid extract_slice()/buffer().as_slice().
}
```

**Step 2: Run tests to verify they fail**

Run:

```bash
cargo test -p tenferro-linalg pack_lu_factors_tensor_native_matches_host_reference -- --exact
cargo test -p tenferro-linalg runtime_capability_guard_keeps_gpu_off_host_only_paths -- --exact
```

Expected: FAIL because public/composite linalg still has host extraction in multiple places.

**Step 3: Write minimal implementation**

Add small reusable tensor-native helpers in `backend/tensor_helpers.rs` for:

- LU packing
- reciprocal-with-threshold
- batched diagonal materialization where needed
- other fixed-shape structural composition built from Tasks 5-6 substrate

Then rewrite public/composite paths in this order:

1. `svd(..., None)`
2. `svdvals`-based `NormKind::Spectral` / `NormKind::Nuclear`
3. `pinv`
4. `det` / `slogdet`
5. `lstsq`, `matrix_power`, `matrix_exp`, `tensor_ops`

If a path is still not expressible generically, keep CUDA capability false instead of adding a fallback.

**Step 4: Run tests to verify they pass**

Run:

```bash
cargo test -p tenferro-linalg pack_lu_factors_tensor_native_matches_host_reference -- --exact
cargo test -p tenferro-linalg runtime_capability_guard_keeps_gpu_off_host_only_paths -- --exact
cargo test -p tenferro-linalg --lib
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tenferro-linalg/src/backend/tensor_helpers.rs tenferro-linalg/src/backend/tensor_helpers/tests/mod.rs tenferro-linalg/src/ad_helpers/lu.rs tenferro-linalg/src/primal/spectral.rs tenferro-linalg/src/primal/least_squares.rs tenferro-linalg/src/primal/linear_systems.rs tenferro-linalg/src/primal/decompositions.rs tenferro-linalg/src/primal/norms.rs tenferro-linalg/src/primal/matrix_functions.rs tenferro-linalg/src/primal/tensor_ops.rs tenferro-linalg/src/tests/runtime_capability.rs
git commit -m "refactor: make composite linalg paths tensor-native or gated"
```

### Task 8: Remove AD-layer host extraction from the generic CUDA path

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

Add AD-layer regressions:

```rust
#[test]
fn ad_helpers_do_not_extract_cpu_slices_for_gpu_capable_paths() {
    // Source-level scan over ad_helpers/rrules/frules.
}

#[test]
fn inv_rrule_and_norm_rrule_keep_cpu_behavior_after_tensor_native_rewrite() {
    // Reuse existing CPU references around the rewritten helpers.
}
```

**Step 2: Run tests to verify they fail**

Run:

```bash
cargo test -p tenferro-linalg ad_helpers_do_not_extract_cpu_slices_for_gpu_capable_paths -- --exact
cargo test -p tenferro-linalg inv_rrule_and_norm_rrule_keep_cpu_behavior_after_tensor_native_rewrite -- --exact
```

Expected: FAIL because AD helpers still depend on `extract_slice()` and related host extraction.

**Step 3: Write minimal implementation**

Rewrite AD helpers and rules to reuse the tensor-native helpers from Task 7. Keep host extraction only for explicitly CPU-only paths. If a rule still cannot be made generic, capability-gate it instead of adding a fallback.

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

Do not proceed to new CUDA linalg kernel tasks until all of these are true:

- `tenferro-device` owns a shared CUDA runtime substrate
- `Tensor::contiguous()` uses shared Layer 0 strided copy instead of host extraction
- `tenferro-prims::CudaContext` reuses shared Layer 0 runtime
- CUDA `resolve_conj()` is device-side
- `_ex` results preserve truthful per-batch `info`
- CUDA scalar arithmetic and reductions needed by linalg are available
- comparison/select/mask substrate exists
- public/composite and AD `tenferro-linalg` paths are tensor-native or correctly gated

Only after that gate is met should a follow-up implementation plan wire:

- `cholesky`
- `qr`
- `lu_factor`
- `solve_triangular`
- `svdvals`
- `thin_svd`
- `eigen_sym` / `eig`

Plan complete and saved to `docs/plans/2026-03-21-tenferro-linalg-cuda-foundations-runtime-layering.md`. Two execution options:

1. Subagent-Driven (this session) - I dispatch fresh subagent per task, review between tasks, fast iteration

2. Parallel Session (separate) - Open new session with executing-plans, batch execution with checkpoints

Which approach?
