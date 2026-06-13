# WebGPU Complex GEMM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans or superpowers:subagent-driven-development to execute this plan task by task. Use test-driven-development for behavior changes.

**Goal:** Complete explicit CUDA/WebGPU provider support without changing CUDA performance behavior, then broaden WebGPU `dot_general`/einsum from rank-2 real matmul to batched real and complex GEMM through a clean planner.

**Architecture:** `CudaBackend` remains the existing cuTENSOR/cuSOLVER/cuBLAS-backed provider. `WebGpuBackend` remains a separate CubeCL-WGPU provider and lowers tenferro `DotGeneralConfig` into CubeK-compatible BGEMM views. When the input layout cannot be represented as a CubeK `[batch..., M, K]` or `[batch..., K, N]` tensor through shape/stride metadata, WebGPU performs an explicit same-device pack kernel. There are no hidden CPU transfers and no CPU fallback paths.

**Tech Stack:** Rust 2021, CubeCL, CubeCL-WGPU, CubeK `cubek-matmul 0.2.0`, `smallvec`, tenferro tensor/runtime/einsum/ad crates, Quarto docs.

---

## Scope And Contracts

- Execution worktree: `/home/shinaoka/.config/superpowers/worktrees/tenferro-rs/webgpu-complex-gemm`.
- Existing dirty files are treated as in-progress work and must not be reverted blindly.
- Do not make per-task commits from this plan; use focused diffs and verification checkpoints. Commit creation is a separate maintainer decision after the full scope is reviewed.
- CUDA performance-preservation contract: do not alter CUDA algorithm selection, cuTENSOR/cuBLAS/cuSOLVER dispatch, packing, workspace allocation, runtime cache, buffer pools, or scratch reuse behavior.
- Feature contract: GPU provider features are explicit and additive: `cuda` and `webgpu`. Do not add a public `gpu` alias and do not enable a GPU provider by default.
- Public provider names: downstream code should see `CudaBackend` and `WebGpuBackend` as concrete providers. `CubeclBackend` may remain as a CUDA compatibility alias.
- `DotGeneralConfig` output contract remains `[lhs_free..., rhs_free..., batch...]`.
- CubeK adapter contract: CubeK receives `[batch..., M, K]`, `[batch..., K, N]`, and `[batch..., M, N]`, where `M`, `K`, and `N` are flattened products derived by the planner.
- Transfer contract: unsupported WebGPU and ROCm paths return explicit backend errors; they do not download, upload, or call CPU implementations behind the user's back.
- ROCm contract for this pass: document the runtime-loaded HIP substrate direction, but keep ROCm unavailable as an execution backend.

## DotGeneral Lowering Design

- Build one private `DotGeneralPlan` in `crates/tenferro-gpu/src/webgpu/gemm.rs`.
- The planner validates ranks with `DotGeneralConfig::validate_dims_with_ranks`, then validates paired contracting sizes and paired batch sizes.
- `lhs_free` and `rhs_free` are the non-contracting, non-batch axes in natural axis order.
- `lhs_contract` and `rhs_contract` preserve the paired order from `DotGeneralConfig`.
- `lhs_batch` and `rhs_batch` preserve the paired order from `DotGeneralConfig`.
- `M = product(lhs_free_shape)`, `N = product(rhs_free_shape)`, `K = product(contract_shape)`.
- `output_shape = lhs_free_shape + rhs_free_shape + batch_shape`.
- `lhs_cubek_shape = batch_shape + [M, K]`; `rhs_cubek_shape = batch_shape + [K, N]`; `out_cubek_shape = batch_shape + [M, N]`.
- Output binding writes directly into tenferro's compact output order with CubeK strides `[M * N * batch_prefix..., 1, M]`.
- Metadata-only input binding is allowed when the flattened free or contract axis group is contiguous in the source tensor layout and ordered the same way the planner unflattens that group.
- Otherwise allocate a WebGPU scratch tensor with the CubeK shape and run a same-device pack kernel before launching CubeK.
- Zero output elements return an allocated empty output. Zero contracting size remains an explicit WebGPU unsupported error until CubeK support is verified.

## Phase 1: Real WebGPU DotGeneral And Provider Contracts

### Task 1: Lock CUDA And Feature Contracts

**Files:**
- `crates/tenferro-gpu/tests/public_surface_contract.rs`
- `crates/tenferro-gpu/Cargo.toml`

- [x] Add `smallvec.workspace = true` under `tenferro-gpu` dependencies for planner metadata.
- [x] Add a contract test named `cuda_dot_general_stays_cutensor_backed_and_not_cubek_rewired`.
- [x] The test reads `crates/tenferro-gpu/src/cubecl/gemm.rs` and asserts:
  - it contains `cutensor.contract(`,
  - it contains `alloc_workspace(backend.runtime(), workspace_size)`,
  - it contains `Plan::new(cutensor, &op_desc`,
  - it does not contain `cubek_matmul`,
  - it does not contain `DotGeneralPlan`.
- [x] Run:

```bash
cargo test -p tenferro-gpu --test public_surface_contract cuda_dot_general_stays_cutensor_backed_and_not_cubek_rewired
```

Expected: pass. This is a characterization contract for existing CUDA behavior, so it may pass immediately.

### Task 2: Add Batched Real WebGPU DotGeneral

**Files:**
- `crates/tenferro-gpu/src/webgpu/gemm.rs`
- `crates/tenferro-gpu/src/webgpu/mod.rs`
- `crates/tenferro-gpu/tests/webgpu_matmul_runtime.rs`

- [x] Add a WebGPU runtime test named `webgpu_dot_general_supports_batched_f32_contract_shape`.
- [x] Test shape:
  - lhs shape `[2, 3, 2]` with `lhs_contracting_dims=[1]`, `lhs_batch_dims=[2]`,
  - rhs shape `[3, 2, 2]` with `rhs_contracting_dims=[0]`, `rhs_batch_dims=[2]`,
  - output shape `[2, 2, 2]`.
- [x] Expected compact column-major F32 values:

```text
[58.0, 139.0, 64.0, 154.0, 5800.0, 13900.0, 6400.0, 15400.0]
```

- [x] Verify the test fails before implementation because the current WebGPU path rejects batch dims.
- [x] Add `SmallVec` planner structs and a `build_dot_general_plan` helper.
- [x] Add crate-private WebGPU binding helper for custom shape/stride metadata rather than exposing a public low-level API.
- [x] Replace `validate_rank2_matmul` in the F32 path with planner-based validation and CubeK binding.
- [x] Run:

```bash
cargo test -p tenferro-gpu --no-default-features --features webgpu,cpu-faer --test webgpu_matmul_runtime webgpu_dot_general_supports_batched_f32_contract_shape
```

Expected: pass on machines with a WebGPU adapter, or return early through `webgpu_available()`.

### Task 3: Add WebGPU Pack Path For Non-Metadata Layouts

**Files:**
- `crates/tenferro-gpu/src/webgpu/kernels.rs`
- `crates/tenferro-gpu/src/webgpu/gemm.rs`
- `crates/tenferro-gpu/src/webgpu/mod.rs`
- `crates/tenferro-gpu/tests/webgpu_matmul_runtime.rs`

- [x] Add a WebGPU runtime test named `webgpu_dot_general_packs_noncontiguous_lhs_free_axes`.
- [x] Test shape:
  - lhs shape `[2, 3, 2]`, contracting axis `[1]`, free axes `[0, 2]`,
  - rhs shape `[3, 2]`, contracting axis `[0]`, free axis `[1]`,
  - output shape `[2, 2, 2]`.
- [x] Choose small known F32 values and compute the expected output in the test with a compact host reference loop over `(m0, m1, n, k)`.
- [x] Verify the test fails before the pack path is implemented.
- [x] Add `pack_lhs_dot_general` and `pack_rhs_dot_general` kernels using `Tensor<E>` metadata, `ABSOLUTE_POS < out.len()`, and one output element per worker.
- [x] Kernel mapping:
  - LHS out axes are `[batch..., M, K]`; unflatten `M` over `lhs_free`, unflatten `K` over `lhs_contract`, copy batch coordinates through `lhs_batch`.
  - RHS out axes are `[batch..., K, N]`; unflatten `K` over `rhs_contract`, unflatten `N` over `rhs_free`, copy batch coordinates through `rhs_batch`.
- [x] Add host-side binding selection:
  - use metadata-only custom shape/stride binding when each flattened group is representable,
  - otherwise allocate scratch with `alloc_output`, launch the pack kernel, and bind the packed scratch.
- [x] Run:

```bash
cargo test -p tenferro-gpu --no-default-features --features webgpu,cpu-faer --test webgpu_matmul_runtime
```

Expected: pass on machines with a WebGPU adapter, or return early through `webgpu_available()`.

## Phase 2: Complex GEMM, ROCm Direction, And User Docs

### Task 4: Route WebGPU C32 Through The Same Planner

**Files:**
- `crates/tenferro-gpu/src/webgpu/gemm.rs`
- `crates/tenferro-gpu/tests/webgpu_matmul_runtime.rs`
- `crates/tenferro-einsum/tests/webgpu_eager_tensor.rs`

- [x] Add a batched `C32` WebGPU `dot_general` test using the Task 2 shape pattern.
- [x] Compute expected values in the test with `Complex32` host multiplication and a single compact comparison helper.
- [x] Update `dot_general_c32` so the split real tensors reuse the same planner and pack/binding path as F32.
- [x] Add assertions that WebGPU `F64` and `C64` `dot_general` return explicit WebGPU backend errors.
- [x] Extend eager/traced WebGPU einsum coverage to one batched or non-rank-2-free case that lowers to supported `dot_general`.
- [x] Run:

```bash
cargo test -p tenferro-gpu --no-default-features --features webgpu,cpu-faer --test webgpu_matmul_runtime
cargo test -p tenferro-einsum --no-default-features --features autodiff,webgpu,cpu-faer --test webgpu_eager_tensor
```

Expected: pass on machines with a WebGPU adapter, or return early through `webgpu_available()`.

### Task 5: Document Common GPU Scratch Pool Direction

**Files:**
- `docs/design/gpu-backend-design.md`
- `docs/worklogs/2026-06-13-webgpu-complex-gemm.md`
- `crates/tenferro-gpu/tests/public_surface_contract.rs`

- [x] Record that a future GPU scratch-pool stats API should be common across CUDA, WebGPU, and future ROCm.
- [x] Name the future common stats fields:

```text
retained_buffers
retained_bytes
acquire_calls
release_calls
reuse_hits
allocation_misses
evictions
high_water_retained_bytes
```

- [x] State that this implementation does not alter CUDA allocation, workspace, or buffer-pool behavior.
- [x] Add a source-contract assertion that `cubecl/gemm.rs` still uses the existing CUDA workspace flow and does not wire a new `GpuScratchPool`.

### Task 6: Record ROCm Compile-Only Direction

**Files:**
- `docs/design/gpu-backend-design.md`
- `docs/guides/devices-and-gpu.md`
- `docs/worklogs/2026-06-13-webgpu-complex-gemm.md`

- [x] Document that future ROCm support should use a CubeCL HIP fork/patch with runtime-loaded HIP libraries, so one binary can remain usable without ROCm installed.
- [x] State that ROCm remains unavailable as a tenferro execution backend until the loader-backed substrate is implemented and tested.
- [x] Keep user docs free of a ROCm quickstart.

### Task 7: Update README And Online User Docs

**Files:**
- `README.md`
- `docs/index.md`
- `docs/guides/devices-and-gpu.md`

- [x] Update README GPU wording from CUDA-only phrasing to explicit CPU, CUDA, and experimental WebGPU backend control where user-facing.
- [x] Update the `tenferro-gpu` crate row to mention CUDA, experimental WebGPU, future ROCm substrate, and explicit device transfers.
- [x] Add or update the provider matrix:

```markdown
| Provider | Status | Feature | Notes |
| --- | --- | --- | --- |
| CPU | Supported | default CPU provider features | Host execution |
| CUDA | Supported | `cuda` | NVIDIA CUDA through CubeCL-CUDA plus CUDA libraries |
| WebGPU | Experimental | `webgpu` | Explicit transfer and limited `dot_general`/einsum coverage |
| ROCm | Not supported for execution | `rocm` reserved | Future compile-only substrate; no runtime quickstart |
```

- [x] Keep the CUDA coverage table intact.
- [x] Add WebGPU coverage rows for allocation/upload/download, `F32` and `C32` `dot_general`, eager/traced binary einsum that lowers to supported `dot_general`, and explicit unsupported rows for `F64`, `C64`, elementwise, reductions, indexing, and linalg.
- [x] Run:

```bash
python3 scripts/check-docs-site.py
```

Expected: pass.

### Task 8: Final Verification And Worklog

**Files:**
- All modified source and docs.
- `docs/worklogs/2026-06-13-webgpu-complex-gemm.md`

- [x] Run formatting:

```bash
cargo fmt --all --check
```

- [x] If formatting fails, run `cargo fmt --all`, then rerun the check.
- [x] Run source checks:

```bash
cargo test -p tenferro-gpu --test public_surface_contract
cargo test -p tenferro-gpu --no-default-features --features webgpu,cpu-faer --test webgpu_backend_contract
cargo test -p tenferro-gpu --no-default-features --features webgpu,cpu-faer --test webgpu_matmul_runtime
cargo check -p tenferro-gpu --no-default-features --features cuda,webgpu,cpu-faer
cargo check -p tenferro-ad --no-default-features --features cuda,webgpu,cpu-faer
cargo check -p tenferro-einsum --no-default-features --features autodiff,cuda,webgpu,cpu-faer
```

- [x] Run docs and diff checks:

```bash
python3 scripts/check-docs-site.py
git diff --check
```

- [x] Update the worklog with design decisions, verification output, hardware availability, and remaining risks.

## Self-Review Checklist

- [x] CUDA source and algorithm behavior are not changed.
- [x] WebGPU and CUDA are still split by provider feature and provider type.
- [x] WebGPU `dot_general` covers real batched BGEMM before complex BGEMM.
- [x] Complex WebGPU GEMM reuses the real planner path.
- [x] Pack kernels are same-device WebGPU kernels with output-domain launches.
- [x] Docs truthfully distinguish supported CUDA, experimental WebGPU, and unavailable ROCm execution.
- [x] The plan contains concrete tests, commands, files, and expected outcomes.
