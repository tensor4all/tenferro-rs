# CubeK Complex GEMM API Implementation Plan

## Goal

Add CubeK-owned `C32` GEMM support with conjugation flags, then route tenferro
WebGPU `dot_general` and `dot_general_with_conj` through that API without
changing CUDA behavior.

## Boundaries

- CubeK owns complex GEMM semantics, temporary real buffers, split/compose
  kernels, conjugation signs, and future native complex-kernel replacement.
- tenferro owns `DotGeneralConfig` planning, WebGPU operand packing, provider
  dispatch, and user-facing documentation.
- CUDA `dot_general` remains cuTENSOR-backed. Do not change CUDA descriptors,
  workspace allocation, buffer pools, or algorithm selection.
- No hidden CPU transfers or CPU fallback are allowed.
- CubeCL kernels may use rank and operation attributes as `#[comptime]`, but
  not tensor shape extents, strides, buffer lengths, or flattened products.
- `C64` remains unsupported for WebGPU until the real `F64` WebGPU path exists.

## Phase 1: CubeK

1. Branch CubeK from the release used by tenferro: `v0.2.0` /
   `cubek-matmul 0.2.0`.
2. Add `ComplexMatmulOptions { lhs_conj, rhs_conj }`.
3. Add `launch_c32_ref` as an additive public API; keep real `launch_ref`
   compatible.
4. Implement initial `C32` lowering through four real `F32` matmuls.
5. Keep split/compose kernels in CubeK and launch over logical tensor domains
   using runtime shape/stride metadata.
6. Cover no-conj, lhs-conj, rhs-conj, both-conj, batched, and strided binding
   cases in CubeK tests.
7. Publish the CubeK commit through an addressable non-local Git source before
   pinning tenferro to it.

## Phase 2: tenferro

1. Pin `cubek-matmul` and `cubek-std` to the CubeK fork commit; do not commit a
   local path dependency.
2. Add WebGPU runtime tests for:
   - `C32` rank-2 `dot_general_with_conj` with lhs conjugation;
   - `C32` rank-2 `dot_general_with_conj` with rhs conjugation;
   - batched `C32` `dot_general_with_conj` with both inputs conjugated;
   - `F32` `dot_general_with_conj` as a real no-op.
3. Add a source contract that requires WebGPU `C32` matmul to call
   `launch_c32_ref` and rejects tenferro-local complex compose kernels.
4. Refactor `crates/tenferro-gpu/src/webgpu/gemm.rs` so `dot_general` delegates
   to `dot_general_with_conj(..., false, false)`.
5. Route `C32` through CubeK `launch_c32_ref`; leave `F32` on CubeK
   `launch_ref`.
6. Remove WebGPU-local complex split/compose kernels and raw complex
   reinterpretation helpers.
7. Override `TensorDot::dot_general_with_conj` for `WebGpuBackend`.
8. Update `docs/design/gpu-backend-design.md`,
   `docs/guides/devices-and-gpu.md`, and a work log under `docs/worklogs/`.

## Verification Scope

Run focused checks before PR:

- CubeK `cubek-matmul` complex tests and full matmul test target.
- tenferro WebGPU runtime tests for the new `dot_general_with_conj` cases.
- tenferro GPU public-surface contract tests.
- tenferro GPU feature compile checks for `webgpu`, `cuda`, and combined
  `cuda,webgpu` where possible.
- `cargo fmt --all --check`, `git diff --check`, and docs-site checks.

Full workspace release tests and coverage remain the repository-wide PR gate.
