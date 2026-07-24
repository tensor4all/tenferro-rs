# Phase 7 WebGPU runtime-registration checkpoint

This worklog records the first Phase 7 local checkpoint after
[`2026-07-25-phase-6-extension-resolution-einsum.md`](2026-07-25-phase-6-extension-resolution-einsum.md).

## Session summary

Phase 7 is being implemented locally on
`codex/execution-engine-phase9-restart`. Per maintainer direction, no PR is
created until Phase 8 and the AMD CPU/CUDA benchmark gate are complete.

The implemented slice is intentionally bounded:

- `tenferro-gpu` exposes a WebGPU runtime engine-registration helper;
- the helper registers `DotGeneralPreparation` and the common runtime-owned
  `TensorBackendExecutor<WebGpuBackend>`;
- `Runtime::run_compiled` is covered for a WebGPU-resident rank-2 F32 matmul;
- CUDA runtime registration is deferred because `CudaBackend` does not yet
  satisfy the cloneable backend ownership contract required by
  `EngineRegistration::with_tensor_backend_executor`.

## Context read

- Workspace and repository rules: `AGENTS.md`, `REPOSITORY_RULES.md`, workspace
  `CODING_RULES.md`, and shared tensor4all rules.
- Design authority:
  `docs/design/execution-engine-provider-architecture.md`.
- Reference implementation:
  `crates/tenferro-cpu/src/runtime_adapter.rs`.
- Runtime execution bridge:
  `crates/tenferro-runtime/src/runtime/engine_registration.rs` and
  `crates/tenferro-runtime/src/runtime/execution.rs`.
- WebGPU backend and tests:
  `crates/tenferro-gpu/src/webgpu/mod.rs` and
  `crates/tenferro-gpu/tests/integration/webgpu_matmul_runtime.rs`.

## Implementation decisions

1. WebGPU uses `tenferro-webgpu.default.v1` as the engine ID and
   `tenferro-webgpu.device.v1` as the hardware class.
2. The registered storage class is `tenferro.storage.device.v1`, matching the
   runtime's `MemoryKind::Device` storage projection. No WebGPU-specific
   storage class is introduced in this slice.
3. The preparation adapter is `dot_general`-only. Other core operation families
   remain absent from the WebGPU registration until their runtime preparation
   contracts are implemented.
4. The prepared-operation metadata mirrors the CPU adapter's specialization
   requirements for dot-general: dtype, rank, concrete dimensions, and layout
   class. Execution itself remains owned by `TensorBackendExecutor<WebGpuBackend>`.
5. `tenferro-gpu` now depends directly on `tenferro-runtime` because the public
   helper returns runtime registration types.

## TDD evidence

The following RED check was observed before the implementation:

```text
cargo test -p tenferro-gpu webgpu_runtime_run_compiled_rank2_f32_matmul_when_adapter_available --features webgpu
  -> compile failure: no `webgpu_runtime_engine_registration` in the crate root
```

The corresponding focused GREEN check passed:

```text
cargo test -p tenferro-gpu webgpu_runtime_run_compiled_rank2_f32_matmul_when_adapter_available --features webgpu
cargo test -p tenferro-gpu webgpu_runtime --features webgpu
cargo test -p tenferro-gpu --features webgpu
cargo test -p tenferro-gpu
cargo fmt --all --check
python3 scripts/check-doc-snippets.py
python3 scripts/check-public-error-docs.py
python3 scripts/test-doc-consistency.py
python3 scripts/check-guide-dependency-snippets.py
git diff --check
```

On machines without a WebGPU adapter, the integration test compiles and exits
early through the existing `webgpu_available()` hardware gate.

## Open-decision ledger

These items are intentionally not implemented in this Phase 7 slice:

- CUDA runtime engine registration;
- generalized `GpuExecutionContext`, stream/queue event slots, and GPU
  admission control;
- WebGPU runtime preparation for elementwise, reduction, indexing, or layout
  families;
- CUDA/WebGPU native einsum engine registration.

Phase 8 remains the owner of XLA/subgraph integration. PR creation, CI
babysitting, and merge remain deferred until Phase 8 and the AMD CPU/CUDA
benchmark gate are complete.
