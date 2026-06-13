# WebGPU And Complex GEMM Implementation

Date: 2026-06-13

## Session Summary

This work adds a first-class WebGPU provider path to `tenferro-gpu` while
keeping CUDA and WebGPU as explicit additive provider features. It introduces
`WebGpuBackend`/`WebGpuRuntime`, preserves `CudaBackend` plus the existing
`CubeclBackend` compatibility alias, and adds `GpuBackendKind::WebGpu` for
placement metadata.

`tenferro-ad` now accepts `WebGpuBackend` through `EagerRuntime`, and
`tenferro-einsum` propagates an explicit additive `webgpu` feature so eager and
traced binary einsum can execute supported WebGPU matmul paths after callers
explicitly upload input tensors.

The implemented WebGPU operation path covers explicit upload/download,
`F32` `dot_general` through a CubeK BGEMM planner, and `C32` `dot_general`
through WebGPU-local real/imag split, four real `F32` CubeK matmuls, and
WebGPU-local compose. The planner maps tenferro output order
`[lhs_free..., rhs_free..., batch...]` to CubeK `[batch..., M, K]`,
`[batch..., K, N]`, and `[batch..., M, N]`. Non-metadata operand layouts are
packed on the same WebGPU provider. `F64`, `C64`, zero-contracting-size matmul,
and non-matmul tensor ops remain explicit unsupported paths.

## Context Read

- `AGENTS.md`
- `REPOSITORY_RULES.md`
- shared tensor4all agent rules for common, Rust, performance, numerical, and
  docs/tests work
- `docs/design/gpu-backend-design.md`
- `crates/tenferro-gpu/src/lib.rs`
- `crates/tenferro-gpu/src/cubecl/dispatch.rs`
- `crates/tenferro-gpu/src/cubecl/gemm.rs`
- `crates/tenferro-gpu/src/cubecl/memory.rs`
- `crates/tenferro-gpu/tests/public_surface_contract.rs`
- `crates/tenferro-gpu/src/webgpu/gemm.rs`
- `crates/tenferro-gpu/src/webgpu/kernels.rs`
- `crates/tenferro-ad/src/eager.rs`
- `crates/tenferro-ad/src/eager_backend.rs`
- `crates/tenferro-einsum/src/eager_tensor.rs`
- `crates/tenferro-tensor/src/types.rs`
- `cubek-matmul 0.2.0`
- `cubek-std 0.2.0`
- the workspace-pinned `tensor4all/cubecl` revision

## Decisions Made

- Did not add a default WebGPU feature or a generic `gpu` alias. Downstream
  users must choose `cuda`, `webgpu`, or both, matching the explicit provider
  style used for CPU BLAS/faer choices.
- Propagated `webgpu` through `tenferro-ad` and `tenferro-einsum` as a
  concrete provider feature, without making it a default and without making it
  imply a CPU provider when defaults are disabled.
- Added `EagerBackend::WebGpu` and `EagerRuntime::with_webgpu_backend` so eager
  execution delegates through the backend trait surface instead of adding
  einsum-specific WebGPU dispatch.
- Kept CUDA runtime/library bindings out of the WebGPU feature. WebGPU depends
  on CubeCL-WGPU and CubeK matmul only.
- Added `[patch.crates-io]` entries for `cubecl` and `cubecl-common` so
  `cubek-matmul`/`cubek-std` use the same tensor4all CubeCL source as
  tenferro. Without this, Cargo builds registry CubeCL and git CubeCL together,
  making `TensorBinding`, `ComputeClient`, and `StorageType` different Rust
  types.
- Started CubeK integration from `cubek-matmul 0.2.0`, the CubeK release paired
  to CubeCL 0.10.0. Any future tensor4all CubeK fork should branch from that
  point and be published as tensor4all-owned crates rather than vendored into
  tenferro.
- Implemented WebGPU `C32` GEMM as real `F32` decomposition instead of using
  CubeCL complex core operations. The active WGPU runtime did not advertise
  `c32` Core complex support for `real_val`/`imag_val`, so split/compose uses
  raw `f32` parts views of `Complex32` allocations.
- Kept `C64` unsupported in the initial WebGPU path because it requires `F64`
  support that is not part of the WebGPU matmul implementation.
- Added a private WebGPU `DotGeneralPlan` rather than exposing CubeK layout
  details as public API. The planner validates dimensions once, flattens free
  and contracting axes, and binds CubeK views with runtime shape/stride
  metadata.
- Kept tensor shape extents, strides, buffer lengths, and flattened products
  out of CubeCL `#[comptime]` parameters. Pack kernels use only axis role lists
  and rank as compile-time launch attributes; shape and stride values are read
  from `TensorBinding` metadata in the kernel.
- Added WebGPU-local pack kernels for operand layouts that cannot be expressed
  as metadata-only `[batch..., M, K]` or `[batch..., K, N]` CubeK views. These
  kernels launch over the packed output domain and write one element per worker.
- Split the WebGPU provider into explicit `runtime`, `memory`, `gemm`, and
  `kernels` modules. `mod.rs` remains the backend facade plus shared buffer
  validation helpers, while CubeK launch details stay in `gemm.rs`.
- Kept WebGPU allocation helpers fallible: shape products and output byte
  lengths are checked before creating device allocations.
- Kept traced WebGPU graph execution aligned with the existing no-hidden-transfer
  contract: non-scalar graph inputs must be explicitly uploaded and bound as
  WebGPU tensors before execution.
- Recorded common future GPU scratch-pool stats fields, but did not wire a new
  scratch pool into CUDA GEMM. CUDA cuTENSOR workspace allocation remains on
  the existing path.
- Recorded future ROCm direction as a CubeCL HIP fork or patch with
  runtime-loaded HIP libraries. ROCm remains unavailable as an execution backend
  in this work.

## Rejected Or Deferred Alternatives

- Did not make WebGPU the default GPU provider. That would surprise downstream
  builds and prevent explicit provider selection.
- Did not hide unsupported WebGPU ops behind CPU fallback. Host/device movement
  remains explicit.
- Did not try CubeK `Strategy::Auto` for WebGPU yet. The initial provider path
  uses `Strategy::Naive`; tuning and adapter-specific selection should be
  benchmarked separately.
- Did not add WebGPU elementwise, reduction, indexing, or linalg kernels.
- Did not expose ROCm execution or a ROCm quickstart without loader-backed HIP
  runtime support and hardware validation.
- Did not add GPU tutorial code in this PR. Follow-up issue
  <https://github.com/tensor4all/tenferro-rs/issues/1046> tracks GPU tutorials
  and running GPU tutorial examples on GPU-capable CI runners.

## Verification Performed

- RED/GREEN: `cargo test -p tenferro-tensor device_model_has_first_class_webgpu_backend_kind`
- RED/GREEN: `cargo test -p tenferro-gpu --test public_surface_contract`
- RED/GREEN: `cargo test -p tenferro-gpu --test public_surface_contract downstream_gpu_features_are_explicit_and_additive`
- RED/GREEN: `cargo test -p tenferro-gpu --test public_surface_contract`
  for the WebGPU runtime/transfer/GEMM module-boundary source contract.
- RED/GREEN: `cargo test -p tenferro-ad --no-default-features --features webgpu,cpu-faer --test eager_runtime_api eager_runtime_accepts_webgpu_backend_constructor`
- RED/GREEN: `cargo test -p tenferro-einsum --no-default-features --features autodiff,webgpu,cpu-faer --test webgpu_eager_tensor`
  for eager WebGPU `F32`/`C32` binary einsum and traced WebGPU `F32` binary
  einsum with explicit WebGPU input bindings.
- RED/GREEN: `cargo test -p tenferro-gpu --no-default-features --features webgpu,cpu-faer --test webgpu_backend_contract`
- RED/GREEN: `cargo test -p tenferro-gpu --no-default-features --features webgpu,cpu-faer --test webgpu_matmul_runtime webgpu_dot_general_runs_rank2_c32_matmul_when_adapter_available -- --nocapture`
- RED/GREEN: `cargo test -p tenferro-gpu --no-default-features --features webgpu,cpu-faer --test webgpu_matmul_runtime webgpu_dot_general_supports_batched_f32_contract_shape_when_adapter_available`
- RED/GREEN: `cargo test -p tenferro-gpu --no-default-features --features webgpu,cpu-faer --test webgpu_matmul_runtime webgpu_dot_general_packs_noncontiguous_lhs_free_axes_when_adapter_available`
- `cargo fmt --all --check`
- `cargo test -p tenferro-gpu --test public_surface_contract`
- `cargo test -p tenferro-gpu --test public_surface_contract`
  includes the WebGPU checked-allocation source contract.
- `cargo test -p tenferro-gpu --no-default-features --features webgpu,cpu-faer --test webgpu_backend_contract`
- `cargo test -p tenferro-gpu --no-default-features --features webgpu,cpu-faer --test webgpu_matmul_runtime`
- `cargo test -p tenferro-einsum --no-default-features --features autodiff,webgpu,cpu-faer --test webgpu_eager_tensor`
- `cargo check -p tenferro-gpu --no-default-features --features cuda,webgpu,cpu-faer`
- `cargo check -p tenferro-ad --no-default-features --features cuda,webgpu,cpu-faer`
- `cargo check -p tenferro-einsum --no-default-features --features autodiff,cuda,webgpu,cpu-faer`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo clippy --manifest-path ext/tropical/Cargo.toml --all-targets -- -D warnings`
- `cargo test --workspace --release`
- `cargo llvm-cov --workspace --release --json --output-path coverage.json`
- `python3 scripts/check-coverage.py coverage.json`
- `cargo doc --workspace --no-deps`
- `python3 scripts/check-docs-site.py`
- `git diff --check`

The local machine had a working WebGPU adapter, so the WebGPU runtime tests ran
the backend paths instead of returning early.

## Remaining Risks

- WebGPU `dot_general` supports `F32` and `C32` through CubeK BGEMM planning,
  but planner stress coverage is still narrower than CUDA coverage.
- `C32` uses four real matmuls and split/compose kernels; performance has not
  been benchmarked.
- WebGPU zero-contracting-size matmul remains unsupported until CubeK behavior
  is validated.
- CUDA runtime tests and CUDA benchmarks were not run in this pass; the contract
  was to preserve CUDA implementation behavior, not to benchmark it.
- ROCm hardware was not available and ROCm remains an unavailable execution
  backend.
