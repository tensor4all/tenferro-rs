# GPU synchronize API work log

Date: 2026-06-01

## Session summary

This change adds an explicit host-side synchronization API for eager and direct
CUDA backend execution:

- `EagerRuntime::synchronize()` for eager code,
- `CudaRuntime::synchronize()` for direct CubeCL CUDA backend integrations.

Eager CPU runtimes return immediately. Eager CUDA runtimes synchronize the
current backend stream without downloading tensor data.

The CubeCL module-level doctest was also guarded with `gpu_available()` so
CUDA-feature doctests remain portable on machines without `libcuda`.

## Context read

| Source | Why it was read | Decision impact |
| --- | --- | --- |
| `AGENTS.md` | Confirm repository workflow, GPU-code requirements, and work-log expectations. | Read the GPU design document before changing CUDA runtime code and added this work log. |
| `REPOSITORY_RULES.md` | Review public API and documentation expectations. | Kept the API small and documented the synchronization boundary. |
| `docs/design/gpu-backend-design.md` | Confirm CubeCL runtime ownership and transfer policy. | Implemented a stream synchronization barrier rather than implicit tensor downloads. |
| `tenferro-gpu/src/cubecl/runtime.rs` | Locate CUDA context and raw stream access. | Added synchronization beside existing runtime stream APIs. |
| `tenferro-ad/src/eager.rs` and `tenferro-ad/src/eager_backend.rs` | Locate eager runtime backend ownership. | Routed eager synchronization through the existing backend mutex. |
| `tenferro-linalg/src/gpu/linalg.rs` | Check existing CUDA stream synchronization usage. | Reused the same `cudaStreamSynchronize` approach on the CubeCL raw stream. |
| `tenferro-gpu/src/cubecl/mod.rs` | Investigate a CUDA-feature doctest failure on a no-GPU host. | Guarded the module quickstart with `gpu_available()`, matching other GPU docs examples. |

## Decisions made

- **Use `synchronize`, not `wait`.** The name matches CUDA and PyTorch
  terminology and describes a backend-wide stream barrier better than a tensor
  readiness flag.
- **Keep synchronization explicit.** Normal eager CUDA ops still submit work and
  return CUDA-resident handles without host blocking after every kernel.
- **Synchronize the current CubeCL CUDA stream.** The method waits for work on
  the active backend stream and does not copy tensor payloads to host memory.
- **Make CPU eager synchronization a no-op.** This keeps the API portable across
  eager backends.

## Rejected or deferred alternatives

- **No implicit eager wait after every GPU op.** That would remove useful CUDA
  overlap and make eager GPU behavior unnecessarily different from PyTorch-style
  asynchronous CUDA execution.
- **No tensor-level ready flag.** The current backend surface has one stream
  boundary; per-tensor readiness would be a broader async ownership design.
- **No stream/event API in this change.** A richer stream/event surface can build
  on this primitive later if the backend exposes multiple public streams.

## Verification performed

- `cargo fmt --all --check`
- `cargo test -p tenferro-ad eager_runtime_synchronize_is_available_and_cpu_noop`
- `cargo test -p tenferro-ad --test eager_runtime_api`
- `cargo test -p tenferro-ad --features cuda --test eager_runtime_api`
- `cargo test -p tenferro-gpu --test cubecl_launch_contract`
- `cargo test -p tenferro-gpu --features cuda --test cubecl_launch_contract`
- `cargo check -p tenferro-ad --features cuda`
- `cargo check -p tenferro-gpu --features cuda --example cuda_quickstart`
- `cargo test --doc -p tenferro-ad`
- `cargo test --doc -p tenferro-gpu --features cuda`
- `git diff --check`

## Remaining risk

- Local verification compile-checks the CUDA synchronization API and doctests,
  but does not run ignored GPU hardware tests. A CUDA machine should run the
  ignored GPU suite before depending on hardware-level behavior changes.
