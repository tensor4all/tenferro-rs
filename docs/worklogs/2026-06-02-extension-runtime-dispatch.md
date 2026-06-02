# Extension runtime dispatch work log

Date: 2026-06-02

## Session summary

This change makes extension dispatch fail explicitly instead of silently
falling back to context-free reference execution:

- eager and compiled runtime-owned paths now require registered extension
  runtimes for `StdTensorOp::Extension` / `ExecOp::Extension`,
- linalg eager wrappers register the linalg runtime before dispatch,
- `EagerBackend` delegates `LinalgBackend` calls to its owned CPU/CUDA backend,
- low-level execution helper free functions were made crate-internal or removed,
- FFT reports GPU-resident inputs as unsupported instead of panicking or
  attempting a host reference path.

The CUDA checks were run on an NVIDIA A100 with CUDA 12.6.

## Context read

| Source | Why it was read | Decision impact |
| --- | --- | --- |
| `REPOSITORY_RULES.md` | Confirm public-surface and extension-boundary rules. | Added durable rules for missing-runtime errors and low-level helper visibility. |
| `docs/spec/extension-op.md` | Check the extension dispatch contract. | Updated stale language that allowed unregistered eager fallback. |
| `docs/spec/backend-contract.md` | Check ExecProgram execution ownership. | Documented `GraphExecutor` as the public runtime/cache owner. |
| `tenferro-ad/src/eager_exec.rs` | Locate eager extension fallback. | Replaced direct `eager_execute` fallback with missing-executor errors. |
| `tenferro-runtime/src/exec.rs` and `src/segment.rs` | Locate compiled extension fallback and public direct eval helpers. | Removed `ExtensionExecutor`-less extension fallback and closed direct eval helpers. |
| `tenferro-linalg/src/eager_tensor.rs` | Locate linalg eager wrapper dispatch. | Added defensive runtime registration before `apply_eager`. |
| `tenferro-fft/src/lib.rs` | Locate FFT runtime and direct execution. | Split host FFT reference execution and added explicit GPU unsupported errors. |

## Decisions made

- **Missing extension runtime is an error.** Runtime-owned execution must use
  `ExtensionExecutor::execute`; unregistered families surface a missing-runtime
  diagnostic with the family id.
- **Direct execution helpers are not public API.** Public free eval functions
  and `tenferro_ad::eager_exec` were removed from the public surface. Tests now
  use `GraphExecutor` or module-local unit tests.
- **Extension runtime composition is owner-scoped.** `ExtensionExecutionContext`
  exposes a narrow core-only subprogram executor for einsum runtime lowering.
  It rejects nested extension ops so it cannot bypass runtime registration.
- **FFT GPU support is deferred, not faked.** FFT remains host/reference
  execution today. GPU-resident inputs get an unsupported error and will be
  tracked separately.
- **Linalg eager CUDA uses the existing backend.** `EagerBackend` implements
  `LinalgBackend` by delegating to its owned CPU/CUDA backend, avoiding fresh
  backend construction in registered linalg eager execution.

## Rejected or deferred alternatives

- **No silent CPU/reference fallback.** It hides performance regressions and
  backend-state loss.
- **No broad nested extension executor in this PR.** The only current need is
  einsum's core-only subprogram. Supporting nested extensions would require
  threading registry ownership through the context deliberately.
- **No FFT CUDA implementation in this PR.** That is a new capability and is
  tracked as follow-up work.

## Verification performed

- `cargo fmt --all --check`
- `cargo test --workspace --release`
- `cargo test -p tenferro-runtime`
- `cargo test -p tenferro-ad`
- `cargo test -p tenferro-ad --test extension_op`
- `cargo test -p tenferro-ad --test exec_dispatch --test segment_tests --test compiler_passes`
- `cargo test -p tenferro-einsum`
- `cargo test -p tenferro-fft --test fft_ops`
- `cargo test -p tenferro-linalg --features autodiff`
- `cargo test -p tenferro-linalg --features autodiff --test eager_tensor`
- `cargo test -p tenferro-linalg --features autodiff --test traced_extension --test eager_device_dispatch`
- `cargo llvm-cov --workspace --release --json --output-path coverage.json`
- `python3 scripts/check-coverage.py coverage.json`
- `cargo doc --workspace --no-deps`
- `python3 scripts/check-docs-site.py`
- `CUBECL_DEBUG_LOG=0 CUDA_PATH=/usr/local/cuda-12.6 LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:/usr/lib/x86_64-linux-gnu/libcutensor/12:$LD_LIBRARY_PATH cargo test -p tenferro-linalg --features 'autodiff cuda' --test eager_tensor cuda_eager_solve_uses_registered_linalg_runtime -- --ignored`
- `CUBECL_DEBUG_LOG=0 CUDA_PATH=/usr/local/cuda-12.6 LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:/usr/lib/x86_64-linux-gnu/libcutensor/12:$LD_LIBRARY_PATH cargo test -p tenferro-ad --features cuda --test segment_tests segmented_dispatch_matches_unsegmented_dispatch_on_cubecl_host_boundaries`
- `CUBECL_DEBUG_LOG=0 CUDA_PATH=/usr/local/cuda-12.6 LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:/usr/lib/x86_64-linux-gnu/libcutensor/12:$LD_LIBRARY_PATH cargo test -p tenferro-fft --features cuda --test fft_ops registered_runtime_reports_gpu_input_as_unsupported`
- `CUBECL_DEBUG_LOG=0 CUDA_PATH=/usr/local/cuda-12.6 LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:/usr/lib/x86_64-linux-gnu/libcutensor/12:$LD_LIBRARY_PATH cargo test -p tenferro-linalg --features cuda --test gpu_linalg test_cubecl_solve_f64_matches_cpu -- --ignored`

## Remaining risks

- CI should still run the full protected-branch matrix.
- FFT CUDA execution remains unsupported by design and needs a follow-up issue.
