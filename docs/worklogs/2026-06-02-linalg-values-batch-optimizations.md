# Linalg Values-Only And Batch Optimizations

## Session Summary

Implemented the remaining linalg optimization cleanup after the prepared-solve
work merged to `origin/main`. The changes remove silent full-decomposition
fallbacks for hidden values/prepared hooks, add values-only Hermitian
eigenvalue execution, reuse CPU packed LU factors, avoid dense diagonal
materialization in traced `pinv`, and reduce CUDA per-batch synchronization.

## Context Read

- `REPOSITORY_RULES.md`
- Shared tensor4all common, Rust, numerical, performance, docs, and benchmark
  rules
- `docs/design/gpu-backend-design.md`
- Existing linalg worklog: `docs/worklogs/2026-06-02-linalg-prepared-solve.md`
- JAX linalg reference under `../jax/jax/_src/lax/linalg.py`

## Decisions Made

- Added hidden backend hooks for values-only paths instead of widening public
  API. Backends that do not implement these hooks now return explicit backend
  errors instead of silently using full decompositions.
- `eigvalsh` now emits an internal `EighVals` op. Its JVP computes the needed
  eigenvectors inside the rule and returns only the eigenvalue tangent.
- CPU `lu_factor` now factors directly through the provider and returns packed
  LU, pivots, and parity. CPU `lu_solve_prepared` consumes those factors through
  pivot application and triangular solves.
- CPU and CUDA values-only SVD/eigh paths return real singular/eigenvalue
  tensors for complex inputs.
- CUDA `cholesky`, `svd`, `svd_values`, `qr`, and `eigh` now write solver info
  into a batched device tensor and download/check it once after the loop.
- CUDA triangular solve uses cuBLAS `trsmBatched` for batched inputs and keeps
  the scalar `trsm` path for a single batch.
- Traced `pinv` scales SVD vectors by broadcasting reciprocal singular values,
  avoiding dense diagonal matrix materialization.

## Rejected Or Deferred Alternatives

- Public `LuFactor`, `LuSolve`, or `EighVals` APIs were not added. The current
  request can be handled by internal extension ops and hidden backend hooks.
- Full non-symmetric CUDA `eig` remains unsupported because cuSOLVER does not
  provide the required operation.
- FFT GPU work remains deferred for issue tracking.

## Verification Performed

- `cargo test -p tenferro-linalg`
- `CUBECL_DEBUG_LOG=0 CUDA_PATH=/usr/local/cuda-12.5 LD_LIBRARY_PATH=/usr/local/cuda-12.5/targets/x86_64-linux/lib:/usr/local/cuda-12.5/lib64:/usr/lib/x86_64-linux-gnu/libcutensor/12:$LD_LIBRARY_PATH cargo test -p tenferro-linalg --features cuda`
- `CUBECL_DEBUG_LOG=0 CUDA_PATH=/usr/local/cuda-12.5 LD_LIBRARY_PATH=/usr/local/cuda-12.5/targets/x86_64-linux/lib:/usr/local/cuda-12.5/lib64:/usr/lib/x86_64-linux-gnu/libcutensor/12:$LD_LIBRARY_PATH cargo test -p tenferro-linalg --features cuda --test gpu_linalg -- --ignored`

## Remaining Risks

- cuBLAS batched TRSM uses pointer-array batched APIs. It is covered by a GPU
  test on the local A100, but larger batch-size benchmarking is still needed.
- Public `eigh` metadata for complex inputs was not changed in this session to
  avoid broad compatibility churn. The new `EighVals` path uses real output
  metadata.
