# CPU linalg: every family through the batched tlinalg providers (#1956)

## Decisions
- The remaining CPU families (Cholesky, triangular solve, LU and solve including the
  direct-output routes, full-pivot LU and its solve, QR, rank-revealing QR, eigh, eig and the
  compact Householder kernels) moved into `tlinalg` (faer) and `tlinalg-blas` (LAPACK/BLAS),
  next to SVD and packed LU. No numerical kernel or batch loop is left in `tenferro-linalg`.
- The shared `tlinalg-traits` crate is gone: the interface the providers are adapted to is
  tenferro's, each provider owns its vocabulary, and the two adapters map their errors
  separately (`tlinalg-blas` adds an `Internal` variant for impossible LAPACK pivots).
- The provider API is torch-style batched (one call per batch over `[rows, cols, batch...]`).
  The host resolves one lane plan per faer call from the existing batch policy; `tlinalg` owns
  the fan-out on the context's pool. This replaces the host's per-item loops and its packed-LU
  `for_each_chunk`.
- Host-side responsibilities that stay here: tenferro's shape errors, empty results, tensor
  construction, the QR gauges, the rank-revealing QR rank decision and its non-finite input
  screen (run before the provider on both routes, so both report it first and identically), the
  compact Householder state compositions, and a one-time compact copy of a borrowed operand with
  negative strides. Both providers give an all-zero rank-revealing QR item the canonical zero-rank
  factors themselves, so every family is one provider call per batch on both routes.
- The tprims linalg adapter in `ext/tenferro-cpu-tprims` is removed with its `tprims-linalg`
  dependency; the injectable-kernel slot itself (`cpu_kernels`) is unchanged.

- Maintainer decision: under `Auto`, the families that did not fan out before (everything but
  packed LU) fan out only when `max(rows, cols) <= 64` (`AUTO_FAN_OUT_MAX_ITEM_DIM`,
  crate-private, not a `CpuBatchThresholds` knob). Lanes help many small items; for a large item
  faer's parallelism inside the item is the better use of the budget, and item counts alone
  cannot tell the cases apart, so above the guard `Auto` keeps the pre-change behaviour. The guard
  is provisional pending the work-model thread policy for CPU linalg (#2000). Forced strategies
  and packed LU are unchanged.

## Verification conclusions and constraints
- Intended behaviour change: non-LU faer families now follow the batch policy, so `Auto` fans
  small-item batches out over the context's lanes and an unservable forced strategy is a typed
  error for them. Its timing effect is not measured here; small-matrix batch cases in
  tenferro-benchmark are the follow-up evidence.
- Steady-state allocations per call (`tests/cpu_linalg_allocation.rs`, one thread) were recorded
  on the pre-move main and the ceilings are now the measured post-move values. No case is above
  its pre-move count; the batched providers allocate less in fourteen cases:

| case | before | after |
|---|---|---|
| faer/lu_factor/48 | 11 | 11 |
| faer/lu_solve_prepared/48 | 5 | 5 |
| faer/lu_factor_solve/48 | 13 | 13 |
| blas/lu_factor/48 | 6 | 6 |
| blas/lu_solve_prepared/48 | 5 | 5 |
| blas/lu_factor_solve/48 | 8 | 8 |
| faer/lu_factor/complex32 | 11 | 11 |
| blas/lu_factor/complex32 | 6 | 6 |
| faer/svd/48 | 12 | 12 |
| faer/svdvals/48 | 5 | 4 |
| faer/svd/tall | 12 | 12 |
| faer/svdvals/tall | 5 | 4 |
| faer/svd/complex32 | 14 | 14 |
| faer/svdvals/complex32 | 5 | 4 |
| faer/cholesky/48 | 4 | 4 |
| faer/triangular_solve/48 | 2 | 2 |
| faer/lu/48 | 13 | 13 |
| faer/full_piv_lu/48 | 17 | 17 |
| faer/full_piv_lu_solve/48 | 10 | 10 |
| faer/solve/48 | 8 | 8 |
| faer/qr/48 | 11 | 11 |
| faer/qr/tall | 11 | 11 |
| faer/householder_qr/tall | 73 | 73 |
| faer/rank_revealing_qr/tall | 15 | 15 |
| faer/eigh/48 | 9 | 9 |
| faer/eigvalsh/48 | 5 | 4 |
| faer/eig/48 | 10 | 10 |
| faer/eigvals/48 | 6 | 6 |
| faer/solve/complex32 | 8 | 8 |
| faer/qr/complex32 | 11 | 11 |
| faer/eigh/complex32 | 11 | 11 |
| faer/eig/complex32 | 9 | 9 |
| blas/cholesky/48 | 3 | 3 |
| blas/triangular_solve/48 | 2 | 2 |
| blas/lu/48 | 12 | 12 |
| blas/full_piv_lu/48 | 16 | 15 |
| blas/full_piv_lu_solve/48 | 5 | 5 |
| blas/solve/48 | 5 | 5 |
| blas/qr/48 | 8 | 7 |
| blas/qr/tall | 8 | 7 |
| blas/householder_qr/tall | 4 | 4 |
| blas/rank_revealing_qr/tall | 14 | 13 |
| blas/eigh/48 | 7 | 7 |
| blas/eigvalsh/48 | 4 | 4 |
| blas/eig/48 | 12 | 7 |
| blas/eigvals/48 | 10 | 4 |
| blas/solve/complex32 | 5 | 5 |
| blas/qr/complex32 | 8 | 7 |
| blas/eigh/complex32 | 8 | 8 |
| blas/eig/complex32 | 11 | 8 |

- Those ceilings are the pre/post-move record for this change. The faer lane's
  numbers were lowered again when every family moved onto `tlinalg 0.3.1`'s
  borrowed-workspace `*_workspace` / `*_into` entries (#2178 item 6); see
  `perf-cpu-inline-metadata-2139.md` for the route-level residual.
