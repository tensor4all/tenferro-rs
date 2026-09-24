# Issue #1884: faer batch lanes for batched LU factor and solve

## Summary

The three batched faer kernels (`lu_factor_batched_in_place`,
`lu_solve_prepared_batched_in_place`, `lu_factor_solve_batched_in_place` in
`crates/tenferro-linalg/src/cpu/linalg/faer_linalg/packed_lu.rs`) iterated the
batch serially and reused one scratch set, so the entered CPU pool's extra
threads only reached faer's parallelism *inside* one small matrix. A four-thread
run was therefore the same as a one-thread run, and at n=64 with a 1024-matrix
batch it was 38% slower.

## Decision

Split the batch across lanes when the context's existing faer policy grants more
than one thread *and* the batch can occupy every lane (`batch_lanes`). Each lane
owns a contiguous batch chunk and one `PackedLuFactorScratch`, and factorizes or
solves its matrices one at a time with `Par::Seq`, so the pool's threads are
spent on independent matrices instead of on one small factorization. A one-thread
or nested context (`Par::Seq`, which includes engine-owned `Outer` children)
keeps the original serial loop unchanged.

No new cross-crate API was added: the lanes read `CpuExecutionContext::faer_parallelism()`,
the existing owner-scoped policy hook, and the scoped work runs on the pool the
context already installs. The LAPACK/Blas path, its provider threading, and its
loop are untouched.

## Measured results

EPYC 7713P, `OPENBLAS_NUM_THREADS=1`, f64, batch 1024, release, median of 10:

| op | n | faer 1T | faer 4T before | faer 4T now |
|---|---:|---:|---:|---:|
| factor | 8 | 0.592 ms | 0.672 ms | **0.289 ms** |
| factor | 16 | 1.812 ms | 2.056 ms | **0.877 ms** |
| factor | 64 | 44.7 ms | 64.6 ms | **38.3 ms** |
| factor | 256 | 861 ms | 1294 ms | **523 ms** |
| fused factor+solve | 16 | 2.201 ms | - | **0.779 ms** |
| solve | 16 | 0.492 ms | 0.541 ms | **0.213 ms** |
| solve | 64 | 4.21 ms | - | **2.17 ms** |
| solve | 256 | 59.4 ms | - | **19.1 ms** |

Four threads are now never slower than one for these shapes, which the
repository's CPU threading contract requires. The Blas provider is unchanged and
still does not scale for a batch of small matrices: that case is the issue's
Accelerate/OpenBLAS half and is deliberately out of scope here, because
parallelizing it would put several provider-threaded `?getrf` calls in flight at
once. It needs the engine-level decision recorded on the issue.

## Alternatives rejected

- **Engine-level batched LU** (mirroring the grouped-GEMM path that uses
  `CpuDomainExecutor::submit_outer`). The right shape for the provider-threaded
  case, but `submit_outer` lives on the entry and an operation only receives an
  entered context whose mode is `Sequential` or `Inner`, so it needs a new
  cross-crate scheduling surface. Out of scope without its own approval.
- **A size threshold on the matrix side instead of the batch.** Measured: with
  the threshold at 64 the n=256 case kept faer's slower-than-serial inner
  parallelism (1294 ms at 4T against 876 ms at 1T). Using the batch whenever it
  can fill the lanes fixes that case too (523 ms).
- **Owning a small-matrix LU kernel** to close the remaining gap against
  PyTorch's `lu_factor` at n=8/16. The public entry adds only 0.03-0.18 ms over
  the loop, and at n=64 the loop already beats PyTorch (2.355 ms against
  2.688 ms for the solve), so the remaining difference is the provider's `dgetrf`
  on tiny matrices, not per-matrix bookkeeping in tenferro.

## Verification

- New test `batched_faer_lanes_reproduce_the_serial_batch` runs the batched
  factor, prepared solve, and fused factor+solve on a one-thread and a
  four-thread faer backend and requires bit-for-bit equality, which pins the
  claim that the lanes only redistribute work.
- The existing batched oracle sweeps (`batched_lu_solves_match_residual_oracle_across_sizes_dtypes_and_flags`
  and neighbours) still pass, and the one-thread path is the previous code.
- Not covered: the fused singularity error at more than one lane. The lanes
  report the same typed error, but which other lanes have already written their
  factors is then unspecified; callers only observe those buffers on success.
  Only the local OpenBLAS/faer providers were exercised; the Apple
  Accelerate/M5 numbers in the issue cannot be reproduced here.
