# Tensor Train Inner Product Overhead Notes

Date: 2026-05-20

This note summarizes the investigation of small-bond-dimension Tensor Train
inner-product overhead in `tenferro-rs`, compared with ITensors.jl.

## Scope

- Repository focus: `tenferro-rs`
- Workload: pairwise contractions used by a TT/MPS inner product
- Threading: one thread for all known thread pools
- Scalar type: `Complex64` / `ComplexF64`
- Physical dimension: `d = 2`
- Bond dimensions: `chi = 1, 2, 4, 8, 16, 32`

## Added Benchmarks

- `tenferro-tensor/benches/pairwise_contraction.rs`
  - Measures `first_site`, `env_bra`, `tmp_ket`, and `site_update`.
  - Compares normal per-call execution with cached GEMM analysis.
- `tenferro-tensor/benches/cpu_install_overhead.rs`
  - Measures `CpuContext::install` and the surrounding buffer-pool wrapper.
- `tenferro-tensor/benches/openblas_direct_overhead.rs`
  - Measures direct `cblas_zgemm` calls for the same small pairwise shapes.
  - Requires the `cpu-blas` feature and is intended for `src-openblas` runs.
- `scripts/bench-itensors-pairwise-contraction.jl`
  - Measures the matching ITensors.jl pairwise contractions.
- `scripts/bench-pairwise-contraction.sh`
  - Runs Rust and/or Julia benchmarks with one-thread environment variables.
- `benchmarks/julia/Project.toml`
  - Minimal Julia environment containing ITensors.

## Commands

Instantiate Julia:

```bash
JULIA_NUM_THREADS=1 julia --startup-file=no --project=benchmarks/julia \
  -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```

Run ITensors.jl pairwise benchmark:

```bash
bash scripts/bench-pairwise-contraction.sh \
  --kind julia \
  --warm-up-time 0.5 \
  --measurement-time 1 \
  --sample-size 10
```

Run Rust pairwise benchmark:

```bash
bash scripts/bench-pairwise-contraction.sh \
  --kind rust \
  --warm-up-time 0.5 \
  --measurement-time 1 \
  --sample-size 10
```

Run CPU install overhead benchmark:

```bash
RAYON_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
cargo bench -p tenferro-tensor --bench cpu_install_overhead -- \
  --warm-up-time 0.5 \
  --measurement-time 1 \
  --sample-size 20
```

Run OpenBLAS-backed tenferro pairwise benchmark:

```bash
RAYON_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
cargo bench -p tenferro-tensor \
  --no-default-features \
  --features src-openblas \
  --bench pairwise_contraction -- \
  --warm-up-time 0.5 \
  --measurement-time 1 \
  --sample-size 10
```

Run direct OpenBLAS microbenchmark:

```bash
RAYON_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
cargo bench -p tenferro-tensor \
  --no-default-features \
  --features src-openblas \
  --bench openblas_direct_overhead -- \
  --warm-up-time 0.5 \
  --measurement-time 1 \
  --sample-size 20
```

## Pairwise Result Summary

`site_update` is the two-contraction pair that corresponds to one inner-product
site update.

| chi | ITensors median | tenferro normal mean | tenferro cached mean | normal / ITensors |
|---:|---:|---:|---:|---:|
| 1 | 3.26 us | 20.38 us | 14.19 us | 6.3x |
| 2 | 3.72 us | 20.65 us | 16.22 us | 5.6x |
| 4 | 3.78 us | 19.21 us | 13.32 us | 5.1x |
| 8 | 5.40 us | 15.59 us | 20.59 us | 2.9x |
| 16 | 10.82 us | 19.50 us | 20.44 us | 1.8x |
| 32 | 48.14 us | 57.19 us | 59.03 us | 1.2x |

Interpretation:

- At small `chi`, tenferro is dominated by per-call overhead, not arithmetic.
- At `chi = 32`, arithmetic dominates more and tenferro approaches ITensors.
- Cached GEMM analysis helps some small cases, but it is not enough to close
  the gap because the main fixed cost is outside layout analysis.

## OpenBLAS Tracking

Julia reports `LBTConfig([ILP64] libopenblas64_.so)` with
`LinearAlgebra.BLAS.get_config()`, and the benchmark script sets
`BLAS.set_num_threads(1)`. Therefore the ITensors comparison is consistent with
a one-thread OpenBLAS-backed Julia setup.

Switching tenferro to `src-openblas` did not, by itself, remove the small-`chi`
fixed cost:

| chi | direct OpenBLAS 2x zgemm | direct + conj copy | tenferro OpenBLAS normal | tenferro OpenBLAS cached |
|---:|---:|---:|---:|---:|
| 1 | 0.17 us | 0.21 us | 20.43 us | 20.05 us |
| 2 | 0.19 us | 0.22 us | 17.02 us | 15.90 us |
| 4 | 0.34 us | 0.31 us | 16.43 us | 14.88 us |
| 8 | 1.41 us | 1.53 us | 17.50 us | 14.78 us |
| 16 | 4.41 us | 8.23 us | 27.35 us | 22.02 us |
| 32 | 39.64 us | 52.39 us | 62.89 us | 68.23 us |

The direct OpenBLAS calls are much faster than the tenferro OpenBLAS backend for
small `chi`. At `chi <= 4`, the BLAS kernels take less than 0.4 us for the
two-GEMM `site_update`, while tenferro takes about 15-20 us. OpenBLAS itself is
therefore not the bottleneck.

After bypassing `rayon::ThreadPool::install` for the `cpu-blas` GEMM and LAPACK
paths, the same `site_update` benchmark improves substantially:

| chi | direct OpenBLAS 2x zgemm | tenferro OpenBLAS normal | tenferro OpenBLAS cached |
|---:|---:|---:|---:|
| 1 | 0.17 us | 2.24 us | 0.98 us |
| 2 | 0.19 us | 5.13 us | 2.75 us |
| 4 | 0.34 us | 5.50 us | 3.32 us |
| 8 | 1.41 us | 6.43 us | 3.91 us |
| 16 | 4.41 us | 10.65 us | 8.18 us |
| 32 | 39.64 us | 38.48 us | 35.63 us |

This removes the earlier "much heavier than faer" state for OpenBLAS at small
bond dimension. The remaining small-shape gap is mostly generic tensor overhead:
shape validation, GEMM analysis/cache lookup, output construction, fallback
materialization for conjugated layouts, and Rust-side dispatch around the BLAS
call.

The OpenBLAS-backed `dot_general_overhead` benchmark also shows the install
cost clearly for an `L = 32` inner-product chain:

| chi | fresh backend cache | persistent cache | single exec session |
|---:|---:|---:|---:|
| 4 | 812.7 us | 547.0 us | 187.1 us |
| 8 | 635.0 us | 658.5 us | 252.9 us |
| 16 | 901.5 us | 1073.4 us | 376.3 us |
| 32 | 2111.7 us | 2365.9 us | 1334.7 us |
| 64 | 9620.6 us | 9854.4 us | 14380.1 us |

For `chi = 4`, moving the 64 GEMMs into one execution session saves roughly
626 us, or about 9.8 us per GEMM. This is consistent with a combination of
`rayon::ThreadPool::install` plus backend entry overhead dominating tiny GEMMs.

After the install cost is removed by a single execution session, remaining
small-`chi` overhead still comes from generic tensor machinery: shape
validation, GEMM analysis/cache lookup, output allocation through `BufferPool`,
`TypedTensor` construction, and complex conjugation/materialization when BLAS
cannot represent the requested conjugation as a transpose flag.

## Install Overhead

Measured on one thread:

| Case | Time |
|---|---:|
| `CpuContext::install(|| {})` | 5.47 us |
| `CpuBackend::install(|| {})` | 5.37 us |
| `BufferPool` `mem::take` / restore only | 0.21 us |
| `ctx.install` + `BufferPool` take/restore | 5.71 us |
| `ctx.install` + `BufferPool` + GEMM cache borrow | 5.97 us |

The bottleneck is mostly `rayon::ThreadPool::install`. Buffer-pool movement is
only about 0.2 us.

For `site_update`, which performs two GEMMs, this adds roughly 11-12 us of
fixed cost when each GEMM enters the backend separately.

## Faer Global-Rayon Upper Bound

The faer backend cannot currently receive a specific Rayon `ThreadPool` handle.
`faer::Par` is only a lightweight parallelism value:

```rust
pub enum Par {
    Seq,
    Rayon(NonZeroUsize),
}
```

Similarly, the lower-level `gemm_common::Parallelism` value is:

```rust
pub enum Parallelism {
    None,
    Rayon(usize),
}
```

Both APIs describe the requested thread count, not the thread pool that must run
the work. faer and spindle rely on the current Rayon context through
`rayon::current_num_threads`, `rayon::scope`, `into_par_iter`, and
`spindle::for_each`. Therefore tenferro currently uses
`CpuContext::install(...)` to make its owned Rayon pool the current execution
context before calling faer.

As an upper-bound experiment, the faer backend was run with an experimental
switch that bypasses `CpuContext::install` and lets faer/spindle use the
current or global Rayon pool directly:

```bash
TENFERRO_EXPERIMENTAL_FAER_GLOBAL_RAYON=1
TENFERRO_PAIRWISE_THREADS=<1|2|4|8>
```

The benchmark below is cached `site_update`, `Complex64`, `d = 2`, with all
known BLAS/OpenMP thread pools set to one thread. Times are Criterion means in
microseconds.

| chi | faer install 1T | faer global 1T | faer install 2T | faer global 2T | faer install 4T | faer global 4T | faer global 8T |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 17.465 | 0.681 | 17.246 | 0.699 | 26.552 | 0.717 | 0.735 |
| 2 | 17.498 | 1.053 | 18.610 | 1.061 | 28.941 | 1.073 | 1.099 |
| 4 | 18.017 | 1.171 | 19.156 | 1.176 | 19.967 | 1.180 | 1.204 |
| 8 | 20.151 | 1.935 | 19.959 | 1.932 | 30.154 | 1.971 | 2.010 |
| 16 | 22.445 | 4.922 | 25.134 | 5.017 | 38.282 | 4.912 | 4.950 |
| 32 | 40.829 | 27.842 | 54.813 | 37.542 | 61.173 | 47.744 | 53.738 |

This shows that the per-call pool entry cost is a dominant faer overhead for
small TT inner-product contractions. However, the fastest upper-bound results
are almost always the one-thread global-Rayon run. For these small shapes,
increasing faer thread count does not help; it usually adds parallel dispatch
overhead. The important optimization is removing per-GEMM `install`, not making
each tiny GEMM parallel.

The experiment should not be treated as a production backend policy. Bypassing
`install` lets faer use the current/global Rayon pool, so `CpuContext` no longer
fully controls pool isolation, thread count, or affinity. It is useful as a
ceiling measurement and as evidence that coarse-grained pool entry is the main
direction to pursue.

Design implications:

- For one-thread faer, `Par::Seq` makes `install` unnecessary and a caller-thread
  fast path is safe.
- For multi-thread faer, keep tenferro pool semantics by entering the pool once
  per larger unit of work, such as one TT contraction, one compiled graph, or
  one linalg batch.
- Add a small-shape policy that keeps tiny faer GEMMs sequential even when the
  backend context has more than one thread.
- A true pool-handle patch would need changes across faer, spindle, and the
  lower-level gemm parallelism API. It is technically possible but invasive
  because `Par`/`Parallelism` are currently `Copy` count-only values.

## What `rayon::ThreadPool::install` Does

When called outside the target Rayon pool, `install` does not run the closure on
the caller thread. It:

1. Creates a stack job holding the closure and result slot.
2. Injects that job into the pool's external job queue.
3. Wakes a worker.
4. Blocks the caller on a latch.
5. Executes the closure on the worker thread.
6. Stores the result and notifies the latch.
7. Wakes the caller and returns the result.

This preserves Rayon pool semantics, but it is expensive for tiny GEMMs.

## Current Code Changes

- Normal `dot_general` and `dot_general_with_conj` no longer store GEMM analysis
  into a throwaway local cache slot. Previously these paths used `Some(0)` even
  though the cache was dropped immediately after the call.
- The internal GEMM analysis cache key now uses `SmallVec` for common small
  ranks and dimension lists. This reduces heap allocation in cached paths.
- For the `cpu-blas` backend, `dot_general`, `dot_general_read`, and
  `dot_general_with_conj` run the GEMM body directly on the caller thread with
  the backend `BufferPool` and GEMM analysis cache, instead of routing through
  `rayon::ThreadPool::install`.
- For the `cpu-blas` backend, LAPACK-backed linalg methods now use the same
  caller-thread pool wrapper. This covers `cholesky`, `triangular_solve`, `lu`,
  `full_piv_lu`, `full_piv_lu_solve`, `svd`, `qr`, `eigh`, and `eig`. The
  `cpu-faer` backend still uses `install` because faer depends on the Rayon CPU
  context for parallel policy.
- The BLAS conjugation path now detects row-contiguous conjugated operands before
  acquiring an output buffer for a direct BLAS attempt that cannot succeed.

No tensor values or contraction results are cached. The cache stores GEMM
layout analysis only:

- lhs/rhs shapes
- contracting and batch dimension lists
- GEMM dimensions `m`, `n`, `k`
- batch count
- input/output strides
- output shape

## Why ITensors.jl Is Faster At Small `chi`

ITensors/NDTensors uses specialized dense tensor contraction paths:

- Scalar-like contractions avoid GEMM entirely.
- Dense contractions reshape/transpose views and call `mul!`.
- Julia specializes the hot path by tensor order and storage type after JIT.

For `chi = 1`, several contractions become scalar-like. ITensors therefore
avoids most of the generic `dot_general` machinery that tenferro currently
executes.

## Compiled Graph Behavior

Compiled `TracedTensor` execution does not always pay one install per GEMM.

- If the whole program can run in a single `TensorExec` session, install happens
  once for the graph evaluation.
- Pure chains of `DotGeneral` / `DotGeneralWithConj` can use this path.
- If segmentation finds a multi-instruction elementwise fused segment, the
  program falls back to segment dispatch.
- In segment dispatch, `Segment::Ffi` currently calls backend methods directly,
  so GEMM instructions can pay one install per GEMM again.

This means graph compilation helps pure GEMM chains, but install overhead can
reappear in mixed graphs.

## Cross-Repository Tracing Direction

The tensor4all-rs side should keep eager values and traced graph construction as
separate types:

- `TensorDynLen` remains an eager concrete tensor. Its operations execute
  immediately and should not implicitly build or cache semantic computation
  graphs.
- `TracedTensorDynLen` should be the explicit graph-building counterpart. It can
  carry `DynIndex` metadata plus tenferro traced payload nodes, and operations
  such as contraction, permutation, and linalg add graph nodes instead of
  executing.
- Runtime graph evaluation should treat each input slot as an ABI. The bound
  `TensorDynLen` must match the traced input signature: index order, dimensions,
  dtype, and storage structure. A different index order should be a different
  graph, not an implicit permutation.
- Explicit graph nodes may represent permutations, but evaluation should reject
  accidental axis-order or structure mismatches. Diagonal/axis-class structure
  should be part of the signature; equivalent diagonal structure may match, but
  ambiguous differences should return an error and require building a new graph.

This keeps the eager API predictable and prevents hidden graph construction from
changing `TensorDynLen` semantics. It also gives GPU/lazy execution a clear
entry point without making normal eager contractions context-dependent.

## Optimization Candidates

1. Keep `CpuContext::install` public semantics unchanged.
2. Avoid broad "skip install when `num_threads == 1`" changes.
3. Consider a private, narrow fast path for one-thread sequential GEMM only:
   - `ctx.num_threads() == 1`
   - faer parallel policy is `Par::Seq`
   - operation is known GEMM-only
4. Improve segmented execution so FFI GEMMs can run through an existing
   `TensorExec` session instead of returning to backend-level calls.
5. Add scalar-like contraction fast paths for `nnz == 1`.
6. Add tiny GEMM direct-loop fast paths for small `m`, `n`, `k`.
7. Consider fixed-rank / tuple-based internal paths for common rank-2/rank-3
   contractions, while keeping the public `DotGeneralConfig` stable unless a
   broader API cleanup is desired.
8. Further reduce BLAS conjugation fallback cost by going directly to a
   preconjugated or specialized contraction path when the stride pattern makes
   `CblasConjTrans` impossible. The obvious failed direct-BLAS/output-buffer
   attempt is already avoided.
9. Add a direct tiny-complex contraction path for TT site updates, especially
   shapes equivalent to `(chi x chi)' * (chi x d*chi)` and
   `(chi*d x chi)' * (chi*d x chi)`.

## Verification Performed

- `cargo fmt --all -- --check`
- `cargo bench -p tenferro-tensor --bench pairwise_contraction --no-run`
- `cargo bench -p tenferro-tensor --bench cpu_install_overhead --no-run`
- `cargo bench -p tenferro-tensor --no-default-features --features src-openblas --bench pairwise_contraction -- --warm-up-time 0.5 --measurement-time 1 --sample-size 10`
- `cargo bench -p tenferro-tensor --no-default-features --features src-openblas --bench dot_general_overhead -- --warm-up-time 0.5 --measurement-time 1 --sample-size 10`
- `cargo bench -p tenferro-tensor --no-default-features --features src-openblas --bench openblas_direct_overhead -- --warm-up-time 0.5 --measurement-time 1 --sample-size 20`
- `cargo bench -p tenferro-tensor --no-default-features --features src-openblas --bench pairwise_contraction --no-run`
- `cargo test -p tenferro-tensor --release dot_general -- --nocapture`
- `cargo test -p tenferro-tensor --no-default-features --features src-openblas --release dot_general`
- `cargo test -p tenferro-tensor --no-default-features --features src-openblas --release`
- `bash -n scripts/bench-pairwise-contraction.sh`
- Julia `Pkg.instantiate()` and `Pkg.precompile()`
- ITensors version check: `0.9.27`
- Julia BLAS config check: `LBTConfig([ILP64] libopenblas64_.so)`, one BLAS thread
