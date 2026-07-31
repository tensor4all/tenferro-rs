# Issue #1516B: pooled full-overwrite output migration

## Scope

This work completes the tenferro side of the post-strided-rs uninitialized
output migration requested by #1516B. The strided-rs dependency is pinned to
merged commit `9da9b9f63e688eaf1bf4a78b718a01ac858b1f9f`, which includes the
MaybeUninit-safe GEMM paths from strided-rs #188 and the follow-up parallel
bound fix from #197.

The migration covers CPU analytic, indexing, reduction, structural, and
elementwise output paths, plus Faer/LAPACK scratch and decomposition paths. It
also removes the obsolete CPU-only pooled uninitialized-output helpers and the
unused legacy `strided_dot.rs` and `indexing_alloc.rs` modules.

## Context read

- `AGENTS.md`
- `REPOSITORY_RULES.md`
- `ai/contribution-workflows/bugfix-pr.md`
- strided-rs #149 execution-context and ownership contract
- strided-rs #188 and #195 uninitialized GEMM design
- tenferro-rs #1516 and merged tenferro-rs #1543

## Decisions

- `PooledUninitOutput` is the canonical owner for full-overwrite pooled
  destinations. Kernel callers receive only `MaybeUninit` storage until the
  successful completion handoff.
- Read-before-write and semantic-zero paths retain safe zeroed acquisition.
  `acquire_empty_with_capacity` is limited to push-only scratch, where the
  vector length remains zero until values are appended.
- Erased strided descriptors are constructed through checked local helpers;
  their raw constructors are used only at the typed storage boundary with
  nearby safety proofs.
- Faer remains typed-unsupported for uninitialized GEMM until the upstream
  Faer API requested by strided-rs #195 exists. This change does not cast
  MaybeUninit storage to initialized Faer references or add zero-fill.
- Empty pathological view tests assert typed validation and no panic without
  coupling the public operation to the internal pooled-output validation name.

## Implementation

- Migrated CPU analytic, indexing, reduction, and structural full-overwrite
  outputs to `PooledUninitOutput` and erased strided uninitialized replay.
- Kept semantic zero-fill for diagonal embedding and other operations whose
  result is defined by untouched zero regions.
- Converted clone and push-only scratch paths to `extend_from_slice` or
  `acquire_empty_with_capacity`, avoiding reads from empty uninitialized
  vectors.
- Updated the tensor erased map/zip boundary for the merged strided raw
  descriptor constructors.
- Removed the public unsafe pool acquisition methods and updated source
  contract tests and examples.
- Removed obsolete `tenferro-cpu` helper modules and the stale tests tied to
  those modules.

## Verification

All commands were run with `CARGO_BUILD_JOBS=4` and
`RUSTFLAGS='-C link-arg=-Wl,--threads=1'` for local resource control.

- `cargo fmt --all`
- `cargo fmt --all -- --check`
- `git diff --check`
- `cargo test -p tenferro-internal-cpu-kernels` (55 unit tests, 24 doctests)
- `cargo test -p tenferro-cpu --lib --no-default-features --features cpu-faer`
  (500 tests)
- `cargo test -p tenferro-linalg --lib --no-default-features --features cpu-faer`
  (116 tests)

The initial CPU run exposed two migration defects: empty pooled vectors were
copied with `copy_from_slice`, and the materialization source-contract test
still required the removed plan-era `copy_into` spelling. Both were corrected;
the final CPU and linalg runs passed.

## Public API measurement

The focused Rust public API runner was used for the exact output-reuse,
indexing, reduction, and structural-shape rows below. The base was the
`origin/main` worktree with strided pin `6885f52`; the candidate was this
worktree with merged strided commit `9da9b9f`. Runs used `BENCH_RUNS=5`,
`BENCH_WARMUPS=2`, and were executed sequentially on the shared host with
`taskset -c 60` for one thread and `taskset -c 60-63` for four threads. Values
are eager medians in milliseconds, shown as `base -> candidate (change)`.

| Public API row | dtype, shape | 1 thread | 4 threads |
| --- | --- | ---: | ---: |
| `cpu/elementwise_reduction/reduce_prod_all` | `f64`, `8192x4096` | 10.555 -> 10.440 (-1.1%) | 8.382 -> 7.827 (-6.6%) |
| `cpu/elementwise_reduction/reduce_sum_all` | `f64`, `8192x4096` | 10.588 -> 10.236 (-3.3%) | 8.396 -> 8.163 (-2.8%) |
| `cpu/indexing_layout/concatenate` | `f64`, `1048576+1048576` | 0.717 -> 0.762 (+6.2%) | 0.704 -> 0.781 (+10.9%) |
| `cpu/indexing_layout/dynamic_slice` | `f64`, `4194304` | 0.995 -> 0.935 (-6.0%) | 0.954 -> 0.867 (-9.1%) |
| `cpu/indexing_layout/dynamic_update_slice` | `f64`, `2097152` | 1.091 -> 1.129 (+3.5%) | 1.123 -> 1.217 (+8.4%) |
| `cpu/indexing_layout/gather` | `f64`, `262144` | 6.611 -> 6.616 (+0.1%) | 1.127 -> 1.104 (-2.1%) |
| `cpu/indexing_layout/pad` | `f64`, `2097152` | 1.182 -> 1.089 (-7.9%) | 1.112 -> 1.179 (+6.0%) |
| `cpu/indexing_layout/reverse` | `f64`, `2097152` | 1.147 -> 1.040 (-9.3%) | 1.240 -> 1.206 (-2.7%) |
| `cpu/indexing_layout/scatter` | `f64`, `262144` | 9.148 -> 5.237 (-42.7%) | 8.924 -> 5.245 (-41.2%) |
| `cpu/indexing_layout/slice` | `f64`, `4194304` | 1.623 -> 1.660 (+2.3%) | 1.776 -> 1.599 (-10.0%) |
| `cpu/output_reuse/add_into` | `f64`, `33554432` | 46.087 -> 45.800 (-0.6%) | 20.008 -> 20.129 (+0.6%) |
| `cpu/output_reuse/conj_into` | `c64`, `16777216` | 27.888 -> 27.062 (-3.0%) | 13.932 -> 13.940 (+0.1%) |
| `cpu/output_reuse/copy_read_into` | `f64`, `33554432` | 13.496 -> 13.114 (-2.8%) | 13.824 -> 13.930 (+0.8%) |
| `cpu/output_reuse/div_into` | `f64`, `33554432` | 70.713 -> 67.235 (-4.9%) | 25.245 -> 20.529 (-18.7%) |
| `cpu/output_reuse/dot_general_read_into` | `f64`, `1024x1024` | 42.525 -> 43.106 (+1.4%) | 12.531 -> 12.520 (-0.1%) |
| `cpu/output_reuse/dot_general_read_into_accum` | `f64`, `1024x1024` | 45.719 -> 43.312 (-5.3%) | 12.526 -> 12.663 (+1.1%) |
| `cpu/output_reuse/mul_into` | `f64`, `33554432` | 65.507 -> 66.315 (+1.2%) | 27.225 -> 20.758 (-23.8%) |
| `cpu/output_reuse/neg_into` | `f64`, `33554432` | 46.876 -> 43.581 (-7.0%) | 14.104 -> 14.117 (+0.1%) |
| `cpu/output_reuse/sub_into` | `f64`, `33554432` | 52.012 -> 49.994 (-3.9%) | 20.348 -> 20.114 (-1.2%) |
| `cpu/structural_shape/broadcast_in_dim` | `f64`, `8192x1 -> 8192x4096` | 143.827 -> 142.912 (-0.6%) | 66.815 -> 73.001 (+9.3%) |
| `cpu/structural_shape/cast_f64_f32` | `f64->f32`, `33554432` | 85.746 -> 86.187 (+0.5%) | 38.945 -> 34.311 (-11.9%) |
| `cpu/structural_shape/embed_diagonal` | `f64`, `8192 -> 8192x8192` | 313.563 -> 320.539 (+2.2%) | 350.295 -> 356.752 (+1.8%) |
| `cpu/structural_shape/extract_diagonal` | `f64`, `8388608x2x2 -> 8388608x2` | 78.814 -> 78.732 (-0.1%) | 88.314 -> 89.622 (+1.5%) |
| `cpu/structural_shape/reshape` | `f64`, `33554432 -> 8192x4096` | 169.083 -> 169.074 (-0.0%) | 186.320 -> 185.346 (-0.5%) |
| `cpu/structural_shape/transpose` | `f64`, `4096x4096` | 127.221 -> 131.984 (+3.7%) | 49.650 -> 50.736 (+2.2%) |
| `cpu/structural_shape/tril` | `f64`, `4096x4096` | 89.741 -> 89.725 (-0.0%) | 100.112 -> 99.725 (-0.4%) |
| `cpu/structural_shape/triu` | `f64`, `4096x4096` | 89.380 -> 90.431 (+1.2%) | 100.661 -> 114.923 (+14.2%) |

The first five-run `gather` t4 measurement was noisy (+44.7%, with a 1.47 ms
IQR). It was repeated with 15 runs and 5 warmups on the same pinned CPUs;
the repeated base/candidate values in the table are the adopted values. The
other rows had no candidate-relative regression over the +20% stop-the-line
threshold in this focused run. These are local indicative measurements, not
a replacement for the full publication-gate campaign, because unrelated
Julia and Rust workloads were active on the shared host.

## Residual risks and follow-ups

- The Faer uninitialized destination path remains intentionally unsupported
  until strided-rs #195 is resolved upstream.
- Release-mode public API before/after performance evidence belongs to the
  #1516 acceptance campaign and is not replaced by these correctness and
  allocation-path tests.
- The merged strided-rs pin must not be moved to an unmerged branch.
