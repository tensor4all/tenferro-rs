# Worklog: W9 Fixed CPU Comparison Dispatch

## Scope

Adopt the fixed-operation comparison entry point merged in strided-rs #180.
This addresses the `compare_lt` attribution recorded under #1490 without
changing comparison semantics, threading policy, or allocation ownership.

## Design

The workspace strided pin advances to
`fa312f9ef3766163540869ee8fb6be3e7199a3e0`.

CPU comparison maps `CompareDir` to `strided_kernel::CompareOp` once, before
entering the element traversal. Owned tensors and borrowed views share the same
helper. Real, integer, and Boolean dtypes use the fixed dispatch. Complex
equality remains on the existing generic closure because complex scalars do not
implement `PartialOrd`; ordered complex comparisons remain typed errors.

The destination is allocated from the existing backend-owned buffer pool.
`compare_into` validates destination injectivity before mutation and inherits
the explicit execution policy already installed by `CpuBackend`.

## Performance Evidence

The paired experiment was fixed before execution:

- baseline tenferro commit: `9e06764a3333b9b3eb464c2b7b873a3187bd128c`;
- candidate implementation commit: `5f9de40ab65535d144f2d6d17263eb2457172df2`
  (the later amendment adds tests and this worklog only);
- benchmark source: tenferro-benchmark
  `3cef513291099591ce94e419ebf06bd8630396b7`;
- release profile, Rust 1.97.1, default/faer CPU provider;
- AMD EPYC 7713P, t1 on CPU 60 and t4 on CPUs 60-63;
- `RAYON_NUM_THREADS`, `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and
  `MKL_NUM_THREADS` fixed to the row thread count;
- separate cold build targets, with both release builds completed before any
  timing;
- exact public API row `cpu/elementwise_reduction`, `compare_lt`, f64,
  33,554,432 elements, direct execution;
- 10 paired cells per thread count, each with 3 warmups and 15 samples;
- odd/even pairs reverse baseline/candidate execution order;
- statistic: paired `(candidate / baseline - 1)` mean with a two-sided 95%
  Student-t interval;
- blocking gate: interval upper bound above +20%;
- validity gates: every status is `ok`, selected CPUs are at least 90% idle in
  the three-second preflight, and every cell has IQR/median at most 15%.

Preflight idle was 98.33% or higher on CPUs 60-63. All 20 cells reported `ok`;
the maximum IQR/median was 3.65% at t1 and 7.73% at t4.

| Threads | Pair | Baseline ms | Candidate ms | Change |
|---:|---:|---:|---:|---:|
| 1 | 1 | 61.666 | 42.230 | -31.52% |
| 1 | 2 | 61.167 | 41.596 | -32.00% |
| 1 | 3 | 61.981 | 42.377 | -31.63% |
| 1 | 4 | 61.075 | 42.417 | -30.55% |
| 1 | 5 | 62.334 | 42.137 | -32.40% |
| 1 | 6 | 61.658 | 42.109 | -31.71% |
| 1 | 7 | 61.786 | 42.153 | -31.78% |
| 1 | 8 | 61.634 | 42.224 | -31.49% |
| 1 | 9 | 61.591 | 42.231 | -31.43% |
| 1 | 10 | 61.685 | 42.231 | -31.54% |
| 4 | 1 | 21.056 | 18.621 | -11.57% |
| 4 | 2 | 21.460 | 18.499 | -13.80% |
| 4 | 3 | 21.952 | 18.319 | -16.55% |
| 4 | 4 | 20.507 | 19.324 | -5.77% |
| 4 | 5 | 21.389 | 18.852 | -11.86% |
| 4 | 6 | 21.655 | 18.274 | -15.61% |
| 4 | 7 | 20.350 | 18.403 | -9.57% |
| 4 | 8 | 20.583 | 18.805 | -8.64% |
| 4 | 9 | 20.777 | 18.856 | -9.25% |
| 4 | 10 | 20.822 | 18.492 | -11.19% |

| Threads | Baseline median | Candidate median | Mean change | 95% CI | Verdict |
|---:|---:|---:|---:|---:|:---|
| 1 | 61.662 ms | 42.227 ms | -31.60% | [-31.94%, -31.27%] | PASS |
| 4 | 20.939 ms | 18.560 ms | -11.38% | [-13.74%, -9.03%] | PASS |

Only comparison dispatch changes. Neighboring elementwise operations retain
their existing call paths.

## Verification

- Owned and borrowed-view paths cover all five comparison directions,
  including unordered NaN behavior.
- The routing source-contract test fails against the baseline source and passes
  against the candidate.
- `cargo fmt --all --check`
- `cargo test -p tenferro-internal-cpu-kernels
  elementwise::tests::ordered_compare_fixed_dispatch_preserves_owned_and_view_semantics
  -- --exact`
- `cargo test -p tenferro-internal-cpu-kernels
  elementwise::tests::ordered_compare_owned_and_read_routes_use_fixed_dispatch
  -- --exact`
- full `tenferro-internal-cpu-kernels` tests: 44 passed, 20 doctests passed
- repository fast check and repository-rules review before PR
