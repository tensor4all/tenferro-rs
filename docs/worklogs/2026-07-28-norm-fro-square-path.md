# 2026-07-28 norm_fro square path

## Summary

Fixed the confirmed `norm_fro` square-path bug from #1505 by replacing the
generic `pow(x, 2.0)` square with `x * x` across the eager, concrete/read, and
traced linalg composites. The same p=2 fast path now delegates to the
Frobenius helper so it cannot regress back to generic `pow`.

I also adopted the issue's second low-risk ambition for real dtypes: real
Frobenius and non-matrix p=2 norms skip the pre-square `abs` materialization.
Complex norms still use the existing absolute-value path because closing that
gap cleanly needs a dedicated complex sum-of-squares expression or the later
single-pass reduction work.

## Context Read

- `REPOSITORY_RULES.md` and `AGENTS.md`
- `CONTRIBUTING.md`
- `ai/contribution-workflows/bugfix-pr.md`
- `ai/contribution-workflows/repository-remediation.md`
- #1505 body
- #1490 benchmark triage context from the task
- `crates/tenferro-linalg/src/eager_composites.rs`
- `crates/tenferro-linalg/src/tensor_ext.rs`
- `crates/tenferro-linalg/src/traced.rs`
- `tenferro-benchmark/src/bin/benchmark_cpu_public_api.rs`

## Decisions

- Keep the PR as a bug-fix PR. It adds no public API, dependency, feature flag,
  or backend policy change.
- Fix the same-root-cause instances in all three linalg surfaces:
  eager composite, concrete/read extension surface, and traced composite.
- Use source-contract tests for the internal path choice because the public
  numerical result remains the same while the bug is specifically a wrong
  internal operator choice.
- Do not implement fused multiply-accumulate sum-of-squares in this PR. That
  would change reduction accumulation behavior and is explicitly blocked until
  the reduction policy step.

## Benchmark Evidence

Focused public API subset:

```bash
PUBLICATION_GATE_PROFILE=full \
BENCH_RUNS=15 \
BENCH_WARMUPS=3 \
TENFERRO_CPU_FEATURES=cpu-faer \
TENFERRO_CPU_BACKEND_KIND=default \
PUBLIC_API_BENCHMARK_FILTER=norm_fro \
taskset -c 8-15 nice -n 10 ... --num-threads {1,4}
```

The host had other CPU-heavy processes, so these are local diagnostic
before/after numbers rather than the final formal benchmark campaign.

| row | before ms | after ms | change |
| --- | ---: | ---: | ---: |
| f64 t1 tenferro-eager | 143.961 | 29.768 | -79.3% |
| f64 t1 tenferro-trace | 60.845 | 7.358 | -87.9% |
| f64 t4 tenferro-eager | 58.554 | 11.176 | -80.9% |
| f64 t4 tenferro-trace | 18.886 | 2.113 | -88.8% |
| c64 t1 tenferro-eager | 62.919 | 31.036 | -50.7% |
| c64 t1 tenferro-trace | 62.555 | 25.485 | -59.3% |
| c64 t4 tenferro-eager | 20.803 | 8.293 | -60.1% |
| c64 t4 tenferro-trace | 17.827 | 6.682 | -62.5% |

The same run measured PyTorch `norm_fro` f64 t1 at 1.304 ms. The remaining
f64 eager gap is therefore still larger than the #1505 3x target on this
machine. A focused `reduce_sum_all` run under the same settings measured
tenferro t1 at 29.352 ms vs PyTorch t1 at 10.278 ms on `8192x4096`, confirming
that the residual belongs to the reduction/single-pass sum-of-squares cluster
rather than another generic `pow` path.

## Verification

Commands run:

```bash
cargo test -p tenferro-linalg --features cpu-faer,autodiff --test integration real_sum_of_squares_norm_skips_abs_materialization_before_square -- --nocapture
cargo test -p tenferro-linalg --features cpu-faer,autodiff --test integration norm_fro_and_p2_norm_square_with_mul_not_generic_pow -- --nocapture
cargo test -p tenferro-linalg --features cpu-faer,autodiff --test integration eager_norm -- --nocapture
cargo test -p tenferro-linalg --features cpu-faer,autodiff --test integration concrete_norm -- --nocapture
cargo test -p tenferro-linalg --features cpu-faer,autodiff --test integration traced_correctness::norm -- --nocapture
cargo test -p tenferro-linalg --features cpu-faer,autodiff --test integration norm -- --nocapture
cargo check -p tenferro-linalg --features cpu-faer,autodiff --tests
cargo fmt --all --check
cargo llvm-cov -p tenferro-linalg --features cpu-faer,autodiff --test integration --profile ci --json --output-path /tmp/tenferro-linalg-coverage-after.json
```

After the first PR run, hosted `coverage` reported
`crates/tenferro-linalg/src/eager_composites.rs: 79.5% < 80%`. I added eager
execution coverage for no-op `dim=[]`, vector infinity norms, vector p-norm,
and complex p=2 norm. The local linalg coverage rerun reported
`eager_composites.rs` at 81.46%.

## Remaining Risk

- #1505 should not be closed from this PR alone unless maintainers accept the
  residual as covered by the reduction cluster. The local f64 t1 public API row
  is still not within 3x of PyTorch.
- Complex Frobenius still materializes `abs` before squaring. It improved by
  removing `pow`, but closing the remaining gap should be handled with the
  reduction policy work or a separately benchmarked complex sum-of-squares path.
