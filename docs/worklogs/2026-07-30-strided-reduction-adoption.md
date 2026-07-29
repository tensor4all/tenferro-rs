# Strided Reduction Kernel Adoption

## Summary

Advanced every workspace strided-rs dependency from `ed3053a6` to merged
commit `1460ab4b`. The new upstream kernel gives compact full sum and product
reductions a fixed multi-accumulator implementation while preserving
tenferro's explicit CPU `ExecContext` threading boundary.

## Context Read

- tenferro-rs issues #1490 and #1505
- strided-rs issues #149 and #167
- strided-rs PR #176 and its accumulation-order policy comments
- `AGENTS.md`, `REPOSITORY_RULES.md`, and the shared tensor4all Rust,
  performance, numerical, documentation, and test rules
- the root-facade oracle replay harness and nightly workflow

## Design And Numerical Contract

- The adoption changes only the merged dependency pin. CPU reduction replay
  already passes the backend-owned bounded `ExecContext`; no ambient Rayon
  state is introduced.
- Compact full reductions now use upstream's documented fixed association.
  `ExecContext::serial()` and `ExecContext::max_threads(1)` select the same
  serial algorithm.
- Floating reassociation may change roundoff, signed zero, NaN details, and
  intermediate overflow or underflow classification. Arithmetic remains in
  the input dtype, fixed-context reproducibility remains required, and no
  tenferro oracle tolerance or benchmark threshold is changed.
- Noncompact and axis reductions retain their existing traversal in this
  adoption. Fused sum-of-squares remains the separate #1505 close-out step.

## Performance Evidence

The focused public API benchmark used the full profile with 15 measured runs
after three warmups. The f64 `8192x4096` rows were pinned to CPU 60 for one
thread and CPUs 56-59 for four threads.

| Row | Baseline t1 | Candidate t1 | Baseline t4 | Candidate t4 |
| --- | ---: | ---: | ---: | ---: |
| sum eager | 28.207 ms | 10.163 ms | 8.040 ms | 7.583 ms |
| sum trace | 28.332 ms | 10.210 ms | 7.652 ms | 7.398 ms |
| product eager | 28.308 ms | 10.162 ms | 8.443 ms | 7.459 ms |
| product trace | 28.315 ms | 10.236 ms | 8.098 ms | 8.314 ms |

Same-machine PyTorch measured 10.130/10.133 ms at one thread and
5.894/5.856 ms at four threads for sum/product. The candidate is within 2x of
that reference, and the worst baseline-relative change is +2.7 percent.

## Verification

- The upstream PR passed strided-rs workspace tests, documentation, formatting,
  and the 53-file coverage policy on Linux CI, plus the macOS test job.
- tenferro's fast-check passed with the focused CPU reduction tests.
- The full root-facade linalg oracle replay used 56 workers and passed all
  2,090 supported-success records plus both expected-error records. It
  classified 7,493 of 9,585 records as intentionally unsupported and skipped
  no records by filter.
- The repository-rules review is run against the committed adoption diff
  before the PR is opened.

## Remaining Work

- #1505 owns the fused single-pass sum-of-squares implementation and its
  `norm_fro` acceptance measurement.
- The final full public API campaign and cross-row gate verdict remain under
  #1490.
