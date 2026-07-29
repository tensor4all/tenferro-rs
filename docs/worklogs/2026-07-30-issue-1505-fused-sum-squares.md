# Issue #1505 fused sum-of-squares close-out

## Session summary

The merged pow-to-mul change removed the generic transcendental pass from
Frobenius norms, but real `f64` still materialized the square before reducing
it. This change adds a closed `ReduceSumSquares { axes }` core semantic
primitive and maps the CPU backend to strided-kernel's one-pass
`ReduceOp::SumSquares`.

The primitive is an execution hook, not a new user-facing norm API. Eager,
direct, and traced Frobenius paths all use it; `p_norm(..., 2.0)` continues to
delegate to the Frobenius helper.

## Context read

- tenferro-rs #1505, including the merged step-1 measurements
- tenferro-rs #1490 benchmark gate policy
- strided-rs #149 reduction accumulation-order policy
- strided-rs #167 and merged PR #177
- `REPOSITORY_RULES.md`, especially explicit execution context, numerical
  oracle, and benchmark evidence requirements

## Decisions made

- Use a closed semantic primitive rather than compiler-only pattern fusion.
  Eager execution materializes `Mul` before graph compilation, so an
  execution-IR rewrite cannot fix the eager row.
- Treat `reduce_sum_squares` as a supported eager/traced reduction operation,
  not a hidden sibling-crate helper. Its same-dtype square, dtype support,
  empty-axis behavior, AD, and backend error contract are documented and
  tested; the linalg layer consumes that public operation without exposing
  lower-level graph construction APIs.
- Keep the hidden backend default explicit: it returns typed unsupported
  instead of materializing a square. `CpuBackend` and `CpuExecSession`
  override with strided-kernel and an explicit bounded `ExecContext`.
- Preserve the previously supported CUDA and XLA surfaces explicitly. CUDA
  uses a CubeCL map-square reduction kernel on the first reduced axis and
  ordinary sum kernels on remaining axes. Its generated operation uses
  `fma(x, x, 0)` as a square-rounding barrier so NVRTC cannot contract the
  later accumulation into FMA. XLA lowers the primitive to a logical
  StableHLO multiply feeding reduce, preserving device execution without a
  host fallback.
- Restrict the primitive to `f32` and `f64`. Complex Frobenius norm retains
  the existing complex magnitude pass, whose real output is then consumed by
  the fused reduction.
- Define empty axes as elementwise square, matching `ReduceSum(Mul(x, x), [])`.
  The CPU read hook delegates that case to the existing pooled elementwise
  multiply implementation.
- Give both semantic AD and the legacy primitive-rule registry explicit JVP
  and transpose rules equivalent to `2 * x * tangent` and
  `2 * x * broadcast(cotangent)`.

## Rejected or deferred alternatives

- Compiler-only `ReduceSum(Mul(x, x))` recognition was rejected because it
  leaves eager execution materializing the square.
- Backend history inference and bespoke eager recording were rejected because
  they introduce hidden state across ownership boundaries.
- Complex magnitude-square fusion remains deferred; this PR preserves the
  existing complex `abs` semantics and targets the confirmed real reduction
  bottleneck.

## Performance evidence

Focused public API subset, full profile, 3 warmups and 15 measured runs. The
baseline is tenferro-rs `39e96af578f04b19b4be9515b82cacda6e26a281`; the
candidate uses strided-rs `95a607c061e95cc3a698a083dd279283a905f99c`.

| row | threads | baseline ms | candidate ms | change |
| --- | ---: | ---: | ---: | ---: |
| `norm_fro` f64 eager | 1 | 22.568 | 1.002 | -95.6% |
| `norm_fro` f64 trace | 1 | 5.506 | 1.022 | -81.4% |
| `norm_fro` c64 eager | 1 | 25.894 | 23.234 | -10.3% |
| `norm_fro` c64 trace | 1 | 23.029 | 19.763 | -14.2% |
| `norm_fro` f64 eager | 4 | 10.113 | 1.070 | -89.4% |
| `norm_fro` f64 trace | 4 | 2.435 | 1.108 | -54.5% |
| `norm_fro` c64 eager | 4 | 10.337 | 9.531 | -7.8% |
| `norm_fro` c64 trace | 4 | 6.778 | 5.607 | -17.3% |

The `f64` single-thread eager result is also below the same-run PyTorch
baseline of 1.205 ms, comfortably satisfying the issue's within-3x
acceptance. Baseline and candidate were both rerun on CPU 60 for one thread
and CPUs 60-63 for four threads to avoid a persistent process on CPU 56.

## Verification performed

- CPU fused reduction tests for `f32`, `f64`, empty axes, and non-compact views
- tensor backend default-hook typed-unsupported test
- semantic JVP/VJP structure test
- Frobenius norm directional JVP and gradient finite-difference tests
- direct eager `reduce_sum_squares` gradient finite-difference oracle
- direct legacy transpose test distinguishing the linear key from the retained
  primal key
- CUDA A100 differential execution for `f32`/`f64`, multi-axis and empty-axis
  semantics, and integer typed errors
- CUDA A100 bitwise test with an input that distinguishes multiply-then-add
  from contracted FMA
- CUDA runtime-preparation routing test
- XLA StableHLO multiply-plus-reduce lowering test
- focused eager, concrete, traced, and AD norm integration tests
- internal primitive AD registry and catalog coverage tests
- focused public API benchmark at one and four threads
- full local oracle replay: 9,585 records, 2,090 supported successes, 2
  expected errors, 7,493 unsupported records, 56 parallel jobs

## Remaining risks and follow-up work

- Complex Frobenius norm still materializes magnitude before the fused real
  reduction. Its residual cost is not addressed without separate profile
  evidence.
- The XLA preservation lowering expresses multiply followed by reduce in
  StableHLO. Physical producer/reduction fusion is owned by XLA and is not
  guaranteed by tenferro's textual lowering; W6's single-pass acceptance and
  benchmark evidence cover the CPU strided implementation. A custom XLA
  reduction is separate backend work if physical non-materialization must
  become a tenferro-level guarantee there.
- The broader final public API campaign and cross-workload gate verdict remain
  tracked by #1490.
