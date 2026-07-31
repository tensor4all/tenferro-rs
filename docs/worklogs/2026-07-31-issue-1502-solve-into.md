# Issue #1502: caller-owned eager solve destination

## Scope

This work adds the public `solve_read_into` path for eager linear solves. The
default backend contract remains allocate-then-copy; the CPU Faer and LAPACK
providers use a validated direct destination path for compatible host,
column-major layouts. `solve` and `solve_read` retain their existing
owned-result APIs and share the same provider kernels.

## Context read

- `AGENTS.md`, `REPOSITORY_RULES.md`, and the shared tensor4all rules.
- Issue #1502 and the merged #1525 session/admission work.
- Existing `TensorRead`/`TensorWrite` metadata and overlap validation.
- CPU buffer-pool ownership and the Faer/LAPACK solve implementations.

## Decisions

- Add one trait hook, `LinalgBackend::solve_read_into`, plus the borrowed
  `TensorReadLinalgExt` forwarding method. External backend implementations get
  the safe allocate-plus-copy default.
- Validate shape, dtype, placement, and detectable input/destination overlap
  before either the direct provider path or the default result allocation.
- The direct CPU path accepts rank-2 A, vector or rank-2 B, and positive
  column-major output with unit row stride. Matrix output permits a padded
  leading dimension and a nonzero view offset. Unsupported layouts use the
  default owned-result fallback.
- Faer and LAPACK pack A directly into one pooled factor buffer, copy strided
  RHS values into the caller destination, and solve in place. The ordinary
  owned-result path uses the same in-place provider kernel after allocating its
  compact result through the shared pooled helper.
- Singular validation occurs before the destination RHS is copied, so the
  documented singular/validation failure leaves `out` unchanged. Provider
  failures after execution begins retain the existing unpublished-output
  boundary.
- No prepared solve plan or process-global cache was introduced.

## Alternatives rejected or deferred

- A fused single-pass solve allocation cache was not added; the issue requests
  caller-owned output, not a new global or prepared-plan lifetime.
- Arbitrary strided output writes were not forced through the direct path. The
  fallback preserves correctness until a provider-native strided destination
  contract exists.
- The #1525 scheduler/session implementation was not changed. The solve path
  reuses its established CPU context admission and is measured separately.

## Implementation

- Added `validate_read_into_destination` to the tensor backend boundary and
  reused it for solve destination validation.
- Added the default and CPU override for `solve_read_into`, including host and
  provider checks and direct-layout eligibility.
- Added shared Faer and LAPACK view-based solve helpers for real and complex
  dtypes, vector and multiple-RHS cases, padded leading dimensions, and view
  offsets.
- Reworked ordinary CPU `solve`/`solve_read` to use the same view-based kernel
  where rank/layout eligibility permits, without adding a second pool entry.
- Added dtype, multiple-RHS, public extension, padded/offset view, sentinel,
  mismatch, overlap, and singular atomicity tests, plus a source-contract test
  that keeps the CPU override from silently disappearing.

## Verification

Passed locally:

- `cargo fmt --all`
- `git diff --check`
- `cargo test -p tenferro-linalg --lib --no-default-features --features cpu-faer`
  (120 tests)
- `cargo test -p tenferro-linalg --test integration --no-default-features --features cpu-faer cpu_linalg_source_contract`
  (17 tests)
- `cargo check -p tenferro-linalg --lib --no-default-features --features cpu-faer`
- `cargo check -p tenferro-linalg --lib --no-default-features --features cpu-blas`
- `cargo check -p tenferro-linalg --tests --no-default-features --features cpu-blas`
- `cargo test -p tenferro-linalg --doc --no-default-features --features cpu-faer`
  (62 doctests)

The BLAS runtime test was not linkable on this host because system BLAS/LAPACK
symbols are unavailable; the `cpu-blas` library and test targets compile. The
repository BLAS CI lane remains required for the PR.

## Focused public API measurement

The release benchmark compared the existing allocating `solve_read` control
with caller-owned `solve_read_into` on the same f64 diagonal matrix and four
RHS columns. It used `taskset -c 0-3`, two warmups, and fixed repetitions. The
allocating row is the before/control path; the caller-owned row is the
candidate path. Values are median-like per-call means from the focused run in
milliseconds.

| threads | shape | allocating `solve_read` | caller-owned `solve_read_into` | ratio |
| ---: | --- | ---: | ---: | ---: |
| 1 | 32x32, 4 RHS | 0.018726 | 0.019038 | 1.0166 |
| 1 | 128x128, 4 RHS | 0.163904 | 0.161107 | 0.9829 |
| 4 | 32x32, 4 RHS | 0.024043 | 0.023406 | 0.9735 |
| 4 | 128x128, 4 RHS | 0.208381 | 0.197415 | 0.9474 |

The small t1 row is within measurement noise and is not claimed as a speedup;
the medium rows show that the direct destination path does not add a material
solve regression and removes the result handoff. This is focused evidence, not
the final public benchmark campaign.

## Remaining verification

Run the repository fast gate, full CPU Faer/BLAS CI, coverage/docs/clippy, and
rules review on the committed PR head. After merge, rerun the #1525 `inv`
direct/trace rows as a non-regression check and report the result on #1502 and
umbrella #1535.
