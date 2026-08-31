# Incremental Householder QR Phase 2 (#1735)

## Summary

Implemented the opaque compact state, concrete/eager/traced operation surfaces,
linalg extension IR, explicit unsupported AD manifest entries, and CPU-faer and
CPU-BLAS/LAPACK primal providers.

## Implementation

- Public `HouseholderQr<T>` keeps packed reflectors and coefficients private.
- Extension operations cover factor, factor import, append, R extraction, and
  contiguous Q-column materialization.
- CPU append applies old reflectors to only the new block and factors only the
  trailing residual.
- LAPACK uses `geqrf` and `ormqr`/`unmqr` with checked LP64 dimensions and
  workspace queries.
- faer uses elementary `make_householder_in_place`, stores reciprocal faer tau
  in the provider-neutral coefficient convention, and applies packed state
  through zero-copy faer matrix views.
- `from_factors` factors Q and folds its triangular factor into R. The accepted
  contract was clarified to require exactly upper-trapezoidal R; arbitrary R
  would require refactorizing the full product or a more complex reflector
  composition and is not a previously computed QR factor.
- CUDA session admission rejects the new operations until Phase 4.
- All five new AD manifest entries remain explicitly unsupported pending the
  Phase-3 oracle family.

## Verification

Passed locally:

- `cargo check -p tenferro-linalg --features autodiff`
- `cargo check -p tenferro-linalg --tests --features autodiff`
- `cargo check -p tenferro-linalg --no-default-features --features cpu-blas,autodiff`
- `cargo check -p tenferro-linalg --features cuda,autodiff`
- focused CPU-faer reconstruction tests for factor, append, factor import,
  rank deficiency, zero append, tall-to-wide transition, and all four dtypes
- concrete, eager, and traced public-surface tests
- linalg AD manifest and extension tests
- all 181 tenferro-linalg doctests
- `scripts/check-public-error-docs.py`
- repository formatting profile
- `scripts/check-pr-fast.sh --coverage-reviewed` with the all-dtype append
  reconstruction test after closure review

CPU-BLAS runtime tests could not link in this local shell because no native
BLAS/LAPACK symbols were configured. The `cpu-blas` Rust code and tests compile;
provider-linked CI remains the runtime verification owner.

## Review gate

- Design review: `reviewer-flash` Round 3, **Correct-to-merge**.
- Post-implementation full-diff review Round 1: **Findings-require-fix**.
  Fixed faer complex append conjugation, added C32/C64 append reconstruction,
  and changed Householder AD from silent absent gradients to explicit
  `Unsupported` with an end-to-end rejection test.
- Closure review: `reviewer-flash`, **Correct-to-merge**. No remaining
  Critical, Important, or Minor findings in the corrected regions.

## Remaining scope

Phase 3 owns oracle and AD support. Phase 4 owns CUDA execution and hardware
tests. Phase 5 owns the predeclared performance gate and final user docs.
