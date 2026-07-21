# Ordinary solve scale invariance

## Session summary

Fixed CPU ordinary solve rejecting invertible matrices solely because every
entry was small. The Faer adapter had reused the tolerance-based singularity
rule from full-pivot LU rank detection, which is not the ordinary solve
contract.

## Context inspected

- `FaerLinalg::solve_2d` for real and complex scalar families.
- Faer's partial-pivot LU factorization and Tenferro's full-pivot LU paths.
- LAPACK `GETRF` solve behavior and the shared `DiagSingularity` validator.
- Owned, borrowed, prepared, and traced solve tests.

## Decisions

- Ordinary solve detects singular Faer pivots by exact zero. Complex pivots are
  checked componentwise so a representable component is not squared into zero.
- Full-pivot LU keeps its tolerance because that operation exposes numerical
  rank detection rather than the ordinary solve contract.
- The shared prepared-factor validator keeps rejecting exact-zero and
  nonfinite values. That is its existing cross-provider validation contract.
- No Faer-only nonfinite rule was added to direct solve. LAPACK may propagate
  nonfinite input, so changing that policy requires a separate provider-wide
  contract rather than an incidental scale-invariance fix.

## Verification

- Tiny nonzero real and complex pivots across compiled Faer solve surfaces.
- Existing exact-singular real and complex solve tests.
- Tiny nonzero provider-parity test with Accelerate LAPACK.
- Shared complex diagonal underflow regression tests.
- Traced complex solve regression test.
- Formatting and strict Clippy for the affected crates.

## Remaining risk

CUDA was outside this CPU/common fix and was not tested. Direct-solve behavior
for NaN and infinity remains provider-defined until Tenferro adopts an explicit
cross-provider nonfinite-input policy.
