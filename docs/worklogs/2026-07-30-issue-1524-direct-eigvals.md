# Direct `eigvals` values-only dispatch

Issue: #1524

## Scope

- Route the owned public `TensorLinalgExt::eigvals` surface through the backend
  values-only hook.
- Add a backend-spy regression test that fails if the public surface dispatches
  to the full eigendecomposition.
- Measure the exact `cpu/linalg_uncovered/eigvals` public API row at one and
  four threads.

## Decision

The traced path already dispatches to `LinalgBackend::eig_values`. The owned
path must use the same values-only backend contract instead of computing and
discarding eigenvectors through `eig`.

The read surface is unchanged because it has a separate borrowed-input
contract and no values-only read hook.

## Verification

- Focused backend-spy regression test.
- `cpu/linalg_uncovered/eigvals`, 3 warmups and 15 samples on an AMD
  EPYC 7713P:
  - threads 1, CPU 60: `9.737 ms` before, `7.447 ms` after (`-23.5%`)
  - threads 4, CPUs 60-63: `11.424 ms` before, `8.507 ms` after (`-25.5%`)
- The after values match the existing traced values-only path (`7.505 ms` and
  `8.560 ms`, respectively), resolving the measured layer asymmetry.
- CPUs 60-63 were 100% idle in the pre-measurement sample.
- Repository fast check and repository-rules review.
