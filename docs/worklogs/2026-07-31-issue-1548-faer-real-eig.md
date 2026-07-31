# 2026-07-31 Faer real eigendecomposition layout fix (#1548)

## Session summary

The Faer real-input `eig` conversion now appends each conjugate eigenvector as
a complete column. Commit `cb388a8` had changed the previous indexed
column-major writes into alternating `push` calls, which interleaved the two
columns by row and broke the returned `A V = V D` relation.

## Context read

- `AGENTS.md`, `REPOSITORY_RULES.md`, `CONTRIBUTING.md`
- `ai/contribution-workflows/bugfix-pr.md`
- `ai/contribution-workflows/repository-remediation.md`
- Current shared repository, performance, documentation/test, and Rust
  numerical rules

## Reference code consulted

- The parent of `cb388a8`, where the two conjugate vectors were written through
  `u[i + j * n]` and `u[i + (j + 1) * n]`
- The LAPACK real-eigenvalue conversion, which still writes complete
  column-major conjugate-pair columns
- The Faer and LAPACK values-only paths; neither constructs eigenvectors and
  neither needs a corresponding change

## Decision

Keep the pooled append-only initialization introduced by `cb388a8`, but use
one loop for column `j` and a second loop for column `j + 1`. This is the
smallest change that preserves both the initialized-length contract and the
public column-major tensor layout.

## Rejected alternatives

- Restoring indexed writes would conflict with the current pooled buffer
  initialization contract.
- Transposing or repairing the tensor after construction would add unnecessary
  work and would hide the incorrect write order.
- Changing the values-only or LAPACK paths was rejected because their existing
  ordering is already correct.

## Verification

Public `CpuBackend::eig` regression tests use the real rotation
`A = [[0, -1], [1, 0]]` for both `f32` and `f64` and every compiled CPU linear
algebra provider. They assert a finite relative residual for `A V - V D`. With
the interleaved Faer ordering restored temporarily, both Faer tests fail with
residual `5e-1`; with the fix, both residuals are exactly `0`. The same tests
also preserve Faer/BLAS provider parity when both providers are compiled.

```text
cargo test -p tenferro-linalg rotation_eig -- --nocapture
2 passed; f32 residual 0e0; f64 residual 0e0
cargo test -p tenferro-linalg --lib
118 passed
cargo fmt --all --check
passed
```

## Remaining risks

The regression covers the minimal two-dimensional conjugate pair directly.
Larger mixed real/conjugate spectra are not covered by a runnable regression
test and remain a residual risk. Add that matrix family if this minimal case
fails to protect later layout changes.
