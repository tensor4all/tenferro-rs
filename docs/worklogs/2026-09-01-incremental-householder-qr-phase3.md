# Incremental Householder QR Phase 3: oracle-backed AD (#1735)

## Session summary

Added JVP/VJP support for the abstract accumulated-matrix tangent carried by
compact Householder QR state. The change imports the separately merged oracle
families, recovers canonical thin Q/R only as fixed AD residuals, reuses the
existing QR derivative, and transposes append through runtime-width state
cotangent splitting.

## Context reviewed

- `REPOSITORY_RULES.md`, especially Oracle Gate, Rule Source Of Truth, and AD
  Rule Coverage
- `docs/design/incremental-householder-qr.md`, `docs/spec/ad-contract.md`, and
  `docs/architecture/primitive-ad.md`
- linalg semantic extension transforms, QR rules, support manifest, extension
  metadata/runtime dispatch, and oracle replay adapter
- upstream `tensor-ad-oracles` generator, replay, schema, and QR math notes

## Oracle-first prerequisite

`tensor4all/tensor-ad-oracles#25` merged first as `8a4f95cd`. Its exact head
`699b2851` passed the required `replay` and `regenerate` checks. The vendored
slice includes the 60 generated cases, generator source, QR math note, and
registry entries without refreshing unrelated divergent vendor content.

The five `(op, family)` tuples are `incremental_householder_qr` with
`factor_qr`, `append_qr`, `from_factors_qr`, `selected_q_columns`, and `r`.
They cover F32/F64/C32/C64, tall/square/wide and tall-to-wide cases, PyTorch
JVP/VJP, centered finite differences, and adjoint consistency.

## Decisions

- `packed` carries the accumulated-matrix tangent; `coeff` is
  non-differentiable auxiliary state.
- Internal `HouseholderQrThinQ` recovers symbolic full thin Q as a fixed
  `OperationRole::Primary` residual. R recovery is fixed in the same gauge.
- R and selected-Q derivatives reuse the existing reduced-QR linearization.
- Factor import projects `dR` and `R_bar` to the upper-trapezoidal domain.
- Append uses an internal linear append operation with fixed primal shape
  anchors. Its custom transpose splits the cotangent at runtime from the old
  and appended widths, avoiding exact-static-shape assumptions.
- Internal residual operations have no public wrappers and remain explicitly
  unsupported as user-differentiable operations in the manifest.

## Review gate

The user-selected reviewer is `reviewer-flash`. The Phase-3 design amendment
received **Correct-to-merge** before Rust implementation.

The broad post-diff attempts timed out without verdicts, so review was split
into bounded lanes:

- Core AD math/rule lane: **Correct-to-merge**, no Critical or Important
  findings. Two non-blocking observations concerned the intentionally rank-2
  contract and a symbolic full-width selector optimization.
- Extension/semantic/oracle lane: **Correct-to-merge**, no Critical or
  Important findings. Static append-row/anchor-row checks and strict legacy
  ThinQ rejection were suggested as Minor consistency improvements and fixed
  before the candidate gate.

No post-review finding remains unresolved.

## Verification completed so far

- 60 incremental oracle records replayed through tenferro JVP/VJP on CPU-faer
- representative family replay including complex append
- `householder_qr_r_grad_matches_finite_difference`
- eager combined Q/R backward
- two sequential appends with gradients for the initial matrix and both blocks
- linalg AD manifest, metadata-invariant, CPU-session-admission tests
- all 168 `tenferro-linalg` library tests with `autodiff,cpu-faer`
- all 226 `tenferro-linalg` integration tests with `autodiff,cpu-faer`
- CPU-BLAS feature compilation
- vendored QR math registry and all 60 new records validate against the vendored
  schema

## Candidate verification

- `RUN_ORACLE_REPLAY=1 ORACLE_REPLAY_OP=incremental_householder_qr ...`:
  all 60 records passed (five families, four dtypes, three shapes).
- `cargo test -p tenferro-linalg --features autodiff,cpu-faer --lib`: 168
  passed.
- `cargo test -p tenferro-linalg --features autodiff,cpu-faer --test integration`:
  226 passed.
- CPU-BLAS-only, CPU-faer, combined providers, and CUDA feature compilation
  passed.
- `python3 scripts/ci/run_profile.py fmt` passed.
- Focused `cargo clippy ... --all-targets -- -D warnings` passed.
- `scripts/check-pr-fast.sh --coverage-reviewed` passed with the 60-record
  oracle replay as its focused test; this also passed workspace and standalone
  extension clippy, formatting, and documentation snippet checks.
- Vendored math registry and all 60 new JSONL records passed the vendored
  schema.

## Remaining risks

- CUDA primal support remains Phase 4; this phase preserves explicit
  unsupported primal behavior.
- Rank-deficient AD remains outside the differentiable domain.
- Hosted CI remains pending.
