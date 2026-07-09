# Issue 1333 Gauge Options

## Session Summary

GitHub issue #1333 was narrowed after #1338 had already landed SVD canonical
gauge support and the linalg AD route cleanup. This session kept the default
decomposition behavior as backend raw gauge and implemented only the remaining
opt-in forward gauge conventions:

- `EighGauge::CanonicalPivot` for Hermitian eigenvectors.
- `QrGauge::PositiveDiagonal` for QR factors.
- Public `QrOptions` and `qr_with_options` on backend, eager, and traced
  surfaces.
- `EighOptions::gauge`, with raw gauge as the default.

## Context Read

- `AGENTS.md`
- `CONTRIBUTING.md`
- `REPOSITORY_RULES.md`
- shared tensor4all common, Rust, performance, numerical, and docs/tests rules
- GitHub issue #1333
- recently merged PR #1338 context via issue and branch state
- linalg backend, eager, traced, extension-op, and AD route code
- linalg guide and existing linalg public-surface tests

## Decisions Made

- Kept gauge fixing opt-in. `svd`, `eigh`, and `qr` still use backend raw
  signs/phases by default.
- Treated gauge fixing as forward-output post-processing at the linalg
  extension/backend boundary. CPU and CUDA decomposition kernels remain
  unchanged.
- Did not add new gauge-aware AD rules. The existing AD formulas still operate
  on the chosen forward outputs; repeated or nearly repeated singular/eigen
  subspaces remain governed by the existing `derivative_eps` regularization.
- When AD internally reconstructs a full `Eigh` carrier for eigenvalue-only
  linearization, it uses `EighGauge::Raw` because that carrier is a derivative
  implementation detail rather than a public forward-output request.

## Rejected Or Deferred Alternatives

- No default gauge fixing.
- No broad gauge-aware AD policy or custom VJP/JVP rules solely for gauge
  conventions.
- No SVD truncation policy; users continue to slice returned factors.
- General non-Hermitian `eig` gauge handling remains separate from this issue.
- LQ remains out of scope because there is no current public LQ operation.

## Verification Performed

- `cargo fmt --all`
- `cargo test -p tenferro-linalg --features autodiff decomposition_options`
- `cargo test -p tenferro-linalg --features autodiff gauge`
- `cargo test -p tenferro-linalg --features autodiff`
- `cargo fmt --all --check`
- `git diff --check`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo clippy --manifest-path ext/tropical/Cargo.toml --all-targets -- -D warnings`
- `cargo test --workspace --release`
- `cargo llvm-cov --workspace --release --json --output-path coverage.json`
- `python3 scripts/check-coverage.py coverage.json`
- `cargo doc --workspace --no-deps`
- `python3 scripts/check-docs-site.py`
- `python3 scripts/repository-rules-review.py --base origin/main --head HEAD --output-json /tmp/repository-rules-review.json`

## Remaining Risks

- Gauge post-processing currently covers host tensors returned by the backend
  boundary. A future backend that returns non-host-resident decomposition
  outputs would need either placement-aware gauge handling or an explicit
  backend override.
