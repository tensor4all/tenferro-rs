# AI Documentation Issue Batch

## Session Summary

This batch addressed the recent AI-reported GitHub issues #1002 through #1014
from a branch based on `origin/main`. The work was initially requested as a
local-only fix batch and was later promoted into a draft PR after the local
batch had been verified.

- Clarified traced and eager AD guide setup, `AdContext` versus
  `TracedTensorAdExt`, and added concrete VJP/JVP value assertions.
- Updated tutorial-code snippets to use column-major constructors consistently.
- Documented eager operation method coverage and added a reverse-mode `matmul`
  example.
- Expanded linalg guide coverage for operation surfaces, decomposition output
  ordering, reconstruction examples, QR/eigh shape contracts, and complete-pivot
  LU conventions.
- Added dependency setup and value assertions across einsum, FFT, tensor, and
  getting-started docs.
- Fixed FFT unsupported AD error messages so `rfft` VJP and `irfft` JVP report
  the failing operation family and AD path accurately.

## Context Read

- `AGENTS.md`
- `CONTRIBUTING.md`
- `REPOSITORY_RULES.md`
- shared tensor4all common, Rust, performance, numerical, docs/tests, and
  repository rules
- `ai/contribution-workflows/bugfix-pr.md`
- GitHub issues #1002 through #1014
- AD traced and eager docs and tutorial-code harness
- FFT AD rule implementation and regression tests
- linalg backend API docs and linalg public guide
- einsum, tensor operations, FFT, and getting-started guides

## Decisions Made

- Kept the FFT behavior unchanged and fixed only the unsupported-error
  reporting path. `rfft` and `irfft` AD remain unsupported until Hermitian
  half-spectrum rules are implemented.
- Mapped VJP errors from the linearization phase to VJP wording at the public
  traced AD boundary, while preserving JVP wording for direct JVP failures.
- Treated guide snippets as executable examples where practical by adding
  concrete value assertions and reconstruction checks instead of shape-only
  assertions.
- Added dependency snippets directly to the affected guides rather than
  assuming readers have already followed the getting-started page.

## Rejected Or Deferred Alternatives

- Did not add new AD rules, public APIs, dependencies, or backend behavior.
- Kept PR preparation deferred during the initial local-only pass. The branch
  was later promoted to a draft PR after the user changed direction.
- Did not run the full release workspace test, coverage, or clippy checklist
  because the requested scope was a local batch fix and the touched code path
  was covered by targeted tests.

## Verification Performed

- `cargo fmt --all`
- `python3 scripts/check-doc-snippets.py`
- `cargo test -p tenferro-tutorial-code`
- `cargo test -p tenferro-linalg --doc`
- `cargo test -p tenferro-fft --features autodiff --test fft_ops unsupported_error_names`
- `python3 scripts/check-doc-snippets.py --check`
- `cargo fmt --all --check`
- `cargo doc --workspace --no-deps`
- `python3 scripts/check-docs-site.py`
- `cargo test -p tenferro-einsum --features autodiff tensordot`
- `cargo test -p tenferro-linalg full_piv_lu`
- `cargo test -p tenferro-ad --doc`
- `cargo test -p tenferro-fft --features autodiff --test fft_ops`
- `git diff --check`

## Remaining Risks

- Markdown guide examples are not all compiled by an automated markdown
  doctest harness; verification relied on existing tutorial-code, doctests,
  package tests, and docs site checks.
- Full release workspace tests, coverage, and clippy were not run in this local
  no-PR pass.
