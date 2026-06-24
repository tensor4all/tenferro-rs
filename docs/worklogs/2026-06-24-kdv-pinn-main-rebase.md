# 2026-06-24 KdV PINN main rebase

## Session Summary

Integrated the KdV PINN sample from PR #1062 onto current `origin/main`,
updated it for the current fallible traced-runtime APIs, and added an online
tutorial page that links the sample into the Quarto docs sidebar.

## Context Read

- `AGENTS.md` and `REPOSITORY_RULES.md`
- Shared repository and docs/test rules from `tensor4all`
- PR #1062 discussion and unresolved review feedback
- Existing tutorial structure under `docs/tutorials`
- CodeGraph index for the worktree before locating code paths

## Decisions Made

- Kept `kdv_pinn` as a workspace package instead of moving it under
  `docs/tutorial-code`, because the full training run is longer than the small
  tutorial binaries.
- Added `docs/tutorials/kdv-pinn.md` as an advanced tutorial that explains the
  sample structure, run commands, output options, and traced-AD residual
  pattern.
- Expanded the KdV tutorial after readability feedback so it defines the KdV
  equation, the neural model, sampled point sets, and the PINN objective with
  rendered math before the run instructions.
- Limited `plotters` features to bitmap PNG/GIF and line-series rendering so
  the sample does not require system `fontconfig` packages in workspace-wide CI.
- Added coverage thresholds for the KdV sample binary, because coverage CI
  exercises module tests but intentionally does not run the full training loop
  or optional image writers.
- Applied the PDE loss weight after `mean(residual^2)` so the scalar weight is
  not squared implicitly.
- Updated KdV PINN graph construction and tests for APIs that now return
  `Result`.

## Verification Performed

- `cargo test -p kdv_pinn`
- `cargo fmt --all --check`
- `cargo clippy -p kdv_pinn --all-targets -- -D warnings`
- `cargo doc --workspace --no-deps`
- `quarto render docs`
- `/opt/homebrew/bin/python3.12 scripts/check-docs-site.py --root-dir . --quiet`
- `cargo test -p tenferro-fft --test fft_ops`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test --workspace --release --no-run`

## Remaining Risks

`cargo test --workspace` reached the `tenferro-fft` integration tests and was
killed by SIGKILL once in this local environment. A later targeted rerun of the
same `tenferro-fft --test fft_ops` binary passed, so the local full-workspace
failure was not reproduced.
