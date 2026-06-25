# FFI Safety Comment Sweep

## Summary

Fixes #1169 by documenting the local safety invariants for LAPACK and CUDA
linalg `unsafe` blocks. The change does not alter linalg algorithms, public API,
or backend dispatch; it keeps the unsafe contracts reviewable at each FFI or
unchecked kernel launch site.

## Context Read

- `CONTRIBUTING.md`
- `AGENTS.md`
- `REPOSITORY_RULES.md`
- `ai/contribution-workflows/bugfix-pr.md`
- `ai/contribution-workflows/repository-remediation.md`
- GitHub issue #1169 and reopen comment after #1203
- `crates/tenferro-linalg/src/cpu/linalg/lapack_linalg/*.rs`
- `crates/tenferro-linalg/src/gpu/linalg.rs`
- `crates/tenferro-linalg/tests/cpu_linalg_source_contract.rs`
- `crates/tenferro-linalg/tests/gpu_linalg_source_contract.rs`

## Classification Ledger

Issue #1169 is a repository-rule remediation finding: many LAPACK and CUDA
linalg `unsafe` blocks existed without nearby `// SAFETY:` comments. It is an
auto-fix because the intended behavior is already defined by
`REPOSITORY_RULES.md`: every unsafe block must document the validation
invariant next to the block. No new public API, dependency, feature flag,
backend, or AD semantics were needed.

The same-root-cause scan covered the full CPU LAPACK linalg wrapper directory
and `src/gpu/linalg.rs`, the file named by the issue for cuSOLVER/cuBLAS/CUDA
kernel call sites. The PR adds source-contract tests that fail if future
`unsafe {` blocks in those scopes lack a nearby `// SAFETY:` comment.

## Decisions

- Keep comments local to each unsafe block rather than moving safety contracts
  into module-level prose, because the repository rule asks reviewers to see
  the validation invariant adjacent to the unsafe operation.
- In the GPU path, group adjacent pointer-construction calls into one unsafe
  block when they share the same checked-offset contract. The grouping keeps the
  unsafe scope narrow while avoiding repeated comments for equivalent pointer
  derivations in a single batch iteration.
- Use source-contract tests instead of behavioral numerical tests for this
  specific finding. The behavior already has linalg coverage; the regression is
  absence of local safety documentation.

## Verification

- `cargo fmt --all --check`
- `cargo test -p tenferro-linalg --test cpu_linalg_source_contract lapack_ffi_unsafe_blocks_document_safety_invariants -- --nocapture`
- `cargo test -p tenferro-linalg --test gpu_linalg_source_contract gpu_linalg_unsafe_blocks_document_safety_invariants -- --nocapture`
- `cargo test -p tenferro-linalg`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo clippy --manifest-path ext/tropical/Cargo.toml --all-targets -- -D warnings`
- `cargo test --workspace --release`
- `cargo llvm-cov --workspace --release --json --output-path /tmp/tenferro-workspace-coverage-1169.json`
- `python3 scripts/check-coverage.py /tmp/tenferro-workspace-coverage-1169.json`
- `cargo doc --workspace --no-deps`
- `python3 scripts/check-docs-site.py`
- `python3 scripts/repository-rules-review.py --base origin/main --head HEAD --worktree --output-json /tmp/repository-rules-review-1169-worktree.json`
- `python3 scripts/repository-rules-review.py --base origin/main --head HEAD --output-json /tmp/repository-rules-review-1169.json`

## Residual Risk

The source-contract tests intentionally check only literal `unsafe {` blocks in
the CPU LAPACK wrapper directory and `src/gpu/linalg.rs`. They will not detect
`unsafe fn`, macro-generated unsafe blocks that do not contain the literal text,
or future FFI files outside those scopes. That is acceptable for #1169 because
the issue named these linalg paths; broader repository-wide unsafe linting can
be handled by a separate rule or audit if needed.
