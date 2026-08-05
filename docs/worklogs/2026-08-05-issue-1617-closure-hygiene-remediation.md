# Issue #1617 closure hygiene remediation work log

Date: 2026-08-05

## Scope and decision

This remediation follows the approved Issue #1617 design in one branch and one
non-squash PR. Apple M5 Max Metal/WebGPU execution was supplied separately and
is recorded in the merged hardware matrix. No checksum, signature, attestation,
nonce, hostile-runner defense, generic runner, extra dependency, compatibility
shim, or hidden transfer is introduced.

The product/checker candidate is selected only after the changes below and the
pre-candidate checks pass. Candidate-bound reports are generated after that
commit; evidence descendants may change only the closed evidence allowlist.

## Classification ledger

| Finding | Disposition | Evidence / implementation |
|---|---|---|
| T0 | Fixed | `apple_context.rs` now destructures through `&mut managed`; CI cross-checks the macOS-gated integration target from Linux. |
| T1 | Fixed by regeneration protocol | Freeze, static-rank, traversal, hardware, docs-audit, and closure reports use one candidate. Stale reports are rejected by the freeze evidence-path check. |
| T2 | Fixed | Closure keeps recorded-evidence mode by default and adds a fixed six-command `--reproduce` mode that delegates receipt validation to the existing ownership checker. |
| T3 | Fixed and merged | Strict merge records positive passing evidence for CPU, CUDA, WebGPU, Metal, and CUDA-AD on candidate C; Apple M5 Max Metal 4/4 and WebGPU 3/3 are included. |
| T4 | Fixed | `static_rank_mismatch.rs` is included in the existing storage `trybuild` compile-fail suite. |
| T5 NonNull | Fixed | Three `NonNull::from(&mut *owner)` sites use `NonNull::from(owner)`. |
| T5 cast_host_vec | Fixed | Size/alignment checks at the existing unsafe boundary are release-active `assert_eq!` checks. |
| T5 sha2 | Closed as stale | `cargo tree -i sha2 --workspace` resolves `sha2` through `tenferro-xla`; `stablehlo.rs` imports `Digest` and `Sha256`. The dependency is retained. |

## Verification completed before candidate selection

- `cargo check -p tenferro-gpu --features webgpu --test integration --target aarch64-apple-darwin` — pass.
- `cargo fmt --all --check` — pass.
- Workflow contract tests — 32 pass after T0; evidence-contract tests — 8 pass after T1–T3 checker work.
- Storage compile-fail suite — 2 tests pass, including the static-rank mismatch fixture.
- Storage static-rank integration test — 1 pass.
- Tensor storage unit tests filtered to `storage::` — 36 pass.
- Ownership v2 contract tests — 24 pass.
- `python3 scripts/check-storage-design-docs.py` — pass.
- Stale freeze report is rejected with `non-evidence path after candidate` while product changes remain after the old candidate.
- Fixture hardware report merge produces five ordered passing lanes and rejects candidate mismatch, duplicate, missing, and skipped lanes.
- Old structured-skip matrix blocks the new closure checker, as required until real required-lane evidence is merged.

## Candidate and merged hardware evidence state

- Product candidate C: `652b5c45f753f04425d71541b387acedc39cfa04`.
- Candidate branch pushed as `origin/codex/issue-1617-remediation-plan`.
- Hardware evidence commit: `885005ca`.
- Candidate-bound freeze, static-rank codegen, traversal, documentation audit,
  and merged hardware matrix reports all name C.
- The merged matrix is `complete: true`, `required_mode: true`, and records
  CPU 3/3, CUDA 4/4, WebGPU 3/3, Metal 4/4, and CUDA-AD 2/2 passing tests.
  Linux lanes use the AMD EPYC host; Apple lanes use arm64 Apple M5 Max,
  Darwin 25.5.0.
- The generated remediation plan was removed from the PR after the repository
  rules review identified it as prohibited standalone AI content.
- The default recorded-evidence closure report passes; bounded reproduction and
  final receipt generation remain the next closure steps.
