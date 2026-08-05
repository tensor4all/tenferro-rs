# Issue #1617 closure hygiene remediation work log

Date: 2026-08-05

## Scope and decision

This remediation follows the approved Issue #1617 design in one branch and one
non-squash PR. It stops before Apple M5 Max Metal/WebGPU execution. No
checksum, signature, attestation, nonce, hostile-runner defense, generic runner,
extra dependency, compatibility shim, or hidden transfer is introduced.

The product/checker candidate is selected only after the changes below and the
pre-candidate checks pass. Candidate-bound reports are generated after that
commit; evidence descendants may change only the closed evidence allowlist.

## Classification ledger

| Finding | Disposition | Evidence / implementation |
|---|---|---|
| T0 | Fixed | `apple_context.rs` now destructures through `&mut managed`; CI cross-checks the macOS-gated integration target from Linux. |
| T1 | Fixed by regeneration protocol | Freeze, static-rank, traversal, hardware, docs-audit, and closure reports will be regenerated from one candidate. Stale reports are rejected by the freeze evidence-path check. |
| T2 | Fixed | Closure keeps recorded-evidence mode by default and adds a fixed six-command `--reproduce` mode that delegates receipt validation to the existing ownership checker. |
| T3 | Prepared for real evidence | Hardware checker supports incomplete host captures and strict merge of one passing record for every required lane. Apple execution occurs only after candidate selection. |
| T4 | Fixed | `static_rank_mismatch.rs` is included in the existing storage `trybuild` compile-fail suite. |
| T5 NonNull | Fixed | Three `NonNull::from(&mut *owner)` sites use `NonNull::from(owner)`. |
| T5 cast_host_vec | Fixed | Size/alignment checks at the existing unsafe boundary are release-active `assert_eq!` checks. |
| T5 sha2 | Closed as stale | `cargo tree -i sha2 --workspace` resolves `sha2` through `tenferro-xla`; `stablehlo.rs` imports `Digest` and `Sha256`. The dependency is retained. |

## Verification completed before candidate selection

- `cargo check -p tenferro-gpu --features webgpu --test integration --target aarch64-apple-darwin` — pass.
- `cargo fmt --all --check` — pass.
- Workflow contract tests — 32 pass after T0; evidence-contract tests — 7 pass after T1–T3 checker work.
- Storage compile-fail suite — 2 tests pass, including the static-rank mismatch fixture.
- Storage static-rank integration test — 1 pass.
- Tensor storage unit tests filtered to `storage::` — 36 pass.
- Ownership v2 contract tests — 24 pass.
- `python3 scripts/check-storage-design-docs.py` — pass.
- Stale freeze report is rejected with `non-evidence path after candidate` while product changes remain after the old candidate.
- Fixture hardware report merge produces five ordered passing lanes and rejects candidate mismatch, duplicate, missing, and skipped lanes.
- Old structured-skip matrix blocks the new closure checker, as required until real required-lane evidence is merged.

## Remaining sequence

1. Run the full product/checker verification on this branch.
2. Commit this work log and the corrected P11/P13 design records.
3. Freeze and record the exact product candidate C; push it.
4. Regenerate non-hardware evidence against C and commit only evidence paths.
5. Capture Linux/CUDA partial evidence.
6. Prepare the Apple host checkout, exact C, command, and temporary report path.
7. Stop immediately before executing the Apple Metal/WebGPU commands.

Apple/Metal results are deliberately absent from this work log until the
separate real-hardware execution begins.
