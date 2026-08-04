# Handoff: Issue #1555 P7 CUDA root/prepared WIP

Date: 2026-08-04 JST

Repository worktree:

/home/shinaoka/tensor4all/tenferro-rs/.worktrees/issue-1558-task2-corrections

Branch:

codex/issue-1558-task3-root-kernel

Remote:

origin/codex/issue-1558-task3-root-kernel

This is a continuation checkpoint, not a completion or closure handoff.
Issue #1555 remains open.

## Objective and authority

The persistent objective is to complete Issue #1555 only after the P13-B
independent closure audit passes. Parent #1555 and child issues #1563 (P7
CUDA) and #1564 (P8 WebGPU/Apple/Metal) are authoritative.

The approved phase order remains:

P1 -> P2 -> P4 -> P5 -> atomic P3+P9 -> P6 -> P7/P8 -> P10 -> P13-A -> P11/P12 -> P13-B.

Only P7 is currently active. P8 and all later phases remain unstarted or
deferred.

The proportional-safety constraints remain binding:

- no cryptographic identity, digest, nonce, or attestation;
- no malicious-runner model;
- no compatibility, migration, recovery, quarantine, or retry machinery;
- no repeated validation whose only purpose is defending trusted local code;
- no safe unscoped raw device pointer or handle API.

The user explicitly requested that this checkpoint be committed and pushed so
the harness can be restarted. No further automatic commit or push is implied
by this document.

## Work completed in this checkpoint

The local P7 WIP now contains:

- a private root/provider preparation seam in tenferro-tensor;
- CubeclBuffer provider preparation returning a non-Clone prepared state;
- CUDA dispatch, GEMM, permutation, and interop paths consuming prepared access
  rather than extracting a provider handle directly from tensor metadata;
- root import of backend buffers with mandatory domain/allocation identity;
- storage tests proving provider preparation is called once and the prepared
  state is retained until the checked access is dropped;
- replacement of the zero-sized device token in storage/prepared.rs with the
  provider state;
- updated CUDA source-contract expectations for prepared binding consumption.

The provider state is currently provider-owned (CubeclPreparedAccess owns the
CubeCL handle). The checked root capability remains live in the prepared
device read/write object.

## Known incomplete work

This checkpoint must not be treated as P7 completion:

1. TypedTensor still has the legacy buffer plus optional group shape in parts
   of the implementation. The backend constructor currently has an untyped
   group fallback (slot: None); the authoritative P7 design requires one
   typed descriptor/root owner without a physical owner split.
2. The P7 artifact
   crates/tenferro-gpu/tests/storage_provider_cuda.rs has not been added.
3. The P7 ledger row remains deferred and must not be activated yet.
4. P8 WebGPU/Apple/Metal work has not started.
5. P10, P13-A, P11/P12, and P13-B have not started.
6. PreparedDeviceRead/Write state is now real, but the remaining root/group
   cutover and exact artifact evidence are still required.

Do not add a compatibility shim to hide item 1. Continue with the direct
single-owner cutover required by #1563, while keeping host behavior unchanged.

## Verification at handoff

These checks passed on the current worktree immediately before this handoff:

    cargo fmt --all -- --check
    cargo check -p tenferro-tensor --quiet
    cargo check -p tenferro-gpu --features cuda --quiet
    cargo test -p tenferro-tensor --lib --quiet
      257 passed
    cargo test -p tenferro-gpu --features cuda --lib --quiet
      86 passed, 116 ignored
    cargo test -p tenferro-gpu --features cuda --test integration --quiet
      68 passed

The full workspace and P7 artifact command have not yet been run for this
checkpoint. Do not claim Issue #1555 closure from these focused checks.

## Exact next work

1. Re-read this handoff, AGENTS.md, REPOSITORY_RULES.md, the parent issue,
   and #1563 before editing.
2. Keep the implementation minimal: remove the untyped backend fallback and
   finish the single root/group owner cutover; do not add new defense layers.
3. Add the required CUDA artifact only after the cutover is complete:

       cargo test -p tenferro-gpu --features cuda --test storage_provider_cuda

4. Run the exact final-commit gates before activating only P7 in the ledger:

       cargo fmt --all -- --check
       cargo check --workspace --quiet
       python3 scripts/check-storage-ownership-contracts.py
       python3 scripts/check-storage-design-docs.py
       python3 scripts/run-storage-ownership-contracts.py --diagnostics-json

5. Only after P7 evidence is accepted, start P8. Do not start P10 or later
   implicitly.
6. Close #1555 only after all later selected phases and the independent P13-B
   closure audit pass on one exact final commit.

## Restart note

The handoff file is part of the checkpoint commit. On restart, inspect git
status first; the branch should be clean after the checkpoint commit, and its
remote tracking branch should be
origin/codex/issue-1558-task3-root-kernel.
