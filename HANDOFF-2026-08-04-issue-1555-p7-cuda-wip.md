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

## Post-checkpoint scope and quality review (2026-08-04)

A proportionality, scope, and implementation-quality review of commit
ef0c90addd1a73e9758bc188d32eea58d211fd67 produced the corrections below.
They are mandatory parts of the remaining P7 cutover, not optional
cleanup.

Remove disproportionate defenses:

1. `prepare_cubecl_access` in crates/tenferro-gpu/src/cubecl/dispatch.rs
   re-validates the request allocation domain, allocation id, and byte
   length against the same buffer that produced those values. Both call
   paths build the request from that buffer instance, and
   `DeviceAccessRequest::new` is `pub(crate)`, so these checks can only
   fire if trusted internal code forges its own output. This is the
   banned repeated-validation pattern and contradicts the G-contract
   rule that prepare/bind accepts no replacement key/range/provider
   values. Stop carrying re-checkable identity in the request, or stop
   re-verifying it in the provider; keep one source of truth.
2. `CubeclPreparedAccess` retains `writable`, `offset`, and
   `element_size` fields that no launch path reads (`writable()` is
   `#[allow(dead_code)]`; GEMM/permutation re-derive offsets from the
   view). Write authority is the Rust borrow, not a runtime flag.
   Delete the dead state.
3. `DeviceAccessRequest.dtype` is never consumed by a provider (only
   asserted in one storage unit test). Delete the field, and keep the
   offset non-negativity check in one layer only.

Collapse duplicated machinery during the cutover:

4. Two device-preparation seams exist. The storage-root path
   (`PreparedRead::Device`/`PreparedWrite::Device` `provider_state` in
   storage/prepared.rs) builds provider state with empty shape/strides
   and offset 0, and no launch path consumes that state; only unit
   tests exercise it. All real CUDA launches use the view/tensor
   `prepare_device_read`/`prepare_device_write` seam in types.rs.
   Finish with exactly one provider-neutral prepared hierarchy; a
   degenerate empty-shape `TensorBinding` must not survive anywhere a
   consumer could bind it.
5. `CubeclPreparedAccess` clones the provider handle twice (inside
   `binding` plus the separate `handle` field); every consumer discards
   one clone. Retain one representation per prepared access.

Implementation-quality notes for the cutover:

6. The view/tensor seam prepares per launch: each helper allocates a
   `Box<dyn PreparedDeviceAccess>`, clones the handle twice, and
   downcasts with a repeated stringly error. Dispatch carries eight
   near-identical downcast helpers. When unifying the seams, prepare
   once per checked access and consolidate the downcast into one
   helper so small-kernel dispatch latency does not regress.
7. The new transitional error branches in types.rs ("untyped backend
   storage has no typed descriptor" and friends) are removed by the
   cutover rather than tested; do not add tests for them.

Reconfirmed correct; do not "fix" these:

- No cryptographic, nonce, attestation, quarantine, retry, or recovery
  machinery exists on the branch (grep hits are `FnOnce` false
  positives).
- Mandatory allocation domain/id at backend root import implements the
  identity design; it is not an extra defense.
- The byte-capacity check in `typed_tensor_array_arg_as` validates a
  caller-chosen length; it is a genuine boundary check and stays.
- Checked arithmetic, structured storage errors, non-consuming failure
  returns, and the counter-based retention tests are sound; unsafe
  blocks follow the established `HostAllocation` discipline with
  accurate SAFETY comments.

## Exact next work

1. Re-read this handoff, AGENTS.md, REPOSITORY_RULES.md, the parent issue,
   and #1563 before editing.
2. Keep the implementation minimal: remove the untyped backend fallback and
   finish the single root/group owner cutover; do not add new defense layers.
   Apply every correction in "Post-checkpoint scope and quality review" as
   part of this step.
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
