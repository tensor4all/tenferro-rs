# Issue #1558 P2 Task 2 correction gate

Date: 2026-08-04

This worklog records the narrow correction gate selected after the P0/P1
checkpoint. The candidate starts from exact `origin/main` commit
`5bbd4a0613b3c801ce924cdc4339df2c77c1ab62` in the isolated branch
`codex/issue-1558-task2-corrections`.

## Scope and authority

The parent issue #1555 and child issue #1558 are authoritative. Only the six
withdrawn Task 2 corrections are addressed here:

1. range-overflow precedence for every participating range end;
2. root-bound span provenance tied to the exact root identity;
3. `RequestedIdentity::{Raw, Keyed, Rooted}` instead of parallel optional
   identity fields;
4. reproducibility from a committed candidate, with later RED modules wired
   only when their phase is selected;
5. executable typed tests rather than lexical source checks as ownership proof;
6. corrected parent/candidate provenance in the design and worklog.

P2 Task 3 ownership/reclaim implementation remains deferred. This change does
not add an owner, claim table, provider import, access preparation, persistent
split, allocation group, legacy bridge, compatibility layer, quarantine,
recovery, cryptographic evidence, or repeated map/enqueue validation.

## Implementation decisions

- `RootResourceExtent` and `ByteRange` use checked end arithmetic before
  alignment or containment checks.
- `RootResourceIdentity` validates its extent before minting its private
  nonzero provenance ID.
- `RootBoundSpan` stores the exact root identity and is the only resolved span
  value accepted by the diagnostic envelope.
- Requested identity is an explicit enum and remains untrusted request data.
- The only malformed extent constructor is test-only and exists to prove the
  required overflow/alignment precedence; it is not a production recovery or
  revalidation path.

## Files

- `crates/tenferro-tensor/src/storage/span.rs`
- `crates/tenferro-tensor/src/storage/identity.rs`
- `crates/tenferro-tensor/src/storage/diagnostics.rs`
- `crates/tenferro-tensor/src/storage/tests/span_validation.rs`
- `docs/design/storage-ownership-contracts.md`

## Verification

The focused RED test was observed before implementation: the storage modules
were absent and compilation failed at the declared module boundary. After the
minimal implementation:

- `cargo build` — passed on the clean base;
- `cargo test -p tenferro-tensor --lib storage::tests::span_validation` — 6
  passed;
- `python3 scripts/test-storage-ownership-contracts-v2.py` — 24 passed on the
  clean base.

The full correction-gate verification is run after the documentation and
contract checks are committed. Evidence is identified by the exact Git commit
and tracked repository-relative paths; no content digest or attestation is
required.
