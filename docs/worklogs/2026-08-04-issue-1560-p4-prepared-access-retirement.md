# Issue #1560 P4 prepared-access and retirement worklog

Date: 2026-08-04

## Authority and scope

This phase follows #1555 and #1560 after the accepted P2 root-claims
candidate. P4 is the selected next phase; P3, P5, and every later phase remain
deferred. The implementation is private to the storage module and does not
migrate the existing public tensor APIs.

The proportional-safety amendment is preserved. The kernel has no
compatibility or provider bridge, cancellation protocol, quarantine/poison
state, retry/recovery registry, cryptographic evidence, or repeated
map/enqueue validation. Unproven completion retains one private record and
returns a diagnostic outcome without an owner or recovery operation.

## Implementation

- `BackendAllocation` now exposes borrowed byte mappings at the existing
  private provider boundary. `StorageRef` and `StorageMut` pass the checked
  root-bound span once; provider mapping guards remain borrowed by prepared
  host access and no provider `Arc` is cloned.
- `CheckedRead` and `CheckedWrite` retain the rank-preserving checked layout,
  dtype/size/alignment, exact span, and mutable injectivity proof. Preparation
  consumes that state into nested `PreparedRead`/`PreparedWrite` host or
  device variants. Host contiguous access uses typed slices; host strided
  access uses an incremental shape/stride/carry cursor. The device variant
  retains only checked state and a private opaque token.
- `PreparedPackage::admit` returns the same package on a rejection known before
  admission. An admitted `RetirementRecord` owns event, bindings, root pins,
  and provider context. Proven completion or a proven provider failure drops
  all resources exactly once. Unproven completion intentionally leaks the
  complete private record and returns `CompletionUnproven`.

## Proof artifacts

The five P4 ledger rows are promoted together:

- `storage_borrow_contract.rs` and the private prepared-access proof cover
  borrowed ownership and unchanged checked failures.
- `storage_prepared_validation.rs` covers rejection before provider mapping.
- `storage_provider_event_retirement.rs` covers proven, failed, unproven, and
  pre-admission lifecycle outcomes.
- `storage_traversal_resolution.rs` covers one provider mapping per prepared
  access independent of element count.
- `storage_prepared_access.rs` covers the enum hierarchy, typed contiguous
  access, reverse strided access, empty layouts, and mutable injective
  traversal.

The existing storage UI compile-contract suite remains active for the public
borrow surface; the new prepared types are private and are therefore proved
through the crate's private unit harness invoked by the five integration
artifacts.

## Verification

The final evidence commit records the exact candidate `HEAD`. Required checks
are:

```text
cargo fmt --all --check
git diff --check
cargo test -p tenferro-tensor --all-targets --quiet
cargo clippy -p tenferro-tensor --all-targets -- -D warnings
python3 scripts/test-storage-ownership-contracts-v2.py
python3 scripts/check-storage-ownership-contracts.py
python3 scripts/check-storage-design-docs.py
```

P3/P5/P9 remain deferred after this phase; no cutover is implied by P4.
