# Issue #1555 P3/P9 owner-bundle follow-up

Date: 2026-08-04

This follow-up continues the explicitly selected P3/P9 atomic cohort from the
existing candidate. It does not activate P6 or any later phase.

## Implemented boundary

- `TensorValue` now stores one owner record (`Tensor`) plus logical layout
  metadata. Cloning a value clones only the immutable record handle; physical
  duplication remains explicit through `duplicate`.
- Compact values can be consumed through `TensorValue::into_tensor` when the
  owner is unshared. Metadata-only views are rejected at that boundary instead
  of being copied implicitly.
- The old public `TensorOwnedView` path and enum-pattern storage variants were
  removed. Runtime and CPU/eager paths use owner/value accessors instead.
- The remaining legacy public storage identifiers were renamed in place to
  `StorageBuffer`, `BackendStorage`, `BackendStorageHandle`,
  `TensorStorageRef`, `TensorStorageRefMut`, and `TypedTensorViewMutSplit`;
  no compatibility aliases retain the old spellings.
- Detached `ExecutionInputs` now owns one `AllocationGroup` and an immutable
  descriptor binding array. Admission and worker execution borrow those
  bindings; ordinary pre-admission failures return the unchanged package, and
  retired execution failures return it only after the worker has finished.

The group wrapper is intentionally limited to direct move-only tensor owners
needed by detached submission. No cancellation, recovery, quarantine,
compatibility alias, hidden materialization, cryptographic evidence, or
repeated provider validation was added.

## Verification

- `cargo fmt --all`
- `cargo check --workspace`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test -p tenferro-tensor --lib`
- `cargo test -p tenferro-tensor --test storage_owner_bundle`
- `cargo test -p tenferro-runtime --test integration runtime_execution -- --nocapture`
- `cargo test -p tenferro-runtime runtime::tests::execution -- --nocapture`
- `python3 scripts/check-storage-ownership-contracts.py --diagnostics-json`
- `python3 scripts/check-storage-design-docs.py`
- `python3 scripts/test-storage-ownership-contracts-v2.py`

The P3/P9 cohort remains open: scoped borrowed submission and the final public
storage-name inventory still require a separate, explicitly selected step.
