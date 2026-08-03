# Issue #1558 P2 root-kernel worklog

Date: 2026-08-04

## Authority and candidate

This phase follows the latest proportional-safety amendment on #1558. The
accepted Task 2 correction candidate is the base:
`e938ed4b55f300bfcfd41d8371a1da0eeeb0218b`.

The implementation commit is
`199e0b94657d9408103c7b92531d41eab7cb5df8`; the P2 ledger/design activation
candidate is exact commit
`f26820c57c6e836ea5f10ed7ad5c9e026b672b6`.

Only `p2-root-claims` is promoted. P3/P4/P5 and all later rows remain
deferred. The implementation contains no claim/hold table, quarantine,
recovery, destructor catch-unwind, persistent split/group, provider access,
legacy bridge, compatibility, cryptographic, or repeated-validation machinery.

## Design

- `BackendAllocation` is the one private unsafe provider boundary and reports
  stable root extent metadata, provider kind, capabilities, and immutable
  diagnostics.
- `import_unique_root` validates one provider extent, mints one exact
  `RootResourceIdentity`, derives one full `RootBoundSpan`, and constructs one
  `Arc<RootResource>` lifetime pin plus one non-`Clone` root claim.
- `StorageRef<'a>` derives only from `&'a OwnedStorage`; `StorageMut<'a>` only
  from `&'a mut OwnedStorage`. Both expose metadata only.
- The final root `Arc` drop destroys the provider allocation exactly once.
- Invalid import returns the existing typed operation envelope before owner
  construction; the consumed provider box is dropped normally.

## Files

- `crates/tenferro-tensor/src/storage/root.rs`
- `crates/tenferro-tensor/src/storage/tests/root_claims.rs`
- `crates/tenferro-tensor/tests/storage_root_claims.rs`
- `crates/tenferro-tensor/tests/ui/storage/fail/private_allocation_core.rs`
- `scripts/storage-ownership-contracts.toml`
- `scripts/test-storage-ownership-contracts-v2.py`
- `scripts/check-storage-design-docs.py`
- `docs/design/storage-ownership-contracts.md`

## Candidate-scoped verification

All commands below were run from the candidate branch at or after
`f26820c57c6e836ea5f10ed7ad5c9e026b672b6`:

- `cargo fmt --all --check` and `git diff --check` — passed;
- `cargo test -p tenferro-tensor --lib` — 228 passed;
- `cargo test -p tenferro-tensor --test storage_compile_contract` — 9 UI
  fixtures passed;
- `cargo test -p tenferro-tensor --test storage_root_claims` — passed;
- `cargo clippy -p tenferro-tensor --all-targets -- -D warnings` — passed;
- `cargo llvm-cov -p tenferro-tensor --lib --summary-only` — storage files
  covered above 90% lines (diagnostics 98.53%, identity 98.15%, root 94.23%,
  span 94.58%);
- `python3 scripts/test-storage-ownership-contracts-v2.py` — 24/24;
- `python3 scripts/check-storage-ownership-contracts.py` — ledger OK;
- `python3 scripts/check-storage-design-docs.py` — passed;
- `python3 scripts/repository-rules-review.py --base origin/main --head
  f26820c57c6e836ea5f10ed7ad5c9e026b672b6 --output-json
  /tmp/p2-task3-rules-review.json` — pass, zero findings.

Fresh independent specification and quality reviews are required against this
exact candidate before the next phase starts.
