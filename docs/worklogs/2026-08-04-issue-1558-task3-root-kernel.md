# Issue #1558 P2 root-kernel worklog

Date: 2026-08-04

## Authority and candidate

This phase follows the latest proportional-safety amendment on #1558. The
accepted Task 2 correction candidate is the base:
`e938ed4b55f300bfcfd41d8371a1da0eeeb0218b`.

The implementation commit is
`199e0b94657d9408103c7b92531d41eab7cb5df8`; the P2 ledger/design activation
commit is `f26820c57c6e836ea5f10ed7ad5c9e026b672b6`; the ledger artifact
binding fix is `8c556863c60f76f3b4028ad5a53cbd939a3e8285`. The evidence commit
containing this updated worklog follows those source commits.

The selected `p2-root-claims` row is promoted on top of the already active P0
and P1 rows. P3/P4/P5 and all later rows remain deferred. The implementation
contains no claim/hold table, quarantine,
recovery, destructor catch-unwind, persistent split/group, provider access,
legacy bridge, compatibility, cryptographic, or repeated-validation machinery.
The G1 ownership sketch was reconciled with this exact P2 shape: the pin is
separate from the identity-bearing claim, and ref/mut capabilities borrow the
owner rather than an `Arc` projection.

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
`8c556863c60f76f3b4028ad5a53cbd939a3e8285`:

- `cargo fmt --all --check` and `git diff --check` — passed;
- `cargo test -p tenferro-tensor --lib` — 228 passed;
- `cargo test -p tenferro-tensor --test storage_compile_contract` — 9 UI
  fixtures passed;
- `cargo test -p tenferro-tensor --test storage_root_claims` — passed and
  delegated to the private four-test root-claims proof surface;
- `cargo clippy -p tenferro-tensor --all-targets -- -D warnings` — passed;
- `cargo llvm-cov -p tenferro-tensor --lib --summary-only` — storage files
  covered above 90% lines (diagnostics 98.53%, identity 98.15%, root 94.23%,
  span 94.58%);
- `python3 scripts/test-storage-ownership-contracts-v2.py` — 24/24;
- `python3 scripts/check-storage-ownership-contracts.py` — ledger OK;
- `python3 scripts/check-storage-design-docs.py` — passed;
- `python3 scripts/repository-rules-review.py --base origin/main --head
  8c556863c60f76f3b4028ad5a53cbd939a3e8285 --output-json
  /tmp/p2-task3-rules-review.json` — pass, zero findings.

Fresh independent specification and quality reviews are required against the
final exact evidence commit (the docs-only commit immediately following the
source candidate above) before the next phase starts. The final review command
binds to `git rev-parse HEAD`; no source changes occur in the evidence commit.
