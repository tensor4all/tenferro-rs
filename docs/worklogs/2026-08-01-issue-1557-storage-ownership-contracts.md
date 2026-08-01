# Worklog: #1557 Storage Ownership Contracts and Phase 1 Verification Harness

This worklog records the contract document and verification harness for the
storage ownership redesign tracked by
[#1555](https://github.com/tensor4all/tenferro-rs/issues/1555) and owned by
[#1557](https://github.com/tensor4all/tenferro-rs/issues/1557).

## Scope

- Added `docs/design/storage-ownership-contracts.md`: the seven design gates
  (span access and retirement, `AllocationGroup`, submission, method
  distribution, raw handles and reclamation, documentation ownership, AD
  value retention) as signature sketches plus state-transition tables, each
  row answering the six review-checklist questions from #1557 (capability,
  borrow, synchronization, failure return, panic/drop state, reclamation
  legality).
- Added the document to the `docs/design/index.md` Core Design table.
- Added the Phase 1 verification harness under #1557: the checked TOML ledger,
  checker tests, trybuild borrow fixtures, six-family read-only parity, and
  the source-inventory drift ledger.
- Completed Task 4 in a separate commit from Task 3: deferred activation
  metadata, compile/property/Miri/provider obligations, private corruption
  test obligations, and the Phase 1 verification documentation.
- No production storage API, unsafe public constructor, corruption hook, or
  test escape hatch was added.

## Phase 1 harness artifacts and evidence

The four tasks are intentionally layered:

- Task 1 added `scripts/storage-ownership-contracts.toml`,
  `scripts/check-storage-ownership-contracts.py`, and their CI-profile unit
  tests. The ledger owns fixture identity, phase ownership, path confinement,
  and source-inventory metadata.
- Task 2 added the trybuild harness and eight current compile contracts: five
  fail fixtures and three pass fixtures. Accepted `.stderr` snapshots are
  checked in; the harness fails on an empty discovery set.
- Task 3 added the six-family API parity test and the current-main source
  inventory. The inventory contains 70 narrow scans (65 inventoried and five
  forbidden) and 78 exact source rows; it is a lexical deletion/drift ledger,
  not the ownership proof.
- Task 4 extends the fixture schema with `future_path`, `command`, and
  `activation_phase`. Deferred rows require all three plus `owner_issue` and
  are activated by the exact future artifact appearing. Active rows use
  `path`; dynamic verification kinds retain an executable command. Each future
  artifact has exactly one deferred fixture row; shared obligations are kept
  together in that row's rationale. The resulting ledger has 28 fixtures: nine
  active and nineteen deferred. The G2 split-property obligations are one
  `storage_group_properties.rs` artifact covering N=0, N=1, N>2, empty,
  reverse-stride, overflow, disjointness, and overlap cases.

The Task 4 checker tests were written RED before implementation and then
GREEN after the schema implementation. They cover missing deferred fields,
invalid phases, active/deferred path exclusivity, traversal and symlink
confinement, the existing-future-artifact promotion error, dynamic active
commands, clean deferred rows, and the generic `provider` kind. The review
fix added a focused duplicate-future-path test: it was restored to the clean
parent, observed RED, then the checker was implemented and the full suite ran
GREEN with 65 tests.

## Context read

- #1555 consolidated body (invariants I1 through I10, maintainer synthesis,
  decomposition-check and evaluation comments).
- Phase issues #1556 through #1569, including the expanded gate 7 (AD value
  retention) and the #1568 AD provider-coverage section.
- Current implementation seams cited in the review threads:
  `crates/tenferro-ad/src/eager.rs` (`materialized()`, `GradSlot`,
  `Arc<TensorValue>`), `crates/tenferro-runtime/src/checkpoint.rs`,
  `crates/tenferro-gpu/src/webgpu/mod.rs` (`map_read`/`map_write` guard
  types), `crates/tenferro-tensor/src/types.rs`
  (`try_multi_slice_mut`, `TypedTensorViewMutPair`, `Placement`).

## Decisions

- One document, sections numbered G1 through G7 to match the #1555 gate list
  one to one, so "gate N" resolves to exactly one contract section.
- `UseLease` is specified as `'static` with provider pins rather than a
  borrow, because leases must move into runtime retirement records that
  outlive the submitting borrow. Capability enforcement stays on the
  acquisition methods (`&self` for reads, `&mut self` for writes); the lease
  itself carries no authority.
- Guard leak (`mem::forget`) is documented as sound but possibly
  liveness-degrading until owner drop, so the contract does not depend on
  `Drop` running for safety.
- AD handle types remain `Clone` explicitly: the non-`Clone` rule applies to
  owners and capabilities, and gate 7 defines handles as read-only
  descriptor references. This resolves the apparent conflict between
  "remove shallow `Clone`" (#1559) and existing `EagerTensor` cloning.
- Copy and allocation accounting use separate reason enums and ledgers.
  Retention has no variant in either ledger, encoding that retention performs
  neither a copy nor an allocation instead of relying on aggregate counts.
- Follow-up contract review after #1570 merged separated a unique
  `OwnedSpanClaim` from a non-authoritative shared provider-resource pin.
  Child claims can only be produced by consuming and proving a split of the
  parent (or at an audited unsafe import boundary), and root deallocation
  waits for every claim and lease.
- Scoped execution now has an exact hybrid result shape: borrowed
  identity/metadata outputs remain `'env`-bounded slots, fresh outputs are
  owned slots, and `wait` returns an outcome that is independent of the
  scope lifetime `'s`. Borrowed slots reject extraction without copying.
- The earlier `lease_unique` sketch was rejected. No transition from a
  shared pin or handle to exclusive authority exists; the unsafe raw-write
  binder consumes an already-proven `StorageMut` capability.
- AD descriptor liveness covers tape, checkpoint, execution, and every
  sibling public handle. Generational IDs prevent a stale handle from
  resolving to a reused descriptor slot, and extraction requires the caller
  to consume the last liveness root.
- The architecture quality gate rejects ad hoc provider or legacy-API
  exceptions. All backends and AD/runtime use one ownership kernel; each
  implementation phase deletes the path it replaces, and every temporary
  bridge has an explicit removal phase.
- Scoped retirement failure is an explicit quarantine outcome. Before a
  borrowed scope can end, the affected root is atomically poisoned so later
  safe access fails and the quarantine registry retains the resource. This
  avoids both hidden copies and unsound return of a borrow whose device use
  has unknown completion.
- `ValueId` is group-qualified and generational, with
  `GenerationalDescriptors` as the only liveness registry. Root claims carry
  private provenance linked to the same root-resource identity as their
  non-authoritative pin.

## Alternatives rejected

- Splitting the contracts into one document per gate: rejected because #1557
  requires one canonical path referenced by all later phases.
- Publishing a `timeline()` accessor in the trait sketch: rejected by the
  maintainer synthesis; access acquisition methods own that state.
- Specifying exact final public names: rejected; #1557 requires shapes and
  capability rules while names stay provisional until owning phases land.

## Verification

- Python 3.12.11 is required for the checker and its `tomllib`-based tests;
  the final environment provides both `python3` and `python3.12` at that
  version.
- Task 4 TDD evidence: the new schema tests were first run RED (62 tests,
  18 failures caused by the unimplemented `activation_phase`/`command`
  fields), then the checker was implemented and the initial expanded suite ran
  GREEN (64 tests passed). The duplicate-deferred-`future_path` review test was
  subsequently observed RED before its checker implementation; the final full
  checker suite ran GREEN with 65 tests.
- `python3.12 scripts/check-storage-ownership-contracts.py`: PASS, ledger OK;
  28 fixtures (nine active, nineteen deferred), 70 source scans (65
  inventoried, five forbidden), and 78 source-inventory rows.
- `cargo test -p tenferro-tensor --test storage_compile_contract`: PASS; all
  eight active trybuild UI cases matched their accepted diagnostics or pass
  expectations.
- `cargo test -p tenferro-tensor --test storage_api_parity`: PASS (one parity
  test).
- `cargo test -p tenferro-tensor`: PASS (216 unit tests, all integration
  suites, eight trybuild cases, and 311 doctests).
- `cargo check -p tenferro-gpu --features webgpu`: PASS. This is the current
  Apple/Metal route; no native Metal implementation was added.
- `python3.12 scripts/ci/run_profile.py ci-config`: the checker tests (65),
  ledger, and all 161 CI-profile unit tests passed. The profile then failed at
  the separate `actionlint` step because `actionlint` is not installed in the
  environment (`/bin/sh: actionlint: not found`).
- `python3.12 scripts/ci/run_profile.py docs`: PASS on one fresh, single
  process run after the review fixes. Rustdoc completed, Quarto rendered all
  84 pages including `performance/cpu-benchmark-results-2026-05-23.md`, and
  the docs-site/link checks passed. No missing performance artifact occurred
  in this run. The optional dependency graph was skipped with the distinct
  environment warning `graphviz (dot) not found`. The run materialized
  untracked rendered `docs/**/*.html` files and `docs/site_libs`; those
  generated artifacts were moved/cleaned after evidence capture to restore a
  clean worktree.
- `bash scripts/check-pr-fast.sh --coverage-reviewed --test 'cargo test -p tenferro-tensor --test storage_compile_contract' --test 'cargo test -p tenferro-tensor --test storage_api_parity' --test 'python3 scripts/check-storage-ownership-contracts.py'`: the required preflight
  stopped before running focused tests because `origin/main` moved from the
  Task 3 base and this candidate is not based on the latest fetched
  `origin/main` (`d41d1716`). This is repository-state drift, not a test
  failure; the focused commands above were run independently.
- `cargo fmt --all --check`: PASS. `git diff --check`: PASS.
- `python3.12 scripts/repository-rules-review.py --base origin/main --head
  HEAD --output-json /tmp/repository-rules-review-task4-final-default.json`:
  PASS, verdict `pass`, no findings. A separate diagnostic run with an
  artificial 30-second timeout reported an external-response timeout; the
  repository-default 120-second run above completed successfully on the
  final candidate.
- Task 2's accepted trybuild snapshots remain checked in; Task 3's source
  inventory remains the lexical deletion/drift ledger, while Rust borrowing
  and private constructors remain the ownership proof.

## Remaining risks

- The G6 command table and deferred `command` fields name current scripts; a
  phase that renames tooling must update the table and ledger in the same PR.
- Signature sketches will drift slightly as owning phases land; the change
  control rule (update document plus tests in the same PR) is the guard.
- The AD `ValueGuard`/`Gradients` sketches are the least implementation-
  tested part of the contract; Phase 3 and Phase 9 may need to refine them
  through the documented change-control path.
- The deferred rows become active only when their exact future artifact is
  introduced. The owning phase must promote the row in the same change as the
  artifact and keep the command and acceptance evidence current.
- The repository-rules review is an external-LLM-dependent gate in addition
  to the deterministic checker; this candidate has a recorded PASS, while
  future environments must report an unavailable external review separately.
