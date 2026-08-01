# Worklog: #1557 storage ownership contract and executable RED checkpoint

This worklog records the Phase 1 contract checkpoint for
[#1557](https://github.com/tensor4all/tenferro-rs/issues/1557), under the
umbrella redesign in
[#1555](https://github.com/tensor4all/tenferro-rs/issues/1555).

## Scope and boundary

This checkpoint delivers two artifacts:

- `docs/design/storage-ownership-contracts.md`: the normative G1--G7 storage
  ownership contract, including signature sketches, six-column transition
  tables, provider-neutral ownership rules, and the phase documentation
  deliverables.
- `scripts/test-storage-ownership-contracts-v2.py`: an executable RED
  specification for the v2 ledger. It invokes the checked-in production
  manifest and creates adversarial temporary repositories with real files and
  real symlinks.

The v2 checker and runner are intentionally not implemented in this
checkpoint. The existing checker file is still the superseded v1
implementation; it is not a compatibility implementation of v2. Phase 1
must later replace it with the v2 checker and add the runner. No production
storage API, unsafe importer, corruption hook, or test escape hatch was added
here.

## Contract decisions fixed by this checkpoint

- P0 (#1556) and P1 (#1557) are independent roots. P2 (#1558) has exactly one
  prerequisite, P1. P0 joins only at the atomic CUTOVER cohort whose
  prerequisites are P0 and P5 and whose members are P3 and P9.
- The ledger has one canonical graph registry and one `[[obligations]]` table.
  Each obligation has one immutable artifact, one typed command bound to that
  artifact, one unit, one or more gate IDs, and a tagged `state`. Parallel
  active/deferred tables, status booleans, terminal flags, synthetic terminal
  artifacts, and stale p3/p4 ownership rows are rejected by the RED contract.
- Deferred-to-active promotion changes only the tagged state. Artifact and
  command identity, including path arguments and binding, remains identical
  across base and candidate manifests. Candidate-bound receipts include the
  manifest, artifact, command, and commit digests. Terminal status is derived
  from obligations, cohort completion, and receipts.
- Path validation is filesystem-aware. Repository-relative lexical checks are
  supplemented by canonical resolution so `..`, absolute paths, and real
  symlink escapes cannot become green. Existing deferred artifacts do not
  promote themselves.
- Command execution is fail-closed and allow-listed by typed command kind.
  Shell strings, empty argv, path escapes, unknown command kinds, unbound
  target links, and a failed active command are errors. Active commands run
  once; deferred commands never run.
- The storage kernel is provider-neutral and allocation-free on resolve/acquire
  hot paths. It owns one provider vtable in `RootResourceState`; no provider
  enum, per-access `Box`/`Arc`, or receiver-plus-resolved authority is part of
  the contract.
- `BackendAllocationAccess` is the sole unsafe provider extension boundary.
  Providers can construct metadata and raw mapping/lease carriers, but cannot
  construct `HostReadGuard`, `HostWriteGuard`, `RootBoundSpan`, claims, or
  `UseLease`. `import_owned_storage` is safe and fallible; rejection returns
  the same allocation box through `ImportRejected`.
- `ResolvedWrite::backend_write_request` explicitly reborrows with
  `let owner = &mut *self.capability.owner;` before borrowing disjoint pin and
  claim fields. Device write failure returns the exact resolved exclusive
  capability. Direct borrowed write is synchronous and retires before its
  borrow ends; scoped execution is read-only. Detached asynchronous execution
  owns `OwnedStorage`.
- AD retention is descriptor/group liveness, not shallow storage cloning.
  Retention has no copy/allocation reason; explicit duplication, transfers,
  operation outputs, and checkpoint recomputation are separately classified.

## RED coverage inventory

The executable specification covers:

- the exact production manifest path and v1 rejection without compatibility;
- nominal v2 parsing, tagged-state shape, canonical graph edges, P0/P1 root
  independence, P2's single prerequisite, and CUTOVER atomicity;
- duplicate and unknown graph targets, duplicate artifact targets, missing or
  escaping paths, real symlink escapes, existing deferred artifacts, stale
  `p3-ad-retention`/`p4-ad-runtime` rows, and synthetic terminal artifacts;
- command allowlist, empty argv, path-argument confinement, exact
  artifact-command binding, duplicate command identity, fail-closed execution,
  active-once/deferred-never runner behavior;
- base-to-candidate immutable identity, candidate-bound receipt fields, and
  derived terminal status;
- rejection of the old fixture/source/ownership parallel tables.

The temporary repository helpers are test-only. They do not replace the
production-manifest test, and the symlink cases create actual filesystem
symlinks rather than checking a string containing a symlink-like path.

## Verification evidence for this checkpoint

Passing deterministic checks:

- `python3.12 -m py_compile scripts/test-storage-ownership-contracts-v2.py`
- generated v2 manifests parse with Python 3.12 `tomllib`, including promoted
  active states;
- `git diff --check`.

The following RED result is intentional and is evidence that the implementation
surface has not been silently added in this checkpoint:

- `python3.12 scripts/test-storage-ownership-contracts-v2.py` fails because the
  v2 checker/runner behavior is not implemented yet; the checked-in production
  manifest is still v1 and therefore also fails the v2 production assertion.

No cargo implementation tests are claimed here because this checkpoint changes
only design, ledger specification tests, and provenance. The next Phase 1
checkpoint must first implement the checker/runner against this RED contract,
then run the full v2 suite, the exact production manifest, the source
inventory, trybuild, parity, docs, and repository quality gates.

## Residual work and change control

- Replace the v1 production ledger with the v2 single-table schema in the
  checker/runner implementation checkpoint; do not add a v1 compatibility
  parser.
- Implement filesystem-aware artifact resolution, promotion comparison,
  candidate-bound receipts, typed command execution, and derived terminal
  reporting to satisfy this RED suite.
- Keep this design document and this RED specification in the same PR for any
  semantic contract change. A later phase may refine provisional names only by
  updating the contract and its executable tests together.
- The obsolete `HANDOFF-2026-07-25-tenferro-unification6-wip.md` remains a
  Phase 13 cleanup item; it is intentionally not deleted in this Phase 1
  checkpoint.
