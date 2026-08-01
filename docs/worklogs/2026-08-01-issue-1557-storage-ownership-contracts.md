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
  manifest, compares it to the canonical graph/obligation model, and creates
  adversarial temporary repositories with real files and real symlinks.

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
  across base and candidate manifests. Every canonical unit has required
  obligations; an empty set is invalid rather than vacuously complete.
  Candidate-bound runner receipts include successful results and the manifest,
  artifact, command, and actual Git commit digests. Terminal status is derived
  from obligations, cohort completion, and complete receipts.
- Receipt tests independently compute SHA-256 values and exercise artifact
  mutation, post-receipt base-manifest digest mutation, and in-repository
  symlink retargeting. The checker must validate resolved bytes rather than a
  path string or a nonzero-looking digest.
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
- Read and write acquisition explicitly split owner fields. One private
  `RootResourceState` method validates the binding, constructs the opaque
  request, and dispatches to that same state's provider before the borrow can
  escape. No request is returned and no owner/provider field is re-accessed
  while it is live. Host and device write use
  `let owner = &mut *self.capability.owner;`; device pre-admission failure ends
  the inner borrow before returning the exact `ResolvedWrite`. Direct borrowed
  write is synchronous and retires before its borrow ends; scoped execution is
  read-only. Detached asynchronous execution owns `OwnedStorage`.
- `BackendRawLease` and `UseLease` are explicitly `Send + !Sync` under the
  thread-transfer clause of the one unsafe provider implementation contract.
  Host raw mappings and guards are explicitly `!Send + !Sync`. Zero-sized
  markers add no allocation. Release callback/context is taken before its
  exactly-once invocation; panic is contained and quarantines the pinned root,
  while a forgotten carrier keeps the pin and can only harm liveness. The RED
  suite locks these signature clauses without introducing another unsafe proof
  boundary.
- AD retention is descriptor/group liveness, not shallow storage cloning.
  Retention has no copy/allocation reason; explicit duplication, transfers,
  operation outputs, and checkpoint recomputation are separately classified.

## RED coverage inventory

The executable specification covers:

- the exact production manifest path, explicit current-v1 state, and the
  post-migration canonical-v2 equality/checker-success gate;
- an independent checked-in legacy v1 fixture/source sample and rejection test
  without compatibility;
- nominal v2 parsing, tagged-state shape, canonical graph edges, P0/P1 root
  independence, P2's single prerequisite, and CUTOVER atomicity;
- duplicate and unknown graph targets, duplicate artifact targets, missing or
  escaping paths, real symlink escapes, existing deferred artifacts, stale
  `p3-ad-retention`/`p4-ad-runtime` rows, and synthetic terminal artifacts;
- command allowlist, empty argv, path-argument confinement, exact
  artifact-command binding, duplicate command identity, fail-closed execution,
  active-once/deferred-never runner behavior;
- actual-Git base-to-candidate immutable identity, candidate-bound runner
  receipt output, exact per-execution obligation/artifact/command/candidate
  bindings, non-vacuous P0/P5 CUTOVER proof, and both positive and
  incomplete-receipt terminal derivation, with no locally constructed positive
  receipts; independently recomputed manifest/artifact/command digest values
  and post-receipt mutation rejection;
- stable JSON diagnostic codes and identifying fields, with checked fixture
  replacement helpers rather than unchecked text mutation. Every one-fault
  case requires an exact diagnostic code set and field shape;
- canonical future production-bound borrow, auto-trait, and provider-release
  artifact/command obligations. No inline synthetic borrow compile is
  canonical evidence and no private-name source scan is used;
- rejection of the independent checked-in legacy v1 fixture/source tables as
  well as rejection of those tables reappearing in a v2 manifest;
- a machine-readable expected RED event set and report that rejects unknown
  failures, errors, skips, duplicate events, and unexpected subtests while
  requiring equal expected/observed event counts.

The temporary repository helpers are test-only. They do not replace the
production-manifest test, and the symlink cases create actual filesystem
symlinks rather than checking a string containing a symlink-like path. A
separate required capability test emits an explicit machine-readable failure
when the host cannot create symlinks; dependent symlink cases cannot turn that
condition into a skipped green result.

## Rejection of checkpoint 546f18be and remediation

Concrete review rejected checkpoint `546f18be` for four contract defects:

- the write request escaped a private binder and acquisition then reborrowed
  the owner/provider while that request was live;
- CUTOVER fixtures had no required P0/P5 obligations or receipt proof, allowing
  prerequisite completion to be interpreted vacuously;
- receipts used invented SHA strings, were handwritten, and lacked runner
  output and a positive terminal case;
- negative assertions depended on unstable prose and unchecked `str.replace`.

This follow-up amends only the RED/design checkpoint. The provider binder is
now a single dispatching kernel method; all canonical units own obligations;
promotion fixtures use real base and HEAD commits; runner output and exact
digest binding are executable requirements; P0/P5 omissions and incomplete
terminal receipts are negative cases; and checker failures use the stable
`tenferro.storage-ownership-diagnostics.v1` envelope. The checker and runner
remain intentionally unimplemented.

An additional coherence review required the worker/reaper auto-trait contract
and release behavior to be explicit. The remediation therefore also fixes
lease transfer to `Send + !Sync`, keeps host mappings thread-bound, and makes
exactly-once panic-contained release/quarantine part of G1.

## Rejection of checkpoint 00295401 and remediation

The second review accepted the normative pseudocode convention but rejected
four executable-proof gaps:

- the positive terminal test supplied a locally constructed receipt instead
  of proving terminal status from runner output;
- one diagnostic helper did not validate the diagnostic schema/array shape;
- lease auto-traits and release lifecycle were checked only as document text,
  not canonical compile/runtime obligations;
- the borrow-checking proof was an ephemeral manual `rustc` invocation rather
  than a repository test.

This follow-up removes receipt construction entirely. Every positive checker
case consumes an actual future runner receipt, and adversarial receipt cases
mutate a schema-checked runner result through verified lookup/removal helpers.
The all-active terminal case executes the runner first and requires receipt
entries for the auto-trait and release-lifecycle obligations before asking the
checker for `terminal: true`. Both diagnostic assertion paths validate the
`tenferro.storage-ownership-diagnostics.v1` envelope and non-empty structured
diagnostic array.

The canonical graph now owns a deferred P4/G1+G4 production-bound compile/test
artifact for the private dispatch borrow and exact `ResolvedWrite` recovery,
plus deferred P3/G1+G4 auto-trait and P4/G1+G3 provider runtime artifacts.
The base snapshot is the coherent P0/P1/P2-complete state; P4/P5 and every
P3/P9 member obligation are activated and receipted in the CUTOVER candidate.
Any synthetic borrow snippet is supplemental only; private-name text matching
is not a proof.

## Fresh SPEC review remediation

The fresh Phase 1 SPEC review found five weaknesses in checkpoint `22ea3d63`.
This follow-up changes only the executable RED specification and its normative
records; the checker and runner remain intentionally unimplemented.

- Production assertions are now independent. The current checked-in manifest
  is explicitly asserted to be v1 and unequal to the canonical v2 model. A
  separate gate requires the future v2 checker to reject that v1 input, and its
  v2 branch requires exact canonical equality before checker success.
- Every receipt execution is checked against the parsed candidate manifest for
  obligation, artifact, command, candidate commit, artifact digest, and
  command digest. Three negative mutations require the exact
  `E_RECEIPT_EXECUTION_BINDING` code and field set.
- RED event matching uses a `Counter`-backed multiset. Extra duplicate events,
  missing events, skips, and total-count mismatches are explicit failures.
- Symlink support is probed by an independent required test. Unsupported
  capability is an unexpected RED event; dependent cases use no optional
  expected-failure escape. Temporary external symlink fixtures are cleaned in
  `finally` blocks.
- The normative production-bound borrow statement now names P4, and the
  diagnostic registry documents the new execution-binding code.

## Verification evidence for this checkpoint

Passing deterministic checks:

- `python3.12 -m py_compile scripts/test-storage-ownership-contracts-v2.py`
- generated v2 manifests parse with Python 3.12 `tomllib`, including promoted
  CUTOVER and all-active terminal candidates, and cover every canonical unit;
- canonical deferred compile/runtime rows for auto-traits and provider release,
  plus a RED that invokes those exact future artifact commands;
- the deferred P4 production-bound borrow artifact/command row, with no
  self-referential inline source accepted as canonical evidence;
- `python3.12 scripts/check-storage-ownership-contracts.py`;
- `python3.12 scripts/test-check-storage-ownership-contracts.py` (65 tests);
- `cargo doc --workspace --no-deps`;
- `python3.12 scripts/check-docs-site.py` and
  `python3.12 scripts/test-check-docs-site.py`;
- `bash scripts/check-pr-fast.sh --coverage-reviewed ...` with py-compile and
  the two implementation-independent RED self-checks (including all root and
  extension-manifest fmt/clippy/doc-snippet checks);
- deterministic `scripts/repository-rules-review.py --worktree --dry-run`
  (pass; external LLM review explicitly skipped);
- `git diff --check`.

The following RED result is intentional and is evidence that the implementation
surface has not been silently added in this checkpoint:

- `python3.12 scripts/test-storage-ownership-contracts-v2.py` runs 39 tests and
  reports exactly 43 expected failure/subtest events. The emitted
  `tenferro.storage-ownership-red-report.v1` has zero unexpected failures and
  zero missing expected events, equal expected/observed event counts, and no
  skipped tests. The causes are machine-readable: v2 checker absent (23
  events), v2 runner absent (17 events), and future production proof artifacts
  absent (3 events). The required symlink capability test passes on this host;
  an unsupported host would add an unexpected capability event and return 2.
  There are no unexpected Python errors.

No cargo implementation tests are claimed here because this checkpoint changes
only design, ledger specification tests, and provenance. The next Phase 1
checkpoint must first implement the checker/runner against this RED contract,
then run the full v2 suite, the exact production manifest, the source
inventory, trybuild, parity, docs, and repository quality gates.

## Residual work and change control

- Replace the v1 production ledger with the v2 single-table schema in the
  checker/runner implementation checkpoint; do not add a v1 compatibility
  parser.
- Keep the production-manifest equality assertion coupled to the canonical
  `UNITS`/`EDGES`/obligation model; do not weaken it to schema or checker-exit
  checks.
- Implement filesystem-aware artifact resolution, promotion comparison,
  candidate-bound receipts, typed command execution, and derived terminal
  reporting to satisfy this RED suite.
- Keep this design document and this RED specification in the same PR for any
  semantic contract change. A later phase may refine provisional names only by
  updating the contract and its executable tests together.
- The obsolete `HANDOFF-2026-07-25-tenferro-unification6-wip.md` remains a
  Phase 13 cleanup item; it is intentionally not deleted in this Phase 1
  checkpoint.
