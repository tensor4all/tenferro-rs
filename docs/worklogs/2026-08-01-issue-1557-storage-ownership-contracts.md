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
checkpoint. The existing checker file and v1 test suite are superseded
deletion debt owned by the immediate atomic checker implementation checkpoint;
they are not a compatibility implementation or an accepted compatibility
surface for v2. That checkpoint must replace them atomically, migrate the
production manifest, and reduce the legacy fixture to a schema-only negative
fixture. No production storage API, unsafe importer, corruption hook, or test
escape hatch was added here.

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
  Shell strings, empty argv, exact argv mutations, absolute/escaping cwd,
  path-bearing argv escapes even when `path_args` lies, cwd/argv symlink
  escapes, unknown command kinds, unbound target links, and a failed active
  command are errors. Active commands run once; deferred commands never run.
- Checker/runner availability is a strict CLI handshake. Each future tool must
  accept `--contract-schema`, return exit code 0, emit the exact JSON contract
  for its role and v2 manifest schema on stdout, and emit no stderr. The RED
  suite invokes and parses that probe; source spelling, comments, unused
  constants, path existence, wrong JSON, extra stdout, stderr noise, and a
  wrong schema/tool are not availability evidence.
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
- the atomic migration checkpoint: exact v2 production registry, v2-only
  checker source, deletion/replacement of the v1 suite, and a schema-only
  negative legacy fixture;
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
  case requires an exact diagnostic code set and field shape. Command
  confinement additionally fixes exact argv allowlist differences, cwd
  lexical escapes, argv path escapes independent of `path_args`, cwd/argv
  symlink escapes, and post-receipt command-path retarget revalidation;
- exact checker and runner `--contract-schema` probes, including positive
  fake-tool cases for both roles and adversarial comment, unused-constant,
  file-existence, wrong-JSON, wrong-schema, stderr, and extra-output cases;
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

## QUALITY follow-up: confinement and atomic migration contract

The remaining Phase-1 QUALITY scope is intentionally limited to the executable
RED contract; no checker or runner implementation is included. The command
contract now treats each command's `argv` as an exact allow-listed vector,
canonicalizes `cwd` and every path-bearing argv value independently of the
advisory `path_args` metadata, and rejects absolute paths, normalized `..`
escapes, and symlink escapes with command-specific identifying fields. A
separate required symlink-capability test remains the prerequisite; dependent
cases never convert capability failure into a skip. The same command-path
symlink diagnostic is required when a path accepted for a receipt is retargeted
outside the repository before checker revalidation.

The atomic migration test is intentionally RED only while the deterministic
legacy predicate proves the current v1 manifest/checker/suite/fixture bytes are
still exactly the frozen quartet below. Once that predicate is false, the test
requires the production manifest to equal the independent v2 verifier
expectation exactly and both future tools to pass their exact CLI probes. It
also rejects the v1 parser/schema surface, requires the v1 suite to be absent,
and requires the legacy fixture to contain only its v1 schema marker. The
production TOML remains the sole machine registry authority after migration;
Python constants in this RED script are independent verifier expectations and
the design graph is explanatory documentation.

The temporary RED-only quartet is byte-pinned as follows:

- `scripts/storage-ownership-contracts.toml` (production v1 manifest):
  `7694da2a07fb702cdc0e2003eeff6b2610d1b8714cd19f78a04b07e4c9082fcf`
- `scripts/check-storage-ownership-contracts.py` (v1 checker):
  `91ab78217adbb74f8f6bf55a48ec6bb0c6c7eea17b9c51251dcdc092627dc718`
- `scripts/test-check-storage-ownership-contracts.py` (v1 test suite):
  `e4dbf32d274f7671430a7a1e474016337b60fcab555087e2d111d093acccbdfe`
- `scripts/fixtures/storage-ownership-contracts-v1.toml` (full v1 fixture):
  `fed8c80e0e5b8969f18a46f729644bad267adeb8a137499638d3a4926ed1b2ec`

Only that exact pre-migration byte state may emit the typed
`v2-atomic-migration-not-landed` RED event. The frozen hashes, their predicate,
and that expected migration event are temporary RED evidence: the atomic v2
implementation commit must delete all of them, with no v1 compatibility
surface.

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

The argv allowlist RED coverage now exercises every index of every distinct
canonical command ID, plus one missing-final-argument and one appended-extra-
argument case per ID. Shared IDs are mutated at every occurrence before
checking exact binding fields or the typed `E_COMMAND_ARGV_LENGTH` fields
(`command_id`, integer `expected`, integer `actual`). The self-tests compare
the complete `(command_id, index)` and length-case sets and multiplicities with
the canonical domains. This is coverage only; runner observed argv/cwd, real
command kinds, TOCTOU, and receipt diagnostics remain deferred.

Checkpoint 1C adds the post-migration tooling inventory. It scans every
regular `.py` and `.toml` file below `scripts`, including renamed/moved files,
using lexical path/token checks only; it is not a v1 parser or compatibility
loader. Exact path+token allowances are purpose-labeled and count-checked for
the schema-only fixture, canonical checker path, and intentional v2 RED
rejection strings. Always-run self-tests prove repository drift is empty,
extra V2-suite occurrences are rejected, clean v2 tooling passes, and renamed
checker/suite/TOML/parser/shim surfaces produce exact path/token evidence. The
atomic implementation commit must remove the frozen v1 hashes, quartet
predicate, and `v2-atomic-migration-not-landed` event.

Checkpoint 2A strengthens runner evidence without implementing the runner. Each
marker child writes a structured observation containing raw and normalized
process argv, a separately named interpreter image, portable PATH-resolved
requested executable identity, resolved cwd, artifact path/digest, and a unique
nonce/challenge. The positive receipt contract binds
those exact child observations into `argv`, `cwd`, `artifact_path`,
`executable`, `observation_nonce`, and `observation_challenge` fields. RED cases
cover missing, duplicate, forged, and swapped observations, plus receipt argv/
cwd/artifact/executable/nonce/challenge mutations, with precise binding
diagnostics. The generic runner path uses the existing canonical `cargo-test`
obligation with a temporary `bin/cargo` PATH shim injected only into the
runner subprocess; it preserves the untouched production cargo argv and kind
while avoiding real cargo/hardware work. Other active obligations remain
Python markers. The nonce/challenge is evidence of independent child emission
under the test model, not cryptographic resistance to a malicious runner
forging child output.

Checkpoint 2B/2C closes the remaining RED evidence gaps without implementing
the checker or runner. Post-receipt revalidation now requires resolved-path
identity for all three filesystem dimensions: command argv path, command cwd,
and artifact path. The cwd case retargets an in-repository symlink to another
in-repository directory while command and artifact bytes stay unchanged. The
command case retargets an in-repository command-path symlink to a different
in-repository executable with identical bytes and uses field
`argv[1].resolved_path`; confinement and command digest therefore cannot mask
the resolved-path change. The artifact case retargets an in-repository artifact symlink
to an external target with identical bytes; confinement is not weakened, and
`E_RECEIPT_PATH_IDENTITY` must report the obligation, field, expected resolved
path, and actual resolved path. Existing command-path retarget coverage remains
explicit.

Receipt digest diagnostics are now typed separately for receipt-level manifest
digests (`E_RECEIPT_MANIFEST_DIGEST`) and obligation-level artifact/command
digests (`E_RECEIPT_DIGEST`). Every such assertion requires exact field,
expected, and actual values, with `obligation_id` where applicable. All
obligation-scoped execution and child-observation mutation cases retain the
same four identifying fields, and the diagnostic envelope self-test rejects a
missing/wrong obligation ID or an empty message. Messages remain wording-flexible
but must be non-empty. The child nonce/challenge remains test-model evidence,
not cryptographic resistance to a malicious runner forging child output.
Digest diagnostics use current canonical recomputation as `expected` and the
recorded receipt claim as `actual`; path-identity diagnostics use the
receipt-time resolved path as `expected` and the post-receipt resolved path as
`actual`.

The final Phase 1 RED remediation closes four spec-review gaps. The atomic
migration branch now inventories the v2 suite itself for every temporary
migration sentinel: all four frozen SHA-256 values, the quartet constant and
predicate, the migration-cause constant and literal slug, and its expected RED
registry event. The independent source assertion and its mutation self-tests
build string/hash targets from split pieces, so the proof does not retain a
forbidden literal merely by naming it. Exact runtime dictionary membership
separately detects the registry event regardless of source spelling. The
atomic implementation commit must delete all of these sentinels together;
current exact-quartet behavior remains the sole typed migration RED.

The tooling inventory now distinguishes storage-specific evidence from
ordinary vocabulary. Exact retired storage signatures and the old suite path
remain forbidden, while generic fixture/legacy/v1/compatibility terms require
a storage-ownership path or source anchor. Temporary trees prove renamed and
moved storage checker/suite/parser/TOML/shim evidence is rejected with exact
path/token records, the canonical checker path is required, and unrelated
scripts using the same generic words and flags remain clean. The canonical
checker is proved v2 by its exact CLI probe, not by a source-spelling rule. The
inventory remains a lexical deletion-debt scanner, never a v1 parser or
compatibility mode.

The receipt envelope is frozen to exactly `schema`, `base_commit`,
`candidate_commit`, `base_manifest_sha256`, `candidate_manifest_sha256`, and
`executions`. Each required field has its own `{case, field}` missing-field RED
subtest, and one extra `terminal` case rejects a second status authority.
`E_RECEIPT_SHAPE` carries exact string `field`, `expected`, and `actual` values
plus a non-empty message; self-tests reject missing or wrong diagnostic fields.

Finally, future proof coverage is derived from all 15 canonical
`DEFERRED_OBLIGATIONS`, not a second three-item list. Every subtest validates
the complete unit/gate/artifact/state/command contract before reporting the
exact future-artifact cause or executing an existing artifact's canonical
command from its canonical cwd. Existing-file-only promotion remains rejected,
and promotion identity now has separate artifact and command mutation cases,
so a noncanonical manifest cannot supply an arbitrary command for execution.

The following RED result is intentional and is evidence that the implementation
surface has not been silently added in this checkpoint:

- `python3.12 scripts/test-storage-ownership-contracts-v2.py` runs 68 tests and
  reports exactly 222 expected failure/subtest events. The emitted
  `tenferro.storage-ownership-red-report.v1` has zero unexpected failures and
  zero missing expected events, equal expected/observed event counts, and no
  skipped tests. The causes are machine-readable: v2 checker absent (168
  events), v2 runner absent (38 events), future production proof artifacts
  absent (15 events), and atomic v2 migration not landed (1 event). The event
  registry matches exception type, failure/error kind, cause, test, and
  subtest parameters as a multiset. The required symlink capability test
  passes on this host; an unsupported host would add an unexpected capability
  event and return 2. There are no unexpected Python errors.

No cargo implementation tests are claimed here because this checkpoint changes
only design, ledger specification tests, and provenance. The next Phase 1
checkpoint must first implement the checker/runner against this RED contract,
then run the full v2 suite, the exact production manifest, the source
inventory, trybuild, parity, docs, and repository quality gates.

## Residual work and change control

- The immediate checker/runner implementation checkpoint owns the atomic
  migration: replace the v1 production ledger with the exact v2 single-table
  registry, delete/replace the v1 checker parser and test suite, and reduce
  the legacy fixture to schema-only rejection. Do not add a v1 compatibility
  parser or retain the old tooling as a supported path. It must also delete
  all four frozen quartet hashes, `LEGACY_V1_QUARTET_SHA256`,
  `_legacy_tooling_is_current`, `MIGRATION_CAUSE`, its literal cause slug, and
  the corresponding expected RED event. The post-migration assertions prove
  their absence; these are temporary RED-only sentinels, not a compatibility
  surface.
- Keep the production-manifest equality assertion coupled to the sole machine
  registry and the independent `UNITS`/`EDGES`/obligation verifier
  expectations; do not weaken it to schema or checker-exit checks.
- Implement filesystem-aware artifact resolution, promotion comparison,
  candidate-bound receipts, typed command execution, and derived terminal
  reporting to satisfy this RED suite.
- Keep this design document and this RED specification in the same PR for any
  semantic contract change. A later phase may refine provisional names only by
  updating the contract and its executable tests together.
- The obsolete `HANDOFF-2026-07-25-tenferro-unification6-wip.md` remains a
  Phase 13 cleanup item; it is intentionally not deleted in this Phase 1
  checkpoint.
