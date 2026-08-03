# Storage Ownership Contracts

Normative contract document for the storage ownership redesign tracked by
issue [#1555](https://github.com/tensor4all/tenferro-rs/issues/1555). This
document is the Phase 1 deliverable of
[#1557](https://github.com/tensor4all/tenferro-rs/issues/1557): it turns the
seven design gates in #1555 into contracts precise enough to review and test
implementation PRs against.

Authority and change control:

- The #1555 issue body owns the architecture invariants (referenced below as
  I1 through I10, numbered as in its "Required invariants" section). This
  document owns the contracts. The phase issues (#1556 through #1569) own
  work decomposition and verification inventories.
- Any semantic change to a contract in this document must update this
  document and its tests in the same PR.
- Signature sketches are normative in shape: which capability a call
  requires, what it borrows and for how long, what it consumes, and what the
  failure carrier is. Exact public names remain provisional until the owning
  phase lands (#1559, #1560, #1561, #1562, #1565). The public API is
  intentionally unsettled until then; do not treat sketch names as released
  surface.
- Gates G1 through G7 below block the storage implementation phases (#1558
  and later). Phase 0 (#1556, engine identity) is independent and may proceed
  at any time.

Long-term architecture quality gate:

- The redesign has one ownership kernel shared by CPU, CUDA, WebGPU, Apple/
  Metal, runtime submission, and AD. Providers may differ in synchronization,
  mapping, and endpoint capabilities, but not in the meaning of owner,
  shared/exclusive capability, descriptor, claim, or lease.
- Owner claims, Rust access capabilities, provider-resource pins, and
  descriptor-liveness roots are four separate concepts. No implementation
  may merge them for convenience or recover one from another through a
  reference count, downcast, raw handle, or provider-specific shortcut.
- New APIs are derived from ownership and lifetime requirements in this
  document, not shaped around preserving legacy call sites. Backward
  compatibility is not a requirement for #1555.
- Old and new storage representations coexist without conversion only until
  the atomic CUTOVER. No migration, legacy, provider, compatibility,
  submission, or conservative-synchronization bridge ships; CUTOVER switches
  every owner, access, retention, runtime, and AD user together and deletes the
  old representation.
- A phase is incomplete when it merely adds the new path: it must also delete
  the replaced path and update contract tests, documentation, and source
  inventories. Exceptions require a contract change reviewed before the
  implementation PR, never an ad hoc implementation waiver.

## Conventions

Terminology:

- **Allocation**: one physical memory root and its claims. `AllocationSpan` is
  metadata (a domain-qualified key plus byte range), never an access
  capability or proof of ownership.
- **Owner**: the unique, non-cloneable ownership token for an allocation span
  (`OwnedStorage`, or a tensor/group wrapping it). Invariant I1.
- **Capability**: the right to access storage, expressed as Rust
  ownership/borrows: shared (`StorageRef`, views), exclusive (`StorageMut`,
  mutable views), or owning (consuming APIs). Invariant I2.
- **Descriptor**: a typed interpretation (dtype, layout, placement) referring
  to an allocation slot. Descriptors never own storage.
- **Group**: one or more owners plus descriptors (`AllocationGroup`), the only
  representation for "one allocation, many logical values".
- **Guard**: a borrow-carrying value granting host byte access to a validated
  span (`HostReadGuard`, `HostWriteGuard`).
- **Lease** (`UseLease`): a provider-private retention and ordering record for
  enqueued device access. A lease is not a capability: it never authorizes
  safe access by itself and cannot mint ownership or write authority
  (invariant I3). It pins provider resources until retirement.
- **Endpoint**: an engine/device access point (`AccessEndpoint`). Distinct
  from `AllocationDomainId`, which is allocation identity (#1555, "Identity
  vs endpoints").
- **Retirement**: the point when all provider events covering an access have
  completed and retained resources may be released. Invariant I6.
- **Prepared access**: the result of `prepare_read` or `prepare_write` after
  the one access-boundary validation. It carries the validated `CheckedLayout`
  and the Rust borrow required by the access; the hot loop consumes it through
  the `iter_contiguous*` typed-slice path or the `iter_strided*` prepared-cursor
  path.
- **Completion-unproven retention**: the typed-error path used when a
  provider cannot prove completion. A provider-private `UnprovenRetirement`
  owns the event, `Arc<RootResource>`, and provider context until completion is
  proven or those resources are intentionally made a permanent leak; this is
  not a quarantine state.

State-table columns. Every state-transition row in this document answers the
six review-checklist questions from #1557, abbreviated as:

| Column | Meaning |
|--------|---------|
| cap | capability required (shared / exclusive / owning / none) |
| borrow | what is borrowed and for how long |
| sync | provider synchronization performed (waits are documented synchronization points, never copies; invariant I4) |
| fail | return on failure (ownership must be recovered or provably retained) |
| panic/drop | state retained if the caller panics or drops mid-operation |
| reclaim | when reclamation of the affected allocation becomes legal |

Error conventions: all failures are structured errors carrying operation,
requested range/ids, and resolved span identity, without raw addresses.

## Phase 1 verification ledger

The machine-readable production registry is
`scripts/storage-ownership-contracts.toml`. It is the sole machine authority
for the production graph, obligations, commands, and lifecycle state. The
graph in this document is explanatory; the v2 checker and runner execute the
same tagged rows and do not maintain parallel active/deferred tables. The
checked-in v2 test suite derives counts from the manifest and checks the
contractual active/deferred IDs without becoming another production registry.

The checker is v2-only, the old v1 test authority is deleted, and the retained
v1 fixture contains only its schema marker so that an old manifest is rejected
without a compatibility parser. The real design-document checker is an active
P1 obligation. The current production state deliberately activates only these
four P1 rows: `p1-ledger`, `p1-contract-document`, `p1-api-parity`, and
`p1-element-access-baseline`. P0 control-plane and P2 root claims remain
deferred until their real artifacts and verifiers land. No missing deferred
artifact is fabricated to make this phase terminal.

### One canonical graph

P0--P13 are stable identifiers, not a numeric merge order. The registry owns
units, edges, gates, and atomic cohorts exactly once. Obligations refer to a
unit ID and gate IDs; they do not repeat issue numbers, phase labels, or
convergence targets in payloads. P13 is split into two units because freeze
and final closure have different evidence requirements.

```text
p0-control-plane (#1556) ────────────────────────────┐
                                                     │
p1-contract-ledger (#1557) ──> p2-root-claims (#1558) │
                                      │              │
                                      v              │
                         p4-access-retirement (#1560)│
                                      │              │
                                      v              │
                         p5-allocation-group (#1561) │
                                      └──────┬───────┘
                                             v
             atomic CUTOVER: p3-host-ownership (#1559)
                    + p9-runtime-ad-cutover (#1565)
                                             │
                                             v
                              p6-reinterpret (#1562)
                              ┌──────────────┴──────────────┐
                              v                             v
                       p7-cuda (#1563)              p8-webgpu-metal (#1564)
                              └──────────────┬──────────────┘
                                             v
                              p10-api-normalization (#1566)
                                             │
                                             v
                              p13-freeze (#1567-A)
                              ┌──────────────┴──────────────┐
                              v                             v
                       p11-hardware (#1568)          p12-docs (#1569)
                              └──────────────┬──────────────┘
                                             v
                              p13-closure (#1567-B)
```

P0 and P1 are independent roots. P2 has exactly one incoming edge, from P1;
P0 does not gate root-claim design or any phase before cutover. The CUTOVER
cohort has both P0 and P5 as prerequisites and activates p3 and p9 atomically.
The cohort contains exactly the host owner and final runtime/AD units. It is
the only place where public host ownership, detached/scoped runtime ownership,
and direct group-based AD retention are introduced. There is no Phase 3
AD-retention adapter, Phase 4 AD bridge, minimal submit bridge, or
conservative pre-retirement synchronization path to promote or delete.
Until atomic CUTOVER, the old and new representations coexist without
conversion. No provider or legacy bridge is a production obligation or
shippable artifact; CUTOVER activates the final representation and deletes the
old one atomically.

### v2 manifest and promotion contract

The manifest has one `[[obligations]]` table. Each row has an immutable
obligation ID, one typed `artifact` specification, one typed `command`
specification, one or more validated gate IDs, one registry `unit` ID, and a
tagged `state` whose discriminant is exactly `active` or `deferred`. The
deferred variant additionally carries its canonical activation unit and
promotion rule.
There are no parallel active/deferred tables, status strings, terminal flags,
fixture registries, or synthetic terminal artifacts. A deferred artifact is a
real future path specification, not an invitation to invent a fixture at
promotion time.

The artifact specification contains a stable ID, kind, exact repository-
relative paths, and an optional structural selector. The command specification
contains a stable ID, an allow-listed typed kind, argv/cwd, path arguments, and
the exact `artifact_id` it is permitted to inspect. Shell strings, arbitrary
commands, and command-to-artifact inference are invalid. A command may be
shared only when its complete typed value and artifact binding are identical.

A promotion compares a base and candidate manifest. For every promoted row it
must preserve the same obligation ID, artifact value, command value, and
registry ownership, changing only the tagged state from deferred to active.
The complete registry value—units, gates, edges, and cohorts—must be exactly
unchanged between base and candidate; graph edits are not promotions.
Every member obligation of an atomic cohort must make that transition in the
same candidate; partial cohort activation is rejected. A changed artifact,
command, ID, unit, or gate is a new obligation, not a promotion.

Every registered unit owns at least one required obligation. A unit is
complete only when all of its required obligations are active and the
candidate-bound receipt contains a successful execution result for every one
of them. An empty obligation set is invalid rather than vacuously complete.
If any obligation in a target unit is active, every obligation in each direct
incoming source unit must already be active. This prerequisite is derived from
the registry edges and obligation tagged states, without a transitive or
parallel lifecycle table. In particular, P2 cannot activate while the P1
element-access baseline remains deferred.
In particular, CUTOVER cannot activate until P0 and P5 are each complete by
this rule; merely naming them as cohort prerequisites is not proof.

The trusted runner emits a small candidate-bound execution log:

```json
{
  "schema": "tenferro.storage-ownership-receipt.v1",
  "candidate_commit": "HEAD from git rev-parse",
  "base_commit": null,
  "executions": [
    {
      "obligation_id": "p1-ledger",
      "argv": ["python3", "scripts/check-storage-ownership-contracts.py"],
      "cwd": ".",
      "artifact_path": "scripts/storage-ownership-contracts.toml",
      "exit_code": 0
    }
  ]
}
```

The top-level fields are exactly `schema`, `candidate_commit`, `base_commit`,
and `executions`. `base_commit` is a fixed field and is null when no
transition comparison was requested. Each execution contains only its
obligation ID, canonical manifest argv, repository-relative cwd and artifact
path, and process exit status. Executions are sorted by obligation ID. Command
and artifact IDs are derived from the candidate row and are not repeated in the
receipt. Git object IDs are opaque strings returned by Git; no length or format
is assumed.

The runner executes every active typed argv exactly once and never executes a
deferred row. The checker derives terminal state from the tagged rows and
successful executions; a receipt cannot declare terminality. Candidate identity
is `git rev-parse HEAD`, and an optional base must be an ancestor of that
candidate. Promotion changes only deferred state and preserves the row's
obligation, unit, gates, artifact, and command values. Atomic cohort members
must transition together and their prerequisites must already be active. The
registry itself is immutable across the promotion comparison.

This is a trusted-runner execution log, not a security attestation. Repository
source, maintainers, build tools, and the CI runner are trusted. For a tracked
manifest, verifier, command target, or artifact, the candidate commit and
repository-relative path identify the bytes. Before executing or accepting a
receipt, the tools require the tracked tree to match candidate `HEAD` using a
single Git cleanliness check. Untracked or ignored receipt output, build
targets, logs, and unrelated user files are allowed. The check is operational
provenance, not anti-tamper machinery, and is not applied to a globally empty
worktree.

The checker confines the manifest, artifact, cwd, and path-bearing argv values
to the repository before execution. Resolution follows ordinary filesystem
paths and rejects a path whose symlink resolves outside the repository. There
is no post-receipt retarget protocol. No content checksum is part of this
tracked-artifact contract; a checksum would be justified only for a concrete
untracked or cross-system artifact boundary, which Phase 1 does not introduce.

The generic runner test supplies a temporary repository with the real active
Python verifiers and a local `cargo` executable that records the exact argv.
This tests structured argv execution and exit-status propagation without adding
a production command mode, child protocol, or runner escape hatch.

The structural shape is intentionally explicit. A production row has one
artifact, one command, and one tagged state; the state is not split into
parallel active/deferred tables:

```toml
[[obligations]]
id = "p4-access-retirement"
unit = "P4"
gates = ["G1"]
artifact = { id = "artifact-corruption-map", kind = "corruption-test", path = "crates/tenferro-tensor/src/storage/tests/corruption_map.rs" }
command = { id = "cmd-corruption-map", kind = "cargo-test", argv = ["cargo", "test", "-p", "tenferro-tensor", "--lib", "storage::tests::corruption_map"], cwd = ".", path_args = [], artifact_id = "artifact-corruption-map" }
state = { kind = "deferred", activation_unit = "P4", promotion = { mode = "activate-in-place" } }
```

Promotion changes only `state.kind` from `deferred` to `active`. The artifact
ID/path/kind, command ID/kind/argv/cwd/path arguments, unit, and gates remain
byte-for-byte identical. A changed value is a new obligation and cannot be
accepted as a promotion. The v2 specification fixes the checker invocation
for base/candidate comparison and the receipt fields so that a later checker
cannot silently weaken this rule.

The v2 checker accepts `--root`, repository-relative `--manifest`, optional
`--base-commit` plus `--receipt` for promotion/receipt validation,
`--summary-json`, and `--diagnostics-json`. The runner accepts `--root`, the
same repository-relative `--manifest`, `--base-commit`, `--receipt-out`, and
`--diagnostics-json`.
Neither tool accepts a caller-supplied candidate commit: candidate identity is
HEAD. An optional base revision is canonicalized with Git using
`git rev-parse --verify <revision>^{commit}` before promotion comparison; the
receipt stores that resulting opaque object ID, never a branch or revision
alias. Neither tool may infer a different manifest or command target from the
current working directory. A receipt written by the runner is the only
execution proof consumed by the checker.

Hosted `ci-config` checks fetch full Git history and pass exactly one canonical
event base to the existing production checker: `pull_request.base.sha` for a
pull request and `github.event.before` for a push. The local `ci-config`
profile omits the base by default and remains a structural developer check.
Supplying a storage-ownership base without selecting `ci-config` is invalid.

Availability is an explicit CLI contract, not source inspection or path
existence. Both tools accept `--contract-schema`, exit successfully
without loading a manifest, write exactly one JSON object to stdout, and write
nothing to stderr. The v2 suite invokes the script with the current Python
interpreter and accepts the result only when its parsed object equals the
corresponding contract below including the complete `options` list; comments,
unused constants, an existing file, non-JSON output, extra keys, stderr noise,
and a wrong schema or tool are unavailable:

```json
{
  "schema": "tenferro.storage-ownership-cli-contract.v1",
  "tool": "check-storage-ownership-contracts",
  "role": "checker",
  "manifest_schema": "tenferro.storage-ownership-contracts.v2",
  "probe": "--contract-schema",
  "options": [
    "--root",
    "--manifest",
    "--base-commit",
    "--receipt",
    "--summary-json",
    "--diagnostics-json"
  ]
}
```

The runner returns the same shape with `tool` equal to
`run-storage-ownership-contracts`, `role` equal to `runner`, and `options`
equal to `--root`, `--manifest`, `--base-commit`, `--receipt-out`, and
`--diagnostics-json`:

```json
{
  "schema": "tenferro.storage-ownership-cli-contract.v1",
  "tool": "run-storage-ownership-contracts",
  "role": "runner",
  "manifest_schema": "tenferro.storage-ownership-contracts.v2",
  "probe": "--contract-schema",
  "options": [
    "--root",
    "--manifest",
    "--base-commit",
    "--receipt-out",
    "--diagnostics-json"
  ]
}
```

This probe is only an availability handshake; it does not execute ledger
commands or provide runner evidence.

Checker and runner failures have a stable machine-readable envelope when
`--diagnostics-json` is selected:

```json
{
  "schema": "tenferro.storage-ownership-diagnostics.v1",
  "diagnostics": [
    {
      "code": "E_RECEIPT_INCOMPLETE",
      "fields": {"obligation_id": "p5-allocation-group"},
      "message": "supplemental human explanation"
    }
  ]
}
```

`code` and the identifying `fields` are compatibility-stable within schema
v1; human `message` text is not. The v2 suite therefore asserts codes and
relevant IDs/paths plus a non-empty human message, without freezing message
wording. Each one-fault case requires the exact one-code set and the exact
field-key shape registered by the suite; duplicate codes, unknown codes,
extra envelope keys, missing or empty `message`, or extra identifying fields
are failures. This prevents a checker from passing a negative case by emitting
every known code or an unrelated diagnostic.

The v2 diagnostic registry covers schema shape, tagged lifecycle state,
registry graph/cohort and direct-prerequisite rules, artifact/path confinement,
exact command allowlists, promotion identity and registry immutability,
command exit status, receipt shape and candidate binding, tracked-tree
provenance, and derived terminality. Each diagnostic has one stable code, an
exact identifying field set, and a non-empty human message. In particular,
command argv binding reports the command ID, index, expected value, and actual
value; path failures are emitted before command identity comparisons; and
receipt execution binding reports the obligation ID, field, expected value,
and actual value. The suite asserts the structured envelope and the relevant
fields without freezing human wording.

The v2 suite uses temporary repositories for reachable path, graph, promotion,
command, receipt, and exit-status mistakes, plus an integration case for the
checked-in production manifest. Counts are derived from the parsed manifest;
the suite does not preserve a migration-event registry or historical totals.
It verifies the four currently implementable P1 rows, including the measured
element-access baseline. P0 control-plane and P2 root claims remain deferred,
and the remaining future rows are not executed or materialized. Fake active
artifacts are rejected because they would turn missing scientific evidence
into a green lifecycle state rather than proving the underlying work.

Command tests cover exact typed argv and repository path confinement, including
the ordering rule that a path escape is reported before a later command-identity
comparison. The runner integration uses the real active Python verifiers and a
temporary local cargo executable to prove that the canonical argv is executed;
this fixture is not a production command mode or a lifecycle authority.

P4 and P5 remain deferred;
the CUTOVER candidate activates every required P4/P5 obligation and obtains
successful runner evidence before atomically activating all P3/P9 obligations.
In particular, the canonical obligation set includes:

- P4/G1+G4: a deferred production-code-bound compile/test artifact for the
  private dispatch borrow shape and exact `ResolvedWrite` failure recovery;
- P3/G1+G4: a compile contract using the repository compile-test harness or
  static assertions for `UseLease`/`BackendRawLease: Send + !Sync` and
  `BackendRawMapping`/host guards: `!Send + !Sync`;
- P4/G1+G3: a provider event-retirement runtime test proving exactly-once
  release after proven completion and intentional retain/leak plus a typed
  `CompletionUnproven` error when completion is unproven.

These are ordinary immutable artifact-command rows, so all-active terminal
proof necessarily includes their successful runner results. A synthetic
borrow snippet may be used as a supplemental design experiment, but it is not
canonical evidence and cannot satisfy the P4 obligation. Private item names in
the pseudocode are provisional; the future production-bound artifact must
prove the structural capability/borrow/recovery contract through the real
crate harness.

The proof layers remain distinct: Rust borrowing/private constructors prove
write safety; trybuild, Miri, property, corruption, and provider tests exercise
dynamic boundaries; source inventories record deletion drift; and the
source-blind documentation audit checks stale public language. None of these
is allowed to manufacture ownership proof from an allocation ID or a lock.

## G1. Span access and retirement

### Controlling permanent model

This subsection is the controlling G1 contract and the contract-side
proportional amendment to invariants I7 and I10 in
[#1555](https://github.com/tensor4all/tenferro-rs/issues/1555). The permanent
lifetime root is `Arc<RootResource>`, which keeps the physical allocation and
the provider resources it retains alive. `OwnedSpanClaim` is the unique
authority for an owned span and is deliberately non-`Clone` (and non-`Copy`).
Writes are authorized only by ordinary Rust exclusive borrows
(`&mut`/`StorageMut`), never by an access-authority, quarantine, or retirement
registry, lease, event, retry, or callback.

`prepare_read` and `prepare_write` are the access-preparation boundary. Together
they validate bounds, layout, dtype, exact root-bound span (`RootBoundSpan`),
alignment, storage, provider, and, for writes, write injectivity exactly once,
then return prepared access carrying `CheckedLayout`. The hot loop uses the
`iter_contiguous*` typed-slice path or the `iter_strided*` prepared-cursor path.
Neither path performs per-element coordinate decode or provider checks.

The concrete provider-private `UnprovenRetirement` owns the completion event,
the `Arc<RootResource>` it protects, and the provider context whenever normal
return cannot prove completion. Provider-private polling retains that owner;
only a proven-completion branch may take and `Drop` its release-capable fields,
exactly once. On a terminal poll result that cannot prove completion, or on
device loss, it intentionally
transitions to a permanent leak by suppressing `Drop` for every release-capable
field (conceptually `mem::forget` or `ManuallyDrop`). The returned typed
`CompletionUnproven` error carries diagnostics only and owns no resource whose
`Drop` could release the allocation. Speculative release is forbidden.

The permanent model explicitly removes quarantine, poison, `catch_unwind`,
access-authority/quarantine/retirement registries, retry, legacy bridge, and
repeated validation. This registry prohibition does not include the Phase 1
contract ledger or a valid descriptor container; neither is access authority
or a retention/reclamation mechanism. **All later conflicting text anywhere in
this document**, including G1, G3, G5, and the test index, is superseded by this
subsection and pending physical deletion in this same PR. That supersession
covers pseudocode, tables, transitions, obligations, and tests that describe
quarantine, poison, unwind containment, removed access-authority/quarantine/
retirement registries, retries, legacy or provider bridges, repeated
validation, or release without proven completion. This commit is internally
authoritative but not final; until that text is deleted, this subsection
controls.

### Types and acquisition surface

The following Rust block fixes normative type, visibility, lifetime, field-
split, and state-transition shape. It is intentionally architecture
pseudocode: unrelated declarations, imports, and routine method bodies may be
elided, and the block is not claimed as one standalone crate. Executability is
proved separately by the canonical P4 production-bound compile/test artifact
and the P3/P4 compile/runtime obligations above. Implementations may rename
provisional private items, but may not replace the private dispatch shape with
a request that escapes and permits a second owner/provider borrow.

```rust
use core::{cell::Cell, marker::PhantomData, ptr::NonNull};
use std::{ffi::c_void, panic::{catch_unwind, AssertUnwindSafe}, rc::Rc};

pub struct AllocationKey {
    domain: AllocationDomainId,
    local: AllocationId,
}

pub struct AllocationSpan {
    key: AllocationKey,
    byte_offset: usize,
    byte_len: usize,
    guaranteed_alignment: usize, // power of two, describes the span start
}

// Metadata only. This value is not accepted by an access or binding method.
// The exact root binding is carried by the private RootBoundSpan type.
struct RootBoundSpan {
    root: RootResourceIdentity,
    range: ByteRange,
    _sealed: PrivateToken,
}

pub(crate) struct RootResourcePin {
    root: RootResourceIdentity,
    state: Arc<RootResource>, // lifetime/deallocator state only
}

pub(crate) struct OwnedSpanClaim {
    root: RootResourceIdentity,
    span: RootBoundSpan,
    provenance: ClaimProvenance, // private, non-Clone and non-Copy
}

pub struct OwnedStorage {
    pin: RootResourcePin,
    claim: OwnedSpanClaim,
}

pub struct StorageRef<'a> {
    owner: &'a OwnedStorage,
}

pub struct StorageMut<'a> {
    owner: &'a mut OwnedStorage,
}

pub(crate) struct RootResource {
    root: RootResourceIdentity,
    // RootResourcePin's Arc pins this one provider allocation/access object.
    // There is no second Arc vtable and no per-access allocation.
    allocation: Box<dyn BackendAllocationAccess>,
}

pub struct BackendAllocationMetadata {
    // This is provider-reported validation metadata only. The kernel assigns
    // RootResourceIdentity during import; a provider cannot construct that
    // identity or any claim from this value.
    byte_len: usize,
    guaranteed_alignment: usize,
}

impl BackendAllocationMetadata {
    pub fn new(byte_len: usize, guaranteed_alignment: usize)
        -> Result<Self, MetadataError>;
    pub fn byte_len(&self) -> usize;
    pub fn guaranteed_alignment(&self) -> usize;
}

pub struct BackendAccessRange {
    pub byte_offset: usize,
    pub byte_len: usize,
    pub guaranteed_alignment: usize,
}

// This is the sole cross-crate provider extension contract. Provider crates
// implement this unsafe trait; the storage kernel alone constructs guards and
// UseLease. The implementation must uphold root identity, checked
// range/alignment, access ordering, lease retirement, and exactly-once
// provider cleanup for the allocation represented by the request. Lease parts
// and their release callback may be moved to an arbitrary retirement worker
// and invoked there exactly once. Host mapping parts need not be transferable.
// These thread-transfer clauses are part of this one unsafe implementation;
// providers do not receive a second Send-proof API. Each successful method
// returns exactly one carrier made from its request; no carrier escapes an Err.
pub unsafe trait BackendAllocationAccess: Send + Sync + 'static {
    fn metadata(&self) -> BackendAllocationMetadata;
    fn map_host_read<'a>(
        &self,
        request: &BackendReadRequest<'a>,
    ) -> Result<BackendRawMapping, AccessError>;
    fn map_host_write<'a>(
        &self,
        request: &BackendWriteRequest<'a>,
    ) -> Result<BackendRawMapping, AccessError>;
    fn acquire_device_read(
        &self,
        request: &BackendReadRequest<'_>,
        endpoint: AccessEndpoint,
    ) -> Result<BackendRawLease, AccessError>;
    fn acquire_device_write(
        &self,
        request: &BackendWriteRequest<'_>,
        endpoint: AccessEndpoint,
    ) -> Result<BackendRawLease, AccessError>;
}

// The request is opaque to provider code and can only be constructed by the
// private RootResource dispatchers below. It contains the exact pin/root
// witness, claim, and span selected by one ResolvedRead/ResolvedWrite.
pub struct BackendReadRequest<'a> {
    _private: (
        &'a RootResourcePin,
        &'a OwnedSpanClaim,
        &'a RootBoundSpan,
        PrivateToken,
    ),
}

pub struct BackendWriteRequest<'a> {
    _private: (
        &'a RootResourcePin,
        &'a mut OwnedSpanClaim,
        &'a RootBoundSpan,
        PrivateToken,
    ),
}

// These are raw extension carriers rather than storage capabilities. A
// provider constructs them from its mapping/order token through the narrow
// request helpers; no carrier contains an owner, claim constructor, mutable
// capability, or public guard/lease constructor.
pub struct BackendRawMapping {
    pin: RootResourcePin,
    provider: Option<ProviderMappingParts>,
    // Zero-sized: host pointers and guards never become Send or Sync merely
    // because a particular pointer representation has permissive auto traits.
    _thread_bound: PhantomData<Rc<()>>,
}

pub struct BackendRawLease {
    pin: RootResourcePin,
    provider: Option<ProviderLeaseParts>,
    // Cell is Send but not Sync. A lease may move to one worker/reaper, but
    // shared concurrent access is not part of the provider contract.
    _not_sync: PhantomData<Cell<()>>,
}

// SAFETY: this is a kernel implementation of the thread-transfer clause on
// the single unsafe BackendAllocationAccess contract, not a provider-facing
// proof boundary. make_raw_lease installs the exact root pin and accepts parts
// only while servicing that provider's opaque request. ProviderLeaseParts is
// intentionally not Send on its own.
unsafe impl Send for BackendRawLease {}

pub struct ProviderMappingParts {
    pointer: NonNull<u8>,
    len: usize,
    release: ProviderReleaseToken,
}

pub struct ProviderLeaseParts {
    token: NonNull<c_void>,
    release: ProviderReleaseToken,
}

pub struct ProviderReleaseToken {
    pending: Option<ProviderReleaseParts>,
}

struct ProviderReleaseParts {
    context: *mut c_void,
    release: unsafe extern "C-unwind" fn(*mut c_void),
}

impl ProviderReleaseToken {
    // This is a raw carrier constructor, not an ownership/uniqueness proof.
    // Its safety obligation is part of the one BackendAllocationAccess
    // provider-extension contract. These constructors create raw carriers
    // only; they do not establish ownership, uniqueness, or a root claim.
    pub unsafe fn from_raw_parts(
        context: *mut c_void,
        release: unsafe extern "C-unwind" fn(*mut c_void),
    ) -> Self;

    fn run_once(&mut self) -> Result<(), ProviderReleasePanic> {
        // Take before calling. Success, panic, and outer unwinding therefore
        // cannot invoke the provider callback a second time.
        let Some(parts) = self.pending.take() else { return Ok(()) };
        catch_unwind(AssertUnwindSafe(|| unsafe {
            (parts.release)(parts.context)
        }))
        .map_err(|_| ProviderReleasePanic)
    }
}

impl ProviderMappingParts {
    pub unsafe fn from_raw_parts(
        pointer: NonNull<u8>,
        len: usize,
        release: ProviderReleaseToken,
    ) -> Self;
}

impl ProviderLeaseParts {
    pub unsafe fn from_raw_parts(
        token: NonNull<c_void>,
        release: ProviderReleaseToken,
    ) -> Self;
}

impl<'a> BackendReadRequest<'a> {
    fn new_private(
        pin: &'a RootResourcePin,
        claim: &'a OwnedSpanClaim,
        span: &'a RootBoundSpan,
    ) -> Self {
        Self {
            _private: (pin, claim, span, PrivateToken::new()),
        }
    }

    pub fn range(&self) -> BackendAccessRange;
    pub fn make_raw_mapping(&self, parts: ProviderMappingParts) -> BackendRawMapping {
        BackendRawMapping {
            pin: self._private.0.clone(),
            provider: Some(parts),
            _thread_bound: PhantomData,
        }
    }
    pub fn make_raw_lease(&self, parts: ProviderLeaseParts) -> BackendRawLease {
        BackendRawLease {
            pin: self._private.0.clone(),
            provider: Some(parts),
            _not_sync: PhantomData,
        }
    }
}

impl<'a> BackendWriteRequest<'a> {
    fn new_private(
        pin: &'a RootResourcePin,
        claim: &'a mut OwnedSpanClaim,
        span: &'a RootBoundSpan,
    ) -> Self {
        Self {
            _private: (pin, claim, span, PrivateToken::new()),
        }
    }

    pub fn range(&self) -> BackendAccessRange;
    pub fn make_raw_mapping(&self, parts: ProviderMappingParts) -> BackendRawMapping {
        BackendRawMapping {
            pin: self._private.0.clone(),
            provider: Some(parts),
            _thread_bound: PhantomData,
        }
    }
    pub fn make_raw_lease(&self, parts: ProviderLeaseParts) -> BackendRawLease {
        BackendRawLease {
            pin: self._private.0.clone(),
            provider: Some(parts),
            _not_sync: PhantomData,
        }
    }
}

impl BackendRawMapping {
    fn release_once(&mut self) -> ReleaseOutcome {
        let Some(mut parts) = self.provider.take() else { return ReleaseOutcome::Retired };
        match parts.release.run_once() {
            Ok(()) => ReleaseOutcome::Retired,
            Err(panic) => {
                self.pin.quarantine(QuarantineReason::ProviderReleasePanic(panic));
                ReleaseOutcome::Quarantined
            }
        }
    }
}

impl Drop for BackendRawMapping {
    fn drop(&mut self) {
        // release_once contains provider panic. Dropping is a liveness fast
        // path; forgetting the carrier retains its root pin and is still safe.
        let _ = self.release_once();
    }
}

impl BackendRawLease {
    fn retire(mut self) -> ReleaseOutcome {
        self.release_once()
    }

    fn release_once(&mut self) -> ReleaseOutcome {
        let Some(mut parts) = self.provider.take() else { return ReleaseOutcome::Retired };
        match parts.release.run_once() {
            Ok(()) => ReleaseOutcome::Retired,
            Err(panic) => {
                self.pin.quarantine(QuarantineReason::ProviderReleasePanic(panic));
                ReleaseOutcome::Quarantined
            }
        }
    }
}

impl Drop for BackendRawLease {
    fn drop(&mut self) {
        // Retirement records normally call retire explicitly. This fallback is
        // idempotent, panic-contained, and never retries an uncertain release.
        let _ = self.release_once();
    }
}

// Private kernel wrappers. Sibling provider crates cannot construct these
// types, HostReadGuard, HostWriteGuard, or UseLease.
struct BackendReadMapping<'a> {
    raw: BackendRawMapping,
    _borrow: PhantomData<&'a [u8]>,
}

struct BackendWriteMapping<'a> {
    raw: BackendRawMapping,
    _borrow: PhantomData<&'a mut [u8]>,
}

pub(crate) struct HostReadGuard<'a> {
    mapping: BackendReadMapping<'a>,
}

pub(crate) struct HostWriteGuard<'a> {
    mapping: BackendWriteMapping<'a>,
}

impl RootResource {
    fn dispatch_host_read<'a>(
        &'a self,
        pin: &'a RootResourcePin,
        claim: &'a OwnedSpanClaim,
        span: &'a RootBoundSpan,
    ) -> Result<BackendRawMapping, AccessError> {
        self.validate_read_binding(pin, claim, span)?;
        let request = BackendReadRequest::new_private(pin, claim, span);
        self.allocation.map_host_read(&request)
    }

    fn dispatch_device_read<'a>(
        &'a self,
        pin: &'a RootResourcePin,
        claim: &'a OwnedSpanClaim,
        span: &'a RootBoundSpan,
        endpoint: AccessEndpoint,
    ) -> Result<BackendRawLease, AccessError> {
        self.validate_read_binding(pin, claim, span)?;
        let request = BackendReadRequest::new_private(pin, claim, span);
        self.allocation.acquire_device_read(&request, endpoint)
    }

    fn dispatch_host_write<'a>(
        &'a self,
        pin: &'a RootResourcePin,
        claim: &'a mut OwnedSpanClaim,
        span: &'a RootBoundSpan,
    ) -> Result<BackendRawMapping, AccessError> {
        self.validate_write_binding(pin, &*claim, span)?;
        let request = BackendWriteRequest::new_private(pin, claim, span);
        self.allocation.map_host_write(&request)
    }

    fn dispatch_device_write<'a>(
        &'a self,
        pin: &'a RootResourcePin,
        claim: &'a mut OwnedSpanClaim,
        span: &'a RootBoundSpan,
        endpoint: AccessEndpoint,
    ) -> Result<BackendRawLease, AccessError> {
        self.validate_write_binding(pin, &*claim, span)?;
        let request = BackendWriteRequest::new_private(pin, claim, span);
        self.allocation.acquire_device_write(&request, endpoint)
    }
    // Each private dispatcher revalidates state.root == pin.root ==
    // claim.root == span.root and checked containment, constructs the opaque
    // request, and calls this state's provider before any borrow escapes.
    // Provider code never receives a receiver selected independently from the
    // request capability.
}

pub(crate) struct ResolvedRead<'a> {
    capability: StorageRef<'a>, // exact owner/claim/pin selected by resolve
    span: RootBoundSpan,         // exact span from that capability's claim
    _sealed: PrivateToken,
}

pub(crate) struct ResolvedWrite<'a> {
    capability: StorageMut<'a>, // exact exclusive owner/claim/pin
    span: RootBoundSpan,         // exact span from that capability's claim
    _sealed: PrivateToken,
}

pub(crate) struct UseLease {
    span: RootBoundSpan,
    mode: AccessMode,
    provider: BackendRawLease,
    // No StorageMut conversion, raw write authority, Clone, or public constructor.
}

fn auto_trait_contract() {
    fn assert_send<T: Send>() {}
    assert_send::<BackendRawLease>();
    assert_send::<UseLease>();
    // Compile-fail/static assertions additionally require BackendRawLease and
    // UseLease to be !Sync and BackendRawMapping/host guards to be !Send+!Sync.
}

impl OwnedStorage {
    fn as_ref(&self) -> StorageRef<'_>;
    fn as_mut(&mut self) -> StorageMut<'_>;

    fn split_claim(
        self,
        children: &[ByteRange],
    ) -> Result<Vec<Self>, (Self, ClaimSplitError)>;
}

impl<'a> StorageRef<'a> {
    fn resolve(
        self,
        descriptor: &ValidatedDescriptor,
    ) -> Result<ResolvedRead<'a>, AccessError>;
}

impl<'a> ResolvedRead<'a> {
    fn acquire_host_read(&self) -> Result<HostReadGuard<'_>, AccessError> {
        let owner = self.capability.owner;
        let (pin, claim) = (&owner.pin, &owner.claim);
        let state = &*pin.state;
        let raw = state.dispatch_host_read(pin, claim, &self.span)?;
        Ok(HostReadGuard {
            mapping: BackendReadMapping {
                raw,
                _borrow: PhantomData,
            },
        })
    }

    fn acquire_device_read(&self, endpoint: AccessEndpoint)
        -> Result<UseLease, AccessError> {
        let owner = self.capability.owner;
        let (pin, claim) = (&owner.pin, &owner.claim);
        let state = &*pin.state;
        let provider = state.dispatch_device_read(pin, claim, &self.span, endpoint)?;
        Ok(UseLease {
            span: self.span.clone(),
            mode: AccessMode::Read,
            provider,
        })
    }
}

impl<'a> StorageMut<'a> {
    fn resolve_write(
        self,
        descriptor: &ValidatedWriteDescriptor,
    ) -> Result<ResolvedWrite<'a>, (Self, AccessError)>;
}

impl<'a> ResolvedWrite<'a> {
    fn acquire_host_write(&mut self) -> Result<HostWriteGuard<'_>, AccessError> {
        let owner = &mut *self.capability.owner;
        let (pin, claim) = (&owner.pin, &mut owner.claim);
        let state = &*pin.state;
        let raw = state.dispatch_host_write(pin, claim, &self.span)?;
        Ok(HostWriteGuard {
            mapping: BackendWriteMapping {
                raw,
                _borrow: PhantomData,
            },
        })
    }

    fn acquire_device_write(self, endpoint: AccessEndpoint)
        -> Result<WriteBinding<'a>, (Self, AccessError)> {
        // A failed private dispatch or provider admission returns this exact
        // ResolvedWrite, preserving the exclusive capability for recovery.
        let this = self;
        let admission = {
            let owner = &mut *this.capability.owner;
            let (pin, claim) = (&owner.pin, &mut owner.claim);
            let state = &*pin.state;
            state.dispatch_device_write(pin, claim, &this.span, endpoint)
        };
        // The inner borrow ended before this match. Validation or provider
        // pre-admission failure therefore returns this exact ResolvedWrite.
        match admission {
            Ok(provider) => {
                let lease = UseLease {
                    span: this.span.clone(),
                    mode: AccessMode::Write,
                    provider,
                };
                Ok(WriteBinding {
                    resolved: this,
                    lease,
                })
            }
            Err(error) => Err((this, error)),
        }
    }
}

pub struct ImportRejected {
    allocation: Box<dyn BackendAllocationAccess>,
    error: ImportError,
}

// This safe importer validates provider metadata before publishing a claim.
// On rejection it returns the same one allocation box, so the provider drops
// it exactly once; on success that box moves into RootResource and is
// pinned by RootResourcePin. The unsafe provider implementation is the only
// authority proof boundary; there is no second proof token or infallible
// unsafe import function.
fn import_owned_storage(
    allocation: Box<dyn BackendAllocationAccess>,
) -> Result<OwnedStorage, ImportRejected>;

impl ImportRejected {
    fn into_parts(self) -> (Box<dyn BackendAllocationAccess>, ImportError);
}

struct WriteBinding<'a> {
    resolved: ResolvedWrite<'a>,
    lease: UseLease,
}

```

Contract points:

- There is no public `timeline()`, `TimelineState`, `map_read`, or
  `map_write`. The provider-internal access state machine stays behind the
  owner-scoped acquisition methods (#1555, "Host-visible memory and device
  timelines").
- `AllocationSpan` is metadata only. It may be copied for diagnostics or
  validation, but it cannot be passed to a provider access method and cannot
  authorize a read, write, map, enqueue, or lease.
- Provider implementations receive only the validated `BackendAccessRange`
  metadata accessor on an opaque request. The range is enough to calculate a
  provider-local pointer or mapping length, but it carries no
  `RootResourceIdentity`, claim provenance, or access authority and cannot be
  constructed into a request by provider code. This keeps the public unsafe
  extension implementable without exposing `RootBoundSpan`.
- `RootBoundSpan` is private and carries the exact `RootResourceIdentity`.
  `ResolvedRead` and `ResolvedWrite` are sealed values constructed only by
  consuming the matching `StorageRef` or `StorageMut`. Each resolved value
  directly owns that capability and its exact root-bound span; there is no
  per-access operation allocation or provider-specific enum. Equal-looking
  offsets from another root are not interchangeable.
- Acquisition methods live only on `ResolvedRead`/`ResolvedWrite`; they do not
  accept a separately supplied span, provider, dispatch object, or resolved
  capability. The method reaches only the vtable stored in
  `self.capability.owner.pin.state`, and that state constructs an opaque
  request from this exact owner claim, pin, and span. The private dispatcher
  rechecks dynamic root equality before dispatch. Thus no public or
  crate-facing safe API has an independently sourced receiver-plus-resolved
  pair to mismatch. Provider crates implement only the narrow unsafe
  `BackendAllocationAccess` extension contract; the storage kernel does not
  enumerate providers and does not ask sibling crates to implement private
  per-access traits.
- `HostReadGuard`/`HostWriteGuard` expose only the validated byte span as
  immutable/mutable bytes and checked typed slices. Guards borrow the
  allocation (`'a`), so the borrow checker excludes moves (consuming
  submission) and exclusive operations while a guard is alive.
- `UseLease` is `'static`, provider-private, span- and access-mode-scoped. It
  holds a root pin inside its raw carrier, not a Rust borrow, so it can move
  into runtime retirement records. `UseLease` and `BackendRawLease` are
  `Send + !Sync`: one worker/reaper may own them, but concurrent shared use is
  forbidden. This `Send` guarantee is implemented once by the kernel under the
  thread-transfer clause of `BackendAllocationAccess`; it is not an authority
  token or a second unsafe provider proof. `BackendRawMapping` and both host
  guards are `!Send + !Sync` and remain borrow-bound even for a backend whose
  current mapping happens to be transferable. All markers are zero-sized and
  root-pin cloning is a refcount operation, so resolve/acquire performs no heap
  allocation. A lease is non-cloneable and non-forgeable, has no conversion to
  `StorageMut`, and cannot authorize a raw write by itself.
- Provider release is exactly once on every explicit-retirement or ordinary
  drop path: the kernel removes the callback/context from its private `Option`
  before invocation. A provider panic is caught, never retried, and changes the
  pinned root to `Quarantined`; it cannot unwind through a guard, runtime
  worker, or destructor. `mem::forget` keeps the root pin forever and degrades
  liveness only. Consequently safety and reclamation never rely on `Drop`
  running to completion. The kernel quarantine/report transition itself is
  infallible and non-panicking. Foreign exceptions must not cross the callback;
  a Rust provider that may panic uses the declared `C-unwind` ABI and is
  contained by the kernel.
- Write resolution and acquisition require the exclusive capability (`&mut`);
  read resolution requires shared. Device write acquisition consumes the
  `ResolvedWrite` into a `WriteBinding`, retaining the exclusive borrow or the
  consumed owner package through enqueue and retirement. Its failure type is
  `(ResolvedWrite<'a>, AccessError)`, so pre-admission failure returns the
  exact exclusive capability. If a direct API would return the owner and end
  the `&mut` borrow earlier, it must synchronously retire the device work
  before returning it.
- Physical-resource lifetime and span authority are separate. An
  `OwnedSpanClaim` is the unique, non-cloneable authority for its byte span.
  A `RootResourcePin` may be shared internally to keep the provider root
  resource and its deallocator alive, but it authorizes neither reads nor
  writes and cannot create a claim. `OwnedStorage` combines exactly one
  claim with such a pin; all safe access starts from that claim through the
  borrow-taking methods above. Raw write bindings retain the originating
  `StorageMut` borrow (or consume the owning package) through enqueue.
- The claim and pin carry the same private, non-forgeable `RootResourceId`.
  `OwnedStorage` construction checks that relation. `split_claim` consumes
  the parent provenance token before creating children; failure returns the
  unchanged parent. Provider import uses the safe, fallible
  `import_owned_storage` path; it reads metadata from that same allocation,
  and rejection returns the unconsumed provider allocation with a typed
  `ImportError`. The unsafe `BackendAllocationAccess` implementation is the
  sole authority proof boundary. Raw carrier constructors are subordinate
  FFI plumbing inside that boundary and are not a second claim/import proof;
  no redundant uniqueness proof token exists.
- Allocation-resource pins do not float unaccounted: every live pin is held
  by an owner claim, an acquired lease/binding, or a retirement/quarantine
  record. Provider endpoint/context handles that are cached independently do
  not own this allocation's deallocator. This makes "last claim and lease"
  an auditable deallocation condition rather than an incidental strong-count
  observation.

### Span rules

- `byte_offset + byte_len` uses checked arithmetic and must fit the provider
  allocation. `guaranteed_alignment` is a power of two describing the start
  of this span, not merely the base allocation.
- `AllocationKey` equality is domain-qualified (I3, #1558); provider kind or
  device ordinal alone never identifies an allocation.
- Suballocations of one provider resource share `key` and differ by byte
  range. Conflict, hazard, and disjointness reasoning always operates on
  `(key, byte range, access mode)` triples, never on object identity.
- Two owners whose spans overlap for the same key must not exist. Group
  construction and provider constructors reject overlapping owner claims.
  Distinct non-overlapping suballocations sharing a key are valid.
- A safe provider constructor creates a claim only for a freshly allocated
  root resource. Further claims for that resource arise only by consuming a
  parent claim and splitting it into proven-disjoint children. Provider
  import or allocator code that cannot establish this provenance statically
  is one audited `unsafe` boundary whose safety contract requires global
  non-overlap for the imported `(key, byte range)`. Cloning a resource pin is
  never a claim-creation mechanism.
- The root provider resource is deallocated exactly once, after the last
  span claim has been released and every lease covering that resource has
  retired. Releasing one child claim never deallocates a root still covered
  by sibling claims. The shared pin may hold the deallocator internally, but
  its reference count is lifetime bookkeeping only, not evidence of access
  uniqueness.
- Zero-length spans: canonically valid when `byte_len == 0` and the offset
  passes checked arithmetic. Guards over empty spans return empty slices.
  Empty access acquires no provider resources and imposes no ordering. No
  code path may dereference a pointer to justify an empty span.

### Hot-path allocation contract

`StorageRef::resolve`, `StorageMut::resolve_write`, and all G1 acquisition
methods are allocation-free in the storage kernel. `ResolvedRead` and
`ResolvedWrite` are fixed-layout values containing the capability, the exact
`RootBoundSpan`, and the seal; they do not contain a provider enum, a
per-access `Box`, or any other heap-backed erased operation. Provider
dispatch uses the one vtable retained in `RootResource`. A provider may
allocate an event or queue object under its own documented backend contract,
but resolution and the core binding path must not allocate.

The Phase 1 acceptance harness records allocator events around a warmed
`resolve -> acquire_host_*` and `resolve -> acquire_device_*` loop. The core
counter must remain zero for both read and write paths (with provider-owned
event allocation measured separately and explicitly reported). A benchmark
receipt records the loop count, allocator counter, resolved-value size, and
backend; a regression that introduces a per-access allocation fails the G1
performance gate.

Resolution is a traversal or launch boundary, never an element boundary.
For one prepared host traversal or backend launch, allocation-key/span
validation, provider dispatch/downcast, host mapping, guard acquisition, and
`UseLease` acquisition each occur a constant number of times independent of
the element count. The resulting loop or kernel receives a monomorphized typed
slice/pointer plus a prevalidated iteration plan. No element iteration may
perform virtual dispatch, `Any` downcast, heap allocation, reference-count
operation, lock acquisition, synchronization, or descriptor-range
revalidation. Contiguous host traversal has a slice-equivalent inner loop;
strided traversal pays only its prepared stride arithmetic and ordinary loop
control. No path in this contract transfers or materializes storage.

Phase 4 proves the constant-count boundary with an instrumented fake provider
(`p4-traversal-resolution-counts`). Phase 10 adds a source-contract proof
(`p10-element-hot-path-structure`) and verifies release traversal performance
against both a direct-slice control and the immutable Phase 1 pre-redesign
report (`p10-storage-traversal-performance`). Timing alone is not a sound CI
proof; the deterministic counters and structural checks are mandatory even
when a machine-dependent benchmark comparison is reported.

The P1 element-access baseline is active after one clean pre-redesign source
measurement. The measured source commit is
`da7b36e699f9f4731dec08de6a4e1ca93f20cd6f`; the benchmark source path is
`crates/tenferro-tensor/benches/element_access.rs`; and the tracked report is
`docs/testing/storage-element-access-baseline.json`. The capture utility was
run with:

```text
python3 scripts/capture-storage-element-access-baseline.py \
  --root . --output docs/testing/storage-element-access-baseline.json
```

Its exact Criterion command was:

```text
cargo bench --locked -p tenferro-tensor --bench element_access -- \
  --warm-up-time 2 --measurement-time 5 --sample-size 100 --noplot
```

The command uses Cargo's optimized `bench` profile, records WallTime values
as nanoseconds, and sets `MKL_NUM_THREADS`, `OMP_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, `RAYON_NUM_THREADS`, and `VECLIB_MAXIMUM_THREADS` to
`1`. It records the actual Cargo/rustc/toolchain, CPU/OS/affinity, and actual
`RUSTFLAGS`/`CARGO_ENCODED_RUSTFLAGS` values (both were empty here). The
report retains explicit warm-up, measurement, sample, and unit fields without
duplicating Criterion arguments, version, provider, or a derivable thread
count. Mutable cases aggregate touched values, and the strided case is a full
logical-order traversal of a rectangular transpose. The required cases include
fixed-rank 3D access, dynamic immutable iteration, and dynamic mutable
iteration. The active canonical command is deliberately a read-only verifier:

```text
python3 scripts/verify-storage-element-access-baseline.py \
  --report docs/testing/storage-element-access-baseline.json
```

It never benchmarks or rewrites the report. On every later candidate it checks
the tracked report at its exact repository-relative path and uses the recorded
measurement commit and source paths as provenance. The exact Git commit plus
path identifies tracked bytes; no content checksum or saved baseline receipt
is required. P10 consumes the baseline report and its commit/path provenance
directly. A benchmark added after the redesign or an unmeasured `--no-run`
build cannot replace the measured artifact.

P10 may compare a candidate traversal with this baseline only in a compatible
environment: the relevant CPU architecture/model and affinity, OS/kernel
class, rustc/Cargo/toolchain and compilation target, optimized profile, thread
environment, and provider/placement configuration where applicable must match
or be explicitly justified as equivalent. On an incompatible environment the
report remains useful provenance but the comparison is inconclusive; no
machine-independent threshold is inferred and no threshold is transferred
between environments.

### Ordering rules

Conceptually each allocation tracks, per span, the last unretired device
write and the set of outstanding uses. The normative ordering behavior:

1. `acquire_host_read(s)` waits until all device writes overlapping `s`
   retire. Providers whose mapping model forbids concurrent host and device
   reads (current WebGPU/CubeCL) also wait for overlapping device reads;
   this is a provider capability, not a contract change.
2. `acquire_host_write(s)` waits until all outstanding device uses
   overlapping `s` retire. New device use is excluded for the guard lifetime
   by the exclusive borrow.
3. `acquire_device_read(e, s)` validates that endpoint `e` may access the
   allocation, then either waits for or records an event dependency on the
   last overlapping write before first device read.
4. `acquire_device_write(e, s)` orders against all outstanding overlapping
   uses (read-after-write, write-after-read, write-after-write) through
   event dependencies on the device timeline where possible, host waits
   otherwise.
5. Every wait above is a documented synchronization point. None of them may
   copy, transfer, materialize, or fall back to another provider (I4).

### Revalidation at map and enqueue boundaries

At every guard acquisition (map) and every binding encode (enqueue), the
implementation receives a `ResolvedRead` or `ResolvedWrite` and revalidates
the descriptor against that value's own claim/pin as defense in depth (I7):

1. use the already root-bound span carried by the resolved value;
2. checked containment: descriptor byte range inside that exact span;
3. alignment: descriptor start satisfies the dtype and provider requirement
   given `guaranteed_alignment`;
4. access mode: write requires the `ResolvedWrite` path;
5. for writes, layout injectivity has been proven (G2).

There is no second receiver or free span to compare. A test-only corruption
hook may alter a private descriptor after resolution, but it cannot replace
the resolved root, claim, pin, or the pin-state access vtable. Tests must assert
that no safe signature contains an independently supplied provider/dispatch
receiver together with a resolved capability or span.

Failure is a structured error naming the operation, requested range, and
resolved span key. Revalidation failure is always an error, never UB, even
if an internal invariant was violated upstream.

### State table

| Transition | cap | borrow | sync | fail | panic/drop | reclaim |
|---|---|---|---|---|---|---|
| allocate fresh root and claim | provider allocator returns owning claim | none | provider allocation rules | no claim and provider cleans up unpublished resource | no partially published claim | root is live under its first claim |
| reject overlapping/imported claim | none until audited construction succeeds | none | none | structured overlap/provenance error; existing claims unchanged | no claim published | unchanged |
| `StorageRef::resolve` | shared | consumes the `StorageRef` wrapper; owner remains immutably borrowed for the resolved value | descriptor range/alignment validation only | unchanged `StorageRef` plus typed error | no resolved value is published | owner remains live |
| `StorageMut::resolve_write` | exclusive | consumes the `StorageMut` wrapper and retains its `&mut OwnedStorage` in `ResolvedWrite` | checked range, alignment, and write-injectivity validation | `(StorageMut, AccessError)` with the exact capability returned | no partial resolved capability is published | owner remains live |
| `acquire_host_read` | shared | allocation for guard lifetime | wait: overlapping device writes (plus reads where provider requires) | no guard, no state change | guard drop unregisters host use | not while guard alive |
| `acquire_host_write` | exclusive | allocation, exclusively, for guard lifetime | wait: all overlapping device uses | no guard, no state change | drop unregisters; writes made so far are visible bytes, no rollback | not while guard alive |
| `acquire_device_read` | shared | none beyond the call (lease is a `Send + !Sync` pin/carrier) | event dependency on last overlapping write | no lease, no state change | lease drop before submission invokes release once; callback panic is contained and quarantines root | not while lease outstanding |
| `acquire_device_write` | exclusive | consumes `ResolvedWrite`; `WriteBinding` retains the owner/`&mut` through enqueue and retirement | event dependencies for RAW/WAR/WAW | unchanged binding capability plus typed error | admitted binding moves to retirement even if its handle is dropped | after covering events retire |
| direct write API returning an owner | owning | no early end of exclusive access is allowed | synchronous retirement before returning the owner | owner returned only after retirement; otherwise typed error retains it | panic retains/quarantines until retirement | after synchronous retirement |
| lease submitted with work | owning (runtime owns inputs) | none (pins) | none at submit; retirement via events | enqueue prep failure releases only unsubmitted leases | admitted leases survive handle drop and panic until retirement | after all covering events complete |
| guard leaked (`mem::forget`) | n/a | borrow ends without `Drop` | none | n/a | provider host-use registration may persist until owner drop; soundness is preserved (access is gone), liveness may degrade; this is documented, not UB | owner drop path below |
| split owner claim | owning (consumes parent claim) | none | none | original owner returned unchanged | no child is observable until all disjoint claims are built | parent is replaced by children; root resource remains pinned |
| drop one of several sibling claims | owning | none | covering leases for that claim follow the next row | n/a | only that claim is released or retired | root remains live under sibling claims/pins |
| owner drop, no outstanding use | owning | none | none | n/a | releases exactly that span claim | root deallocated exactly once only if this was the last claim and no lease remains |
| owner drop, outstanding leases | owning | none | none | n/a | claim, deallocator pin, and leases move into a retirement record | claim releases after its events; root deallocates exactly once after the last claim and lease |
| last root pin/claim release | owning/provider-internal | none | all covering events already retired | n/a | exactly-once deallocator runs or the resource remains quarantined | now, and only now |
| retirement wait fails | n/a | none | attempted wait/poll | error reported on the runtime/provider error channel | resources quarantined: retained and reported | never speculatively; only if a later drain proves completion |
| provider release callback panics | carrier has already consumed its one callback token | unchanged | no retry | structured provider-release failure | panic is caught; pinned root enters `Quarantined`; outer Drop/worker continues | never from the failed release proof |

Persistent owner-claim splitting above is distinct from G2 `split_mut`.
Claim splitting consumes one owner and changes the persistent ownership set;
`split_mut` only derives temporary disjoint Rust borrows from an unchanged
owner/group and cannot create a claim or affect root-resource lifetime.

Quarantine is root-resource state, not merely a runtime log entry. Marking a
root quarantined is atomic and visible to every claim sharing its private
`RootResourceId`. All safe acquisition and extraction paths revalidate this
state and fail before exposing bytes or raw bindings. A quarantine record
retains the root pin/deallocator and provider context even after every public
claim is dropped; only a later provider-specific proof of retirement may
release it.

## G2. AllocationGroup

The group is the only sound representation for one owner with many logical
values (#1555, "Disjoint views and allocation groups"; #1561).

### Types

```rust
pub struct AllocationGroup {
    allocations: Vec<Option<OwnedStorage>>, // private, stable move-out slots
    descriptors: Vec<Option<DescriptorRecord>>, // private, append-only slots
}

#[derive(Clone, Copy)]
pub struct DescriptorSlot(u32); // opaque; meaningful only under its group borrow

struct DescriptorRecord {
    allocation: AllocationSlot, // index into `allocations`
    dtype: DType,
    layout: ValidatedLayoutMetadata,
    byte_range: ValidatedRootBoundRange,
    placement: Placement,
    storage: ValidatedStorageMetadata,
    provider: ValidatedProviderMetadata,
    write_injectivity: Option<WriteInjectivityProof>,
}
```

Construction preconditions (safe constructors):

- every occupied allocation entry contains one `OwnedStorage`, and each owner
  appears in exactly one entry. Duplicate owner tokens are impossible by move
  semantics, and overlapping owner spans for one key are rejected;
- allocation entries have stable indices. Moving an owner out leaves `None`
  and never renumbers another `AllocationSlot`;
- every descriptor references an occupied allocation entry. Before publishing
  the record, safe construction validates its dtype, checked layout and byte
  range, alignment, placement, storage, and provider compatibility against
  that entry's exact root-bound span, then retains the validated
  layout/range/storage/provider metadata in `DescriptorRecord`. This metadata
  is non-owning and non-authoritative;
- descriptors may alias freely, including exact duplicates;
- descriptor slots are append-only. Insertion always appends a new table
  entry; removing a descriptor leaves its entry vacant for the rest of the
  group's lifetime, and that slot is never rebound or reused;
- `DescriptorSlot` is only a local lookup key. It carries no allocation,
  root, provider, or write authority, and it is meaningful only when resolved
  through a borrow of the group that owns the table. Copying a slot copies
  only this metadata and grants no capability;
- physical lifetime comes from each occupied entry's `OwnedStorage` claim and
  its `Arc<RootResource>` root (G1). Mutation and extraction require the
  exclusive group borrow; `split_mut` derives only temporary disjoint mutable
  capabilities from that borrow;
- descriptor records are ordinary group-owned metadata. No out-of-band
  descriptor liveness roots or cross-group identity participate in access or
  reclamation.

### Operation contracts

```rust
fn view(&self, slot: DescriptorSlot) -> Result<TensorView<'_>, GroupError>;
fn view_mut(&mut self, slot: DescriptorSlot)
    -> Result<TensorViewMut<'_>, GroupError>;
fn split_mut(&mut self, slots: &[DescriptorSlot])
    -> Result<Vec<TensorViewMut<'_>>, DisjointViewError>;
fn try_extract(&mut self, slot: DescriptorSlot) -> Result<Tensor, ExtractError>;
fn into_tensor(self, slot: DescriptorSlot) -> Result<Tensor, (Self, ExtractError)>;
```

- Every operation resolves its `DescriptorSlot` through the borrowed group:
  a shared receiver yields a borrowed descriptor record and read view, while
  an exclusive receiver yields the write or extraction path. A slot alone
  cannot expose storage or a provider binding. Slot resolution checks only
  the local table position and occupancy; it does not repeat the descriptor's
  construction-time invariant validation.
- An access path combines the retained metadata with its Rust borrow and any
  operation-specific proof, then constructs the G1 prepared-access object
  once. Provider map and enqueue consume that prepared object and do not
  repeat bounds, layout, range, storage, or provider validation.
- `view` borrows the group shared; any number of aliasing read views may
  coexist. The returned view is bounded by that borrow.
- `view_mut` exclusively borrows the whole group; one mutable view exists at
  a time. The Rust borrow is the write authority; the slot is only the record
  selected by that borrow.
- `split_mut` resolves all requested slots under one exclusive group borrow,
  reads their retained validated metadata, and returns N simultaneous mutable
  views only after the central disjointness proof (below). It performs no
  layout, range, storage, provider, map, or enqueue revalidation. Its only
  additional proofs are write injectivity when a record does not already
  retain that proof, and pairwise disjointness for the requested mutable
  views. Children are non-cloneable and hold the exclusive borrow of the
  group; the root is inaccessible while any child lives.
- `try_extract` resolves `slot` under `&mut AllocationGroup`. It succeeds
  only when the selected record is the sole descriptor in this group that
  refers to its `AllocationSlot`; the record is removed and its owned
  allocation is moved out by replacing the occupied allocation entry with
  `None`, without renumbering any other entry. If another local descriptor
  aliases that allocation, the operation returns a typed reason and leaves
  the group unchanged. The removed descriptor entry remains vacant for the
  rest of the group's lifetime, and no copy or materialization fallback is
  permitted (I4).
- `into_tensor` consumes the group, resolves one local slot, and explicitly
  discards all other descriptor records. It never duplicates ownership to
  preserve them. On failure it returns the unchanged group.
- Persistent AD handle behavior is outside G2 and is specified by G7; these
  operations do not consult AD handle bookkeeping.

### Central disjointness proof

One audited module owns the proof. Normative order (#1561):

1. resolve each requested occupied descriptor slot under the exclusive group
   borrow;
2. read its retained validated layout, root-bound byte range, storage, and
   provider metadata without recomputing those facts;
3. use the retained write-injectivity proof, or compute that proof once when
   the record does not already contain it;
4. partition requests by their retained allocation slot and root span;
5. treat empty descriptors as non-overlapping;
6. prove pairwise disjointness of the retained reachable byte envelopes;
7. derive non-cloneable disjoint mutable child capabilities whose lifetimes
   remain bounded by the exclusive group borrow.

After slot resolution, `split_mut` performs only the write-injectivity proof
for records that do not retain one and requested-view pairwise disjointness.
It neither maps nor enqueues storage. A later map or enqueue consumes prepared
access and does not repeat construction-time validation.

Conservative rejection is required rather than element enumeration:
interleaved strided requests whose byte envelopes overlap return
`NotProvablyDisjoint`. Error variants are invalid or empty descriptor slot,
non-injective write layout when no retained proof exists, pairwise overlap,
and not provably disjoint. Construction-time layout, range, storage, and
provider errors cannot originate from `split_mut`. Every error leaves the
group unchanged.

### State table

| Transition | cap | borrow | sync | fail | panic/drop | reclaim |
|---|---|---|---|---|---|---|
| construct group | owning (consumes owners) | none | none | typed invariant-validation error returns owners or drops them exactly once, per constructor contract | no partially observable record; validated layout/range/storage/provider metadata is published only after all checks pass | owners' G1 rules |
| `view` | shared | group borrow resolves a local slot and returns a borrowed view | none (host access goes through G1 guards) | invalid/empty slot error, group unchanged | view drop ends the borrow | n/a |
| `view_mut` | exclusive | exclusive group borrow resolves the record and supplies write authority | none | invalid/empty slot error, group unchanged | borrow end; bytes written stay written | n/a |
| `split_mut` | exclusive | requested slots consume retained metadata under one group borrow; children receive temporary disjoint mutable borrows/capabilities, never persistent claims | no map/enqueue; no repeated validation | invalid/empty slot, non-injective write layout when its proof was absent, pairwise overlap, or not provably disjoint; group unchanged | children drop ends borrow; no partial child set is observable on panic (proof precedes construction) | n/a |
| `try_extract` | exclusive | direct borrowed slot; local descriptor count proves allocation uniqueness | none | invalid/empty or aliased-allocation reason, group unchanged | descriptor and allocation entries become vacant without renumbering; the descriptor slot is never reused; no borrowed view can coexist with the exclusive borrow | extracted owner follows G1 |
| `into_tensor` | owning | consuming group resolves one local slot and discards other records | none | group returned unchanged with reason | unselected records and claims follow G1 drop rules | selected owner follows G1 |

## G3. Submission

G3 has two submission surfaces. Detached owning execution remains
asynchronous. Scoped read-only execution is synchronous to retirement and
accepts only immutable tensor-view borrows.

### Detached submission

```rust
pub struct ExecutionInputs {
    group: AllocationGroup,
    bindings: Box<[DescriptorSlot]>,
}

pub fn submit(
    &self,
    program: &CompiledGraph,
    inputs: ExecutionInputs,
) -> Result<ExecutionHandle, SubmitRejected>;

impl ExecutionHandle {
    pub fn wait(self) -> ExecutionOutcome;
}

pub struct SubmitRejected {
    cause: SubmitError,
    inputs: ExecutionInputs,
}

impl SubmitRejected {
    pub fn into_parts(self) -> (SubmitError, ExecutionInputs);
}

pub enum ExecutionOutcome {
    Completed(ExecutionBundle),
    RetiredFailed {
        cause: ExecutionError,
        inputs: ExecutionInputs,
    },
    CompletionUnproven {
        cause: CompletionError,
        diagnostic_keys: Box<[DiagnosticKey]>,
    },
}

pub struct ExecutionBundle {
    group: AllocationGroup,
    outputs: Box<[ExecutionOutput]>,
}

pub enum ExecutionOutput {
    Tensor(DescriptorSlot),
    Metadata(OutputMetadata),
}

pub enum OutputRef<'a> {
    Tensor(TensorView<'a>),
    Metadata(&'a OutputMetadata),
}

pub enum OutputExtractError {
    InvalidOutput,
    MetadataOutput,
    Extract(ExtractError),
}

impl ExecutionBundle {
    pub fn output(&self, output: usize)
        -> Result<OutputRef<'_>, OutputAccessError>;
    pub fn into_output(self, output: usize)
        -> Result<Tensor, (Self, OutputExtractError)>;
}
```

`SubmitRejected` returns the exact owning `ExecutionInputs` that were not
admitted. After admission, `Completed` and `RetiredFailed` expose their
resources only after provider retirement. `ExecutionBundle::output` returns a
borrowed tensor view or metadata reference. A tensor output slot is resolved
in the returned group and may be an existing identity or repeated slot, or a
slot newly inserted for a fresh allocation; neither case copies storage.
`into_output` consumes the entire bundle and delegates the selected tensor slot
to G2 `into_tensor`. On success, repeated or duplicate output aliases, the
remaining group, and the output map disappear together; no extracted-state
flags remain. On rejection it returns the exact bundle and typed error.
`Metadata` is genuinely storage-free.

`CompletionUnproven` exposes only its typed cause and diagnostic keys; it never
returns an owner or other owning resource. The provider-private permanent
record retains the consumed `Arc` roots for that outcome. No public result can
recover those roots.

### Scoped read-only execution

```rust
pub struct ScopedReadInputs<'env> {
    bindings: Box<[ScopedReadBinding<'env>]>,
}

pub struct ScopedReadBinding<'env> {
    tensor: TensorView<'env>,
}

pub fn execute_scoped_read_only<'env>(
    &self,
    program: &CompiledGraph,
    inputs: ScopedReadInputs<'env>,
) -> Result<ScopedExecutionOutcome<'env>, ScopedSubmitRejected<'env>>;

pub enum ScopedExecutionOutcome<'env> {
    Completed(ScopedExecutionBundle<'env>),
    RetiredFailed {
        cause: ExecutionError,
        inputs: ScopedReadInputs<'env>,
    },
}

pub struct ScopedSubmitRejected<'env> {
    cause: SubmitError,
    inputs: ScopedReadInputs<'env>,
}

impl<'env> ScopedSubmitRejected<'env> {
    pub fn into_parts(self) -> (SubmitError, ScopedReadInputs<'env>);
}

pub struct ScopedExecutionBundle<'env> {
    owned: AllocationGroup,
    outputs: Box<[ScopedOutput<'env>]>,
}

pub enum ScopedOutput<'env> {
    Borrowed(TensorView<'env>),
    Owned(DescriptorSlot),
    Metadata(OutputMetadata),
}

pub enum ScopedOutputExtractError {
    BorrowedOutput,
    Output(OutputExtractError),
}

impl<'env> ScopedExecutionBundle<'env> {
    pub fn output(&self, output: usize)
        -> Result<OutputRef<'_>, OutputAccessError>;
    pub fn into_owned_output(self, output: usize)
        -> Result<Tensor, (Self, ScopedOutputExtractError)>;
}
```

`ScopedReadInputs<'env>` contains only immutable `TensorView<'env>` bindings;
there is no writable binding shape. Rejection returns the exact borrowed
package that was not admitted. Once admitted, `execute_scoped_read_only` is
synchronous to retirement and returns only `Completed` or `RetiredFailed`
after all provider work has retired.

A completed scoped bundle distinguishes borrowed and owned tensor results.
Identity and repeated outputs are `Borrowed` descriptor views bounded by
`'env`; fresh results are inserted into `owned` and named by group-local
`Owned` slots. `Metadata` is storage-free. None of these paths copies or
materializes input storage, and fresh owned outputs become observable only
after retirement. `output` reborrows either tensor form as an immutable view.
`into_owned_output` consumes the whole bundle and succeeds only for an `Owned`
slot by delegating to G2 `into_tensor`. Success discards repeated or duplicate
owned aliases and the remaining output map together. A `Borrowed` output
returns the exact bundle with `ScopedOutputExtractError::BorrowedOutput`; a
metadata output returns the exact bundle with the typed metadata rejection.
Neither rejection copies, and no extracted-state flags exist.

Scoped read-only execution supports only host/CPU providers whose operation is
synchronous through retirement. CUDA, WebGPU, Metal, and any provider that can
leave asynchronous work live across unwind are rejected before admission or
report the operation unsupported. No borrowed device work is outstanding at
any unwind point. Safety follows from the synchronous-provider contract,
never from panic catching or `Drop`.

### Lifecycle

Detached execution follows `Prepared` -> `Admitted`/`Running` -> `Draining`
-> `Retired(Completed | Failed)` or `CompletionUnproven`. The public terminal
variants are `Completed`, `RetiredFailed`, and `CompletionUnproven`.

`Prepared` covers validation and planning. Rejection returns the exact
unadmitted owners. Admission consumes `ExecutionInputs`; the worker or reaper
owns its inputs and leases until a terminal outcome.

A detached worker or provider panic is contained at the existing worker,
thread, or FFI boundary and enters `Draining`. If completion is proven,
`RetiredFailed` returns the exact input owners with a typed panic cause. If
completion cannot be proven, `CompletionUnproven` returns only its typed cause
and diagnostics while a provider-private permanent record retains the `Arc`
roots. No public recovery path returns those owners.

Dropping a detached handle detaches observation; the reaper owns resources
until the terminal outcome.

Scoped read-only execution is limited to synchronous host/CPU providers.
Rejection returns the exact borrowed inputs; accelerator and asynchronous
providers reject before admission or do not support this call. An admitted
call retires before return, so no borrowed work is outstanding at unwind and
safety does not depend on panic catching or `Drop`.

| Transition | cap | borrow | sync | fail | panic/drop | reclaim |
|---|---|---|---|---|---|---|
| `Prepared` -> submit result | owning (consumes `ExecutionInputs`) | none | validation and planning only | `SubmitRejected` returns the exact unadmitted owners | no admitted work or provider retention | owners return to the caller under G1 |
| `Admitted` -> `Running` | owning (worker) | none | leases acquired before enqueue | post-admission preparation or enqueue failure enters `Draining` | handle drop detaches observation; reaper retains owners and leases | only at a terminal outcome |
| `Running` -> `Draining` | owning (worker/reaper) | none | all enqueued work and event domains drain | execution failure or worker/provider panic enters `Draining` | panic is typed at the existing worker/thread/FFI boundary; reaper retains ownership | not yet |
| `Draining` -> `Retired(Completed)` | owning (worker/reaper) | none | completion proven | returns `ExecutionBundle` | n/a | returned bundle follows G1 |
| `Draining` -> `Retired(Failed)` | owning (worker/reaper) | none | completion proven | returns exact input owners with the typed execution or panic cause | n/a | returned owners follow G1 |
| `Draining` -> `CompletionUnproven` | no public owner; provider-private retention | none | completion cannot be proven | returns no owner, only the typed completion or panic cause and diagnostics | permanent record retains the `Arc` roots | retained by that permanent record |
| scoped call rejected | shared | `ScopedReadInputs<'env>` borrows | none | returns exact borrowed inputs; non-host or asynchronous providers are unsupported | no work is admitted | caller-owned borrows remain valid |
| scoped admitted call | shared | input borrows remain for `'env` | host/CPU work executes and retires synchronously before return | returns only completed or failed after retirement | no work survives to an unwind point; no panic-catch or `Drop` safety | owned outputs follow G1; borrowed outputs remain bounded by `'env` |

## G4. Method distribution

Rules (#1555, "Capability surface and method distribution"; #1559):

1. Read-only tensor behavior is implemented once, on `TypedTensorView`.
2. `TypedTensor` and `TypedTensorViewMut` delegate through cheap
   canonicalization:

   ```rust
   impl<T, R: TensorRank> TypedTensor<T, R> {
       fn as_view(&self) -> TypedTensorView<'_, T, R>;
       fn as_view_mut(&mut self) -> TypedTensorViewMut<'_, T, R>;
   }
   impl<'a, T, R: TensorRank> TypedTensorViewMut<'a, T, R> {
       fn as_view(&self) -> TypedTensorView<'_, T, R>; // reborrow
   }
   ```

3. Mutation behavior is implemented once, on `TypedTensorViewMut`; the owner
   delegates through `as_view_mut()`. Consuming operations (`into_*`,
   extraction, consuming reinterpretation) live on owners and groups only.
4. The dtype-erased family mirrors the triad: `Tensor`, `TensorView<'a>`,
   `TensorViewMut<'a>` with `as_view()`/`as_view_mut()` dispatching to the
   typed implementations. Erased behavior must be observably equivalent to
   typed behavior. There is no public mutable downcast (`as_any_mut`);
   mutable dispatch stays behind the opaque write capability and the
   provider boundary.
5. There is no `Deref`/`DerefMut` between tensor families, no common sized
   header type, and no fake DST. Rationale is recorded in #1555: the sized
   deref target reproduces the ndarray 0.17.0 header-swap unsoundness, and a
   fake DST needs a host address that device storage does not have.
6. Swap-safety: no public API returns `&mut OwnedStorage`,
   `&mut Box<dyn BackendAllocationAccess>`, or any mutable projection of an owner
   container; `StorageMut` is an opaque write capability. Layout mutation is
   available only through operations that revalidate the resulting
   descriptor against the span (G1), so layout and storage cannot be
   decoupled through safe code.
7. Parity is enforced: the canonical read-only method list has one source of
   truth, and an API-parity contract test fails when owner, view, view-mut,
   or the erased family drifts.

Method-home table:

| Method class | Home | Others |
|---|---|---|
| read-only (shape, dtype, layout inspection, host read via G1 guard, formatting) | `TypedTensorView` | owner and view-mut delegate via `as_view()` |
| in-place mutation (write guard acquisition, mutable slicing, `split_mut` entry points) | `TypedTensorViewMut` | owner delegates via `as_view_mut()` |
| consuming (into-conversions, consuming reinterpretation, extraction) | `TypedTensor` / `AllocationGroup` | not available on views |
| duplication (explicit copy) | `TypedTensorView` (reads source) returning a new owner | owner/view-mut delegate |

### Rank preservation and element-access cost model

The optional rank parameter is part of the public type and performance
contract. Rank-preserving operations retain `R` on the owner and both views:

```rust
struct TypedTensor<T, R: TensorRank = DynRank> {
    storage: OwnedStorage,
    layout: TensorLayout<R>,
    placement: Placement,
    _scalar: PhantomData<T>,
}

struct TypedTensorView<'a, T, R: TensorRank = DynRank> {
    storage: StorageRef<'a>,
    layout: TensorLayoutRef<'a, R>,
    placement: Placement,
    _scalar: PhantomData<T>,
}

struct TypedTensorViewMut<'a, T, R: TensorRank = DynRank> {
    storage: StorageMut<'a>,
    layout: TensorLayoutRef<'a, R>,
    placement: Placement,
    _scalar: PhantomData<T>,
}
```

The sketches are normative in shape, while names remain provisional until
their owning implementation phase. `TensorLayoutRef` denotes a borrowed
metadata representation: an ordinary `as_view()` or `as_view_mut()` is O(1),
allocation-free, and does not clone heap-backed dynamic shape/stride metadata,
clone or increment a storage/provider reference count, resolve a provider,
synchronize, transfer, or materialize data. It only reborrows the existing
owner capability and layout. `DynRank` remains a first-class supported rank;
it is not permission to erase fixed rank from unrelated APIs.

`p3-as-view-zero-allocation` is a combined allocation, counter, and structural
contract rather than an allocator-only test. Around warmed owner and mutable
view reborrows for both fixed and dynamic rank, it asserts zero allocator
events, zero storage/provider clone or strong-count operations, and zero
dynamic shape/stride metadata clones. Its source-contract inventory proves
that view storage and dynamic layout metadata are borrow fields and that
`as_view*` contains no ownership clone path. An `Arc::clone` that happens not
to allocate is therefore still a gate failure.

Checked random `get(&[usize])` and `get_mut(&[usize])` may validate bounds and
perform O(rank) offset arithmetic per call. They are not the canonical hot-loop
interface. Contiguous bulk access resolves once and exposes a typed slice or
guard. Strided iteration resolves once and carries a prevalidated incremental
offset/stride plan. Backend execution resolves and leases once per launch.
Static-rank traversal remains monomorphized and eligible for loop unrolling;
dynamic-rank support must not route every typed element through opaque
per-element dispatch. The release codegen artifact
`p10-static-rank-codegen` records at least one contiguous fixed-rank loop and
must show a slice-equivalent inner loop without storage/provider abstraction
work.

Phase 4 must expose a concrete prepared-access boundary equivalent to this
shape:

```rust
enum CheckedLayout<R: TensorRank> {
    Contiguous {
        element_range: core::ops::Range<usize>,
    },
    Strided(CheckedStrided<R>),
}

enum PreparedHostRead<'a, T, R: TensorRank> {
    Contiguous(PreparedContiguousRead<'a, T, R>),
    Strided(PreparedStridedRead<'a, T, R>),
}

enum PreparedHostWrite<'a, T, R: TensorRank> {
    Contiguous(PreparedContiguousWrite<'a, T, R>),
    Strided(PreparedStridedWrite<'a, T, R>),
}

struct PreparedContiguousRead<'a, T, R: TensorRank> {
    guard: HostReadGuard<'a, T>,
    element_range: core::ops::Range<usize>,
    _rank: PhantomData<R>,
}

struct PreparedContiguousWrite<'a, T, R: TensorRank> {
    guard: HostWriteGuard<'a, T>,
    element_range: core::ops::Range<usize>,
    _rank: PhantomData<R>,
}

struct PreparedStridedRead<'a, T, R: TensorRank> {
    guard: HostReadGuard<'a, T>,
    plan: CheckedStrided<R>,
}

struct PreparedStridedWrite<'a, T, R: TensorRank> {
    guard: HostWriteGuard<'a, T>,
    plan: CheckedInjectiveStrided<R>,
}

struct StrideCursor<R: TensorRank> {
    coordinate: RankIndex<R>,
    next_element_offset: usize,
    remaining: usize,
}

struct PreparedStridedIter<'i, T, R: TensorRank> {
    base: core::ptr::NonNull<T>,
    plan: &'i CheckedStrided<R>,
    cursor: StrideCursor<R>,
    _borrow: PhantomData<&'i T>,
}

struct PreparedStridedIterMut<'i, T, R: TensorRank> {
    base: core::ptr::NonNull<T>,
    plan: &'i CheckedInjectiveStrided<R>,
    cursor: StrideCursor<R>,
    _borrow: PhantomData<&'i mut T>,
}

impl<'a, T, R: TensorRank> TypedTensorView<'a, T, R> {
    fn prepare_host(
        self,
    ) -> Result<PreparedHostRead<'a, T, R>, (Self, AccessError)>;
}

impl<'a, T, R: TensorRank> TypedTensorViewMut<'a, T, R> {
    fn prepare_host_mut(
        self,
    ) -> Result<PreparedHostWrite<'a, T, R>, (Self, AccessError)>;
}

impl<'a, T, R: TensorRank> PreparedContiguousRead<'a, T, R> {
    fn as_slice(&self) -> &[T];
    fn iter_contiguous(&self) -> core::slice::Iter<'_, T>;
}

impl<'a, T, R: TensorRank> PreparedContiguousWrite<'a, T, R> {
    fn as_slice_mut(&mut self) -> &mut [T];
    fn iter_contiguous_mut(&mut self) -> core::slice::IterMut<'_, T>;
}

impl<'a, T, R: TensorRank> PreparedStridedRead<'a, T, R> {
    fn iter_strided(&self) -> PreparedStridedIter<'_, T, R>;
}

impl<'a, T, R: TensorRank> PreparedStridedWrite<'a, T, R> {
    fn iter_strided_mut(&mut self) -> PreparedStridedIterMut<'_, T, R>;
}

impl<'i, T, R: TensorRank> Iterator for PreparedStridedIter<'i, T, R> {
    type Item = &'i T;
    fn next(&mut self) -> Option<Self::Item>;
}

impl<'i, T, R: TensorRank> Iterator for PreparedStridedIterMut<'i, T, R> {
    type Item = &'i mut T;
    fn next(&mut self) -> Option<Self::Item>;
}
```

`RankIndex<R>` is the rank-preserving cursor representation: inline for fixed
rank and initialized once outside iteration for dynamic rank.
`CheckedStrided<R>` owns the checked start offset, extents, strides, element
count, and incremental carry plan; it contains no provider or storage receiver.
`CheckedInjectiveStrided<R>` is constructible only after the write-injectivity
proof and otherwise has the same traversal data. The fallible `prepare_host*`
constructor resolves the storage capability, validates checked
shape/stride/offset arithmetic, bounds, span containment, alignment, layout
injectivity for writes, provider compatibility, mapping, and synchronization
before constructing or publishing `CheckedLayout`, `PreparedHostRead`, or
`PreparedHostWrite`. Failure rolls back any partial mapping/registration and
returns the unchanged input capability with a typed `AccessError`; no prepared
object or iterator exists on failure. The constructor consumes the checked
layout into exactly one `PreparedHost*` enum variant. Matching that variant is
the only contiguous/strided state transition and performs no validation or
provider work.

`as_slice*` and `iter_contiguous*` perform only typed slice access after one
range extraction outside the loop. `PreparedStridedIter*::next` performs only
typed pointer/slice access, the necessary incremental stride/carry updates,
and loop termination. It does not decode a flat index into coordinates or
repeat bounds, layout, span, alignment, capability, provider, map, or
synchronization checks. The `PreparedHost*` and `CheckedLayout` enums are the
state authorities; independent booleans such as `is_checked`, `is_contiguous`,
`is_mapped`, and `is_writable` must not encode these states.

`iter_strided*` borrows its prepared guard for `'i`, takes the already checked
base pointer, and initializes `StrideCursor` once; it performs no validation,
mapping, synchronization, or provider operation. Exhaustion has one authority:
`cursor.remaining == 0`; `next()` returns `None` without pointer arithmetic in
that state. Otherwise it dereferences the previously proven in-span offset,
decrements `remaining`, and advances the incremental carry plan. The immutable
iterator's `Item = &'i T` is bounded by the read guard. The mutable iterator's
`Item = &'i mut T` is sound because `CheckedInjectiveStrided` proves that no
two yielded offsets overlap for `size_of::<T>()`; the iterator owns the sole
mutable borrow of its guard for `'i` and never yields an offset twice. The
unsafe pointer dereference is private, adjacent to these checked-plan
invariants, and covered by empty, singleton, negative/reverse-stride,
noncontiguous, overflow-rejection, exhaustion, and Miri tests.

These type names are provisional, but the prepared boundary and contiguous
specialization are normative. An owning phase that chooses different names or
splits the objects differently must update this contract in the same PR and
include an explicit mapping from every sketch type/method/state transition to
its replacement. It must show, through `p4-prepared-access-api`,
`p4-traversal-resolution-counts`, `p10-element-hot-path-structure`, and
`p10-static-rank-codegen`, that its API has the same prevalidation, inner-loop,
and code-generation properties. The P4 artifact combines compile/runtime API
tests with a source-contract inventory proving all validation and provider
work precede construction, iterator bodies contain only the permitted typed
access and increments, and no boolean fields duplicate enum state. P10 repeats
the loop-boundary structural proof over the final normalized API.

Rank-changing reinterpretation is separate from ordinary views. Phase 6 must
define each operation's result-rank policy explicitly and test it under
`p6-reinterpret-rank-policy`. A stable-Rust limitation in expressing a type
level result such as `N + 1` may require a dynamic result or an explicit
caller-selected result rank for that operation only; it must never force
rank-preserving view, slice, or traversal APIs to erase `R`.

The v2 ledger carries these executable obligations:

| Obligation | Phase | Artifact and proof |
|---|---|---|
| `p1-element-access-baseline` | P1 | active measured direct-slice/contiguous/strided report and verifier; later candidates use its exact Git commit and repository-relative path, subject to P10 compatible-environment comparison |
| `p3-static-rank-preservation` | P3 | compile/API contract for owner, immutable view, and mutable view preserving `R` |
| `p3-as-view-zero-allocation` | P3 | warmed allocator/refcount/provider-clone/layout-clone counters plus borrow-only source contract for owner/view-mut reborrows, including dynamic rank |
| `p4-traversal-resolution-counts` | P4 | fake provider counters proving resolve/map/lease/dispatch counts are independent of element count |
| `p4-prepared-access-api` | P4 | compile/runtime and source contract for typed failure, enum-authoritative preparation, contiguous slice/iterator access, and incremental strided iteration |
| `p6-reinterpret-rank-policy` | P6 | behavior and compile contract for every rank-changing reinterpretation |
| `p10-element-hot-path-structure` | P10 | source-contract check that provider/capability resolution is outside element loops |
| `p10-storage-traversal-performance` | P10 | release contiguous and representative strided report that explicitly verifies and consumes the P1 result JSON plus its measured commit/path provenance |
| `p10-static-rank-codegen` | P10 | release codegen/assembly report for a contiguous static-rank typed loop |
| `p12-element-access-guide` | P12 | executable content check for the guide and required rustdoc cost claims |
| `p12-element-access-examples` | P12 | release runnable owner/view/view-mut traversal tutorial |

## G5. Raw handles and reclamation

- Raw provider handles and pointers are lifetime-only. They may keep a
  provider resource usable for a binding or retirement record, but they carry
  no owner claim, `StorageRef`, `StorageMut`, or write authority. Cloning a
  provider handle or retaining an `Arc` does not change that contract.

### Prepared binding

`prepare_read` and `prepare_write` are the single access-boundary validation
described by G1. They validate bounds, layout, dtype, the exact root span,
alignment, storage, and provider compatibility, plus write injectivity for a
write. The result is sealed provider-ready access carrying the checked layout
and the Rust capability needed by the operation.

```rust
fn prepare_read<'a>(
    access: StorageRef<'a>,
    request: ReadRequest,
    provider: &Provider,
) -> Result<PreparedRead<'a>, (StorageRef<'a>, AccessError)>;

fn prepare_write<'a>(
    access: StorageMut<'a>,
    request: WriteRequest,
    provider: &Provider,
) -> Result<PreparedWrite<'a>, (StorageMut<'a>, AccessError)>;

fn bind_read<'a>(
    prepared: PreparedRead<'a>,
) -> Result<DeviceRead<'a>, (PreparedRead<'a>, BindError)>;

fn bind_write<'a>(
    prepared: PreparedWrite<'a>,
) -> Result<DeviceWrite<'a>, (PreparedWrite<'a>, BindError)>;
```

Binding consumes provider-ready prepared access. Neither binding nor enqueue
repeats these checks or compares a second request, key, or range; those values
are carried by the prepared object, and the binding/enqueue signatures accept
no replacement values. `PreparedRead`, `PreparedWrite`, `DeviceRead`, and
`DeviceWrite` are distinct sealed states, not boolean state combinations.

There is no shared-to-exclusive conversion. A provider handle, `Arc`, lease,
event, raw pointer, or refcount cannot become `StorageMut`, an owner claim, or
a write binding. `prepare_write` starts with an exclusive owner borrow or a
newly allocated output, and that Rust capability remains the source of write
authority.

Safe APIs never return an escaping raw pointer. Provider-specific unsafe
interop may expose a pointer only with documentation that the caller keeps the
binding alive for the required lifetime, obeys the provider's synchronization
rules, and does not use the pointer after retirement. There is no safe generic
`device_ptr` accessor.

### Detached and borrowed submission

Safe asynchronous device submission is detached and owning only. The task
consumes the G3 `ExecutionInputs`, owns its `AllocationGroup` and the
`Arc<RootResource>` roots held by that group, and derives the prepared read and
write bindings inside the task before enqueue. A safe detached call does not
accept a caller-borrowed prepared binding. A write binding is derived only from
an exclusive owner/group borrow or a newly allocated output.

```rust
fn submit_detached(
    inputs: ExecutionInputs,
) -> Result<ExecutionHandle, SubmitRejected>;

fn submit_borrowed<'a>(
    bindings: BorrowedBindings<'a>,
) -> Result<RetiredBorrowed<'a>, BorrowedSubmitRejected<'a>>;

struct BorrowedSubmitRejected<'a> {
    cause: SubmitError,
    bindings: BorrowedBindings<'a>,
}

enum RetiredBorrowed<'a> {
    Completed(BorrowedBindings<'a>),
    RetiredFailed {
        cause: ExecutionError,
        bindings: BorrowedBindings<'a>,
    },
}
```

The admission boundary is the first provider call that may enqueue device
work. A failure is pre-admission only when it occurs before that call or the
provider result proves that no work was enqueued. `SubmitRejected` and
`BorrowedSubmitRejected` are reserved for that proven case and return the exact
unchanged owning package or borrowed bindings. Binding failure likewise
returns the exact unchanged `PreparedRead` or `PreparedWrite` because binding
precedes the enqueue-capable call.

Once enqueue may have happened, the task retains the package and bindings and
enters G3 `Draining`; an immediate error never returns them. On a post-boundary
failure, ownership returns only as G3 `RetiredFailed` after completion is
proven. If completion cannot be proven, `CompletionUnproven` returns diagnostics
without owners while the provider-private permanent record retains the roots
and context.

A borrowed operation is optional: if offered, it is synchronous through
retirement and is supported only by a provider that guarantees no asynchronous
work survives unwind. After its enqueue-capable call, it returns bindings only
inside `RetiredBorrowed::Completed` or `RetiredBorrowed::RetiredFailed` after
retirement. Asynchronous providers reject borrowed submission as unsupported
before admission.

### Event retirement

After detached admission, a provider-private retirement record owns the event,
the `Arc<RootResource>` roots, and the provider context until completion is
proven.

```rust
struct RetirementRecord {
    event: ProviderEvent,
    roots: Box<[Arc<RootResource>]>,
    context: Arc<ProviderContext>,
}
```

`CompletionUnproven` returns only its typed cause and diagnostics. Its
provider-private record permanently retains the roots and context; it does not
free speculatively and exposes no safe recovery path. There is no
quarantine/poison state, access or retirement registry, or retry transition.
Completion handles may be dropped without changing this retention. When event
retirement is proven, the record releases its retained event, root `Arc`s, and
context reference exactly once before publishing `Completed` or
`RetiredFailed`.

### Raw-handle state table

| Transition | cap | borrow | sync | fail | panic/drop | reclaim |
|---|---|---|---|---|---|---|
| owner/view -> `PreparedRead` or `PreparedWrite` | shared / exclusive | Rust capability carried by prepared access | one bounds/layout/dtype/root-span/alignment/storage/provider validation; write injectivity for writes | exact input capability, no prepared object | no provider state is published | owner follows G1 |
| `Prepared*` -> device binding | prepared shared / exclusive | binding keeps its capability and provider lifetime | binding work only; no second check or request/key/range comparison | exact prepared object | unadmitted binding drops without changing ownership | no device resource is released before binding drop |
| prepared submission -> proven pre-admission rejection | owning / borrowed | package or bindings have not crossed the enqueue-capable call | no enqueue occurred | exact unchanged package/bindings | no event-retirement record exists | caller retains ownership |
| enqueue may have happened -> G3 `Draining` | owning task | task-local prepared bindings; no caller lifetime | event domains drain | no immediate owner return | worker/reaper retains package, event, roots, and context | only after proven event retirement |
| G3 `Draining` -> `RetiredFailed` | owning worker/reaper | none | completion proven | returns owners with typed failure | retirement-held event/root/context references release exactly once | returned owners follow G1 |
| G3 `Draining` -> `Completed` | owning worker/reaper | none | completion proven | returns completed bundle | retirement-held event/root/context references release exactly once | returned bundle follows G1 |
| asynchronous provider rejects borrowed submission | borrowed | unchanged bindings | none; rejection precedes admission | unsupported with exact unchanged bindings | no work survives | caller retains bindings |
| admitted synchronous borrowed operation -> retired result | shared / exclusive | binding borrow remains until return | provider work retires before return | returns bindings only in retired completed/failed outcome | provider contract leaves no async work across unwind | after synchronous retirement |
| retirement -> `CompletionUnproven` | provider-private owning record | no public borrow | completion cannot be proven | diagnostics only; no owner is returned | record permanently retains roots/context | never released by this outcome |

## G6. Documentation ownership

Per-phase documentation deliverables and the commands that validate them.
Commands refer to the repository's current tooling; a phase that renames a
script updates this table in the same PR.

Common validation commands:

- `bash scripts/check-pr-fast.sh` (docs-only mode where applicable)
- `python3 scripts/ci/run_profile.py docs`
- `cargo test --doc --workspace --profile ci`
- `cargo test -p tenferro-tutorial-code --release`
- `python3 scripts/check-public-error-docs.py`
- `python3 scripts/check-operation-categories.py --fail-on-findings --include-rendered`
- `python3 scripts/check-docs-site.py`

| Phase | Owns | Extra validation |
|---|---|---|
| 0 (#1556) | runtime/API docs for discovery, caller-selected engine IDs, endpoint routing; examples must not assume CUDA device 0 or a fixed engine ID | common commands |
| 1 (#1557) | this document; the per-phase ownership table itself | common commands |
| 2 (#1558) | internal architecture/safety rustdoc for the unsafe allocation boundary; the legacy-bridge inventory | common commands |
| 3 (#1559) | `docs/spec/tensor-semantics.md` section III rewritten in the PR that removes public `Buffer<T>`; rustdoc/examples broken by clone/Buffer removal; final owner/view migration notes | common commands |
| 4 (#1560) | G1 state tables kept current; API rustdoc for guards/leases; waits documented as synchronization points, explicitly not copies | common commands |
| 5 (#1561) | storage design updates for immutable aliasing, conservative disjointness, N-way borrow lifetimes, extraction | common commands |
| 6 (#1562) | reinterpretation rustdoc; the reserved section of the views guide (representation view vs numeric cast, supported pairs) | common commands |
| 7 (#1563) | CUDA design doc, device guide, unsafe interop rustdoc, synchronization/reclamation behavior, explicit duplication examples | common commands |
| 8 (#1564) | GPU backend design, device guide, Apple tutorials; synchronization/map transitions vs transfers; one owner with multiple access endpoints | common commands |
| 9 (#1565) | detached vs synchronous scoped ownership, outcome recovery, handle detachment, extraction; G3 state tables kept current | common commands |
| 10 (#1566) | GPU quickstarts, provider matrix, namespace rustdoc; `# Errors` sections for every public `Result` API | common commands |
| 11 (#1568) | hardware evidence recorded in the test profile/worklog with candidate Git commit | common commands |
| 12 (#1569) | `docs/guides/views-and-slicing.md` plus sidebar entry and an **Element access and performance** section; `docs/getting-started/core-concepts.md`; README/tutorials; rustdoc for `as_view`, random access, contiguous guard/slice access, iterators, and rank conversion; runnable owner/view/view-mut traversal examples; the rendered stale-language checker (`scripts/check-storage-docs.py`); the source-blind audit | common commands plus `python3 scripts/check-storage-docs.py --include-rendered`, `python3 scripts/check-storage-element-access-docs.py docs/guides/views-and-slicing.md`, and the exact `p12-element-access-examples` release command |

The Phase 12 element-access section must distinguish O(rank) checked random
access, contiguous typed-slice/guard traversal, prepared strided traversal,
and one-resolution-per-launch backend execution. It documents host-visible
versus device-only storage, including the explicit download boundary, and
warns against repeated multidimensional `get` in a hot loop when a slice,
iterator, or prepared strided plan is available. Every affected rustdoc entry
states whether the operation allocates, dispatches through a provider,
synchronizes, performs per-element bounds/stride work, preserves static rank,
or can transfer/materialize. The source-blind reviewer must be able to select
the zero-overhead path without reading implementation source.
| 13 (#1567) | final worklog linking candidate Git commit, scaffolding disposition, hardware/docs/audit reports; deletion of `HANDOFF-2026-07-25-tenferro-unification6-wip.md` and inbound references | common commands plus closure validation from #1567 |

## G7. AD value retention

The AD layer is the one workload where a single logical value has two
consumers by design: the caller and the tape (or checkpoint state). Today it
is built on shallow clones and `Arc<Tensor>`; under linear ownership it is
built on groups. Blanket replacement of shallow clones with `duplicate()` is
not an accepted migration.

### Ownership root

- Each autodiff context (eager tape, traced execution, checkpoint store)
  owns retained primal allocations through a directly owned group/container
  record. The exact names remain implementation choices; the ownership shape
  is normative:

  ```rust
  struct TapeRetention {
      tape: Arc<TapeRecord>,
      retained: HashMap<ValueKey, Arc<AdValueRecord>>,
  }

  struct AdValueRecord {
      descriptor: DescriptorRef,       // read-only group-local descriptor
      container: Arc<RetentionContainer>,
  }

  struct RetentionContainer {
      group: AllocationGroup,         // directly owns group/root resources
  }

  struct TapeRecord {
      container: Arc<RetentionContainer>,
  }
  ```

- An `EagerTensor` (and any traced value handle) contains an `Arc<AdValueRecord>`
  plus non-owning node metadata. Cloning a handle clones only that `Arc`:
  storage, the descriptor, and the write authority are not cloned. The
  record directly retains the `Arc<RetentionContainer>` that owns the group
  and its root resources. A tape or checkpoint retains the same kind of
  record/container directly; no external table is needed to keep an
  allocation alive.
- Handle types may remain `Clone` because they are read-only descriptor
  references, not owners or capabilities. A handle has no method that returns
  an owner, creates a write lease, or produces a mutable view. The non-`Clone`
  rule (I1) applies to owners and capabilities.
- `ValueKey` is only a local associative key inside a tape/checkpoint
  container. It is never used to reconstruct a descriptor, prove uniqueness,
  or authorize access.
- A descriptor record contains only read-only metadata and shared container
  ownership. It does not allocate a per-element storage/provider object or
  repeat layout/provider validation.
- Handle drop is ordinary `Arc`/container lifetime. Dropping a tape/context
  releases its owning reference; a surviving handle keeps its directly
  retained descriptor record, container, group, and root resources alive.
  When the last reference disappears, normal ownership/drop of the container
  releases the allocation according to G1/G2. Lifetime is represented only by
  these direct owners; no side table participates in release.
- Mutable access is available only from an exclusive owning
  `&mut RetentionContainer`/`&mut TapeRecord` path. The shared handle type
  cannot be used to obtain that borrow. If an owning `Arc` cannot be made
  unique because a handle, tape, checkpoint, or execution record still
  retains it, the mutable operation is unavailable and returns a typed
  uniqueness error; the caller may request an explicit duplicate instead.
  No implicit duplication is provided.
- Retention policy: an operation output is retained iff a declared
  VJP/JVP rule declares it needed for backward, or the user explicitly
  requests retention. Values nobody declares needed are not retained.
- When the caller wants a standalone owner of a retained value, the only
  paths are the G2 paths: consume the handle and uniquely unwrap the direct
  descriptor/container ownership, then move the selected owner out of the
  group, or make an explicit duplicate (classified below). There is no hidden
  copy path and no external identifier check.

### Public API replacement

The `Arc<Tensor>`-returning surface is replaced. No retention adapter appears
in any public or crate-private runtime boundary; the cutover lands directly on
the group-qualified descriptor model. The sketch intentionally exposes the
ownership shape, not final public names.

| Current | Replacement sketch | Semantics |
|---|---|---|
| `materialized(&self) -> Result<Arc<Tensor>>` | `value(&self) -> Result<ValueGuard<'_>>` | materializes if lazy, then exposes a borrowed `TensorView`; host bytes go through G1 guards |
| owned copy of a value | `duplicate_value(&self) -> Result<Tensor>` | explicit copy, reason `ExplicitDuplicate` |
| owned move of a value | `into_value(self) -> Result<Tensor, IntoValueError<Self>>` | consumes the handle; `NotUnique(Self)` is returned before group extraction when direct Arc ownership is shared, while a local G2 failure returns `Extract { value: Self, error: ExtractError }` |
| backward result `Vec<Arc<Tensor>>`, `GradSlot = Arc<Mutex<Option<Arc<Tensor>>>>` | `Gradients` bundle (a G2 group specialization) with `grad(&self, key) -> Option<TensorView<'_>>` and `take_grad(&mut self, key) -> Result<Option<Tensor>, ExtractError>` | one owner per gradient allocation; `Ok(None)` means no gradient, and extraction failure leaves the bundle unchanged |
| traced attached-data maps `HashMap<ValueKey, Arc<Tensor>>` | `ExecutionInputs` bindings over directly retained group descriptors (G3) | no shared tensor owners or mutable authority in the runtime boundary |

```rust
pub enum IntoValueError<H> {
    NotUnique(H),
    Extract { value: H, error: ExtractError },
}
```

`ValueGuard` borrows the record's prepared descriptor view and can request the
G1 host-read guard. A guard does not retain a second owner or perform a new
layout/provider validation. `duplicate_value` is the explicit destination
allocation and data movement path. `into_value` is the consuming path; an
`Arc::try_unwrap`-equivalent structural uniqueness test runs before any G2
group extraction. Failure returns `IntoValueError::NotUnique` with the
original usable handle. Only after Arc uniqueness succeeds may G2 extraction
run; its local error reconstructs and returns the same usable handle in
`IntoValueError::Extract`. These are the only two error variants. Neither
records nor reports the category of the remaining direct owner.

### Checkpoint semantics

- Boundary values (checkpoint region inputs and outputs) are retained as
  descriptor records whose direct container reference keeps the checkpoint
  group and required root resources alive.
- Interior values are deliberately discarded at record time; checkpointing
  must not accidentally retain every intermediate. Their owners are never
  inserted into the checkpoint retained group, and no checkpoint descriptor
  record is created for them. After forward use, each interior owner therefore
  drops normally unless a separate tape, execution record, or external handle
  directly owns it.
- Backward recomputation executes the stored subgraph and produces fresh
  owners; its allocations are classified `CheckpointRecomputeOutput`,
  distinct from retention (which allocates nothing) and from explicit
  duplicates.
- A checkpoint record and a tape record retain their required containers
  directly. Releasing one record is ordinary ownership drop and cannot
  invalidate a descriptor record still held by a handle.

### Reinterpretation and aliases

- A retained complex/real reinterpretation of a retained value is another
  read-only descriptor reference to the same group allocation, with the same
  direct container record and no second owner.
- Mutable reinterpretation requires an exclusive or owning capability. A
  handle cannot supply it. If a tape, checkpoint, execution record, or sibling
  handle still retains the container, the unique owning path cannot be
  obtained and the ownership acquisition returns undifferentiated
  `NotUnique`. This is a direct ownership/borrow property, verified by
  structural-uniqueness tests and compile-fail tests for mutable access
  through a shared handle. An explicit duplicate is the only alternative;
  reinterpretation never duplicates implicitly.

### Copy and allocation accounting

Copy events and allocation events are distinct ledgers:

```rust
enum CopyReason {
    ExplicitDuplicate,
    Transfer,
}

enum AllocationReason {
    OperationOutput,
    CheckpointRecomputeOutput,
    ExplicitDuplicateDestination,
    TransferDestination,
}
```

- A byte-preserving duplicate or transfer records a `CopyReason`; allocating
  its destination independently records the matching `AllocationReason`.
  An operation kernel records an allocation reason for a fresh output but is
  not thereby a copy. Checkpoint recomputation records fresh allocations as
  `CheckpointRecomputeOutput`; it is not reclassified as a copy merely
  because the values reproduce prior results.
- Retention has neither a copy nor an allocation reason because retaining a
  descriptor performs neither operation.
- Every physical allocation emits exactly one allocation event. Every
  byte-preserving explicit duplication or transfer emits exactly one copy
  event, whether or not its destination allocation is new. Kernel writes,
  initialization, reductions, and newly computed operation results are not
  copy events; a fresh kernel result records only its allocation reason.
  Descriptor aliasing, metadata-only outputs, and public-handle cloning emit
  neither event.
- Acceptance for an AD scenario (forward plus backward, with and without
  checkpointing): every observed byte-preserving duplication/transfer and
  every observed allocation carries a reason from its own enum, and each
  ledger matches the scenario's expected multiset. Kernel writes and new
  computed results must not increment the copy ledger. Copies and allocations
  attributable to retention are therefore both exactly zero.
- Aggregate pre-migration versus post-migration copy counts are not an
  acceptance criterion.

### Contract tests

- A direct-lifetime test drops the tape/context while a cloned handle remains,
  reads through that handle, then observes release only after the final direct
  record/container reference is dropped.
- Structural-uniqueness tests cover a sibling handle, tape record,
  checkpoint record, and execution record. Each shared case rejects
  `into_value` with `IntoValueError::NotUnique`, preserves the same usable
  handle, and proves that G2 extraction was not attempted. A uniquely owned
  Arc proceeds to G2: local extraction failure returns
  `IntoValueError::Extract` with the usable handle, while success moves one
  owner through G2. All shared-owner setups assert the same `NotUnique`
  result.
- `take_grad` tests distinguish `Ok(None)`, `Ok(Some(owner))`, and
  `Err(ExtractError)`; the error case leaves the `Gradients` bundle unchanged.
- Compile-fail tests show that a shared handle has no mutable view or owner
  projection, and that mutable reinterpretation requires an exclusive owning
  borrow. An explicit duplicate test verifies that duplication is requested by
  the caller and that retention itself never copies.
- A checkpoint test inserts only boundary owners into its retained group,
  asserts that no interior owner or descriptor was inserted, observes the
  omitted interior allocation release after forward use, and checks fresh
  recomputation ownership.
- Forward/backward CPU and designated asynchronous-provider tests compare the
  reason-classified copy/allocation multisets and require zero events caused
  by retention. Kernel writes and fresh results contribute allocation events
  where applicable but no copy events.

### Atomic cutover

Public host ownership (#1559) and final detached/scoped runtime plus direct
group-based AD retention (#1565) form one atomic promotion cohort. They land
the final `AllocationGroup`, lease, retirement, and descriptor-ownership
semantics together. There is no interim AD-retention adapter or compatibility
path. Each provider consumes the final prepared access and event-retirement
contracts directly when its owning phase lands, with no additional ownership
seam.

### Validation lanes

- CPU is mandatory: eager and traced forward plus backward, and a
  checkpoint boundary-retention/recomputation case, with reason-classified
  copy and allocation counters compared as separate expected multisets.
- One supported asynchronous accelerator lane is designated before the
  Stage A freeze (CUDA preferred, else WebGPU or Apple/Metal) and runs the
  same retention contract with allocation, copy-reason, and retirement
  counters on real asynchronous provider work. Hidden CPU fallback or
  staging is a failure. Required-hardware mode cannot pass by skipping
  (#1568).

### State table: retained value lifecycle

| Transition | cap | borrow | sync | fail | panic/drop | reclaim |
|---|---|---|---|---|---|---|
| record op output into tape | owning (tape/container takes the output owner into its group) | none | none | op error: no retained record | unwind drops the partially constructed direct owner exactly once | container/group ownership follows G1/G2 |
| clone `EagerTensor` handle | none (read-only `Arc<AdValueRecord>` reference) | none | none | n/a; clone does not resolve or validate storage | ordinary `Arc` clone; no owner or write authority is created | record/container remains while any direct owner exists |
| drop a handle while tape/checkpoint retains | none | none | none | n/a | ordinary `Arc` drop; tape/checkpoint record remains independent | allocation remains under direct container ownership |
| tape/context drop while a handle remains | none | none | outstanding work follows G1 | n/a | tape's owning record drops; the handle's record retains the container/group/root | normal container drop after the last direct owner |
| tape retains/releases descriptor record | owning tape/container reference / none on release | none | none | local `ValueKey` lookup may fail before a record is acquired; no storage is changed | an `Arc<AdValueRecord>` keeps the descriptor/container valid for its lifetime | group/container drops when no direct owner remains |
| checkpoint retains/releases boundary record | owning checkpoint/container reference / none on release | none | none | local boundary-key lookup may fail before a record is acquired | ownership keeps an acquired record valid; interior values get no record | boundary allocation follows direct ownership |
| `value()` guard | shared | record/container borrow for guard lifetime | G1 host-read rules if host bytes requested | error, record/container unchanged | guard drop ends the borrow | n/a |
| backward execution | shared reads of retained descriptors; new owners for grads | tape/container shared during execution | G3 rules | typed failure, tape unchanged, grads dropped after retirement | per G3 panic row | grads owned by `Gradients` bundle |
| `take_grad` | exclusive on `Gradients` | exclusive bundle borrow until return | none | `Ok(None)` for no gradient; `Err(ExtractError)` leaves the bundle unchanged | owner is removed only after successful extraction | `Ok(Some(owner))` follows G1 |
| `into_value` while another direct owner exists | owning attempt (consumes one handle) | none | none | `IntoValueError::NotUnique(Self)`; G2 extraction is not attempted | the original usable handle and all direct owners are preserved | n/a |
| `into_value` after unique Arc ownership | owning (consumes and uniquely unwraps the record/container before G2) | none | none | local G2 failure returns `IntoValueError::Extract { value: Self, error: ExtractError }` | failure reconstructs the usable handle; success moves the owner exactly once | extracted owner per G1 |
| checkpoint record | owning (only boundary owners/descriptors enter the checkpoint retained group) | none | none | error: no partial checkpoint record is published | interior owners are never inserted and drop after forward use absent another direct owner | boundary containers after direct owners drop |
| checkpoint recompute (backward) | shared reads of boundary; fresh owners for recomputed values | checkpoint container shared | G3 rules | typed failure after retirement | per G3 | recomputed owners dropped after use |
| tape drop | owning tape/container reference | none | retirement per G1 for in-flight work | n/a | direct records and event-retirement ownership follow G1; ordinary ownership only | after the last direct owner, exactly once |

## Contract test index

Each gate's clauses are enforced by tests owned by the listed phases. The
phase issues carry the full inventories; this index is the cross-reference.

| Gate | Enforcement | Owning phases |
|---|---|---|
| G1 ordering, guards, revalidation, retirement | deterministic fake-timeline transition tests; claim provenance/split/overlap and exactly-once root-deallocator tests; compile-fail (guard across consuming submit, write guard from shared); corrupt-descriptor rejection at map and enqueue; immediate-drop-after-enqueue; quarantine poisoning; Miri on host guard slices; constant resolve/map/lease/dispatch counts and no per-element abstraction work | #1560, performance evidence in #1566, providers in #1563/#1564 |
| G2 group, splitting, extraction | construction-time invalid layout/range/storage/provider rejection and retained-metadata counters; N-way split cases (N=0,1,>2, empty, reverse-stride) proving validation counters do not increase; write injectivity checked only when its retained proof is absent; pairwise-disjointness and permutation-independence property tests; direct borrowed-slot resolution for shared/exclusive group borrows, including empty entries; structural extraction-uniqueness tests (aliased records reject, sole record moves one owner, consuming extraction discards the rest); compile-fail (root access while children live); extraction counters; map/enqueue tests assert no validation rerun | #1561 |
| G3 submission terminal semantics | executable checks prove exact detached/scoped rejection recovery; host/CPU synchronous scoped acceptance and CUDA/WebGPU/Metal or asynchronous-provider rejection before admission; no borrowed work at return or unwind and no panic-catch/`Drop` safety; borrowed output-view coverage; consuming `into_output`/`into_owned_output` cases prove repeated and duplicate-output aliases plus the remaining map disappear together, failures return the exact bundle, and scoped borrowed/metadata rejection never copies; source checks reject extracted-state flags; worker/provider panic drains to typed `RetiredFailed` when completion is proven and ownerless `CompletionUnproven` otherwise; handle-detach and terminal-outcome suites; compile-fail (host guard across submit) | #1565, hardware in #1568 |
| G4 method distribution | API-parity contract with one canonical method list; compile-fail (no `Clone` on owners/capabilities); source scan (no mutable owner projections); static-rank preservation; allocation-free O(1) view construction; release traversal and fixed-rank codegen evidence | #1557 harness, #1559, #1566 |
| G5 raw handles, reclamation | executable prepared-once resolution counts; source checks that bind/enqueue accept no replacement request/key/range and perform no repeated static validation; compile-fail shared write plus source inventory proving raw handles, `Arc`, and refcounts cannot mint write authority; provider-matrix checks that asynchronous providers accept detached owning submission only and reject borrowed submission before admission; exact-return tests limited to failures proving no enqueue occurred, with post-boundary failures routed through G3 terminal outcomes; proven-retirement tests releasing event/roots/context exactly once; `CompletionUnproven` tests returning no owners and permanently retaining roots/context; raw-binder source inventory and unsafe-interop rustdoc checks for binding lifetime, synchronization duties, and post-retirement invalidity | #1558, #1563, #1564 |
| G6 documentation | rendered stale-language checker; doctests; checked cost-model content; runnable owner/view/view-mut traversal tutorial; tutorial-code checks; source-blind audit | #1569, #1567 |
| G7 AD retention | byte-preserving duplicate/transfer copy counters separated from allocation/kernel-write counters (zero retention events in both); ordered Arc-uniqueness-before-G2 tests with `IntoValueError<Self>` preservation; `take_grad` three-outcome extraction tests; acquired-record validity and local-key-miss tests; checkpoint retained-group exclusion/interior-release test; compile-fail mutable-reinterpret exclusion; CPU plus designated async accelerator lanes | #1557 contract, atomic #1559/#1565 cutover, #1568 evidence |

## Relationship to phase issues

- #1556 and #1557 are independent roots of the canonical DAG; #1557 owns the
  v2 ledger and its executable test gate.
- `p1-element-access-baseline` is active with the tracked measured report and
  verifier. The report is identified by measured commit
  `da7b36e699f9f4731dec08de6a4e1ca93f20cd6f` and repository-relative path
  `docs/testing/storage-element-access-baseline.json`; P10 may compare it only
  under the compatible-environment rule above.
- #1558 owns the root pin, non-`Clone` claim, and the single typed provider
  bridge. #1560 owns access/retirement. #1561 owns groups and direct borrowed
  descriptor slots. None waits for the public host cutover.
- #1559 and #1565 are one atomic promotion cohort. The cohort lands public
  host ownership, final detached/scoped runtime ownership, and direct AD group
  retention together.
- #1562, #1563, #1564, and #1566 follow the graph above. #1567-A freezes and
  cleans the candidate; #1568 and #1569 validate the same candidate; #1567-B
  is the final independent closure audit.
- A later phase that needs to deviate from a contract here must change this
  document (and its tests) first, in the same PR, with the deviation called
  out in the PR description. It may not introduce an interim adapter to avoid
  the change-control path.
