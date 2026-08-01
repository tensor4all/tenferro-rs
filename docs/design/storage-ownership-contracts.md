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
- One typed, crate-private provider bridge is permitted while an accelerator
  provider is still being migrated. It must already use the final
  claim/pin/access/lease/retirement contract, cannot mint ownership or writes,
  and is removed by the CUDA and WebGPU/Metal migration phases. No AD,
  submission, or conservative-synchronization adapter is permitted. The
  completed redesign has no dual storage stack, permanent compatibility
  adapter, or provider-specific authority escape hatch.
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
- **Quarantine**: the terminal state for resources whose retirement cannot be
  proven; they are retained and reported, never freed speculatively.

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
`scripts/storage-ownership-contracts.toml`. After the atomic migration, that
file is the sole machine authority for the production graph and obligations.
The Python tuples in `scripts/test-storage-ownership-contracts-v2.py`
(`UNITS`, `EDGES`, and the obligation expectations) are independent verifier
expectations used to detect drift; they are not a second production registry.
The graph and tables in this document are explanatory documentation. The v2
checker and runner are `scripts/check-storage-ownership-contracts.py` and
`scripts/run-storage-ownership-contracts.py`; both are deliberately absent in
their v2 form in this RED checkpoint (the first path still contains the
superseded v1 checker). The executable RED specification is
`scripts/test-storage-ownership-contracts-v2.py`; it must remain checked in
and must invoke the exact production manifest as well as adversarial temporary
repositories.

The checked-in production manifest, v1 checker, v1 test suite, and full v1
fixture are current superseded deletion debt. They are owned by the immediate
atomic checker implementation checkpoint and are not an accepted compatibility
surface. That checkpoint must replace the manifest with exact v2 registry
content, make the checker v2-only, delete/replace the v1 suite, and leave only
the minimal schema-only legacy fixture for rejection. The RED contract records
that migration deterministically rather than accepting a compatibility
parser. After migration, the production manifest must equal the independent
v2 verifier expectation exactly before checker success is required.

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
The sole provider bridge is a separate typed obligation under the root/provider
phase; its implementation already obeys the final lease and retirement
contract and its removal is part of the accelerator migration.

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
Every member obligation of an atomic cohort must make that transition in the
same candidate; partial cohort activation is rejected. A changed artifact,
command, ID, unit, or gate is a new obligation, not a promotion.

Every registered unit owns at least one required obligation. A unit is
complete only when all of its required obligations are active and the
candidate-bound receipt contains a successful execution result for every one
of them. An empty obligation set is invalid rather than vacuously complete.
In particular, CUTOVER cannot activate until P0 and P5 are each complete by
this rule; merely naming them as cohort prerequisites is not proof.

The runner emits a candidate-bound receipt containing:

```json
{
  "schema": "tenferro.storage-ownership-receipt.v1",
  "base_commit": "...",
  "candidate_commit": "...",
  "base_manifest_sha256": "...",
  "candidate_manifest_sha256": "...",
  "executions": [
    {
      "obligation_id": "...",
      "artifact_id": "...",
      "artifact_sha256": "...",
      "command_id": "...",
      "command_sha256": "...",
      "candidate_commit": "...",
      "exit_code": 0
    }
  ]
}
```

The runner executes each active typed command exactly once, passes the exact
artifact binding, and records one result per active obligation. Every
execution must bind the manifest-derived `obligation_id`, `artifact_id`, and
`command_id`, repeat the actual candidate commit, and carry the
manifest-derived artifact and command digests. The receipt also carries the
exact base/candidate manifest digests. It does not execute deferred commands.
The checker validates the receipt; it does not manufacture command results.
Both tools resolve `candidate_commit` from `git rev-parse HEAD` and
load the base manifest from the same repository-relative manifest path at the
actual `base_commit` Git object. The supplied base must be an ancestor of
HEAD. Thus matching fake strings in the CLI and receipt cannot substitute for
repository history. The checker resolves every artifact and command path with
filesystem-aware traversal and symlink checks. Terminal status is derived
only from zero deferred obligations, complete atomic promotions, and one
successful candidate-bound receipt result for every required obligation. No
boolean terminal switch can make a manifest complete.

Digest canonicalization is part of the receipt contract: manifest digests are
SHA-256 over the exact manifest bytes, artifact digests are SHA-256 over the
resolved repository file bytes, and command digests are SHA-256 over the
UTF-8, sorted-key, compact JSON encoding of the typed command value. A receipt
with a correct-looking ID but a digest for another candidate is invalid. The
verification harness independently recomputes all four digest classes and
mutates artifact bytes, the base-manifest digest, and candidate files after a
receipt. An in-repository symlink is retargeted after receipt. A separate
machine-readable capability test is required before the symlink cases; an
unsupported host is an explicit failure/event, never a skipped or optional
green result. On capable hosts the checker must reject the changed resolved
artifact bytes.

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
accepted as a promotion. The RED specification fixes the checker invocation
for base/candidate comparison and the receipt fields so that a later checker
cannot silently weaken this rule.

The v2 checker accepts `--root`, repository-relative `--manifest`, optional
`--base-commit` plus `--receipt` for promotion/receipt validation,
`--summary-json`, and `--diagnostics-json`. The runner accepts `--root`, the
same repository-relative `--manifest`, `--base-commit`, `--receipt-out`, and
`--diagnostics-json`.
Neither tool accepts a caller-supplied candidate commit: candidate identity is
HEAD. Neither tool may infer a different manifest or command target from the
current working directory. A receipt written by the runner is the only
execution proof consumed by the checker.

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
v1; human `message` text is not. The RED suite therefore asserts codes and
relevant IDs/paths rather than prose. Each one-fault case requires the exact
one-code set and the exact field-key shape registered by the RED harness;
duplicate codes, unknown codes, extra envelope keys, missing `message`, or
extra identifying fields are failures. This prevents a checker from passing a
negative case by emitting every known code or an unrelated diagnostic.

The v2 diagnostic code registry is grouped by failed invariant: `E_SCHEMA_VERSION`,
`E_SCHEMA_PARALLEL_TABLE`, `E_SCHEMA_UNKNOWN_TABLE`,
`E_OBLIGATION_TAGGED_STATE`, `E_UNIT_OBLIGATION_MISSING`,
`E_GRAPH_P2_PREREQUISITE`, `E_GRAPH_DUPLICATE_EDGE`,
`E_GRAPH_UNKNOWN_UNIT`, `E_COHORT_DEFINITION`,
`E_COHORT_PARTIAL_PROMOTION`, `E_COHORT_PREREQUISITE_INCOMPLETE`,
`E_OBSOLETE_OWNERSHIP_TABLE`, `E_ARTIFACT_SYNTHETIC_TERMINAL`,
`E_ARTIFACT_DUPLICATE_TARGET`, `E_ARTIFACT_MISSING`, `E_PATH_ESCAPE`,
`E_PATH_SYMLINK_ESCAPE`, `E_DEFERRED_ARTIFACT_EXISTS`,
`E_COMMAND_KIND`, `E_COMMAND_ARGV`, `E_COMMAND_ARGV_BINDING`,
`E_COMMAND_CWD_ESCAPE`, `E_COMMAND_PATH_ESCAPE`,
`E_COMMAND_ARGV_PATH_ESCAPE`, `E_COMMAND_CWD_SYMLINK_ESCAPE`,
`E_COMMAND_ARGV_SYMLINK_ESCAPE`, `E_COMMAND_ARTIFACT_BINDING`,
`E_COMMAND_TARGET_BINDING`, `E_COMMAND_ID_CONFLICT`, `E_COMMAND_FAILED`,
`E_PROMOTION_IDENTITY`, `E_RECEIPT_COMMIT`, `E_RECEIPT_DIGEST`,
`E_RECEIPT_EXECUTION_BINDING`, `E_RECEIPT_INCOMPLETE`, and
`E_TERMINAL_DECLARED`. Command kinds have an exact allow-listed `argv` vector:
`E_COMMAND_ARGV_BINDING` has exactly `command_id`, `index`, `expected`, and
`actual`. `E_COMMAND_CWD_ESCAPE` has exactly `command_id` and `cwd`, and
covers absolute cwd values and cwd values whose normalized path escapes the
repository. `E_COMMAND_ARGV_PATH_ESCAPE` has exactly `command_id`, `index`,
and `argument`; every path-bearing argv element is canonicalized independently
of `path_args`, so lying by omitting an argv value from `path_args` cannot make
it safe. `E_COMMAND_CWD_SYMLINK_ESCAPE` has exactly `command_id` and `cwd`;
`E_COMMAND_ARGV_SYMLINK_ESCAPE` has exactly `command_id`, `index`, and
`argument`. These symlink diagnostics are also required on post-receipt
revalidation when a previously internal command path is retargeted outside
the repository. The execution-binding code has exactly `obligation_id`,
`field`, `expected`, and `actual` fields and is used for a swapped or forged
execution identity. Adding a code is compatible; changing the meaning or
required identifying fields of an existing code requires a diagnostic schema
revision.

The RED suite includes both a structured temporary repository for adversarial
path, graph, promotion, and command-binding cases and an integration case that
invokes the checked-in production manifest. Temporary repositories contain
real files and real symlinks; they do not stand in for the production gate.

The RED command emits a machine-readable
`tenferro.storage-ownership-red-report.v1` record. Its expected-failure set is
keyed by test and subtest parameters and records both a named cause and the
expected exception type. The runner compares observed failure/error/subtest/
skip events with that exact set as a multiset, preserves duplicate
multiplicity, and requires equal total event counts. An unlisted event,
duplicate event, wrong exception/cause, skipped required test, or missing
expected event makes the RED harness itself fail, so an implementation cannot
relabel an unexpected regression as intentional.

The base RED snapshot has completed P0, P1, and P2. P4 and P5 remain deferred;
the CUTOVER candidate activates every required P4/P5 obligation and obtains
successful runner evidence before atomically activating all P3/P9 obligations.
In particular, the canonical obligation set includes:

- P4/G1+G4: a deferred production-code-bound compile/test artifact for the
  private dispatch borrow shape and exact `ResolvedWrite` failure recovery;
- P3/G1+G4: a compile contract using the repository compile-test harness or
  static assertions for `UseLease`/`BackendRawLease: Send + !Sync` and
  `BackendRawMapping`/host guards: `!Send + !Sync`;
- P4/G1+G3: a provider runtime test for take-before-call, exactly-once release,
  callback panic containment, and root quarantine after release panic.

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
    state: Arc<RootResourceState>, // lifetime/deallocator state only
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

pub(crate) struct RootResourceState {
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
// private RootResourceState dispatchers below. It contains the exact pin/root
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

impl RootResourceState {
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
// it exactly once; on success that box moves into RootResourceState and is
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
dispatch uses the one vtable retained in `RootResourceState`. A provider may
allocate an event or queue object under its own documented backend contract,
but resolution and the core binding path must not allocate.

The Phase 1 acceptance harness records allocator events around a warmed
`resolve -> acquire_host_*` and `resolve -> acquire_device_*` loop. The core
counter must remain zero for both read and write paths (with provider-owned
event allocation measured separately and explicitly reported). A benchmark
receipt records the loop count, allocator counter, resolved-value size, and
backend; a regression that introduces a per-access allocation fails the G1
performance gate.

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
the resolved root, claim, pin, or the pin-state access vtable. RED must assert
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
    allocations: Vec<OwnedStorage>,   // private: each owned span exactly once
    values: GenerationalDescriptors,  // private: interpretation + slot
}

pub struct TensorDescriptor {
    slot: AllocationSlot,             // index into `allocations`
    dtype: DType,
    layout: TensorLayout,
    placement: Placement,
}

pub struct ValueId {
    group: GroupId,
    slot: u32,
    generation: u32,
}

struct GenerationalDescriptors {
    group: GroupId,
    slots: SlotMap<DescriptorSlot>,
}

struct DescriptorSlot {
    generation: u32,
    descriptor: Option<TensorDescriptor>,
    roots: DescriptorRoots, // handles, tape, checkpoint, execution
}
```

Construction preconditions (safe constructors):

- every `OwnedStorage` appears once; duplicate owner tokens are impossible by
  move semantics, and overlapping owner spans for one key are rejected;
- every descriptor is validated against its slot's span (G1 revalidation
  rules) at construction;
- descriptors may alias freely, including exact duplicates.
- A `ValueId` is stable only while its descriptor slot and generation are
  live. Removing a descriptor tombstones the slot; reuse increments the
  generation. Stale IDs fail with a structured error and can never resolve
  to a later value. The group component prevents an ID from resolving in a
  different group even when slot and generation happen to match. Slot
  indices, vector addresses, and provider handles are not public identity.
- `GroupId` is opaque, non-forgeable outside the registry, and is never
  reused while a stale `ValueId` could exist. Exhaustion is a structured
  construction error, never wraparound. Group identity is descriptor-table
  identity only and cannot authorize allocation access.
- `GenerationalDescriptors` is the sole authoritative descriptor-liveness
  registry. Root registration/release and descriptor lookup are atomic with
  respect to slot tombstoning. G2 extraction and G7 handle operations consult
  this registry; no side table or provider reference count may override it.

### Operation contracts

```rust
fn view(&self, id: ValueId) -> Result<TensorView<'_>, GroupError>;
fn view_mut(&mut self, id: ValueId) -> Result<TensorViewMut<'_>, GroupError>;
fn split_mut(&mut self, ids: &[ValueId])
    -> Result<Vec<TensorViewMut<'_>>, DisjointViewError>;
fn try_extract(&mut self, id: ValueId) -> Result<Tensor, ExtractError>;
fn into_tensor(self, id: ValueId) -> Result<Tensor, (Self, ExtractError)>;
```

- `view` borrows the group shared; any number of aliasing read views may
  coexist.
- `view_mut` exclusively borrows the whole group; one at a time.
- `split_mut` returns N simultaneous mutable views only after the central
  disjointness proof (below). Children are non-cloneable and hold the
  exclusive borrow of the group; the root is inaccessible while any child
  lives.
- `try_extract` removes descriptor `id` and moves its allocation out as a
  standalone owner only when no remaining descriptor or registered external
  descriptor handle references the same allocation slot. On failure the
  group is unchanged and the error carries a typed reason. Removing the
  descriptor invalidates its generation. There is no copy or materialization
  fallback (I4).
- `into_tensor` consumes the group, selecting one descriptor and explicitly
  discarding the rest; it never duplicates ownership to preserve them. On
  failure it returns the unchanged group.

### Central disjointness proof

One audited module owns the proof. Normative validation order (#1561):

1. validate each layout with checked shape/stride/offset arithmetic;
2. resolve dtype-sized byte ranges against the exact root-bound claim span;
3. prove each mutable layout internally injective;
4. partition requests by allocation key and root span;
5. treat empty descriptors as non-overlapping;
6. prove pairwise disjoint reachable byte envelopes;
7. split the root exclusive capability into non-cloneable children.

Conservative rejection is required rather than element enumeration:
interleaved strided requests whose byte envelopes overlap return
`NotProvablyDisjoint`. Error variants: invalid layout, foreign allocation,
internal overlap, pairwise overlap, not provably disjoint, overflow,
unsupported provider span. Every error leaves the group unchanged.

### State table

| Transition | cap | borrow | sync | fail | panic/drop | reclaim |
|---|---|---|---|---|---|---|
| construct group | owning (consumes owners) | none | none | owners returned to caller or dropped exactly once, per constructor contract | no partially observable group | owners' G1 rules |
| `view` | shared | group, guard-free descriptor view | none (host access goes through G1 guards) | error, group unchanged | view drop is borrow end | n/a |
| `view_mut` | exclusive | whole group for view lifetime | none | error, group unchanged | borrow end; bytes written stay written | n/a |
| `split_mut` | exclusive | whole group, transferred to children | none | structured `DisjointViewError`, group unchanged | children drop ends borrow; no partial child set is observable on panic (proof precedes construction) | n/a |
| `try_extract` | exclusive | none after return | none | typed reason, group unchanged | n/a | extracted owner follows G1 |
| `into_tensor` | owning | none | none | group returned unchanged with reason | n/a | discarded owners follow G1 drop rules |

## G3. Submission

Two complementary APIs (#1555, "Runtime ownership and asynchronous
execution"; #1565). Detached execution returns an owned group-based result;
scoped execution returns the hybrid borrowed/owned result defined below.

### Signatures

```rust
pub struct ExecutionInputs {
    group: AllocationGroup,
    bindings: Box<[ValueId]>, // graph input i reads descriptor bindings[i]
}

pub fn submit(
    &self,
    program: &CompiledGraph,
    inputs: ExecutionInputs,
) -> Result<ExecutionHandle, SubmitRejected>;

pub struct SubmitRejected {
    error: Error,
    inputs: ExecutionInputs, // the exact unaccepted package
}

pub struct ExecutionFailure {
    cause: ExecutionError,
    inputs: ExecutionInputs,
}

pub struct CancelledExecution {
    inputs: ExecutionInputs,
}

pub struct QuarantinedExecution {
    cause: RetirementError,
    quarantine: QuarantineId,
    affected: Box<[AllocationKey]>, // diagnostic identity, never owners
}

pub enum ExecutionOutcome {
    Completed(ExecutionBundle),
    Failed(ExecutionFailure),       // recovered inputs, typed cause
    Cancelled(CancelledExecution),  // recovered inputs
    Quarantined(QuarantinedExecution), // runtime registry retains resources
}

pub struct ExecutionBundle {
    group: AllocationGroup,
    outputs: Box<[ExecutionOutput]>,
    retained_inputs: Box<[Option<ValueId>]>,
}

pub enum ExecutionOutput {
    Tensor(ValueId),
    Metadata(OutputMetadata),
}

impl SubmitRejected {
    pub fn into_parts(self) -> (Error, ExecutionInputs);
}

impl ExecutionFailure {
    pub fn into_parts(self) -> (ExecutionError, ExecutionInputs);
}

impl CancelledExecution {
    pub fn into_inputs(self) -> ExecutionInputs;
}

impl QuarantinedExecution {
    pub fn into_parts(self) -> (RetirementError, QuarantineId, Box<[AllocationKey]>);
}

pub fn scope<'env, R>(
    &self,
    f: impl for<'s> FnOnce(&'s SubmitScope<'s, 'env>) -> R,
) -> Result<R, ScopeExitError<R>>;

impl<'s, 'env> SubmitScope<'s, 'env> {
    pub fn submit_read_only(
        &'s self,
        program: &CompiledGraph,
        inputs: ScopedReadInputs<'env>,
    ) -> Result<ScopedHandle<'s, 'env>, ScopedSubmitRejected<'env>>;
}

impl<'s, 'env> ScopedHandle<'s, 'env> {
    pub fn wait(self) -> ScopedExecutionOutcome<'env>;
}

pub enum ScopedExecutionOutcome<'env> {
    Completed(ScopedExecutionBundle<'env>),
    Failed(ScopedExecutionFailure<'env>),
    Cancelled(ScopedCancelledExecution<'env>),
    Quarantined(ScopedQuarantinedExecution<'env>),
}

pub struct ScopedExecutionBundle<'env> {
    allocations: Box<[ScopedAllocation<'env>]>,
    values: GenerationalDescriptors,
    outputs: Box<[ScopedOutput]>,
}

pub enum ScopedOutput {
    Tensor(ValueId),
    Metadata(OutputMetadata), // genuinely storage-free graph result
}

enum ScopedAllocation<'env> {
    Borrowed(StorageRef<'env>),
    Owned(OwnedStorage),
}

pub struct ScopedSubmitRejected<'env> {
    error: Error,
    inputs: ScopedReadInputs<'env>,
}

pub struct ScopedExecutionFailure<'env> {
    cause: ExecutionError,
    inputs: ScopedReadInputs<'env>,
}

pub struct ScopedCancelledExecution<'env> {
    inputs: ScopedReadInputs<'env>,
}

pub struct ScopedQuarantinedExecution<'env> {
    cause: RetirementError,
    quarantine: QuarantineId,
    inputs: ScopedReadInputs<'env>,
}

pub struct ScopeExitError<R> {
    value: R,
    unobserved: Box<[ScopedTaskFailure]>,
}

impl<'env> ScopedSubmitRejected<'env> {
    pub fn into_parts(self) -> (Error, ScopedReadInputs<'env>);
}

impl<'env> ScopedExecutionFailure<'env> {
    pub fn into_parts(self) -> (ExecutionError, ScopedReadInputs<'env>);
}

impl<'env> ScopedCancelledExecution<'env> {
    pub fn into_inputs(self) -> ScopedReadInputs<'env>;
}

impl<'env> ScopedQuarantinedExecution<'env> {
    pub fn into_parts(
        self,
    ) -> (RetirementError, QuarantineId, ScopedReadInputs<'env>);
}

impl<R> ScopeExitError<R> {
    pub fn into_parts(self) -> (R, Box<[ScopedTaskFailure]>);
}
```

- Repeated or aliased bindings reference descriptors; they never duplicate
  owners.
- `ExecutionBundle` fields are private. Tensor `output()` returns a borrowed
  view; `output_mut()` exclusively borrows the whole bundle; extraction
  follows G2. Identity, metadata-only tensor transforms, repeated-input, and
  duplicate-output graphs keep exactly one owner per physical allocation,
  with no hidden copy. A genuinely storage-free output uses
  `ExecutionOutput::Metadata`, parallel to scoped execution.
- `ScopedReadInputs` borrows immutable tensor/group views for `'env` and
  declares its access mode explicitly. Provider read leases are still
  acquired (G1), because logically read-only host and device uses can
  conflict on some providers. G3 contains no scoped writable-input contract;
  scoped submission is read-only and no writable scoped row may be inferred
  from this surface.
- `ScopedHandle<'s, 'env>` cannot escape the scope (higher-ranked `'s`). Its
  `wait` result contains no `'s` borrow and may leave the scope, but remains
  bounded by the original input lifetime `'env`.
- A completed scoped bundle is deliberately hybrid. Identity,
  metadata-only, repeated-input, and duplicate-output results are
  tensor descriptors whose slot is `ScopedAllocation::Borrowed`; newly
  computed tensor results use `ScopedAllocation::Owned`. A genuinely
  storage-free result is `ScopedOutput::Metadata` and never receives a fake
  allocation slot. `output()` borrows the bundle and returns a view bounded
  by both that borrow and `'env`. Extraction is available only for an
  `Owned` slot satisfying G2; requesting extraction from a `Borrowed` slot
  returns `BorrowedOutput` and never copies.
- `ScopedSubmitRejected<'env>` returns the exact unadmitted
  `ScopedReadInputs<'env>` with its typed cause. After
  admission, `ScopedExecutionFailure<'env>` and
  `ScopedCancelledExecution<'env>` retain the exact borrowed input
  descriptors for diagnosis; caller ownership was never transferred. Partial
  or uninitialized owned outputs remain private and are retired and dropped
  or quarantined before either outcome becomes observable. Consuming
  accessors return the error and exact input package; private fields are not
  the recovery contract.
- Scope exit joins and retires every admitted task whose handle was not
  waited. Dropping a handle abandons observation, not execution. A waited
  `ScopedExecutionBundle<'env>` may be returned from the closure because it
  contains only `'env` borrows plus its own fresh owners, never a scope
  borrow.
- Scope exit is an explicit synchronous `join_and_retire_all` transition, not
  a `Drop` side effect. The read-only scope-owned task registry still joins
  every task and retires every lease before ending the `'env` borrow. If the
  closure panics, the implementation catches the unwind long enough to
  perform the same synchronous join/retirement, records secondary failures,
  and then resumes the original panic. `Drop` is diagnostic cleanup only and
  never establishes safety or retirement.
- A normal scope exit reports every unobserved task failure through
  `ScopeExitError<R>` while preserving the closure result. If the closure
  panics, the scope guard still drains, records secondary failures in the
  documented runtime error sink, and resumes the original panic; it never
  replaces that panic with a second one.
- If retirement cannot be proven, `Quarantined` is a distinct terminal
  outcome, not `Retired`. Before a scoped borrow can end, provider state for
  every affected root is atomically marked quarantined. All later safe map,
  enqueue, extraction, and deallocation attempts return `Quarantined`; the
  quarantine registry retains the root resource and context. Thus dropping
  a scoped error cannot expose borrowed storage to work of unknown status.
- The detached runtime is the only G3 asynchronous path that owns writable
  inputs: `submit` consumes `ExecutionInputs`, whose `AllocationGroup` owns
  its `OwnedStorage` values after admission. Those owners move into the
  retirement record with the leases; no detached record stores a Rust borrow.
  A static `UseLease` alone never makes an owner externally writable while
  device work remains.
- A direct borrowed device-write API is synchronous in its public lifetime
  shape: it may use `WriteBinding<'a>` internally, but it must join and retire
  all device work before returning and therefore cannot place `'a` in a
  `'static` retirement record. It returns the exact borrowed input on
  pre-admission failure. This is distinct from the read-only scoped API and
  detached owning submission; there is no scoped write API in this contract.

### Lifecycle

States: `Prepared` (validation/planning), `Admitted` (worker owns inputs),
`Running`, `Draining` (event domains drain after completion, error, panic,
or cancellation), then either `Retired(outcome)` or
`Quarantined(quarantine_id)`.

| Transition | cap | borrow | sync | fail | panic/drop | reclaim |
|---|---|---|---|---|---|---|
| `submit` validation/preparation/spawn | owning (consumes `ExecutionInputs`) | none | none | `SubmitRejected` returns the exact unaccepted package | no worker exists yet; nothing retained | inputs back with caller; G1 rules |
| admitted, running | owning (worker) | none | leases per G1 acquired before each enqueue | execution error leads to Draining then `Failed` | worker panic leads to Draining/quarantine then `Failed` with a typed panic cause | only after retirement |
| `ExecutionHandle::wait` | none | none | blocks until Retired or Quarantined | returns explicit terminal `ExecutionOutcome`; quarantined resources are not recovered | n/a | per retired outcome; quarantine registry otherwise |
| handle drop before completion | none | none | none | n/a | detach: reaper retains owners and leases until retirement; completion is not cancelled | after retirement, by the reaper |
| cancellation request | none | none | none | n/a | cooperative: honored at pre-enqueue boundaries only; already enqueued device work is never revoked | after retirement |
| unobserved failure (detached, handle dropped) | none | none | none | reported through the documented runtime error sink/callback; never silent | n/a | after retirement |
| scoped submit rejected | shared borrow packaged for `'env` | no runtime borrow admitted | none | exact `ScopedReadInputs<'env>` returned with cause | no worker or partial bundle exists | caller-owned inputs unaffected |
| scoped admitted and `wait`ed | shared borrows of inputs for `'env` | input storage for `'env`; handle for `'s` only | leases per G1; `wait` observes post-retirement outcome | typed scoped outcome; borrowed descriptors retained, partial owned outputs private | panic enters draining/quarantine before outcome | fresh owners after retirement; borrowed slots never reclaimed by bundle |
| scoped handle dropped | none beyond admitted shared input borrows | inputs remain borrowed for `'env`; handle observation ends | none at drop | n/a | scope registry retains task, owners, and leases | only after scope join/retirement |
| scope exit with unobserved tasks | none | ends `'s`, not `'env` | joins and drains every admitted task | `ScopeExitError<R>` preserves closure result and aggregates failures/quarantine IDs | panic during closure still drains, reports secondary failures, and resumes original panic | after proven retirement; quarantine registry otherwise |

Detached `Failed`/`Cancelled` outcomes return the exact owning input package
only after all relevant event domains retire; normal shared, exclusive, and
extraction APIs are then available again. A `Quarantined` outcome returns no
owner: the runtime quarantine registry retains affected resources. Possibly
partial or uninitialized outputs stay private and are dropped after
retirement or retained by quarantine.

## G4. Method distribution

Rules (#1555, "Capability surface and method distribution"; #1559):

1. Read-only tensor behavior is implemented once, on `TypedTensorView`.
2. `TypedTensor` and `TypedTensorViewMut` delegate through cheap
   canonicalization:

   ```rust
   impl<T> TypedTensor<T> {
       fn as_view(&self) -> TypedTensorView<'_, T>;
       fn as_view_mut(&mut self) -> TypedTensorViewMut<'_, T>;
   }
   impl<'a, T> TypedTensorViewMut<'a, T> {
       fn as_view(&self) -> TypedTensorView<'_, T>; // reborrow
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

## G5. Raw handles and reclamation

- Provider handles (CubeCL handles, wgpu resources, CUDA device pointers,
  Metal resources) are lifetime or in-flight pins only. No handle can mint
  ownership, recover uniqueness, or authorize a write (I3). `Handle`
  cloneability inside a provider is acceptable exactly because handles carry
  no authority.
- Raw access is lease-bounded. The launch-session shape (#1563):

  ```rust
  let read_binding = resolved_read.acquire_device_read(endpoint)?;
  let write_binding = match unsafe { bind_raw_write(resolved_write, request) } {
      Ok(binding) => binding,
      Err((resolved, error)) => return Err((resolved, error)),
  };
  let completion = session.enqueue_borrowed(read_binding, write_binding)?;
  ```

  ```rust
  impl<'a> LaunchSession<'a> {
      fn enqueue_borrowed(
          self,
          read: UseLease,
          write: WriteBinding<'a>,
      ) -> Result<RetiredBorrowed<'a>, (Self, EnqueueError)>;
  }
  ```

  `RetiredBorrowed<'a>` is returned only after the device work and its lease
  have retired. A pending completion cannot carry `'a` into a runtime
  retirement record; detached submission uses the separate owning package.

  `resolved_read` and `resolved_write` are sealed capability wrappers, not
  public descriptor constructors. A read value is built only from `StorageRef`
  plus a validated descriptor; a write value is built only from `StorageMut`
  plus a validated, injective descriptor. Their fields and constructors remain
  in the audited ownership module. `session` owns only endpoint/event
  submission state: it cannot resolve a descriptor, select a provider, or
  supply a second span or claim. Storage authority is carried by the resolved
  value and then by the binding.

  ```rust
  struct TensorWrite<'a> {
      resolved: ResolvedWrite<'a>,
      _sealed: PrivateToken,
  }

  struct OwnedTensorWrite {
      owner: OwnedStorage,
      span: RootBoundSpan,
      _sealed: PrivateToken,
  }

  // `WriteBinding<'a>` is the G1 type: it owns this resolved operation and
  // its `UseLease` until enqueue admission and retirement finish.
  ```

  The binding retains the exclusive borrow (or the consumed `OwnedTensorWrite`
  package)
  until enqueue admission has either returned the unchanged package or moved
  it into the runtime retirement record. A static `UseLease` alone never
  permits reacquiring mutable access while work is pending.

  Bindings expose raw pointers only for the session lifetime; every binding
  revalidates domain/endpoint, span, layout arithmetic, alignment, access
  mode, and write injectivity (G1). A safe unleased
  `device_ptr(&Tensor) -> u64` does not exist; any escaping raw-pointer API
  is explicitly `unsafe` and documents its retirement obligations, and it
  stays provider-specific (no false parity).
- There is no shared-internal-to-exclusive transition. In particular, a
  provider pin, `Arc`, raw handle, lease, reference count, or completion
  token cannot be converted into `StorageMut`, an owner claim, or a write
  binding. Raw write binding starts with an exclusive capability already
  proven by Rust ownership. The sole audited unsafe boundary has this shape:

  ```rust
  unsafe fn bind_raw_write<'a>(
      capability: ResolvedWrite<'a>,
      request: ValidatedWriteRequest,
  ) -> Result<WriteBinding<'a>, (ResolvedWrite<'a>, AccessError)>;
  ```

  `ResolvedWrite` carries the exclusively borrowed owner, its matching claim
  and pin, the root-bound span, and the provider dispatch together, so the
  binder cannot be given an unrelated pin or provider receiver. The binder
  rechecks request key/range against that resolved capability before exposing
  raw state. Its safety proof covers only conversion of an existing exclusive
  capability into provider raw state; it does not establish uniqueness.
  The call-site inventory is enforced by a source-contract test. Strong
  counts may be diagnostics for leaked pins or retirement latency, never a
  precondition or proof for access authority (I3).
- Successful enqueue consumes the launch session. For detached execution,
  the runtime task continues to own the containing `OwnedStorage` and makes
  it unreachable to callers until event retirement; the encoding borrow may
  end, but no safe reborrow is possible. The completion lease and root pin
  move to the retirement record. Scoped submission in G3 is read-only. A
  direct borrowed write retires synchronously; the only asynchronous write
  package is the owning `OwnedTensorWrite` path. No other async write mode is
  permitted.
- Enqueue failure returns the unchanged session and all unadmitted
  capabilities in `(Self, EnqueueError)`. No raw binding or lease remains
  active, and the owning task/borrow can retry or recover without inference.
- Stream-ordered reclamation (I6): a retirement record holds the root-resource
  deallocator and all pins for a resource whose claim dropped with
  outstanding work. It never owns a deallocator for an individual subspan.
  Records are keyed by event domain. Completion tokens visible to users are
  never the sole owner of a lease, because users may drop them. Runtime
  drop drains only its own retirement queue before releasing its context.
  A failed drain quarantines (retains and reports); it never frees early and
  never releases a provider context that pending work may still use.

### Raw-binding state table

| Transition | cap | borrow | sync | fail | panic/drop | reclaim |
|---|---|---|---|---|---|---|
| request write from shared pin/handle/lease | shared or none (insufficient) | none | none | structured capability error; no binding or state change | n/a | unchanged |
| bind raw read | `StorageRef<'a>` | shared capability and resource for binding/session lifetime | dependencies from G1 | no binding; capability remains valid | session retains lease after admitted enqueue | after covering read retires |
| bind raw write | sealed `TensorWrite<'a>` or `OwnedTensorWrite` from a matching exclusive capability | exclusive claim+pin capability for binding/session lifetime | RAW/WAR/WAW dependencies from G1 | no binding; exact borrowed/owning package remains valid | no shared state can recover the consumed exclusivity; admitted lease retires normally | after covering write retires |
| enqueue validated bindings | consumes session; detached task retains owners | capabilities cannot be reused during admission or before retirement | provider enqueue/event registration | returns unchanged session and all capabilities in `(Self, EnqueueError)` | admitted leases and root pins move to runtime retirement even if completion handle is dropped | after event-domain retirement |
| drop last claim with provider pins in flight | owning claim | none | none at drop | n/a | claim, root pin, and leases enter retirement record | root deallocator only after all sibling claims and pins retire |
| retirement proof fails | provider-internal retirement record | none | attempted wait/poll | error reported | resource and context are quarantined | never speculatively |

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
| 9 (#1565) | detached vs scoped ownership, outcome recovery, detach vs cancel, extraction; G3 state tables kept current | common commands |
| 10 (#1566) | GPU quickstarts, provider matrix, namespace rustdoc; `# Errors` sections for every public `Result` API | common commands |
| 11 (#1568) | hardware evidence recorded in the test profile/worklog with candidate SHA | common commands |
| 12 (#1569) | `docs/guides/views-and-slicing.md` plus sidebar entry; `docs/getting-started/core-concepts.md`; README/tutorials; the rendered stale-language checker (`scripts/check-storage-docs.py`); the source-blind audit | common commands plus `python3 scripts/check-storage-docs.py --include-rendered` |
| 13 (#1567) | final worklog linking candidate SHA, scaffolding disposition, hardware/docs/audit reports; deletion of `HANDOFF-2026-07-25-tenferro-unification6-wip.md` and inbound references | common commands plus closure validation from #1567 |

## G7. AD value retention

The AD layer is the one workload where a single logical value has two
consumers by design: the caller and the tape (or checkpoint state). Today it
is built on shallow clones and `Arc<Tensor>`; under linear ownership it is
built on groups. Blanket replacement of shallow clones with `duplicate()` is
not an accepted migration.

### Ownership root

- Each autodiff context (eager tape, traced execution, checkpoint store)
  owns retained primal allocations through one retention group:

  ```rust
  struct TapeRetention {
      group: AllocationGroup,
      index: HashMap<ValueKey, ValueId>,
  }
  ```

- An `EagerTensor` (and any traced value handle) is a node handle plus a
  descriptor reference. Cloning a handle clones neither storage nor
  ownership: handles are read-only descriptor handles and can never mint a
  storage owner or a write capability. Handle types may remain `Clone`
  because they are not owner-like; the non-`Clone` rule (I1) applies to
  owners and capabilities.
- Every descriptor reference is registered against a generational `ValueId`.
  The liveness set includes the tape, checkpoint records, execution bundles,
  and every public handle clone. This bookkeeping may use reference counts,
  but those counts govern descriptor-slot liveness only; they never prove
  storage uniqueness or authorize a write. A slot is reclaimed or reused
  only after all liveness roots are gone, and reuse increments the generation
  so stale handles fail rather than resolving to a new value.
- Public handles pin the retention table and its group as liveness roots,
  without gaining access authority. Dropping a tape/context releases only
  its own root; owners needed by surviving public handles remain in the table
  until those handles are dropped or one last handle successfully extracts
  the value.
- Retention policy: an operation output is retained iff a registered
  VJP/JVP rule declares it needed for backward, or the user explicitly
  requests retention. Values nobody declares needed are not retained.
- When the caller wants a standalone owner of a retained value, the paths
  are exactly the G2 paths: `try_extract` after every other registered root
  (tape, checkpoint, execution, and sibling public handles) is absent, or an
  explicit duplicate (classified below). There is no hidden copy path.

### Public API replacement

The `Arc<Tensor>`-returning surface is replaced. No retention adapter appears
in any public or crate-private runtime boundary; the cutover lands directly on
the group-qualified descriptor model.

| Current | Replacement sketch | Semantics |
|---|---|---|
| `materialized(&self) -> Result<Arc<Tensor>>` | `value(&self) -> Result<ValueGuard<'_>>` | materializes if lazy, then exposes a borrowed `TensorView`; host bytes go through G1 guards |
| owned copy of a value | `duplicate_value(&self) -> Result<Tensor>` | explicit copy, reason `ExplicitDuplicate` |
| owned move of a value | `into_value(self) -> Result<Tensor, (Self, ValueStillReferenced)>` | extraction via G2; succeeds only after consuming the last public handle and when tape, checkpoint, execution bundle, and sibling-handle liveness roots are absent; failure returns the handle |
| backward result `Vec<Arc<Tensor>>`, `GradSlot = Arc<Mutex<Option<Arc<Tensor>>>>` | `Gradients` bundle (a G2 group specialization) with `grad(&self, key) -> Option<TensorView<'_>>` and `take_grad(&mut self, key) -> Option<Tensor>` | one owner per gradient allocation; extraction when unique |
| traced attached-data maps `HashMap<ValueKey, Arc<Tensor>>` | `ExecutionInputs` bindings over group descriptors (G3) | no shared owners in the runtime boundary |

### Checkpoint semantics

- Boundary values (checkpoint region inputs and outputs) are retained as
  descriptors in the checkpoint group.
- Interior values are deliberately discarded at record time; checkpointing
  must not accidentally retain every intermediate. A contract test asserts
  the checkpoint adds no liveness root for an interior value. Its allocation
  is released after the boundary is recorded only when no tape, execution,
  or external handle root independently keeps it live.
- Backward recomputation executes the stored subgraph and produces fresh
  owners; its allocations are classified `CheckpointRecomputeOutput`,
  distinct from retention (which allocates nothing) and from explicit
  duplicates.

### Reinterpretation and aliases

- A retained complex/real reinterpretation of a retained value is another
  descriptor of the same slot in the same group (G2 duplicate-descriptor
  semantics), never a second owner.
- Mutable reinterpretation requires an exclusive or owning capability. While
  any tape, checkpoint, execution bundle, or sibling handle descriptor
  references a span, the owner lives in the retention group, so no caller can
  hold the owner: consuming or mutable reinterpretation of that allocation
  is unreachable, and `try_extract` fails with a typed
  `ValueStillReferenced` reason (`Tape`, `Checkpoint`, `Execution`, or
  `SiblingHandle`). This exclusion is structural (borrow/owner placement),
  verified by compile-fail plus runtime extraction and stale-generation
  tests.

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
  physical data movement emits exactly one copy event, whether or not a new
  destination was allocated. Descriptor aliasing, metadata-only outputs,
  and public-handle cloning emit neither event.
- Acceptance for an AD scenario (forward plus backward, with and without
  checkpointing): every observed copy and every observed allocation carries
  a reason from its own enum and each ledger matches the scenario's expected
  multiset. Copies and allocations attributable to retention are therefore
  both exactly zero.
- Aggregate pre-migration versus post-migration copy counts are not an
  acceptance criterion.

### Atomic cutover and provider bridge

Public host ownership (#1559) and final detached/scoped runtime plus direct
group-based AD retention (#1565) form one atomic promotion cohort. They land
the final `AllocationGroup`, lease, retirement, and descriptor-liveness
semantics together. There is no interim AD-retention adapter, minimal
consuming-submit bridge, or pre-retirement synchronization adapter.

Exactly one typed, inventoried crate-private provider bridge may keep an
unmigrated accelerator provider buildable. The bridge is implemented against
the final root-bound claim, access, authority-free lease, and retirement
contracts; it cannot expose an owner-like shared handle, mint a claim, or
authorize a write. #1563 and #1564 remove the bridge as their provider paths
are migrated. If the bridge cannot satisfy the final contract, provider
migration moves earlier instead of adding a second seam.

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
| record op output into tape | owning (retention table takes the output owner into its group) | none | none | op error: no retention entry | unwind releases the tape root but preserves independent roots | after the last liveness root, via G1/G2 |
| clone `EagerTensor` handle | none (descriptor liveness only) | none | none | n/a; clone does not resolve storage | clone registers another liveness root; drop unregisters it | descriptor slot only after all roots disappear; never storage authority |
| drop last descriptor handle while tape/checkpoint retains | none | none | none | n/a | public-handle root disappears; retention root remains | not until all remaining roots disappear |
| tape/context drop while public handle remains | owning tape root only | none | outstanding work follows G1 | n/a | table/group pin transfers no authority; public-handle root remains | not until last independent root disappears |
| tape retains/releases descriptor | owning tape bookkeeping / none on release | none | none | invalid or stale ID is typed error | update is atomic; release cannot invalidate sibling roots | only after all roots disappear |
| checkpoint retains/releases boundary descriptor | owning checkpoint bookkeeping / none on release | none | none | invalid or stale ID is typed error | no interior root is added implicitly | only after all roots disappear |
| `value()` guard | shared | tape group (shared) for guard lifetime | G1 host-read rules if host bytes requested | error, tape unchanged | guard drop ends borrow | n/a |
| backward execution | shared reads of retained descriptors; new owners for grads | tape group shared during execution | G3 rules | typed failure, tape unchanged, grads dropped after retirement | per G3 panic row | grads owned by `Gradients` bundle |
| `take_grad` | exclusive on `Gradients` | none after return | none | `None`/typed reason, bundle unchanged | n/a | extracted owner per G1 |
| `into_value` while any other root remains | owning attempt (consumes one handle) | none | none | `ValueStillReferenced` identifies tape/checkpoint/execution/sibling handle; consumed handle is returned or remains usable | no liveness root is lost on failure | n/a |
| `into_value` as last root | owning (consumes last handle and group slot) | none | none | stale/invalid descriptor leaves group unchanged | generation tombstoned; owner moves exactly once | extracted owner per G1 |
| stale-handle access after tombstone/reuse | none | none | none | deterministic `StaleValueId`; no storage is touched | no state change | unchanged |
| reuse descriptor slot | owning group bookkeeping | none | none | n/a | generation increments before publication | new slot follows its own roots; old IDs remain stale |
| checkpoint record | owning (boundary owners/descriptors into checkpoint group) | none | none | error: no partial checkpoint | no checkpoint root is added for interiors; independent roots remain valid | boundary owners after last liveness root |
| checkpoint recompute (backward) | shared reads of boundary; fresh owners for recomputed values | checkpoint group shared | G3 rules | typed failure after retirement | per G3 | recomputed owners dropped after use |
| tape drop | owning tape root | none | retirement per G1 for owners with no other roots and in-flight work | n/a | quarantine path per G1; independent handle/checkpoint roots remain | after retirement and the last liveness root, exactly once |

## Contract test index

Each gate's clauses are enforced by tests owned by the listed phases. The
phase issues carry the full inventories; this index is the cross-reference.

| Gate | Enforcement | Owning phases |
|---|---|---|
| G1 ordering, guards, revalidation, retirement | deterministic fake-timeline transition tests; claim provenance/split/overlap and exactly-once root-deallocator tests; compile-fail (guard across consuming submit, write guard from shared); corrupt-descriptor rejection at map and enqueue; immediate-drop-after-enqueue; quarantine poisoning; Miri on host guard slices | #1560, providers in #1563/#1564 |
| G2 group, splitting, extraction | N-way split cases (N=0,1,>2, empty, reverse-stride, overflow); permutation-independence property tests; group-qualified stale-generation tests; compile-fail (root access while children live); extraction counters | #1561 |
| G3 submission terminal semantics | rejection carriers return identical allocation keys; hybrid scoped identity/metadata/new-output bundles; borrowed-output extraction rejection; scoped result bounded by `'env` but not `'s`; explicit scope-exit and quarantine outcomes; cancellation/panic/detach/unobserved-error suites; compile-fail (scoped handle escape, host guard across submit) | #1565, hardware in #1568 |
| G4 method distribution | API-parity contract with one canonical method list; compile-fail (no `Clone` on owners/capabilities); source scan (no mutable owner projections) | #1557 harness, #1559 |
| G5 raw handles, reclamation | fake backend proving internal `Arc` clones cannot write or mint owners; sealed `TensorWrite` construction and audited raw-write binder inventory proving `StorageMut` input; enqueue-failure capability recovery; retirement/quarantine tests; source scans (no shared-to-exclusive transition, no safe unleased pointer) | #1558, #1563, #1564 |
| G6 documentation | rendered stale-language checker; doctests; tutorial-code checks; source-blind audit | #1569, #1567 |
| G7 AD retention | separate reason-classified copy/allocation counters (zero retention events in both); generational stale-handle and all-liveness-root extraction tests; checkpoint interior-release test; mutable-reinterpret exclusion; CPU plus designated async accelerator lanes | #1557 contract, atomic #1559/#1565 cutover, #1568 evidence |

## Relationship to phase issues

- #1556 and #1557 are independent roots of the canonical DAG; #1557 owns the
  v2 ledger and its executable RED/green gate.
- #1558 owns the root pin, non-`Clone` claim, and the single typed provider
  bridge. #1560 owns access/retirement. #1561 owns groups and generational
  descriptors. None waits for the public host cutover.
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
