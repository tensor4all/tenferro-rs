# Storage Ownership Contracts

Normative contract document for the storage ownership redesign tracked by
issue [#1555](https://github.com/tensor4all/tenferro-rs/issues/1555). This
document is the Phase 1 deliverable of
[#1557](https://github.com/tensor4all/tenferro-rs/issues/1557): it turns the
seven design gates in #1555 into contracts precise enough to review and test
implementation PRs against.

Authority and change control:

- The #1555 issue body owns the named bullets in its "Fixed architecture"
  section. This document owns the detailed contracts. The phase issues (#1556
  through #1569) own work decomposition and verification inventories.
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
  shared/exclusive capability, descriptor, claim, prepared access, or
  retirement.
- Owner claims, Rust access capabilities, direct root lifetime, and
  descriptors are separate concepts. No implementation may recover ownership
  or write authority from a reference count, downcast, raw handle, allocation
  ID, or provider-specific shortcut.
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

- **Allocation**: one physical memory root and its claims. A requested byte
  range is metadata (optionally carrying a domain-qualified key), never an
  access capability or proof of ownership. A `RootBoundSpan` is the checked
  range form and carries the exact `RootResourceIdentity` from which it was
  derived.
- **Owner**: the unique, non-cloneable ownership token for an allocation span
  (`OwnedStorage`, or a tensor/group wrapping it); this is the umbrella's
  one-owner rule.
- **Capability**: the right to access storage, expressed as Rust
  ownership/borrows: shared (`StorageRef`, views), exclusive (`StorageMut`,
  mutable views), or owning (consuming APIs); Rust borrowing is the write
  authority.
- **Descriptor**: a typed interpretation (dtype, layout, placement) referring
  to an allocation slot. Descriptors never own storage.
- **Group**: one or more owners plus descriptors (`AllocationGroup`), the only
  representation for "one allocation, many logical values".
- **Guard**: a borrow-carrying value granting host byte access to a validated
  span (`HostReadGuard`, `HostWriteGuard`).
- **Endpoint**: an engine/device access point (`AccessEndpoint`). Distinct
  from `AllocationDomainId`, which is allocation identity (#1555, "Identity
  vs endpoints").
- **Retirement**: the point when all provider events covering an access have
  completed and retained resources may be released.
- **Prepared access**: the result of pairing a Rust capability with retained
  construction-time descriptor proofs and completing access-time provider
  mapping/synchronization. It carries the `CheckedLayout` and Rust borrow
  required by the access. Its host variant exposes the hot loop through the
  `iter_contiguous*` typed-slice path or the `iter_strided*` prepared-cursor
  path; its device variant retains provider-ready binding state and exposes no
  host pointer or iterator.
- **Completion-unproven retention**: the typed-error path used when a
  provider cannot prove completion. A provider-private `UnprovenRetirement`
  owns provider retirement bindings, the event, `Arc<RootResource>`, and
  provider context until completion is proven or those resources are
  intentionally made a permanent leak; this is
  not a quarantine state.

State-table columns. Every state-transition row in this document answers the
six review-checklist questions from #1557, abbreviated as:

| Column | Meaning |
|--------|---------|
| cap | capability required (shared / exclusive / owning / none) |
| borrow | what is borrowed and for how long |
| sync | provider synchronization performed (waits are documented synchronization points, never copies) |
| fail | return on failure (ownership must be recovered or provably retained) |
| panic/drop | state retained if the caller panics or drops mid-operation |
| reclaim | when reclamation of the affected allocation becomes legal |

Error conventions: all failures are structured errors carrying operation,
requested range/ids, and resolved span identity, without raw addresses.

### P2 identity and span correction boundary

P2 constructs the private identity/span vocabulary used by later owners and
prepared descriptors. `RootResourceExtent` checks every half-open range end
with checked arithmetic before alignment or containment decisions.
`RootResourceIdentity` pairs one private root provenance ID with that exact
extent. `RootBoundSpan` can be created only from that identity and retains the
identity in its value; equal extents from two roots therefore cannot be
interchanged as resolved spans.

Compound relative-range validation checks, in order, the root end, relative
end, base-plus-relative offset, and child end. Only after those checks does it
evaluate containment and alignment. This makes a relative-range overflow win
over a simultaneous malformed-alignment condition without adding runtime
recovery or repeated access validation.

Operation request metadata uses the single sum type
`RequestedIdentity::{Raw, Keyed, Rooted}`. It is untrusted and may differ from
the resolved value; the resolved side retains a `RootBoundSpan`. No diagnostic
identity contains a pointer, provider handle, or write authority.

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

Contract revision is distinct from promotion. `registry.revision` is a
positive monotonic integer (the original field-less v2 registry is revision
1). A candidate may advance it by exactly one while preserving the complete
unit/gate/edge/cohort topology, obligation membership, and every tagged state.
During that revision-only transition, active obligation identities remain
immutable; only still-deferred rows may revise their gates, artifact, or
command to reflect the reviewed implementation contract. Revision and
promotion cannot occur in the same candidate. Once an obligation is active,
later contract revisions cannot rewrite its evidence. This is the explicit
design-amendment path; it is not an exception to promotion immutability.

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
artifact = { id = "artifact-prepared-validation-boundary", kind = "validation-boundary-test", path = "crates/tenferro-tensor/tests/storage_prepared_validation.rs" }
command = { id = "cmd-prepared-validation-boundary", kind = "cargo-test", argv = ["cargo", "test", "-p", "tenferro-tensor", "--test", "storage_prepared_validation"], cwd = ".", path_args = [], artifact_id = "artifact-prepared-validation-boundary" }
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
  prepared read/write borrow shape and exact capability recovery before
  admission;
- P3/G1+G4: a compile contract using the repository compile-test harness for
  non-`Clone` owners, exclusive write preparation, allocation-free views, and
  host guards that cannot escape their borrow;
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
write safety; trybuild, Miri, property, invalid-constructor/input-boundary, and
provider tests exercise dynamic boundaries before checked descriptors or
prepared access are published; source inventories record deletion drift; and
the source-blind documentation audit checks stale public language. None of
these is allowed to manufacture ownership proof from an allocation ID or a
lock. These tests do not introduce a post-construction corruption hook or
repeated map/enqueue validation protocol.

## G1. Span access and retirement

G1 defines the permanent correctness boundary for storage access. Tenferro is
scientific-computing software, not a security boundary. This gate protects Rust
aliasing and memory safety, checked layout arithmetic, numerical interpretation,
provider compatibility, and asynchronous device lifetime. It does not defend
against a malicious maintainer, runner, provider implementation, or process
that can already execute arbitrary code in the repository.

### Ownership and capability types

The physical resource has one direct lifetime graph. There is no second
authority, liveness table, or reconstructable identity protocol.

```rust
struct RootResource {
    provider: Arc<ProviderContext>,
    allocation: ProviderAllocation,
    capacity_bytes: usize,
    diagnostics: AllocationDiagnostics,
}

struct OwnedSpanClaim {
    root: Arc<RootResource>,
    byte_range: Range<usize>,
}

struct OwnedStorage {
    claim: OwnedSpanClaim,
}

struct StorageRef<'a> {
    root: &'a Arc<RootResource>,
    byte_range: Range<usize>,
}

struct StorageMut<'a> {
    root: &'a mut Arc<RootResource>,
    byte_range: Range<usize>,
}

struct RootBoundSpan {
    byte_range: Range<usize>,
    dtype: DType,
}
```

`OwnedStorage` and `OwnedSpanClaim` are non-`Clone`. `Arc<RootResource>` is
cloneable only where direct physical lifetime must survive asynchronous work or
read-only retained records; cloning it grants neither an owner nor write
authority. Allocation IDs and diagnostics identify observations only.

`StorageRef` is created from a shared borrow of a matching claim. `StorageMut`
is created only from an exclusive borrow of a matching claim or from a freshly
allocated output that has not escaped. Neither can be constructed from an ID,
raw handle, event, provider context, reference count, or read-only handle.
Splitting an owned claim consumes it and creates checked disjoint child claims.
Temporary mutable splitting distributes one existing exclusive borrow and does
not change physical ownership.

Provider import is the audited unsafe boundary that constructs the initial
root and full-range claim. Its safety contract proves that the allocation is
valid for its reported capacity, deallocation contract, alignment, provider,
and device context. Safe code may narrow or split that claim but never widen it.

### Validate once, then traverse

Safe tensor/group construction validates the descriptor once and retains the
result. Every host or device access then consumes a capability already paired
with that checked descriptor into a prepared object:

```rust
enum CheckedLayout<R: TensorRank> {
    Contiguous {
        element_range: Range<usize>,
    },
    Strided(CheckedStrided<R>),
}

struct CheckedDescriptor<R: TensorRank> {
    span: RootBoundSpan,
    layout: CheckedLayout<R>,
    placement: Placement,
}

struct CheckedInjectiveDescriptor<R: TensorRank> {
    descriptor: CheckedDescriptor<R>,
    injectivity: WriteInjectivityProof,
}

struct CheckedRead<'a, R: TensorRank>(private::CheckedReadBundle<'a, R>);
struct CheckedWrite<'a, R: TensorRank>(private::CheckedWriteBundle<'a, R>);

enum AccessTarget {
    Host,
    Device,
}

enum PreparedRead<'a, T, R: TensorRank> {
    Host(PreparedHostRead<'a, T, R>),
    Device(PreparedDeviceRead<'a, T, R>),
}

enum PreparedWrite<'a, T, R: TensorRank> {
    Host(PreparedHostWrite<'a, T, R>),
    Device(PreparedDeviceWrite<'a, T, R>),
}

fn prepare_read<'a, T, R: TensorRank>(
    checked: CheckedRead<'a, R>,
    target: AccessTarget,
) -> Result<PreparedRead<'a, T, R>, (CheckedRead<'a, R>, AccessError)>;

fn prepare_write<'a, T, R: TensorRank>(
    checked: CheckedWrite<'a, R>,
    target: AccessTarget,
) -> Result<PreparedWrite<'a, T, R>, (CheckedWrite<'a, R>, AccessError)>;
```

Before any `CheckedDescriptor` is published, its safe constructor validates:

1. checked shape, stride, offset, and byte-range arithmetic;
2. logical bounds and the exact root-bound span;
3. dtype size and interpretation;
4. required alignment;
5. storage and provider compatibility; and
6. non-overlapping element addresses for a write layout.

Views and group records retain these proofs. Slicing or reinterpretation
constructs a new checked descriptor and validates only the newly derived
arithmetic and invariants. `prepare_read` and `prepare_write` do not recompute
them: they consume the checked capability/descriptor pairing and perform only
provider operations that cannot be established until access time, such as
mapping, synchronization, and timeline admission. Provider lifetime is reached
through the borrowed root; preparation performs no provider-context `Arc`
clone. Preparation failure returns the unchanged checked pairing and a typed
error. Any temporary host mapping is released before returning. No partially
prepared state is published.

`CheckedRead` and `CheckedWrite` are opaque module-private bundles, not structs
with independently constructible public or crate-wide fields. There is no
`new(access, descriptor)` function. A tensor/view method creates the bundle by
moving or borrowing its co-located storage capability and checked descriptor;
an `AllocationGroup` method creates it only after resolving the descriptor's
local `AllocationSlot` to that same occupied owner entry. Those are the only
safe constructors. Consequently a descriptor cannot be paired with another
root without entering the audited unsafe storage module, and ordinary access
does not need a root-identity comparison or repeated range validation.

Host mapping or device preparation consumes the checked object and publishes
the matching `Prepared*::Host` or `Prepared*::Device` variant. The device
payload retains the checked capability/layout plus the provider's opaque
prepared mapping or binding state; it does not contain or construct a host
guard. Subsequent binding and enqueue consume that device payload. None of
these operations accepts a replacement descriptor, range, key, provider, or
access mode or repeats the static checks above. This is an API-shape
requirement, not a convention.

### Contiguous and strided hot paths

Host preparation selects the traversal representation once. A contiguous
prepared access exposes the already checked typed range as a slice and an
`iter_contiguous()`-equivalent slice iterator. A strided prepared access owns a
precomputed incremental plan:

The authoritative `PreparedRead`/`PreparedWrite` enums select exactly one host
or device state. Within the host variant, `PreparedHostRead` and
`PreparedHostWrite` select exactly one contiguous or strided traversal state.
Their payloads and traversal methods are specified once in G4 below. Device
payloads retain `CheckedLayout` for launch/binding but expose no host pointer,
slice, or iterator. There is no optional second traversal surface.

The exact names may change in the owning phase, but equivalent code generation
and verification properties are mandatory. Contiguous inner loops perform only
ordinary typed slice access. Strided `next()` performs only loop termination,
typed pointer access, and necessary precomputed stride/carry increments. It
does not resolve storage, dispatch through a provider, check bounds, inspect
dtype, map, synchronize, allocate, decode a flat index into coordinates, or
repeat layout arithmetic. Fixed-rank plans remain monomorphized; dynamic-rank
cursor state is allocated or initialized once outside the element loop.

`CheckedInjectiveDescriptor` retains descriptor-level write injectivity, and
`CheckedInjectiveStrided` carries the corresponding traversal proof used by
the private mutable strided iterator to yield each writable element at most
once. The iterator owns the sole mutable borrow of its prepared guard.
Independent booleans such as `is_checked`,
`is_mapped`, `is_contiguous`, and `is_writable` do not encode lifecycle state;
the prepared enum/newtype variants do.

Ordinary `as_view()` and `as_view_mut()` only reborrow owner storage and layout
metadata. They are O(1), allocation-free, and perform no provider operation,
reference-count increment, synchronization, transfer, or materialization.

### Provider use and retirement

Host/CPU borrowed access is synchronous when the provider guarantees that all
work and temporary mapping retire before the call returns. An asynchronous
CUDA, WebGPU, or Metal operation uses detached owning submission. It consumes
prepared bindings into a task-owned retirement record:

```rust
struct RetirementRecord {
    event: ProviderEvent,
    bindings: Box<[ProviderRetirementBinding]>,
    roots: Box<[Arc<RootResource>]>,
    provider: Arc<ProviderContext>,
}
```

Enqueue consumes each `DeviceRead`/`DeviceWrite` into a
`ProviderRetirementBinding` that owns any mapping, reservation, or raw-binding
lifetime the provider requires after enqueue. The record owns every binding,
event, root, and provider context until completion is proven. Dropping a
user-visible completion handle only detaches
observation; the worker or provider reaper retains the record. After proven
completion, the record releases bindings, event, and root/context references
exactly once and publishes a completed or typed failed outcome.

If completion cannot be proven, the public outcome contains diagnostics and no
owner. A provider-private record permanently retains the bindings, event,
roots, and provider context because neither binding/event destruction nor
memory reuse is known to be safe. This is a terminal leak-for-soundness case,
not a recoverable state.
There is no retry API, global recovery table, or safe extraction path from it.

Soundness does not depend on `Drop` or a callback running: `mem::forget` may
reduce liveness, but cannot create writable aliases or early reclamation. Panic
is handled at the existing thread/task/FFI boundary. After possible enqueue it
drains to a proven retired failure or the same ownerless
`CompletionUnproven` outcome; G1 introduces no panic-catching access protocol.

### Transition contract

| Transition | capability | borrow | synchronization | failure | panic/drop | reclaim |
|---|---|---|---|---|---|---|
| owner -> shared view | shared | tied to owner borrow | none | infallible | borrow rules remain authoritative | owner/root lifetime |
| owner -> mutable view | exclusive | tied to exclusive owner borrow | none | infallible | borrow rules remain authoritative | owner/root lifetime |
| checked shared pairing -> prepared host/device read | shared | capability and checked descriptor carried by target variant | retained static proofs; provider may map/synchronize once | exact unchanged checked pairing plus typed error | temporary provider state released | owner/root lifetime |
| checked exclusive pairing -> prepared host/device write | exclusive | capability and checked injective descriptor carried by target variant | retained static proofs; provider may map/synchronize once | exact unchanged checked pairing plus typed error | temporary provider state released | owner/root lifetime |
| prepared host access -> synchronous return | shared/exclusive | guard lives through call | provider work retires before return | typed retired error | no work survives return/unwind | after guard and owner release |
| prepared device access -> pre-admission rejection | owning | no caller borrow escapes | no enqueue occurred | exact unchanged package | no retirement record exists | caller retains owners |
| possible enqueue -> draining | task-owned | no caller borrow | event domains drain | no immediate owner return | worker/reaper owns retirement bindings, event, roots, context | not yet |
| draining -> retired completed/failed | task-owned | none | completion proven | typed result; owners only after retirement | record releases bindings/event/roots/context once | normal root lifetime |
| draining -> completion unproven | provider-private | none | completion not proven | diagnostics only, no owner | permanently retains bindings, event, roots, context | never by this outcome |

### Acceptance evidence

G1 is accepted only with executable evidence for all of the following:

- compile-fail tests reject `Clone` for owners/claims, write preparation from a
  shared borrow, overlapping mutable splits, and prepared guards escaping their
  borrow;
- property/Miri tests cover empty, singleton, reverse-stride, noncontiguous,
  overflow, out-of-span, misaligned, wrong-dtype, and non-injective layouts;
- fake-provider counters prove validation, provider resolution, mapping,
  synchronization, and dispatch counts are independent of element count;
- source/API contracts prove binding and enqueue accept only prepared access and
  no replacement descriptor/range/provider/access mode;
- contiguous release benchmarks/codegen show slice-equivalent loops, and
  strided structure checks show only typed access plus stride/carry increments;
- `as_view()` and `as_view_mut()` tests prove zero allocation, zero provider or
  storage clone/refcount work, and no dynamic layout clone;
- event tests cover immediate handle drop, successful completion, execution
  failure, panic after possible enqueue, and completion-unproven retention of
  provider retirement bindings, event, roots, and provider context;
- CPU, CUDA, WebGPU, and Metal use the same capability and retirement contract,
  with explicit unsupported errors where a provider cannot offer a mode.


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
  permitted by the no-hidden-copy rule.
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
record retains the consumed retirement bindings, event, `Arc` roots, and
provider context for that outcome. No public result can recover them.

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
owns its inputs, provider retirement bindings, events, roots, and provider contexts until
a terminal outcome.

A detached worker or provider panic is contained at the existing worker,
thread, or FFI boundary and enters `Draining`. If completion is proven,
`RetiredFailed` returns the exact input owners with a typed panic cause. If
completion cannot be proven, `CompletionUnproven` returns only its typed cause
and diagnostics while a provider-private permanent record retains the
retirement bindings, event, `Arc` roots, and provider context. No public
recovery path returns those owners.

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
| `Admitted` -> `Running` | owning (worker) | none | prepared bindings cross the enqueue-capable boundary and become provider retirement bindings | post-admission preparation or enqueue failure enters `Draining` | handle drop detaches observation; reaper retains owners, retirement bindings, events, roots, and contexts | only at a terminal outcome |
| `Running` -> `Draining` | owning (worker/reaper) | none | all enqueued work and event domains drain | execution failure or worker/provider panic enters `Draining` | panic is typed at the existing worker/thread/FFI boundary; reaper retains ownership | not yet |
| `Draining` -> `Retired(Completed)` | owning (worker/reaper) | none | completion proven | returns `ExecutionBundle` | n/a | returned bundle follows G1 |
| `Draining` -> `Retired(Failed)` | owning (worker/reaper) | none | completion proven | returns exact input owners with the typed execution or panic cause | n/a | returned owners follow G1 |
| `Draining` -> `CompletionUnproven` | no public owner; provider-private retention | none | completion cannot be proven | returns no owner, only the typed completion or panic cause and diagnostics | permanent record retains retirement bindings, event, `Arc` roots, and provider context | retained permanently because completion and safe binding/event destruction are unproven |
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
   available only through operations that validate the resulting
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
offset/stride plan. Backend execution prepares and binds once per launch.
Static-rank traversal remains monomorphized and eligible for loop unrolling;
dynamic-rank support must not route every typed element through opaque
per-element dispatch. The release codegen artifact
`p10-static-rank-codegen` records at least one contiguous fixed-rank loop and
must show a slice-equivalent inner loop without storage/provider abstraction
work.

Phase 4 implements the authoritative G1 `CheckedLayout`, `PreparedRead`, and
`PreparedWrite` hierarchy. The host variants use these nested traversal
variants and payloads; the device variants are shown afterward. This is one
preparation hierarchy, not a second host-preparation surface:

```rust
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

struct PreparedDeviceRead<'a, T, R: TensorRank> {
    checked: CheckedRead<'a, R>,
    provider_state: ProviderPreparedRead<'a>,
    _scalar: PhantomData<T>,
}

struct PreparedDeviceWrite<'a, T, R: TensorRank> {
    checked: CheckedWrite<'a, R>,
    provider_state: ProviderPreparedWrite<'a>,
    _scalar: PhantomData<T>,
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

These view methods are public convenience wrappers around the private G1
`prepare_read`/`prepare_write` transition, not another preparation layer. They
create the opaque checked bundle from the view's co-located capability and
descriptor, call that transition, and reconstruct the unchanged view when it
returns the checked bundle with an error. Both paths publish the same
`PreparedRead::Host`/`PreparedWrite::Host` payloads and return those payloads
without introducing another lifecycle state.

`PreparedDeviceRead` and `PreparedDeviceWrite` are the device payloads of the
same G1 hierarchy. Their opaque `ProviderPrepared*` state represents only the
provider mapping/binding work selected for the descriptor's placement. The
embedded checked bundle retains the capability and `CheckedLayout`; these
payloads neither map host memory nor contain a `HostReadGuard` or
`HostWriteGuard`. CUDA, WebGPU, and Metal therefore prepare device bindings
without manufacturing a host-visible access path.

`RankIndex<R>` is the rank-preserving cursor representation: inline for fixed
rank and initialized once outside iteration for dynamic rank.
`CheckedStrided<R>` owns the checked start offset, extents, strides, element
count, and incremental carry plan; it contains no provider or storage receiver.
`CheckedInjectiveStrided<R>` is constructible only after the write-injectivity
proof and otherwise has the same traversal data. View/group construction has
already retained the checked shape/stride/offset arithmetic, bounds, exact
root-span containment, alignment, provider compatibility, and write
injectivity required here. The fallible `prepare_host*` constructor consumes
that checked capability/descriptor pairing without recomputing those proofs;
it performs only access-time mapping and synchronization before publishing
`PreparedHostRead` or `PreparedHostWrite`. Failure releases any temporary
provider mapping and returns the unchanged view with a typed `AccessError`; no
prepared object or iterator exists on failure. The constructor selects exactly
one `PreparedHost*` enum variant from the retained `CheckedLayout`. Matching
that variant performs no validation or provider work.

`as_slice*` and `iter_contiguous*` perform only typed slice access after one
range extraction outside the loop. `PreparedStridedIter*::next` performs only
typed pointer/slice access, the necessary incremental stride/carry updates,
and loop termination. It does not decode a flat index into coordinates or
repeat bounds, layout, span, alignment, capability, provider, map, or
synchronization checks. The `PreparedRead`/`PreparedWrite`, `PreparedHost*`,
and `CheckedLayout` enums are the state authorities; independent booleans such
as `is_checked`, `is_contiguous`, `is_mapped`, and `is_writable` must not
encode these states.

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
tests with a source-contract inventory proving static descriptor proofs are
retained rather than recomputed, access-time provider work precedes prepared
object construction, iterator bodies contain only the permitted typed access
and increments, and no boolean fields duplicate enum state. P10 repeats
the loop-boundary structural proof over the final normalized API.

Rank-changing reinterpretation is separate from ordinary views. Phase 6 must
define each operation's result-rank policy explicitly and test it under
`p6-reinterpret-rank-policy`. A stable-Rust limitation in expressing a type
level result such as `N + 1` may require a dynamic result or an explicit
caller-selected result rank for that operation only; it must never force
rank-preserving view, slice, or traversal APIs to erase `R`.

Every reinterpretation is a descriptor operation over the same physical root,
not a copy or a new ownership path. Its sealed scalar-pair rule validates byte
divisibility, alignment, shape/stride/offset arithmetic, and resulting exact
root-bound span before publishing the new checked descriptor. A consuming
owner operation preserves and returns the original owner on failure; a
read-only view remains tied to its source borrow. The resulting descriptor
retains the same root `Arc`, allocation diagnostics, provider placement, and
device/managed-resource state. Mutable reinterpretation additionally requires
an exclusive borrow and an injective resulting layout, and is unavailable
while retained aliases prevent that exclusive path. `p6-reinterpret` proves
same-root preservation, zero allocation/copy, numerical element mapping, and
typed failure recovery; the rank-policy obligation is supplementary rather
than the whole Phase 6 contract.

The v2 ledger carries these executable obligations:

| Obligation | Phase | Artifact and proof |
|---|---|---|
| `p1-element-access-baseline` | P1 | active measured direct-slice/contiguous/strided report and verifier; later candidates use its exact Git commit and repository-relative path, subject to P10 compatible-environment comparison |
| `p3-static-rank-preservation` | P3 | compile/API contract for owner, immutable view, and mutable view preserving `R` |
| `p3-as-view-zero-allocation` | P3 | warmed allocator/refcount/provider-clone/layout-clone counters plus borrow-only source contract for owner/view-mut reborrows, including dynamic rank |
| `p4-traversal-resolution-counts` | P4 | fake provider counters proving prepare/map/bind/dispatch counts are independent of element count |
| `p4-prepared-access-api` | P4 | compile/runtime and source contract for typed failure, enum-authoritative preparation, contiguous slice/iterator access, and incremental strided iteration |
| `p6-reinterpret` | P6 | same-root, zero-copy, numerical/layout, exclusivity, and typed failure-recovery contract for every sealed scalar pair |
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

G5 consumes the exact `PreparedDeviceRead` and `PreparedDeviceWrite` variant
payloads of the G1 `PreparedRead` and `PreparedWrite` hierarchy; it does not
redefine a second preparation surface. Descriptor construction has already established the static
bounds/layout/dtype/root-span/alignment/storage/provider proofs and write
injectivity. Preparation consumes those retained proofs with the matching Rust
capability and performs only access-time provider mapping, synchronization, or
timeline admission. An immutable owner or view can yield
`PreparedRead::Device`; only an owner or mutable view borrowed exclusively can
yield `PreparedWrite::Device`.

```rust
fn bind_read<'a, T, R: TensorRank>(
    prepared: PreparedDeviceRead<'a, T, R>,
) -> Result<DeviceRead<'a, T, R>, (PreparedDeviceRead<'a, T, R>, BindError)>;

fn bind_write<'a, T, R: TensorRank>(
    prepared: PreparedDeviceWrite<'a, T, R>,
) -> Result<DeviceWrite<'a, T, R>, (PreparedDeviceWrite<'a, T, R>, BindError)>;
```

The checked access already borrows the root that owns its provider context.
Consequently binding needs no provider argument, additional provider lifetime,
or provider-context `Arc` clone. Detached execution owns the roots in its task
package, so the same relationship remains valid through event retirement.

Binding consumes provider-ready prepared access. Neither binding nor enqueue
repeats these checks or compares a second request, key, or range; those values
are carried by the prepared object, and the binding/enqueue signatures accept
no replacement values. The host/device and host traversal variants plus
`DeviceRead` and `DeviceWrite` are distinct sealed states, not boolean state
combinations.

There is no shared-to-exclusive conversion. A provider handle, `Arc`,
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
returns the exact unchanged `PreparedDeviceRead` or `PreparedDeviceWrite` because binding
precedes the enqueue-capable call.

Once enqueue may have happened, the task retains the package and bindings and
enters G3 `Draining`; an immediate error never returns them. On a post-boundary
failure, ownership returns only as G3 `RetiredFailed` after completion is
proven. If completion cannot be proven, `CompletionUnproven` returns diagnostics
without owners while the provider-private permanent record retains the
retirement bindings, event, roots, and provider context.

A borrowed operation is optional: if offered, it is synchronous through
retirement and is supported only by a provider that guarantees no asynchronous
work survives unwind. After its enqueue-capable call, it returns bindings only
inside `RetiredBorrowed::Completed` or `RetiredBorrowed::RetiredFailed` after
retirement. Asynchronous providers reject borrowed submission as unsupported
before admission.

### Event retirement

After detached admission, a provider-private retirement record owns the
provider retirement bindings, event, the `Arc<RootResource>` roots, and the
provider context until completion is proven.

```rust
struct RetirementRecord {
    event: ProviderEvent,
    bindings: Box<[ProviderRetirementBinding]>,
    roots: Box<[Arc<RootResource>]>,
    context: Arc<ProviderContext>,
}
```

`CompletionUnproven` returns only its typed cause and diagnostics. Its
provider-private record permanently retains the bindings, event, root `Arc`s,
and provider context because binding/event destruction while completion
remains unproven is not known to be safe. It does not
free speculatively and exposes no safe recovery path. There is no
quarantine/poison state, access or retirement registry, or retry transition.
Completion handles may be dropped without changing this retention. When event
retirement is proven, the record releases its retained bindings, event, root
`Arc`s, and context reference exactly once before publishing `Completed` or
`RetiredFailed`.

### Raw-handle state table

| Transition | cap | borrow | sync | fail | panic/drop | reclaim |
|---|---|---|---|---|---|---|
| checked shared pairing -> `PreparedRead::Device` | shared | `StorageRef` and checked descriptor borrow carried by device payload; provider reached through root | retained static proofs; access-time provider work only | exact checked pairing, no prepared object | temporary provider state released | owner follows G1 |
| checked exclusive pairing -> `PreparedWrite::Device` | exclusive | `StorageMut` and checked injective descriptor borrow carried by device payload; provider reached through root | retained static proofs; access-time provider work only | exact checked pairing, no prepared object | temporary provider state released | owner follows G1 |
| `PreparedDevice*` -> device binding | prepared shared / exclusive | binding keeps its capability and provider lifetime | binding work only; no second check or request/key/range comparison | exact device payload | unadmitted binding drops without changing ownership | no device resource is released before binding drop |
| prepared submission -> proven pre-admission rejection | owning / borrowed | package or bindings have not crossed the enqueue-capable call | no enqueue occurred | exact unchanged package/bindings | no event-retirement record exists | caller retains ownership |
| enqueue may have happened -> G3 `Draining` | owning task | task-local prepared bindings; no caller lifetime | event domains drain | no immediate owner return | worker/reaper retains package, retirement bindings, event, roots, and context | only after proven event retirement |
| G3 `Draining` -> `RetiredFailed` | owning worker/reaper | none | completion proven | returns owners with typed failure | retirement-held bindings/event/root/context references release exactly once | returned owners follow G1 |
| G3 `Draining` -> `Completed` | owning worker/reaper | none | completion proven | returns completed bundle | retirement-held bindings/event/root/context references release exactly once | returned bundle follows G1 |
| asynchronous provider rejects borrowed submission | borrowed | unchanged bindings | none; rejection precedes admission | unsupported with exact unchanged bindings | no work survives | caller retains bindings |
| admitted synchronous borrowed operation -> retired result | shared / exclusive | binding borrow remains until return | provider work retires before return | returns bindings only in retired completed/failed outcome | provider contract leaves no async work across unwind | after synchronous retirement |
| retirement -> `CompletionUnproven` | provider-private owning record | no public borrow | completion cannot be proven | diagnostics only; no owner is returned | record permanently retains bindings/event/root `Arc`s/provider context | bindings, event, roots, and context are never released by this outcome |

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
| 2 (#1558) | internal architecture/safety rustdoc for the unsafe allocation boundary and direct root/span ownership model | common commands |
| 3 (#1559) | `docs/spec/tensor-semantics.md` section III rewritten in the PR that removes public `Buffer<T>`; rustdoc/examples broken by clone/Buffer removal; final owner/view migration notes | common commands |
| 4 (#1560) | G1 state tables kept current; API rustdoc for prepared host access and provider event retirement; waits documented as synchronization points, explicitly not copies | common commands |
| 5 (#1561) | storage design updates for immutable aliasing, conservative disjointness, N-way borrow lifetimes, extraction | common commands |
| 6 (#1562) | reinterpretation rustdoc; the reserved section of the views guide (representation view vs numeric cast, supported pairs) | common commands |
| 7 (#1563) | CUDA design doc, device guide, unsafe interop rustdoc, synchronization/reclamation behavior, explicit duplication examples | common commands |
| 8 (#1564) | GPU backend design, device guide, Apple tutorials; synchronization/map transitions vs transfers; one owner with multiple access endpoints | common commands |
| 9 (#1565) | detached vs synchronous scoped ownership, outcome recovery, handle detachment, extraction; G3 state tables kept current | common commands |
| 10 (#1566) | GPU quickstarts, provider matrix, namespace rustdoc; `# Errors` sections for every public `Result` API | common commands |
| 11 (#1568) | hardware evidence recorded in the test profile/worklog with candidate Git commit | common commands |
| 12 (#1569) | `docs/guides/views-and-slicing.md` plus sidebar entry and an **Element access and performance** section; `docs/getting-started/core-concepts.md`; README/tutorials; rustdoc for `as_view`, random access, contiguous guard/slice access, iterators, and rank conversion; runnable owner/view/view-mut traversal examples; the rendered stale-language checker (`scripts/check-storage-docs.py`); the source-blind audit | common commands plus `python3 scripts/check-storage-docs.py --include-rendered`, `python3 scripts/check-storage-element-access-docs.py docs/guides/views-and-slicing.md`, and the exact `p12-element-access-examples` release command |
| 13 (#1567) | final worklog linking candidate Git commit, scaffolding disposition, hardware/docs/audit reports; deletion of `HANDOFF-2026-07-25-tenferro-unification6-wip.md` and inbound references | common commands plus closure validation from #1567 |

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
  an owner, creates a write capability, or produces a mutable view. The non-`Clone`
  one-owner rule applies to owners and capabilities.
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
the final `AllocationGroup`, prepared-access, retirement, and descriptor-ownership
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
| G1 prepared access and retirement | deterministic transition tests; claim provenance/split/overlap and exactly-once root-deallocator tests; compile-fail (guard across consuming submit, write preparation from shared); invalid-descriptor rejection before prepared-object construction; immediate-drop-after-enqueue; Miri on host guard slices; permanent binding/event/root/context retention when completion is unproven; constant prepare/map/bind/dispatch counts and no per-element abstraction work | #1560, performance evidence in #1566, providers in #1563/#1564 |
| G2 group, splitting, extraction | construction-time invalid layout/range/storage/provider rejection and retained-metadata counters; N-way split cases (N=0,1,>2, empty, reverse-stride) proving validation counters do not increase; write injectivity checked only when its retained proof is absent; pairwise-disjointness and permutation-independence property tests; direct borrowed-slot resolution for shared/exclusive group borrows, including empty entries; structural extraction-uniqueness tests (aliased records reject, sole record moves one owner, consuming extraction discards the rest); compile-fail (root access while children live); extraction counters; map/enqueue tests assert no validation rerun | #1561 |
| G3 submission terminal semantics | executable checks prove exact detached/scoped rejection recovery; host/CPU synchronous scoped acceptance and CUDA/WebGPU/Metal or asynchronous-provider rejection before admission; no borrowed work at return or unwind and no panic-catch/`Drop` safety; borrowed output-view coverage; consuming `into_output`/`into_owned_output` cases prove repeated and duplicate-output aliases plus the remaining map disappear together, failures return the exact bundle, and scoped borrowed/metadata rejection never copies; source checks reject extracted-state flags; worker/provider panic drains to typed `RetiredFailed` when completion is proven and ownerless `CompletionUnproven` otherwise; handle-detach and terminal-outcome suites; compile-fail (host guard across submit) | #1565, hardware in #1568 |
| G4 method distribution | API-parity contract with one canonical method list; compile-fail (no `Clone` on owners/capabilities); source scan (no mutable owner projections); static-rank preservation; allocation-free O(1) view construction; release traversal and fixed-rank codegen evidence | #1557 harness, #1559, #1566 |
| G5 raw handles, reclamation | executable prepared-once resolution counts; API/source checks that device-prepared access reaches provider context through its borrowed root without an extra `Arc` clone, contains no host guard, and bind accepts only the G1 device variant payload without a provider argument or lifetime; source checks that bind/enqueue accept no replacement request/key/range and perform no repeated static validation; acquisition and compile-fail checks that shared owner/immutable view yields only read preparation, while only exclusive owner/mutable view yields write preparation; source inventory proving raw handles, `Arc`, and refcounts cannot mint write authority; provider-matrix checks that asynchronous providers accept detached owning submission only and reject borrowed submission before admission; exact-return tests limited to failures proving no enqueue occurred, with post-boundary failures routed through G3 terminal outcomes; proven-retirement tests releasing bindings/event/roots/context exactly once; `CompletionUnproven` tests returning no owners and permanently retaining bindings/event/roots/provider context; raw-binder source inventory and unsafe-interop rustdoc checks for binding lifetime, synchronization duties, and post-retirement invalidity | #1558, #1563, #1564 |
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
- #1558 owns direct root lifetime and the non-`Clone` claim. #1560 owns
  prepared access and retirement. #1561 owns groups and direct borrowed
  descriptor slots. No phase introduces a provider bridge, and none waits for
  the public host cutover.
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
