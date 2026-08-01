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

## Conventions

Terminology:

- **Allocation**: one physical memory span owned by a provider, identified by
  an `AllocationSpan` (domain-qualified key plus byte range).
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

## G1. Span access and retirement

### Types and acquisition surface

```rust
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

unsafe trait BackendAllocation: Debug + Send + Sync + 'static {
    fn span(&self) -> AllocationSpan;
    fn provider(&self) -> ProviderKind;
    fn as_any(&self) -> &dyn Any; // provider-private diagnostics only

    fn acquire_host_read<'a>(&'a self, span: AllocationSpan)
        -> Result<HostReadGuard<'a>, AccessError>;
    fn acquire_host_write<'a>(&'a mut self, span: AllocationSpan)
        -> Result<HostWriteGuard<'a>, AccessError>;
    fn acquire_device_read(&self, endpoint: AccessEndpoint, span: AllocationSpan)
        -> Result<UseLease, AccessError>;
    fn acquire_device_write(&mut self, endpoint: AccessEndpoint, span: AllocationSpan)
        -> Result<UseLease, AccessError>;
}
```

Contract points:

- There is no public `timeline()`, `TimelineState`, `map_read`, or
  `map_write`. The provider-internal access state machine stays behind these
  four acquisition methods (#1555, "Host-visible memory and device
  timelines").
- Signatures are span-aware from the start. A provider may conservatively
  track whole allocations in v1; the API and validation stay span-scoped.
- `HostReadGuard`/`HostWriteGuard` expose only the validated byte span as
  immutable/mutable bytes and checked typed slices. Guards borrow the
  allocation (`'a`), so the borrow checker excludes moves (consuming
  submission) and exclusive operations while a guard is alive.
- `UseLease` is `'static`, provider-private, span- and access-mode-scoped. It
  holds provider pins (internally reference-counted handles are permitted as
  pins per I3), not Rust borrows, so it can move into runtime retirement
  records. Leases are non-cloneable and non-forgeable outside the provider.
- Write acquisition (`acquire_host_write`, `acquire_device_write`) requires
  the exclusive capability (`&mut`); read acquisition requires shared. This
  is the provider-side counterpart of "write mapping and write enqueue
  require the mutable capability" in #1555.

### Span rules

- `byte_offset + byte_len` uses checked arithmetic and must fit the provider
  allocation. `guaranteed_alignment` is a power of two describing the start
  of this span, not merely the base allocation.
- `AllocationKey` equality is domain-qualified (I3, #1558). Provider kind or
  device ordinal alone is never identity.
- Suballocations of one provider resource share `key` and differ by byte
  range. Conflict, hazard, and disjointness reasoning always operates on
  `(key, byte range, access mode)` triples, never on object identity.
- Two owners whose spans overlap for the same key must not exist. Group
  construction and provider constructors reject overlapping owner claims.
  Distinct non-overlapping suballocations sharing a key are valid.
- Zero-length spans: canonically valid when `byte_len == 0` and the offset
  passes checked arithmetic. Guards over empty spans return empty slices.
  Empty access acquires no provider resources and imposes no ordering. No
  code path may dereference a pointer to justify an empty span.

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
implementation resolves the owner and revalidates the descriptor against the
allocation as defense in depth (I7):

1. resolve `span()` of the owning allocation;
2. checked containment: descriptor byte range inside the span byte range;
3. alignment: descriptor start satisfies the dtype and provider requirement
   given `guaranteed_alignment`;
4. access mode: write requires the exclusive/owning path;
5. for writes, layout injectivity has been proven (G2).

Failure is a structured error naming the operation, requested range, and
resolved span key. Revalidation failure is always an error, never UB, even
if an internal invariant was violated upstream.

### State table

| Transition | cap | borrow | sync | fail | panic/drop | reclaim |
|---|---|---|---|---|---|---|
| `acquire_host_read` | shared | allocation for guard lifetime | wait: overlapping device writes (plus reads where provider requires) | no guard, no state change | guard drop unregisters host use | not while guard alive |
| `acquire_host_write` | exclusive | allocation, exclusively, for guard lifetime | wait: all overlapping device uses | no guard, no state change | drop unregisters; writes made so far are visible bytes, no rollback | not while guard alive |
| `acquire_device_read` | shared | none beyond the call (lease is a pin) | event dependency on last overlapping write | no lease, no state change | lease drop before submission releases pin | not while lease outstanding |
| `acquire_device_write` | exclusive | `&mut` for the call; lease pins after | event dependencies for RAW/WAR/WAW | no lease, no state change | same as device read | not while lease outstanding |
| lease submitted with work | owning (runtime owns inputs) | none (pins) | none at submit; retirement via events | enqueue prep failure releases only unsubmitted leases | admitted leases survive handle drop and panic until retirement | after all covering events complete |
| guard leaked (`mem::forget`) | n/a | borrow ends without `Drop` | none | n/a | provider host-use registration may persist until owner drop; soundness is preserved (access is gone), liveness may degrade; this is documented, not UB | owner drop path below |
| owner drop, no outstanding use | owning | none | none | n/a | n/a | immediately, exactly once, via the original deallocator |
| owner drop, outstanding leases | owning | none | none | n/a | deallocator and pins move into a retirement record | after the record's events complete, exactly once |
| retirement wait fails | n/a | none | attempted wait/poll | error reported on the runtime/provider error channel | resources quarantined: retained and reported | never speculatively; only if a later drain proves completion |

## G2. AllocationGroup

The group is the only sound representation for one owner with many logical
values (#1555, "Disjoint views and allocation groups"; #1561).

### Types

```rust
pub struct AllocationGroup {
    allocations: Vec<OwnedStorage>,   // private: each owned span exactly once
    values: Vec<TensorDescriptor>,    // private: interpretation + slot
}

pub struct TensorDescriptor {
    slot: AllocationSlot,             // index into `allocations`
    dtype: DType,
    layout: TensorLayout,
    placement: Placement,
}

pub struct ValueId(/* stable index for the group's lifetime */);
```

Construction preconditions (safe constructors):

- every `OwnedStorage` appears once; duplicate owner tokens are impossible by
  move semantics, and overlapping owner spans for one key are rejected;
- every descriptor is validated against its slot's span (G1 revalidation
  rules) at construction;
- descriptors may alias freely, including exact duplicates.

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
  standalone owner only when no remaining descriptor references the same
  slot. On failure the group is unchanged and the error carries a typed
  reason. There is no copy or materialization fallback (I4).
- `into_tensor` consumes the group, selecting one descriptor and explicitly
  discarding the rest; it never duplicates ownership to preserve them. On
  failure it returns the unchanged group.

### Central disjointness proof

One audited module owns the proof. Normative validation order (#1561):

1. validate each layout with checked shape/stride/offset arithmetic;
2. resolve dtype-sized byte ranges against the current `AllocationSpan`;
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
execution"; #1565). Both return group-based results.

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

pub enum ExecutionOutcome {
    Completed(ExecutionBundle),
    Failed(ExecutionFailure),       // recovered inputs, typed cause
    Cancelled(CancelledExecution),  // recovered inputs
}

pub struct ExecutionBundle {
    group: AllocationGroup,
    outputs: Box<[ValueId]>,
    retained_inputs: Box<[Option<ValueId>]>,
}

pub fn scope<'env, R>(
    &self,
    f: impl for<'s> FnOnce(&'s SubmitScope<'s, 'env>) -> R,
) -> R;

impl<'s, 'env> SubmitScope<'s, 'env> {
    pub fn submit_read_only(
        &'s self,
        program: &CompiledGraph,
        inputs: ScopedReadInputs<'env>,
    ) -> Result<ScopedHandle<'s>, ScopedSubmitError>;
}
```

- Repeated or aliased bindings reference descriptors; they never duplicate
  owners.
- `ExecutionBundle` fields are private. `output()` returns a borrowed view;
  `output_mut()` exclusively borrows the whole bundle; extraction follows G2.
  Identity, metadata-only, repeated-input, and duplicate-output graphs keep
  exactly one owner per physical allocation, with no hidden copy.
- `ScopedReadInputs` borrows immutable tensor/group views for `'env` and
  declares its access mode explicitly. Provider read leases are still
  acquired (G1), because logically read-only host and device uses can
  conflict on some providers. Mutable scoped inputs are out of scope for the
  redesign; a future mutable-input graph requires an exclusive borrow and a
  separate signature contract.
- `ScopedHandle<'s>` cannot escape the scope (higher-ranked lifetime), and
  scope exit joins and retires every admitted task before returning.

### Lifecycle

States: `Prepared` (validation/planning), `Admitted` (worker owns inputs),
`Running`, `Draining` (event domains drain after completion, error, panic,
or cancellation), `Retired(outcome)`.

| Transition | cap | borrow | sync | fail | panic/drop | reclaim |
|---|---|---|---|---|---|---|
| `submit` validation/preparation/spawn | owning (consumes `ExecutionInputs`) | none | none | `SubmitRejected` returns the exact unaccepted package | no worker exists yet; nothing retained | inputs back with caller; G1 rules |
| admitted, running | owning (worker) | none | leases per G1 acquired before each enqueue | execution error leads to Draining then `Failed` | worker panic leads to Draining/quarantine then `Failed` with a typed panic cause | only after retirement |
| `ExecutionHandle::wait` | none | none | blocks until Retired | returns `ExecutionOutcome` (all variants are post-retirement) | n/a | per outcome: bundle owns allocations |
| handle drop before completion | none | none | none | n/a | detach: reaper retains owners and leases until retirement; completion is not cancelled | after retirement, by the reaper |
| cancellation request | none | none | none | n/a | cooperative: honored at pre-enqueue boundaries only; already enqueued device work is never revoked | after retirement |
| unobserved failure (detached, handle dropped) | none | none | none | reported through the documented runtime error sink/callback; never silent | n/a | after retirement |
| scoped submit | shared borrows of inputs for `'env` | inputs until scope exit | leases per G1 | error carrier returns without admitting | dropping a `ScopedHandle` abandons observation only; scope exit still joins and retires every admitted task | after scope join |

Recovered inputs on `Failed`/`Cancelled` are exposed read-only and only
after all relevant event domains retire. Possibly partial or uninitialized
outputs stay private and are dropped after retirement or quarantine.

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
   `&mut Box<dyn BackendAllocation>`, or any mutable projection of an owner
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
  session.read(TensorRead<'_>)   -> Result<ReadBinding<'_>>;
  session.write(TensorWrite<'_>) -> Result<WriteBinding<'_>>;
  session.enqueue(...)           -> Result<Completion>;
  ```

  Bindings expose raw pointers only for the session lifetime; every binding
  revalidates domain/endpoint, span, layout arithmetic, alignment, access
  mode, and write injectivity (G1). A safe unleased
  `device_ptr(&Tensor) -> u64` does not exist; any escaping raw-pointer API
  is explicitly `unsafe` and documents its retirement obligations, and it
  stays provider-specific (no false parity).
- Shared-internal to exclusive transitions funnel through one crate-internal
  `unsafe fn` (working name `lease_unique`) with a documented uniqueness
  argument per call site. The call-site inventory is enforced by a
  source-contract test. `debug_assert!` on strong counts at hand-offs is
  diagnostics only, never the proof (I3).
- Stream-ordered reclamation (I6): a retirement record holds the deallocator
  and all pins for an allocation whose owner dropped with outstanding work.
  Records are keyed by event domain. Completion tokens visible to users are
  never the sole owner of a lease, because users may drop them. Runtime
  drop drains only its own retirement queue before releasing its context.
  A failed drain quarantines (retains and reports); it never frees early and
  never releases a provider context that pending work may still use.

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
| 3 (#1559) | `docs/spec/tensor-semantics.md` section III rewritten in the PR that removes public `Buffer<T>`; rustdoc/examples broken by clone/Buffer removal; AD interim-rule notes | common commands |
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
- Retention policy: an operation output is retained iff a registered
  VJP/JVP rule declares it needed for backward, or the user explicitly
  requests retention. Values nobody declares needed are not retained.
- When the caller wants a standalone owner of a retained value, the paths
  are exactly the G2 paths: `try_extract` once the tape no longer references
  the slot, or an explicit duplicate (classified below). There is no hidden
  copy path.

### Public API replacement

The `Arc<Tensor>`-returning surface is replaced. The Phase 3 retention
adapter must not appear in any public signature.

| Current | Replacement sketch | Semantics |
|---|---|---|
| `materialized(&self) -> Result<Arc<Tensor>>` | `value(&self) -> Result<ValueGuard<'_>>` | materializes if lazy, then exposes a borrowed `TensorView`; host bytes go through G1 guards |
| owned copy of a value | `duplicate_value(&self) -> Result<Tensor>` | explicit copy, reason `ExplicitDuplicate` |
| owned move of a value | `into_value(self) -> Result<Tensor, RetainedByTape>` | extraction via G2; fails while the tape still references the slot |
| backward result `Vec<Arc<Tensor>>`, `GradSlot = Arc<Mutex<Option<Arc<Tensor>>>>` | `Gradients` bundle (a G2 group specialization) with `grad(&self, key) -> Option<TensorView<'_>>` and `take_grad(&mut self, key) -> Option<Tensor>` | one owner per gradient allocation; extraction when unique |
| traced attached-data maps `HashMap<ValueKey, Arc<Tensor>>` | `ExecutionInputs` bindings over group descriptors (G3) | no shared owners in the runtime boundary |

### Checkpoint semantics

- Boundary values (checkpoint region inputs and outputs) are retained as
  descriptors in the checkpoint group.
- Interior values are deliberately discarded at record time; checkpointing
  must not accidentally retain every intermediate. A contract test asserts
  interior allocations are released after the boundary is recorded.
- Backward recomputation executes the stored subgraph and produces fresh
  owners; its allocations are classified `CheckpointRecompute`, distinct
  from retention (which allocates nothing) and from explicit duplicates.

### Reinterpretation and aliases

- A retained complex/real reinterpretation of a retained value is another
  descriptor of the same slot in the same group (G2 duplicate-descriptor
  semantics), never a second owner.
- Mutable reinterpretation requires an exclusive or owning capability. While
  any tape or checkpoint descriptor references a span, the owner lives in
  the retention group, so no caller can hold the owner: consuming or mutable
  reinterpretation of that allocation is unreachable, and `try_extract`
  fails with `RetainedByTape`. This exclusion is structural (borrow/owner
  placement), verified by compile-fail plus runtime extraction tests.

### Copy accounting

Copy and allocation events carry a reason:

```rust
enum CopyReason { ExplicitDuplicate, Transfer, CheckpointRecompute, OperationOutput }
```

- Retention itself has no reason variant because retention performs no copy.
- Acceptance for an AD scenario (forward plus backward, with and without
  checkpointing): every observed copy carries one of the enumerated reasons
  and matches the scenario's expected multiset; copies attributable to
  retention are therefore exactly zero.
- Aggregate pre-migration versus post-migration copy counts are not an
  acceptance criterion.

### Interim Phase 3 rule

Between the Phase 3 cutover and the Phase 9 group representation:

- one inventoried crate-private read-only retention adapter may bridge AD
  retention. It cannot authorize writes, cannot mint public owners, and
  cannot appear in public signatures;
- before Phase 4 retirement machinery exists, provider completion is
  synchronized conservatively before input owners are released; a failed
  synchronization retains or quarantines the allocation and never releases
  or reuses it speculatively;
- the adapter is listed in the scaffolding inventory; Phase 9 removes it.

### Validation lanes

- CPU is mandatory: eager and traced forward plus backward, and a
  checkpoint boundary-retention/recomputation case, with reason-classified
  copy counters.
- One supported asynchronous accelerator lane is designated before the
  Stage A freeze (CUDA preferred, else WebGPU or Apple/Metal) and runs the
  same retention contract with allocation, copy-reason, and retirement
  counters on real asynchronous provider work. Hidden CPU fallback or
  staging is a failure. Required-hardware mode cannot pass by skipping
  (#1568).

### State table: retained value lifecycle

| Transition | cap | borrow | sync | fail | panic/drop | reclaim |
|---|---|---|---|---|---|---|
| record op output into tape | owning (tape takes the output owner into its group) | none | none | op error: no retention entry | tape unwind drops group per G1/G2 | via tape drop |
| clone `EagerTensor` handle | none | none | none | n/a | handle drop is free | n/a |
| `value()` guard | shared | tape group (shared) for guard lifetime | G1 host-read rules if host bytes requested | error, tape unchanged | guard drop ends borrow | n/a |
| backward execution | shared reads of retained descriptors; new owners for grads | tape group shared during execution | G3 rules | typed failure, tape unchanged, grads dropped after retirement | per G3 panic row | grads owned by `Gradients` bundle |
| `take_grad` | exclusive on `Gradients` | none after return | none | `None`/typed reason, bundle unchanged | n/a | extracted owner per G1 |
| `into_value` while retained | owning attempt | none | none | `RetainedByTape`, value usable, tape unchanged | n/a | n/a |
| checkpoint record | owning (boundary owners into checkpoint group) | none | none | error: no partial checkpoint | interior values already released | boundary owners via checkpoint drop |
| checkpoint recompute (backward) | shared reads of boundary; fresh owners for recomputed values | checkpoint group shared | G3 rules | typed failure after retirement | per G3 | recomputed owners dropped after use |
| tape drop | owning | none | retirement per G1 for every owner with in-flight work | n/a | quarantine path per G1 | after retirement, exactly once |

## Contract test index

Each gate's clauses are enforced by tests owned by the listed phases. The
phase issues carry the full inventories; this index is the cross-reference.

| Gate | Enforcement | Owning phases |
|---|---|---|
| G1 ordering, guards, revalidation, retirement | deterministic fake-timeline transition tests; compile-fail (guard across consuming submit, write guard from shared); corrupt-descriptor rejection at map and enqueue; immediate-drop-after-enqueue; Miri on host guard slices | #1560, providers in #1563/#1564 |
| G2 group, splitting, extraction | N-way split cases (N=0,1,>2, empty, reverse-stride, overflow); permutation-independence property tests; compile-fail (root access while children live); extraction counters | #1561 |
| G3 submission terminal semantics | rejection carriers return identical allocation keys; cancellation/panic/detach/unobserved-error suites; compile-fail (scoped handle escape, host guard across submit) | #1565, hardware in #1568 |
| G4 method distribution | API-parity contract with one canonical method list; compile-fail (no `Clone` on owners/capabilities); source scan (no mutable owner projections) | #1557 harness, #1559 |
| G5 raw handles, reclamation | fake backend proving internal `Arc` clones cannot write or mint owners; `lease_unique` call-site inventory; retirement/quarantine tests; source scan (no safe unleased pointer) | #1558, #1563, #1564 |
| G6 documentation | rendered stale-language checker; doctests; tutorial-code checks; source-blind audit | #1569, #1567 |
| G7 AD retention | reason-classified copy counters (zero retention copies); extraction-blocked tests; checkpoint interior-release test; mutable-reinterpret exclusion; CPU plus designated async accelerator lanes | #1557 contract, #1559 interim, #1565 final, #1568 evidence |

## Relationship to phase issues

- #1556 (Phase 0) is independent of every gate.
- #1558 through #1566 implement G1 through G5 and the control plane against
  this document.
- #1567 through #1569 validate, document, and close against the frozen
  candidate.
- A later phase that needs to deviate from a contract here must change this
  document (and its tests) first, in the same PR, with the deviation called
  out in the PR description.
