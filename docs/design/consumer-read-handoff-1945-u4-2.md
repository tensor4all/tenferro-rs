# Consumer read handoff: a registry-bound consumer copy of a published value (U4-2) (#1945 U4)

Revision 2. Revision 1 (a borrowed `PublishedRead` view admitted through a new driver-level hook) was
rejected by its pre-implementation review, and the rejections were verified against the source:

| Revision 1 claim | Source | Disposition |
| --- | --- | --- |
| "the publication's frontier already implements `EventToken`" | the publication stores `frontier: Option<CUevent>` (`crates/tenferro-gpu/src/cubecl/transfer.rs:1332-1345`); the admissible token is the private `CudaEventToken` (`crates/tenferro-gpu/src/cubecl/event_domain.rs:198-215`) | **wrong**: revision 2 mints a registry-provenanced completion from the recorded event |
| a driver-level `admit_foreign_dependency(&self, domain, token, source)` orders "the next run" | every run already exists before execution (`crates/tenferro-runtime/src/runtime/execution.rs:2198-2231`) and the destination run is located before classification (`:2259-2290`) | **wrong shape**: admission moves onto the destination run's enqueue |
| the held session obtains a domain id from the snapshot | `CudaHeldSession` holds only the backend borrow, reservation, stream identity and stats (`crates/tenferro-gpu/src/cubecl/exec_session.rs`); the frozen registry is established by `cuda_runtime_engine_registration` (`crates/tenferro-gpu/src/cubecl/runtime_adapter.rs`) | **gap**: revision 2 specifies one owner-scoped registry binding |
| `PublishedRead<'p>` borrowing `&'p SourcePublication` protects the asynchronous read | the source half queues the source tensor behind the *producer* frontier only (`transfer.rs:1599-1617`), and its drainable retirement queue holds owned records (`:251-256, 396-410`) | **unsound**: a borrow cannot exclude `download_into`, `mem::forget` or later writes; revision 2 replaces the view with an owned consuming copy |
| "readiness across streams is a view property" | consumption re-applies the captured stream through `StreamId::executes` (`transfer.rs:1519-1541`); CubeCL aligns resources by `handle.stream` (`cubecl@1811309/crates/cubecl-runtime/src/stream/event.rs:257-312`) | **incomplete**: revision 2 binds admission, allocation, copy and completion to the *consumer* stream |

## Review status (read this first)

Revision 2 is **not approved and not implemented**. Its own pre-implementation review
(`openai/gpt-6.1-sol`, fresh read-only process) returned "do not implement revision 2 as written" and confirmed
that revision 2 fixes the *direction* of revision 1 (consuming ownership, destination-run admission,
consumer-stream ordering) while four blockers remain, all verified against the source:

1. **No acquisition boundary for the capability.** `CudaBackend::open_session` takes no registration, and a
   domain is minted only when a snapshot freezes, so "`open_session` acquires the capability from the
   registration" is asserted rather than designed. The capability must be owned by the *selected frozen
   executable slot*, and a non-`Clone` capability cannot move into every session - a credential shape is
   needed.
2. **`ForeignAdmission::Admitted` must carry the destination completion token.** The scheduler validates and
   records that token for downstream dependencies; an `Admitted` without it cannot be inserted, and calling
   the ordinary `enqueue` afterwards would launch twice or record a second completion.
3. **The retirement queue cannot support pre-enqueue ownership.** Its records are owned and consumed with no
   reservation/recovery operation, and a drainable record needs a *recorded* completion (an unrecorded event
   may query as complete). The smallest correct shape is the existing pending-transfer model: an owning
   local submission guard, moved into `PendingRead` on success, detached on drop, permanently retained after
   an unproven submission - no pre-enqueue insertion.
4. **Consumer retirement ownership is unsettled.** A copy can outlive the session borrow, so its package must
   retain source *and* destination allocations plus provider lifetimes through success, error, drop and
   forgetting; the source runtime's teardown must not be assumed to retire a consumer stream; and putting a
   strong runtime-owning completion into that runtime's own queue creates a retention cycle (the pinned
   buffer uses a weak reference for exactly this reason). Adding a normal `CudaEventDomainRun` to the session
   would also add implicit synchronization on session destruction.

Also recorded by the same review: exact-event completion is feasible with one shared event owner
(`Arc<CudaEventHandle>`) but requires updating `release_frontier` and the detach paths and matching the
admission's `CudaEventToken` downcast; scheduler admission does **not** make the *scheduled transfer* path
asynchronous (its provider contract is blocking), so the "host-wait-free" claim must be narrowed to the
held-session consumer route; and same-device eligibility must be explicit.

Revision 3 addresses these before implementation. Nothing in this record may be implemented as it stands.

## 1. Scope

#1945 item 1's consumer half: a consumer session reads a value published by another session, ordered by a
**registry-established dependency** rather than the runtime's cross-domain host bridge, with no
caller-constructed domain id and no public driver escape. The producer half is revision 6 of
`docs/design/source-publication-1945-u4.md`.

## 2. Settled decisions

### 2.1 The registry binding is the provenance, not an id

A frozen engine witness (`RuntimeId`, `RuntimeEpoch`, `RegistrationIdentity` → `EventDomainId`, minted in
`crates/tenferro-runtime/src/runtime/snapshot.rs`) is bound to the exact CUDA runtime and the consumer's
logical stream by one owner-scoped admission capability:

```rust
// crates/tenferro-gpu/src/cubecl/runtime_adapter.rs (CUDA engine registration)
/// Admission capability for one frozen engine witness.
///
/// Minted by the registration that owns the frozen witness; not `Clone`, not constructible by callers,
/// and never exposed as a driver handle. It binds the witness to the CUDA runtime the registration
/// drives, so a completion minted or admitted through it carries that witness's provenance.
pub struct CudaDomainAdmission { /* private: witness (domain), runtime pin, CUDA provider handle */ }

impl CudaDomainAdmission {
    /// The frozen domain this capability speaks for.
    pub fn domain(&self) -> EventDomainId;
}
```

- `CudaHeldSession::open_session` acquires the capability from the registration it was opened through and
  carries it privately, so every held session is registry-backed from the start (no new constructor and no
  caller-supplied id).
- `EventDomainId` is *not* accepted as an argument anywhere in this design. `EventDomainId::new` stays
  `pub(crate)`; the public snapshot accessor (`EngineSnapshotView::event_domain_id`) and the public transfer
  accessors (`TransferRequest::source_event_domain_id`/`destination_event_domain_id`) only expose ids of
  existing things - possessing one proves nothing, which is exactly why the capability, not the id, is the
  provenance (this corrects revision 1's stronger "only public source" claim).

### 2.2 The dependency is a registry-provenanced completion of the recorded frontier

`publish` mints, through the session's capability, an owned completion that keeps the **exact recorded
`CUevent`** and carries the producer's frozen domain. It is the private CUDA token machinery
(`CudaEventToken`) with a publication-specific record, not a new public token type:

```rust
// crates/tenferro-gpu/src/cubecl/transfer.rs (private)
struct PublicationCompletion {
    domain: EventDomainId,          // from the session's CudaDomainAdmission
    event: CUevent,                 // the exact event publish recorded, never a replacement
    provider: /* runtime pin */,
}
impl EventToken for PublicationCompletion { /* as_any, origin, repeatable wait */ }
```

`SourcePublication` stores this completion instead of the bare event, and `frontier_ready`/`await_frontier`
keep their current query/synchronize semantics through it. No replacement event is recorded on an unrelated
stream, and no new public token type is added.

### 2.3 Admission happens on the destination run, before its enqueue

The destination run exists before classification, so the runtime gains no driver-global state. The
classification (`crates/tenferro-runtime/src/runtime/execution.rs`, `ScheduledEventDomains`) resolves a
foreign-domain transfer dependency into the run's enqueue:

```rust
// crates/tenferro-runtime/src/runtime/event_domain.rs
pub trait EventDomainRun {
    // ... existing methods ...

    /// Enqueue `launch` with the given same-domain dependencies plus `foreign`.
    ///
    /// Returning [`ForeignAdmission::Unsupported`] keeps the runtime's documented host bridge for that
    /// edge: the caller never sees a silently reordered dependency, only a slower but correct one.
    fn enqueue_with_foreign(
        &mut self,
        dependencies: &[Arc<dyn EventToken>],
        foreign: &[Arc<dyn EventToken>],
        launch: &mut dyn FnMut() -> crate::Result<()>,
    ) -> crate::Result<ForeignAdmission> {
        let _ = (dependencies, foreign, launch);
        Ok(ForeignAdmission::Unsupported)
    }
}

pub enum ForeignAdmission { Admitted, Unsupported }
```

- The CUDA run applies an admitted foreign completion **inside its captured-stream execution**, exactly as
  it already applies same-domain waits (`crates/tenferro-gpu/src/cubecl/event_domain.rs`), so the wait lands
  on the destination stream and no ambient stream is involved.
- Only `Unsupported` falls back to `dependency_completion.wait()`; the outcome is counted with
  test-driver instrumentation rather than a new public snapshot surface (§4).
- The consumer *read* route (§2.4) enqueues the same way through its own session's run, so the held-session
  path and the scheduled path share one admission mechanism.

### 2.4 The consumer read is an owned consuming copy

Revision 1's borrowed view is replaced by a consuming operation whose ownership is explicit before any
enqueue:

```rust
// crates/tenferro-gpu/src/cubecl/transfer.rs
impl SourcePublication {
    /// Consume the publication by copying it into `consumer`'s domain, ordered after its frontier.
    ///
    /// The publication is consumed: the source value, allocation and completion move into the returned
    /// pending read, so no later `download_into`, write or reuse can race the consumer's copy.
    pub fn copy_into(
        self,
        consumer: &mut CudaHeldSession<'_>,
    ) -> Result<PendingRead, ReadAdmissionError>;
}

/// An admitted consumer copy in flight.
pub struct PendingRead { /* owns the source-side retention record + the consumer completion */ }

impl PendingRead {
    /// Byte length of the copy.
    pub fn bytes(&self) -> usize;
    /// Wait for the copy's completion and take the consumer-side tensor.
    pub fn wait(self) -> Result<Tensor, ReadAdmissionError>;
}

pub enum ReadAdmissionError {
    /// Nothing was enqueued: the publication is returned unchanged, with its retention intact.
    Refused { publication: Box<SourcePublication>, error: crate::Error },
    /// The enqueue may have happened: the source retention stays queued, no value is returned.
    Unproven { error: crate::Error },
}
```

Semantics, all settled:

1. **Ownership before enqueue.** `copy_into` moves the publication into a source-side retention record
   *before* it enqueues anything; a refusal hands the publication back, an `Unproven` outcome keeps the
   record queued. Dropping or forgetting the returned `PendingRead` cannot release the source: the record is
   already on the bounded retirement queue (drainable) or its permanent-retention side, and `Drop` never
   waits - the same contract the pending transfers state.
2. **Exclusion.** Because the publication is consumed, the revision-1 counterexamples (enqueue then
   `download_into`; `mem::forget` of the view; `Unproven` followed by reuse) cannot occur: there is no
   publication left to consume, write or reuse.
3. **Consumer-stream ordering.** The copy is enqueued on the **consumer's captured stream** (reapplied
   through `StreamId::executes`), with the admitted foreign wait enqueued before it on that same stream; the
   consumer completion is recorded on that stream. A copy on an ambient third stream is not permitted, and
   the held session's own run is the execution authority - the operation does not outlive it (it borrows the
   session for the call and the session's stream for the copy).
4. **Readiness.** `PendingRead::wait` is the explicit boundary; `frontier_ready` on the publication is
   unchanged for the caller who has not consumed it (and takes `&mut self`, so it is called before
   `copy_into`, not during).
5. **Terminal publications.** A publication that recorded a failure is refused before any enqueue with
   `ReadAdmissionError::Refused`; the source half's terminal record is untouched.

### 2.5 No public driver escape

The capability is a private field of the held session, the run admission is a trait method on the runtime's
run, and nothing in this design returns a driver, a raw device handle or an `EventDomainId` chosen by a
caller. The new public surface is exactly `SourcePublication::copy_into`, `PendingRead` (two methods) and
`ReadAdmissionError`; the private surface is the completion type, the capability and the run admission.

## 3. Non-goals

Unchanged from the source half's deferrals: pageable staging, P2P routes, differentiated transfers,
publishing from the *scoped* callback entry, and U2-AD (D12). Pageable/overlapped copies of publications
are explicitly out of scope: this slice copies one published value once, on the consumer's stream.

## 4. Evidence plan

Two evidence layers, deliberately separated (revision 1 conflated them):

**(a) Scheduler-level, no GPU:** a test driver whose run returns `Unsupported` keeps the host bridge and
increments the host-wait counter (`crates/tenferro-runtime/src/runtime/tests/event_domain.rs` already
injects a host-wait failure); a run that admits increments the admitted counter and the launch proceeds.
Claim proven: *no scheduler dependency host wait* for an admitted edge. This does **not** claim that
publication or copying never host-waits - the source half's resolution/dispatch waits stay as documented.

**(b) CUDA consumer path (local A100, GPU model/driver/toolkit recorded):**

| Claim | Evidence |
| --- | --- |
| Genuine cross-stream read | producer session A and consumer session B hold *different* streams (asserted, not assumed); publish on A, `copy_into(B)`, `wait()` returns the exact bytes; A's frontier was not ready at admission |
| Consumer-stream ordering | after `copy_into`, a probe operation enqueued on B observes the copied bytes; a probe on A before the admitted wait does not (the wait is on B) |
| Ownership before enqueue | a refusal returns the publication with its retention intact; an `Unproven` outcome leaves the source record queued; dropping/forgetting a `PendingRead` never releases the source early (asserted through the retirement counters and a `synchronize` after) |
| Exclusion | the publication cannot be reused after `copy_into` (compile-fail fixture: `download_into`/`copy_into` on a moved value), and no safe write path exists (#2093) |
| Fault-injected terminal paths | a simulated provider failure at the completion query/submission seam drives: consumer admission refusal (publication returned), copy success followed by a completion-record failure (no value, record retained), and a terminal publication's refusal; each asserts the retirement/ownership state, not just the error |
| No regression | the pending-transfer, publication and overlap suites keep passing; the audited CUDA test partition is updated; the `transfer_overlap` bench (`crates/tenferro-gpu/benches/transfer_paths.rs`) gains a two-session consumer arm |

## 5. Open questions for maintainers

1. Should the run admission be a new trait method (`enqueue_with_foreign`, as specified) or a parameter on
   the existing enqueue? The classification resolves dependencies immediately before the run's enqueue,
   which is why this record adds the variant rather than a second call.
2. Should `CudaDomainAdmission` be acquired by `open_session` (this record's choice: every held session is
   registry-backed) or handed to `publish`/`copy_into` explicitly (fewer hidden fields, more call sites)?
3. Does the registration want the capability to be per-*engine* or per-*stream*? This record mints it per
   engine and binds the stream at the call site, because one registration can drive several sessions.
