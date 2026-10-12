# Consumer read handoff: an admitted consumer copy of a published value (U4-2) (#1945 U4)

Revision 3. Revisions 1 and 2 were both rejected by independent pre-implementation reviews (recorded below);
revision 3 takes the reviewers' smallest-correct-alternative direction: **scope registry authorization to the
publication/consumer pair** instead of every held session, **reuse** the existing token, retirement and error
machinery instead of parallel types, and **change no scheduled-transfer contract**.

## Review status

| Revision | Verdict | Blockers that forced the next revision |
| --- | --- | --- |
| 1 | do not implement | the frontier is a native `CUevent`, not an `EventToken`; a driver-level hook cannot identify the destination run; nothing connects the held session to the frozen registry; a borrow cannot protect the asynchronous read; the view can outlive its session |
| 2 | do not implement | `open_session` has no frozen registration to acquire from; `Admitted` must carry the destination completion token; the retirement queue cannot support pre-enqueue ownership; consumer retirement ownership (both allocations, teardown authority, the runtime-owning-completion cycle) is unsettled |
| 3 | do not implement | `bind(backend, view)` has no exact-instance validation seam (the view exposes nominal identities and ingress predicates, not backend-instance or registration-candidate equality); one frozen witness does not authorize two distinct runtimes (the producer's and the consumer's); the consumer path has no destination run for the admission to land on; and adopting `CudaEventHandle` reintroduces a runtime-owning-completion cycle |

Every rejection was verified against the source before it was recorded (citations appear inline). Revision 3
is not implemented and must be re-reviewed before implementation starts.

## Revision 4 direction (recorded after the third review)

The third review confirms the *copy* is the bounded, implementable piece and that the registry/token/admission
half needs substrate contracts this repository does not have yet:

1. an exact-instance binding seam - either a provider-neutral predicate such as
   `EngineSnapshotView::matches_executable_registration(&EngineRegistration)`, or a new executable-instance
   identity contract; today the view exposes nominal identities and ingress predicates, and `CudaBackend` does
   not retain its registration candidate token (`crates/tenferro-runtime/src/runtime/snapshot.rs`,
   `crates/tenferro-runtime/src/runtime/engine_registration.rs`, `crates/tenferro-gpu/src/cubecl/runtime_adapter.rs`);
2. a two-runtime authorization model - one witness cannot authorize a producer runtime and a consumer runtime
   at once;
3. a destination-run (or equivalent admission) surface on the consumer path - the held-session copy never
   enters a run, so `enqueue_with_foreign` has nothing to admit into;
4. a cycle-free completion owner - the adopted `CudaEventHandle` holds a strong `CudaRuntime`, which a
   runtime-owned retirement record cannot hold (`crates/tenferro-gpu/src/cubecl/event_domain.rs`).

Those four are a substrate decision, not a bounded consumer slice. Revision 4 therefore splits this record:

- **4a (implementable now, review-endorsed):** an **unadmitted same-runtime, same-device consumer copy**, specified in
  `docs/design/consumer-copy-1945-u4-2a.md` -
  `SourcePublication::copy_into(&mut CudaHeldSession<'_>)` with an eligibility check that both ends belong to
  the same CUDA runtime and device, the pending-transfer ownership model, the consumer-stream copy and
  completion, and the private fault-injection seam. No registry, no token, no run admission, no new
  substrate contract. Claims bounded to "device-side consumer copy on the same runtime and device".
- **4b (deferred, needs the substrate decision above):** the registry-authorized, cross-runtime handoff with a
  domain-scoped dependency token admitted into the consumer's run - i.e. the part #1945 item 1 calls
  "authorized read access plus a domain-scoped dependency token obtained from the runtime registry".

Nothing in the revision-3 design is implemented; 4a is specified below only as a scope decision and needs its
own review before implementation.

## 1. Scope

#1945 item 1's consumer half: a consumer session copies a value published by another session on the **same
device**, ordered by a dependency admitted into the consumer's own execution stream instead of the runtime's
cross-domain host bridge, with **no caller-constructed domain id**, **no public driver escape**, and an
explicit registry authorization. The producer half is revision 6 of `docs/design/source-publication-1945-u4.md`.

Explicitly out of scope, because the reviewers showed they are separate contract changes: making the
**scheduled blocking transfer** provider asynchronous
(`crates/tenferro-runtime/src/runtime/transfer.rs`: a blocking provider must return an immediately readable
tensor), making ordinary held sessions registry-backed, pageable/P2P/differentiated transfers,
scoped-callback publishing, and U2-AD (D12).

## 2. Settled decisions

### 2.1 Registry authorization: one admitted pair, not every session

Authorization is a value the caller obtains from the **selected frozen executable slot** and passes to the two
entries that need it. Ordinary `CudaBackend::open_session` keeps its current signature and stays
registry-independent.

```rust
// crates/tenferro-gpu/src/cubecl/runtime_adapter.rs (public in tenferro_gpu::cuda)
/// Authorization for one frozen CUDA engine witness on one device.
///
/// Not `Clone`, not constructible by callers, and never a driver handle: `bind` validates that `view`
/// froze the CUDA engine this backend drives, and keeps only the witness, the runtime pin and the device
/// id. The type is public because callers hold it; its fields and its constructor are not.
pub struct CudaAdmittedPair<'a> { /* private: witness (EventDomainId), runtime pin, device id */ }

impl<'a> CudaAdmittedPair<'a> {
    /// Bind a frozen snapshot view to the backend it froze.
    ///
    /// # Errors
    ///
    /// [`AdmissionError`] when the view did not freeze this backend's engine (a different runtime, epoch,
    /// or device), so a caller cannot authorize itself against a witness it does not own.
    pub fn bind(
        backend: &CudaBackend,
        view: &'a EngineSnapshotView<'a>,
    ) -> Result<Self, AdmissionError>;

    /// The device this authorization is valid for.
    pub fn device(&self) -> DeviceId;
}
```

- No entry point accepts an `EventDomainId`. `EventDomainId::new` stays `pub(crate)`
  (`crates/tenferro-runtime/src/runtime/schedule.rs`), and the public accessors
  (`EngineSnapshotView::event_domain_id`, `TransferRequest::source_event_domain_id`/
  `destination_event_domain_id`, `EventToken::origin`) only expose ids of *existing* things. Possessing an id
  or a token with a matching `origin` is therefore **not** authorization - the admitted pair is.
- Authorization is scoped to the pair: the producer publishes *through* it and the consumer copies *through*
  it, so the two ends provably belong to the same frozen witness and device.

### 2.2 The completion reuses the existing event-token machinery

`publish` records the frontier event exactly as it does today and then **adopts** it into the existing owner
instead of storing a bare handle; there is no parallel completion type:

- the recorded `CUevent` is wrapped in the private `Arc<CudaEventHandle>` that `CudaEventToken` already owns
  (`crates/tenferro-gpu/src/cubecl/event_domain.rs`: the handle owns the event, has justified `Send`/`Sync`
  impls, waits without consuming the event, and destroys it in its owner);
- `SourcePublication` stores that shared owner, and `frontier_ready`/`await_frontier` query through it, so
  there is one owner and no double destruction;
- `release_frontier` and the terminal/detach paths pass the **same shared owner** into the retirement record
  rather than moving a raw event, so the existing drop path keeps working;
- the publication exposes the frontier as a `CudaEventToken` (the type CUDA admission already downcasts to,
  `crates/tenferro-gpu/src/event_domain_admission.rs`), so an admitted pair can hand it to a consumer run.

`publish` through an admitted pair records the pair's witness on the completion; the unadmitted `publish`
keeps its current behaviour and mints nothing that can be admitted.

### 2.3 Admission happens on the destination run and returns the completion

```rust
// crates/tenferro-runtime/src/runtime/event_domain.rs
pub trait EventDomainRun {
    // ... existing methods ...

    /// Enqueue `launch` with same-domain `dependencies` plus `foreign`, which came from other frozen domains.
    ///
    /// Contract:
    /// - `Unsupported`: nothing was enqueued, `launch` was not invoked, no dependency was partially
    ///   admitted. The caller keeps its documented host bridge.
    /// - `Admitted(token)`: every dependency (same-domain and foreign) was accepted, `launch` was invoked
    ///   exactly once, and `token` is the destination completion the scheduler records for downstream
    ///   dependencies.
    /// - `Err`: the same partial-enqueue semantics as the existing enqueue; the caller reports it as the
    ///   node's failure.
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

pub enum ForeignAdmission {
    /// Dependencies accepted; the destination completion for downstream scheduling.
    Admitted(Arc<dyn EventToken>),
    /// Nothing was enqueued; the runtime host-waits this edge as before.
    Unsupported,
}
```

- The CUDA run applies an admitted foreign completion **inside its captured-stream execution**, exactly as it
  already applies same-domain waits (`crates/tenferro-gpu/src/cubecl/event_domain.rs`), so the wait lands on
  the destination stream and no ambient stream is involved.
- The default implementation changes nothing; the contract above is what an override must satisfy, so an
  implementation cannot enqueue waits and then answer `Unsupported`.
- **Only the consumer copy (§2.4) uses this in this revision.** The scheduled transfer path keeps its blocking
  provider contract: the scheduler's explicit source-token `wait()` may be removable for an admitted edge, but
  the provider still must return a readable tensor, so this revision claims **no** host-wait-free scheduled
  transfer and does not relax that contract.

### 2.4 The consumer copy reuses the pending-transfer ownership model

```rust
// crates/tenferro-gpu/src/cubecl/transfer.rs
impl SourcePublication {
    /// Consume the publication by copying it into `consumer`'s device, ordered after its frontier.
    ///
    /// # Errors
    ///
    /// [`PublicationError`] - the existing carrier - when the admitted pair does not authorize this
    /// publication (a different frozen witness or device), when the publication is terminal, or when the
    /// submission is unproven. A refusal before any enqueue returns the publication unchanged.
    pub fn copy_into(
        self,
        admitted: &CudaAdmittedPair<'_>,
        consumer: &mut CudaHeldSession<'_>,
    ) -> Result<PendingRead, PublicationError>;
}

/// An admitted consumer copy in flight. `wait` is the only accessor, as in `PendingDownload`.
pub struct PendingRead { /* private */ }

impl PendingRead {
    /// Byte length of the copy.
    pub fn bytes(&self) -> usize;
    /// Wait for the consumer completion and take the destination tensor.
    pub fn wait(self) -> Result<Tensor, PublicationError>;
}
```

Settled semantics, in the order they matter:

1. **Eligibility before any enqueue.** The pair must authorize this publication (same frozen witness and same
   device as the publication's own admission) and the consumer session must be on that device; otherwise the
   call refuses with the existing `PublicationError::Rejected`, returning the publication unchanged. Nothing
   is enqueued, so no ownership has moved.
2. **Owning submission guard, not a pre-enqueued record.** The call builds a local guard that owns the
   publication (value, allocation, completion) and the destination allocation. On success the guard moves
   into `PendingRead`; on refusal it returns the publication; if the submission outcome is unproven the guard
   is detached into permanent retention. `mem::forget(PendingRead)` leaks the owned package in place, exactly
   as forgetting a pending transfer does. There is **no** queue insertion before the enqueue, so the existing
   queue needs no reservation or recovery operation.
3. **Consumer-stream ordering, one stream.** The copy is enqueued on the **consumer's captured stream**
   (reapplied through `StreamId::executes`), with the admitted foreign wait enqueued before it on that same
   stream, and the consumer completion recorded on that stream. A same-device address copy on that stream is
   sufficient: the producer frontier already represents the producer's writes, so the producer's stream is not
   reapplied. No `CudaEventDomainRun` is created inside the session, so session destruction keeps its current
   semantics (`close` submits; it does not add a retirement barrier).
4. **Retention authority and no cycle.** The in-flight package is owned by the **consumer runtime's** transfer
   retirement queue - the runtime whose stream carries the completion - and it holds:
   - a **strong** pin on the *source* runtime (a different runtime, so this is not a self-cycle) plus the
     source value/allocation, which is what keeps the published storage alive until the consumer completes;
   - a **weak** reference to the *consumer* runtime, following the pinned-buffer precedent
     (`crates/tenferro-gpu/src/cubecl/transfer.rs`), so the queue record cannot pin its own runtime and the
     last-owner teardown qualification is preserved;
   - the destination allocation and the consumer completion.
   The source runtime's teardown is **not** assumed to retire a consumer stream: the record's strong source pin
   means the source runtime cannot tear down first, and the consumer runtime's own teardown retires its stream
   and then drains its queue, as it already does for pending transfers.
5. **Terminal publications** are refused before any enqueue (rule 1), so a failed publication can never be
   copied as if it were ready.

### 2.5 No public driver escape

The admitted pair is a value with a crate-private constructor; the run admission is a trait method on the
runtime's own run; nothing in this design returns a driver, a raw device handle, an `EventDomainId` chosen by
a caller, or the frozen executable witness. New public surface: `CudaAdmittedPair` (`bind`, `device`),
`SourcePublication::copy_into`, `PendingRead` (`bytes`, `wait`), `ForeignAdmission`, and the run's
`enqueue_with_foreign` (a provider contract on the already-public `EventDomainRun`).

## 3. Evidence plan

**(a) Scheduler-level, no GPU** (`crates/tenferro-runtime/src/runtime/tests/event_domain.rs`, whose test
tokens already count waits and inject failures):

| Claim | Evidence |
| --- | --- |
| Admission returns a usable completion | the returned token is recorded and consumed by a downstream node; the node runs exactly once |
| `Unsupported` changes nothing | launch count is zero, no dependency is enqueued, the host bridge is used and counted |
| Override contract | a test run that enqueues and then answers `Unsupported` fails the contract assertion (the trait's documented rule) |

**(b) CUDA consumer path (local A100; GPU model, driver and toolkit recorded):**

| Claim | Evidence |
| --- | --- |
| Genuine cross-stream copy | producer and consumer sessions hold **distinct physical stream handles/slots** (asserted on the handles, not the logical ids, which map modulo the bounded pool); sessions run sequentially on one thread (two live held sessions are rejected) or on separate threads; the copied bytes equal the produced bytes |
| Consumer-stream ordering | after the copy, work enqueued on the consumer stream sees the copied bytes; the consumer stream is synchronized **explicitly** (ambient synchronization is not evidence). No negative "the producer stream cannot see it" assertion is made: missing ordering permits an early read but does not guarantee one |
| Authorization | a pair bound to a different snapshot view, runtime or device refuses before enqueue with the publication returned; `CudaAdmittedPair` has no public constructor (compile-fail fixture) |
| Ownership before enqueue | refusal returns the publication with its retention intact; an unproven submission leaves the guard permanently retained; forgetting a `PendingRead` leaks the owned package in place; retention is observed through controlled pending work and resource-release observations, not aggregate counters |
| Exclusion | the publication cannot be reused after `copy_into` (compile-fail fixture: `download_into`/`copy_into` on a moved value); no safe write path exists (#2093) |
| Fault-injected terminal paths | a private seam at the completion query/submission boundary (part of this slice, `#[cfg(test)]`-reachable) drives: a terminal source query/wait failure, consumer admission refusal, and a copy whose completion record fails; each asserts the resulting ownership/retirement state |
| No regression | pending-transfer, publication and overlap suites keep passing; the audited CUDA test partition is updated; the `transfer_overlap` group (`crates/tenferro-gpu/benches/transfer_paths.rs`) gains a two-session consumer arm |

## 4. What this revision does not claim

- No host-wait-free **scheduled transfer**: the blocking provider contract is unchanged.
- No registry authorization for ordinary held sessions.
- No fault-injected evidence for the source half's own terminal path beyond the seam listed above; the source
  record's gap stays open until that seam is implemented.
- No pageable, P2P, differentiated or scoped-callback publication routes, and no U2-AD.

## 5. Open questions for maintainers

1. Should the admitted pair bind one device (this record's choice: the copy is same-device) or one engine
   across devices (which would make the eligibility check a per-call device comparison)?
2. Should the in-flight record live in the consumer runtime's queue (this record's choice, matching the
   completion's stream) or in a dedicated consumer-side queue with its own teardown rule?
3. Is `EngineSnapshotView` the right public anchor for `bind`, or should the registration handle expose the
   binding (the view is the only public object that identifies a frozen witness today)?
