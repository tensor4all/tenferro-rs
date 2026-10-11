# Source publication: an owned value with a producer frontier and a domain-scoped dependency (#1945 U4, source publication)

This record specifies the source-publication half of #1945's transfer contract. Its first revision
proposed publishing from a runtime handle and was rejected in review; §1 records why, and what has
changed since. §2 onwards is the revised design.

## 1. What the first revision got wrong, and what has changed

| Rejection | State now |
| --- | --- |
| An asynchronous *borrow* contradicts the storage-ownership contract (owning submission, no escaped caller borrow, soundness independent of `Drop`/`mem::forget`) | **closed**: both pending transfers own their admitted package, detach on drop and leak on `mem::forget` ([owning-pending-transfers-1945-u4.md](./owning-pending-transfers-1945-u4.md)) |
| A borrow is not an immutable lease while a *safe* writable-binding escape exists | **closed**: `ComputeServer::write` is `unsafe` with an explicit authority contract (tensor4all/cubecl#29), so an immutability guarantee is enforceable rather than implied |
| `publish_source(&CudaRuntime, &Tensor)` receives no producer identity (CubeCL records the *creating* stream; `check_errors`/`get_resource` use the current one, so a frontier recorded on the wrong stream can precede the value's production) | **this design**: publication is entered from the held session, which captured its logical stream when it opened and runs every operation on it |
| Cross-stream producer failures, waiting publication, and a destination that hides session entry or lazy source work | addressed by §3 below: the boundary is the same non-blocking one the pending transfers use, and consumption takes no session |

## 2. Specified API

```rust
// crates/tenferro-gpu/src/cubecl/exec_session.rs, re-exported from tenferro_gpu::cuda
impl CudaHeldSession<'_> {
    /// Publish a value whose production ran on this session's stream.
    pub fn publish(&mut self, value: Tensor) -> Result<SourcePublication, TransferSubmitError>;
}

// crates/tenferro-gpu/src/cubecl/transfer.rs
pub struct SourcePublication { /* owns §3 */ }

impl SourcePublication {
    /// Byte length of the published value.
    pub fn bytes(&self) -> usize;
    /// Whether the producer's frontier has completed. Never waits.
    pub fn frontier_ready(&mut self) -> crate::Result<bool>;
    /// Explicit readiness boundary: wait for the producer's frontier.
    pub fn await_frontier(&mut self) -> crate::Result<()>;
    /// Consume the publication by copying it into `buffer`.
    pub fn download_into(self, buffer: PinnedHostBuffer) -> Result<PendingDownload, TransferSubmitError>;
    /// The frontier as a dependency token for exactly `domain`.
    pub fn dependency(&self, domain: EventDomainId) -> crate::Result<Arc<dyn EventToken>>;
}

// The consumer side needs the driver that owns the domain identity.
impl CudaRuntime {
    /// The CUDA event-domain driver, for a caller that owns the domain identity itself.
    pub fn event_domain_driver(&self) -> Arc<dyn tenferro_runtime::EventDomainDriver>;
}
```

## 3. What a publication owns, and what it guarantees

```text
SourcePublication
  ├── value: Tensor            // owned; the caller has no other path to it
  ├── allocation: ManagedResource   // pins; resolved while publishing
  ├── frontier: CUevent        // recorded on the session's captured stream
  ├── stream: StreamId         // the producer binding, captured by the session
```

1. **Producer identity.** `publish` runs inside the held session's captured stream: it dispatches
   pending work through the non-blocking error check and then records the frontier there, so the
   frontier is ordered after every operation the session ran — including the one that produced the
   value. A frontier recorded on an arbitrary current stream cannot promise that, which is why the
   entry point is the session rather than a runtime handle.
2. **Ownership.** The value is moved in; only `download_into` (which consumes the publication) or a
   drop ends it. Dropping detaches the record — value, retention, event, provider pin — into the
   runtime's bounded transfer-retirement queue without waiting for the device; an unproven completion
   or a full queue retains it permanently. `mem::forget` leaks it in place.
3. **Immutability is enforceable.** The safe value model cannot reach the moved value, and reaching
   its storage for a *write* requires an explicit `unsafe` assertion of authority (tensor4all/cubecl#29),
   so the publication can state immutability for storage it owns.
4. **No session entry and no lazy source work in the destination.** `download_into` enqueues the copy
   on the publication's stream using the retained allocation; it takes no session, resolves nothing
   lazily, and never calls the default/global route.
5. **Readiness is explicit and never implicit.** `frontier_ready` is a non-blocking event query;
   `await_frontier` is the documented boundary. Consuming without awaiting is allowed and correct
   because the copy is stream-ordered after the frontier's work; what a caller must not do is read the
   *host* buffer before the copy's own completion, which `PendingDownload` already prevents.
6. **A dependent consumer can wait on the device.** `dependency(domain)` mints the frontier as a token
   that admission accepts for exactly `domain` (origin and token type are both checked), so a consumer
   that begins an event-domain run for that domain and enqueues with the token orders its own work
   after the producer without a host wait.
7. **Cross-domain stays explicit.** A consumer in a *different* execution domain still needs the
   runtime's scheduler to carry a transfer dependency across domains, which today host-waits; that is
   out of scope here and remains recorded in
   [asynchronous-transfer-1945-u4.md](./asynchronous-transfer-1945-u4.md) §5 item 2.

## 4. Evidence plan

| Claim | Evidence |
| --- | --- |
| Producer identity | publish a value produced by a held session, then close the session before consuming: the copy returns exactly the produced bytes |
| Frontier ordering | publish immediately after a deliberately large producer: `frontier_ready` reports `false`, `await_frontier` then succeeds, and the copy is correct |
| Dependent consumer | a consumer thread begins an event-domain run for the publication's domain on a different stream, enqueues with `dependency(domain)`, and its operation observes the produced values; the admission is device-ordered, not host-ordered |
| Domain scoping | a token minted for one domain is refused typed (`DependencyDomainMismatch`) by a run for another |
| Ownership and drop | the value is unreachable after publishing (compile-fail fixture); dropping detaches into the retirement queue and `synchronize` releases it; `mem::forget` leaks in place |
| No session in consumption | `download_into` runs after the source session closed, with no other session open, and takes no session argument |
| No regression | the pending transfer tests, the CUDA test partition, the example, the guide and the benchmark are updated in the same change |

## 5. Out of scope

- Cross-domain scheduling in the runtime (the scheduler host-waits source-domain transfer
  dependencies), pageable staging for publications, P2P routes and differentiated transfers.
- Publishing from the *scoped* callback entry: only the held session is specified here, because it is
  the entry that owns an explicit binding; the scoped case can follow if a caller needs it.

## 6. Open questions for maintainers

1. Should `CudaRuntime::event_domain_driver` be public, or should the driver stay internal and the
   token be minted only for a domain the caller obtained from the runtime registry?
2. Should a publication expose `dependency` at all in this slice, or is `download_into` plus explicit
   readiness the only consumption route worth specifying until cross-domain scheduling exists?

## 7. Revision 3 review: still blocked, and what the second review added

Revision 2 was reviewed and rejected again. The findings are recorded here because they narrow the
problem rather than merely repeating it:

1. **Provenance cannot be established by a safe API taking an arbitrary value.** `publish(value: Tensor)`
   says the value's production ran on the session's stream, but a value can be allocated on one stream
   and written through another (native `_into` operations enqueue writes without changing the
   allocation's *creating* stream identity, which is what CubeCL's binding alignment consults). Such a
   value published through the first session lets the frontier precede its production. The entry point
   therefore has to be an `unsafe` contract ("the caller asserts the value's last write is ordered
   before this session's stream"), or the provenance has to be carried by the type.
2. **A dependency token orders work but does not hand the consumer the value.** Admission validates the
   token's domain and type and then enqueues a stream wait; nothing in that route gives the consumer's
   launch closure a read operand. A publication needs an owning read handoff (a read accessor usable
   inside the run's launch closure) for the dependent-consumer test to be writable at all.
3. **A caller-supplied domain is not provenance.** `EventDomainId`s are minted by a runtime freeze and
   retrieved from the runtime snapshot; the CUDA driver stores whatever domain it is given and does not
   verify that the domain belongs to its runtime. Admission therefore proves equality and token type,
   not "the exact frozen event domain that produced this token". The driver must stay owned by the
   runtime registration rather than exposed for callers to mint into.
4. **The immutability blocker is only half closed.** `ComputeServer::write` is `unsafe` now, but a
   *checked kernel launch* is a second, still-safe write route: a binding obtained for a borrowed tensor
   can be passed as the output of a `#[cube(launch)]` kernel, whose generated launcher and
   `ComputeClient::launch` are safe, and bounds checking does not supply write authority. Until that
   route carries write authority too, a publication cannot unconditionally claim immutable storage.
5. **Lifecycle details** remain to specify: the error carrier for a bufferless publication, failed
   readiness as a terminal outcome, sharing an event with an escaped dependency token, the last-owner
   drop qualification, and re-applying the captured stream in the consumption path (calling the existing
   `download_pending` would re-resolve the producer and reintroduce the submission boundary).

The review's own conclusion is the useful summary: for a device→host-only slice the separate publication
object and a public driver route are unnecessary, and the full "source publication, not mandatory host
export" contract needs a distinct ownership boundary that has not been completed. Two prerequisites are
outside this repository's control as it stands: write authority on the kernel-launch route (a CubeCL-side
mechanism) and a domain-ownership contract for tokens (the runtime registry).

**Status: blocked.** Nothing here is implemented. The pending transfer routes keep their documented
behaviour; a publication built on them must not be assumed to have established provenance, domain
ownership, or immutable storage.
