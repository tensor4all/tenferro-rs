# U4-2a: an unadmitted same-runtime, same-device consumer copy of a published value (#1945 U4)

Revision 4a of the consumer half. The registry-authorized, cross-runtime handoff (4b) is deferred in
`docs/design/consumer-read-handoff-1945-u4-2.md`: three independent pre-implementation reviews rejected every
token/registry design because the substrate contracts it needs (an exact-instance binding seam, a two-runtime
authorization model, a consumer-path admission surface, a cycle-free completion owner) do not exist. This
record specifies the part those reviews endorsed as bounded and implementable: **a device-side copy of a
published value into another session on the same CUDA runtime and the same device**, using only existing
machinery.

## 1. Scope and claims

- **In:** `SourcePublication::copy_into(&mut CudaHeldSession<'_>)` - the publication is consumed, the copy runs
  on the consumer session's captured stream behind the publication's frontier, and the caller gets a
  `PendingRead` following the pending-transfer ownership rules.
- **Claimed:** a device-side consumer copy of a published value on the same runtime and device, ordered after
  the producer's frontier, with the source allocation retained until the copy completes.
- **Not claimed:** any authorization beyond same-runtime/same-device eligibility; any dependency token,
  registry binding or run admission (4b); any cross-runtime or cross-device copy; any change to the scheduled
  blocking transfer contract; pageable/P2P/differentiated transfers; scoped-callback publishing; U2-AD.

## 2. Settled decisions

### 2.1 Eligibility is an exact-instance comparison, before any enqueue

The publication already holds its `CudaRuntime`; the consumer session holds the backend it was opened from.
Both sides expose public exact-instance identity - `CudaBackend::runtime_identity()` /
`CudaRuntime::runtime_identity()` and `device_id()` (`crates/tenferro-gpu/src/cubecl/mod.rs`,
`crates/tenferro-gpu/src/cubecl/runtime.rs`) - so the check is a plain comparison:

```rust
impl SourcePublication {
    /// The runtime instance and device this publication belongs to.
    pub fn runtime_identity(&self) -> CudaRuntimeIdentity;
    pub fn device_id(&self) -> CudaDeviceId;
}
```

`copy_into` refuses *before any enqueue* unless both identities and both devices match, so no ownership moves
and no device work is submitted for an ineligible pair. This is the "exact-instance validation" the third
review asked for, without a registry: the two ends are the same runtime instance by construction, which is
what makes the copy safe without a token.

### 2.2 The copy reuses the raw session primitives on the consumer's captured stream

No new substrate, no run, no event-token machinery:

1. **Destination allocation**: `raw::Session::alloc_bytes(nbytes, op) -> DeviceBytes` on the consumer's
   captured stream (`crates/tenferro-gpu/src/cubecl/raw/mod.rs`).
2. **Ordering**: the publication's recorded frontier is waited on the **consumer's** stream with
   `cuda_result::stream::wait_event(stream, frontier, CU_EVENT_WAIT_DEFAULT)` - the same primitive the CUDA
   run already uses for same-domain dependencies (`crates/tenferro-gpu/src/cubecl/event_domain.rs`). The
   producer's stream is not reapplied: the frontier event already represents the producer's writes, and a
   same-device address copy ordered behind it is sufficient.
3. **Copy**: `raw::Session::copy_bytes(dst, src, nbytes, op)` (a device-to-device `memcpy_dtod_async` on that
   stream) from the publication's retained address into the destination allocation.
4. **Completion**: an event recorded on the consumer's stream, so the copy's completion is observable exactly
   as a pending transfer's is.
5. **No `CudaEventDomainRun` is created inside the session**, so session destruction keeps its current
   semantics (`close` submits; it adds no retirement barrier).

### 2.3 Ownership reuses the pending-transfer record, with no cycle

`copy_into` builds an owning submission guard and moves it into the returned `PendingRead`:

- on **refusal** (eligibility, terminal publication) nothing was enqueued, so the publication is returned
  unchanged;
- on **success** the guard becomes the in-flight package;
- on an **unproven submission** the guard is detached into permanent retention, exactly as an unproven
  pending transfer is;
- dropping or forgetting the `PendingRead` never releases the source early: the record is already on the
  consumer runtime's bounded retirement queue (drainable) or its permanent-retention side, and `Drop` never
  waits.

The record is the existing `TransferRetirement { event, value, buffer, resource }`
(`crates/tenferro-gpu/src/cubecl/transfer.rs`), filled as `event` = the consumer completion (a **raw**
`CUevent`, not an owning `CudaEventHandle`), `value` = the source value (which keeps the source allocation
alive), `buffer` = `None`, `resource` = the destination allocation. Because the record holds no strong
`CudaRuntime` and no `CudaEventHandle`, the runtime-owning-completion cycle the third review found cannot
arise, and the last-owner teardown qualification is preserved.

### 2.4 The refusal carrier is explicit and small

The existing `PublicationRejected` carries a `PinnedHostBuffer` that the device-side copy path does not have,
and `PublicationError`'s other variants describe the publication's own consumption. `copy_into` therefore
returns:

```rust
/// A copy that was refused before any enqueue.
#[derive(Debug)]
pub struct CopyRefusal {
    /// Why the copy was refused.
    pub error: crate::Error,
    /// The publication, unchanged: nothing was enqueued.
    pub publication: SourcePublication,
}

impl SourcePublication {
    pub fn copy_into(
        self,
        consumer: &mut CudaHeldSession<'_>,
    ) -> Result<PendingRead, CopyRefusal>;
}
```

`PendingRead` exposes `bytes()` and a consuming `wait() -> Result<Tensor, crate::Error>`; a terminal
publication is refused by the same rule (2.1), so a failed publication can never be copied as if it were
ready.

## 3. Evidence plan (local A100; GPU model, driver and toolkit recorded)

| Claim | Evidence |
| --- | --- |
| Same-runtime/same-device copy is correct | producer and consumer sessions hold **distinct physical stream handles/slots** (asserted on handles, not the logical ids that map modulo the bounded pool); sessions run sequentially on one thread (two live held sessions are rejected) or on separate threads; the copied bytes equal the produced bytes; the consumer stream is synchronized explicitly |
| Eligibility refusals | a consumer session on a different runtime instance, and one on a different device, are refused before any enqueue with the publication returned; the refusal leaves no device work (asserted through the pending-work counters) |
| Ordering | the copy observes the producer's frontier: a producer whose frontier is deliberately still pending yields a `PendingRead` that completes only after the producer finishes, and `wait()` returns the final bytes |
| Ownership | a refusal returns the publication with its retention intact; an unproven submission leaves the record permanently retained; forgetting a `PendingRead` leaks the owned package in place; retention is observed with controlled pending work and resource-release observations, not aggregate counters |
| Exclusion | the publication cannot be reused after `copy_into` (compile-fail fixture: `download_into`/`copy_into` on a moved value); no safe write path exists (#2093) |
| Fault injection | a private, `#[cfg(test)]`-reachable seam at the completion/submission boundary drives a copy whose completion record fails and an allocation failure after the guard is armed; each asserts the resulting ownership/retirement state |
| No regression | pending-transfer, publication and overlap suites keep passing; the audited CUDA test partition is updated; the `transfer_overlap` group (`crates/tenferro-gpu/benches/transfer_paths.rs`) gains a same-runtime consumer arm |

## 4. Open questions for maintainers

1. Should the destination tensor be a `Tensor` in the consumer's session domain (this record's choice, so
   `PendingRead::wait` hands back a normal device tensor) or a `DeviceBytes`-like raw view?
2. Should the eligibility check also require the same *stream pool* (a different pool on the same runtime and
   device would still copy correctly, but the wait-then-copy ordering crosses pools)? This record assumes one
   pool per runtime and asserts distinct handles in the evidence rather than adding a pool check.
