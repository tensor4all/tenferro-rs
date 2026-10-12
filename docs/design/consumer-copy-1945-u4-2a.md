# U4-2a: an unadmitted same-runtime consumer copy of a published value (#1945 U4)

Revision 2. Revision 1 was reviewed before implementation; the review confirmed the scope ("a bounded,
plausible slice", "the identity comparison is sound") and the ordering shape, and required corrections that
do not reopen registry authorization or add substrate. They are applied below and recorded in §1.

The registry-authorized, cross-runtime handoff (4b) stays deferred in
`docs/design/consumer-read-handoff-1945-u4-2.md`.

## 1. What revision 2 changed

| Revision-1 issue | Correction |
| --- | --- |
| `CopyRefusal` cannot represent an unproven submission (it always returns the publication) | the error channel is the existing `PublicationError`, with its rejection representation extended for a device copy (no pinned buffer); `Unproven` stays diagnostics-only |
| the rationale misidentified the carrier | the pinned buffer belongs to `PublicationError::Rejected`; `PublicationRejected` holds only `error` and `value` |
| `DeviceBytes` cannot inhabit the retirement record (`!Send + !Sync`, session-bound) or become the promised tensor | the destination is an **owned typed tensor** allocated through the existing typed allocation path, matching the source's shape and dtype |
| "already on the queue" contradicts an owning `PendingRead` | the package is owned by `PendingRead` until `wait`, detachment on drop, or terminal failure; the queue receives it only at detachment |
| stream reapplication alone was claimed to be enough | the route goes through the consumer's captured-stream `with_session`/`with_raw` protocol, which reapplies the logical stream **and** activates/restores the CUDA context |
| public publication identity getters | deleted: `copy_into` compares the existing runtime fields internally |
| stream-pool eligibility question | deleted: the runtime owns its bounded stream mapping |

## 2. Scope and claims

- **In:** `SourcePublication::copy_into(&mut CudaHeldSession<'_>)` - the publication is consumed, the copy runs
  on the consumer session's captured stream behind the publication's frontier, and the caller gets a
  `PendingRead` following the pending-transfer ownership rules.
- **Claimed:** a device-side consumer copy of a published value on the same runtime, ordered after the
  producer's frontier, with the source allocation retained until the copy completes.
- **Not claimed:** authorization beyond same-runtime eligibility; any dependency token, registry binding or
  run admission (4b); cross-runtime or cross-device copies; any change to the scheduled blocking transfer
  contract; pageable/P2P/differentiated transfers; scoped-callback publishing; U2-AD.

## 3. Settled decisions

### 3.1 Eligibility is an internal exact-runtime comparison, before any enqueue

`copy_into` runs in the module that owns `SourcePublication.runtime`, and the consumer session exposes its
runtime through the existing execution-session visitor, so no new public accessor is needed. Exact runtime
identity is per-instance (`runtime_identity()` compares with `Arc::ptr_eq`; every construction creates a
fresh marker), so comparing the two runtime identities is sufficient - device equality is implied and is not
checked separately. Independently constructed backends never compare equal; clones do.

A mismatch refuses **before any enqueue**, so no ownership moves and no device work is submitted.

### 3.2 The copy goes through the captured-stream protocol, then waits, copies and records

The route uses the consumer session's existing captured-stream entry (`with_session`/`with_raw`), which
reapplies the logical stream **and** activates/restores the CUDA context - `StreamId::executes` alone does
not, and neither `raw::Session` nor `copy_bytes` activates a context. Inside that scope:

1. **Destination**: an owned typed device tensor matching the source's shape and dtype, allocated through the
   existing typed allocation path on the consumer runtime. It is owned by the package (not a session-bound
   `DeviceBytes`), so it survives into the retirement record and can become the tensor `PendingRead::wait`
   returns.
2. **Ordering**: the publication's recorded frontier is waited on the consumer's stream with
   `cuda_result::stream::wait_event(stream, frontier, CU_EVENT_WAIT_DEFAULT)` - the primitive the CUDA run
   already uses for same-domain dependencies. The producer's stream is not reapplied: the frontier event
   already represents the producer's writes, and a same-device address copy ordered behind it is sufficient.
   The recorded source address stays valid because the publication retains the owned value (the resolution
   guard is dropped only after the value and address are kept).
3. **Copy**: `raw::Session::copy_bytes(dst, src, nbytes, op)` (device-to-device on that stream).
4. **Completion**: an event recorded on the consumer's stream, exactly as a pending transfer records one.
5. **No `CudaEventDomainRun` is created inside the session**, so session destruction keeps its current
   semantics.

### 3.3 Ownership follows the pending-transfer model exactly

`PendingRead` **owns** the package - the source value (which keeps the source allocation alive), the
destination tensor and the completion event - until one of:

- **`wait()`** succeeds: the record is removed and released once its event has completed, and the destination
  tensor is returned;
- **drop**: the package detaches into the consumer runtime's bounded retirement queue (drainable), which
  releases it once the event completes; `Drop` itself never waits;
- **terminal failure**: the package detaches into permanent retention.

`mem::forget(PendingRead)` leaks the owned package in place; it does not enqueue it. The record is the
existing `TransferRetirement { event, value, buffer, resource }` filled as `event` = the consumer completion
(a **raw** `CUevent`), `value` = the source value, `buffer` = `None`, `resource` = the owned destination
tensor (erased on detachment). Neither `SourcePublication` nor an owning `CudaEventHandle` ever enters the
queue, so no runtime-owning completion cycle arises and the last-owner teardown qualification is preserved:
detachment adds no device wait, but releasing the **last** runtime pin can still trigger runtime teardown,
which synchronizes initialized streams.

### 3.4 Errors reuse the existing carrier

`copy_into` returns `Result<PendingRead, PublicationError>`:

- `PublicationError::Rejected` for a refusal before any enqueue (eligibility mismatch, terminal publication,
  allocation failure), with the rejection representation extended so a device copy without a pinned buffer can
  be returned along with the unchanged publication;
- `PublicationError::Unproven` for an unproven submission, carrying diagnostics only, with the package
  permanently retained.

A terminal publication is refused with its `failure`, `value`, `frontier` and `terminal_recorded` fields
intact, so its eventual drop still follows the source half's permanent-retention path.

## 4. Evidence plan (local A100; GPU model, driver and toolkit recorded)

| Claim | Evidence |
| --- | --- |
| Same-runtime copy is correct | **sequential** sessions on one thread (separate threads are rejected: the reservation is contended and `SourcePublication` is not `Send`), with explicitly selected logical streams and asserted **distinct physical stream handles/slots**; the copied bytes equal the produced bytes; the consumer stream is synchronized explicitly |
| Ordering | the frontier is observed still pending **at the copy boundary** (not earlier, since destination preparation can consume time); a producer whose frontier is deliberately pending yields a `PendingRead` that completes only after the producer finishes, and `wait()` returns the final bytes |
| Eligibility refusal | a consumer session on a different runtime instance is refused, with the publication returned and **zero visits to the copy/wait/record submission boundary** asserted through the private seam (session counters and retirement counts do not prove "no enqueue") |
| Different-device refusal | exercised only with a second real device; otherwise reported as untested rather than manufactured |
| Ownership | a refusal returns the publication with its retention intact; an unproven submission leaves the package permanently retained; forgetting a `PendingRead` leaks it in place; release is observed on package/handle state (pooled allocation handles, not address reuse) |
| Exclusion | the publication cannot be reused after `copy_into` (trybuild fixture in the existing harness: `download_into`/`copy_into` on a moved value); no safe write path exists (#2093) |
| Fault injection | a private, `#[cfg(test)]`-reachable seam at the submission boundary drives a failed completion record and an allocation failure after the guard is armed (a refusal, not an unproven submission); the existing frontier query/wait failure path is driven for the terminal-source claim |
| No regression | pending-transfer, publication and overlap suites keep passing; the audited CUDA test partition is updated; the `transfer_overlap` group (`crates/tenferro-gpu/benches/transfer_paths.rs`) gains a same-runtime consumer arm |

## 5. Open question for maintainers

1. Should the extended rejection representation make the existing buffer field optional, or add a
   device-copy rejection variant? This record has no preference; both keep `Rejected`/`Unproven` semantics.
