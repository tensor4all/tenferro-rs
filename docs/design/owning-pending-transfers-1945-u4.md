# Owning pending transfers: detached owning submission for device→host (#1945 U4 item 3)

This record specifies the change the source-publication review named as the prerequisite for any
session-independent consumption contract, and follows the repository's normative
[storage-ownership contract](./storage-ownership-contracts.md) rather than the borrowed admission the
delivered routes use.

It supersedes the first revision of this design, which kept blocking resolution in `Drop` and
inherited the borrowed API's error carrier; an independent review rejected both as contract
discrepancies. The first slice is device→host; the host→device direction follows.

## 1. Why, and what is dropped

`download_pending(runtime, &Tensor, buffer)` borrows its source and retains the allocation with a
`ManagedResource`. Nothing can be reclaimed while the copy runs, but the contract requires owning
submission: with a borrow, `mem::forget(pending)` ends the compiler-enforced exclusion while the DMA
can still be outstanding, and retention prevents reclamation, not writes. The contract's transition
table names the three outcomes this slice must have — pre-admission rejection returns the exact
unchanged package, proven completion releases the record once and publishes the owners, and unproven
completion publishes diagnostics with **no owner** while a provider-private record retains everything.

## 2. Specified API

```rust
// crates/tenferro-gpu/src/cubecl/transfer.rs, re-exported from tenferro_gpu::cuda
pub fn download_pending(
    runtime: &CudaRuntime,
    source: Tensor,
    buffer: PinnedHostBuffer,
) -> Result<PendingDownload, TransferSubmitError>;

pub struct PendingDownload { /* owns §3 */ }

impl PendingDownload {
    /// Whether the copy's event has completed. Never waits.
    pub fn is_ready(&mut self) -> crate::Result<bool>;
    /// Wait for the copy.
    ///
    /// `Ok((source, buffer))` after proven completion; `Err(CompletionUnproven)` with diagnostics and
    /// no owner when completion cannot be proven. Never panics.
    pub fn wait(self) -> Result<(Tensor, PinnedHostBuffer), CompletionUnproven>;
}

/// A submission that could not be admitted.
#[derive(Debug)]
pub enum TransferSubmitError {
    /// No enqueue happened: the caller gets the submitted package back.
    Rejected(Box<TransferRejected>),
    /// The enqueue may have happened: the package stays provider-private.
    Unproven { error: crate::Error },
}

/// Pre-admission rejection: no enqueue happened, so the caller keeps what it submitted.
#[derive(Debug)]
pub struct TransferRejected {
    pub error: crate::Error,
    pub source: Tensor,
    pub buffer: PinnedHostBuffer,
}

/// Completion could not be proven: diagnostics only, and the package stays provider-private.
#[derive(Debug)]
pub struct CompletionUnproven {
    pub error: crate::Error,
}
```

Rules:

1. **Owned admission.** The value and the pinned buffer are moved in; only `wait` hands them back. No
   caller borrow escapes, so `mem::forget` leaks the whole package instead of restoring access, and
   soundness does not depend on `Drop` running.
2. **Detached drop.** Dropping an unwaited handle moves its record — event, moved value, pinned buffer,
   allocation retention and runtime pin — into the runtime's transfer-retirement queue and returns
   immediately; there is no device wait in `Drop`. The queue is drained non-blockingly on every
   subsequent transfer and on `synchronize`, releases each record exactly once after its event
   completes, and falls back to a bounded blocking drain (and, if even that fails, permanent retention)
   exactly as the existing workspace-retirement queue does for its own records. A dropped transfer's
   value and buffer are therefore never handed back, and never freed while the device may touch them.
   `mem::forget` never runs `Drop`, so a forgotten handle is not detached: its record — value, buffer,
   retention and event — is leaked in place, which is the same guarantee with less liveness, and is the
   property the borrowed form lacked.
3. **Pre-admission rejection returns the package.** Validation (dtype served, byte length, device
   residency) happens before anything is enqueued, and `TransferRejected` carries the unchanged source
   and buffer back. After enqueue there is no rejection path.
4. **Unproven completion returns no owner.** If the event synchronize fails, `wait` reports
   `CompletionUnproven` and the record is retained provider-privately; the value and buffer are not
   released and not returned. A failure to *destroy* the event after a proven completion is a teardown
   diagnostic, not an unproven completion, because completion was already established.
5. **`!Send + !Sync`.** The copy is enqueued on the opening thread's stream and the event belongs to
   that device context; a transfer is not moved between threads.
6. **Compact `Tensor` only.** `Tensor` is already a move-only physical owner that the blocking routes
   take, so it is the operand here too; no view or `TensorValue` wrapping is introduced, and a
   metadata-only view is simply not an operand of this route (the typed error says to materialize it).

The change replaces the borrowed signatures rather than adding a parallel API: slices 1-2 are
unreleased, and the workspace callers are the GPU tests, one compile-fail fixture, the example, the
benchmark and the crate exports.

## 3. Evidence plan

| Claim | Evidence |
| --- | --- |
| Owned admission and round trip | submit a known device payload and pinned buffer by move, close the producing held session, `wait`, and assert the exact bytes plus the returned source value and buffer (byte-identical, reusable in the next transfer) |
| Exclusion is compile-checked | compile-fail fixture: using the tensor after moving it into `download_pending` fails to compile, including after `mem::forget(pending)` (the value is unreachable in both cases) |
| Detached drop | dropping an unwaited handle returns immediately (no device wait), the value is neither handed back nor freed early, a later transfer is correct, and the retirement queue drains once the event completes |
| Pre-admission rejection | a byte-length mismatch and a rejected dtype return `TransferRejected` carrying the same source and buffer, with no enqueue |
| Unproven completion | a fault-injected failed synchronize reports `CompletionUnproven` and releases nothing; the queue's retention is observable (the record count stays) rather than a freed alias |
| `Drop`/`mem::forget` soundness | neither restores access to the admitted package; a forgotten handle leaks and the runtime stays usable |
| Ordering and numerics | the existing pending tests (delayed copy observed as not-ready, exact bytes after `wait`, completion after the producing session closes, two transfers in flight) adapted to the owning signature |
| No regression | the blocking `upload_tensor`/`download_tensor` routes, the CUDA test partition, the example, the guide and the benchmark are updated in the same change |

## 4. What this does not fix

- **The host→device direction** stays borrowed in this slice and is converted next, with the same
  rules.
- **The safe writable-binding escape** (`Session::tensor_binding` + `Client::with_server` +
  `ComputeServer::write`) still allows safe code to write into storage it does not own. That is a
  pre-existing boundary defect recorded in [source-publication-1945-u4.md](./source-publication-1945-u4.md);
  owning the transfer removes the borrow-based hole, not that one.
- **Source publication, dependent-consumer admission, cross-domain/P2P, view readiness, chunked
  staging, event batching and measured overlap** remain undelivered and are listed in
  [asynchronous-transfer-1945-u4.md](./asynchronous-transfer-1945-u4.md) §5.
- The publication boundary still dispatches through CubeCL's host queue, whose cross-stream GC
  backpressure and staging retirement can wait; this package claims no new evidence there.

## 5. Open questions for maintainers

1. Does the transfer-retirement queue belong in `CudaRuntimeState` next to the workspace queue (as
   specified here), or in the runtime's event-domain retirement substrate so both queues share one
   drain policy?
