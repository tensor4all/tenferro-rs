# Source publication: an owned value with a producer frontier (revision 6) (#1945 U4 item 1)

This record specifies source publication - "a source session publishes an immutable storage/read lease
plus the producer's ready token or submitted completion frontier ... the source root session/eager owner
need not stay held ... no lazy source work or session entry is hidden inside the destination copy"
(#1945 transfer contract, item 1) - and records what earlier revisions got wrong.

## 1. History, and what changed

Two revisions were rejected, and both rejections were about the substrate rather than the API shape:

| Rejection | State now |
| --- | --- |
| An asynchronous *borrow* contradicts the storage-ownership contract | closed: the pending transfers own their admitted package (#2073, #2076) |
| A borrow is not an immutable lease while a *safe* writable-binding escape exists | closed for tenferro's surface: `ComputeServer::write` is `unsafe` (tensor4all/cubecl#29) and the extension accessors that derive a binding from a borrow are `unsafe` with the write-authority contract (#2093). The residue is cubecl's own checked launch, which is the upstream-facing part of finishing the fork |
| A safe API taking an arbitrary value cannot establish producer provenance | addressed by §2: publication is `unsafe`, and the contract is the caller's assertion |
| A dependency token orders work but hands the consumer no read operand | addressed by narrowing: this revision exposes **no read accessor at all**, so the unsupported device-consumer route is not advertised |
| A caller-supplied event-domain id is not provenance, and a public driver escape is wrong | **deferred to U4-2** (§5): with no consumer-side route in this revision, no token is minted at all; the consumer half is specified in `docs/design/consumer-read-handoff-1945-u4-2.md` |
| The consumer half was unmeasured, and the second review asked for the cross-stream and overlap claims to be stated as gaps | closed as far as the source half can close it: revision 4's implementation merged (#2101) and producer/consumer overlap is now measured with a controlled schedule (#2106); readiness across *different* streams and host-wait-free publication remain the U4-2 gap (§4, §5) |
| The source-side immutability residue is cubecl's own checked launch | closed for tenferro's surface (#2093); the substrate decision and the fork's exit are recorded in `docs/design/cubecl-fork-exit-1945.md` (#2102) |

## 2. Specified API

```rust
// crates/tenferro-gpu/src/cubecl/exec_session.rs, re-exported from tenferro_gpu::cuda
impl CudaHeldSession<'_> {
    /// Publish a value whose production is ordered before this session's captured stream.
    ///
    /// # Safety
    /// The caller asserts that the value's last write is ordered before this session's stream: a value
    /// produced by another stream (for example one written into through a `_into` route on a different
    /// session) does not satisfy the contract, because neither the allocation handle nor the session can
    /// prove who wrote last.
    pub unsafe fn publish(&mut self, value: Tensor) -> Result<SourcePublication, PublicationAdmissionError>;
}

// crates/tenferro-gpu/src/cubecl/transfer.rs
pub struct SourcePublication { /* owns §3 */ }

impl SourcePublication {
    /// Byte length of the published value.
    pub fn bytes(&self) -> usize;
    /// Whether the producer's frontier has completed. Never waits.
    ///
    /// # Errors
    ///
    /// Returns the recorded failure once the publication is terminal, so a failed publication can never
    /// report readiness.
    pub fn frontier_ready(&mut self) -> crate::Result<bool>;
    /// Explicit readiness boundary. A failure is terminal (§3.4).
    pub fn await_frontier(&mut self) -> Result<(), PublicationAdmissionError>;
    /// Consume the publication by copying it into `buffer`.
    ///
    /// Reuses the transfer's own error carrier, because consumption can refuse before any enqueue and must
    /// then hand both submitted owners back.
    pub fn download_into(
        self,
        buffer: PinnedHostBuffer,
    ) -> Result<PendingDownload, PublicationError>;
}

/// A publication that could not be admitted, or whose completion could not be proven.
#[derive(Debug)]
pub enum PublicationAdmissionError {
    /// Nothing was enqueued, so the caller keeps the value.
    Rejected(Box<PublicationRejected>),
    /// The enqueue may have happened, or the publication already failed terminally: no owner is returned.
    Unproven { error: crate::Error },
}
```

## 3. Semantics

1. **Producer identity.** `publish` runs inside the session's captured stream and records the frontier
   there after the same non-blocking dispatch the pending transfers use, so the frontier is ordered
   after everything the session ran. It does *not* make the frontier correct for a value produced
   elsewhere - that is what the `unsafe` contract is for, and why the entry point is the session rather
   than a runtime handle.
2. **Ownership and consumption, with no public read accessor.** The value moves into the publication and
   is reachable only through `download_into`, which copies it into a caller-owned pinned buffer. There is
   deliberately **no public `value()`**: an owned `&Tensor` carries storage lifetime and Rust read authority,
   but not the publication's readiness, because ordinary CubeCL resource alignment follows the allocation's
   *creating* stream rather than this publication's event - so a consumer handed that borrow on a third
   stream could read before the producer's work completed. A device-consumer route needs the dependency
   half (U4-2); until then the honest route back is the retained copy.
   `download_into` copies from the retained allocation on the captured stream: it does not call
   `get_resource`, does not run the producer's dispatch or error check again, enters no session, and never
   falls back to the default route. The retained fields are the runtime pin, the **owned value plus the
   resolved device address** (the resolution guard itself is dropped once the address is recorded, because
   the owned value keeps the allocation alive), the byte extent, the captured logical `StreamId` and the
   recorded frontier event; consumption re-applies the captured stream through `StreamId::executes` and
   enqueues the copy there.
3. **Immutable by construction in tenferro's safe API.** Safe code cannot reach the value for a write:
   the value model cannot alias it, and the extension accessors that could produce a writable
   descriptor require an `unsafe` assertion of ownership (#2093). The remaining hole is cubecl's own
   checked launch, recorded in §5.
4. **Terminal failure.** A failed `await_frontier` marks the publication terminal: the failure is recorded,
   `frontier_ready` reports it instead of readiness, a further `await_frontier` refuses without releasing
   anything, and a further `download_into` refuses **before any enqueue** with
   `PublicationError::Rejected`, handing the publication and the buffer back. The
   record is retained until the publication is dropped, at which point it goes to the permanent-retention
   side of the retirement queue rather than the drainable side. A *successful* `await_frontier` does **not**
   release the retention - unlike a pending transfer's `resolve` - because the publication still needs its
   allocation for a later `download_into`. Dropping a publication detaches its record into
   the runtime's bounded transfer-retirement queue without waiting; `mem::forget` leaks it in place; and
   dropping the *last* runtime owner still runs the runtime's documented teardown, which synchronizes
   its streams (the same qualification the owning transfer record states).
5. **Rejection returns the value.** Validation that happens before any enqueue - an unserved dtype, a
   host tensor, an unresolvable allocation, a producer failure reported at the dispatch boundary -
   returns the value unchanged as `PublicationError::Rejected`.

## 4. Evidence plan

| Claim | Evidence |
| --- | --- |
| Publication survives its session | produce a value in a held session, publish it, `close()` the session, then `download_into(...).wait()`: the exact produced bytes come back |
| Producer frontier | publish immediately after a deliberately large producer: `frontier_ready()` reports `false`; `await_frontier()` then succeeds and the copy is correct |
| Retention without the source handles | drop the `CudaBackend` handle (and close the session) while only the publication holds the runtime pin and the allocation, then consume: the exact bytes come back |
| Immutability in safe API | compile-fail fixture: the value cannot be used, moved or mutated after `publish`; and the extension accessors that could write are `unsafe` (#2093) |
| Lifecycle | dropping a publication detaches into the retirement queue and `synchronize` releases it; `mem::forget` leaks in place; a rejected publication returns the value |
| No session in consumption | `download_into` runs after the source session closed, with no other session open, and takes no session argument |
| No regression | the pending transfer tests keep passing and the audited CUDA test partition is updated; the example, guide and benchmark still describe and measure only the pending routes, which this slice does not change |

A fault-injected failed `await_frontier` (the terminal path) is **not** delivered in this revision: the
path records the failure, reports it through `frontier_ready` and retains the record, but no test drives it.
A simulated provider failure (rather than a malformed native handle) is the way to exercise it, and #1945
asks for fault-injected production-path evidence, so it is listed as a gap rather than dressed up.

Publication is also **not** claimed to be host-wait-free: the dispatch and resource resolution it performs
can still meet CubeCL's bounded GC backpressure and staging retirement, exactly as the pending transfers
document.
**Cross-stream readiness and host-wait-free publication are still the U4-2 gap.** Every test here uses the
ambient stream a session captures when it opens. Producer/consumer *overlap* is now measured with a
controlled schedule (#2106, the `transfer_overlap` group in `crates/tenferro-gpu/benches/transfer_paths.rs`),
but readiness
across *different* streams and a publication that never host-waits remain unestablished - which is also why
the rejected-length path returns the publication with its frontier instead of relying on them. The consumer
half is specified in `docs/design/consumer-read-handoff-1945-u4-2.md`; it claims a device-side consumer copy,
**not** a host-wait-free scheduled transfer (that path's provider contract stays blocking).


## 5. What this revision defers, and why

- **Dependent-consumer dependency tokens and cross-domain admission (U4-2).** Specified in
  `docs/design/consumer-read-handoff-1945-u4-2.md`: an admitted pair obtained from the selected frozen
  executable slot authorizes one publication/consumer route, the recorded frontier is adopted into the
  existing event-token owner, and the destination run admits it in place of the runtime's host bridge for
  that edge. With no consumer-side route here, this revision mints no token: event-domain ids are minted by
  a runtime freeze, the CUDA driver stays owned by the runtime registration rather than exposed, and the
  scheduler currently host-waits source-domain dependencies. Possessing an id or a token with a matching
  `origin` is not authorization - `EngineSnapshotView::event_domain_id` and the transfer accessors expose
  ids of existing things, and the immediate driver can mint a token bearing an id it was given - which is
  why the admitted pair, not the id, is the provenance.
- **cubecl's own surface.** Its checked launch still accepts a lifetime-free binding as a writable
  argument; marking launches with writable arguments `unsafe` is the fork-finishing change owed to
  cubecl's users (the fork's own test harness and samples included).
- **The consuming-copy rejection path** hands the publication back, including its frontier, rather than
  destroying the frontier and returning a bare value: a value whose handle identifies its creating
  stream carries no readiness, which is the same reason there is no public read accessor. Returning the
  publication keeps the caller able to fix the buffer and consume, and nothing is destroyed.
- Pageable staging for publications, P2P routes, differentiated transfers, and publishing from the
  *scoped* callback entry (the held session is the entry that owns an explicit binding).

## 6. Open questions for maintainers

1. Is `unsafe fn publish` the right provenance contract, or should the value carry its producer (a
   session-tagged value type threaded through the operation routes, at the cost of touching every
