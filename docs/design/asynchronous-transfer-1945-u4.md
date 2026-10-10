# Asynchronous CPU↔CUDA transfer (#1945 U4)

This record specifies U4 and, after three independent reviews of the full contract, what the first
implementable slice of it is. The full package is **not** delivered here, and this record says
exactly which requirements remain open and why.

Downstream consumer: [tensor4all-rs #859](https://github.com/tensor4all/tensor4all-rs/issues/859)
milestone B2. Performance tracker: [#2009](https://github.com/tensor4all/tenferro-rs/issues/2009).

Revision history: draft 1 proposed a generic `PendingTransfer<T>` framework submitted through
`client.flush`; review round 1 falsified its API shape and its publication claim. Draft 2 replaced
the flush with the `get_resource`/host-dispatch boundary and borrowed its DMA storage; review round 2
showed that a borrowed-DMA handle cannot be retired asynchronously (the repository's
[storage-ownership contract](./storage-ownership-contracts.md) requires owning asynchronous
submission, no escaped caller borrow, and soundness independent of `Drop`/`mem::forget`), that a
same-domain token grants no access to the pending destination, and that `get_resource`'s
`ignore: true, flush: false` mode bypasses producer-error reporting. What follows is the slice the
review recommended.

## 1. Slice 1: owned-buffer pending device→host handoff

Delivered:

- `PinnedHostBuffer`: owned pinned host storage (`cudaHostAlloc`), handed to a transfer **by value**
  and returned by it, so no caller borrow has to outlive a DMA it cannot control. Dropping the
  buffer frees it; a transfer that cannot prove completion abandons it instead (§3).
- `download_pending(rt, src: &Tensor, buffer: PinnedHostBuffer) -> PendingDownload<'_>`: one
  `cuMemcpyDtoHAsync_v2` into that buffer, an event recorded on the copy's stream, and the source
  allocation retained (`ManagedResource`) for the whole flight. `src` is borrowed for the handle's
  lifetime, so the source cannot be dropped or mutated while the copy may read it.
- `PendingDownload::{is_ready, wait}`: `is_ready` queries the event and distinguishes not-ready from
  a device error; `wait(self)` synchronizes the event once and returns the filled buffer.
- `Drop` resolves the event before releasing anything; when it cannot, the buffer and the retained
  resource are leaked rather than freed early. `mem::forget` is sound by construction: the handle
  owns everything it uses, so forgetting it leaks (never frees early).

Explicitly **not** claimed in this slice:

- **Submission is not nonblocking.** The copy uses the existing audited submission path
  (`flush_cubecl`, which also flushes producer errors — see §3). Host dispatch can block on
  cross-stream GC backpressure, and preceding staging retirement can wait on a fence; the slice says
  so instead of claiming device-free publication. The `get_resource`-only variant of draft 2 was
  rejected because it silently drops producer error reporting.
- **No dependent-device-consumer contract.** There is no `dependency()`: a same-domain CUDA event
  token grants ordering but no access to the pending destination, and minting an admissible token
  needs the exact frozen event-domain identity, which this API does not receive.
- **No source-publication/root handoff.** The transfer retains the source allocation; releasing a
  source session or owner *before* completion while keeping the read authority is part of the full
  package (§4).
- **No cross-domain/P2P route, no staging policy enum, no caller-output alias, no AD transfer.**

## 2. Requirements this slice does and does not meet

| Issue #1945 requirement | Slice 1 |
| --- | --- |
| "Pinned H2D/D2H can enqueue and return without a host wait" | **partly**: D2H returns a pending handle without waiting for the copy; the submission itself is not claimed to be wait-free (§1) |
| "An owned/guarded pending result retains source read leases, destination allocation, pinned/staging buffers and provider/event resources" | **yes**, as owned state: `ManagedResource`, the owned pinned buffer, the `CudaRuntime` pin and the event |
| "CPU reads/mutation cannot observe a pending or aliased buffer" | **yes**: the buffer is owned by the handle until `wait()` returns it, and the source is borrowed for the handle's lifetime |
| "GPU consumers import a valid stream/event dependency without host synchronization" | **no**: needs the frozen-domain admission seam (§4) |
| "Completion errors surface at poll/wait/dependency admission; failed outputs are not published as ready" | **partly**: `is_ready`/`wait` report errors and `wait` publishes nothing on failure; there is no dependency admission |
| "Pending-handle drop/cancellation does not release buffers while DMA/kernels may access them" | **yes** (§1) |
| "Reusable caller-owned pinned buffers/output buffers use lifetime guards through completion" | **yes**, by ownership transfer rather than a guard |
| "Staging is bounded/chunked with backpressure and no tensor-sized zero-fill or redundant pageable copies" | **partly**: the pinned buffer is caller-sized and reused; chunking and the pageable route are open (§4) |
| "Source publication, not mandatory host export" | **no** (§4) |
| "Reuse/batch events at producer/transfer frontiers" | **no**: one event per transfer |
| AD is a separate layer | **yes**: plain copies only |

## 3. Ownership and failure semantics

```text
PendingDownload<'a>            // 'a borrows the source tensor
  ├── runtime: CudaRuntime     // provider pin: context and stream stay alive
  ├── buffer: PinnedHostBuffer // owned destination
  ├── resource: ManagedResource<GpuResource>   // source allocation retention
  ├── event: CUevent           // recorded after the copy on the same stream
  └── resolved: bool
```

- The copy is submitted through the existing `flush_cubecl` boundary, which submits pending CubeCL
  work *and* flushes producer errors; a producer failure therefore surfaces as the transfer's own
  error instead of being masked by a successfully completed copy.
- `wait()`: `cuEventSynchronize`; on success the retained resource is released and the buffer is
  returned; on failure nothing is published and the buffer/resource are abandoned (leaked), because a
  failed copy cannot prove what the device is still touching.
- `Drop` without `wait`: resolves the event if it can (non-blocking query, then a bounded
  synchronize), then releases; otherwise abandons.
- `is_ready()`: `cuEventQuery`; a non-ready result is `Ok(false)`, and a device error is reported
  rather than collapsed into "not ready".

## 4. Remaining U4 work (needs maintainer decisions)

1. **Source publication and source-root handoff**: capture an immutable storage/read lease plus the
   producer frontier, end the source's numerical admission, and let the destination be entered
   afterwards. The U1 record leaves this to U4; `ManagedResource` retains an allocation but does not
   grant or transfer read authority. This is the first blocker for a real source-publication
   contract.
2. **Dependent-consumer access**: a device consumer must receive both ordering *and* authorized
   access to the pending destination. That needs a destination-access handoff in the value model, and
   tokens minted for the exact frozen event domain (not derived from a runtime handle). Cross-domain
   admission currently host-waits source-domain dependencies.
3. **Owning asynchronous submission** as the repository's storage-ownership contract requires, with
   the publication boundary shown not to block on device progress (cross-stream GC backpressure and
   staging retirement both can today).
4. **Bounded chunked staging, event amortization, and the route capability model** (including why
   P2P is unavailable on a given device pair).
5. **#2009's measurements** (small/large, pageable/pinned/caller-output, copy bytes, event/wait cost,
   measured overlap) once the routes above exist.

## 5. Open questions for maintainers

1. Does slice 1 belong in `tenferro-gpu`'s CUDA module only, or should the pending contract be stated
   in `tenferro-tensor` so other backends can adopt it deliberately?
2. For §4.1, is the intended source lease a new owned type in `tenferro-tensor`, or an extension of
   the existing borrowed-read contract?
