# Asynchronous CPU↔CUDA transfer (#1945 U4)

This record specifies U4 and, after three independent reviews of the full contract, what the first
implementable slice of it is. The full package is **not** delivered here, and this record says
exactly which requirements remain open and why.

Downstream consumer: [tensor4all-rs #859](https://github.com/tensor4all/tensor4all-rs/issues/859)
milestone B2. Performance tracker: [#2009](https://github.com/tensor4all/tenferro-rs/issues/2009).

Two slices are delivered: the D2H handoff and its H2D mirror.

Revision history: draft 1 proposed a generic `PendingTransfer<T>` framework submitted through
`client.flush`; review round 1 falsified its API shape and its publication claim. Draft 2 replaced
the flush with the `get_resource`/host-dispatch boundary and borrowed its DMA storage; review round 2
showed that a borrowed-DMA handle cannot be retired asynchronously (the repository's
[storage-ownership contract](./storage-ownership-contracts.md) requires owning asynchronous
submission, no escaped caller borrow, and soundness independent of `Drop`/`mem::forget`), that a
same-domain token grants no access to the pending destination, and that `get_resource`'s
`ignore: true, flush: false` mode bypasses producer-error reporting. What follows is the slice the
review recommended.

## 1. Delivered: owned-buffer pending handoffs (slices 1 and 2)

Delivered:

- `PinnedHostBuffer`: owned pinned host storage (`cudaHostAlloc`), handed to a transfer **by value**
  and returned by it, so no caller borrow has to outlive a DMA it cannot control. Dropping the
  buffer frees it; a transfer that cannot prove completion abandons it instead (§3). Bytes can be
  initialized (`as_mut_slice`) before a H2D and inspected (`as_slice`) after a D2H.
- `download_pending(rt, src: &Tensor, buffer: PinnedHostBuffer) -> PendingDownload<'_>`: one
  `cuMemcpyDtoHAsync_v2` into that buffer, an event recorded on the copy's stream, and the source
  allocation retained (`ManagedResource`) for the whole flight. `src` is borrowed for the handle's
  lifetime, so the source cannot be dropped or mutated while the copy may read it.
- `PendingDownload::{is_ready, wait}`: `is_ready` queries the event and distinguishes not-ready from
  a device error; `wait(self)` synchronizes the event once and returns the filled buffer.
- `upload_pending(rt, source: PinnedHostBuffer, dst: &mut Tensor) -> PendingUpload<'_>`: the mirror
  copy on the same submission and event discipline. The handle holds the destination's **mutable
  borrow** for the whole flight — enforced by the compiler, with a compile-fail fixture in
  `tests/ui` — so no reader, writer or dropper of the device tensor can run while the copy may
  still write it. `wait(self)` returns the source buffer for reuse.
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
| "Pinned H2D/D2H can enqueue and return without a host wait" | **partly**: both directions return a pending handle without waiting for their copy; the submission itself is not claimed to be wait-free (§1) |
| "An owned/guarded pending result retains source read leases, destination allocation, pinned/staging buffers and provider/event resources" | **yes**, as owned state: `ManagedResource`, the owned pinned buffer, the `CudaRuntime` pin and the event |
| "CPU reads/mutation cannot observe a pending or aliased buffer" | **yes**: the D2H buffer is owned by the handle until `wait()` returns it, the D2H source is borrowed for the handle's lifetime, and the H2D destination is mutably borrowed |
| "GPU consumers import a valid stream/event dependency without host synchronization" | **no**: needs the frozen-domain admission seam (§4) |
| "Completion errors surface at poll/wait/dependency admission; failed outputs are not published as ready" | **partly**: `is_ready`/`wait` report errors and `wait` publishes nothing on failure; there is no dependency admission |
| "Pending-handle drop/cancellation does not release buffers while DMA/kernels may access them" | **yes** (§1) |
| "Reusable caller-owned pinned buffers/output buffers use lifetime guards through completion" | **yes** in both directions: the source/destination buffer is owned by the handle and returned to the caller on `wait`, and the H2D destination is exclusively borrowed |
| "Staging is bounded/chunked with backpressure and no tensor-sized zero-fill or redundant pageable copies" | **partly**: the pinned buffer is caller-sized and reused; chunking and the pageable route are open (§4) |
| "Source publication, not mandatory host export" | **no** (§4) |
| "Reuse/batch events at producer/transfer frontiers" | **no**: one event per transfer |
| AD is a separate layer | **yes**: plain copies only |

## 3. Measured: the pending routes against the existing blocking paths (#2009)

Protocol: `crates/tenferro-gpu/benches/transfer_paths.rs`, criterion, single-threaded harness, one
CUDA stream, 50 samples with 2 s warm-up and 5 s measurement per case, on an NVIDIA A100 80GB PCIe
with driver 580.126.09 (no CPU pinning; the measured work is device-side). Pinned cases reuse one
buffer across iterations — `wait` hands it back and the next iteration hands it in — so the
measurement covers the copy and its event, not allocation. `pinned_wait` puts the enqueue in
criterion's untimed setup, so it is the completion wait alone. Medians in µs:

| case | 8 B | 4 KB | 1 MB |
| --- | ---: | ---: | ---: |
| download `pageable` (`download_tensor`) | 29.7 | 32.4 | 482.8 |
| download `pinned` (`download_pending` + `wait`) | 29.6 | 29.5 | 350.3 |
| download `pinned_wait` (wait only) | 5.18 | 6.01 | 323.3 |
| upload `staging` (`upload_tensor`) | 3.85 | 18.2 | 3594.6 |
| upload `pinned` (`upload_pending` + `wait`) | 29.2 | 30.9 | 1167.9 |
| pinned allocation, 1 MB | — | — | 1703.4 |

What the numbers support, and what they do not:

- **Large payloads favour the pinned routes**: 1 MB D2H is 350 µs pinned against 483 µs pageable
  (~27% less) and 1 MB H2D is 1.17 ms pinned against 3.59 ms staged (~3x less). The staged upload
  number reflects that CubeCL's staging path pays a completion wait per call, which is exactly the
  per-call cost #2009 tracks.
- **Small payloads do not**: the pending route carries a ≈25 µs fixed cost (audited flush, resource
  resolution, event creation) against 3.9 µs for a staged 8 B upload, so it is not a drop-in
  replacement for the existing paths. That is why slices 1-2 add a route rather than reroute the
  blocking APIs.
- **The absolute D2H rate is still below the reference**: 1 MB in 323 µs is about 3.2 GB/s against
  #2009's 5-13 GB/s cudarc reference, so the copy itself, the submission boundary and the retirement
  of staged bytes remain the open performance work. These numbers are not an app-level speedup
  claim, and the pending route's point is that a caller may wait later than the copy ends.

## 4. Ownership and failure semantics

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

## 5. Remaining U4 work (needs maintainer decisions)

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

## 6. Open questions for maintainers

1. Does slice 1 belong in `tenferro-gpu`'s CUDA module only, or should the pending contract be stated
   in `tenferro-tensor` so other backends can adopt it deliberately?
2. For §4.1, is the intended source lease a new owned type in `tenferro-tensor`, or an extension of
   the existing borrowed-read contract?
