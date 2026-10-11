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
- `download_pending(rt, source: Tensor, buffer: PinnedHostBuffer) -> Result<PendingDownload, TransferSubmitError>`:
  one `cuMemcpyDtoHAsync_v2` into that buffer, an event recorded on the copy's stream, and the source
  allocation retained (`ManagedResource`) for the whole flight. The value and the buffer are **moved
  in** — the owning admission described in
  [owning-pending-transfers-1945-u4.md](./owning-pending-transfers-1945-u4.md) — so neither is
  reachable while the copy may read or write them; `wait` hands both back, a refused submission returns
  the package untouched, dropping an unwaited handle detaches the record into the runtime's retirement
  queue, and forgetting the handle leaks it in place.
- `PendingDownload::{is_ready, wait}`: `is_ready` queries the event and distinguishes not-ready from
  a device error; `wait(self)` synchronizes the event once and returns the filled buffer.
- `upload_pending(rt, source: PinnedHostBuffer, destination: Tensor) -> Result<PendingUpload, TransferSubmitError>`:
  the mirror copy on the same submission, ownership and event discipline: the buffer and the
  destination are **moved in**, `wait` returns both, a refusal returns them unchanged, and a
  compile-fail fixture proves the destination cannot be used after the move.
- `Drop` detaches the record into the runtime's bounded retirement queue, which releases it once its
  event proves completion and retains it permanently otherwise
  ([owning-pending-transfers-1945-u4.md](./owning-pending-transfers-1945-u4.md)); a completion that
  cannot be proven never frees the value or the buffers the device may still touch. `mem::forget` is sound by construction: the handle
  owns everything it uses, so forgetting it leaks (never frees early).

Explicitly **not** claimed in this slice:

- **Publication does not wait for *its own* retirement, but a device-free publication is not claimed.**
  The copy is published through CubeCL's `check_errors` (§3), which reports producer errors and
  dispatches the host queue without rotating the drop queue, so the boundary no longer performs a
  device wait of its own — the two earlier alternatives failed here: `flush` waits the previous fence,
  and `get_resource` alone silently drops producer error reporting. What remains is host dispatch
  (a blocking host round trip), cross-stream resource resolution (bounded GC backpressure) and a
  preceding kernel that trips the staging policy. Those are named, not hidden.
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
| "Pinned H2D/D2H can enqueue and return without a host wait" | **partly**: both directions return a pending handle without waiting for their copy, and their publication boundary no longer performs its own drop-queue rotation (CubeCL's `check_errors`). Two waits remain: host dispatch is a blocking host round trip, and both cross-stream resource resolution (bounded GC backpressure) and a preceding kernel that trips the staging policy can still wait for device progress (§3, §5) |
| "An owned/guarded pending result retains source read leases, destination allocation, pinned/staging buffers and provider/event resources" | **yes**, as owned state: `ManagedResource`, the owned pinned buffer, the `CudaRuntime` pin and the event |
| "CPU reads/mutation cannot observe a pending or aliased buffer" | **yes in both directions**: the value and the buffer are owned by the handle until `wait()` returns them |
| "GPU consumers import a valid stream/event dependency without host synchronization" | **no**: needs the frozen-domain admission seam (§4) |
| "Completion errors surface at poll/wait/dependency admission; failed outputs are not published as ready" | **partly**: `is_ready`/`wait` report errors and `wait` publishes nothing on failure; there is no dependency admission |
| "Pending-handle drop/cancellation does not release buffers while DMA/kernels may access them" | **yes** (§1) |
| "Reusable caller-owned pinned buffers/output buffers use lifetime guards through completion" | **yes in both directions**: the buffer and the value are owned until `wait` hands them back |
| "Staging is bounded/chunked with backpressure and no tensor-sized zero-fill or redundant pageable copies" | **partly**: the pinned buffer is caller-sized and reused; chunking and the pageable route are open (§4) |
| "Source publication, not mandatory host export" | **no** (§4) |
| "Reuse/batch events at producer/transfer frontiers" | **no**: one event per transfer |
| AD is a separate layer | **yes**: plain copies only |

## 3. Measured: the pending routes against the existing blocking paths (#2009)

The boundary these numbers measure became non-blocking for device work in the same change that
adopted it: tensor4all/cubecl#27 adds `Client::check_errors`, which reports a stream's producer errors
and dispatches the host task queue **without** retiring staged bytes, and therefore without performing
its own drop-queue rotation. Tenferro's raw-copy boundaries call it instead of `flush`. Adopting it
needs one CubeCL revision in the whole graph, which is why tensor4all/cubek#16 and #17 bump that fork's
pin on its release branch, and tensor4all/cubecl#28 makes the CUDA **write** path enforce the staging
policy as well — with the boundary no longer flushing, a transfer-only workload that stages host bytes
and runs no kernel would otherwise retain them, because the policy counters are only evaluated where
they are checked.

Protocol: `crates/tenferro-gpu/benches/transfer_paths.rs`, criterion, single-threaded harness, one CUDA
stream, 50 samples with 2 s warm-up and 5 s measurement per case, on an NVIDIA A100 80GB PCIe with
driver 580.126.09, no CPU pinning. Payloads are `f64` element counts; the table labels their byte
sizes. Pinned cases reuse one buffer across iterations — `wait` hands it back and the next iteration
hands it in — so they cover the copy and its event, not allocation. `pinned_wait` puts the enqueue in
criterion's untimed setup. These are host wall-clock numbers that include host dispatch work, and the
values are criterion's displayed central estimates, not medians.

| case | 16 B | 64 B | 32 KiB | 8 MiB |
| --- | ---: | ---: | ---: | ---: |
| download `pageable` (`download_tensor`) | 22.4 -> 22.0 | 30.3 -> 29.7 | 32.5 -> 32.0 | 517.6 -> 488.5 us |
| download `pinned` (`download_pending` + `wait`) | 30.3 -> **22.0** | 29.8 -> **21.1** | 30.6 -> **22.2** | 350.6 -> 345.8 us |
| download `pinned_wait` (wait only) | 5.16 -> 5.18 | 5.02 -> 5.17 | 5.99 -> 6.07 | 323.1 -> 323.2 us |
| upload `staging` (`upload_tensor`) | 3.89 -> 3.73 | 3.93 -> 3.87 | 18.68 -> 18.40 | 3.77 -> 4.15 ms |
| upload `pinned` (`upload_pending` + `wait`) | 28.4 -> **22.0** | 29.4 -> **21.8** | 31.7 -> **23.1** | 1.10 ms -> **349.4 us** |
| pinned allocation, 8 MiB | — | — | — | 6.35 -> 6.32 ms |

What the numbers support, and what they do not:

- **The publication path got cheaper**: the pending routes are 22-29% faster from 16 B to 32 KiB in
  both directions, and an 8 MiB pinned upload is ~68% faster. The copy-bound 8 MiB download moves only
  ~1.4% (the pageable variant moves 5.6%, which is the boundary's share of that path).
- **The wait row is not pure copy time and does not attribute the change**: `pinned_wait` excludes the
  enqueue but still includes synchronization, retention release and event teardown, and it is
  essentially unchanged (0-3% at the small sizes, identical at 8 MiB) — which bounds the change to the
  publication path but does not prove "exactly the boundary cost".
- **The mechanism behind the 8 MiB upload win is not established here**: this serial
  enqueue-then-wait benchmark cannot separate a previous-fence wait from the boundary's own overhead,
  and a completion wait cannot overlap in this loop by construction. The observation stands; the
  attribution does not.
### 3.1 Route cost attribution and host-copy counts (8 MiB)

A second measurement pass separates the staged upload's pieces, so the route's cost is attributed rather
than guessed. Same protocol, same machine, measured in one session with the `transfer_host_copies` group
of the same bench; 8 MiB payloads:

| piece | 8 MiB |
| --- | ---: |
| `pageable_clone` (one ownership copy of the payload) | 213 us (36 GiB/s) |
| `pageable_to_pinned` (one pageable -> pinned copy) | 258 us (30 GiB/s) |
| `pinned_dma` (`upload_pending` + `wait`) | 343 us (22.8 GiB/s) |
| `staged_total` (`upload_tensor`) | 1.006 ms (7.8 GiB/s) |
| upload `staging` / upload `pinned` (`transfer_upload`) | 1.087 ms / 343 us |
| download `pageable` / `pinned` / `pinned_wait` | 509 us / 345 us / 324 us |
| fresh pinned allocation, 8 MiB (`alloc_8MiB`) | 7.01 ms |

- **The staged route already performs exactly the copies it needs.** The three pieces above sum to
  814 us against 1.006 ms measured, so the remainder is reservation and dispatch, not another copy of
  the payload. Source confirms the count: `staged_bytes` copies a borrowed slice once into max-aligned
  owned storage (with its own unit test), `Client::staging` converts that pageable `Bytes` to pinned
  memory once, and `do_create` hands the already-owned `Bytes` to the device server without re-copying
  it (tensor4all/cubecl, "stop re-copying uploads in `ComputeClient::create*`").
- **Host passes over an 8 MiB payload**: staged upload 3 (own copy, staging copy, DMA), pinned upload 1
  (the device reads the caller's pinned buffer), staged download 2 (DMA into pageable, then the result
  `Vec`), pinned download 1 (DMA straight into the caller's buffer).
- **Buffer reuse is not an optimization, it is the contract.** A fresh 8 MiB pinned allocation costs
  7.01 ms, twenty times the 343 us DMA it enables. That is why `wait` hands the buffer back and why the
  pending routes exist: allocating a fresh pinned buffer per iteration costs far more than the transfer.
- **The staged-upload row above is not reproduced by this pass.** The earlier table records 3.77 -> 4.15
  ms for the 8 MiB staged upload; the same case now measures 1.087 ms, and the 32 KiB case 7.0 us
  against 18.4 us there. Nothing here attributes that gap, the two passes are not the same bench
  revision, and the earlier "varies by up to 10% run to run" statement is therefore withdrawn rather
  than stretched to cover it. The current numbers are the ones to reuse; the pinned columns agree
  across both passes (345 us vs 349 us at 8 MiB, 324 us vs 323 us for the wait-only row).
### 3.2 Large payloads fall off the substrate's small-transfer policy (256 MiB)

A 256 MiB case (above the substrate's 100 MB pinned-staging threshold) shows what the blocking routes do
once the payload stops fitting that policy. Same protocol, `transfer_large` group, 32 Mi
`f64` elements = 256 MiB:

| case | 256 MiB | effective |
| --- | ---: | ---: |
| download `pageable` (`download_tensor`) | 176.9 ms | 1.41 GiB/s |
| download `pinned` (`download_pending` + `wait`) | 10.46 ms | 23.9 GiB/s |
| upload `staging` (`upload_tensor`) | 166.7 ms | 1.50 GiB/s |
| upload `pinned` (`upload_pending` + `wait`) | 10.08 ms | 24.8 GiB/s |

- **The gap is the transport, not the payload, the direction or PCIe.** Same payload, both directions,
  same host and device: pageable 1.41/1.50 GiB/s against pinned 23.9/24.8 GiB/s, about 17x. Both blocking
  routes hand the driver pageable memory - the borrowed upload owns a native (pageable) copy and
  `do_create`'s `staging(..., file_only = true)` policy leaves `Native` storage alone, while the numeric
  download writes straight into its result `Vec<T>`.
- **An earlier revision of this paragraph attributed the gap to the substrate's 100 MB threshold, and that
  account is withdrawn.** Neither route takes that branch: the numeric download above 16 bytes never calls
  `reserve_cpu` (it allocates its result vector and copies into it), and the borrowed upload is pageable at
  every size rather than only above 100 MB.
- **Which share of the 166-177 ms is pageable DMA, host page population, queue dispatch, or staging
  retirement is not attributed here.** The measurements above are observations; the causal split is
  something the implementation of `docs/design/large-transfer-staging-1945-u4.md` has to measure on this
  same group rather than inherit. What the numbers do fix is the size of the prize: a pinned DMA of this
  payload is ~10 ms and a host-side payload copy is ~8.3 ms.

- **The copy is not below the reference**: 8 MiB in 323 us is ~26 GB/s, above #2009's 5-13 GB/s cudarc
  reference. An earlier revision of this record claimed the opposite because the payload labels were
  wrong by a factor of eight (element counts read as bytes); that claim is withdrawn, and the staged
  upload path varies by up to 10% run to run, so no conclusion is drawn from its small deltas.
- **The pending routes are still not a drop-in replacement at tiny sizes**: ~22 us at 16 B against
  3.7 us for a staged 16 B upload, because the remainder is host dispatch and resource resolution.
- These are device-side measurements of a route, not an app-level speedup claim: the pending route's
  point is that the caller may wait later, which a serial enqueue-then-wait benchmark cannot show.

## 4. Ownership and failure semantics

```text
PendingDownload                 // owns the submitted value and the pinned buffer
  ├── runtime: CudaRuntime     // provider pin: context and stream stay alive
  ├── buffer: PinnedHostBuffer // owned destination
  ├── resource: ManagedResource<GpuResource>   // source allocation retention
  ├── event: CUevent           // recorded after the copy on the same stream
  └── resolved: bool
```

- The copy is published through CubeCL's `check_errors` (tensor4all/cubecl#27): it reports the
  stream's accumulated producer errors and dispatches every queued task to the server thread — so the
  producer's kernels are on the stream before the raw copy — without rotating the drop queue, so the
  call does not wait for device progress. A producer failure therefore surfaces as the transfer's own
  error instead of being masked by a successfully completed copy, and the transfer's own CUDA event
  is the completion witness.
- Staged host bytes are retired by CubeCL's staging policy, which the write path enforces as well
  (tensor4all/cubecl#28): with the boundary no longer flushing, a transfer-only workload that stages
  bytes and runs no kernel would otherwise retain them, because the policy counters are only evaluated
  where they are checked.
- `wait()`: `cuEventSynchronize`; on success the retained resource is released and the buffer is
  returned; on failure nothing is published and the buffer/resource are abandoned (leaked), because a
  failed copy cannot prove what the device is still touching.
- `Drop` without `wait`: detaches the record into the runtime's retirement queue (see
  [owning-pending-transfers-1945-u4.md](./owning-pending-transfers-1945-u4.md)); the old note said it
  resolved non-blockingly and then waited, which is the behaviour this package replaced. (Original
  wording follows, kept only as a pointer: it resolves the event if it can (non-blocking query, then a
  bounded
  synchronize), then releases; otherwise abandons.
- `is_ready()`: `cuEventQuery`; a non-ready result is `Ok(false)`, and a device error is reported
  rather than collapsed into "not ready".

## 5. Remaining U4 work (needs maintainer decisions)

1. **Source publication and source-root handoff**: capture an immutable storage/read lease plus the
   producer frontier, end the source's numerical admission, and let the destination be entered
   afterwards. The U1 record leaves this to U4; `ManagedResource` retains an allocation but does not
   grant or transfer read authority. A design for it was written and rejected in review — see
   [source-publication-1945-u4.md](./source-publication-1945-u4.md), which records the four blockers
   (owning submission first, the safe writable-binding escape, the missing producer identity, and
   cross-stream error masking) and the prerequisites for a second attempt.
2. **Dependent-consumer access**: a device consumer must receive both ordering *and* authorized
   access to the pending destination. That needs a destination-access handoff in the value model, and
   tokens minted for the exact frozen event domain (not derived from a runtime handle). Cross-domain
   admission currently host-waits source-domain dependencies.
3. **Owning asynchronous submission** as the repository's storage-ownership contract requires: both
   transfer directions now own their admitted package, detach on drop and leak on `mem::forget`
   ([owning-pending-transfers-1945-u4.md](./owning-pending-transfers-1945-u4.md)). What stays open is
   the publication boundary: it no longer performs its own drop-queue rotation (CubeCL's
   `check_errors`), but host dispatch, cross-stream GC backpressure and a preceding kernel that trips
   the staging policy can still wait for device progress.
4. **Bounded chunked staging, event amortization, and the route capability model** (including why
   P2P is unavailable on a given device pair).
5. **#2009's measurements** (small/large, pageable/pinned/caller-output, copy bytes, event/wait cost,
   measured overlap) once the routes above exist.

## 6. Open questions for maintainers

1. Does slice 1 belong in `tenferro-gpu`'s CUDA module only, or should the pending contract be stated
   in `tenferro-tensor` so other backends can adopt it deliberately?
2. For §4.1, is the intended source lease a new owned type in `tenferro-tensor`, or an extension of
   the existing borrowed-read contract?
