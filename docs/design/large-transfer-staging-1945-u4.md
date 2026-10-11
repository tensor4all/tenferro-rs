# Bounded chunked staging for large blocking transfers (#1945 U4 item 4)

Short design for the remaining U4-staging gap. **Revision 2** after an independent pre-review
(`openai/gpt-6.1-sol`) returned "not ready to implement": revision 1 misidentified the bottleneck (the
substrate's 100 MB threshold), understated the readiness change on upload, and specified no failure
retention after a chunk was enqueued.

## 1. Problem, with the measured size of it

The blocking `upload_tensor` / `download_tensor` routes move 256 MiB at pageable-transport speed while the
owning pending routes, on the same host, device, and payload, move it at pinned speed. A dedicated
diagnostic (`transfer_large` group, same protocol as `docs/design/asynchronous-transfer-1945-u4.md` §3,
A100 80GB PCIe, driver 580.126.09, CUDA 12.0):

| case | 256 MiB | effective |
| --- | ---: | ---: |
| `download_pageable` (`download_tensor`) | 176.9 ms | 1.41 GiB/s |
| `download_pinned` (`download_pending` + `wait`) | 10.46 ms | 23.9 GiB/s |
| `upload_staging` (`upload_tensor`) | 166.7 ms | 1.50 GiB/s |
| `upload_pinned` (`upload_pending` + `wait`) | 10.08 ms | 24.8 GiB/s |

**What this establishes**: the gap is the transport, not the payload, the direction, or PCIe. Both
blocking routes hand the driver pageable memory - the borrowed upload owns a native (pageable) copy and
`do_create`'s `staging(..., file_only = true)` policy leaves `Native` storage alone, while the numeric
download writes straight into its result `Vec<T>` - and both pending routes hand it pinned memory. Two
further facts bound the prize: a pinned DMA of this size takes ~10 ms, and a host-side payload copy costs
~8.3 ms at this size (8 MiB copy = 258 us, §3.1).

**What it does not establish**, and revision 1 got wrong:

- The substrate's 100 MB threshold does not route these two paths: the numeric download above 16 bytes
  never calls `reserve_cpu` (it allocates its result vector and copies into it), and the borrowed upload
  is pageable at *every* size, not only above 100 MB.
- Which share of the 166-177 ms is pageable DMA, host page population, queue dispatch, or staging
  retirement is **not** attributed here. The implementation must measure its own before/after on the same
  group rather than inherit a causal story.
- `docs/design/asynchronous-transfer-1945-u4.md` §3.2 states the threshold account; that paragraph is
  corrected in the same change as this design.

## 2. Scope

- **In scope**: the blocking `upload_tensor` / `download_tensor` routes in `crates/tenferro-gpu` for the
  numeric dtypes they already serve. The Rust signatures and return types do not change.
- **Explicitly narrowed**: `Bool` keeps its current route. Its upload builds a converted byte vector and
  its download normalizes with `byte != 0`, so chunking it would need a chunk-local encoding/decoding
  contract that this package does not specify; the gap is recorded rather than half-done.
- **Out of scope**: the owning pending routes (already pinned and measured), P2P and device-to-device
  routes, differentiated transfers, source publication, and any new public API. No cubecl type enters the
  public API.

## 3. Readiness: state the change instead of claiming none

- `download_tensor` returns a completed result today (the small path synchronizes; the large path waits),
  and keeps that contract.
- `upload_tensor` today **enqueues** the write and returns: `upload_typed`'s own comment says the device
  write runs after the call returns. The chunked route cannot keep a borrowed host source and a reused
  staging buffer safe while returning early without an owned asynchronous scheduler, so **large uploads
  become completion-blocking**: the call waits for its own chunks. Uploads below the routing threshold
  keep today's enqueue behaviour, so the asymmetry is deliberate and documented in the guide.
- Consequently the bench comparison must separate *enqueue/staging* from *completed transfer* for uploads
  (the current `upload_staging` arm measures the former and `upload_pinned` the latter). The
  implementation adds a completed-upload arm and reports both, instead of comparing the two as if they
  measured the same thing.

## 4. Design

A runtime-owned, reused pinned staging buffer, plus one event per call.

1. **One staging buffer per runtime, bounded.** A lazily allocated `PinnedHostBuffer` of
   `STAGING_CHUNK_BYTES` (8 MiB), owned by the runtime behind the same kind of mutex as the existing
   pinned scalar slot. Per-call allocation is rejected on measurement: one 8 MiB allocation costs ~7 ms
   (§3.1), so a per-call buffer would make the route *slower* than today's for payloads below ~1 GiB.
   `ponytail: one shared staging slot serializes concurrent large transfers; a small pool is the
   follow-up if measurements show contention, and callers who need throughput now use the pending routes
   with their own buffer.`
2. **Boundary, once, before the loop.** Reuse the existing discipline the pending routes already
   implement, in this order: residency/placement and dtype checks; the producer-error publication
   boundary (`check_errors`, which flushes and reports queued producer errors without rotating the drop
   queue); `get_resource(handle)` on the captured stream to preserve cross-stream ordering; retain that
   resolved resource for the whole call; activate the context and capture the raw stream. `get_resource`
   alone uses `ignore: true, flush: false`, so it is not the publication boundary, and a cached address is
   not a substitute for it. A host upload's freshly reserved destination is resolved the same way before
   its address is used.
3. **Chunk loop.** `chunk = min(payload, STAGING_CHUNK_BYTES)`, offsets in checked arithmetic:
   - upload: copy the host chunk into the staging buffer, enqueue `cuMemcpyHtoDAsync(dst + offset,
     staging, n)`, record the call's event, wait for it, then reuse the buffer for the next chunk;
   - download: enqueue `cuMemcpyDtoHAsync(staging, src + offset, n)`, record and wait the same event, then
     copy the chunk out of the staging buffer into the result.
   One event is recorded and waited repeatedly, never allocated per chunk - the event-amortization the
   transfer contract asks for. The per-chunk wait is what makes the single buffer safe: the CPU must not
   overwrite an H2D source or read a D2H destination until that chunk's copy has completed.
4. **Failure policy: retention, not just an error.** Before the first possible enqueue the call owns the
   staging buffer and the resolved resource. If an enqueue, event record, or wait fails - or anything
   unwinds - a guard keeps both: a proven completion releases them, an unproven one hands them to the
   existing retirement queue exactly as the pending transfers do. A failed chunk must not drop the staging
   buffer (`cudaFreeHost` while DMA may still touch it) and must not fall back to a plain `?`. An event
   that was never recorded queries as complete, so it can never authorize reclamation.
5. **Result publication.** The numeric download allocates its result with capacity and fills completed
   ranges, publishing length only after the loop succeeds; no zero-fill of the payload and no
   uninitialized element is ever exposed as a normal slice. The device tensor is not published as ready
   before its last chunk completes.
6. **Routing.** Payloads at or below the existing small path stay there (host dispatch dominates). Above
   it, the chunked route is used; with a reused buffer there is no per-call allocation to amortize, so the
   cutoff is decided by measurement near the boundary rather than assumed: measure 32 KiB, 256 KiB, 1 MiB,
   8 MiB and 256 MiB with both routings and keep the faster. The chunk capacity and the routing cutoff are
   separate constants.

## 5. Non-goals and explicit non-claims

- Not a change to the pending routes or their admission/retirement semantics; not a new transfer
  framework, pool, or double buffer.
- Not evidence of producer/consumer overlap: this is a blocking route whose per-chunk wait is on the
  critical path by construction.
- Not a claim that the whole 17x is recoverable, and not a claim at other sizes or devices until measured.
- Not full U4 acceptance: this closes the blocking-staging coverage item, not source publication, device
  consumers, P2P, or AD.
- An 8 MiB per-call staging bound is not a bound on aggregate concurrent transfer memory.

## 6. Evidence plan

1. **Before/after on the same group**: `transfer_large` (256 MiB) and the small sizes, both routings,
   including a completed-upload arm, so the enqueue/staging and completed costs are reported separately.
   Declare the non-regression gate before measuring: no size may regress beyond the bench's run-to-run
   spread, and the 256 MiB blocking routes must land within a factor of ~2 of the pinned DMA time.
2. **A100 tests**: round-trip equality for a payload above the chunk size, exactly the chunk size, one
   element either side, and a byte length that is not a multiple of the chunk; the empty and <= 16-byte
   paths unchanged; device/placement mismatch still refused before any enqueue.
3. **Post-enqueue failure**: fault-inject a record failure and a wait failure after an earlier chunk
   succeeded, and assert that the staging buffer and device resource are retained (not freed, not
   returned) - the case revision 1 left untested.
4. **Staging bound**: assert the largest live staging allocation equals the chunk, and record the copy
   count per route in the async-transfer record next to the measured rows.
5. **Repository gates**: fmt; clippy with the repository flags and `--features cuda`; rustdoc
   `-D warnings`; doctests; non-CUDA all-targets check; the audited CUDA partition kept an exact inventory
   match; coverage >= 75% per touched file.

## 7. Open questions

1. Does the per-chunk wait cost enough to justify a ping-pong double buffer in the same change, or is it a
   follow-up? Decided by §6.1, not by preference.
2. Does the shared staging slot measurably serialize realistic concurrent workloads? If yes, the follow-up
   is a small pool, not a per-call allocation.
