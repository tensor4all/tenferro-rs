# Bounded chunked staging for large blocking transfers (#1945 U4 item 4)

Design for the remaining U4-staging gap, **revision 3**, after two independent pre-reviews
(`openai/gpt-6.1-sol`) returned "not ready to implement" on revisions 1 and 2. Both reviews' required
changes are applied below; the measured problem and the narrowed scope are unchanged.

Measured size of the prize: `docs/design/asynchronous-transfer-1945-u4.md` §3.2.

## 1. Problem

The blocking `upload_tensor` / `download_tensor` routes move a large payload at pageable-transport speed
while the owning pending routes move the same payload, on the same host and device, at pinned speed
(256 MiB: 1.41 / 1.50 GiB/s against 23.9 / 24.8 GiB/s, about 17x). The gap is the transport: both blocking
routes hand the driver pageable memory - the borrowed upload owns a native copy and `do_create`'s
`staging(..., file_only = true)` leaves `Native` storage alone, while the numeric download writes into its
result vector. Which share of the residual is pageable DMA, host page copy, dispatch or retirement is
**not** attributed by the measurement, and the implementation measures its own effect instead of
inheriting that story. The substrate's 100 MB threshold does not route either path; the earlier account
that claimed it did is withdrawn.

## 2. Scope

- **In scope**: the blocking `upload_tensor` / `download_tensor` routes for the numeric dtypes they
  already serve. Public signatures and return types do not change.
- **Narrowed**: `Bool` keeps its separate dispatch (`byte != 0` normalization, byte-encoded upload;
  `CubeElement`'s `Pod` bound excludes Rust `bool`, so no chunked numeric route can reach it). A
  regression test covers it; chunk-local Bool encoding is not part of this package.
- **Out of scope**: the owning pending routes, P2P and device-to-device routes, differentiated transfers,
  source publication, any new public API. No cubecl type enters the public API.

## 3. Readiness: the change, stated

- `download_tensor` returns a completed result today and keeps that contract.
- `upload_tensor` today enqueues and returns; a large chunked upload becomes **completion-blocking**,
  because a borrowed source and a reused staging buffer cannot stay safe behind an early return without
  an owned asynchronous scheduler. Uploads below the routing cutoff keep the enqueue behaviour, so the
  asymmetry is deliberate, documented in the guide, and measured separately: the bench gains a
  completed-upload arm instead of comparing an enqueue-only arm with a waiting one.

## 4. Design

### 4.1 One private staging lease

A single runtime-owned pinned staging buffer of `STAGING_CHUNK_BYTES` (8 MiB), never grown: it is
allocated before the first enqueue (with the result capacity and the call's event), so allocation failure
returns an ordinary error and **no allocation or growth happens inside the chunk loop**.

It is handed out as a **lease**, not shared in place:

- The lease takes the buffer *out* of the slot (`Option::take`), so no other call can touch it while a
  chunk may be in flight; the slot is a private field guarded by its own mutex, and raw access stays
  crate-private (the public `PinnedHostBuffer` API is unchanged).
- The lease restores the buffer only when the last chunk's completion is proven.
- Completion is marked **unproven before each possible enqueue**. Any enqueue, record or wait failure -
  including one after an earlier chunk succeeded - retires the buffer, the event and the resolved
  resource through the existing soundness-retention entry point, never the query-drainable detach path,
  and leaves the slot empty so the next call allocates a replacement.
- **A reused event carries no stale authority.** A successful record proves only its captured work, so
  after a failed record the previous success must never be read as this chunk's completion. Marking
  unproven first and retiring on failure is what keeps the single event sound on the success path; on the
  failure path it is retired, not reused.
- Unwinding between enqueue and record goes through the same lease, so `Drop` cannot free a buffer DMA
  may still touch.

Teardown: the cached buffer is released explicitly, while the context is still retained, on the same
successful-teardown path that already cleans up the pinned scalar slot; the weak-reference `Drop` on
`PinnedHostBuffer` stays the fallback for buffers that outlive the state. The runtime's existing unsafe
`Send`/`Sync` justification is extended to cover this slot.

Lock discipline: publication, resource resolution and lease acquisition happen before the staging mutex
is held, and terminal records are handed to retirement after it is released, so no new
"retirement -> staging" edge can invert "staging -> retirement".

### 4.2 Boundary, once, before the loop

Reuse the discipline the pending routes already implement, in this order: residency/placement and dtype
checks; the producer-error publication boundary (`check_errors`: flushes and reports queued producer
errors without rotating the drop queue); `get_resource(handle)` on the captured stream for cross-stream
ordering; retain that resolved resource for the whole call; activate the context and capture the raw
stream. `get_resource` alone uses `ignore: true, flush: false`, so it is not the publication boundary, and
a cached address is not a substitute for it.

Two conditions the current routes already enforce are preserved as gates, not replaced by offset
arithmetic: the direct-download route runs only when the handle spans exactly the payload
(`handle.size_in_used() == byte_len`), otherwise the whole-allocation read and its element-count
validation remain; and the resolved resource's available size is checked before any raw copy.

### 4.3 Chunk loop

`chunk = min(payload, STAGING_CHUNK_BYTES)`, offsets in checked arithmetic:

- upload: copy the host chunk into the leased buffer, enqueue `cuMemcpyHtoDAsync(dst + offset, staging,
  n)`, record the call's event, wait for it, then reuse the lease for the next chunk;
- download: enqueue `cuMemcpyDtoHAsync(staging, src + offset, n)`, record and wait the same event, then
  copy the chunk into the result.

One event per call, recorded and waited repeatedly, never allocated per chunk. The per-chunk wait is what
makes the single lease safe: the CPU must not overwrite an H2D source or read a D2H destination before
that chunk's copy completes.

Result publication: the result is allocated with capacity and filled in completed ranges, with its length
published only after the loop succeeds - no payload zero-fill and no uninitialized element ever exposed
as a normal slice. The device destination is not published as ready before its last chunk completes.

### 4.4 Routing

Eligibility for the chunked route is **above the preserved fast path** (empty and `<= 16` byte cases, the
exact-handle gate, and the existing small-payload paths stay; the small routes keep their current
behaviour). The final cutoff is **decided by measurement**, not assumed: no cutoff is chosen in this
design. The chunk capacity and the routing cutoff are separate constants.

## 5. Non-goals and explicit non-claims

- Not a change to the pending routes or their admission/retirement semantics; not a new transfer
  framework, pool, or double buffer. `ponytail: the single slot serializes concurrent large transfers; a
  small pool is the follow-up if measurement shows contention, and callers who need throughput now use the
  pending routes with their own buffer.`
- Not evidence of producer/consumer overlap; this is a blocking route whose per-chunk wait is on the
  critical path by construction.
- Not a claim that the whole 17x is recoverable, and not a claim at other sizes or devices until measured.
- Not full U4 acceptance: this closes the blocking-staging coverage item, not source publication, device
  consumers, P2P, or AD.
- The 8 MiB lease bounds per-call staging, not aggregate concurrent transfer memory (retained failures
  are outside the one-slot bound by design).

## 6. Evidence plan

1. **Acceptance, defined before measuring.** Same harness and protocol as
   `docs/design/asynchronous-transfer-1945-u4.md` §3; warm steady state reported separately from
   first-use (the first call allocates the lease). Cases: 64 B, 32 KiB, 256 KiB, 1 MiB, 8 MiB, 256 MiB,
   in both directions, each routing. Gate: no case may regress beyond the criterion harness's reported
   confidence interval at the same settings, and the 256 MiB blocking routes must land within a factor of
   2 of the pinned DMA time for that payload; the routing cutoff is the smallest size where the chunked
   route is faster by more than that interval.
2. **A100 tests**: round-trip equality above the chunk size, exactly the chunk size, one element either
   side, and a byte length that is not a multiple of the chunk; the empty, `<= 16` byte, exact-handle
   fallback, resource-extent and `Bool` paths unchanged.
3. **Post-enqueue failure**: fault-inject a record failure and a wait failure after an earlier chunk
   succeeded, then drain/synchronize retirement and attempt another transfer, asserting the old buffer was
   neither reclaimed nor reused, plus unwind coverage between enqueue and record.
4. **Bounds**: assert the largest live staging allocation equals the chunk, and record per-route copy
   counts in the async-transfer record beside the measured rows.
5. **Repository gates**: fmt; clippy with the repository flags and `--features cuda`; rustdoc
   `-D warnings`; doctests; non-CUDA all-targets check; the audited CUDA partition kept an exact inventory
   match; coverage at the repository's per-file thresholds (`coverage-thresholds.json`).

## 7. First implementation step

The private staging lease and its failure-state tests, before either numeric route is wired into it: the
lease's take/restore/retire transitions, the unproven-before-enqueue marking, the empty-slot-on-retirement
rule, and the teardown release are the parts a later bug would hide in, and they are testable without
touching a public route.
