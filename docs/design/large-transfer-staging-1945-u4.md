# Bounded chunked staging for large blocking uploads (#1945 U4 item 4)

**Revision 4, implemented.** Revisions 1-3 were rejected by two independent pre-reviews
(`openai/gpt-6.1-sol`); revision 4 carries their required changes and adds what the implementation
measured. The measurement baseline is `docs/design/asynchronous-transfer-1945-u4.md` §3.1-3.2.

## 1. Outcome, measured

Blocking uploads above the staging chunk go through one runtime-owned pinned staging lease, chunk by
chunk. A100 80GB PCIe, driver 580.126.09, CUDA 12.0, `transfer_paths` bench, criterion's displayed
central estimates:

| case | before | after |
| --- | ---: | ---: |
| upload `staging`, 256 MiB (`upload_tensor`) | 166.7 ms (1.50 GiB/s) | **24.2 ms (10.6 GiB/s)** |
| upload `staging`, 8 MiB (at the cutoff, previous route) | 1.044 ms | unchanged |
| upload `staging`, 8 MiB + 8 B (chunked) | — | **0.651 ms** |
| download `pageable`, 256 MiB | 176.9 ms | unchanged (169-204 ms across runs) |
| download/upload `pinned` (owning routes) | 10.5 / 10.1 ms | unchanged |

Two findings shape the design, and one of them removed half of it:

- **The upload direction is where chunking pays.** A pinned upload of 256 MiB already ran at 24.7 GiB/s,
  so the blocking route's pageable transport was the whole gap, and staging through pinned memory closes
  6.9x of it. The chunked route is already faster at the smallest payload it can serve (8 MiB + 8 B:
  0.651 ms against 1.044 ms), so the cutoff stays at the chunk size rather than moving up.
- **A chunked download does not pay, and is not shipped.** It was implemented, measured, and removed:
  DMA into pinned staging took the expected 10.4 ms for 256 MiB, but copying that staging into the
  result ran at about **2 GB/s** - reading this machine's pinned allocations from the host is an order
  of magnitude slower than writing them (measured 30 GiB/s the other way). The chunked download
  therefore measured 153 ms against the pageable route's 177 ms, and the pageable route keeps its place
  with a comment saying why.

## 2. Scope

- The blocking `upload_tensor` route for the numeric dtypes it already serves; above
  `STAGING_CHUNK_BYTES` (8 MiB) only. Public signatures and return types do not change.
- `Bool`, the empty and `<= 16` byte paths, the exact-handle gate and the non-exact-handle fallback,
  the placement checks and the owning pending routes are untouched.
- No new public API; no cubecl type enters one.

## 3. Readiness: the change, stated

`upload_tensor` used to enqueue and return. A chunked upload now waits for its own chunks, because a
borrowed host source and a reused staging buffer cannot stay safe behind an early return without an
owned asynchronous scheduler. Uploads at or below the cutoff keep the enqueue behaviour, so the
asymmetry is deliberate and documented.

## 4. Design as implemented

### 4.1 The staging lease

`PinnedStagingSlot` on `CudaRuntimeState` holds `Option<PinnedHostBuffer>`, fixed at
`STAGING_CHUNK_BYTES` and allocated before the first enqueue (the chunk loop allocates nothing).
`CudaRuntime::acquire_pinned_staging` hands it out as a `PinnedStagingLease` that *takes* the buffer out
of the slot, so two concurrent chunked uploads never stage through the same memory.

The lease's transitions are the contract:

- `begin_chunk` marks the lease unproven **before** each possible enqueue; `end_chunk` clears it only
  after the wait that follows. A reused event proves only the work it captured, so a previous chunk's
  success is never read as this chunk's completion.
- `release` returns a proven buffer to the slot; `retire` hands an unproven one to the runtime's
  permanent soundness-retention queue together with the event and the device resource. The slot stays
  empty after a retirement, so the next transfer allocates a replacement instead of reusing a buffer DMA
  may still be touching.
- `Drop` on an unproven lease is the unwinding path: it leaks the buffer rather than freeing it.
- Teardown frees a buffer in the slot on the same successful path that already cleans up the pinned
  scalar slot, and leaks it when stream retirement failed.

Lock discipline: the publication boundary, resource resolution and lease acquisition all happen before
the slot mutex is taken, and a terminal record is handed to retirement after it is released, so no
"retirement -> staging" edge can invert "staging -> retirement". The runtime's unsafe `Send`/`Sync`
justification now names the staging slot as well.

### 4.2 The loop

Per chunk, in checked arithmetic: copy the host slice into the leased buffer, `cuMemcpyHtoDAsync_v2`,
record the call's event, wait for it, then reuse the lease. One event per call, recorded and waited
repeatedly, never allocated per chunk. Failures after the first possible enqueue retire the lease, the
event and the retained `get_resource` guard, so nothing is freed while DMA may be in flight.

Before the loop, once: the placement/dtype checks the caller already made, `check_errors` (the
publication boundary: it flushes and reports queued producer errors without rotating the drop queue),
`get_resource` on the captured stream, the resolved resource's extent check, and context plus stream
capture.

### 4.3 What the tests caught

The first implementation copied the staging buffer's *full* length on the last chunk
(`memcpy_dtoh_async`'s destination slice length is its copy length), which failed the three-chunk case
with `CUDA_ERROR_INVALID_VALUE`. The tests that exist for that - above the chunk, exactly the chunk, one
element either side, and several chunks with a remainder, all with per-index distinct values - are the
reason it was found before the PR. The download removal later in the same work left them exercising the
chunked upload against the unchanged download.

## 5. Non-goals and non-claims

- Not a change to the owning pending routes; callers who need throughput with their own buffer still use
  them. `ponytail: one shared staging slot serializes concurrent chunked uploads; a small pool is the
  follow-up if measurement shows contention.`
- Not evidence of producer/consumer overlap: the per-chunk wait is on the critical path by construction.
- Not a claim at other devices, drivers or payload sizes, and not a claim about the pageable download
  route's run-to-run spread (169-204 ms observed).
- Not full U4 acceptance: source publication, device consumers, P2P and AD remain separate.
- An 8 MiB lease bounds per-call staging, not aggregate concurrent transfer memory.

## 6. Evidence

- 4 new A100 tests in `crates/tenferro-gpu/tests/chunked_staging.rs` (chunk boundary and remainders, both
  directions, distinct per-index values), 4 new A100 lease tests in the runtime's unit tests (take/restore
  reuse, unproven retention with an empty slot, unwinding leak, populated-slot teardown), the existing
  209 CUDA unit tests and the pending-download/pending-upload/held-session targets re-run unchanged.
- `scripts/ci/cuda_test_partition.tsv` extended with all eight, so the GPU lane runs them.
- Bench: `transfer_large` (256 MiB, both directions and families) and `transfer_chunk_boundary`
  (8 MiB and 8 MiB + 8 B, the routing decision).
