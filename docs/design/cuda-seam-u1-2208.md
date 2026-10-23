# U1: the backend-neutral `gpu/` seam and the `cuda/` adapter (issue #2208)

Status: design revision 2 for review, 2026-10-23 (revision 1's review is answered in §8). Parent design:
[cuda-backend-t4a-2208.md](cuda-backend-t4a-2208.md) §3-§4 (unit U1). Upstream
prerequisites U0a and U0b are merged: `t4a-gpu-backend#6` (by-value parameters,
raw-scope launch, retention) and `t4a-cuda-kernels#15`/`#16` (ABI revisions 9 and
10).

U1 delivers the foundation only: device discovery, one engine over one
`t4a-cuda-runtime`, storage with allocation-domain identity, transfers, pending
handles, source publication, workspace counters, the error vocabulary, and the two
seam traits with CUDA implementations stubbed. **No kernel dispatch and no vendor
call lands in U1**; its acceptance is that the existing CUDA device, storage,
transfer, publication and consumer-copy tests run against the new path.

## 1. Module layout

```
crates/tenferro-gpu/src/
  gpu/                     backend-neutral; no CUDA type may appear here
    mod.rs                 the seam's public vocabulary and the two traits
    error.rs               GpuError (the shared cause vocabulary)
    counter.rs             counter snapshots shared by both backends
    storage.rs             Buffer: allocation-domain identity + an opaque handle enum
    transfer.rs            the ordered transfer protocol and the pending state machine
    publication.rs         SourcePublication: a produced value across engines
  cuda/                    CUDA adapter over t4a-cuda-runtime
    mod.rs                 the public `cuda::` surface (device, engine, transfers)
    device.rs              discovery over t4a_cuda_runtime::device_count/name
    engine.rs              CudaEngine over one Runtime + one ModuleCache
    storage.rs             Buffer internals: t4a_cuda_runtime::Buffer/UninitBuffer
    transfer.rs            upload/download/pending over the runtime's copies
    publication.rs         the CUDA publication guard
    vendor.rs              VendorOps stub: every route returns Unsupported
```

`src/lib.rs` keeps exporting the same `cuda::` names, so no downstream import
changes in U1. `src/cubecl/` stays in place and keeps serving the operation
surface until U2 replaces it; the new tree is private until then (the parent
design's "no dual-storage switch").

## 2. The seam

Two traits, as settled in the parent design (§3.1-3.3):

```rust
/// One runtime's source of launchable programs. Implemented by the kernel
/// adapter (U2/U3); in U1 the CUDA implementation returns `Unsupported`.
pub trait KernelSource {
    fn plan(&mut self, op: &OpDescriptor) -> Result<Program, GpuError>;
}

/// The vendor libraries a backend serves: BLAS, contraction, linalg, FFT.
/// Implemented by the CUDA adapters in U4; in U1 every method returns
/// `Unsupported`.
pub trait VendorOps {
    fn gemm(&mut self, call: &GemmCall<'_>) -> Result<(), GpuError>;
    fn contraction(&mut self, call: &ContractionCall<'_>) -> Result<(), GpuError>;
    fn linalg(&mut self, call: &LinalgCall<'_>) -> Result<(), GpuError>;
    fn fft(&mut self, call: &FftCall<'_>) -> Result<(), GpuError>;
}
```

The traits are used as `Box<dyn KernelSource + Send>` / `Box<dyn VendorOps + Send>`:
the engine moves to another thread, so its parts must. `CudaEngine` keeps `!Sync`
explicitly (`PhantomData<Cell<()>>`) rather than relying on the runtime's auto
traits, and a compile-time assertion in the engine module pins `Send` and `!Sync`.

`Program` is the backend-neutral plan type of U2/U3: an opaque handle produced by
`KernelSource::plan` and consumed by the CUDA launcher, holding the runtime's bound
function plus the prepared artifact key. In U1 it is a placeholder type with no
constructor outside `Unsupported`. The vendor call descriptors (`GemmCall`,
`ContractionCall`, `LinalgCall`, `FftCall`) borrow the engine's operands and scalars;
their concrete shapes land with their adapters in U4, and cuFFT descriptors stay in
`tenferro-fft` (the adapter there calls into the engine through `VendorOps`).

No storage trait: `gpu::Buffer` is a concrete type holding the backend handle in a
private enum (`StorageHandle::Cuda(...)`, later `Rocm(...)`). A third trait or a
generic storage parameter is a design change, not a detail.

`GpuError` carries the shared causes — `Unavailable`, `Unsupported`,
`ForeignDomain`, `Backpressure`, `Closed`, `Transfer { staged: bool }`,
`Broken { detail }` — and each backend maps its own errors into it. The existing
CUDA error types keep their names and variants and gain a `From` conversion, so the
provider tests keep asserting on them.

## 3. Ownership and lifetime rules (from the runtime's contract)

- **Two submission paths, one runtime, explicitly reconciled with the parent design.**
  The parent design says "the engine is the sole submission owner … no cloneable
  serialized owner", which is about *numerical* submission. The public transfer API
  that the acceptance tests use is `upload_tensor(&CudaRuntime, …)` /
  `upload_pending(&CudaRuntime, …)` with a cloned `CudaRuntime`, so explicit host
  transfers are a second, serialized path: `CudaRuntime` is a cheap handle owning
  `Arc<RuntimeHandle>`, and every transfer takes the runtime's submission lock
  (`Mutex<t4a_cuda_runtime::Runtime>`) for the duration of one admission. The engine
  keeps the numerical path (`&mut` on the caller thread, no lock) and borrows the
  same handle. The parent design sentence is amended accordingly in this slice's
  PR: the invariant that survives is "no second CUDA stream, no background thread,
  and one `&mut` at a time on the runtime", which the lock enforces for transfers.
- One `t4a_cuda_runtime::Runtime` per engine; the engine's numerical submission
  borrows it mutably and the engine is `Send`, not `Sync`. A buffer owns its allocation-domain identity
  and outlives the engine; a pending handle retains the runtime, the source buffer
  and the producer frontier.
- A buffer's `Drop` enqueues its free on the owning stream; the runtime's
  stream-ordered allocator already orders it behind every admitted use, so no
  per-buffer completion event is recorded.
- Everything that used CubeCL's relocation model (buffer pins, `ordered` flags,
  memoized device addresses) disappears: `t4a-cuda-runtime` addresses are stable
  for the allocation's life. The corresponding assertions in the existing tests
  move to "the address is stable" rather than "the pin keeps it stable".
- Pooled output reuse stays a runtime-owned, byte-length-keyed free list; the
  handle is a `t4a_cuda_runtime::Buffer`, so returning it to the pool is a move. Its
  existing byte bound, `clear`/trim controls and counters are preserved unchanged —
  U1 does not redesign the pool.
- **`CudaEngine::drop` synchronises before it releases the runtime handle.** The
  publication test requires the frontier to be complete immediately after
  `drop(backend)`, and a buffer can outlive the engine, so the drop is a
  best-effort `Runtime::synchronize` (draining queued work) followed by releasing
  the handle, never a `close` of a runtime another handle still uses.
- **CubeCL's pin/`ordered` machinery is not ported, and its guards stay until U6.**
  `src/cubecl/` and its assertions (`memoized_addr_tests.rs`,
  `cubecl_launch_contract.rs`) are untouched in U1 while CubeCL still serves the
  operation surface. The new path states the two properties separately: a
  `t4a-cuda-runtime` address is stable for the allocation's life (asserted by a new
  U1 test), and producer-before-vendor ordering holds because every vendor call runs
  inside `Runtime::with_raw` on the same stream (U4's contract, recorded here and
  tested there).

## 4. The acceptance surface, item by item

The six CUDA test files are the acceptance. Each moves to the new path when its
operation lands, in this order:

| Test file (tests) | U1 delivers |
| --- | --- |
| `storage_provider_cuda.rs` (4) | `CudaEngine::open`, `cuda_devices`, `upload_tensor`, view/allocation identity, duplicate placement, no safe raw access |
| `pending_upload.rs` (6) | `PinnedHostBuffer`, `upload_pending`, `PendingUpload::{is_ready, wait}`, drop detaches, forget leaks, rejection returns the package, two in flight |
| `pending_download.rs` (6) | `download_pending`, `PendingDownload::{is_ready, wait}`, completion after the producing engine drops, drop/forget semantics, rejection returns the package, unserved dtype rejected |
| `source_publication.rs` (6) | `SourcePublication` (admission, `frontier`, completion), survives and keeps its allocation after the engine drops, drop detaches, forget leaks, rejection returns the value |
| `chunked_staging.rs` (4) | the runtime's bounded staging ring: payload sizes around one chunk |
| `consumer_copy.rs` (6) | cross-engine copy ordering, publication into another engine's tensor, transfer outlives its producer, forgotten transfer leaks |

Three pieces of accounting and lifetime have no upstream equivalent and therefore
belong to the *compatibility package* (the thin CUDA-provider types) rather than to
upstream retirement:

- **Detached-handle count.** The tests require a count that grows when a pending or
  publication handle is dropped, stays unchanged on `forget`, and drops to zero after
  a synchronisation. Upstream `pending_records` counts retirement epochs, leases and
  witnesses, which is a different quantity; the package keeps its own counter and
  reports it in the stats snapshot next to (not instead of) the runtime's.
- **The publication frontier.** `Runtime::checkpoint()` (merged in
  `t4a-gpu-backend#7`) returns the `Completion` handle for "everything queued
  before now", so a publication retains one checkpoint recorded on the producing
  engine's stream and a reader polls it; a failed or broken stream surfaces through
  the checkpoint's own `poll`/`wait` error, which is the failure-observation path.
  Publication admission never synchronises.
- **Staging.** Upstream owns the bounded pinned ring; U1 asserts round-trip bytes
  *and* the runtime's staging/pending counters, and adds a small-budget case so a
  single whole-payload pinned allocation cannot pass the four staging tests.

The invariants those tests encode, restated so U1 can be reviewed against them:

1. A published value is readable after its producing engine is dropped, and its
   allocation is not released while the reader holds it.
2. A pending handle reports `is_ready == false` immediately after submission for a
   payload large enough that the copy cannot have completed, then yields the bytes.
3. Dropping a pending handle detaches it into the retirement queue and releases
   nothing that is still in use; `forget` leaks deliberately.
4. A rejected submission returns the submitted package unchanged (source, value).
5. A copy that waits on another engine's write observes the written bytes, and the
   producer may be dropped immediately afterwards.
6. Pinned host bytes round-trip unchanged, and a payload of one chunk, one element
   below, one above, and several chunks all round-trip.
7. A dtype the transfer route does not serve is rejected with a typed error and the
   package comes back.

## 5. What U1 deliberately does not do

- No `KernelSource`/`VendorOps` implementation beyond `Unsupported`: the trait
  objects are wired into the engine so U2/U4 fill them in, but a call fails with a
  typed error rather than a stub that silently returns zeros.
- No operation surface: `TensorElementwise`/`TensorStructural`/… keep coming from
  the CubeCL backend until U2. U1's tests therefore do not include the operation
  suites.
- No feature removal: `cubecl*`/`webgpu` dependencies, the macOS lane and the
  `t4a-cubecl`/`t4a-cubek` pins stay until U6.
- No `CudaEngine::open` signature change; the concurrency contract is documented
  where it already is.

## 6. Gates

- `cargo test -j 16 --workspace` (CPU) unchanged and green.
- `cargo test -j 16 -p tenferro-gpu --features cuda -- --ignored` on the local
  A100 for the six files above, recorded with the GPU model, driver and toolkit in
  a `docs/worklogs/` entry.
- `cargo clippy -j 16 --all-targets --features cuda -- -D warnings`, `cargo fmt`.
- Independent review of this design before implementation and of the finished diff
  before the PR.

## 7. Risks and open questions

- **Error vocabulary.** The five CubeCL-era error types (`CudaDeviceError`,
  `CudaEngineError`, `TransferSubmitError`, `PublicationAdmissionError`,
  `PublicationError`) carry ownership packages (`Rejected(...)`, `Unproven(...)`) and
  source errors that a bare `GpuError` cause cannot reconstruct. U1 therefore keeps
  them as the *primary* vocabulary, unchanged, and defines `GpuError` only as the two
  seam traits' error type with a one-way `From` into the provider errors where a
  transfer is involved. There is no merger in U6: the provider types either survive
  or are replaced when their CubeCL-only variants disappear, and that decision is
  U6's, not U1's.
- `PinnedHostBuffer::new(runtime, bytes)` is used by the tests directly; on the
  runtime it becomes `Runtime::alloc_pinned`. Its `as_mut_slice` stays a host view.
- Whether `CudaRuntime` remains a cloneable handle (it is `Clone` today and the
  tests clone it) while the engine is the submission owner: recommendation is yes,
  with the runtime handle holding `Arc<RuntimeHandle>` that owns the mutex-guarded
  `t4a_cuda_runtime::Runtime` — the same shape the runtime's own contract assumes
  (one owner, `&mut` at submission).

## 8. Findings from the revision 1 review and where they are answered

| Finding | Answer |
| --- | --- |
| Cloneable `Arc<Mutex<Runtime>>` contradicts the parent's "no cloneable serialized owner" | §3 separates the two paths explicitly: the engine owns numerical submission (`&mut`, no lock), the `CudaRuntime` handle owns serialized host transfers (one submission lock, one runtime, one stream). The parent design sentence is amended in this PR |
| Boxed traits break engine `Send`, `!Sync` not enforced, pinned/pending auto traits | §2: `+ Send` object bounds, an explicit `!Sync` marker with a compile-time assertion, and the pinned/pending handles keep upstream's `Send` |
| Retaining the runtime after engine destruction delays the synchronisation the publication test needs | §3: `CudaEngine::drop` synchronises (best effort) before releasing its handle; a surviving buffer keeps the domain, not the engine |
| Upstream `pending_records` is not the detached-handle count the tests assert | §4: the compatibility package keeps its own detached-handle counter, reported beside the runtime's counters |
| No upstream checkpoint for arbitrary producer work | §4: `Runtime::checkpoint()` (merged upstream in `t4a-gpu-backend#7`) is the publication frontier; its `poll`/`wait` is the failure path; admission never synchronises |
| The staging tests cannot see bounded staging | §4: round-trip **and** runtime counters, plus a small-budget case |
| Replacing the pin with "addresses are stable" loses #1868's producer-before-vendor ordering | §3: CubeCL's guards stay until U6; the new path asserts address stability and states the ordering contract that U4 implements inside `with_raw` |
| `GpuError` cannot express allocation/validation/driver causes or the ownership packages | §7: provider errors stay primary in U1; `GpuError` is the traits' error only, with a one-way conversion; the merge decision is U6's |
| `Program`/call descriptors undefined; vendor objects have no borrow path to the runtime | §2: `Program` and the call descriptors are named with their ownership and borrow direction; cuFFT stays in `tenferro-fft` |
| Delete list | §3/§4 adopt it: no second module cache (the runtime's own), no pin cache or `ordered` flags, no duplicate transfer scheduler, no boxed plumbing before its caller exists, no pool redesign |
