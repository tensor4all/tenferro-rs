# CUDA sessions and pinned transfers

Two things make a CUDA stage cheap: entering the backend once instead of once per operation, and
moving host data without making the host wait for the device. This guide covers the entry shapes and
the transfer routes that provide them, the rules each one keeps, and what is deliberately not
supported yet. The runnable driver is
`crates/tenferro-gpu/examples/cuda_session_transfer.rs`; the reasoning and the measurements are in
[the held-session record](../design/held-cuda-session-1945-u3.md) and
[the transfer record](../design/asynchronous-transfer-1945-u4.md).

## Two entry shapes

| Entry | Use it for | Notes |
| --- | --- | --- |
| `CudaBackend::with_backend_session(\|session\| ...)` | one operation or one unrelated burst | unchanged; one callback, one entry |
| `CudaBackend::open_session()` | a stage that keeps its execution binding open | one entry for the whole stage; the session is closed explicitly |

<!-- snippet-source: crates/tenferro-gpu/examples/cuda_session_transfer.rs#cuda_held_session -->
```rust
    let mut session = backend.open_session()?;
    let mut state = session.with_session(|view| {
        view.add_read(
            TensorRead::from_tensor(&seed),
            TensorRead::from_tensor(&seed),
        )
    })??;
    for _ in 0..2 {
        state = session.with_session(|view| {
            view.add_read(
                TensorRead::from_tensor(&state),
                TensorRead::from_tensor(&state),
            )
        })??;
    }
    session.submit()?;
    let stats = session.close()?;
```
<!-- end-snippet-source -->

`with_session` hands out the same `&mut dyn BackendSession` view the scoped entry hands out, so every
concrete route and the extension path (`with_cuda_exec_session`) work unchanged; the difference is
only when the backend is entered. `submit` dispatches pending work at a boundary the caller chooses,
`synchronize` adds an explicit host barrier, and `close` performs a final submission and returns the
counters.

## Rules the session keeps

- **One owner, one thread.** The session is `!Send + !Sync` and captures the logical CubeCL stream of
  the thread that opened it, so its operations run on that binding even if other code changes the
  ambient stream. Closing restores nothing global because nothing global was taken.
- **One root per backend state.** While a session is live, a second `open_session` — on the same handle
  or through a clone, which shares the state — and a new scoped `with_backend_session` are rejected
  with a typed `SessionEntryError`. Extension *visitation* is not a root and keeps working; that
  distinction is what lets linalg and runtime extension dispatch run inside a held session.
- **The reservation is visible for its whole lifetime.** A thread holding a session carries a
  lifetime marker that the eager runtime's owner-lock path and the runtime executor's lease consult
  before blocking, so a session holder never waits for a lock its own session may need.
- **`Drop` never panics and never reports.** Dropping a session releases the reservation; submission
  errors are reported by `submit`/`close`, which is why `close` exists next to `Drop`.

## Where the host waits

`submit` and `close` call the submission boundary, which reports a producer's failures and dispatches
the host task queue without retiring staged bytes; that is why the boundary itself does not wait for
device progress. Host waits still happen at: the explicit `synchronize`, readbacks and host exports,
CubeCL's own staging-policy retirement when a preceding kernel trips it, cross-stream resource
resolution under GC backpressure, and documented destruction/error-recovery paths. `submit` is a
submission boundary, not a synchronization.

## Pinned transfers

`upload_tensor`/`download_tensor` remain the blocking convenience routes. When a stage wants to keep
the data in memory it controls, or to decide *when* to wait, use the pending routes:

<!-- snippet-source: crates/tenferro-gpu/examples/cuda_session_transfer.rs#cuda_pinned_transfer -->
```rust
    let host = vec![2.0, 4.0, 6.0, 8.0];
    let bytes = as_bytes(&host);
    let mut destination = upload_tensor(&runtime, &vector(&[0.0; 4])?).expect("device destination");
    let mut source = PinnedHostBuffer::new(&runtime, bytes.len())?;
    source.as_mut_slice().copy_from_slice(&bytes);
    let mut pending_upload = upload_pending(&runtime, source, &mut destination)?;
    let mut observed_in_flight = false;
    for _ in 0..64 {
        if !pending_upload.is_ready()? {
            observed_in_flight = true;
            break;
        }
    }
    let reusable = pending_upload.wait()?;
    // A 32-byte copy can finish before the first poll, so either answer is correct here; the
    // point is that `is_ready` never waits and the bytes are only handed over by `wait`.
    println!("pinned upload: observed in flight before wait = {observed_in_flight}");

    // Pending download into a caller-owned pinned buffer: the bytes are readable only from `wait`.
    let buffer = PinnedHostBuffer::new(&runtime, bytes.len())?;
    let filled = download_pending(&runtime, &destination, buffer)?.wait()?;
    assert_eq!(from_bytes(filled.as_slice()), host);
    assert_eq!(reusable.as_slice(), filled.as_slice());
```
<!-- end-snippet-source -->

- `PinnedHostBuffer` owns pinned host memory (`cudaHostAlloc`), is filled or read through
  `as_mut_slice`/`as_slice`, and is handed to a transfer **by value** so no caller borrow has to
  outlive a copy it cannot control.
- `upload_pending` borrows the destination tensor mutably for the handle's lifetime, so nothing can
  read, mutate or drop the device tensor while the copy may still write it. The compiler enforces it.
- `download_pending` borrows the source tensor for the handle's lifetime and owns the destination
  buffer; the bytes are only handed over by `wait`.
- `is_ready` queries the copy's CUDA event and distinguishes "not ready" from a device error;
  `wait` synchronizes once and returns the buffer; dropping without `wait` resolves before releasing,
  and a completion that cannot be proven leaks the buffers instead of freeing memory the device may
  still touch. Forgetting a pending transfer is safe for the same reason: the handle owns everything
  it uses.
- Small copies can complete before the first poll, so an `is_ready` that already returns `true` is not
  an error.
- The pending routes carry a fixed cost (host dispatch plus resource resolution) of roughly 20 µs on
  the reference machine, so they pay off above a few kilobytes or when the caller can wait later; the
  measured comparison with the blocking routes is in
  [the transfer record](../design/asynchronous-transfer-1945-u4.md#3-measured-the-pending-routes-against-the-existing-blocking-paths-2009).

## What is not supported

- **No dependent-device-consumer handoff.** A device consumer cannot yet receive authorized access to
  a pending destination; it waits for the transfer and then uses the tensor.
- **No cross-domain or peer-to-peer route.** Transfers are same-device; a capability the provider does
  not have fails typed rather than falling back.
- **No differentiated transfer.** Plain copies create no tape; a tracked transfer is a separate
  package.
- **No CUDA held session for the eager/AD runtime.** The session is concrete-only.

Unsupported combinations return a typed error; nothing falls back to the global/default route or to a
CPU numerical implementation.

## See also

- [Devices and GPU](devices-and-gpu.md) for placement and the CUDA compatibility matrix.
- [Custom CUDA kernels](custom-cuda-kernels.md) for raw/CubeCL interop inside a session.
- [CPU session entry and Rayon dispatch cost](session-entry-cost.md) for the CPU counterpart and the
  measured entry/transfer costs.
