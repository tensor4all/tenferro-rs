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

### Prepared graphs inside a held session

`CudaHeldSession::with_session` can run a `PreparedCompiledGraph` directly through
`Runtime::run_prepared_in`, so a steady-state loop keeps one held session instead of paying a scoped
entry per `Runtime::run_prepared` call:

<!-- snippet-source: crates/tenferro-gpu/examples/cuda_session_transfer.rs#cuda_prepared_in_held -->
```rust
    let mut prepared_builder = tenferro_runtime::Runtime::builder();
    prepared_builder.register_engine(tenferro_gpu::cuda::cuda_runtime_engine_registration(
        &backend,
        tenferro_runtime::EngineId::new("tenferro.example.held-prepared.v1")?,
    )?)?;
    let prepared_runtime = prepared_builder.build()?;
    let x = tenferro_runtime::TracedTensor::input_symbolic_shape(tenferro_runtime::DType::F64, 1)?;
    let y = (&x * &x)?;
    let program = tenferro_runtime::GraphCompiler::new()
        .compile_with_input_specs(&y, &[(&x, tenferro_runtime::DType::F64, &[4])])?;
    let prepared = prepared_runtime.prepare_compiled(&program, &[&seed])?;
    let prepared_inputs: [&Tensor; 1] = [&seed];
    let mut prepared_session = backend.open_session()?;
    let outputs = prepared_session.with_session(|view| {
        prepared_runtime.run_prepared_in(view, &prepared, &prepared_inputs)
    })??;
    prepared_session.close()?;
    assert_eq!(outputs[0].shape(), &[4]);
```
<!-- end-snippet-source -->

`run_prepared_in` runs every graph node on the borrowed session, leases the runtime executor state
once without waiting, and retires its device work before returning. Before any launch it rejects a
multi-engine graph, a session that belongs to a different backend state, and a prepared operation
that does not support a borrowed session, each with a typed error that names `Runtime::run_prepared`
as the alternative. The scoped form `with_backend_session(|view| runtime.run_prepared_in(view, ...))`
works on CPU and CUDA; the eager/AD adapter is still not part of this route.

### Deferred completion for a batch of prepared runs

When a loop runs the same prepared graph several times and only needs completion
before reading the batch's results, defer retirement instead of paying one stream
barrier per call:

```text
let mut session = backend.open_session()?;
let inputs: [&Tensor; 1] = [&input];
let batches = session.with_session(|view| {
    let mut deferred = runtime.begin_prepared_deferred(view, &prepared)?;
    for _ in 0..64 {
        deferred.run_prepared_deferred(&inputs)?;
    }
    deferred.finish_prepared()
})??;
// Every tensor in `batches` is now valid; `batches[i]` is run i's output package.
```

`run_prepared_deferred` enqueues a run and returns its index without the
retirement barrier. `finish_prepared` retires all runs since the previous finish
(one dispatch fence plus one stream barrier for a coalesced single-domain graph)
and returns their outputs in enqueue order. The outputs are **not** valid before
`finish_prepared`; the guard owns them internally, and the inputs it borrows must
outlive the guard, so neither can be released while device work is pending. The
guard also borrows the session and holds the executor-state lease, so unrelated
work cannot use that session between deferred runs. A device-side retirement
failure is returned by `finish_prepared`; dropping an unfinished guard retires
best-effort and cannot report an error. A host-side validation failure still
comes from the deferred call itself, after it retires earlier work in that
batch.

Use `run_prepared_in` when each individual call must return retired outputs. A
graph that spans more than one engine is outside both caller-borrowed routes;
use `run_prepared` outside a session instead. The deferred guard supports the
same single-engine graphs as `run_prepared_in`.

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
    let destination = upload_tensor(&runtime, &vector(&[0.0; 4])?).expect("device destination");
    let mut source = PinnedHostBuffer::new(&runtime, bytes.len())?;
    source.as_mut_slice().copy_from_slice(&bytes);
    let mut pending_upload = upload_pending(&runtime, source, destination)?;
    let mut observed_in_flight = false;
    for _ in 0..64 {
        if !pending_upload.is_ready()? {
            observed_in_flight = true;
            break;
        }
    }
    let (destination, reusable) = pending_upload.wait()?;
    // A 32-byte copy can finish before the first poll, so either answer is correct here; the
    // point is that `is_ready` never waits and the values are only handed over by `wait`.
    println!("pinned upload: observed in flight before wait = {observed_in_flight}");

    let buffer = PinnedHostBuffer::new(&runtime, bytes.len())?;
    let (returned, filled) = download_pending(&runtime, destination, buffer)?.wait()?;
    assert_eq!(from_bytes(filled.as_slice()), host);
    assert_eq!(reusable.as_slice(), filled.as_slice());
```
<!-- end-snippet-source -->

- `PinnedHostBuffer` owns pinned host memory (`cudaHostAlloc`), is filled or read through
  `as_mut_slice`/`as_slice`, and is handed to a transfer **by value** so no caller borrow has to
  outlive a copy it cannot control.
- **Reuse the buffer; do not allocate one per transfer.** A fresh 8 MiB pinned allocation costs about
  7 ms on an A100 against 343 us for the 8 MiB DMA it enables, so `PinnedHostBuffer::new` inside a
  transfer loop dominates everything else. Every route here hands the buffer back (`wait` returns it, a
  refused submission returns it, a dropped handle retains it), which is what makes reuse the intended
  shape rather than a workaround.
- `upload_pending` takes ownership of the source buffer *and* the destination value, so nothing can
  read, mutate or drop the device tensor while the copy may still write it, and `wait` returns the
  destination and the buffer together.
- `download_pending` takes ownership of the value *and* the buffer, so neither is reachable while the
  copy is in flight and both come back from `wait`. Dropping an unwaited handle detaches the record
  into the runtime's retirement queue, which releases it once its event proves completion and retains
  it otherwise; forgetting the handle leaks it in place. A refused submission (wrong length, unserved
  dtype, host tensor) returns the package untouched as `TransferSubmitError::Rejected`.
- `is_ready` queries the copy's CUDA event and distinguishes "not ready" from a device error;
  `wait` synchronizes once and returns the value and the buffer; dropping without `wait` detaches the
  record into the runtime's retirement queue, which releases it after its event proves completion and
  retains it otherwise. Forgetting a pending transfer is safe for the same reason: the handle owns
  everything it uses, so forgetting it leaks the record instead of handing the caller access.
- Small copies can complete before the first poll, so an `is_ready` that already returns `true` is not
  an error.
- The pending routes carry a fixed cost (host dispatch plus resource resolution) of roughly 20 µs on
  the reference machine, so they pay off above a few kilobytes or when the caller can wait later; the
  measured comparison with the blocking routes is in
  [the transfer record](../design/asynchronous-transfer-1945-u4.md#3-measured-the-pending-routes-against-the-existing-blocking-paths-2009).

### Blocking readback into a reused buffer

`download_tensor_into` is the blocking readback for a caller that reads the same size back
repeatedly. It takes a caller-owned `Vec<T>`, resizes it to the tensor's element count without
reallocating when the capacity already suffices, and returns it filled and ready to pass in again, so
the loop does not allocate and first-touch a fresh pageable buffer on every iteration. It uses the
same driver copy into host memory as the large `download_tensor` path; the saving is the allocation
and its page faults, not a different transport. Prefer it for repeated readback of same-size results,
keep `download_tensor` for a one-off download, and use the pending routes when the caller wants to
decide *when* to wait. A `bool` destination is rejected with a typed error because raw device bytes
are not guaranteed to be 0/1; `download_tensor` decodes them.

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
