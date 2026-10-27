# CUDA engines and pinned transfers

Two things make a CUDA stage cheap: holding one engine instead of opening a
device per operation, and moving host data without making the host wait for the
device. This guide covers the engine shape, the prepared-run route on that
engine, the pinned transfer routes, the rules each one keeps, and what is
deliberately not supported yet. The runnable driver is
`crates/tenferro-gpu/examples/cuda_session_transfer.rs`; the engine ownership
model is recorded in
[execution engine ownership](../design/execution-engine-ownership.md), and the
transfer reasoning and measurements in
[the transfer record](../design/asynchronous-transfer-1945-u4.md) and the
[explicit-transfer work log](../worklogs/2026-07-29-issue-1471-explicit-transfers.md).

## One engine, one device, one stream

`CudaEngine::open(ordinal)` opens the device and returns a value you hold. It
owns the CubeCL runtime, the device's stream, the cuTENSOR/cuBLAS/cuSOLVER
handles and the engine's caches. Every operation borrows it through `&mut`, so
there is no entry protocol: a chain of dependent operations is just a chain of
calls, and one `flush` or `synchronize` covers them.

<!-- snippet-source: crates/tenferro-gpu/examples/cuda_session_transfer.rs#cuda_held_session -->
```rust
    let mut state = backend.add_read(
        TensorRead::from_tensor(&seed),
        TensorRead::from_tensor(&seed),
    )?;
    for _ in 0..2 {
        state = backend.add_read(
            TensorRead::from_tensor(&state),
            TensorRead::from_tensor(&state),
        )?;
    }
    backend.runtime().synchronize()?;
```
<!-- end-snippet-source -->

`flush` dispatches pending work at a boundary the caller chooses; `synchronize`
adds an explicit host barrier. Neither is implicit: entering a raw scope is not
a host barrier either, so raw kernels and vendor calls can be interleaved with
eager operations without stalling the pipeline.

## Prepared graphs on the same engine

A compiled graph prepared for this engine runs on it, so steady-state replay
reuses its stream, handles and caches:

<!-- snippet-source: crates/tenferro-gpu/examples/cuda_session_transfer.rs#cuda_prepared_in_held -->
```rust
    let mut prepared_builder = tenferro_runtime::Runtime::builder();
    prepared_builder.register_engine(tenferro_gpu::cuda::cuda_runtime_engine_registration(
        &backend,
    )?)?;
    let prepared_runtime = prepared_builder.build()?;
    let x = tenferro_runtime::TracedTensor::input_symbolic_shape(tenferro_runtime::DType::F64, 1)?;
    let y = (&x * &x)?;
    let program = tenferro_runtime::GraphCompiler::new()
        .compile_with_input_specs(&y, &[(&x, tenferro_runtime::DType::F64, &[4])])?;
    let prepared = prepared_runtime.prepare(
        &program,
        &[TensorRead::from_tensor(&seed)],
        &[backend.target()],
    )?;
    let outputs = prepared_runtime.run(
        &mut [&mut backend],
        &prepared,
        &[TensorRead::from_tensor(&seed)],
    )?;
    assert_eq!(outputs.len(), 1);
    drop(outputs);
    backend.synchronize()?;
```
<!-- end-snippet-source -->

`Runtime::run` returns once the work is queued and no longer reads caller memory
that could change; later work on the same engine runs after it in stream order.
An output that is one of the inputs, or a view of one, comes back borrowed;
every other output is a new compact tensor. `backend.synchronize()` is the
explicit host barrier before a download or a host read.

## Where the host waits

`flush` reports a producer's failures and dispatches the host task queue without
retiring staged bytes; that is why the boundary itself does not wait for device
progress. Host waits still happen at: the explicit `synchronize`, readbacks and
host exports, CubeCL's own staging-policy retirement when a preceding kernel
trips it, cross-stream resource resolution under GC backpressure, and documented
destruction/error-recovery paths. `flush` is a submission boundary, not a
synchronization.

## Pinned transfers

`upload_tensor`/`download_tensor` remain the blocking convenience routes. When a
stage wants to keep the data in memory it controls, or to decide *when* to wait,
use the pending routes:

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

## Rules the engine keeps

- **One device per engine.** `CudaEngine::open` selects one device and captures its stream; the
  engine's operations run on that stream even if other code changes the ambient one.
- **Raw scopes restore what they touched.** `with_raw` and `with_cubecl` save the calling thread's
  device/context, activate this engine's primary context for the callback, and restore the previous
  one on return, on `Err` and on unwind. Entering a scope is not a host barrier.
- **`Drop` never panics and never reports.** Dropping the engine releases its resources; submission
  errors are reported by `flush`, `synchronize` or the operation that hit them.

## What is not supported

- **No dependent-device-consumer handoff.** A device consumer cannot yet receive authorized access to
  a pending destination; it waits for the transfer and then uses the tensor.
- **No cross-domain or peer-to-peer route.** Transfers are same-device; a capability the provider does
  not have fails typed rather than falling back.
- **No differentiated transfer.** Plain copies create no tape; a tracked transfer is a separate
  package.

Unsupported combinations return a typed error; nothing falls back to the global/default route or to a
CPU numerical implementation.

## See also

- [Devices and GPU](devices-and-gpu.md) for device selection and the CUDA compatibility matrix.
- [Devices and GPU](devices-and-gpu.md) for the raw session boundary inside an engine.
- [Dispatch and engine cost](dispatch-and-engine-cost.md) for the CPU counterpart and the
  measured per-call costs.
