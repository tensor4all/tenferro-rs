# Source publication: blocked, and what has to change first (#1945 U4, source publication)

This is a **blocker record**, not a delivered design. It specifies what "a source session publishes an
immutable storage/read lease plus the producer's ready token or submitted completion frontier"
(#1945's transfer contract, item 1) requires, and records why the design that was reviewed cannot be
implemented on the current architecture. It follows
[asynchronous-transfer-1945-u4.md](./asynchronous-transfer-1945-u4.md), whose §5 lists source
publication as the first remaining blocker.

## 1. A design was written, reviewed, and rejected

The reviewed design was: a `SourcePublication<'source>` bundling a `&'source Tensor` read lease, a
`ManagedResource` retention, the runtime pin and a `CUevent` producer frontier, published by
`publish_source(runtime, &Tensor)` and consumed by `SourcePublication::download_into`, with
`download_pending` re-expressed as that composition. The independent review found, with source
evidence:

1. **BLOCKER - the asynchronous borrow contradicts this repository's storage-ownership contract.**
   `docs/design/storage-ownership-contracts.md` requires accelerator submission to own its admitted
   package, with no caller borrow escaping, and soundness independent of `Drop` or `mem::forget`.
   Forgetting a borrowed-handle transfer ends the compiler-enforced exclusion while the DMA can still
   be outstanding, and `ManagedResource` retention prevents reclamation but not later writes through
   the original tensor. The existing pending transfers have the same limitation, which the package
   record already leaves open as its item 3 ("owning asynchronous submission"); a publication layer
   cannot promote that limitation into a complete immutable-lease contract.
2. **BLOCKER - a borrow is not an immutable lease while a safe writable-binding escape exists.**
   Safe code can take `Session::tensor_binding(&TypedTensor)` (an owned, cloneable `TensorBinding`),
   keep the client (`Session::client`), and call the *safe* `Client::with_server(...)` to reach the
   *safe* `ComputeServer::write(...)` with that binding — overwriting the storage a publication holds
   borrowed, without any `&mut`, `unsafe` block or second tensor owner. A compile-fail fixture for
   `&mut source` does not close this. This is a pre-existing boundary defect of the extension surface,
   not something the publication introduces, and it must be resolved or excluded by an enforceable
   contract rather than documented away.
3. **BLOCKER - the producer frontier is not identified by `publish_source(&CudaRuntime, &Tensor)`.**
   `check_errors` and `get_resource` select the client's *current* stream and `raw_cuda_stream` selects
   `StreamId::current()`; a held session instead executes on its saved stream. CubeCL's handle records
   the stream where storage was *created*, not the last writer, so for a reused allocation written on
   another stream — which native caller-output execution can do and return from without a barrier —
   publishing on the allocation-origin stream can record a frontier and copy without waiting for that
   later producer.
4. **IMPORTANT - cross-stream producer failures can be masked**, because publication's error check
   covers only the publication stream while `get_resource` resolves the allocation's stream with
   `ignore: true, flush: false`; and **the "does not wait" claim overstates the boundary** (host
   dispatch is a blocking round trip, cross-stream resource resolution can block on GC backpressure,
   and dispatching preceding work can trip staging retirement).
5. **IMPORTANT - the acceptance inventory is broader than the plan**: pending H2D, view readiness,
   producer/consumer overlap, bounded staging, route/P2P errors and dependent-consumer admission are
   not covered by a CUDA-source D2H publication, and "session-free" cannot be established by running
   without a session — it needs the session-entry audit.

The review's own conclusion about the smaller alternatives is the useful part: a frontier-only helper
is smaller but gives up source access authority and a resolved consumption path; and for a two-phase
handoff an **opaque ownership package** is justified, whose "access authority cannot consist solely of
an escaped caller borrow". It also recommends cutting the public `source()` accessor, the public
`await_frontier` (the consuming copy's `wait` is the only boundary this package enables) and the
provider-seam speculation.

## 2. What has to change before source publication can be specified

1. **Owning asynchronous submission** (#1945 U4 item 3): the pending transfers — and any publication
   built on them — must own their admitted package (value, buffers, retention, event) instead of
   borrowing it, so that a forgotten handle leaks rather than ending exclusion while the DMA runs.
   That is the prerequisite, and it is a change to the delivered transfer API, not an addition to it.
2. **A producer identity**: publication has to be created by whatever produced the value, or be given
   the producing stream/binding and frontier explicitly, because neither the allocation handle nor the
   current stream identifies the last writer.
3. **An enforceable immutability contract for published storage**, which means closing or explicitly
   excluding the safe writable-binding path (`Session::tensor_binding` plus `Client::with_server` plus
   `ComputeServer::write`). That path is CubeCL-side as much as tenferro-side, so this is a maintainer
   decision with a possible fork change.
4. **Cross-domain admission** (#1945 U4 item 2) before a published frontier can serve a device consumer
   in another execution domain.

## 3. What is not claimed

Nothing in this record is implemented. The pending transfer routes were subsequently converted to
owning admission in both directions (see
[owning-pending-transfers-1945-u4.md](./owning-pending-transfers-1945-u4.md)), which removes the
borrow-based hole this record identified but not the writable-binding escape or the missing producer
identity. Their "no source publication" limitation stays open, and no caller should build a
session-independent consumption path on the assumption that a transfer makes the source immutable.

## 4. Recommendation

Do not attempt source publication again before items 1-3 of §2 are settled; the owning-submission
package (U4 item 3) is the first of them and is implementable inside tenferro-rs without a fork
change, while item 3 needs a maintainer decision about the extension surface.
