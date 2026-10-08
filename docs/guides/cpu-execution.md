# CPU Execution and NUMA Placement

tenferro distinguishes a CPU thread count from CPU affinity. A thread count
limits parallelism; affinity determines which logical CPUs may execute those
threads. On NUMA machines, setting only the count does not keep work in one
memory-locality domain.

For session versus worker-dispatch overhead and ways to amortize it, see
[CPU Session Entry and Rayon Dispatch Cost](session-entry-cost.md).

## What a NUMA Node Means

For this API, a NUMA node is an operating-system topology domain identified by
its sparse OS node ID and its set of logical CPUs. A node is usable only through
the intersection of that OS CPU set with the process affinity mask. Therefore:

- `NumaNodeId(2)` means OS node 2, not the third entry in a dense array;
- CPUs excluded by a container, scheduler, cpuset, or `taskset` are never added
  back by tenferro;
- an OS node with an empty process-visible intersection is unavailable; and
- `AllAllowed` means the complete process affinity mask, not every CPU installed
  in the host.

Inspect the process-visible topology before selecting a node:

<!-- snippet-source: docs/tutorial-code/src/bin/core_tensor_snippets.rs#cpu_execution_28 -->
```rust
use tenferro_cpu::{CpuBackend, CpuPlacement};

let backend = CpuBackend::new();
for node in backend.topology().nodes() {
    println!("OS node {}: {:?}", node.id(), node.cpus().as_usize_vec());
}

if backend.supports_placement(CpuPlacement::AllAllowed) {
    let all = backend.for_placement(CpuPlacement::AllAllowed)?;
    println!("{:?}", all.placement());
} else {
    println!("{:?}", backend.placement());
}
```
<!-- end-snippet-source -->

## Managed Placement

Placement is resolved against the process topology and always constructs a
tenferro-owned engine:

| `CpuPlacement` | Resolution |
| --- | --- |
| `Auto` | the managed all-allowed engine |
| `AllAllowed` | a managed engine pinned to the process-permitted CPU set |
| `NumaNode(id)` | a managed engine pinned to that OS node's CPUs |

On platforms where tenferro cannot set and verify worker affinity, `Auto` uses
an unpinned compatibility context. Explicit placement still returns a typed
error instead of silently weakening the request.

All placement choices belong to the same backend: the CPU numerical provider is
selected at compile time (`native`, the default, or `blas`), never per handle.
tenferro creates one fixed Rayon engine for the resolved CPU set and confines
**every worker to that whole set** when the engine is constructed; workers
share the domain mask instead of owning one CPU each. They share it because a
thread created by a provider (BLAS/LAPACK, or your own library call) inherits
the creating thread's mask: a one-CPU worker mask would confine the provider's
entire thread team to one CPU. `CpuBackend` clones are cheap handles: they
share topology, engines, arbitration, and engine-owned caches.

## Scoped direct faer calls

Downstream code that needs a faer routine not exposed by tenferro can use
`tenferro_cpu::FaerParallelismExt` on the active `BackendSession`:

```text
backend.with_backend_session(|session| {
    session.with_faer_parallelism(|parallel| {
        faer_operation(..., parallel)
    })
})??;
```

The callback receives the same policy as an internal faer operation: bounded
`Par::rayon(n)` for managed multi-thread inner execution, and `Par::Seq` for
one-thread or already-inner/outer-worker execution. The capability is lexical;
it does not expose the Rayon pool, executor handle, mutable CPU context, or a
value that can outlive the callback. Calling tenferro backend/session methods
from the callback remains unsupported and retains the existing reentrancy
diagnostics. This guarantee applies to direct faer/Rayon-compatible calls, not
to workers created internally by OpenBLAS, MKL, Accelerate, or OpenMP.

## Vendor BLAS/LAPACK

A `blas` build adds cpueinsum-blas and tlinalg-blas as compiled adapters; it is
not a vendor-only execution world, and the native path stays available inside
it. tenferro calls the vendor from the coordinator thread and makes no placement
or thread-count promise for the threads the vendor creates. The calling
thread's mask is the one tenferro guarantees: because placement only ever
narrows the caller's own mask and never widens it, a caller already pinned to
one CPU also confines a newly created vendor team.

Provider variables such as `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, and
`OMP_NUM_THREADS` still control vendor counts where supported. They are outside
tenferro's resource contract, and `threads(1)` on a tenferro backend does not
imply a one-thread vendor call.

## Batched Operation Strategy

Lane selection, vendor batching, and the per-item route for batched work
(strided-batched and grouped GEMM, packed LU factor/solve, and the batched
linalg families) belong to the lower numerical libraries: cpueinsum, tlinalg,
and tlinalg-blas. tenferro passes one parallelism token - the selected pool with
a thread budget, or `Sequential` - and the lower library decides its own lanes
within that budget. There is no host batch-policy setting.

A contraction whose axes are all batch axes is an elementwise product and never
lowers to per-element GEMMs.

## CPU Affinity Is Not NUMA Memory Placement

Pinned workers restrict where computation may run. They do not make tensor
allocation NUMA-local, choose a first-touch policy, migrate existing pages, or
configure page interleaving. Input and output pages may therefore remain remote
from the selected node. Applications that require memory locality must arrange
allocation/first-touch or OS memory policy separately and measure the result on
their deployment topology.

Fresh CPU results record the selected execution domain in
`Placement::cpu_affinity`. This is routing and locality metadata: it identifies
the domain that produced the result and can guide later scheduling. It is not
proof that the storage was allocated by that domain, that its pages are pinned
or resident there, or that an allocation-domain owner changed. Metadata-only
views and reshapes retain their storage metadata, while caller-owned `_into`
destinations are never retagged.

The worker budget is not mapped onto individual CPUs: every worker is confined
to the same domain CPU set and the operating system schedules the pool inside
it. That is not a promise to prefer physical cores over SMT siblings; tenferro
does not infer core/sibling topology for a domain.

## Where Elementwise Rayon Runs

For a faer backend, elementwise, analytic, reduction, structural, indexing, and
faer GEMM work execute inside the selected fixed Rayon engine. With the default
`Auto` placement this is the all-allowed process CPU set; with `NumaNode(id)` it
is that node's process-visible CPU set.

For a BLAS backend, a graph keeps one exclusive coordinator permit. Native
tenferro segments and BLAS/LAPACK provider calls both cross the selected domain
executor exactly once. The BLAS call runs inside that admitted operation
region, but the provider runtime owns its worker fan-out; it does not use the
executor's Rayon team as BLAS workers. Thus an elementwise operation adjacent
to BLAS does not run on an unconstrained global Rayon pool, and no provider path
bypasses domain admission.

Supported Host instructions, native instructions, and session-capable GEMM FFI
instructions share one backend session. Extension runtimes that cannot execute
through `BackendSession` remain explicit session boundaries.

## Diagnostics

Use `CpuBackend::placement()`, `CpuBackend::num_threads()`,
`CpuBackend::topology()`, and `CpuBackend::buffer_pool_stats()` for logs.
`tenferro_cpu::cpu_provider_id()` names the compile-time provider
(`tenferro.cpu.faer` or `tenferro.cpu.blas`).

Runtime registration uses the opaque `CpuRuntimeIdentity` witness token for
exact backend identity. Clones of one backend share the token; a newly
constructed backend or a backend whose placement or shared allocation domain
changes receives a distinct token. The token carries no execution or storage
authority and is not a provider/device identifier.
