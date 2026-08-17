# CPU Execution and NUMA Placement

tenferro distinguishes a CPU thread count from CPU affinity. A thread count
limits parallelism; affinity determines which logical CPUs may execute those
threads. On NUMA machines, setting only the count does not keep work in one
memory-locality domain.

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

let all = backend.for_placement(CpuPlacement::AllAllowed)?;
println!("{:?}", all.execution_info());
```
<!-- end-snippet-source -->

## Managed Placement Is a faer Contract

The initial capability matrix is deliberately conservative:

| Runtime CPU provider | `Auto` | `NumaNode(id)` | `AllAllowed` |
| --- | --- | --- | --- |
| faer and tenferro-native kernels | managed all-allowed engine | managed pinned node engine | managed pinned all-allowed engine |
| OpenBLAS, MKL, Accelerate, or another external BLAS | provider-default, process-wide exclusive | unsupported | unsupported |

On platforms where tenferro cannot set and verify worker affinity, faer's
`Auto` mode uses an unpinned compatibility context. Explicit managed placement
still returns an error instead of silently weakening the request.

`CpuPlacement::NumaNode` and `CpuPlacement::AllAllowed` are supported by
`CpuBackendKind::Faer`. tenferro creates a fixed Rayon engine for the resolved
CPU set and pins every worker when the engine is constructed. `CpuBackend`
clones are cheap handles: they share topology, engines, arbitration, and
engine-owned caches.

<!-- snippet-source: docs/tutorial-code/src/bin/core_tensor_snippets.rs#cpu_execution_29 -->
```rust
use tenferro_cpu::{CpuBackend, CpuBackendKind, CpuPlacement};

let coordinator = CpuBackend::with_threads_and_kind(4, CpuBackendKind::Faer)?;
if let Some(node) = coordinator.topology().nodes().first() {
    let local = coordinator.for_placement(CpuPlacement::NumaNode(node.id()))?;
    let another_handle = local.clone();
    assert_eq!(local.resolved_placement(), another_handle.resolved_placement());
}
```
<!-- end-snippet-source -->

`Auto` resolves to `AllAllowed` for faer. An all-allowed engine can use all CPUs
granted to the process, so splitting work by NUMA node does not prevent a
separate all-node computation. Overlapping placements are serialized by the
process-wide arbiter, including placements created by independently constructed
backends; disjoint node placements may execute concurrently. Other top-level
executions remain subject to these overlap rules.

CPU backend execution is not reentrant. Do not call a backend clone or another
CPU backend directly from `CpuBackend::install` or a backend session, and do not
make backend calls from Rayon tasks spawned inside one. A managed scope rejects
same-thread, spawned, stolen, and shared-context re-entry with a panic before a
second permit is acquired. Work moved to an unrelated executor cannot always
inherit that diagnostic marker and may instead wait for the outer permit, so
waiting for it from the outer execution can deadlock. Finish the outer backend
execution before launching new top-level backend calls. Ordinary Rayon work
that does not re-enter a CPU backend remains supported.

### Scoped direct faer calls

Downstream code that needs a faer routine not exposed by tenferro can use
`tenferro_cpu::FaerParallelismExt` on the active `BackendSession`:

```text
backend.with_backend_session(|session| {
    session.with_faer_parallelism(|parallel| {
        faer_operation(..., parallel)
    })
})?;
```

The callback receives the same policy as an internal faer operation: bounded
`Par::rayon(n)` for managed multi-thread inner execution, and `Par::Seq` for
one-thread or already-inner/outer-worker execution. The capability is lexical;
it does not expose the Rayon pool, executor handle, mutable CPU context, or a
value that can outlive the callback. Calling tenferro backend/session methods
from the callback remains unsupported and retains the existing reentrancy
diagnostics. This guarantee applies to direct faer/Rayon-compatible calls, not
to workers created internally by OpenBLAS, MKL, Accelerate, or OpenMP.

## External BLAS Providers

OpenBLAS, Intel MKL, Apple Accelerate, and OpenMP-backed BLAS implementations
own their worker creation and affinity. Their thread-count settings do not prove
that workers stay inside a requested tenferro CPU set. Consequently external
BLAS backends accept only `CpuPlacement::Auto`; explicit `NumaNode` and
`AllAllowed` requests return `CpuPlacementError`.

`Auto` for `CpuBackendKind::Blas` uses provider-default execution under a
process-wide exclusive permit. This prevents tenferro-managed CPU work from
overlapping a provider call whose worker CPU set is unknown. Provider variables
such as `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, and `OMP_NUM_THREADS` still
control counts where supported, but they do not upgrade the provider to a
tenferro-managed affinity contract.

If strict NUMA placement is required, select `CpuBackendKind::Faer`. If an
application configures and pins a BLAS provider independently, that remains an
application/provider responsibility outside the tenferro placement guarantee.

Fallible backend constructors return `CpuBackendError`. Configuration failures
appear as `CpuBackendError::Tensor`, while topology discovery and engine
placement failures remain inspectable through `CpuBackendError::placement_error`.

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

The reduced worker budget is spread deterministically over the logical CPU IDs
in a domain. This is not a promise to prefer physical cores over SMT siblings;
tenferro does not currently infer core/sibling topology for that selection.

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

Use `CpuBackend::execution_info()` for logs. `CpuBackendKind::{Faer, Blas}` is
the stable public provider identity. `provider_diagnostic()` may mention a
compiled provider such as OpenBLAS or a runtime-injected provider, but that
string is diagnostic only and may change.

Runtime registration uses the opaque `CpuRuntimeIdentity` witness token for
exact backend identity. Clones of one backend share the token; a newly
constructed backend or a backend whose provider bundle, placement, or shared
allocation domain changes receives a distinct token. The token carries no
execution or storage authority and is not a provider/device identifier.

<!-- snippet-source: docs/tutorial-code/src/bin/core_tensor_snippets.rs#cpu_execution_30 -->
```rust
let backend = tenferro_cpu::CpuBackend::new();
let info = backend.execution_info();
println!("kind={:?} provider={}", info.backend_kind(), info.provider_diagnostic());
println!("mode={:?} workers={}", info.execution_mode(), info.worker_count());
println!("topology={:?} requested={:?} resolved={:?}", info.topology(),
    info.requested_placement(), info.resolved_placement());
```
<!-- end-snippet-source -->

See [Parallelism and Caching](parallelism-and-caching.md) for thread budgets,
cache limits, and oversubscription guidance.
