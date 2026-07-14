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

```rust
use tenferro_cpu::{CpuBackend, CpuPlacement};

let backend = CpuBackend::new();
for node in backend.topology().nodes() {
    println!("OS node {}: {:?}", node.id(), node.cpus().as_usize_vec());
}

let all = backend.for_placement(CpuPlacement::AllAllowed)?;
println!("{:?}", all.execution_info());
# Ok::<(), tenferro_cpu::CpuPlacementError>(())
```

## Managed Placement Is a faer Contract

`CpuPlacement::NumaNode` and `CpuPlacement::AllAllowed` are supported by
`CpuBackendKind::Faer`. tenferro creates a fixed Rayon engine for the resolved
CPU set and pins every worker when the engine is constructed. `CpuBackend`
clones are cheap handles: they share topology, engines, arbitration, and
engine-owned caches.

```rust
use tenferro_cpu::{CpuBackend, CpuBackendKind, CpuPlacement};

let coordinator = CpuBackend::with_threads_and_kind(4, CpuBackendKind::Faer)?;
if let Some(node) = coordinator.topology().nodes().first() {
    let local = coordinator.for_placement(CpuPlacement::NumaNode(node.id()))?;
    let another_handle = local.clone();
    assert_eq!(local.resolved_placement(), another_handle.resolved_placement());
}
# Ok::<(), Box<dyn std::error::Error>>(())
```

`Auto` resolves to `AllAllowed` for faer. An all-allowed engine can use all CPUs
granted to the process, so splitting work by NUMA node does not prevent a
separate all-node computation. Overlapping placements are serialized by the
shared coordinator; disjoint node placements may execute concurrently.

## External BLAS Providers

OpenBLAS, Intel MKL, Apple Accelerate, and OpenMP-backed BLAS implementations
own their worker creation and affinity. Their thread-count settings do not prove
that workers stay inside a requested tenferro CPU set. Consequently external
BLAS backends accept only `CpuPlacement::Auto`; explicit `NumaNode` and
`AllAllowed` requests return `CpuPlacementError`.

`Auto` for `CpuBackendKind::Blas` uses provider-default execution under an
exclusive coordinator permit. This prevents tenferro-managed CPU work from
overlapping a provider call whose worker CPU set is unknown. Provider variables
such as `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, and `OMP_NUM_THREADS` still
control counts where supported, but they do not upgrade the provider to a
tenferro-managed affinity contract.

If strict NUMA placement is required, select `CpuBackendKind::Faer`. If an
application configures and pins a BLAS provider independently, that remains an
application/provider responsibility outside the tenferro placement guarantee.

## Where Elementwise Rayon Runs

For a faer backend, elementwise, analytic, reduction, structural, indexing, and
faer GEMM work execute inside the selected fixed Rayon engine. With the default
`Auto` placement this is the all-allowed process CPU set; with `NumaNode(id)` it
is that node's process-visible CPU set.

For a BLAS backend, a graph keeps one exclusive coordinator permit. Native
tenferro segments enter the pinned all-allowed Rayon engine, while BLAS/LAPACK
provider calls execute outside that Rayon pool. Thus an elementwise operation
adjacent to BLAS does not run on an unconstrained global Rayon pool, and the
provider is not nested inside tenferro's Rayon workers.

Supported Host instructions, native instructions, and session-capable GEMM FFI
instructions share one backend session. Extension runtimes that cannot execute
through `BackendSession` remain explicit session boundaries.

## Diagnostics

Use `CpuBackend::execution_info()` for logs. `CpuBackendKind::{Faer, Blas}` is
the stable public provider identity. `provider_diagnostic()` may mention a
compiled provider such as OpenBLAS or a runtime-injected provider, but that
string is diagnostic only and may change.

```rust
let backend = tenferro_cpu::CpuBackend::new();
let info = backend.execution_info();
println!("kind={:?} provider={}", info.backend_kind(), info.provider_diagnostic());
println!("requested={:?} resolved={:?}",
    info.requested_placement(), info.resolved_placement());
```

See [Parallelism and Caching](parallelism-and-caching.md) for thread budgets,
cache limits, and oversubscription guidance.
