# CPU Backend Execution Contract

`CpuBackend` is a cloneable handle to a shared coordinator. The coordinator
owns process-visible topology, lazily constructed fixed engines, overlap-aware
execution arbitration, and engine-local buffer and GEMM-analysis caches.

Topology uses sparse OS CPU and NUMA node IDs. A usable node CPU set is
`OS node cpuset ∩ process affinity`; `AllAllowed` is the process affinity set.
No code may reinterpret node IDs as dense indexes or widen process affinity.

For `CpuBackendKind::Faer`, `Auto` resolves to managed `AllAllowed`, and
explicit node/all-allowed placement is supported. Each managed engine has a
fixed Rayon pool whose workers are pinned and verified at construction.
Overlapping CPU sets cannot hold permits concurrently; disjoint sets can.

For `CpuBackendKind::Blas`, only `Auto` is valid. It resolves to a
provider-default exclusive permit because tenferro cannot establish the CPU
affinity of provider-owned OpenMP or native workers. Native graph segments enter
the fixed all-allowed Rayon engine; provider calls remain outside that pool.

A supported graph execution holds one permit and one backend session across
Host operations, native operations, and session-capable FFI operations.
Non-session extension runtimes are boundaries. Cache ownership follows engine
ownership so clone handles do not duplicate retained execution state.

Backend execution is non-reentrant. An active managed Rayon scope marks both
the root call and every owned worker; direct nesting, a spawned or stolen child
task, and unrelated work submitted to the same active `CpuContext` are rejected
before acquiring another permit. External-provider execution also rejects
same-thread nesting.

Each managed Rayon worker registers the engine's shared execution-scope state
once during `CpuContext` construction, before the constructor returns. An
execution changes that shared active owner under RAII; workers consult the
shared state when a nested backend entry is attempted. Entry must not broadcast
owner metadata to every worker because that makes empty warm execution scale
with the pool and allocate per call.

This propagation contract covers Rayon workers owned by the active
`CpuContext`. Ambient global Rayon workers are not part of a managed execution
scope; tests for child-task propagation must use an explicit owned context.

This intentionally does not infer permission from an execution owner. During a
cross-pool wait Rayon may schedule a parallel sibling on the same OS worker as
the direct call chain, so thread-local identity cannot distinguish those cases.
Treating that sibling as reentrant could enter overlapping engine resources or
an external BLAS provider concurrently. Separate top-level executions remain
governed by process-wide overlap and provider exclusion.

The stable public identity is `CpuBackendKind::{Faer, Blas}`. Concrete provider
names are diagnostic strings, not dispatch or compatibility keys.
