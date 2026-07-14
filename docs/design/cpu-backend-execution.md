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

Synchronous nested execution propagates one logical owner across Rayon pools,
so clone and independent-backend work can reenter permits already held by that
execution. Reentrant operations use transient scratch buffers and analysis
caches rather than re-locking an outer session's engine resources. Separate
logical executions remain governed by process-wide overlap and provider
exclusion.

The stable public identity is `CpuBackendKind::{Faer, Blas}`. Concrete provider
names are diagnostic strings, not dispatch or compatibility keys.
