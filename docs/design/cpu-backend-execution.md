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
affinity of provider-owned OpenMP or native workers. Native graph segments and
provider calls both cross the selected all-allowed domain executor exactly
once. The executor entry owns admission and caller-thread placement; the BLAS
runtime, rather than the executor's Rayon workers, owns provider fan-out.

The unentered crate-private `CpuOperationEntry` holds the selected domain and
resource permit. It is the only CPU backend value allowed to call executor
`install` or `submit`. Provider-facing `CpuExecutionContext` values are created
inside an installed job or an outer child, so they are always already entered
and expose only immutable policy/accessors. This separates executor entry,
logical `ParallelMode`, and provider worker ownership and prevents a provider
or operation-family implementation from re-entering the executor.

A supported graph execution holds one permit and one backend session across
Host operations, native operations, and session-capable FFI operations.
Non-session extension runtimes are boundaries. Cache ownership follows engine
ownership so clone handles do not duplicate retained execution state.
The session stores `CpuOperationEntry`, not a provider context; each operation
performs exactly one executor entry. Direct and session execution therefore
have identical placement and call-count contracts.

After that single executor entry, the already-entered
`CpuExecutionContext` scopes tenferro-native strided work. `Inner` plus a
selected executor that advertises Rayon uses that executor's Rayon region,
capped by the validated operation budget. `Sequential`, every `Outer` child,
and `Inner` backed by external workers use a sequential native policy. Thus
native kernels never inherit an unrelated ambient Rayon pool, outer fan-out
cannot create nested native fan-out, and an external BLAS runtime may still
fan out independently of the sequential strided policy.

If direct GEMM dispatch reports exactly `Layout(Lhs)`, `Layout(Rhs)`, or
`Conjugation` as unsupported, dot-general materializes canonical operands and
retries the same provider once. Conjugation is fused into materialization and
the retry flags are cleared. Output-layout and every other unsupported reason,
a typed provider error, or a second unsupported result are terminal. Both
temporary operands return to the engine buffer pool on every retry exit.

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
with the pool and adds mandatory per-entry allocations. Rayon may still perform
occasional scheduler maintenance allocations; the backend does not promise that
an unbounded sequence of calls remains allocation-free.

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
