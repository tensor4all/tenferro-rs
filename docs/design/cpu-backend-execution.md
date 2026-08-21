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

## Caller-managed external admission

An external domain explicitly selects one of two admission contracts:

- **cooperative CPU-set admission** retains the existing resolved placement,
  placement guarantee, and process-wide overlap arbitration; or
- **caller-managed admission** declares no CPU set. The supplied executor's
  workers are the complete CPU universe for that domain, and the caller owns
  admission and oversubscription across caller-managed domains.

`ExternalCpuDomain::new` remains the cooperative constructor.
`ExternalCpuDomain::new_caller_managed(id, executor, thread_budget)` selects the
second contract. `ExternalCpuDomain::admission_mode()` and
`CpuExecutionInfo::admission_mode()` expose the distinction. Placement and CPU
set accessors on `ExternalCpuDomain`, `CpuExecutionInfo`, and provider-facing
`CpuExecutionContext` return `None` for caller-managed domains rather than
inventing a placement claim. `CpuBackend::for_domain(CpuDomainId)` selects any
registered external domain by stable ID; placement selection remains available
only for cooperative domains.

Caller-managed execution never enters `ResourceArbiter`, so unrelated
caller-managed domains do not conflict because their process affinity overlaps.
Instead, each caller-managed domain owns a small RAII admission guard carrying
tenferro's execution-owner identity. It rejects a second public backend entry
while that domain is active, including entry from another worker of the same
caller pool, and clears on success, error, or unwind. This preserves public
`CpuBackend` recursive-entry rejection without requiring TLS installation on a
pool built by the caller. The permit never reports arbiter re-entry, so the
existing engine-resource lock continues to protect that domain's caches and
buffer pool. Cooperative, managed, compatibility, and provider-exclusive
permits are unchanged.

The standard caller-owned Rayon adapter retains an `Arc<rayon::ThreadPool>` and
implements the generic `CpuDomainExecutor` contract without constructing or
shutting down a pool. Its synchronous `install` delegates to that exact pool,
including when called by a worker already inside the same pool; inner faer and
strided parallelism therefore sees only that pool and the validated domain
thread budget. `submit` partitions only the indexed jobs supplied by tenferro.
No path may fall back to ambient/global Rayon.

Caller-managed domains explicitly select `CpuBackendKind::Faer`, independent of
`CpuBackendKind::default_compiled()`; construction returns a typed unsupported
configuration error when faer is not compiled. Provider-domain validation has a
distinct caller-managed branch with no advisory or process-all-allowed bypass.
It accepts only thread controls `PerCallUpperBound`, `Sequential`, or
`BinaryClampToOne` and placement controls `EngineWorkers` or `CallingThread`.
Caller-managed domains carry no `CpuPlacementGuarantee`. A bundle with external
workers, uncontrolled thread count, or no placement control fails with a typed
construction error before execution. The current BLAS/LAPACK operation-family
path is unsupported because its process-global or provider-owned workers cannot
be confined to the supplied executor. It must not run and then weaken
isolation, mutate an output, or fall back silently.

Diagnostics report caller-managed admission, absent verified placement, caller
executor/shutdown ownership, worker count, and thread budget. Fresh output
metadata may retain the stable CPU domain ID for routing, but it is not evidence
of CPU affinity or NUMA residency. Executor ownership stays with the caller and
is retained by the domain for every active synchronous job.

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
The session stores `CpuOperationEntry` plus an optional entered execution
context. Tenferro-managed sessions enter once and reuse that executor boundary
while selecting an explicit logical mode for each native or provider
operation. Fallible external executors retain operation-level entry so their
typed admission failures are not replaced by panic or fallback.

Every successfully returned fresh CPU allocation records the selected resource
domain as `Placement::cpu_affinity`. The tag is routing/locality metadata, not
allocation ownership or evidence of NUMA page placement, worker pinning, or
memory residency. Device, memory-kind, and backend allocation-domain metadata
remain independent. Storage-sharing views and metadata-only reshapes preserve
the source metadata; caller-owned `_into` outputs preserve the caller's
metadata. Validation and provider/domain compatibility checks precede fresh
allocation or caller-output mutation, and failures never cause tagging.

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
