# CPU Backend Execution Contract

`CpuBackend` is a cloneable handle to a shared coordinator. The coordinator
owns process-visible topology, lazily constructed tenferro-owned engines,
overlap-aware execution arbitration, and engine-local buffer and plan caches.

Topology uses sparse OS CPU and NUMA node IDs. A usable node CPU set is
`OS node cpuset ∩ process affinity`; `AllAllowed` is the process affinity set.
No code may reinterpret node IDs as dense indexes or widen process affinity.

## Provider Selection Is Compile-Time

Exactly one CPU backend is compiled: `native` (the default) or `blas`.

- `native` uses faer-backed tlinalg plus cpueinsum/tprims.
- `blas` adds the `cpueinsum-blas` and `tlinalg-blas` adapters and keeps the
  native path for steps those adapters decline.

Enabling both, or neither, is a compile-time error. There is no runtime
provider kind, provider bundle, kernel slot, or per-handle provider selection.
`tenferro_cpu::cpu_provider_id()` names the compiled provider for diagnostics
only; it is not a dispatch key.

## Placement And Ownership

Every engine is tenferro-owned. `CpuPlacement::{Auto, NumaNode, AllAllowed}`
resolves against the process topology: `Auto` and `AllAllowed` use the
process-permitted CPU set, `NumaNode(id)` a node's CPUs. `CpuBackend::builder()`
selects a CPU set, NUMA node, thread count, worker stack, and buffer limit;
`CpuBackend::new()` / `with_threads(n)` use the environment thread count and the
process CPU set.

An engine has a fixed Rayon pool confined to its declared CPU set and verified
at construction. Workers share the domain mask instead of owning one CPU each,
because provider-created threads inherit the creating worker's mask and a
one-CPU mask would confine a provider's whole thread team. Overlapping CPU sets
cannot hold permits concurrently; disjoint sets can. On platforms without
verified worker affinity, `Auto` uses an unpinned compatibility context and
explicit placement returns a typed error rather than weakening the request.

A clone shares topology, engines, arbitration, and engine-owned caches.
tenferro never borrows an application-owned pool or executor.

## Session Entry Is Admission, Not Installation

`with_backend_session` and `with_execution_scope` are admission-only: the
permit, the caller-affinity guard, and the workspace lease are acquired, and
the callback and the coordinator stay on the calling OS thread. tenferro does
not install the callback into a pool.

Entry with an explicitly declared CPU set observes the current thread mask `O`
and narrows to `O ∩ D`, never widening it; a disjoint request returns a typed
error before numerical work. `AllAllowed` performs no affinity syscall. The
guard restores the mask on normal return, error, and unwind; the unwind path
makes a best-effort restore and records a diagnostic rather than panicking.

A Rayon worker never waits for a conflicting permit or an eager backend owner:
worker entry acquires non-blocking and reports the typed `Contended` /
`Reentered` state on conflict. Ordinary independent callers keep blocking FIFO
admission. Direct recursion is rejected from caller-local session state, so
reentry rejection does not depend on pool-wide registration.

## The Lower-Library Seam

`CpuExecutionContext` is the one owner-scoped interop seam. cpueinsum, tprims,
and tlinalg receive one parallelism token: `Sequential`, or the selected pool
with a thread budget. The lower library owns lane selection, vendor batching,
scratch, and scheduling within that budget; tenferro owns tensor semantics,
placement, output allocation, validation, and cache lifetime.

One lower-library numerical call may install the selected pool for the duration
of that call. Strided-rs work and FFT lane fan-out use the same bounded,
operation-local exception. tenferro never installs the caller's continuation.

Vendor BLAS/LAPACK is called from the coordinator thread. tenferro guarantees
the calling thread's mask and nothing about the vendor's own worker team,
thread count, or placement; `threads(1)` on a tenferro backend does not imply a
one-thread vendor call.

The token is not uniform across the two linalg providers, by design: the
faer-backed `tlinalg` owns batch lanes and takes `Parallel` (pool plus budget),
while `tlinalg-blas` is deliberately token-free because LAPACK/BLAS own their
own threading, its batch loop is serial, and a Rayon fan-out around a vendor
call would fight the vendor's own pool. The host places the vendor call instead;
its scratch still comes from tenferro's buffer pool through the workspace seam.

## Data Path

CPU floating/complex `dot_general` and grouped GEMM go to cpueinsum prepared
binary/grouped plans; CPU N-ary concrete einsum delegates to cpueinsum with the
order tenferro's planner selected; CPU linalg families go to tlinalg /
tlinalg-blas. The integer/all-batch elementwise contraction stays in
tenferro-cpu. tenferro keeps axis, layout, dtype, placement, and output-form
validation, the `MaybeUninit` complete-coverage proof for fresh binary outputs,
and the typed unsupported mapping.

Prepared plans are immutable and cacheable: they hold no live operands, no
execution lease, and no mutable scratch. Execution scratch lives in an
owner-scoped workspace with its own retention bound and accounting; retained
plan and buffer caches are bounded and clearable through the runtime cache
owner.

Every successfully returned fresh CPU allocation records the selected resource
domain as `Placement::cpu_affinity`. The tag is routing/locality metadata, not
allocation ownership or evidence of NUMA page placement, worker pinning, or
memory residency. Storage-sharing views and metadata-only reshapes preserve the
source metadata; caller-owned `_into` outputs preserve the caller's metadata.

## Diagnostics

`CpuBackend::placement()`, `num_threads()`, `topology()`, and
`buffer_pool_stats()` are the observation accessors. `CpuRuntimeIdentity` is an
opaque witness token for backend identity; clones share it and a backend whose
placement or shared allocation domain changes receives a new one. The token
carries no execution or storage authority.
