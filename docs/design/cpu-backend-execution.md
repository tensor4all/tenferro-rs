# CPU Backend Execution Contract

`CpuBackend` is a cloneable handle to a shared coordinator. The coordinator
owns process-visible topology, lazily constructed tenferro-owned engines,
overlap-aware execution arbitration, and engine-local buffer and plan caches.

Topology uses sparse OS CPU and NUMA node IDs. A usable node CPU set is
`OS node cpuset ∩ process affinity`; `AllAllowed` is the process affinity set.
No code may reinterpret node IDs as dense indexes or widen process affinity.

## Provider Selection Is Compile-Time

Exactly one CPU backend is compiled: `native` (the default) or `blas`.

- `native` uses faer-backed tlinalg plus cpueinsum/tcontract.
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

`CpuExecutionContext` is the one owner-scoped interop seam. cpueinsum, tcontract,
and tlinalg receive one parallelism token: `Sequential`, or the selected pool
with a thread budget. The lower library owns lane selection, vendor batching,
scratch, and scheduling within that budget; tenferro owns tensor semantics,
placement, output allocation, validation, and cache lifetime.

One lower-library numerical call may install the selected pool for the duration
of that call. Strided-rs work and FFT lane fan-out use the same bounded,
operation-local exception. tenferro never installs the caller's continuation.

One caller-owned exception exists, and it is explicit rather than implicit: a
[held session's phase lease](./held-cpu-session-1945-phase-lease.md) runs the
caller's own lane callbacks on the workers of the context's pool. A lane is the
coordinator of the numerical calls it makes and reaches tenferro through an
owner-inheriting child session, so ordinary session entry stays admission-only
and lower-library lane ownership is unchanged. A phase never creates a pool, and
vendor threading remains outside tenferro's budget.

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

## A Fused Elementwise Region Is Planned And Selected

The runtime plans a fused elementwise region for a prepared program and asks the
session to execute it as one command. The CPU session selects the region and
replays it through `tenferro_cpu_fused::elementwise_fusion_with_pool`; the
executor still declines (returns `None`, so the runtime dispatches the region's
instructions one by one) below its 16,384-element floor, above its four-input
limit, and for dtypes or ops without a fused kernel.

`strided-fused 0.5.2` made the selection viable. Its chunked register machine
runs the `elem-simd-math` slice kernels over chunk-sized buffers and now treats a
stride-`0` operand as a broadcast value, so a broadcast on the contiguity axis —
the case that fell back to `interpret_inner_loop_uninit` and lost about 16x on
0.5.1 — stays on the vectorized replay. `tenferro-cpu-fused` also compiles one
`strided-fused` plan carrying every live output and executes it with
`ErasedFusedPlan::execute_outputs_uninit`, so a region with several live outputs
replays its instruction list once instead of once per output.

The selection is scoped to what was measured. Only `F32`/`F64` regions are
selected, because 0.5.2's vectorized chunk replay covers only those dtypes;
C32/C64, integers and bool stay unfused rather than run the unmeasured
per-element interpreter. An ordered real op (`Maximum`/`Minimum`/`Clamp`) also
declines, because the upstream kernels use Rust `.max`/`.min`, which drop NaN
while tenferro's ordinary kernels propagate it.

The measurement (2026-10-17, `AMD EPYC 7713P`, 1T and 4T) is recorded in
[`docs/worklogs/perf-cpu-fusion-declined-2140.md`](../worklogs/perf-cpu-fusion-declined-2140.md),
section "Re-enabled on `strided-fused 0.5.2` (#2161)". The decision rule actually
applied was "median over all recorded runs per case `<= 1.05`", and it passes
with the largest per-case median 1.02; the contiguity-axis broadcast and the
two-live-output region, the two cases that lost on 0.5.1, are among the wins.
The GPU backend reaches the same runtime machinery with its own `#[cube]` code
generator and is unaffected.

`CpuBackend::max_fused_region_inputs` reports the executor's four-input limit:
the limit describes the executor, and the session selects a region only inside
it.

## Diagnostics

`CpuBackend::placement()`, `num_threads()`, `topology()`, and
`buffer_pool_stats()` are the observation accessors. `CpuRuntimeIdentity` is an
opaque witness token for backend identity; clones share it and a backend whose
placement or shared allocation domain changes receives a new one. The token
carries no execution or storage authority.
