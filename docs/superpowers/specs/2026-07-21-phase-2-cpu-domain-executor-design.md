# Phase 2 Design: CPU Domain Executors and External NUMA Domains

## Status

Accepted for implementation as child issue
[#1436](https://github.com/tensor4all/tenferro-rs/issues/1436) under the
maintainer directive to execute the accepted issue #1433 architecture through
phase 9. This child design is bounded to phase 2. It refines exact Rust
mechanisms without changing the umbrella's semantic-program, scheduler,
extension, GPU, or XLA scope.

The ordered TDD tasks are recorded in
[`docs/superpowers/plans/2026-07-21-phase-2-cpu-domain-executor.md`](https://github.com/tensor4all/tenferro-rs/blob/codex/execution-engine-through-phase9/docs/superpowers/plans/2026-07-21-phase-2-cpu-domain-executor.md).

The implementation branch is `codex/execution-engine-through-phase9`, based on
the preserved phase-1 evidence head `8a0baf42`. Phase 1 remains an
`INCONCLUSIVE` performance handoff rather than a promoted result. Phase 2 may
use its contracts as temporary internal staging, but it must compare every
changed eager path directly with the tracked current-`main` baseline.

## Objective

Phase 2 makes CPU execution domains explicit and replaceable while preserving
one execution owner, one resource lease, and one parallel fan-out decision per
operation. It provides:

- an object-safe `CpuDomainExecutor` contract;
- explicit `ParallelMode::{Sequential, Outer, Inner}` policy;
- tenferro-owned `Managed` domains and caller-owned `ExternalManaged` domains;
- a placement-indexed external-domain registry on one `CpuBackend`
  coordinator;
- placement-bound eager execution without creating another runtime, pool, or
  backend session; and
- independent thread-count and CPU-placement capability validation.

The existing managed NUMA implementation, topology discovery, global
`ResourceArbiter`, engine-local buffers/caches, and backend re-entry rejection
are migration inputs. They are generalized rather than replaced by a second
resource system.

## Selected Approach

Use the external-domain registry proposed as approach B in closed issue #1417.
Each registered domain retains an `Arc<dyn CpuDomainExecutor>`, its placement
identity, declared CPU set, thread budget, placement guarantee, and resource
owners. `CpuBackend::for_placement` resolves both managed and external domains
through the same registry-facing engine path.

Rejected alternatives are:

1. One `CpuBackend` per external pool. This cannot provide one
   placement-indexed coordinator and makes placement selection and shared
   arbitration awkward.
2. Accept only `Arc<CpuContext>`. This makes the public contract Rayon-specific
   and cannot represent executors that support only outer scheduling or only
   sequential execution.
3. Bypass `ResourceArbiter`. Work submitted directly by the application remains
   the application's responsibility, but tenferro work must still conflict with
   overlapping managed/external domains and provider-exclusive execution.

## Public and Provider Contracts

### Parallel mode

The phase-1 `CpuKernelParallelism` staging enum is replaced, not retained as a
compatibility alias:

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ParallelMode {
    Sequential,
    Outer,
    Inner,
}
```

`Outer` means the engine-owned composite partitions jobs through the selected
domain executor and gives every delegated provider a sequential child context.
`Inner` means the outer loop is sequential and one provider kernel may enter an
inner parallel region. `Sequential` permits neither fan-out. A provider never
changes the mode and never invokes ambient Rayon independently.

### Object-safe executor

The executor operates on borrowed, stack-owned jobs without allocating on the
synchronous eager path:

```rust
pub trait ScopedCpuJob: Send {
    fn run(&mut self) -> Result<(), CpuDomainExecutorError>;
}

pub trait ScopedCpuJobs: Sync {
    fn len(&self) -> usize;
    fn run(&self, index: usize) -> Result<(), CpuDomainExecutorError>;
}

pub trait CpuDomainExecutor: Debug + Send + Sync + 'static {
    fn capabilities(&self) -> CpuDomainExecutorCapabilities;

    // Synchronous fork/join submission for engine-owned outer scheduling.
    // All borrowed jobs have completed when this method returns.
    fn submit(&self, jobs: &dyn ScopedCpuJobs)
        -> Result<(), CpuDomainExecutorError>;

    // Synchronous entry into one provider-owned inner parallel region.
    fn install(&self, job: &mut dyn ScopedCpuJob)
        -> Result<(), CpuDomainExecutorError>;
}
```

`submit` is deliberately synchronous and indexed. This gives arbitrary safe
executors a borrowed fork/join boundary without a Rayon-specific iterator or
an unsafe escaping closure. Asynchronous graph enqueue and completion events
belong to the phase-5 scheduler, not this executor trait.

The executor error reports executor admission, scheduling, cancellation, or
panic-bridge failures. A tensor/provider result is stored by the stack-owned
job and recovered by the caller, so this boundary does not erase an operation
error into `CpuDomainExecutorError`.

The standard implementation adapts the existing pinned Rayon pool. Managed
construction owns that adapter. Applications may supply another implementation
without exposing its pool type through tenferro. A public convenience Rayon
adapter may be added only if it wraps, rather than reconstructs, a caller-owned
pool and its API does not become the sole external-managed path.

### Executor capabilities

Capabilities are immutable construction-time facts:

```rust
pub struct CpuDomainExecutorCapabilities {
    pub worker_count: NonZeroUsize,
    pub outer_parallelism: bool,
    pub inner_parallelism: CpuInnerParallelism,
    pub reentrancy: CpuExecutorReentrancy,
    pub affinity: CpuExecutorAffinity,
    pub shutdown: CpuExecutorShutdown,
}
```

`CpuInnerParallelism` distinguishes Rayon-compatible inner regions from no
inner-region support. `CpuExecutorAffinity` distinguishes tenferro-pinned and
verified workers, caller-declared/unverified workers, and no placement claim.
`CpuExecutorShutdown` identifies tenferro versus external ownership. Diagnostics
report these facts; tenferro never upgrades a caller declaration to verified
affinity.

Capability booleans are enforced. Selecting `Outer` without outer support or
`Inner` without compatible inner-region support returns a typed error before a
kernel mutates output.

### Resource domain and execution context

`CpuEngine` owns one immutable crate-private `CpuResourceDomain`:

```rust
pub(crate) struct CpuResourceDomain {
    id: CpuDomainId,
    placement: ResolvedCpuPlacement,
    executor: Arc<dyn CpuDomainExecutor>,
    thread_budget: NonZeroUsize,
    placement_guarantee: CpuPlacementGuarantee,
    ownership: CpuDomainOwnership,
}

pub struct CpuExecutionContext<'a> {
    domain: &'a CpuResourceDomain,
    parallel_mode: ParallelMode,
    lease: &'a ResourcePermit,
}
```

The lease remains crate-private. Providers can query the domain ID, declared
CPU set, thread budget, placement guarantee, and mode, and can request the
context's checked `submit`/`install` helpers. They cannot acquire another
permit, choose another pool, or access mutable engine resources directly.

Phase-1 provider traits take `&CpuExecutionContext<'_>` in place of
`&CpuProviderContext<'_>`. The old staging type is removed atomically. Direct
eager and prepared/session dispatch construct the same context below the one
backend-session entry.

### Provider count and placement axes

Provider traits expose an immutable `CpuProviderExecutionCapabilities` value.
It records independently:

- thread-count control (`Sequential`, per-call upper-bound control, binary
  single/auto control, or uncontrolled/global control);
- placement control (engine workers, calling thread only, external workers, or
  none);
- whether a worker-local sequential inner kernel can be forced; and
- whether the provider accepts `Sequential`, `Outer`, and `Inner` calls.

Faer and native providers run on engine workers. A one-thread external BLAS
call runs inline and therefore honors both axes. A strict exact domain with an
external-BLAS budget greater than one is rejected unless the domain is exactly
the process-allowed CPU set. Advisory placement remains allowed only when the
count upper bound can still be enforced, and the advisory status remains
observable.

The existing explicit BLAS features classify conservatively at construction:

- MKL uses its thread-local setter for count but never claims exact per-domain
  placement above one thread;
- recent pthread OpenBLAS claims per-call count only after the local setter and
  pthread mode are both observed;
- OpenMP OpenBLAS and ArmPL `_mp` do not claim exact thread-local count;
- macOS 15 Accelerate exposes binary single/auto count control; older
  Accelerate exposes only startup-global control;
- serial ArmPL/NVPL-style providers are sequential by construction; and
- injected or unknown BLAS remains conservative unless the injector supplies
  an explicit capability descriptor.

No symbol probing occurs per operation.

Installing a replacement `CpuProviderBundle` becomes fallible because the
bundle must be checked against every registered domain. The current infallible
`with_provider_bundle` staging method is changed directly rather than retained
beside a second `try_*` API.

## Managed and ExternalManaged Construction

### Managed

Managed construction keeps the existing behavior:

- topology is intersected with the process-allowed CPU set;
- tenferro creates and pins Rayon workers;
- workers are verified during construction;
- the global arbiter admits exact CPU-set requests;
- engine-local resources remain construct-once/reuse; and
- arbitrary nested `CpuBackend` entry continues to panic with the existing
  documented re-entry message.

Internal provider/composite delegation reuses `CpuExecutionContext`; it is not
backend re-entry.

### ExternalManaged descriptor and registry

```rust
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct CpuDomainId(u64);

pub struct ExternalCpuDomain { /* private fields */ }

impl ExternalCpuDomain {
    pub fn new(
        id: CpuDomainId,
        placement: ResolvedCpuPlacement,
        executor: Arc<dyn CpuDomainExecutor>,
        thread_budget: NonZeroUsize,
        placement_guarantee: CpuPlacementGuarantee,
    ) -> Result<Self, ExternalCpuDomainError>;
}

impl CpuBackend {
    pub fn from_external_managed_domains(
        default_domain: CpuDomainId,
        domains: impl IntoIterator<Item = ExternalCpuDomain>,
    ) -> Result<Self, CpuBackendError>;
}
```

`CpuDomainId` is supplied or selected inside one coordinator; it is not issued
by a process-global counter. Long-running computation therefore cannot exhaust
IDs merely by executing operations.

The builder validates nonempty registries and CPU sets, unique domain and
placement identities, process-allowed CPU membership, a present default
domain, nonzero executor workers, and a thread budget no greater than the
declared worker count. Distinct overlapping CPU sets are accepted and recorded;
the arbiter serializes them. This is required for deterministic overlap tests
and for coexistence with independently constructed managed coordinators.

The backend retains each executor `Arc` through every engine, active session,
and submitted scoped operation. It never reconstructs a pool, repins workers,
or verifies live external affinity. `CpuExecutionMode::ExternalManaged` and
`CpuExecutionInfo` report domain identity, declared CPUs, worker count, thread
budget, exact/advisory guarantee, unverified affinity, and external shutdown
ownership.

`CpuPlacement::Auto` resolves to the explicitly selected default domain.
`NumaNode` and `AllAllowed` resolve only to registered matching external
domains; they never create a managed pool as a fallback.

## Placement-Bound Eager Context

Phase 2 adds a CPU placement binding to the current eager runtime without
creating a second `EagerRuntime` or changing tensor runtime identity:

```rust
let socket0 = runtime.on_cpu(CpuPlacement::NumaNode(node0))?;
socket0.with_eager_session(|session| run_job(session, input0))?;
```

`CpuPlacementBoundEager` retains the original runtime and a resolved, cheap
`CpuBackend` handle. It does not own a lease or hold a backend mutex while idle.
`on_cpu` locks the original eager runtime once, verifies that it is CPU-backed,
clones a resolved placement handle, and releases the lock. The placement
context is used mutably, so it needs no second backend mutex.
`with_eager_session` enters exactly one selected backend session and passes a
borrowed `&mut dyn BackendSession` adapter to the closure. The bridge operates
on concrete `Tensor` values through the existing public backend-session traits;
it does not ask ordinary `EagerTensor` methods to recursively lock the runtime.
Nested operation-family calls through the borrowed session delegate below that
session boundary and do not call `CpuBackend::install` again.

This scoped surface is the phase-2 bridge. Phase 4 attaches the same placement
binding to the general immutable `RuntimeSnapshot` and adds AD-aware
operation-family ergonomics such as `cpu0.matmul(...)`; it must not replace the
resource-domain or executor contract introduced here. Phase 2 does not add a
duplicate set of placement-specific `EagerTensor` methods.

The default unbound eager path remains byte-for-byte on its current dispatch
shape except for type-level context migration. No thread-local placement,
ambient runtime singleton, second backend mutex, or long-lived permit is added.

## Arbitration, Re-entry, and Failure

- Exact managed and external domains acquire `ResourceRequest::CpuSet` with
  their concrete declared sets.
- Advisory external domains use the same declared sets for cooperative
  arbitration while diagnostics state that worker placement is not verified.
- Disjoint requests may coexist; overlaps and provider-exclusive requests
  serialize through the existing FIFO-aware arbiter.
- The same internal execution owner may delegate within its active session,
  but public/backend entry from the active thread or worker scope keeps the
  current rejection contract.
- Executor, provider, validation, configuration, poisoned-resource, and
  unsupported-capability failures remain typed. No path falls back to a
  different domain/provider or silently executes in the global Rayon pool.
- RAII releases permits and mutable resources after success, error, or unwind.
  External executor ownership remains alive until all synchronous submitted
  jobs have returned.

## Data Flow

```text
placement request
    -> resolve managed/external CpuResourceDomain
    -> validate provider count + placement axes
    -> acquire one ResourceArbiter lease
    -> create borrowed CpuExecutionContext
    -> engine composite selects one ParallelMode
       -> Outer: executor.submit(indexed jobs), child provider Sequential
       -> Inner: sequential outer loop, executor.install(one provider region)
       -> Sequential: direct provider call
    -> return/reclaim output in its AllocationDomainId
    -> release lease
```

The default mixed-CPU-affinity selection policy described by the umbrella is
staged here as a pure resolver shared by eager binding and later prepared input
binding. It chooses the domain with greatest logical input bytes and breaks ties
by `CpuDomainId`; unknown affinities do not contribute. No implicit copy or
rehome occurs. `RequireSingleDomain` is a typed opt-in rejection policy.

## Testing

TDD tasks must first add tests that fail for the missing contract. Required
coverage includes:

- object safety and stack-borrowed `submit`/`install` jobs;
- managed Rayon adapter capability truthfulness and no ambient-pool leakage;
- `Sequential`, `Outer`, and `Inner` each producing exactly one fan-out level;
- per-job provider contexts becoming sequential under outer fan-out;
- unsupported executor/mode combinations failing before output mutation;
- two fake external domains selected from one coordinator without new pool
  construction;
- executor owner retention through a delayed job and release after completion;
- disjoint-domain concurrency, overlapping-domain serialization,
  provider-exclusive conflict, fairness, error release, and unwind release;
- duplicate/empty/out-of-cpuset/default/budget validation errors;
- exact versus advisory diagnostics and no live-affinity claim;
- provider count and placement capability tables, including conservative
  unknown/ArmPL/NVPL classifications;
- strict external BLAS budget-one acceptance and budget-greater-than-one
  rejection for a subdomain;
- existing Managed, Compatibility, and ProviderDefaultExclusive behavior;
- placement-bound eager execution using one runtime identity and one backend
  session entry;
- mixed CPU affinity dominant-input selection and `RequireSingleDomain`;
- no allocation or string lookup in the unbound eager/provider dispatch path;
  and
- Linux opt-in observed-CPU integration without making verification part of
  ExternalManaged semantics.

All new public types and methods require runnable doctests. Production modules
keep only `#[cfg(test)] mod tests;`; substantive tests live in module-local test
files or integration tests.

## Performance Gate

Before candidate measurement, phase 2 fixes the following protocol:

- reuse the tracked current-`main` eager benchmark, lockfile, compiler/profile,
  CPU pin, provider, thread count, output consumption, and phase-1 interleaved
  pair runner;
- run the complete small elementwise, reduction, indexed, and contraction
  matrix, not only placement-bound calls;
- require all three valid 95% interval upper bounds to be at most +5% for each
  default-path case, with the existing A/A sentinel and invalid-pair rules;
- require no new allocation, allocated bytes, backend/session entry, string
  lookup, or downcast on the default eager path;
- separately measure placement-bound session entry, managed/external executor
  `submit` and `install`, and grouped outer/inner scheduling at 1, 2, and 4
  threads; and
- record NUMA hardware evidence only when the machine actually exposes
  multiple usable nodes. Fixture correctness is mandatory everywhere.

An `INCONCLUSIVE` campaign is preserved as evidence and is not called a pass.
It does not authorize changing the baseline to the phase-1 or phase-2 branch.

## Documentation and Migration

The phase updates:

- `docs/design/execution-engine-provider-architecture.md` for final signatures;
- `docs/design/cpu-backend-execution.md` for executor/domain ownership;
- `docs/guides/cpu-execution.md` and
  `docs/guides/parallelism-and-caching.md` for user responsibilities and
  provider limits;
- the issue #1433 phase table and the phase-2 child issue; and
- a curated phase-2 worklog with tests, benchmarks, decisions, and residual
  risks.

The phase may make clean pre-1.0 breaking changes. It does not leave aliases for
`CpuKernelParallelism` or `CpuProviderContext`, and it does not expose mutable
resource pools, permits, or `CpuEngine` internals. Its placement bridge reuses
the existing public `BackendSession` contract. Temporary internal staging is
removed before the phase is called complete.

## Excluded Work

- `SemanticProgram`, graph specialization, and runtime snapshots (phases 3-4);
- common `ScheduledGraph`, event domains, and asynchronous graph enqueue
  (phase 5);
- extension lowering and operation-family migration (phase 6);
- CUDA/WebGPU resources, XLA, and multi-GPU scheduling (phases 7-9);
- NUMA page migration or a public distributed tensor;
- arbitrary nested public `CpuBackend` entry; and
- core MPI communication.

## Exit Criteria

Phase 2 exits only when its child issue is accepted, every required behavior
test and doctest passes, the default eager performance and allocation gates
pass, all touched docs and the worklog are current, repository-rule review has
no unresolved finding, and no temporary phase-2 dispatcher or duplicated
resource path remains.
