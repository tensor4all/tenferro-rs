# Accepted Umbrella Plan: Execution Engine and Provider Architecture

## Purpose and status

This document is the single planning entry point for issue
[#1433](https://github.com/tensor4all/tenferro-rs/issues/1433). It controls
scope, decomposition, ordering, status, and acceptance gates for the execution
engine and provider redesign. It intentionally does not repeat the detailed
architecture or prototype mechanics stored in child documents.

The umbrella architecture round is closed and this document is accepted for
child planning. It does not freeze exact public signatures, authorize a
monolithic implementation, or replace current normative runtime, backend,
extension, or GPU specifications. Each implementation phase requires an
accepted child issue and its own reviewed implementation plan; further design
iteration belongs to those bounded children rather than to expansion of this
umbrella.

This revision incorporates every issue #1433 contract added after `f777b52e`,
including the two-axis CPU capability model, additional provider
classifications, batched-contraction target policy, MPI compatibility, the
requirements consolidated from closed issues, and the post-review fast-path,
extension-AD, ownership, control-flow, lifecycle, transfer/collective,
module-first, and pre-1.0 migration contracts.

## Document authority

The documents have distinct roles:

| Document class | Authority |
| --- | --- |
| Current repository rules and normative specs | Define behavior that implementation must preserve until deliberately updated |
| This umbrella | Defines project scope, child boundaries, dependencies, status, and cross-cutting gates |
| Architecture detail | Defines the proposed target contracts and their rationale |
| Child design | Defines one bounded mechanism or experiment without expanding the umbrella scope |
| Worklog | Records measurements and completed-work evidence; it does not introduce requirements |
| Child implementation plan | Defines the ordered code changes for one accepted child issue |

When two documents appear to conflict, their owning roles decide the result.
For example, a worklog measurement may update evidence but cannot weaken an
acceptance gate, and a child prototype may refine its mechanism but cannot add
a second scheduler contrary to the umbrella. Current normative specifications
remain authoritative over this accepted planning architecture until an
accepted implementation updates them.

## Tracked documents

| Artifact | Role | Status | Next gate |
| --- | --- | --- | --- |
| [Execution-engine architecture detail](../../design/execution-engine-provider-architecture.md) | Full semantic, compiler, runtime, provider, resource, and device design | Accepted architecture for child planning | Refine exact mechanisms in child issues |
| [Current-main eager baseline](../../worklogs/2026-07-20-eager-main-baseline.md) | Reproducible pre-refactor public eager evidence | Complete | Reuse unchanged for candidate comparisons |
| [ResourceArbiter uncontended fast path](https://github.com/tensor4all/tenferro-rs/blob/23d152f3/docs/superpowers/specs/2026-07-20-resource-arbiter-uncontended-fast-path-design.md) | Independent prototype for reducing an existing CPU entry cost | Formally inconclusive; not promoted | Preserve branch and evidence; no prototype PR |

Files under `docs/superpowers/plans/` are not planning authority for this
umbrella. A child implementation plan may be generated there for execution,
but durable decisions stay in the tracked specs and completed-work evidence
stays in worklogs.

## Objective

The redesign restores a prism-like dependency direction:

- backend-neutral operation semantics form a portable `SemanticProgram`;
- runtimes implement only the operation-family capabilities they support;
- provider traits isolate replaceable algorithms such as faer, BLAS/LAPACK,
  TBLIS, and device-native kernels;
- one runtime owns compilation, scheduling, resources, caches, events, and
  admission policy; and
- eager and graph execution share semantic and provider contracts without
  forcing eager calls through graph-only artifacts.

CPU, tenferro-native GPU, XLA, and third-party runtimes fit the same semantic
boundary. Multi-NUMA and multi-GPU resources are represented from the start,
while automatic sharding, distributed tensors, collectives, and structured
control flow remain later child designs.

## Cross-cutting invariants

Every child issue and implementation plan must preserve these invariants:

1. `SemanticProgram` is backend- and runtime-neutral. Runtime resources appear
   only in prepared or scheduled artifacts.
2. Operation-family traits are small and explicit. Unsupported capabilities
   return typed errors; they do not trigger implicit provider fallback, CPU
   fallback, or device transfer.
3. Registration-time discovery may use dynamic identity, but steady-state
   dispatch uses pre-resolved typed slots without string lookup or downcasting.
4. Providers execute with runtime-supplied resources. They do not own another
   scheduler, thread pool, stream registry, admission controller, or cache
   hierarchy.
5. CPU thread policy has one source of truth. A provider called inside an
   existing CPU execution session must not call `CpuContext::install` or
   acquire a second resource permit.
6. The migration reuses the current `CpuBackend` session boundary. A
   Tenferro-managed eager session amortizes backend admission and executor entry
   over the session, and a compatible graph session amortizes them over one
   same-domain segment. Per-operation `ParallelMode` contexts are derived
   inside that entered scope without another install. Fallible
   `ExternalManaged` executors retain operation-level entry until a separately
   accepted fallible session API can return pre-callback admission failures.
   New provider layers must not add another entry at either surface. Eager
   remains a first-class supported API, but this architecture promises
   current-main non-inferiority rather than a positive fixed-overhead speedup.
7. Internal provider/composite calls may reuse the session's context, owner,
   and permit only by direct delegation. This is not arbitrary backend
   re-entry. Current `CpuBackend` re-entry rejection remains unchanged unless
   a separately accepted child design replaces that safety contract.
8. ResourceArbiter fast-path work reduces an existing cost only. It is not a
   prerequisite for provider extraction, eager-context design, or the broader
   architecture.
9. CPU and GPU graph execution converge on the same dependency, buffer
   lifetime, event, failure, and observability contracts without pretending
   that their executors or memory domains are identical.
10. Caches have explicit owners, bounded defaults, clear/configuration APIs,
   and entry/byte statistics.
11. No phase makes a performance claim without release-mode measurements that
    pin relevant thread, placement, and provider configuration.
12. CPU thread-count and CPU-placement capabilities are independent. Strict
    exact-domain placement rejects external BLAS with a budget greater than
    one unless the domain is the process's complete allowed CPU set; advisory
    placement remains explicit and observable.
13. `Managed` CPU topology is restricted to the process-allowed cpuset, and
    each MPI rank may own an independent runtime without process-global mutable
    tenferro state. Communication stays at an explicit application boundary.
14. Removed by amendment (2026-07-24). DMRG-class workload contracts left the
    umbrella scope; see [Amendment record](#amendment-record-2026-07-24).
15. Eager fast-path eligibility is decided after pure eager-local lowering.
    Exactly one executable semantic operation may remain direct; two or more
    lowered operations require prepared-graph execution.
16. Extension AD migrates explicitly from the current linearize, linear
    transpose, and optional direct-primal-VJP roles to validated
    `SemanticProgram -> SemanticProgram` transforms owned by `AdContext`.
17. The strong ownership graph is acyclic: runtime state, caches, prepared
    plans, events, and in-flight runs never strongly own public tensor wrappers
    or point back to their runtime owner.
18. Dropping execution, tensor, or runtime handles is non-blocking. In-flight
    state retains storage, resources, external owners, and events through
    completion; cancellation and blocking shutdown are explicit operations.
19. Transfer and collective are scheduler-owned core node families with
    provider implementations. A transfer explicitly bridges event domains;
    neither family may be hidden as an arbitrary extension operation.
20. The program component begins as `tenferro_runtime::program`. A public
    `tenferro-program` crate is extracted only after semantic stability and a
    direct external-consumer need are demonstrated without a dependency cycle.
21. Tenferro is pre-1.0: phases may make clean breaking changes and update
    in-repository callers atomically. They do not add deprecation periods or
    public compatibility shims; temporary internal staging ends with its phase.
22. Structured control flow remains deferred, but its child must start from a
    region/block model with block arguments, terminators, yields, and typed
    shape joins and prove representative branch and loop programs before API
    work begins.

## Workstreams and dependencies

The architecture lane changes ownership and execution boundaries. The
independent `ResourceArbiter` performance experiment is complete and was not
promoted. Its negative or inconclusive result does not block the architecture
lane. Future eager fixed-overhead work is optional, begins with a fresh cost
decomposition, and is not an acceptance dependency for phase 1.

### Architecture lane

The numbered order is the default migration order. A later phase may begin
design review early, but implementation cannot assume an earlier contract that
has not landed or been provided by temporary internal staging in that child's
own implementation.

| Phase | Scope | Required predecessor | Status |
| ---: | --- | --- | --- |
| 0 | Record current-main eager dispatch and CPU-entry evidence | None | Eager baseline complete; CPU-entry diagnostics collected |
| 1 | [Borrowed validated requests, typed CPU capability slots, and layout/GEMM/general-contraction provider composites](https://github.com/tensor4all/tenferro-rs/issues/1434) with temporary internal staging | Phase 0 evidence and accepted child issue | Complete on the restart branch |
| 2 | Placement-bound eager contexts, `CpuDomainExecutor`, explicit parallel modes, and managed/external NUMA domains | Phase 1 contracts | Complete on the restart branch; managed session-entry repair is `b5a3dcd2` |
| 3 | `tenferro_runtime::program`, immutable `SemanticProgram`, fingerprints, effects, aliases, extension AD trait migration, and `TraceContext`/`GraphCompiler` separation | Phase 1 request/schema decisions and accepted semantic-IR child design | Complete under issue #1449 |
| 4 | Immutable runtime snapshots, remaining core capabilities, extension modules, prepared operations, specialization, and bounded caches | Phase 3 semantic artifact | Ready for a bounded child design after the 2026-07-24 amendment |
| 5 | Common `ScheduledGraph`, core transfer/collective node boundary, event domains, buffer planning, admission, and runtime-owned `GraphExecutor`; port CPU graph execution | Phases 2 and 4 | Not started |
| 6 | Extension capability resolution and pure core lowering, validated first with N-ary einsum including per-call-changing shapes, then linalg, FFT, sparse, and permutation families | Phases 4 and 5 | Planned terminal implementation phase; exact-commit audit runs at its boundary |
| 7 | Split CUDA/WebGPU resources from algorithms and port native GPU execution to the common scheduler | Phase 5 common executor | Deferred by maintainer direction on 2026-07-24 |
| 8 | Integrate XLA through `SubgraphCompiler` and prepared operations; retire executor-shaped portable artifacts | Phases 4 and 5 | Deferred by maintainer direction on 2026-07-24 |
| 9 | Schedule independent work across multiple GPUs | Phase 7 device-resource model | Deferred by maintainer direction on 2026-07-24 |
| 10 | Design logical sharding, collective providers, resharding, and region/block structured control flow with representability proofs | Explicitly accepted follow-up designs | Deferred |

Phase numbers describe architectural dependency, not pull-request size. Each
phase may be split into multiple accepted child issues, but every issue must
name the phase and the exact contract it advances.

### Consolidated issue-to-phase mapping

| Issue | Phase ownership | Retained contract |
| --- | ---: | --- |
| Open #1434 | 1 | Borrowed CPU requests, direct capability slots, provider composites, temporary internal staging, and eager current-main non-inferiority |
| Closed #1432 | 1 | Validated general contraction, provider composites and extension-owned linalg bundles, temporary internal staging, engine-owned fan-out, and dispatch evidence |
| Closed #1417 | 2 | Externally managed executor lifetime, exact declared-domain arbitration, placement-resolved registries, and honest affinity diagnostics |
| Closed #1422 | 3 | Opaque builder tokens, typed cross-builder rejection, atomic import/finish, supported extension construction, and private representation |
| Open #1426 H8 | Phase 1/2 target-policy child | Batched `dot_general` thresholds and tiny-kernel policy remain open and benchmark-owned |

Closing the first three issues did not discard their requirements; the detailed
architecture is their durable design home. Tactical fixes remain independent
when they do not alter these contracts or the phase dependencies above.

### Independent ResourceArbiter performance lane

Current CPU execution already pays the session-entry cost under investigation.
The proposed architecture is required to reuse that entry rather than adding a
second one.

Preliminary same-host diagnostics for an empty closure measured these medians:

| Entry path | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| `CpuContext::install` | 0.56 ns | 5.15 us | 5.60 us |
| `CpuBackend::install` | 6.92 us | 6.45 us | 7.02 us |

The one-thread context has no Rayon pool and executes the closure directly.
For two and four threads, Rayon pool entry itself is already about 5-6 us. The
minimal ResourceArbiter prototype can remove only backend admission/release
overhead; it is not expected to remove multi-thread Rayon entry cost.

The completed paired run was formally `INCONCLUSIVE` because its predeclared
host-noise gate failed. Even if that noise failure is ignored, the primary
one-thread `CpuBackend::install` improvement was about 11.8%, below the
predeclared 20% continuation threshold. The prototype is therefore preserved
as negative/inconclusive evidence with no PR and no promotion. No further
session-entry prototype is required before phase 1.

The accepted eager policy is non-inferiority: eager remains first-class, but
this architecture is not required to reduce the current 9-11 us small-call
baseline. A future eager fixed-overhead child may reconsider a budget-one
direct path only after separating validation, eager bookkeeping, admission,
session entry, provider dispatch, and output allocation. Budget one already
avoids Rayon pool entry in `CpuContext::install`, so thread count alone is not
an adequate mechanism or explanation.

### Adjacent work outside this umbrella

Issue [#1377](https://github.com/tensor4all/tenferro-rs/issues/1377), prepared
caller-owned factorizations and reusable provider workspace, proceeds
independently on the current architecture. Its present-day LAPACK work-array
and faer scratch costs do not wait for this redesign.

When phase 6 migrates linalg capabilities, it adopts #1377's established
mapping: prepared plan to `PreparedOperation`, reusable workspace to
`BufferContract` plus engine-owned resources, and factorization session to the
engine execution context and lease. It must not invent an incompatible second
plan/workspace lifecycle. Conversely, #1377 does not need to implement this
umbrella's runtime before its accepted slices can proceed.

## Acceptance gates

### Gate A: issue and scope

No implementation phase starts from this umbrella alone. Its child issue must
be accepted and must state public/API breakage, in-repository migration and
temporary internal staging, excluded work, and the normative documents it will
update.

### Gate B: execution ownership

Every child touching CPU dispatch must demonstrate that:

- no second scheduler or thread pool was introduced;
- provider code does not nest `CpuContext::install` inside the current backend
  session;
- internally composed provider operations receive and reuse the outer context,
  resource owner, and permit by direct delegation;
- arbitrary nested `CpuBackend` entry retains the current rejection behavior
  unless a separate accepted design changes it;
- faer and native kernels remain within the selected `CpuContext`; and
- BLAS/LAPACK global or provider-local thread controls are represented
  honestly rather than modeled as a supplied Rayon executor;
- thread-count and CPU-placement capability are probed and reported separately;
  and
- strict placement returns a typed configuration or prepare error when an
  external BLAS budget greater than one cannot honor the exact domain.

### Gate C: eager non-inferiority

Before a child changes the eager path, it must declare its comparison statistic,
allowed regression, repetition policy, and noisy-run handling. Those choices
are fixed before candidate measurements. Candidate runs reuse the tracked
current-main benchmark source, compiler/profile, CPU pinning, backend thread
count, provider selection, and result-consumption semantics.

The tracked source starts with
`crates/tenferro-ad/benches/eager_dispatch_baseline.rs` and must cover small
elementwise, reduction, contraction, and indexed eager operations before a
candidate is measured. Existing einsum and linalg benchmarks cover extension
and multi-operation cases. Acceptance requires no statistically significant
regression beyond the predeclared threshold and no new steady-state allocation
or string lookup in the measured eager hot path.

The umbrella does not select one universal percentage for all phases. Each
child owns a threshold appropriate to the surface it changes, but cannot
replace the current-main evidence with a faster post-refactor baseline.

### Gate D: hot-path cost attribution

Performance-sensitive children measure validation, request construction,
dispatch, resource entry, provider call, and kernel work separately where
practical. They must cover representative ranks, shapes, batch counts, thread
counts, layouts, NUMA placements, and devices for the behavior they change.
No steady-state path adds per-call heap allocation, whole-program hashing,
string lookup, or graph-level scheduling to an eligible eager operation.

Phase 6 additionally benchmarks eager N-ary einsum over a predeclared sequence
whose concrete shapes change on every call and therefore miss exact-shape
specializations. The baseline and candidate use the same TCI-representative
shape sequence and output-consumption boundary. Acceptance requires
current-main non-inferiority for the complete call, including lowering,
preparation, and execution; a same-shape cache-hit benchmark cannot substitute
for this gate.

### Gate E: correctness and explicit failure

Contract tests cover capability resolution, unsupported behavior, placement,
resource release on error or unwind, effect ordering, buffer lifetime, and
numeric parity. Execution never retries silently on another engine or moves
tensor payloads across a device boundary implicitly.

Runtime tests also prove acyclic ownership with weak sentinels, non-blocking
handle/tensor/runtime drop, retained resources through delayed completion, and
safe event-generation reuse across success, failure, and both cancellation
timings. Transfer tests prove source-to-destination event-domain bridging.
The phase 3 AD child migrates FFT plus test extension rules and establishes JVP
and VJP parity before removing the old graph-specific rule traits.

CPU tests cover process-allowed cpuset discovery, strict versus advisory
placement, provider count/placement classification, and externally managed
domain diagnostics. MPI-compatibility tests use independent rank-like runtimes,
explicit contiguous mutable host export/import, and reproducible cross-rank
planning without introducing a core MPI runtime.

### Gate F: documentation and evidence

When a child lands, it updates the affected normative design or spec, rendered
parallelism documentation, this status table, and a concise worklog. Completed
measurements remain evidence artifacts and are not rewritten as plans.

## Amendment record: 2026-07-24

Issue #1433
[proposed](https://github.com/tensor4all/tenferro-rs/issues/1433#issuecomment-5064768643)
and
[accepted](https://github.com/tensor4all/tenferro-rs/issues/1433#issuecomment-5066101808)
four changes before phase 4:

- core lowering is a portability contract; an eager-latency-supported
  extension supplies a native engine when lowering emits multiple executable
  core operations;
- einsum uses one shared planning core with mandatory native engines on eager
  CPU and GPU surfaces. Its pure planning inputs include resolved runtime
  defaults, budgets, hard workspace limits, and determinism seed. Identical
  paths are required for `Reproducible` under the same resolved snapshot,
  inputs, and hardware class, but `Fast` has no cross-backend path-identity
  requirement. Concrete-shape planning completes before executor/resource
  admission;
- executor entry is amortized across Tenferro-managed sessions and compatible
  same-domain graph segments. Fallible external executors keep operation-level
  entry until a typed fallible session API is accepted; and
- DMRG-class workload contracts leave this umbrella while the TCI
  changing-shape einsum gate and independent MPI contracts remain.

The fast-path eligibility rule, practical non-inferiority policy, phase order,
and completed phase 3 contracts are unchanged.

The subsequent
[implementation direction](https://github.com/tensor4all/tenferro-rs/issues/1433#issuecomment-5066159995)
limits the active sequence to phases 4, 5, and 6. Phases 7 through 9 are
deferred. The restart design's exact-commit cross-phase audit is not umbrella
phase 9: it runs at the phase 6 boundary, covers phases 1 through 6, and is part
of phase 6 completion. The CUDA/WebGPU native einsum engine defers with phase
7; the CPU native engine remains mandatory in phase 6.

## Immediate next actions

1. Create and review the bounded phase 4 child for immutable runtime snapshots,
   remaining core capabilities, extension modules, prepared operations,
   specialization, and bounded caches.
2. Preserve the ResourceArbiter prototype branch and inconclusive evidence;
   create no prototype PR. A future eager fixed-overhead experiment is optional
   and starts from a new complete cost decomposition.
3. Begin phase 4 implementation only after its child design and implementation
   plan are accepted. Do not begin later phases by growing this umbrella.
4. Update this document whenever a child is accepted, completed, superseded,
   or blocked; detailed mechanics stay in the child document.

## Status vocabulary

- **WIP design recorded**: discussion is captured but does not authorize code.
- **Accepted for planning**: maintainers accepted scope; a child implementation
  plan may be written.
- **Implementation in progress**: an accepted child plan is being executed.
- **Evidence pending**: code exists but required correctness, performance, or
  documentation gates have not passed.
- **Complete**: implementation and all child gates passed and the umbrella was
  updated.
- **Deferred**: deliberately outside the active migration sequence.
- **Superseded**: replaced by a linked design; retained only for history.
