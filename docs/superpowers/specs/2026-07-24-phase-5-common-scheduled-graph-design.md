# Phase 5 common scheduled graph design

Date: 2026-07-24
Status: draft child design candidate; not accepted until maintainer review records
zero Critical or Important findings.

## Purpose

Implement the Phase 5 boundary from the accepted execution-engine umbrella:
runtime-owned scheduled execution, explicit transfer/collective/barrier node
families, event-domain metadata, buffer-plan ownership, admission metadata, and
CPU graph execution through the runtime preparation path.

This child deliberately does not implement Phase 6 extension-family migration,
Phase 7 GPU native execution, Phase 8 XLA `SubgraphCompiler`, or umbrella
Phase 9 multi-GPU scheduling. It creates the common representation those
phases consume.

## Authority

- Issue #1433 latest maintainer direction: implement Phases 4 through 8, keep
  umbrella Phase 9 deferred, and include a Phase 5 two-mock-engine transfer
  test with no GPU hardware.
- `docs/superpowers/specs/2026-07-20-execution-engine-provider-umbrella-design.md`.
- `docs/design/execution-engine-provider-architecture.md`.
- `docs/superpowers/specs/2026-07-24-phase-4-runtime-preparation-design.md`.
- Phase 4 closeout worklog
  `docs/worklogs/2026-07-24-phase-4-runtime-preparation.md`.

## Non-negotiable constraints

- No Phase 5 production code starts from the umbrella alone; this child design
  is the Phase 5 authority once accepted.
- `Runtime::prepare_for` and its options-aware crate-private companion remain
  the only preparation entries used by new runtime-owned execution.
- `PreparedProgram` remains binding-free. Phase 5 execution may borrow
  `(&Arc<PreparedProgram>, &ProgramBindings)` for one call after fresh input
  signature validation and exact specialization-projection equality; it must
  retain neither.
- Preparation/cache capacity remains distinct from run and node admission.
- Transfer and collective are scheduler node families, never hidden extension
  operations.
- Managed CPU graph execution enters a same-domain segment once, not once per
  operation. Fallible `ExternalManaged` keeps the existing operation-level
  entry until a separate fallible session API is accepted.
- The old `GraphExecutor<B>` compatibility facade may remain during Phase 5
  only as a legacy facade. Phase 8 owns retirement of executor-shaped portable
  artifacts. New Phase 5 runtime-owned execution must not depend on that facade
  for its public contract.

## Scope decisions

### Public surface

Phase 5 adds one explicit runtime-owned execution surface:

```text
impl Runtime {
    pub fn run_compiled(
        &self,
        program: &CompiledGraph,
        inputs: &[&Tensor],
    ) -> tenferro_runtime::Result<Vec<Tensor>>;

    pub fn run_compiled_values(
        &self,
        program: &CompiledGraph,
        inputs: &[&Tensor],
    ) -> tenferro_runtime::Result<Vec<TensorValue>>;
}
```

These methods are blocking convenience paths. They derive an `InputSignature`
from supplied tensors or semantic defaults, call crate-private
options-aware runtime preparation, validate that the per-call signature projects
exactly to the prepared specialization, then execute the prepared scheduled
graph.

Phase 5 also extends engine registration with one runtime-owned execution
bridge:

```text
impl EngineRegistration {
    pub fn with_tensor_backend_executor<B>(self, backend: B) -> Self
    where
        B: TensorBackend + Clone + Send + Sync + 'static;

    pub fn has_execution_engine(&self) -> bool;
}
```

The bridge is implemented inside `tenferro-runtime` as a private erased
adapter over `B`, `B::RuntimeCache`, and `ExtensionExecutor<B>`. This is not a
new backend trait for downstream authors to implement. It lets `Runtime`
execute its private scheduled/staging representation without depending on
`tenferro-cpu`, `tenferro-gpu`, or any other concrete backend crate.

`tenferro-cpu` adds a public helper:

```text
pub fn runtime_engine_registration(
    backend: &CpuBackend,
) -> Result<EngineRegistration, RuntimeConfigError>;
```

The helper registers the same CPU engine ID, hardware class, storage class,
direct core preparation slots, cache owner, and tensor-backend execution bridge.
The existing `tenferro-ad` eager private helper delegates to this public CPU
helper after Phase 5. No runtime production dependency on `tenferro-cpu` is
introduced.

Phase 5 does not expose public `PreparedGraph`, `ScheduledGraph`,
`ExecutionHandle`, `submit`, cancellation, or asynchronous tensor completion.
Those remain crate-private or deferred because the accepted architecture
defines their direction but not enough public API detail for a stable surface.

The existing public `GraphExecutor<B>` remains for in-repository compatibility
through Phase 5. Documentation must identify it as legacy staging while
`Runtime::run_compiled` is the new runtime-owned path. Phase 8 owns its
retirement after XLA and GPU migration no longer require executor-shaped
artifacts.

`CompiledGraph` remains the carrier for the immutable frozen semantic program,
bindings, and the `CompilerOptions` used to lower it. It must not retain
`ExecProgram` or any execution staging. Preserving `CompilerOptions` is required
so `GraphCompiler::with_compiler_options` keeps the same observable execution
semantics after staging moves behind runtime/legacy execution boundaries.

### Scheduled representation

Create `crates/tenferro-runtime/src/runtime/schedule.rs` with crate-private
types:

```text
pub(crate) struct ScheduledGraph {
    nodes: Box<[ScheduledNode]>,
    input_slots: Box<[usize]>,
    output_slots: Box<[usize]>,
    value_count: usize,
    buffer_plan: BufferPlan,
    segments: Box<[ScheduleSegment]>,
}

pub(crate) enum ScheduledNode {
    Operation(ScheduledOperation),
    Transfer(ScheduledTransfer),
    Collective(ScheduledCollective),
    Barrier(ScheduledBarrier),
}
```

`ScheduledOperation` stores semantic provenance, input/output value slots, the
prepared-operation handle, operation-family metadata, and an execution-staging
payload for Phase 5. The payload is temporary and private; it exists only to
port current CPU graph behavior without freezing old `ExecProgram` as the
Phase 5 API.

`ScheduledTransfer`, `ScheduledCollective`, and `ScheduledBarrier` are
constructible and visible in schedule tests. Phase 5 executes CPU operation
nodes and synchronous mock transfer nodes. Collectives are representable but
return an explicit unsupported execution error until a later collective child
defines providers.

### Event domains

Create crate-private IDs in `runtime/schedule.rs`:

```text
pub(crate) struct EventDomainId(u32);
pub(crate) struct EventSlotId(u32);
pub(crate) struct EventDependency {
    domain: EventDomainId,
    slot: EventSlotId,
    generation: u64,
}
```

Phase 5 uses these IDs for schedule validation and transfer bridging tests. It
does not allocate backend-native event handles or expose asynchronous public
completion. Blocking CPU execution treats operation completion as immediately
observable in the CPU event domain.

### Buffer planning and admission

Phase 5 implements a minimal deterministic `BufferPlan`:

- value liveness from semantic/scheduled slot use;
- output slot list;
- alias/view metadata sufficient to preserve existing lazy `TensorValue::View`
  outputs;
- retained-byte estimates for plan metadata;
- run-level `RunAdmissionSummary` and per-node `NodeAdmissionSummary`.

The initial admission implementation is synchronous and non-queuing: it
validates prepared memory feasibility and then executes. It must still keep the
type boundary separate from the Phase 4 cache. No API may call cache limits
“admission”.

### Runtime-owned CPU execution

`Runtime::run_compiled` uses one snapshot-derived prepared schedule. For a
single CPU engine it forms one contiguous CPU segment and executes that segment
through the registered tensor-backend execution bridge inside one compatible
backend session. This preserves the Phase 2 session-entry repair: a k-operation
same-domain segment performs one managed executor install.

The runtime-owned executor must not create a second scheduler or thread pool.
It reuses the registered CPU engine/cache owner, direct core capabilities, and
tensor-backend execution bridge. If the selected engine can prepare but has no
execution bridge, `Runtime::run_compiled` returns a typed runtime-state error
before admission or node execution.

Runtime-owned preparation for a `CompiledGraph` uses the graph's stored
`CompilerOptions` when constructing the private transitional staging root.
The older crate-private `Runtime::prepare_for(&FrozenProgram, ...)` test seam
may remain with default options, but Phase 5 execution must call an options-aware
crate-private preparation helper so non-default compiler options are preserved.

### Transfer test required by the updated Phase 8 direction

Phase 5 includes a mock-engine schedule test with two engines and one explicit
transfer node:

- engine A produces a value in event domain A;
- transfer A→B consumes that event and produces a destination-domain event;
- engine B consumes the transferred value;
- no GPU hardware or real device transfer is required;
- validation proves the transfer is a first-class scheduled node and that the
  executor does not assume all nodes complete in one event domain.

The test may execute through a synchronous mock transfer provider. It must not
encode the transfer as an extension operation.

## Deletion and retention policy

Phase 5 removes the Phase 4-only private semantic-staging adapter name
`lower_semantic_to_exec_staging` from `compiler/semantic_staging.rs`.
`stage_semantic_program` may remain as a private staging builder consumed by
`Runtime::prepare_for` and the temporary compatibility facade until Phase 8.

Phase 5 may retain old `ExecOp`/`ExecInstruction` implementation helpers
behind the new schedule module for the CPU compatibility path, but only if:

- no public `CompiledGraph` field or public method exposes them;
- `GraphCompiler` no longer constructs execution staging as its compile
  output, though it may transiently stage during compilation to preserve
  existing compile-cache validation/statistics until Phase 8 removes those
  executor-shaped artifacts;
- `CompiledGraph` records the compiler options used by `GraphCompiler`, and
  temporary legacy execution restages with those exact options;
- new runtime-owned execution reaches them only through
  options-aware runtime preparation -> `PreparedProgram` -> `ScheduledGraph`;
- source tests record that Phase 8 owns final removal of executor-shaped
  compatibility artifacts.

## Required tests

Red-first tests must cover:

1. `GraphCompiler` preserves only `FrozenProgram`, bindings, and
   `CompilerOptions` in `CompiledGraph`; it does not retain execution staging.
2. `Runtime::run_compiled` prepares via options-aware runtime preparation,
   increments prepared-plan cache miss/hit statistics, and returns the same
   tensor results as the legacy `GraphExecutor<CpuBackend>` on representative
   add, reduction, indexing, and `dot_general` programs.
3. A stale prepared entry caused by runtime reconfiguration is not executed by
   an explicit prepared execution helper; the convenience path re-prepares.
4. Input dtype, rank, shape, shape-guard, and missing-default failures happen
   before any run-admission or node-execution counter increments.
5. A same-domain managed CPU graph segment executes k operations with one
   managed executor install.
6. Fallible `ExternalManaged` keeps the existing operation-level behavior and
   returns typed executor failures.
7. A two-mock-engine graph contains an explicit transfer node bridging event
   domains, and execution never treats the source event as a destination event.
8. Collective nodes are representable and return explicit unsupported errors;
   they are not accepted through extension registration.
9. `PreparedProgram` and `ProgramBindings` are borrowed for one execution call
   and not retained; weak-sentinel tests prove no strong ownership cycle from
   runtime caches or in-flight execution to public tensor wrappers.
10. Source contracts prove the old semantic-staging adapter name is gone,
    `CompiledGraph` has no `staging` field, and Phase 6 extension migration has
    not started.
11. `EngineRegistration::with_tensor_backend_executor` provides runtime-owned
    execution without a runtime-to-CPU dependency, and missing execution bridge
    is reported before admission.

## Verification

Each implementation node runs:

- focused runtime/graph tests for the changed module;
- `cargo test -p tenferro-runtime`;
- affected CPU integration tests when CPU execution changes;
- `cargo clippy -p tenferro-runtime --all-targets -- -D warnings`;
- `cargo fmt --all --check`;
- `git diff --check`;
- repository-rules review. If the external LLM reviewer remains unusable
  because of HTTP 400, deterministic dry-run review evidence is recorded.

Phase 5 closeout additionally runs:

- `cargo test -p tenferro-runtime`;
- `cargo test -p tenferro-cpu`;
- `cargo test -p tenferro-ad`;
- doc consistency/snippet/site checks;
- fast PR gate with focused test commands, without creating a PR.

## Stop conditions

Stop for maintainer direction if implementation requires:

- public `PreparedGraph`, `ScheduledGraph`, `submit`, or `ExecutionHandle`
  signatures beyond the `Runtime::run_compiled*` blocking convenience methods;
- a runtime production dependency on `tenferro-cpu`;
- a second scheduler, worker pool, or hidden CPU executor entry;
- segment-wide entry for fallible `ExternalManaged`;
- implicit device transfer or collective hidden inside an operation;
- Phase 6 extension-family migration;
- Phase 7 GPU resource split; or
- Phase 8 XLA `SubgraphCompiler`.
