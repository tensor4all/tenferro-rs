# WIP Design: Pluggable Execution Engines and Resource Domains

## Status

This is a work-in-progress architecture proposal. It records the agreed design
direction for discussion and issue decomposition; it is not an accepted public
API or an implementation plan.

The proposal returns tenferro to a prism-like dependency direction: operation
semantics are expressed in a backend-neutral IR and pure schemas, execution
capabilities are expressed as small operation-family traits, and tenferro's
eager and compiled surfaces call the selected engine. Backend libraries remain
replaceable behind smaller provider traits, while resource ownership stays
with the engine.

This document refines rather than immediately replaces the current contracts in:

- `docs/spec/backend-contract.md`
- `docs/spec/extension-op.md`
- `docs/design/gpu-backend-design.md`
- `docs/design/xla-backend.md`
- `docs/superpowers/specs/2026-04-05-execution-layer-separation-design.md`
- `docs/superpowers/specs/2026-07-14-numa-cpu-execution-design.md`

Those documents remain authoritative until an accepted implementation updates
the corresponding normative specs.

## Motivation

The current architecture has accumulated several kinds of coupling:

- a broad backend interface mixes semantic operations with implementation
  choices;
- CPU provider choice, thread ownership, and operation dispatch are difficult
  to vary independently;
- extension dispatch and built-in operation dispatch follow different paths;
- CUDA runtime resources and operation implementations are coupled inside one
  backend object;
- multi-NUMA and multi-GPU scheduling need resources that no individual kernel
  provider should own;
- linalg crates should not require the upstream runtime to know every linalg
  family in advance.

The target is not a generic hook system. Hooks make ownership, fallback, and
capability behavior implicit. The target is a backend-neutral semantic schema,
explicit operation-family and provider traits, and explicit runtime resources.

## Goals

1. Let CPU eager, CUDA/WebGPU eager, and XLA lowering consume the same semantic
   operation contract.
2. Allow runtime replacement of CPU and GPU implementation providers without a
   process-global registry.
3. Keep operation-family crates such as linalg independent of an upstream
   facade or monolithic backend trait.
4. Make thread pools, streams, allocators, caches, topology, and admission
   control explicit engine-owned resources.
5. Support managed and externally managed NUMA execution.
6. Include multi-GPU task scheduling in the resource model from the start.
7. Keep steady-state dispatch allocation-free and avoid string-keyed lookup in
   hot paths.
8. Preserve explicit device transfer, materialization, fallback, and error
   behavior.
9. Make graph compilation produce one backend-neutral semantic artifact that
   CPU, GPU, XLA, and third-party runtimes can all consume.
10. Run CPU and GPU graphs through one runtime-owned scheduler and executor
    contract, including heterogeneous and multi-device graphs.

## Non-goals

- Implementing this architecture in the design branch.
- Adding a root `tenferro` facade crate or `tenferro::cpu` facade path.
- Replacing all existing backend code in one change.
- Automatic tensor sharding, distributed QR/SVD, multi-host execution, or NCCL
  implementation in the initial refactor.
- A public `DistributedTensor` type in the initial refactor.
- Implicit CPU fallback, implicit device transfer, or implicit provider
  fallback.
- Making provider-specific global state, such as OpenBLAS worker management,
  appear runtime-local when the external library does not provide that
  isolation.

## Architectural Overview

The graph path has two compilation stages separated by a portable semantic
artifact:

```text
TracedGraph
    |
    v
GraphCompiler
    |
    v
SemanticProgram                  backend and runtime neutral
    |
    v
Runtime::prepare_for
    +-------------------------+
    |                         |
    v                         v
PreparedGraph             XlaExecutable
    |                     whole-region target
    v
runtime-owned GraphExecutor
```

Eager execution does not run a whole graph compiler for each operation. It
constructs the same semantic descriptors, uses the same validation contracts,
and enters the same runtime capability, provider, resource, and prepared-plan
path through a single-operation fast path.

The current `GraphProgram`, `ExecProgram`, `GraphExecutor<B>`, and
`GraphProgramLoweringView` are migration inputs, not the final abstraction
boundaries. In particular, an executor-shaped instruction stream is not the
portable compiler artifact.

The dependency direction is deliberate:

- tenferro core defines or depends on semantic operation contracts;
- an engine implements those contracts;
- a backend-specific engine delegates algorithms to providers;
- providers receive a per-execution context but do not own scheduling
  resources;
- extension crates add semantic and provider traits without requiring the core
  runtime to enumerate their operation families.

## Compiler and Artifact Boundaries

### `GraphCompiler` and `SemanticProgram`

`GraphCompiler` is a pure semantic compiler. It performs graph merging and root
resolution, SSA construction, dtype and symbolic-shape inference, shape-guard
construction, extension payload validation, view/copy classification, effect
and alias analysis, and target-independent optimization.

Target-independent optimization may include dead-code elimination, common
subexpression elimination, algebraic simplification, propagation and
canonicalization of dtype and shape metadata, view and transpose composition,
constant handling, canonical `dot_general`, and identification of fusion
candidates. It does not select providers, materialize layouts, allocate
buffers, create vendor plans, or decide an actual fusion kernel.

The resulting `SemanticProgram` contains:

- semantic operations and extension payloads;
- SSA values, inputs, and outputs;
- dtypes, symbolic shapes, and shape guards;
- view, copy, effect, alias, and dependency semantics;
- placement requirements and constraints, but not resolved devices;
- target-independent canonicalization results and stable node provenance.

It does not contain:

- provider or extension-runtime selection;
- thread pools, streams, allocators, leases, or runtime handles;
- backend packing, materialization, fusion kernels, or vendor plans;
- buffer allocation and reuse decisions;
- host, FFI, or backend dispatch categories chosen for an executor.

This makes `SemanticProgram` the backend-neutral replacement for the portable
role currently attributed to `GraphProgram`. Existing `GraphProgram` and
`ExecProgram` remain temporary adapters while migration is in progress.

### Runtime plan compilation

`Runtime::prepare_for` is the second compiler stage. Given a
`SemanticProgram`, concrete `InputSignature`, runtime configuration epoch, and
prepare options, it performs:

1. capability-aware legalization, including extension-native versus core
   lowering decisions;
2. provider and extension-slot resolution;
3. final placement and region partitioning;
4. layout materialization and explicit transfer insertion;
5. engine-specific preparation of kernels, library plans, and workspace
   requirements;
6. dependency scheduling, buffer lifetime planning, events, and barriers.

CPU engines may prepare GEMM, decomposition, packing, and thread-domain plans.
GPU engines may prepare kernels, launch configurations, vendor handles, and
stream requirements. XLA consumes the same `SemanticProgram` but may lower a
whole region to StableHLO and a PJRT executable rather than produce per-node
provider operations.

## Semantic IR and Execution Traits

The semantic IR uses a closed core vocabulary with an extension carrier:

```rust
pub enum SemanticOp {
    Core(CoreSemanticOp),
    Extension(Arc<dyn ExtensionOp>),
}
```

The closed core enum enables exhaustive compiler passes, structural hashing,
canonicalization, and common-subexpression analysis. It is an IR vocabulary,
not a monolithic backend abstraction. Third-party operation families remain
possible through the extension carrier.

Compiler semantics and runtime execution are separate contracts. A pure
semantic schema or catalog owns metadata inference, effects, aliasing, and
validation. It must not call an execution provider. Runtime execution is
expressed through small operation-family traits such as
`DotGeneralRuntime`, `ReductionRuntime`, and extension-owned linalg traits.
An engine implements only the operation-family traits it supports.

## Two Trait Layers

### Runtime operation-family traits

Runtime operation-family traits consume validated semantic requests and
produce prepared or executable operations. They do not define compiler shape
inference and do not mention ambient Rayon, process-global BLAS state, or
unresolved device selection.

Illustrative shapes are:

```rust
pub trait DotGeneralRuntime {
    fn dot_general(
        &self,
        request: &PreparedDotGeneral<'_>,
    ) -> Result<ExecutionValue>;
}

pub trait LayoutRuntime {
    fn materialize_layout(
        &self,
        request: &PreparedLayoutTransform<'_>,
    ) -> Result<ExecutionValue>;
}
```

The names and exact signatures are not fixed by this WIP document. The
contract is that validation and semantic preparation happen once at a public or
planning boundary, and engines receive validated borrowed requests.

CPU and CUDA/WebGPU engines implement only the runtime families they support:

- a CPU engine executes immediately or schedules CPU work;
- a CUDA/WebGPU engine enqueues device work;
- an XLA engine consumes `SemanticProgram` and lowers supported semantic
  regions to compiler IR.

XLA does not implement CPU or GPU provider traits and does not receive CPU
thread-pool or CUDA-stream contexts.

### Backend-specific provider traits

Provider traits are narrower algorithm families inside an eager backend. The
initial likely families are:

- `GemmProvider`;
- `LayoutTransformProvider`;
- `ReductionProvider`;
- `IndexingProvider`;
- `ElementwiseProvider` where replacement is useful;
- extension-owned decomposition or solver provider traits;
- `TransferProvider` and `CollectiveProvider` for multi-device execution.

Providers may be implemented by faer, general BLAS/LAPACK, TBLIS, CubeCL,
cuTENSOR, or user code. A provider does not select a resource domain, create a
thread pool, acquire an arbiter permit, choose a stream, or perform an implicit
fallback.

## Provider Bundles and Dispatch

Standard provider families are stored as direct trait-object fields:

```rust
pub struct CpuProviderBundle {
    gemm: Option<Arc<dyn CpuGemmProvider>>,
    layout: Option<Arc<dyn CpuLayoutTransformProvider>>,
    reduction: Option<Arc<dyn CpuReductionProvider>>,
    indexing: Option<Arc<dyn CpuIndexingProvider>>,
    extensions: ExtensionProviderRegistry,
}
```

This is preferred over a `HashMap` for built-in hot-path dispatch. A family may
be delegated to another provider explicitly. For example, a custom GEMM
provider may override `dot_general` while an engine uses the default provider
for layout transforms and reductions. Delegation is composition, not an
implicit fallback performed after an unsupported error.

Provider bundle granularity is per operation family rather than one trait per
operation or one trait for every operation in the system. This keeps runtime
configuration manageable while avoiding a new monolithic `TensorBackend`.

Provider selection is a build or prepare-time action. Steady-state execution
uses a resolved trait-object field or slot. Missing optional families are
reported during preparation; the execution loop never branches on whether a
family exists.

## Prepared Graphs and Common Execution

`PreparedGraph` contains a common `ScheduledGraph` plus runtime-binding
metadata. The schedule is shared by CPU, GPU, extension, transfer, and
multi-device execution. Its nodes are conceptually:

```rust
pub enum ScheduledNode {
    Host(PreparedHostOperation),
    Engine {
        engine: EngineId,
        operation: PreparedOperation,
        resources: ResourceRequirements,
    },
    Transfer(PreparedTransfer),
    Barrier(PreparedBarrier),
}
```

The schedule also records value slots, the dependency DAG, buffer lifetimes,
output bindings, and event dependencies. A `PreparedOperation` has already
resolved its provider or extension adapter. There is no family-id lookup or
capability discovery in the execution loop.

The common `GraphExecutor` is owned by `Runtime`; it is not generic over one
`TensorBackend`. This permits a single graph to contain CPU work, GPU work,
explicit transfers, barriers, and work on more than one device. An opaque
`dyn ExecutableGraph` was considered but rejected because it would hide the
common scheduling and lifetime model. A generic `GraphExecutor<E>` was also
rejected because it makes heterogeneous and multi-GPU execution awkward.

### Public execution path

The explicit prepared path is conceptually:

```rust
let semantic = GraphCompiler::compile(&traced_graph)?;
let prepared = runtime.prepare_for(
    &semantic,
    &input_signature,
    &prepare_options,
)?;
let outputs = runtime.execute_prepared(&prepared, inputs)?;
```

The ordinary convenience path is:

```rust
let outputs = runtime.run(&semantic, inputs)?;
```

`Runtime::run` validates input metadata and shape guards, derives the
`InputSignature`, looks up or creates a prepared specialization, acquires
resource leases, and executes the common schedule. `execute_prepared` is the
fast explicit path and does not silently re-prepare a foreign or stale plan.

Each `PreparedGraph` is bound to:

- one `RuntimeId`;
- one runtime configuration epoch;
- one semantic-program fingerprint;
- one concrete or polymorphic input specialization.

Using it with a different runtime returns `RuntimeMismatch`. Changing provider,
extension, topology, or resource configuration advances the epoch and makes
the plan stale. The convenience `run` path may create a new cached plan; the
explicit prepared path returns `StalePreparedGraph`.

## Dynamic Shapes and Specialization

`SemanticProgram` retains symbolic dimensions and guards. Before any provider
work or resource acquisition, the runtime validates input metadata, evaluates
shape guards, and selects a prepared specialization.

When a provider or lowering can remain polymorphic, it may prepare one
polymorphic plan. Otherwise, preparation is keyed by `InputSignature`. This is
required for operations such as N-ary einsum, where contraction-path selection
depends on concrete operand dimensions.

Planning is not performed inside the steady-state executor. On the first call
for a signature, `Runtime::run` crosses a planning boundary before acquiring
execution resources. Later calls reuse the bounded specialization cache. An
explicit `prepare_for` API lets latency-sensitive users prepare a known
signature in advance.

The compiler and runtime caches have separate ownership and keys:

```text
semantic graph + compiler options
    -> SemanticProgram

semantic fingerprint + RuntimeId/config epoch + placement
    + InputSignature + prepare options/provider version
    -> PreparedGraph or XlaExecutable
```

Every long-lived cache follows the repository cache contract: bounded default,
entry and retained-byte statistics, configuration, clear APIs, and aggregate
runtime introspection. Cache keys use compact structural fingerprints plus
exact collision checks rather than formatted programs. The semantic cache is
owned by an explicit compiler service or `GraphCompiler`; specialization and
prepared-plan caches are owned by `Runtime`.

## Request and Dispatch Cost Contract

Requests are borrowed, validated views over existing metadata. Their creation
must not allocate in steady state. Variable-rank fields should use one
repository-wide representation at the boundary rather than repeatedly convert
between `Vec`, `SmallVec`, and slices.

The prototype for issue #1432 is stored on branch
`codex/issue-1432-provider-overhead` at commit `1b6223ce`. Its release-mode
microbenchmarks measured approximately:

| Dispatch path | Measured cost |
| --- | ---: |
| direct dynamic dispatch | 5.69 ns |
| string-keyed hash map | 31.47 ns |
| pre-resolved extension slot | 6.31 ns |

The prototype also found a borrowed request near 7 ns, while constructing a
per-call `SmallVec` request was worse. These results are evidence for direct
trait-object fields and resolved slots, not a permanent performance guarantee.
Future implementation must benchmark representative ranks and request shapes,
including `Vec` versus `SmallVec`, validation cost, and cache-key hashing.

## Extensions

An extension crate owns four pieces:

1. its semantic payload, schema, and optional core lowering;
2. backend-specific provider traits for the backends it supports;
3. its prepared-operation representation;
4. a typed runtime adapter that connects its operation-family traits to core
   extension dispatch.

The core runtime stores an erased `ExtensionRuntimeAdapter`, but extension
authors and engines work with typed traits. Registration uses the stable
`family_id` contract from `docs/spec/extension-op.md` during build or prepare.
The result is a resolved `ExtensionSlot`; execution does not repeat a string
hash lookup.

```text
family_id registration -> validation -> ExtensionSlot resolution
                                           |
                                           v
                                dyn adapter dispatch
```

This supports both first-party and third-party operation crates. Linalg remains
owned by its operation crate: upstream tensor/runtime crates need not know SVD,
QR, eigensolvers, or future decompositions. A `CpuLinalgRuntime` may bundle
smaller extension-owned traits, while a user may implement only one small trait
and explicitly delegate the rest.

Extension payload identity remains semantic. Provider handles, caches, streams,
thread pools, and mutable runtime state do not participate in `ExtensionOp`
hashing or equality.

### Generic extension lowering and specialization

Extension lowering is not an einsum-specific runtime hook. Every extension may
optionally express an equivalent graph of core semantic operations:

```rust
pub enum ExtensionLoweringOutcome {
    Lowered(Vec<SemanticValue>),
    NeedsSpecialization(SpecializationRequirements),
    Unsupported,
}
```

Conceptually, the semantic and native sides are separate object-safe
interfaces:

```rust
pub trait ExtensionOp {
    fn lower_to_core(
        &self,
        context: &mut ExtensionLoweringContext<'_>,
    ) -> Result<ExtensionLoweringOutcome>;
}

pub trait ExtensionEngine {
    fn family_id(&self) -> ExtensionFamilyId;

    fn prepare(
        &self,
        operation: &dyn ExtensionOp,
        context: &ExtensionPrepareContext<'_>,
    ) -> Result<PrepareCapability<PreparedOperation>>;
}
```

`SpecializationRequirements` may request concrete shapes, dtype, layout,
placement, runtime topology, or resource limits. It must not request arbitrary
input tensor values. Data-dependent algorithms remain dynamic execution
operations rather than compile-time inspection of user data.

The current `lower_to_standard_ops` result uses `Ok(None)` for both temporary
metadata insufficiency and permanent lack of a lowering. Migration replaces
that ambiguity with `NeedsSpecialization` versus `Unsupported`.

Lowering is invoked by runtime preparation, because the decision depends on
available engines and may depend on a concrete input signature. The pure
`GraphCompiler` preserves the extension payload in `SemanticProgram`. Runtime
preparation first checks native capability, then attempts core lowering, then
uses an explicitly configured fallback engine if one exists.

A successful lowering preserves the extension's declared metadata, effects,
alias behavior, output arity, and node provenance. The newly emitted core
region re-enters target-independent optimization and capability resolution
before final placement and scheduling.

Core lowering may emit only `CoreSemanticOp`. This guarantees termination and
avoids cyclic extension-to-extension legalization. A native extension engine
may depend explicitly on another extension provider, but that dependency is
part of runtime configuration rather than an implicit lowering chain.

The same contract supports shape-dependent einsum paths, FFT radix plans,
shape- and dtype-dependent linalg algorithms, sparse formats, specialized
permutation kernels, and topology-dependent collectives. Runtime code dispatches
through the extension interface and does not downcast to an einsum type.

### Extension capability resolution

An extension engine reports capability separately from failure:

```rust
pub enum PrepareCapability<T> {
    Prepared(T),
    NeedsSpecialization(SpecializationRequirements),
    Unsupported(UnsupportedReason),
}
```

Only `Unsupported` permits trying another provider or lowering. Invalid
payloads, internal planning failures, and broken workspace plans are errors and
must not be converted into capability misses.

### Einsum planning policy

Einsum illustrates the separation between semantic policy, runtime policy, and
a resolved plan:

- the extension payload stores an operation-local policy such as automatic
  search options, left-to-right execution, nested notation, a JAX-compatible
  path, or an explicitly supplied path;
- the runtime stores inherited defaults, provider versions, global planning
  budgets, hard workspace limits, and resource policy;
- the prepared operation stores the concrete-shape-dependent contraction tree,
  intermediate layouts, selected engines, and workspace requirements.

The current `EinsumOptimize::Tree` is a concrete-shape-dependent input. When
accepted, its fixed path may be preserved as semantic user intent, but the
resolved `ContractionTree` itself does not live in portable payload identity.
The current `static_tree` execution hint therefore migrates to the prepared
operation.

Automatic search options such as TreeSA temperatures, trial and iteration
counts, and time/space/read-write score weights are semantic operation-local
policy when explicitly supplied. A runtime default is represented as
`InheritRuntimeDefault`, not copied into every payload. A hard byte limit is a
runtime resource constraint, not merely a space-complexity score; infeasible
candidate paths are rejected during preparation.

## Layout and Permutation

Logical view-only permutation remains a core metadata operation. It does not
call a provider and must preserve the repository's explicit view semantics.

Actual data movement is replaceable through `LayoutTransformProvider`, for
example:

- materializing a non-contiguous view;
- packing for GEMM or LAPACK;
- making a contiguous copy;
- device-native transpose or permutation kernels.

A provider may change how materialization is performed, but not whether a
public operation is a view or a copy. Hidden layout conversion remains
forbidden.

## Runtime Ownership

The primary runtime is explicit and user-owned:

```rust
pub struct Runtime {
    engines: EngineRegistry,
    devices: DeviceRegistry,
    extensions: ExtensionRuntimeRegistry,
    resources: ResourceArbiter,
    plans: PreparedPlanCache,
    executor: GraphExecutor,
}
```

This is illustrative, not a frozen public struct. There is no process-global or
thread-local singleton in the core contract. A convenience default runtime may
exist only as a high-level facade.

`EagerTensor` may hold `Arc<Runtime>`. The physical tensor/storage object does
not own its engine, avoiding runtime-storage reference cycles. Physical tensor
state includes storage, placement, and any pending completion event.

Mixing tensors from different runtime or allocation domains is an error. The
runtime does not silently transfer or import them. Cross-runtime movement uses
an explicit `import` or `transfer` API with visible cost and error behavior.

Runtime configuration has an epoch. Changes to providers, extensions,
topology, resource domains, or planning defaults advance the epoch and
invalidate prepared-plan reuse. Runtime-owned caches and resource pools expose
individual and aggregate bounds, clear operations, entry counts, and retained
byte estimates.

## CPU Resource Model

### Resource domains

`CpuEngine` owns resource domains. Each resolved domain contains at least:

- an OS NUMA-node identity when applicable;
- a concrete `CpuSet`;
- an executor or thread pool;
- a node-local buffer pool and scratch owner;
- cache ownership;
- a thread budget;
- admission-control state.

The engine selects a domain and passes a per-execution context:

```rust
pub struct CpuExecutionContext<'a> {
    domain: &'a CpuResourceDomain,
    lease: &'a CpuResourceLease,
    thread_budget: usize,
}
```

Providers consume this context. They do not choose a NUMA node, query ambient
Rayon state, or acquire a second resource permit. The outer execution acquires
one lease; nested operations reuse it. This prevents nested oversubscription
and makes the thread budget a single engine-owned policy.

### Managed mode

In `Managed` mode, tenferro:

- discovers and validates topology;
- builds and pins executor workers;
- owns admission control;
- creates node-local buffer and scratch pools;
- applies first-touch allocation within the selected executor where possible;
- validates and enforces the selected thread budget for tenferro-controlled
  kernels.

### ExternalManaged mode

In `ExternalManaged` mode, the user supplies node domains, executors or pools,
CPU sets, and thread budgets. Tenferro validates static configuration such as
empty CPU sets, duplicate domain identities, and overlapping declared CPU sets.

The external owner is responsible for:

- admission and fairness among submitted work;
- maintaining worker affinity;
- coordinating pools used by other libraries;
- avoiding oversubscription;
- external BLAS/OpenMP global state;
- executor lifetime and shutdown.

Tenferro must not reconstruct or silently replace an external executor.

### BLAS and faer behavior

Faer and tenferro-native parallel kernels can use the selected executor when
their APIs permit it. General BLAS/LAPACK providers often expose only a global
or provider-specific thread count, not a per-call executor. Such a provider
receives the execution thread budget and provider exclusivity policy, but it
does not receive a Rayon pool as if BLAS could execute on that pool.

If a BLAS library cannot provide the requested isolation, the engine returns a
typed unsupported or configuration error for explicit placement modes. It does
not claim NUMA placement that it cannot enforce.

### Memory locality

The NUMA model includes memory locality, not just worker affinity. Buffers and
scratch should be allocated or first-touched within the chosen domain executor
and returned to a node-local pool. A buffer records its allocation domain so it
cannot be reused silently by an incompatible domain.

## GPU Resource Model

Tenferro-owned CUDA and WebGPU backends use the same engine/provider split.
The runtime owns a `DeviceRegistry`; each device has a `GpuDeviceRuntime` with:

- a provider bundle;
- streams or queues;
- allocator and scratch ownership;
- vendor handles and plan caches;
- dependency tracking;
- admission-control state.

The current CUDA runtime, cuTENSOR handle, and extension cache responsibilities
would migrate into the per-device runtime. A GPU provider receives a resolved
`GpuExecutionContext` containing the selected device, stream or queue, scratch
access, and dependency events. It does not create or globally select a stream.

## Asynchronous Execution

The public eager API remains synchronous-looking, while GPU execution is
internally asynchronous:

- a GPU provider enqueues work and returns an `ExecutionEvent`;
- the engine attaches dependencies to subsequent work;
- a CPU event is immediately ready;
- an XLA/PJRT event can wrap the corresponding future;
- providers do not call a global synchronize after each operation.

Synchronization occurs only at an observable boundary:

- host read or download;
- explicit synchronize;
- an external provider whose contract is synchronous;
- safe reuse or destruction of a resource;
- a host-visible error flag that requires completion.

In externally managed GPU execution, the user owns stream lifetime and
interoperation constraints, while tenferro still tracks ordering for work it
submits.

## Multi-GPU Design Boundary

Multi-GPU support is included in the resource model now, but the initial goal
is task parallelism: independent contractions, batches, or graph jobs can be
placed on different GPUs.

The engine owns:

- a `DeviceMesh` describing logical device organization;
- transfer providers;
- collective providers;
- canonical multi-device lease acquisition ordered by `DeviceId` to avoid
  deadlock;
- scheduling policy for independent work.

An ordinary physical `Tensor` remains single-device in the initial refactor.
No initial public `DistributedTensor` is introduced.

### Future sharding model

The design reserves a future logical-tensor placement layer with:

- `Shard`;
- `Replicate`;
- `Partial`;
- explicit `Reshard`, `AllReduce`, `AllGather`, and related IR operations.

This follows the long-term direction of a unified logical tensor carrying
sharding, similar in shape to JAX's `Array + Sharding`, rather than duplicating
the entire Rust tensor API in a separate distributed tensor type. PyTorch's
`DTensor + DeviceMesh + Placement` remains a useful comparison, especially for
explicit placement vocabulary.

Automatic sharding, multi-host scheduling, distributed decomposition kernels,
and collective-library implementation are deferred. Future resharding and
collectives must remain observable; the engine must not hide communication
inside an unrelated tensor operation.

## Capability and Fallback Contract

Capability resolution happens before steady-state execution. A prepared
operation resolves to a provider, extension slot, or lowering implementation.
Missing capabilities return a typed prepare error. `PreparedGraph` cannot
contain an unresolved operation.

The default resolution order is:

1. a native engine implementation;
2. semantic lowering to core operations;
3. an explicitly configured fallback engine;
4. `PrepareError::UnsupportedOperation` or
   `PrepareError::UnsupportedExtension`.

Fallback is permitted only through explicit runtime configuration, a decorator,
or a composite provider. A provider must not catch an arbitrary error and
silently move work to CPU, another device, a reference implementation, or a
full decomposition. Only an explicit capability result of `Unsupported`
continues resolution. Once execution has begun, a kernel or device failure does
not trigger implicit retry on another engine because earlier effectful or
asynchronous work may already have run.

## Error Stages

Errors are separated by lifecycle stage:

1. **Semantic compile:** invalid graph, contradictory dtype or shape semantics,
   malformed extension payload, or semantic-schema failure.
2. **Prepare:** unsupported semantic capability, unresolved provider or
   extension family, specialization failure, infeasible placement, or a
   resource requirement exceeding known runtime capacity.
3. **Enqueue:** resource-lease failure, allocation failure, immediate launch
   failure, stopped runtime, foreign runtime, or stale prepared plan.
4. **Completion:** deferred device, kernel, transfer, collective, or
   asynchronous provider failure.

Public errors preserve these distinctions without exposing unstable provider
internals as public API and preserve typed source chains. Input metadata and
shape guards are validated before specialization work, provider work, or
resource acquisition. Resource leases are acquired only after preparation has
produced a feasible schedule.

Runtime construction errors such as overlapping NUMA domains, malformed
external executors, or incompatible provider configuration remain a separate
configuration boundary before this program lifecycle.

## Observability

Compiler and runtime decisions must be inspectable without exposing provider
implementation objects as stable public API. The intended capabilities are
conceptually:

```rust
semantic.explain();
prepared.explain();
runtime.plan_cache_stats();
runtime.execution_metrics();
```

A prepared-plan explanation includes semantic node provenance, lowering
provenance, specialization signature, selected engine or provider, placement,
resource requirements, buffer lifetime, transfers, barriers, and any explicit
fallback decision. Metrics distinguish semantic compilation, specialization,
cache lookup, planning, resource wait, enqueue, and completion so planning cost
is not accidentally reported as kernel cost.

Explanations and diagnostics are not cache keys. They may format human-readable
strings on demand, while steady-state lookup uses compact structural keys.

## Testing and Performance Evidence

The implementation must provide a shared semantic conformance suite across at
least:

- CPU with faer providers;
- CPU with general BLAS/LAPACK providers where available;
- CUDA/CubeCL and custom CUDA providers;
- WebGPU providers;
- XLA lowering plus a reference execution path.

Required focused tests include:

- CPU, GPU, and XLA preparation observe the same semantic fingerprint for the
  same graph;
- `SemanticProgram` contains no runtime, engine, thread-pool, stream, allocator,
  or lease handle;
- no `PreparedGraph` contains an unresolved operation;
- provider replacement preserves semantic results;
- request construction and resolved dispatch allocate nothing in steady state;
- extension execution performs no string lookup after prepare;
- extension native execution, core lowering, and explicit fallback obey the
  documented priority;
- generic extension specialization works without runtime knowledge of the
  extension's concrete Rust type;
- different concrete einsum shapes select distinct cached plans when their
  resolved contraction plans differ;
- a runtime epoch change invalidates old prepared plans;
- CPU and GPU graphs use the same dependency, buffer lifetime, transfer, and
  barrier rules;
- input and shape-guard failure occurs before resource acquisition;
- execution failure never causes an implicit cross-engine retry;
- eager and graph paths apply the same provider-selection policy;
- nested CPU operations reuse the outer lease;
- NUMA buffers remain associated with their allocation domain;
- `ExternalManaged` does not reconstruct a supplied executor;
- GPU dependency tracking avoids unnecessary global synchronization;
- independent multi-GPU work can enqueue concurrently;
- foreign runtime tensors do not transfer implicitly;
- unsupported provider paths return explicit errors rather than fallback.

Performance work must measure representative shapes, ranks, batch counts,
thread counts, NUMA placements, and device counts. Microbenchmarks must keep
validation, request construction, dispatch, provider call, and kernel work
separable.

## Migration Strategy

This is an umbrella architecture. Migration is incremental, keeps the current
backend path working behind adapters, and is decomposed into independently
accepted child issues:

1. Introduce `SemanticProgram` and make `GraphCompiler` produce it. Preserve
   current `GraphProgram` and `ExecProgram` behavior through an adapter without
   changing execution.
2. Introduce runtime plan compilation, `Engine`, small operation-family traits,
   `PreparedOperation`, specialization requirements, and bounded plan caches.
3. Introduce common `ScheduledGraph` and runtime-owned `GraphExecutor`. Port
   CPU execution first while retaining a compatibility adapter for
   `GraphExecutor<B>`.
4. Lift NUMA topology, arbiter, executor, buffer, and cache ownership into CPU
   resource domains. Add `Managed` and `ExternalManaged` validation.
5. Migrate extension capability resolution and core lowering. Use N-ary einsum
   to validate shape specialization, operation-local planning policy, and
   resolved-slot execution; then migrate FFT, linalg, sparse, and permutation
   families.
6. Split CUDA/WebGPU device runtime resources from provider algorithms, port
   GPU execution to the common scheduler, and introduce dependency-aware
   `ExecutionEvent` resource reuse.
7. Make XLA consume `SemanticProgram` directly, then retire
   `GraphProgramLoweringView` and the executor-shaped portable artifact.
8. Add multi-GPU task scheduling for independent work.
9. Add logical sharding, collectives, and resharding only if accepted by a
   later design issue.

Each phase must be independently reviewable. Normative specs and online
parallelism documentation are updated as the corresponding behavior lands.
No child implementation issue is authorized merely by this WIP umbrella
design; maintainers must accept its scope before implementation begins.

## Documentation Requirements

The rendered documentation needs a dedicated parallelism section that states,
for each backend and provider family:

- whether parallelism is outer, inner, or both;
- which object owns the thread pool, stream, or queue;
- whether thread counts are per-runtime, per-domain, per-provider, or global;
- what `Managed` and `ExternalManaged` guarantee;
- how faer differs from general BLAS/LAPACK;
- when synchronization occurs;
- what single-device and multi-device placement mean;
- which behavior is unsupported rather than silently degraded.

SVD, QR, and other batched linalg documentation must explicitly describe outer
batch parallelism and inner decomposition parallelism. Users who need a
different policy select or implement providers; tenferro does not bury that
policy in operation hooks.

## Alternatives Considered

### Treat the current `ExecProgram` as the portable graph artifact

This minimizes type migration but preserves executor categories, slots,
lifetimes, and dispatch decisions in the supposedly backend-neutral program.
Rejected in favor of a semantic artifact followed by runtime plan compilation.

### Use `GraphExecutor<E>` for each engine type

This retains static engine typing but makes CPU/GPU graphs, transfers,
multi-device scheduling, and extension engines difficult to combine. Rejected
in favor of a runtime-owned executor over a common `ScheduledGraph`.

### Return only an opaque `dyn ExecutableGraph`

This gives every target complete freedom but hides common dependency, transfer,
buffer-lifetime, resource, and observability contracts. Rejected for CPU/GPU
execution. XLA may still produce an opaque whole-region executable after
consuming the same `SemanticProgram`.

### Keep one broad `TensorBackend`

This preserves a simple call site but couples unrelated operation families,
resources, and extensions. It also requires upstream crates to know new linalg
families. Rejected as the target architecture; retained temporarily behind
adapters during migration.

### Add per-operation hooks to `CpuBackend`

Hooks are easy to add but make ownership, capability discovery, and fallback
unclear. They do not generalize cleanly to extension crates or XLA lowering.
Rejected in favor of semantic and provider traits.

### Use a string-keyed provider map on every call

This makes third-party registration uniform but adds repeated hashing and
downcasting to the hot path. Rejected for steady-state execution. String family
identity remains at registration and prepare boundaries, followed by slot
resolution.

### Introduce `DistributedTensor` immediately

This provides an explicit distributed type but would duplicate Rust APIs,
operation traits, and AD integration before the execution-resource model is
stable. Deferred. Initial multi-GPU support schedules independent single-device
tensors; future logical sharding is designed separately.

## References

- [Provider-overhead prototype and measurements, issue #1432](https://github.com/tensor4all/tenferro-rs/issues/1432#issuecomment-5017993877)
- [Prototype branch at commit `1b6223ce`](https://github.com/tensor4all/tenferro-rs/commit/1b6223cee988af8e98a4a79d05d977024482573f)
- [JAX array migration](https://docs.jax.dev/en/latest/jax_array_migration.html)
- [JAX explicit sharding](https://docs.jax.dev/en/latest/notebooks/explicit-sharding.html)
- [PyTorch `DTensor`](https://docs.pytorch.org/docs/stable/distributed.tensor.html)
- [PyTorch tensor parallel APIs](https://docs.pytorch.org/docs/stable/distributed.tensor.parallel.html)

## Decisions Required Before Child Implementation Planning

The artifact boundaries, generic extension model, specialization ownership,
fallback behavior, common CPU/GPU schedule, and staged migration direction are
agreed in this WIP design. An accepted umbrella issue must still assign the
following API-level details to child design issues before implementation:

1. exact crate ownership, public visibility, and names of `SemanticProgram`,
   engine traits, and prepared types;
2. exact built-in provider-family boundaries and delegation APIs;
3. serialized versus process-local portability requirements for extension
   payloads and semantic programs;
4. runtime identity and configuration-epoch representation;
5. executor abstraction required by `ExternalManaged`;
6. `ExecutionEvent` object-safety, cancellation, and error propagation;
7. exact structural key and equality strategy for semantic and specialization
   caches;
8. which migration phases become separately accepted child issues.

Until those details and child scopes are accepted, this branch is design
evidence only and must not be treated as authorization for a feature
implementation PR.
