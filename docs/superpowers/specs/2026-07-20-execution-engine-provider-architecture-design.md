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

1. Let CPU eager, CUDA/WebGPU eager, tenferro's native graph execution, XLA,
   and third-party runtimes consume the same semantic operation contract.
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

The graph path has two compilation stages separated by a backend- and
runtime-neutral semantic artifact. Initial portability is deliberately
process-local; serialization and wire compatibility are not part of this
design.

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
    |
    v
PreparedGraph                   runtime-bound scheduled artifact
    |
    v
runtime-owned GraphExecutor
```

Eager execution does not run a whole graph compiler for each operation. It
constructs the same borrowed semantic request and uses the same validation,
capability, provider, placement, and resource contracts through a
single-operation fast path. Sharing contracts does not require eager execution
to construct graph-only artifacts or pay graph-level orchestration costs.

The current `GraphProgram`, `ExecProgram`, `GraphExecutor<B>`, and
`GraphProgramLoweringView` are migration inputs, not the final abstraction
boundaries. In particular, an executor-shaped instruction stream is not the
portable compiler artifact.

XLA follows this same path. A whole-program XLA compilation is represented as
one prepared subgraph operation in a one-node `ScheduledGraph`; the internal
PJRT executable is not a second public execution pipeline.

The dependency direction is deliberate:

- tenferro core defines or depends on semantic operation contracts;
- an engine implements those contracts;
- a backend-specific engine delegates algorithms to providers;
- providers receive a per-execution context but do not own scheduling
  resources;
- extension crates add semantic and provider traits without requiring the core
  runtime to enumerate their operation families.

### Crate ownership

A new public `tenferro-program` crate owns `SemanticProgram`,
`CoreSemanticOp`, `ExtensionOp`, semantic metadata, shape guards, effects,
alias declarations, and process-local structural fingerprints.
`tenferro-runtime` owns `TraceContext`, `GraphCompiler`, runtime and engine
traits, `PreparedGraph`, `ScheduledGraph`, `GraphExecutor`, `ResourceArbiter`,
and prepared-plan caches. XLA depends on `tenferro-program` and integrates with
`tenferro-runtime` as an engine. Operation crates depend on
`tenferro-program` for semantic payloads and optionally on `tenferro-runtime`
for execution adapters.

`GraphCompiler` remains in `tenferro-runtime` initially because its compiler
service, trace integration, and cache are closely tied to runtime-facing
workflows. This ownership may be split later without changing the program
artifact.

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

`SemanticProgram` has private fields and read-only views and iterators. A
validation-preserving builder is available to low-level frontends, but callers
cannot mutate an already frozen program. `CoreSemanticOp` is a public,
`#[non_exhaustive]` enum. The initial program contains one acyclic
`SemanticRegion`. The container reserves future nested regions for structured
control flow, but the first implementation has no `if`, `while`, block
arguments, yields, shape joins, or loop AD.

Terminology is fixed as follows:

- `SemanticRegion` is the future control-flow nesting unit;
- `SemanticSubgraph` is a fusion or multi-node compilation candidate;
- `SubgraphCompiler` prepares a multi-node candidate for one engine;
- `ScheduledGraph` is the executable dependency DAG.

### Trace and compiler responsibilities

`TraceContext` owns mutable graph construction, traced value identities,
captures and defaults, trace-time parsing caches, and metadata or constraint
scopes. It produces an immutable `TracedGraph`. A pure `GraphCompiler`
consumes that graph and emits `CompiledGraph { program, bindings }`, where
`program` is an `Arc<SemanticProgram>` and `bindings` is process-local
`ProgramBindings`.

The program contains input schemas and only small backend-neutral
`ConstantLiteral` values. Actual captured or default tensors, and all large
constants, remain in `ProgramBindings`. Constantizing a large tensor is an
explicit operation governed by a size policy. A tensor from another runtime is
never imported implicitly as a default. Cross-`TraceContext` tensor use also
requires an explicit import; tracing does not merge graphs implicitly.

Extension tracing APIs target `TraceContext`. Existing APIs such as
`GraphCompilerEinsumExt` become compatibility adapters rather than permanent
compiler extension points.

### Verified semantic transforms and AD

The fixed compiler pipeline is deterministic and target-independent:

1. trace normalization;
2. metadata, constraints, and shape guards;
3. effects and aliases;
4. canonicalization plus DCE/CSE and view composition;
5. extension output pruning;
6. validation, freeze, and fingerprinting.

Custom compiler work uses an explicit `SemanticTransform`: a read-only input
and a validated builder output. Transforms cannot mutate a program, inspect
runtime providers or input values, or use a process-global pass registry.
Ordered custom transforms participate in a compiler cache key and preserve
node provenance. Extension callbacks admitted inside the fixed pipeline are
limited to pure semantic operations such as output pruning.

Automatic differentiation is a separate
`SemanticProgram -> SemanticProgram` transform in `tenferro-ad`, before
runtime preparation. `tenferro-program` has no AD dependency. Extension-owned
rules are installed explicitly in an `AdContext`; there is no global rule
inventory. AD tries the extension rule before runtime lowering. Differentiating
through a lowering is an explicit fallback policy, effectful operations require
a rule, and in-place semantics require functionalization.

The current recorded eager VJP maps to this pipeline as
`record -> TracedGraph -> SemanticProgram -> AD transform -> Runtime::prepare_for`.
The eager recorder remains a frontend; derivative execution does not preserve
a separate legacy graph-runtime path.

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

### Effects and aliases

An effect is observable state beyond returned tensor values, such as stateful
random-number generation, a host callback, access to an external mutable
buffer, or use of a collective communicator. Pure operations declare an empty
effect set. Effects use typed resource identities with read and write access,
so operations touching independent resources may still execute concurrently.
An extension may not silently default to pure; it must declare its effects.

Aliasing is a separate semantic contract with `Fresh`, `ViewOf`, `MustAlias`,
and `ExternalAlias` forms. Physical reuse of a dead buffer is permitted even
for semantically `Fresh` outputs when the buffer planner proves it safe.
Extension lowering must preserve output arity, metadata, effects, aliases, and
provenance, and validation rejects any mismatch.

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

The core engine capability bundle has typed optional slots for:

- `ElementwiseRuntime` for one semantic elementwise operation;
- `ReductionRuntime` for reductions and arg reductions;
- `IndexingRuntime` for gather, scatter, dynamic slice/update, concatenate,
  padding, and indexing copies;
- `DotGeneralRuntime` for semantic generalized contraction;
- `LayoutRuntime` for actual materialization, packing, and copies;
- `SubgraphCompiler` for optional multi-node compilation or fusion.

Metadata-only reshape, transpose, broadcast, and view slicing do not call a
provider. Allocation and scheduling are runtime resources, not capabilities.
Transfers use a separate registry keyed by source and destination storage
class, and collectives use a separate registry. Neither may hide a host
transfer.

Providers may be implemented by faer, general BLAS/LAPACK, TBLIS, CubeCL,
cuTENSOR, or user code. A provider does not select a resource domain, create a
thread pool, acquire an arbiter permit, choose a stream, or perform an implicit
fallback.

## Provider Bundles and Dispatch

Standard core families are stored as direct typed trait-object fields:

```rust
pub struct CpuProviderBundle {
    elementwise: Option<Arc<dyn ElementwiseRuntime>>,
    reduction: Option<Arc<dyn ReductionRuntime>>,
    indexing: Option<Arc<dyn IndexingRuntime>>,
    dot_general: Option<Arc<dyn DotGeneralRuntime>>,
    layout: Option<Arc<dyn LayoutRuntime>>,
    subgraph: Option<Arc<dyn SubgraphCompiler>>,
}
```

This is preferred over a `HashMap` or `TypeId`/`Any` query for built-in hot-path
dispatch. A family may delegate explicitly to another implementation.
Delegation is composition, not an implicit fallback after an arbitrary error.

Provider bundle granularity is per operation family rather than one trait per
operation or one trait for every operation in the system. This keeps runtime
configuration manageable while avoiding a new monolithic `TensorBackend`.

Provider selection is a build or prepare-time action. Steady-state execution
uses a resolved trait-object field or slot. Missing optional families are
reported during preparation; the execution loop never branches on whether a
family exists.

### CPU contraction and linalg composition

`CpuGemmProvider` exposes GEMM, strided-batched GEMM, and grouped GEMM
primitives; it does not implement semantic `dot_general` or own batch
scheduling. `DotGeneralRuntime` is an engine-owned composite that may use a
`CpuGeneralContractionProvider` such as TBLIS for a direct binary contraction,
or a `CpuLayoutTransformProvider` plus `CpuGemmProvider` for decomposition.
N-ary einsum first resolves a contraction path into binary `dot_general`
operations. A user may replace the complete composite, the
general-contraction provider, or only GEMM/layout providers.

The composite chooses the batch parallelization level before invoking the
provider. Under `ParallelMode::Outer`, it splits a grouped request into jobs,
fans them out through the domain executor, and calls the provider with a
sequential single-job request. Under `ParallelMode::Inner`, the outer loop is
sequential and each provider call may use `CpuDomainExecutor::install` for
inner kernel parallelism. A grouped provider entry point is therefore a
provider-native grouped primitive, not permission to create its own outer task
fan-out. Replacing the provider cannot change the selected parallel level.

Linalg remains extension-owned and uses a family-level capability bundle with
optional SVD, QR, eigen, Cholesky, LU, and solve slots. This avoids both one
monolithic linalg trait and one public trait per operation. A values-only
request may not silently invoke a full decomposition; such behavior requires
an explicit decomposition adapter. Batch scheduling belongs to the engine,
while a prepared provider operation describes one matrix algorithm and its
workspace. All providers obey the engine-selected `ParallelMode`.

## Prepared Graphs and Common Execution

`PreparedGraph` contains a common `ScheduledGraph` plus runtime-binding
metadata. The schedule is shared by CPU, GPU, extension, transfer, and
multi-device execution. Its nodes are conceptually:

```rust
pub enum ScheduledNode {
    Operation {
        operation: Arc<dyn PreparedOperation>,
        resources: ResourceRequirements,
    },
    Transfer(PreparedTransfer),
    Barrier(PreparedBarrier),
}
```

The schedule also records value slots, the dependency DAG, buffer lifetimes,
output bindings, and event dependencies. Each `PreparedOperation` is
self-contained: it retains the resolved engine/provider and algorithm plan.
There is one dynamic operation dispatch and no family-id, provider-registry, or
capability lookup in the execution loop.

Provider code receives an `ErasedExecutionContext` and performs one safe
`TypeId` check/downcast to its typed context. Preparation validates the same
context identity. Inputs and outputs remain borrowed, and execution performs
no context or plan allocation. An unsafe custom vtable fast path is deferred
unless benchmarks prove the safe check material; increasing fusion is the
preferred way to amortize dispatch.

The common `GraphExecutor` is owned by `Runtime`; it is not generic over one
`TensorBackend`. This permits a single graph to contain CPU work, GPU work,
explicit transfers, barriers, and work on more than one device. An opaque
`dyn ExecutableGraph` was considered but rejected because it would hide the
common scheduling and lifetime model. A generic `GraphExecutor<E>` was also
rejected because it makes heterogeneous and multi-GPU execution awkward.

### Subgraph compilation and fusion

`SubgraphCompiler` is an optional engine capability for multi-node
compilation. `GraphCompiler` records target-independent legality facts and
candidate relationships but does not commit to a target fusion. Runtime
preparation performs legalization, asks engines for proposals, applies a
deterministic partition, prepares selected subgraphs, and falls back to
single-operation preparation for uncovered nodes.

Selection follows explicit constraints, engine preference, larger legal
subgraphs, and stable semantic node order. CPU elementwise fusion, GPU fusion,
and whole-program XLA compilation use this same mechanism. Every selected
subgraph must preserve effects, aliases, external inputs, outputs, and
provenance.

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

The ordinary synchronous convenience path is:

```rust
let outputs = runtime.run(&semantic, inputs)?;
```

`Runtime::submit` is the asynchronous primitive and returns an
`ExecutionHandle`. `ExecutionHandle::wait` reports completion errors, and
`run` is exactly `submit` plus `wait`. Pending outputs may feed more work in the
same runtime without host synchronization. Dropping a handle does not cancel
already submitted work. Eager GPU operations likewise return tensors carrying
pending completion; host reads, explicit synchronization, or export observe
completion and deferred errors.

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
polymorphic plan. Otherwise, the cache key uses a runtime-owned finite typed
projection, `SpecializationRequirements`. It may include concrete dimensions,
dtype, placement, layout class, exact strides, and alignment. It does not
include tensor values, pointers, current free memory, or scheduler load.
Topology, configuration, and hard workspace policies belong to the runtime
epoch or prepare options.

A provider may respond with `NeedsSpecialization` and strictly broader
requirements. Repeating or narrowing the same request is a
`ProviderContractError`. Since the vocabulary is finite and widening is
monotonic, preparation terminates. This is required for operations such as
N-ary einsum, where contraction-path selection depends on concrete operand
dimensions.

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
    + specialization projection + prepare options
    -> PreparedGraph
```

Every long-lived cache follows the repository cache contract: bounded default,
entry and retained-byte statistics, configuration, clear APIs, and aggregate
runtime introspection. A fixed process-local structural fingerprint is computed
once when a program freezes. Cache lookup uses it first and exact structural
equality only within a collision bucket; the bucket retains an
`Arc<SemanticProgram>`. Extension identity contributes `family_id` plus
`payload_hash`, followed by `payload_eq` on collision. The algorithm has no
wire-stability guarantee. The prepared-plan root key is the fingerprint,
`RuntimeId`, configuration epoch, and `PrepareOptionsKey`; its specialization
projection is part of the selected entry. The semantic cache is
owned by an explicit compiler service or `GraphCompiler`; specialization and
prepared-plan caches are owned by `Runtime`.

Prepared-plan creation uses key-level single-flight states: `Preparing`,
`Ready`, and `Failed`. A global cache lock is never held while planning. The
same key waits; different keys prepare concurrently. Deterministic failures may
be negative-cached within one epoch, while transient failures are not cached.
Self-recursive preparation returns `PreparationCycle`. An in-progress entry is
not evicted, and ready entries remain safe through `Arc` ownership. Metrics
include hits, misses, waits, negative hits, preparation, and eviction.

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

### Eager single-operation cost contract

An eager operation shares semantic validation and resolved provider behavior
with graph execution, but it must not pay graph orchestration costs merely to
execute one already-resident operation. The direct fast path applies when:

- there is exactly one semantic operation;
- all inputs belong to the same runtime and compatible storage domain;
- placement resolves to one CPU domain or one GPU device;
- no transfer, collective, cross-domain barrier, or multi-operation core
  lowering is required;
- effects and output storage can be represented by the operation-local
  contract; and
- the selected provider can prepare or execute without global liveness
  analysis.

Under those conditions the path uses a validated borrowed request and a
pre-resolved typed capability slot. It does not construct or freeze a
`SemanticProgram`, derive a program fingerprint, consult the graph
specialization cache, build a `ScheduledGraph`, integrate a global
`BufferPlan`, allocate a run event-slot table, or issue a
`RunAdmissionRequest`. Output and scratch allocation use the provider's local
`BufferContract`. A provider-specific bounded plan cache remains allowed when
the algorithm genuinely requires shape-dependent preparation. An
operation-local composite such as layout-plus-GEMM may remain on the fast path
when its temporary storage, dependencies, and effects are fully described by
that one operation contract; it need not be expanded into a general schedule.

A standalone eager call acquires at most one node-level resource lease. An
active explicit execution scope may reuse its existing compatible lease; lease
reuse is carried by an `EagerExecutionContext`, not ambient thread-local state.
If any fast-path condition fails, execution promotes explicitly to the normal
one-node prepared-graph path without changing semantics or hiding a transfer.

Before a child implementation changes eager dispatch, it records release-mode
baselines for representative no-op or metadata-light operations and small
elementwise, reduction, and contraction calls on current `main`. The child
issue fixes a non-inferiority threshold before implementation results are
known. Acceptance requires no new steady-state allocation or string lookup, no
new microsecond-scale orchestration step, and no statistically significant
regression beyond that predeclared threshold.

### Placement-bound eager API

The explicit low-level API remains `Runtime::submit(ExecutionRequest)`, while
ordinary eager calls use a lightweight placement-bound context:

```rust
let cpu0 = runtime.on(CpuPlacement::Domain(socket0));
let cpu1 = runtime.on(CpuPlacement::Domain(socket1));

let y0 = cpu0.matmul(&a0, &b0)?;
let y1 = cpu1.matmul(&a1, &b1)?;
```

The same surface binds `GpuPlacement::Device(gpu_id)`. Device placement and
CPU NUMA affinity follow different rules:

- conflicting GPU devices, CPU versus GPU, or incompatible storage domains
  remain errors and require an explicit transfer or import;
- CPU NUMA domains share one address space, so tensors with different NUMA
  affinities may participate in one operation without a copy;
- an explicit `runtime.on(CpuPlacement::Domain(id))` selects the execution
  domain even when some inputs were allocated elsewhere;
- without an explicit context, the default
  `CpuAffinityPolicy::DominantInputBytes` chooses the domain having the largest
  sum of logical input bytes, breaking ties by stable `CpuDomainId` order;
- inputs with unknown or no affinity do not contribute to that score, and the
  runtime default domain is used when no input contributes;
- `CpuAffinityPolicy::RequireSingleDomain` is an opt-in diagnostic or tuning
  policy that rejects mixed CPU affinities.

Outputs and scratch are allocated or first-touched in the selected execution
domain. Inputs remain in place and may be read through remote NUMA access.
Users may request an explicit CPU `rehome` copy when locality justifies its
cost, but eager dispatch never inserts one. Input-free allocation uses an
explicit context or runtime default.

The CPU affinity resolver is shared by eager dispatch and prepared-graph input
binding. An explicit semantic or execution-request placement constraint wins;
otherwise the runtime applies its configured `CpuAffinityPolicy` after input
metadata is known. The chosen domain and reason are observable in execution or
prepared-plan diagnostics.

`runtime.on(...)` binds a runtime and placement and may cache the current
configuration epoch and resolved capability slots. Each eager call performs a
cheap epoch check and refreshes those slots after reconfiguration; a long-lived
context does not silently pin obsolete policy. It also does not hold scarce CPU
threads or a stream indefinitely. Lease reuse occurs only inside an explicit
resource scope or already admitted executor job, preventing a long-lived eager
context from starving other work.

## Extensions

An extension crate owns four pieces:

1. its semantic payload, schema, and optional core lowering;
2. backend-specific provider traits for the backends it supports;
3. its prepared-operation representation;
4. a typed runtime adapter that connects its operation-family traits to core
   extension dispatch.

The core runtime stores an erased `ExtensionRuntimeAdapter`, but extension
authors and engines work with typed traits. Registration uses the stable
`family_id` contract from `docs/spec/extension-op.md` during runtime
configuration. An explicit `ExtensionModule` may install multiple CPU, CUDA,
XLA, or reference engine adapters into `RuntimeConfigBuilder`. Semantic payload
use does not require runtime registration. Runtime extension modules and AD
rules are installed separately, and no process-global inventory is consulted.
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

`ExtensionOp` itself is a pure deterministic semantic object: family identity,
payload identity, arity and metadata, effects and aliases, optional
`lower_to_core`, and output pruning. It cannot allocate runtime storage,
inspect providers or input values, or access device and global state. A host
reference implementation is an explicitly registered
`HostReferenceExtensionEngine`, not a method on the semantic payload. The
current `host_reference` bridge may remain only as a deprecated compatibility
adapter.

Duplicate registration is an error only for the same `(family_id, EngineId)`.
Replacing an entry is explicit. The current per-execution `register_runtime`
pattern migrates to a deprecated transactional configuration bridge; it does
not auto-register on every call.

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

`SpecializationRequirements` uses the finite runtime-owned projection described
above. It must not request arbitrary input tensor values or private provider
cache keys. Data-dependent algorithms remain dynamic execution operations
rather than compile-time inspection of user data.

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

`EagerTensor` pairs a physical tensor with `Arc<Runtime>`. The physical storage
does not own its engine, avoiding runtime-storage reference cycles. Identity is
split into:

- `RuntimeId`, for runtime ownership and configuration compatibility;
- `StorageDomainId`, for address-space, allocator, lifetime, and device
  interoperability;
- `AllocationDomainId`, for the allocator or pool that must reclaim storage;
- `EventDomainId`, for completion-token interoperability; and
- placement metadata, including an optional CPU NUMA affinity or a strict GPU
  device placement.

Storage records these identities, its allocation owner, and pending
completion. A prepared graph records `RuntimeId` and epoch. CPU allocations on
different NUMA nodes may share one compatible `StorageDomainId` while retaining
different `AllocationDomainId` and affinity metadata.

Mixing tensors from different runtimes or incompatible storage domains is an
error. Different compatible CPU allocation domains are not an error: they
affect locality and reclamation, not addressability. Foreign or unregistered
storage is rejected. Cross-runtime movement uses explicit
`runtime.import(tensor, ShareIfCompatible | Copy)` or transfer APIs. Zero-copy
sharing requires compatible allocator, device, event interoperation, and
lifetime contracts. There is no implicit copy fallback.

Runtime configuration is an immutable `RuntimeConfigSnapshot` containing its
epoch, engines, extensions, transfers, collectives, provider-selection policy,
and topology. Reconfiguration builds and validates a complete replacement,
increments the epoch once, and publishes it atomically; failure leaves the old
snapshot active. Preparation pins one snapshot. In-flight work retains old
prepared-operation `Arc`s and may finish, while explicit execution of an old
prepared graph returns stale and the convenience path re-prepares.

Registering the same family and registration identity is a no-op; a conflicting
duplicate is an error and replacement must be explicit. A builder is the
normal construction API, while transactional reconfiguration supports plugins,
tuning, and migration. The steady executor takes no registry lock.
Runtime-owned caches and resource pools expose individual and aggregate bounds,
clear operations, entry counts, and retained-byte estimates.

## Buffer Planning

Each prepared provider returns a `BufferContract`; `Runtime` integrates all
contracts into one `BufferPlan`. Output storage is one of
`RuntimeAllocated`, `ProviderAllocated`, or `ViewOf`. The plan owns logical
slots, liveness, alias and view relations, allocation classes, scratch,
dynamic-size bounds, and a peak-memory estimate.

Provider-allocated storage is registered with its storage domain, allocation
domain and owner, byte size, completion event, destructor, layout, dtype, and
placement or affinity metadata, and remains visible in explain and statistics.
Exact dynamic sizes allocate from the resolved plan; bounded sizes reserve the
bound; unbounded sizes require an explicit dynamic-allocation contract and
cannot claim a fixed peak.

Internal values may be donated automatically at proven last use. External
inputs are borrowed by default and become candidates only through explicit
`BindingMode::Donatable`. Donation is refused for shared, viewed, foreign, or
still-pending storage.

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

CPU execution uses an object-safe `CpuDomainExecutor` with two distinct
operations: `submit` for outer scheduling and synchronous `install` for an
inner parallel region. `ScopedCpuJob` permits a borrowed synchronous job, and
the standard distribution includes a Rayon adapter. The executor advertises
worker count, submit/install support, reentrancy, and external affinity and
shutdown guarantees. CPU set, NUMA identity, and thread budget belong to
`CpuResourceDomain`, not to the executor object.

Every operation receives `ParallelMode::Outer`, `Inner`, or `Sequential`.
Providers may not use ambient Rayon or independently submit nested work. A
composite contraction or batched linalg implementation chooses outer versus
inner parallelism once, and all delegated providers honor that choice.

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
CPU sets, and thread budgets through the same `CpuDomainExecutor` contract.
Tenferro validates static configuration such as empty CPU sets, duplicate
domain identities, and overlapping declared CPU sets.

The external owner is responsible for:

- fairness and lifecycle of jobs after submission to the external executor;
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

A thread budget is an upper bound, not an exact-width requirement. A provider
may use fewer threads, and a sequential provider satisfies every positive
budget. A violation occurs only when a provider may exceed the cap and has no
mechanism to prevent it.

Thread-control capability is resolved when constructing the provider or during
preparation, never by probing on every call. A system OpenBLAS provider may use
`dlsym` to detect `openblas_set_num_threads_local` and
`openblas_get_parallel()`. It claims exact thread-local control only when the
local setter exists and `openblas_get_parallel()` reports the pthread variant;
an OpenMP build does not receive that claim because local isolation is not
reliable. Strict enforcement without the required capability returns a typed
configuration or prepare error and never degrades to a global setter.

Known control granularity is:

| Provider | Mechanism | Granularity |
| --- | --- | --- |
| faer / native kernels | per-call executor or parallelism argument | any `N` |
| MKL | `mkl_set_num_threads_local` | any `N`, thread-local |
| OpenBLAS, recent pthread build | `openblas_set_num_threads_local` | any `N`, thread-local |
| Accelerate, macOS 15+ | `BLASSetThreading` | binary single/auto, thread-local |
| Accelerate, macOS 14 and older | `VECLIB_MAXIMUM_THREADS` | global, effectively startup-fixed |

For binary control, a budget of one selects single-threaded mode. An
intermediate budget such as eight is an explicit runtime policy choice between
clamping to one, which respects the upper bound, and returning unsupported.
Auto mode is invalid if it may exceed the cap. If a BLAS library cannot provide
the requested isolation, the engine does not claim NUMA placement that it
cannot enforce.

### Normative faer batched-GEMM policy

Extracting the GEMM provider must preserve current `main` behavior:

- for grouped or batched jobs with more than one job and a context with more
  than one thread, the engine composite uses outer `jobs().par_iter()` on the
  selected domain executor and forces every per-job faer kernel to
  `faer_seq()`;
- with one job or a single-threaded context, the job loop is sequential and
  the faer kernel may use inner parallelism through `faer_par()`;
- for strided-batched `dot_general`, the outer batch loop remains sequential
  and the inner faer kernel parallelizes.

The invariant is one Rayon fan-out level: either jobs parallelize and kernels
are sequential, or the loop is sequential and kernels parallelize. All work
stays under one outer lease and its upper-bound thread budget. The
engine-owned composite, not `CpuGemmProvider`, owns this choice. Changing it
requires benchmark evidence and a normative specification update rather than
occurring as a side effect of provider replacement.

### Memory locality

The NUMA model includes memory locality, not just worker affinity. Buffers and
scratch should be allocated or first-touched within the chosen domain executor
and returned to their owning node-local pool. A buffer's
`AllocationDomainId` determines reclamation and pool reuse, while its NUMA
affinity is a scheduling hint. Neither prevents another CPU domain in the same
compatible storage domain from reading or writing the allocation through the
shared address space.

For mixed-affinity inputs, the selected execution domain may incur remote NUMA
traffic but performs no hidden migration. The output and scratch belong to the
selected domain; each input retains its original allocation owner. Reuse by a
different node-local pool remains forbidden unless an explicit rehome or
allocator-compatibility contract permits it.

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

## Asynchronous Execution and Events

The runtime owns event storage rather than allocating an
`Arc<dyn CompletionEvent>` for every node or defining a closed backend-event
enum. One illustrative implementation preplans slots identified by
`EventDomainId`, `EventSlotId`, and a generation, with the actual CUDA, WebGPU,
PJRT, or CPU completion token stored in engine-owned per-run workspace. Exact
slot identity, recycling, and generation management are reserved for the event
child design.

The engine attaches dependencies to subsequent work and providers never call a
global synchronize after each operation. Cross-event-domain dependencies
require an explicit `Transfer` or `Barrier`; the executor does not synthesize a
hidden global synchronization. Synchronization occurs only at an observable
boundary:

- host read or download;
- explicit synchronize;
- an external provider whose contract is synchronous;
- safe reuse or destruction of a resource;
- a host-visible error flag that requires completion.

In externally managed GPU execution, the user owns stream lifetime and
interoperation constraints, while tenferro still tracks ordering for work it
submits.

Cancellation is best-effort. It removes waiting resource requests and prevents
unsubmitted nodes from being enqueued, but cannot assume already enqueued
device work is cancellable. Submitted work is drained, leases remain held until
completion, effects are not rolled back, and dropping an execution handle does
not imply cancellation.

## Admission Control and Resource Leases

The illustrative arbiter design has two levels. A `RunAdmissionRequest`
reserves persistent, maximum-live, and provider-bound memory derived from
`BufferPlan`. Once admitted, each ready node submits one atomic
`NodeLeaseRequest` containing all required CPU domains or thread budgets,
device streams, scratch memory, and exclusivity constraints. The selected
invariant is that the arbiter grants a node's complete resource set or queues
it while it holds nothing, acquiring multi-domain resources in a deadlock-free
order. A child design may combine or split the two admission levels if it
preserves this invariant and enforces the prepared memory feasibility bound.

Leases live through asynchronous completion and, for buffers, through their
planned last use. An `AnyCompatible` resource may be selected when granting the
lease, but it may vary only within the compatibility class accepted during
preparation; the provider and algorithm do not change. The umbrella contract
requires starvation avoidance but does not select FIFO, aging, priorities, or
deadline semantics; queue policy belongs to the resource-arbiter child design.
Nested provider operations reuse the current lease. Recursively calling
`Runtime::run` from a provider is prohibited.

The executor uses run-level fail-fast semantics. The first failure becomes the
primary error, no new node is enqueued, waiting and unsubmitted nodes are
cancelled, and already submitted work is drained safely. Additional drain
failures are retained as suppressed errors. Partial outputs are not returned as
a successful run.

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

The deterministic resolution order is:

1. an operation-local explicit override;
2. placement constraints;
3. runtime family preference;
4. a native engine capability;
5. semantic lowering to core operations;
6. an explicitly configured fallback engine;
7. `PrepareError::UnsupportedOperation` or
   `PrepareError::UnsupportedExtension`.

Provider and algorithm are fixed at preparation. The initial design has no
adaptive current-load selection or implicit cost model. An equivalent resource
lease may be selected late only within its prepared compatibility class. The
selected implementation and reason are recorded by `prepared.explain()`. A
future cost model must be an explicit versioned policy.

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

Ownership follows crate boundaries: `tenferro-program` owns build, validation,
extension-schema, and transform errors; `tenferro-runtime` owns configuration,
prepare, enqueue, completion, and execution errors; operation crates own their
planning and preparation errors. `ExecutionError` is the sum of `Prepare`,
`Enqueue`, and `Completion`; compilation remains separate. A convenience
trace-and-run API may expose an outer `TraceRunError`.

Public errors preserve typed source chains and a runtime classification without
making unstable provider internals part of the public enum. Only a typed
`Unsupported` result continues capability selection. Input metadata and shape
guards are validated before specialization work, provider work, or resource
acquisition. Resource leases are acquired only after preparation has produced
a feasible schedule.

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

## Determinism Policy

`ExecutionPolicy::determinism` has two initial levels:

- `Fast`, the default, guarantees deterministic semantic compilation,
  provider selection, partitioning, and cache identity, but not a fixed
  concurrent execution order or bitwise-identical parallel reductions;
- `Reproducible` admits only provider algorithms declaring reproducible
  behavior and fixes reduction trees, contraction paths, and subgraph
  partitioning. Unsupported operations fail during preparation with
  `DeterminismUnsupported` rather than falling back to `Fast`.

The reproducibility scope is the same runtime snapshot, inputs, and hardware
class. Cross-backend bitwise equality and reproducibility across arbitrary
library versions are not initial guarantees. A stricter `Bitwise` level may be
designed later.

## Testing and Performance Evidence

Testing is organized around reusable contracts rather than duplicating one
large integration suite for every backend:

- `tenferro-program` property tests cover builders, validation, fingerprint
  collisions, effects and aliases, transforms, and extension lowering;
- runtime tests use mock engines, event domains, and resource arbiters to make
  single-flight preparation, epochs, atomic leases, fail-fast draining,
  cancellation, and buffer lifetimes deterministic;
- a provider contract suite checks capability declarations, monotonic
  specialization, buffer contracts, aliases, event completion,
  `ParallelMode`, and determinism claims;
- a small backend-parity suite executes common programs on faer, general
  BLAS/LAPACK, TBLIS where available, CUDA/CubeCL, WebGPU, and XLA, comparing
  numeric tolerances, effect order, and typed failure behavior;
- fault injection covers prepare, allocation, enqueue, device completion, and
  transfer failures, including primary and suppressed errors and resource
  release.

Bitwise equality is tested only where the selected `Reproducible` provider
contract promises it.

Required focused tests include:

- CPU, GPU, and XLA preparation observe the same semantic fingerprint for the
  same graph;
- `SemanticProgram` contains no runtime, engine, thread-pool, stream, allocator,
  or lease handle;
- no `PreparedGraph` contains an unresolved operation;
- provider replacement preserves semantic results;
- request construction and resolved dispatch allocate nothing in steady state;
- eligible eager single-operation calls avoid graph artifacts, global buffer
  planning, run admission, and schedule construction;
- eager fast-path latency satisfies the predeclared non-inferiority threshold
  against the current implementation;
- placement-bound eager contexts apply the documented distinct CPU-affinity
  and GPU-device rules without an implicit copy or device transfer;
- mixed CPU NUMA affinities execute in the deterministic dominant-input domain,
  retain every input's allocation owner, and place outputs in the selected
  domain;
- `CpuAffinityPolicy::RequireSingleDomain` rejects mixed CPU affinities while
  the default policy accepts them;
- conflicting GPU devices still require an explicit transfer;
- BLAS capability probing rejects strict budgets that cannot be enforced;
- extracting or replacing `CpuGemmProvider` preserves the normative grouped
  and strided-batched faer parallelization policy;
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
- NUMA buffers remain associated with their allocation domain even when an
  operation executes from another CPU domain;
- `ExternalManaged` does not reconstruct a supplied executor;
- GPU dependency tracking avoids unnecessary global synchronization;
- independent multi-GPU work can enqueue concurrently;
- foreign runtime tensors do not transfer implicitly;
- unsupported provider paths return explicit errors rather than fallback;
- same-key preparation is single-flight while different keys make progress;
- non-monotonic specialization requests fail as provider contract violations;
- atomic node lease requests never hold a partial resource set while queued;
- run-level failure stops new enqueue and drains already submitted operations;
- `Reproducible` rejects unsupported algorithms during prepare rather than
  silently using `Fast`.

Performance work must measure representative shapes, ranks, batch counts,
thread counts, NUMA placements, and device counts. Microbenchmarks must keep
validation, request construction, dispatch, provider call, and kernel work
separable.

## Migration Strategy

This is an umbrella architecture. Migration is incremental, keeps the current
backend path working behind adapters, and is decomposed into independently
accepted child issues:

1. Extract validated borrowed requests, typed CPU capability slots, and the
   layout/GEMM/general-contraction provider composites behind adapters on the
   current eager and backend path. Preserve current performance and faer batch
   policy while delivering provider replacement early.
2. Introduce placement-bound eager contexts, `CpuDomainExecutor`, explicit
   parallel modes, and managed/external NUMA resource domains behind the same
   compatibility layer. Establish the eager fast-path benchmark gate.
3. Introduce `tenferro-program`, private immutable `SemanticProgram`, builders,
   fingerprints, effects, aliases, and adapters from current graph artifacts.
   Split mutable `TraceContext` from pure `GraphCompiler`.
4. Introduce immutable runtime snapshots, the remaining typed core
   capabilities, explicit extension modules, `PreparedOperation`, finite
   specialization requirements, and single-flight bounded plan caches.
5. Introduce common `ScheduledGraph`, runtime-owned event domains, buffer
   planning, resource admission, and `GraphExecutor`. Port CPU graph execution
   while retaining a compatibility adapter for `GraphExecutor<B>`.
6. Migrate extension capability resolution and pure core lowering. Use N-ary
   einsum to validate shape specialization, planning policy, resolved slots,
   AD registration, and host-reference fallback; then migrate FFT, linalg,
   sparse, and permutation families.
7. Split CUDA/WebGPU resources from provider algorithms, port GPU execution to
   the common scheduler, and use runtime-owned event slots and explicit
   transfers.
8. Integrate XLA through `SubgraphCompiler` and `PreparedOperation`, then retire
   `GraphProgramLoweringView` and the executor-shaped portable artifact.
9. Add multi-GPU task scheduling for independent work.
10. Add structured control flow, logical sharding, collectives, and resharding
    only through later accepted designs.

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
buffer-lifetime, resource, and observability contracts. Rejected as a public
execution boundary. XLA may still produce an opaque whole-region executable
internally, retained by a `PreparedOperation` in the common schedule.

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
- [StableHLO specification](https://openxla.org/stablehlo/spec)
- [PJRT uniform device API](https://openxla.org/xla/pjrt)

## Child Design and Implementation Boundary

The umbrella design selects the architectural invariants needed for child
decomposition: crate ownership, process-local portability, immutable public
program access, typed core capabilities, pure extensions, runtime identity and
epochs, safe prepared-operation dispatch, external CPU executors,
runtime-owned events, explicit buffer contracts, all-or-none multi-resource
acquisition, finite specialization projections, structural collision checks,
no rollback of effects, draining of submitted work, and the determinism policy.

Mechanism details are reserved for workload-informed child designs. These
include event-slot identifier and generation representation, slot recycling,
buffer-donation heuristics, exact dynamic-buffer reservation, the precise
two-level admission accounting algorithm, queue ordering or priority policy,
and the public cancellation state machine. Child designs must preserve the
selected invariants but may replace the illustrative representations used in
this document.

This is still an umbrella design rather than authorization for one monolithic
implementation PR. Each numbered migration phase must become a reviewable
child issue with exact public signatures, compatibility impact, benchmarks,
and acceptance tests. Later work may refine names and reserved mechanisms
without changing the selected ownership and behavioral invariants.
Implementation planning starts only
after maintainers review this written WIP design.
