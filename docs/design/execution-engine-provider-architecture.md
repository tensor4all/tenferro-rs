# Accepted Architecture Design: Pluggable Execution Engines and Resource Domains

> Planning authority and current phase status live in the
> [execution-engine umbrella issue](https://github.com/tensor4all/tenferro-rs/issues/1433).
> This child document owns the detailed proposed architecture and rationale.

## Status

The umbrella architecture is accepted for child planning and governed by the
execution-engine umbrella plan. This document records the agreed design
direction and invariants; it does not freeze exact public signatures or serve
as an implementation plan. Mechanism refinement and implementation approval
occur in bounded phase children, beginning with issue #1434.

This revision incorporates all contracts added to issue #1433 after
`f777b52e`: separate CPU thread-count and placement capabilities, ArmPL/NVPL
classification, the PyTorch-aligned batched-contraction target policy,
MPI-compatible process boundaries, the DMRG-class driving workload, and the
consolidated requirements retained from closed issues #1432, #1417, and #1422.

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

### Component and crate ownership

The first implementation keeps the logical program component in
`tenferro_runtime::program`. That component owns `SemanticProgram`,
`CoreSemanticOp`, `ExtensionOp`, semantic metadata, shape guards, effects,
alias declarations, validation, and process-local structural fingerprints.
`tenferro-runtime` also owns `TraceContext`, `GraphCompiler`, runtime and
engine traits, `PreparedGraph`, `ScheduledGraph`, `GraphExecutor`,
`ResourceArbiter`, and prepared-plan caches. This module-first boundary avoids
freezing a new public crate while the semantic representation and extension
interfaces are still changing.

The logical program component has no dependency on runtime resources,
providers, scheduling, or AD even while it is physically a runtime module.
Phase 3 must keep that dependency boundary mechanically auditable. Extraction
to a public `tenferro-program` crate occurs only after the representation is
stable and a direct external consumer such as XLA or an operation crate needs
the semantic types without the runtime surface. The extraction child must show
an acyclic dependency graph and must not change the artifact semantics.

`GraphCompiler` remains in `tenferro-runtime` initially because its compiler
service, trace integration, and cache are closely tied to runtime-facing
workflows. It may be split later without changing the program artifact.

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
`ExecProgram` may be used as temporary internal staging types, but they are not
compatibility surfaces and are removed when their owning migration phase
completes.

`SemanticProgram` has private fields and read-only views and iterators. A
validation-preserving builder is available to low-level frontends, but callers
cannot mutate an already frozen program. `CoreSemanticOp` is a public,
`#[non_exhaustive]` enum. The initial program contains one acyclic
`SemanticRegion`; the first implementation has no `if`, `while`, block
arguments, yields, shape joins, or loop AD.

The reserved structured-control-flow model is at least an MLIR-style
region/block model: a `SemanticRegion` contains ordered blocks, a block has
typed block arguments, operations, and exactly one terminator, successor edges
pass explicit operands, and region operations declare yielded values. Branch
and loop validation must define dtype and symbolic-shape joins, including the
guards required when incoming shapes are not statically equal. Nested regions
cannot capture runtime resources or bypass effect and alias analysis. Before
adding public control flow, the phase 10 child must prove that representative
`if` and `while` programs, their shape joins, and their AD requirements are
representable in this model; reserving `SemanticRegion` alone is not evidence
of that capability.

Builder-issued graph and value handles are opaque tokens scoped to one builder;
raw integer identities are never public. Using a token with another builder is
a typed error. Import is atomic across the source graph, value map, bindings,
roots, checkpoint, and metadata or constraint scopes. Finishing is likewise
atomic for both metadata and tensors and returns an error without requiring a
caller panic. Extensions emit through the same supported builder interface,
with a source-contract check that prevents them from depending on private
representation fields.

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
`GraphCompilerEinsumExt` are replaced atomically when the new compiler
extension point lands; they are not permanent compiler extension points.

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
runtime preparation. The logical program component has no AD dependency.
Extension-owned rules are installed explicitly in an `AdContext`; there is no
global rule inventory. AD tries the extension rule before runtime lowering.
Differentiating through a lowering is an explicit fallback policy, effectful
operations require a rule, and in-place semantics require functionalization.

The phase 3 AD child must define semantic-rule traits and migrate all three
current extension roles explicitly:

| Current role | Semantic-program role |
| --- | --- |
| `ExtensionLinearizeRule` | Emit a validated primal-plus-linear semantic fragment from one extension op and active tangent inputs. |
| `ExtensionLinearTransposeRule` | Transpose an already linearized semantic fragment and emit cotangents for its active inputs. |
| `ExtensionPrimalVjpRule` | Optional direct-VJP transform retained only as a measured optimization; the canonical reverse path remains linearize then transpose. |

The new callbacks consume immutable semantic op/value views and emit through a
validation-preserving semantic AD builder. They do not expose
`ValueKey<StdTensorOp>`, `PrimitiveRuleBuilder`, `ShapeGuardContext`, or the
current executor graph. The child must specify active-input and absent-tangent
encoding, multi-output ordering, residual capture, effect rejection,
provenance, typed failures, and cache identity. It must migrate the FFT and
test extension rules and compare JVP/VJP results against the current rules
before the old traits are removed in the same migration phase.

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

The names and exact signatures are not fixed by this architecture-level
document. The contract is that validation and semantic preparation happen once
at a public or planning boundary, and engines receive validated borrowed
requests.

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
scheduling. `CpuGeneralContractionProvider` consumes a full validated binary
`dot_general` request, including TBLIS-style label groups, without forcing the
engine to flatten it into GEMM. `DotGeneralRuntime` is an engine-owned composite
that may use such a provider for a direct binary contraction, or a
`CpuLayoutTransformProvider` plus `CpuGemmProvider` for decomposition. N-ary
einsum first resolves a contraction path into binary `dot_general` operations.
A user may replace the complete composite, the general-contraction provider,
or only GEMM/layout providers.

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
    Collective(PreparedCollective),
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

`Transfer` and `Collective` are scheduler-owned node families, not arbitrary
extension operations. A transfer has explicit source and destination storage
domains and bridges their event domains by producing a destination-domain
completion dependency; the scheduler understands its buffer lifetime,
ordering, failure, and resource requirements. A collective similarly has
explicit participants, ordering, event-domain behavior, and communication
resources. Provider registries may supply their implementations, but cannot
hide them inside an opaque extension op. The initial refactor implements
transfers and may leave collectives unavailable; reserving the core collective
node prevents a later sharding design from bypassing common scheduling and
lifetime rules.

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
or wait for already submitted work. Eager GPU operations likewise return
tensors carrying pending completion; host reads, explicit synchronization, or
export observe completion and deferred errors.

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

- eager-local pure legalization yields exactly one executable semantic
  operation, whether native or one core operation produced by lowering;
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

Fast-path eligibility is evaluated after eager-local pure lowering, not merely
from the source extension op count. An extension with a native engine remains
one operation. A `lower_to_core` result may also remain eligible only when it
emits exactly one core operation and its validation, placement, effects,
aliases, output and scratch needs are operation-local. Lowering to two or more
operations always promotes to prepared-graph execution, even when the lowered
sequence is short. Consequently, an extension author who requires predictable
eager latency must provide either a native engine or a guaranteed single-core-
operation lowering; providing generic `lower_to_core` alone does not guarantee
the eager fast path. N-ary einsum and decompositions lowered to multiple core
ops intentionally pay preparation and scheduling unless a native engine
represents them as one prepared operation.

Eager remains a first-class supported API. The architecture acceptance promise
is current-main non-inferiority, not a positive reduction of the measured
9-11 us fixed cost for small calls. Phase 1 reuses the existing single backend
session entry and must not add another; it is not blocked on a budget-one eager
fast path. Current measurements show why thread count alone is not a design:
one-thread `CpuContext::install` already executes directly at about 0.56 ns,
while one-thread `CpuBackend::install` remains about 6.9 us. Any future eager
fixed-overhead child must first separate validation, eager bookkeeping,
admission, session entry, provider dispatch, and output allocation before
selecting a mechanism.

A standalone eager call acquires at most one node-level resource lease. An
active explicit execution scope may reuse its existing compatible lease; lease
reuse is carried by an `EagerExecutionContext`, not ambient thread-local state.
If any fast-path condition fails, execution promotes explicitly to the normal
one-node prepared-graph path without changing semantics or hiding a transfer.

Before a child implementation changes eager dispatch, it records release-mode
baselines for representative no-op or metadata-light operations and small
elementwise, reduction, contraction, and indexed calls on current `main`. The
canonical starting source is the existing
`crates/tenferro-ad/benches/eager_dispatch_baseline.rs`; it must be extended
with an indexed case before candidate code is measured. Existing einsum and
linalg benchmarks supplement this dispatch suite for extension-native and
promoted multi-operation paths. The child issue fixes a non-inferiority
statistic, threshold, repetition policy, and noisy-run handling before
implementation results are known. Acceptance requires no new steady-state
allocation or string lookup in the measured eager hot path, no new
microsecond-scale orchestration step, and no statistically significant
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
current `host_reference` bridge is removed when that engine lands; it does not
remain as a parallel compatibility surface.

Duplicate registration is an error only for the same `(family_id, EngineId)`.
Replacing an entry is explicit. The current per-execution `register_runtime`
pattern may be routed through a temporary internal transactional staging path
within its migration phase, but the old public entry point is then removed. It
does not auto-register on every call.

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

Graph lowering is invoked by runtime preparation, because the decision depends
on available engines and may depend on a concrete input signature. The pure
`GraphCompiler` preserves the extension payload in `SemanticProgram`. Runtime
preparation first checks native capability, then attempts core lowering, then
uses an explicitly configured fallback engine if one exists. The eager
dispatcher may invoke the same pure lowering callback before constructing a
program only to decide the single-operation path described above; it may
execute the result directly only when exactly one core operation is emitted and
every other eager fast-path condition holds.

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

The strong-ownership graph is acyclic by contract:

```text
EagerTensor -> Arc<RuntimeHandle> -> Arc<RuntimeState>
     |                                  |
     v                                  v
TensorStorage -> PendingCompletion   snapshots/caches/executor services
                     |
                     v
              Arc<InFlightRun> -> storage + leases + events + prepared ops
```

There is no strong edge from `RuntimeState`, a cache entry, a prepared plan, an
event, or an `InFlightRun` back to `EagerTensor`, `Tensor`, or another public
tensor wrapper. Runtime registries that index public values use opaque IDs and
`Weak` backreferences only. `InFlightRun` owns exactly the storage, prepared
operations, configuration snapshot, leases, event tokens, and completion
state required to finish submitted work; it does not own `RuntimeHandle` or
`RuntimeState`. Pending output storage may own its `InFlightRun` completion,
but completion records never own the output wrapper. Cache insertion must be
rejected in review if its value can introduce a strong path back to the cache
owner. The runtime ownership child includes a cycle test using weak sentinels
for completed, cancelled, failed, and dropped-handle runs.

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

The engine first creates one crate-private, unentered operation capability:

```rust
pub(crate) struct CpuOperationEntry<'a> {
    domain: &'a CpuResourceDomain,
    permit: &'a ResourcePermit,
}

pub struct CpuExecutionContext<'a> {
    domain: &'a CpuResourceDomain,
    parallel_mode: ParallelMode,
}
```

`CpuOperationEntry` alone owns checked executor `install` and `submit`. It
constructs a `CpuExecutionContext` only after entering the selected executor:
inside the installed sequential/inner job, or separately inside each submitted
outer child. Consequently every public `CpuExecutionContext` is already
entered. It exposes immutable domain facts and logical policy, but has no
install, submit, mode-mutation, child-construction, permit, or scheduling-error
surface.

Providers consume the already-entered context. They do not choose a NUMA node,
query ambient Rayon state, acquire a second resource permit, or re-enter the
executor. The outer operation holds one permit. Internal provider and composite
calls reuse the borrowed context by direct delegation and do not re-enter
`CpuBackend`. This prevents nested oversubscription and makes the thread budget
a single engine-owned policy.

CPU execution uses an object-safe `CpuDomainExecutor` with two distinct
operations: `submit` for outer scheduling and synchronous `install` for one
sequential or inner operation entry. `ScopedCpuJob` permits a borrowed
synchronous job, and the standard distribution includes a Rayon adapter. The
executor advertises worker count, submit/install support, reentrancy, and
external affinity and shutdown guarantees. CPU set, NUMA identity, and thread
budget belong to `CpuResourceDomain`, not to the executor object.

Every operation receives `ParallelMode::Outer`, `Inner`, or `Sequential`.
The executor-entry mechanism, logical mode, and owner of provider workers are
separate axes:

| Selected operation | Executor calls | Entered provider context | Fan-out owner |
|---|---:|---|---|
| Sequential top-level | install = 1, submit = 0 | Sequential | none |
| Inner engine workers (faer/native) | install = 1, submit = 0 | Inner | selected Rayon executor |
| Inner external workers (BLAS) | install = 1, submit = 0 | Inner | external provider runtime |
| Outer | install = 0, submit = 1 | Sequential in every child | executor |

`ParallelMode::Inner` does not itself imply that
`CpuDomainExecutorCapabilities::inner_parallelism` is Rayon. An external-worker
provider still crosses the selected executor once for admission and placement,
then owns its provider fan-out. Task 7 adds construction-time count and
placement capability classification; until then the BLAS selection is the
unavoidable backend-kind special case. No path may use that pending work to
bypass executor installation.

Providers may not use ambient Rayon or independently submit nested executor
work. A composite contraction or batched linalg implementation chooses outer
versus inner parallelism once, and all delegated providers honor that choice.
Direct backend calls and backend-session operations use the same
`CpuOperationEntry` boundary. `with_linalg_pool` enters internally and passes
an already-entered context to the operation-family implementation.

### Re-entry during migration

Current `main` rejects arbitrary `CpuBackend` re-entry on the active thread or
managed Rayon scope through `BACKEND_REENTRY_PANIC`. The architecture does not
silently reverse that public safety contract. Phases 1 and 2 preserve the
rejection while moving internal composition below the session boundary, where
providers receive `CpuExecutionContext` and call each other directly without a
second backend entry.

Thus, "reuse the outer lease" means trusted runtime composition within one
already-entered session. It does not mean that an application closure may call
`CpuBackend::install` recursively. Supporting arbitrary nested backend calls,
or replacing the panic with a typed scoped API, requires a separate accepted
child design with oversubscription, provider-exclusivity, unwind, and
compatibility tests.

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

The runtime retains the supplied executor or pool owner for at least the
lifetime of every domain, lease, and submitted job that refers to it. It never
repins external workers and does not attempt to verify their live OS affinity;
the declared `CpuSet` is a caller contract, and an inaccurate declaration is a
placement or performance contract violation rather than a memory-safety
mechanism. Arbitration still uses the declared exact CPU sets. A runtime may
register multiple external domains and resolves them by placement instead of
collapsing them into one ambient pool. Diagnostics distinguish declared
placement, unverifiable external affinity, executor capabilities, and the
owner responsible for shutdown.

The external owner is responsible for:

- fairness and lifecycle of jobs after submission to the external executor;
- maintaining worker affinity;
- coordinating pools used by other libraries;
- avoiding oversubscription;
- external BLAS/OpenMP global state;
- executor lifetime and shutdown.

Tenferro must not reconstruct or silently replace an external executor.
External registry construction and provider validation are one atomic
operation: the ordinary constructor validates the standard bundle, while
`from_external_managed_domains_with_provider_bundle` accepts a caller bundle
and returns no backend if any domain rejects it. This custom route is required
when the compiled default is an uncontrolled BLAS adapter. In Task 7a the
bundle covers `dot_general` providers only; linalg provider capability remains
the separate Task 7b boundary.

### Two capability axes: thread count and placement

CPU resource capability has two independent axes:

1. **Thread-count control** asks whether a provider can enforce the execution
   budget as an upper bound for this call without interfering with another
   domain.
2. **CPU-placement control** asks whether every worker used by that call is
   confined to the selected domain's concrete `CpuSet`.

Construction or preparation probes both axes once and records the result in
the prepared provider capability. They are not inferred from a library name
and are not reprobed during domain validation or execution. The current CPU
bundle samples each provider slot's `execution_capabilities()` exactly once at
bundle construction and retains that immutable snapshot for the bundle
lifetime. A thread budget remains an upper bound: a provider may use fewer
threads, and a serial provider satisfies every positive budget.

Placement for a budget greater than one is enforceable only when the kernel
runs on engine-supplied workers, as with faer and tenferro-native kernels.
External BLAS worker pools are not modeled as a `CpuDomainExecutor`:

- OpenBLAS pthread builds use a process-global, lazily created worker pool. The
  misleadingly named `openblas_set_num_threads_local` reads the old count,
  calls the process-global setter, stores additional non-TLS global state, and
  returns the old count for later restoration. The changed count is visible to
  concurrent threads. Distributions may use `NO_AFFINITY` or their own
  affinity policy, and there is no per-call or per-domain binding API. Its
  parallel BLAS server executes only one multithreaded job at a time, so
  simultaneous multithreaded calls from different domains serialize on its
  internal lock.
- MKL's `KMP_AFFINITY` placement is process-global rather than a per-call
  domain binding.
- Accelerate schedules through Grand Central Dispatch, and macOS exposes no
  API for binding those workers to a selected set of cores.

A budget of one is sound for both axes only when the provider can force
worker-local sequential execution, because that call then runs inline on the
already pinned calling thread. Under strict `Managed` or `ExternalManaged`
placement, an exact domain `CpuSet` combined with a count-controlled external
BLAS budget greater than one is a typed placement error unless the domain is
the process's complete allowed CPU set or placement is explicitly advisory.
An uncontrolled provider such as parallel OpenBLAS fails the independent
thread-count contract for every finite strict bundle budget; it remains
available only through the process-global `ProviderDefaultExclusive`
compatibility policy.

### BLAS and faer behavior

Faer and tenferro-native parallel kernels can use the selected executor when
their APIs permit it. General BLAS/LAPACK providers often expose only a global
or provider-specific thread count, not a per-call executor. Such a provider
receives the execution thread budget and provider exclusivity policy, but it
does not receive a Rayon pool as if BLAS could execute on that pool.

Thread-control capability is resolved when constructing the provider or during
preparation, never by probing on every call. OpenBLAS symbol presence does not
provide per-call count control: in OpenBLAS 0.3.32,
`openblas_set_num_threads_local` is a process-global set-and-restore helper for
both pthread and OpenMP builds, not TLS. Parallel OpenBLAS therefore remains
`GlobalOrUncontrolled` even when that symbol is present and an adapter invokes
it. Strict enforcement without the required capability returns a typed
configuration or prepare error and never degrades to a global setter.

Known count and placement capability is:

| Provider | Count mechanism | Count granularity | Placement with budget > 1 |
| --- | --- | --- | --- |
| faer / native kernels | per-call executor or parallelism argument | any `N` | exact engine-supplied domain workers |
| MKL | `mkl_set_num_threads_local` | any `N`, thread-local | process-global `KMP_AFFINITY`; not exact per domain |
| OpenBLAS, pthread or OpenMP | `openblas_set_num_threads_local` calls global set and returns the old value | process-global set-and-restore; no per-call claim | process-global worker pool; not exact per domain |
| OpenBLAS, sequential build | no worker pool | always one thread | pinned `CallingThread` |
| Accelerate, macOS 15+ | `BLASSetThreading` | binary single/auto, thread-local | GCD workers; not exact per domain |
| Accelerate, macOS 14 and older | `VECLIB_MAXIMUM_THREADS` | global, effectively startup-fixed | GCD workers; not exact per domain |
| ArmPL `_mp` | OpenMP controls | probe at construction; no exact thread-local claim | external OpenMP workers; not exact per domain |
| ArmPL serial | no worker pool | always one thread | pinned calling thread |
| NVPL on Grace | vendor/runtime-dependent controls | probe and classify at construction | external workers; not exact per domain |

The table above is the target adapter policy, not a claim that every genuinely
local setter is already wired. The current conservative Task 7a implementation
does not yet apply and restore MKL or Accelerate local controls around a call.
It also does not identify the linked OpenBLAS build mode in production, so it
cannot claim the sequential-build exception. Consequently every current
built-in BLAS descriptor, plus TBLIS, remains `GlobalOrUncontrolled` external
workers in Task 7a. A future adapter may classify a positively identified
sequential OpenBLAS build as `Sequential` on `CallingThread`; parallel OpenBLAS
cannot be upgraded by wiring its `_local` symbol because the underlying count
is process-global. `ProviderDefaultExclusive` preserves legacy `Auto`
execution under the process-wide permit, while explicit strict provider
bundles are rejected rather than receiving a false thread-count or
NUMA-placement guarantee. Task 7b owns scoped guards for providers with true
local controls and may separately classify OpenBLAS global set-and-restore for
exclusive compatibility and diagnostics, never as per-call control.

`BinaryClampToOne` has one unambiguous finite-budget meaning: its adapter
selects single-threaded mode for every resource-domain call, including a
budget such as eight, and never selects provider-controlled auto mode. Using
fewer threads respects the upper-bound contract. An adapter that cannot make
that guarantee reports `GlobalOrUncontrolled` (or rejects the request) instead.
If a BLAS library cannot provide the requested isolation, the engine does not
claim NUMA placement that it cannot enforce.

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

### Target policy for strided-batched `dot_general`

The normative faer extraction policy above remains the migration-preservation
rule. Issue #1426 H8 owns a later PyTorch-aligned target policy and its actual
thresholds:

1. Prefer a provider-native strided-batched primitive when it satisfies the
   selected count, placement, and single-fan-out contracts.
2. For small matrices, parallelize the outer batch with a grain size roughly
   `GRAIN_SIZE / (m * n * k)` only when each inner kernel can be forced
   sequential independently on the executing worker. Eligible examples are
   faer, MKL local control, macOS 15 Accelerate, and intrinsically serial
   kernels. Parallel OpenBLAS is ineligible because its count control is
   process-global.
3. Providers without per-worker sequential enforcement cannot use outer
   fan-out. Large matrices use a sequential outer loop with parallel inner
   kernels.

PyTorch's `m * n * k < 400` cutoff and `GRAIN_SIZE = 32768` are starting
points, not accepted tenferro constants. The #1426 H8 child benchmarks
tensor-network shapes before selecting thresholds. Whether to retain a tiny
naive kernel is also a child decision. At every size there is exactly one
fan-out level; provider replacement cannot create nested outer and inner
parallelism.

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

`ExecutionHandle::drop` is non-blocking detach: it removes only that observer
and neither waits nor requests cancellation. `ExecutionHandle::cancel` is an
explicit best-effort request, and `wait` or `await` is the error-observing
completion boundary. Dropping an eager tensor or ordinary `Runtime` handle is
also non-blocking. `Runtime::shutdown` is the explicit API for callers that
want to stop admission and wait for all submitted work; default runtime drop
signals shutdown and hands remaining run records to the runtime driver or
external executor owner for draining.

An `InFlightRun` retains every input/output allocation, prepared operation,
resource lease, event token, external executor owner, and engine/device handle
that submitted work may access. These are released only after the completion
event, including failure or cancellation drain. Event-slot generations cannot
be recycled until that release. Thus detaching a handle cannot cause use after
free, while public handle and tensor destructors never synchronize a device.
The event/cancellation child must test dropped handles and runtimes with delayed
success, delayed failure, cancellation before enqueue, and cancellation after
device enqueue; completion errors from a detached run remain observable in
runtime diagnostics even though no handle receives them.

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

## MPI and Multi-Process Compatibility

The initial architecture provides no core MPI implementation and no
multi-host runtime. An MPI application creates one `Runtime` per rank and owns
communication, rank placement, communicator lifetime, and progress. Tenferro
must not require process-global mutable runtime, provider, topology, or
registration state, so independently configured rank processes remain valid.

In `Managed` mode, discovered topology is restricted to the process's actual
allowed CPUs, using `sched_getaffinity` and cgroup cpuset constraints where
available. The runtime must not construct domains from machine CPUs that the
rank cannot schedule on. User-specified domains are validated against that
allowed set before worker creation.

The host interop boundary includes an explicit contiguous mutable host
export/import suitable for calls such as `MPI_Allreduce`. Export is zero-copy
when compatible contiguous mutable host storage is already available;
otherwise the API reports and performs at most one stated staging copy.
Import similarly adopts or makes one explicit copy according to its ownership
contract. A future device/DLPack interop path uses the core transfer boundary,
and a future `CollectiveProvider` implements the reserved core collective node;
neither is required by the initial CPU refactor. MPI remains application-owned
until a collective child explicitly defines a provider, but a collective is
never encoded as an arbitrary extension operation.

`ExecutionPolicy::Reproducible` must produce identical provider selection,
partitioning, contraction paths, and planning decisions on ranks with the same
runtime snapshot, input signatures, and hardware class. This planning
guarantee does not claim bitwise cross-rank equality for providers that do not
declare it.

## Driving Workload: DMRG-Class Sweeps

DMRG-class workloads are a design driver rather than an application-specific
core API. Block-sparse semantics enter through an extension family and
block-sparse providers. During a sweep, matrix shapes and block structure
change often enough that common cache misses must keep preparation cheap; a
child must bound planning cost and cache footprint or define an explicit
polymorphic plan rather than assuming shape-stable reuse.

Prepared Davidson operations must support deliberate reuse of subspace,
workspace, and provider plans while preserving effects and invalidation rules.
At each user-managed MPI iteration boundary, tenferro adds no extra
synchronization, allocation, or admission beyond the applicable eager
single-operation contract. The application remains responsible for the MPI
collective and for making the resulting host mutation visible before the next
tenferro call.

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

Ownership follows component boundaries: `tenferro_runtime::program` initially
owns build, validation, extension-schema, and transform errors;
`tenferro-runtime` owns configuration, prepare, enqueue, completion, and
execution errors; operation crates own their planning and preparation errors.
If the program component is later extracted, those errors move with it.
`ExecutionError` is the sum of `Prepare`, `Enqueue`, and `Completion`;
compilation remains separate. A convenience trace-and-run API may expose an
outer `TraceRunError`.

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

- program-component property tests cover builders, validation, fingerprint
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
- native extensions and pure one-core-op lowerings remain eager-fast-path
  eligible, while two-or-more-op lowerings always promote to prepared graphs;
- eager fast-path latency satisfies the predeclared non-inferiority threshold
  against the current implementation for elementwise, reduction, contraction,
  and indexed calls, with no new eager-path allocation or string lookup;
- semantic extension AD preserves JVP and VJP results while removing all
  graph-specific rule-builder types from the new traits;
- placement-bound eager contexts apply the documented distinct CPU-affinity
  and GPU-device rules without an implicit copy or device transfer;
- mixed CPU NUMA affinities execute in the deterministic dominant-input domain,
  retain every input's allocation owner, and place outputs in the selected
  domain;
- `CpuAffinityPolicy::RequireSingleDomain` rejects mixed CPU affinities while
  the default policy accepts them;
- conflicting GPU devices still require an explicit transfer;
- transfers bridge source and destination event domains as first-class
  scheduled nodes, and collectives cannot be registered as arbitrary extension
  operations;
- count and placement capability are tested independently: a count-controlled
  budget-one external BLAS runs inline on a pinned domain thread, while strict
  exact-`CpuSet` placement plus a controlled external BLAS budget greater than
  one is rejected unless the domain is the complete process-allowed set;
  explicitly advisory placement is accepted only after count validation and
  remains visible in diagnostics; parallel OpenBLAS remains uncontrolled;
- OpenBLAS, MKL, Accelerate, ArmPL `_mp`, ArmPL serial, and NVPL construction
  probes classify thread-count and placement capability without making an
  unsupported thread-local or per-domain claim;
- extracting or replacing `CpuGemmProvider` preserves the normative grouped
  and strided-batched faer parallelization policy;
- #1426 target-policy tests select outer batching only for providers that can
  force a per-worker sequential inner kernel, prefer an eligible native batched
  primitive, select parallel inner kernels for large matrices, and detect any
  nested fan-out;
- `CpuGeneralContractionProvider` receives validated `dot_general` label groups,
  `DotGeneralRuntime` composition stays engine-owned, linalg family bundles stay
  extension-owned, and an execution error cannot trigger provider-specific
  fallback;
- extension execution performs no string lookup after prepare;
- extension native execution, core lowering, and explicit fallback obey the
  documented priority;
- generic extension specialization works without runtime knowledge of the
  extension's concrete Rust type;
- different concrete einsum shapes select distinct cached plans when their
  resolved contraction plans differ;
- phase 6 eager N-ary einsum over a predeclared TCI/DMRG-representative shape
  sequence that changes every call and misses exact-shape specializations does
  not regress against current `main` for the complete lower-prepare-execute
  boundary; a same-shape cache-hit case cannot satisfy this requirement;
- a runtime epoch change invalidates old prepared plans;
- CPU and GPU graphs use the same dependency, buffer lifetime, transfer, and
  barrier rules;
- input and shape-guard failure occurs before resource acquisition;
- execution failure never causes an implicit cross-engine retry;
- eager and graph paths apply the same provider-selection policy;
- internal CPU provider/composite operations reuse the outer lease without
  re-entering `CpuBackend`, while arbitrary nested backend entry retains the
  documented rejection;
- NUMA buffers remain associated with their allocation domain even when an
  operation executes from another CPU domain;
- `ExternalManaged` does not reconstruct a supplied executor;
- `ExternalManaged` retains the supplied executor lifetime, arbitrates exact
  declared CPU sets across multiple placement-resolved domains, never repins or
  claims live OS-affinity verification, and diagnoses caller-owned affinity;
- managed topology discovery never includes CPUs outside the
  `sched_getaffinity`/cgroup allowed set;
- contiguous mutable host export/import is zero-copy when eligible and otherwise
  reports no more than the specified single copy, including an in-place
  `MPI_Allreduce` integration test;
- separate rank-like runtimes require no process-global mutable state, and
  `Reproducible` produces identical planning decisions for identical snapshots;
- DMRG-like changing shapes and block structures meet a predeclared common-miss
  prepare budget, Davidson state reuse invalidates correctly, and the
  user-managed MPI boundary adds no synchronization, allocation, or admission
  beyond the eager single-operation contract;
- builder tokens reject cross-builder use, graph import and finish are atomic
  across values, roots, bindings, checkpoints, and metadata scopes, and
  extension construction passes the public source-contract check without raw
  representation access;
- GPU dependency tracking avoids unnecessary global synchronization;
- independent multi-GPU work can enqueue concurrently;
- foreign runtime tensors do not transfer implicitly;
- unsupported provider paths return explicit errors rather than fallback;
- same-key preparation is single-flight while different keys make progress;
- non-monotonic specialization requests fail as provider contract violations;
- atomic node lease requests never hold a partial resource set while queued;
- run-level failure stops new enqueue and drains already submitted operations;
- runtime/cache/run ownership has no strong cycle, public drops are
  non-blocking, and delayed completion retains resources and external owners;
- `Reproducible` rejects unsupported algorithms during prepare rather than
  silently using `Fast`.

Performance work must measure representative shapes, ranks, batch counts,
thread counts, NUMA placements, and device counts. Microbenchmarks must keep
validation, request construction, dispatch, provider call, and kernel work
separable.

The architecture does not require another fixed-overhead eager prototype.
When one is proposed independently, it follows the repository performance-gate
protocol and does not block provider, NUMA, program, scheduler, or device
phases.

## Consolidated Requirements from Prior Issues

Closing an exploratory issue does not discard the contract it established.
The following requirements are carried into the named architecture phases:

- **Closed #1432, phase 1:** `CpuGeneralContractionProvider` accepts a complete
  validated binary `dot_general` request, including TBLIS label groups;
  `DotGeneralRuntime` remains the engine-owned composite; linalg exposes
  extension-owned family bundles rather than an upstream facade; and phase 1
  may use temporary internal staging around current implementations. The
  engine alone owns batch fan-out and `ParallelMode`. Provider-specific
  fallback after an error is forbidden. The direct-dispatch and
  borrowed-request prototype evidence at `1b6223ce` remains part of the
  phase's performance baseline.
- **Closed #1417, phase 2:** `ExternalManaged` retains the external pool owner
  for all dependent work, never repins workers or claims live OS-affinity
  verification, and treats an inaccurate declared `CpuSet` as a caller
  placement contract violation rather than a memory-safety concern. Exact
  declared sets participate in domain arbitration; multiple external domains
  are registered and resolved by placement; diagnostics state the limits of
  external affinity, fairness, oversubscription, and shutdown guarantees.
- **Closed #1422, phase 3:** builder-issued graph and value tokens are opaque,
  never expose raw IDs, and reject cross-builder use with a typed error. Import
  is atomic across the complete graph/value mapping, bindings, roots,
  checkpoint, and metadata scopes. Finish is atomic for metadata and tensors
  and returns errors without requiring caller panic recovery. Extensions use
  only the supported builder with a source-contract check; raw representation
  fields stay private.
- **Open #1426 H8:** the strided-batched `dot_general` child owns target-policy
  thresholds and tiny-kernel decisions. Tactical fixes that do not change
  these contracts remain independent of the architecture migration.

## Migration Constraints

Phase definitions, dependency order, status, and acceptance gates live only in
the [umbrella issue](https://github.com/tensor4all/tenferro-rs/issues/1433).
This detailed design imposes the following migration constraints:

- tenferro is pre-1.0: public breaking changes are allowed and no deprecation
  period or compatibility shim is required. Migration is incremental only to
  keep each merged phase buildable, testable, and reviewable; temporary
  internal staging types are removed by the end of their owning phase;
- every phase is independently reviewable and may be split into smaller child
  issues without weakening the selected architectural invariants;
- normative specs and rendered parallelism documentation are updated as the
  corresponding behavior lands; and
- no child implementation is authorized by this detailed architecture alone.

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

### Create `tenferro-program` before the semantic model stabilizes

An early crate makes the intended dependency boundary visible, but also freezes
a public package while operation payloads, AD transforms, effects, aliases, and
builder contracts are still being migrated. Rejected for the first
implementation. The logical boundary starts as `tenferro_runtime::program` and
is extracted only when stability and a direct external-consumer dependency
justify it.

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
families. Rejected as the target architecture. A child may use it as an
internal staging boundary while replacing all in-repository callers, but it is
not preserved as a compatibility API.

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

- [Execution-engine and provider umbrella issue #1433](https://github.com/tensor4all/tenferro-rs/issues/1433)
- [External-managed NUMA requirements, issue #1417](https://github.com/tensor4all/tenferro-rs/issues/1417)
- [Semantic builder integrity requirements, issue #1422](https://github.com/tensor4all/tenferro-rs/issues/1422)
- [Batched parallelism policy child, issue #1426](https://github.com/tensor4all/tenferro-rs/issues/1426)
- [Provider-overhead prototype and measurements, issue #1432](https://github.com/tensor4all/tenferro-rs/issues/1432#issuecomment-5017993877)
- [Prototype branch at commit `1b6223ce`](https://github.com/tensor4all/tenferro-rs/commit/1b6223cee988af8e98a4a79d05d977024482573f)
- [JAX array migration](https://docs.jax.dev/en/latest/jax_array_migration.html)
- [JAX explicit sharding](https://docs.jax.dev/en/latest/notebooks/explicit-sharding.html)
- [PyTorch `DTensor`](https://docs.pytorch.org/docs/stable/distributed.tensor.html)
- [PyTorch tensor parallel APIs](https://docs.pytorch.org/docs/stable/distributed.tensor.parallel.html)
- [StableHLO specification](https://openxla.org/stablehlo/spec)
- [PJRT uniform device API](https://openxla.org/xla/pjrt)

## Child Design and Implementation Boundary

The [umbrella issue](https://github.com/tensor4all/tenferro-rs/issues/1433)
governs decomposition, ordering, status, and acceptance gates. This detailed
design selects the architectural invariants that children must preserve: crate
ownership, process-local portability, immutable public program access, typed
core capabilities, pure extensions, runtime identity and epochs, safe
prepared-operation dispatch, external CPU executors, runtime-owned events,
explicit buffer contracts, all-or-none multi-resource acquisition, finite
specialization projections, structural collision checks, no rollback of
effects, draining of submitted work, and the determinism policy.

Mechanism details are reserved for workload-informed child designs. These
include event-slot identifier and generation representation, slot recycling,
buffer-donation heuristics, exact dynamic-buffer reservation, the precise
two-level admission accounting algorithm, queue ordering or priority policy,
and the public cancellation state machine. Child designs must preserve the
selected invariants but may replace the illustrative representations used in
this document.

This detailed design is not authorization for one monolithic implementation
PR. Each umbrella phase must become a reviewable child issue with exact public
signatures, breaking-API and in-repository migration impact, benchmarks, and
acceptance tests. Later work may refine names and reserved mechanisms without
changing the selected ownership and behavioral invariants. Implementation
planning starts only after maintainers accept the relevant child design; the
umbrella itself is no longer the open review gate.
