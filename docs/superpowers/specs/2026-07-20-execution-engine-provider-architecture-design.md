# WIP Design: Pluggable Execution Engines and Resource Domains

## Status

This is a work-in-progress architecture proposal. It records the agreed design
direction for discussion and issue decomposition; it is not an accepted public
API or an implementation plan.

The proposal returns tenferro to a prism-like dependency direction: operation
semantics are expressed as small traits, execution engines implement those
traits, and tenferro's eager and compiled surfaces call the selected engine.
Backend libraries remain replaceable behind smaller provider traits, while
resource ownership stays with the engine.

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
capability behavior implicit. The target is a set of explicit semantic traits,
explicit provider traits, and explicit runtime resources.

## Goals

1. Let the same semantic operation contract be implemented by CPU eager,
   CUDA/WebGPU eager, and XLA lowering engines.
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

The architecture has four layers:

```text
User API / graph compiler
          |
          v
Small semantic operation traits and prepared requests
          |
          v
Execution engine: policy, resources, scheduling, dependency tracking
          |
          v
Small backend-specific provider traits: algorithms only
```

The dependency direction is deliberate:

- tenferro core defines or depends on semantic operation contracts;
- an engine implements those contracts;
- a backend-specific engine delegates algorithms to providers;
- providers receive a per-execution context but do not own scheduling
  resources;
- extension crates add semantic and provider traits without requiring the core
  runtime to enumerate their operation families.

## Two Trait Layers

### Semantic operation traits

Semantic traits describe observable tensor behavior. They are shared across
execution styles and do not mention Rayon, BLAS, CUDA streams, or PJRT.

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

CPU, CUDA/WebGPU, and XLA engines share these semantic traits:

- a CPU engine executes immediately or schedules CPU work;
- a CUDA/WebGPU engine enqueues device work;
- an XLA engine lowers the same semantic request to compiler IR.

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
    gemm: Arc<dyn CpuGemmProvider>,
    layout: Arc<dyn CpuLayoutTransformProvider>,
    reduction: Arc<dyn CpuReductionProvider>,
    indexing: Arc<dyn CpuIndexingProvider>,
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
uses a resolved trait-object field or slot.

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

1. its semantic operation trait and request or prepared plan;
2. backend-specific provider traits for the backends it supports;
3. its `ExtensionOp` graph payload and lowering behavior;
4. a typed runtime adapter that connects its semantic trait to core extension
   dispatch.

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
    cpu: Option<Arc<CpuEngine>>,
    devices: DeviceRegistry,
    extensions: ExtensionRuntimeRegistry,
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
Missing capabilities return a typed error.

Fallback is permitted only through an explicit decorator or composite provider
configured by the user or standard engine builder. A provider must not catch an
unsupported error and silently move work to CPU, another device, a reference
implementation, or a full decomposition.

## Error Stages

Errors are separated by lifecycle stage:

1. **Build:** invalid topology, overlapping resource domains, incompatible
   provider configuration, or invalid external resources.
2. **Prepare:** unsupported semantic capability, unresolved provider or
   extension family, invalid validated request, or incompatible placement.
3. **Enqueue:** allocation failure, immediate kernel-launch error, or resource
   acquisition failure.
4. **Completion:** deferred device, transfer, collective, or asynchronous
   provider failure.

Public errors preserve these distinctions without exposing unstable provider
internals as public API.

## Testing and Performance Evidence

The implementation must provide a shared semantic conformance suite across at
least:

- CPU with faer providers;
- CPU with general BLAS/LAPACK providers where available;
- CUDA/CubeCL and custom CUDA providers;
- WebGPU providers;
- XLA lowering plus a reference execution path.

Required focused tests include:

- provider replacement preserves semantic results;
- request construction and resolved dispatch allocate nothing in steady state;
- extension execution performs no string lookup after prepare;
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

Migration is incremental and keeps the current backend path working behind
adapters:

1. Define validated semantic requests and small semantic traits.
2. Add adapters over the current `TensorBackend` execution path.
3. Introduce the CPU provider bundle and `CpuExecutionContext`.
4. Lift existing NUMA topology, arbiter, executor, buffer, and cache ownership
   into CPU resource domains.
5. Add `Managed` and `ExternalManaged` construction and validation.
6. Resolve extension families to typed adapters and steady-state slots.
7. Split CUDA/WebGPU device runtime resources from provider algorithms.
8. Introduce `ExecutionEvent` and dependency-aware resource reuse.
9. Add multi-GPU task scheduling for independent work.
10. Add logical sharding, collectives, and resharding only if accepted by a
    later design issue.

Each phase must be independently reviewable. Normative specs and online
parallelism documentation are updated as the corresponding behavior lands.

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

## Decisions Required Before Implementation Planning

The architecture direction is agreed, but an accepted umbrella issue should
still record maintainer decisions on these concrete contracts before code is
planned:

1. exact crate ownership and public visibility of semantic traits;
2. exact provider-family boundaries and naming;
3. request lifetime and prepared-plan types;
4. runtime and tensor identity representation;
5. executor abstraction required by `ExternalManaged`;
6. `ExecutionEvent` object-safety and error propagation;
7. transition rules from the current extension runtime registry;
8. which migration phases become separate accepted child issues.

Until those decisions are accepted, this branch is design evidence only and
must not be treated as authorization for a feature implementation PR.
