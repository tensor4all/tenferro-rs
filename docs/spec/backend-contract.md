# Backend Architecture

**Date:** 2026-05-28
**Repos:** tenferro-rs
**Related:** `../architecture/ad-pipeline.md`, `primitive-catalog.md`, `../reference/stablehlo-primitives.md`, `../reference/jax-primitives.md`

---

## I. Overview

All computation, primal and derivative, flows through the same execution
pipeline.

### Execution path

```text
TraceContext -> TracedGraph -> GraphCompiler
  │
  │ compile
  ▼
CompiledGraph
  │
  │ Runtime::prepare(program, inputs, targets)
  │   - validate ordered input metadata
  │   - bind one caller-held engine per target slot
  │   - lower semantic operations to private ExecProgram staging
  │   - build the private ScheduledGraph
  ▼
private runtime preparation and staging
  │
  │ Runtime::run(engines, prepared, inputs)
  │   - bind slot i to the i-th engine the caller passes
  │   - enqueue operations, transfers, and barriers through event domains
  │   - drain submitted work before returning execution failure
  ▼
private ScheduledGraph
  │
  ├── ScheduledNode::Operation
  ├── ScheduledNode::Transfer (endpoint-pair provider)
  └── ScheduledNode::Barrier / reserved collective nodes
```

There is no in-process `StableHloProgram` / `StableHloOp` layer. The current
public runtime contract is centered on `CompiledGraph` and runtime preparation.
`ExecProgram` and `ScheduledGraph` are private runtime artifacts: staging lowers
the immutable semantic graph, and scheduling binds storage endpoints, transfer
providers, and event-domain dependencies for one runtime snapshot.

The optional XLA path is a peer executor over a compiled `CompiledGraph`
lowering view, not a native backend. `tenferro-xla` may inspect immutable
lowering metadata, emit StableHLO, and load PJRT plugins at runtime, but it does
not implement `Engine` and does not bypass `Runtime::prepare` / `Runtime::run`
dispatch.

---

## II. Execution IR

`ExecProgram` is private runtime staging for standard tensor programs and
registered extension runtimes. It is not the public graph artifact and callers
must not construct or depend on it.

```rust
pub(crate) struct ExecProgram {
    pub instructions: Vec<ExecInstruction>,
    pub input_slots: Vec<usize>,
    pub output_slots: Vec<usize>,
    pub n_slots: usize,
    pub shape_guards: Vec<ShapeGuard>,
}

pub struct ExecInstruction {
    pub op: ExecOp,
    pub input_slots: Vec<usize>,
    pub output_slots: Vec<usize>,
    pub dtype: DType,
    pub output_shapes: Vec<Vec<DimExpr>>,
    pub output_extents: Vec<Vec<ShapeExtent<DimExpr>>>,
    pub last_use: Vec<bool>,
}
```

`ScheduledGraph` is the crate-private executable dependency DAG. It records
prepared operation nodes, first-class endpoint transfer nodes, value-slot
locations, event dependencies, and output bindings for one runtime snapshot.

### Core guarantees

- Each instruction is SSA over slots: outputs are written once.
- `dtype` is a representative instruction dtype used by legacy single-output
  paths. For multi-output extensions, per-output slot metadata is the
  authoritative dtype source.
- `output_shapes` contains one symbolic shape per output slot.
- `output_extents` contains one extent vector per output slot.
- `last_use` is populated after lowering and is used for buffer reclamation.
- `shape_guards` retains normalized symbolic shape obligations that could not
  be discharged from compile-time input metadata. Guard semantics are the
  relation and normalized left/right expressions; diagnostic source and
  instruction provenance are not semantic cache identity.
- Multi-output extension instructions write directly to multiple output slots
  and may use mixed output dtypes.

Before dispatching any backend, host, or extension instruction, execution must
evaluate every retained shape guard against the concrete program-input shapes
in stored order and return the first typed shape-constraint error on failure.
This validation is enforced immediately after execution-program input-count
validation and before segmentation, tensor allocation, backend-buffer
allocation, execution-output allocation, backend sessions, host work, or
extension dispatch. Validation may allocate small metadata vectors. The
input-count error therefore takes precedence over a guard expression's
`MissingInput` evaluation error.

The guarantee covers owned tensor and owned value-output execution, borrowed
and non-consuming reads, the `Runtime::prepare` / `Runtime::run` entry points,
segmented execution, extension-owned core-program execution through the
nonsegmented path, and programs with no instructions. Programs with no guards
retain their previous behavior. Compiler-cache hits reuse semantic plans while
restoring the current compilation's guard provenance, so executor failures
report the current extension family and final instruction index.

### Private staging vocabulary

The private staging operations keep StableHLO-aligned names where they remain
useful. They are lowered into scheduled operation nodes before execution; the
public contract is the semantic graph plus runtime preparation and scheduling:

- Elementwise: `Add`, `Multiply`, `Negate`, `Conj`, `Divide`, `Abs`, `Sign`,
  `Maximum`, `Minimum`, `Compare`, `Select`, `Clamp`, `Exp`, `Log`, `Sin`,
  `Cos`, `Tanh`, `Sqrt`, `Rsqrt`, `Pow`, `Expm1`, `Log1p`, `Erf`
- Structural: `Transpose`, `Reshape`, `BroadcastInDim`, `Convert`,
  `ExtractDiag`, `EmbedDiag`, `Tril`, `Triu`
- Reductions: `ReduceSum`, `ReduceProd`, `ReduceMax`, `ReduceMin`
- Indexing / shape: `Gather`, `GatherDynamicSliceSizes`, `Scatter`, `Slice`,
  `DynamicSlice`, `DynamicUpdateSlice`, `Pad`, `Concatenate`, `Reverse`, `ShapeOf`,
  `DynamicTruncate`, `PadToMatch`
- Contraction: `DotGeneral`
- Extension boundary: `Extension`
- Constants: `Constant`

String `CustomCall` dispatch is gone. Structured linalg variants are first
class extension operations owned by `tenferro-linalg`, not core `ExecOp`
variants.

---

## III. Lowering Contract

### Semantic operation staging

Runtime preparation extracts the frozen `SemanticProgram` from a
`CompiledGraph` and passes it to the private semantic-staging step, together
with compiler options and concrete input metadata:

- `SemanticProgram`
- input dtypes
- input shapes as `Vec<DimExpr>`

For each semantic instruction it:

1. reads semantic operations and infers output dtype, shape, and extent
   metadata; extension instructions use
   constraint-aware output-meta inference for one `(dtype, shape)` pair per
   output slot and collect local shape obligations
2. resolves output extents
3. lowers semantic operations to private `ExecOp` staging
4. records output slot dtype/shape/extent metadata
5. runs the compiler passes on the resulting `ExecProgram` and populates
   `last_use`
6. resolves retained constraint provenance against the final instruction
   stream, discharges compile-time-provable obligations, and stores the
   remaining normalized obligations in `shape_guards`

### Current pass pipeline

The active optimizer passes are:

- `DotDimensionSorter`
- `TransposeFolding`
- `DotDecomposer`
- `DeadCodeElimination`

`ReductionSimplification` was deleted and is not part of the current backend
contract.

For pass algorithms and rationale, see
[`optimizer-passes.md`](optimizer-passes.md).

---

## IV. Dispatch Categories

Execution is divided into three instruction categories.

### Engine instructions

These run through `Engine` on the caller-held engine the runtime was given.
They are the operations eligible for grouped segmented execution and, when
supported by the engine, elementwise fusion planning.

Examples:

- elementwise ops
- structural ops such as `Transpose`, `Reshape`, `BroadcastInDim`
- reductions such as `ReduceSum`, `ReduceProd`, `ReduceMax`, `ReduceMin`
- indexing ops such as `Gather`, `GatherDynamicSliceSizes`, `Scatter`, `Slice`,
  `DynamicSlice`, `DynamicUpdateSlice`, `Pad`, `Concatenate`, `Reverse`

The helper that executes one such instruction is `execute_backend_op()`.

### Host instructions

These are handled without calling backend kernels:

- `ShapeOf`
- `DynamicTruncate`
- `PadToMatch`
- `Constant`

`GatherDynamicSliceSizes` resolves its symbolic `slice_sizes` against concrete
runtime tensor shapes in the execution layer, then calls the backend through the
normal concrete `Gather` path.

`Constant` uses the engine's `upload_host_tensor()` so device-specific
execution still receives correctly placed tensors without implicit transfer of
user-supplied inputs.

### FFI / boundary instructions

These stay as single-instruction boundaries in segmented execution:

- `DotGeneral`
- `Extension`

`DotGeneral` dispatches through the engine. `Extension` dispatch routes
through the installed `ExtensionModule` for the operation family; linalg,
einsum, and FFT provide those modules from their owning crates.

---

## V. Segmented vs. Unsegmented Execution

`Runtime::prepare` and `Runtime::run` are the public execution entry points for
a `CompiledGraph`. Runtime preparation owns engine binding,
registered extension modules, cache ownership, and the private
`CompiledGraph -> ExecProgram staging -> ScheduledGraph` transition required to
preserve dispatch invariants. The runtime executes the prepared schedule one
node at a time and dispatches each operation through its selected engine bridge.
When a downstream operation uses a different execution endpoint, preparation
adds a `ScheduledNode::Transfer` with the registered endpoint-pair provider;
execution bridges the source completion into the destination event domain
before dispatch. Missing providers are runtime errors, not implicit host
fallbacks.

The segmented internal path groups fusible backend instructions:

```text
private ExecProgram staging
  │
  │ internal preparation/execution optimization
  ▼
segment_exec_program()
  │
  ├── fused backend-session segments
  ├── single-instruction FFI segments
  └── single-instruction host segments
```

Segmented execution exists to:

- reuse one backend execution session across consecutive backend ops
- enable elementwise fusion planning where the backend supports it
- preserve the same observable behavior as unsegmented execution

The unsegmented internal path evaluates one instruction at a time and is used
by the current scheduled runtime loop, parity checks, and narrow owner-scoped
extension-module composition. Neither this path nor the staging IR is a
general public execution surface.
Extension instructions must run through a registered `ExtensionModule`; missing
module registration is an error, not a fallback to a host/reference path.

The engine uses `last_use` metadata to reclaim buffers via
`Engine::reclaim_buffer()`.

---

## VI. Backend Traits

### Engine

`Engine` is the only execution trait, declared in
`crates/tenferro-tensor/src/backend.rs`. It includes:

- elementwise arithmetic and analytic ops
- structural ops
- reductions
- `dot_general`
- indexing ops
- `to_contiguous_read` and the caller-output `_into` forms
- `download_to_host` / `upload_host_tensor`
- `reclaim_buffer`

Each indexing and structural operation also has a `*_read` entry that takes
`TensorRead` operands. A backend whose kernel reads strided input overrides it so
a borrowed view is consumed in place; the CPU engine does this for every such
entry. A default that cannot do that passes an owned tensor straight to the
owned-input method and stages only a borrowed view through `to_contiguous_read`.
`tenferro_gpu::cuda::CudaEngine` overrides the entries whose catalogue kernel
addresses the operand's own layout (offset, strided and negative-stride
regions); the entries that still resolve a compact copy document that boundary
where they are defined.

Every method takes `&mut self`, so one computation uses one engine at a time and
the borrow is the exclusion contract. There is no session entry, no admission
permit, no nested-entry error and no shared execution scope: the caller owns the
engine and decides which operations share it.

`EngineRegistration` (tenferro-runtime) describes an engine *kind* to a
`Runtime`: its stable `EngineId`, its hardware and storage classes, its
preparation capabilities and its transfer endpoints. The registration holds no
engine; `Runtime::prepare` records which target slot needs which kind, and
`Runtime::run` binds each slot to the engine the caller passes. A CPU engine
registers through `tenferro_cpu::runtime_engine_registration`, and
`runtime_engine_registration_with_id` allows several CPU engines in one process.

`ExecutionEngine` (tenferro-runtime) extends `Engine` with what the runtime
needs from a concrete engine value: `target()` reports the `EngineTarget` the
engine runs as, and `take_extension_caches` / `restore_extension_caches` lend
the engine's extension caches to extension executors for the duration of a run.
`tenferro_cpu::CpuEngine` and `tenferro_gpu::cuda::CudaEngine` implement it.

A backend leaf that exposes native services returns an opaque
`NativeSessionRef` from `Engine::native_engine()`, recovered only by that leaf's
visitor; the default is `None`. The raw CUDA seam
(`CudaEngine::with_raw` / `with_cubecl`) is the sanctioned way to reach vendor
and CubeCL APIs on tenferro buffers.

Custom operation families do not add a second execution trait. They lower to
`ExecOp::Extension` and dispatch through their installed `ExtensionModule`.
The owning extension crate is responsible for deciding whether that runtime
uses the active engine, a provider-specific library, or an internal
implementation.

---

## VII. Layout and Device Contract

### Layout

All runtime tensors are dense contiguous column-major tensors. The backend
contract does not expose arbitrary stride-aware dispatch in the private
`ExecProgram` staging IR.

This means:

- private `ExecProgram` staging does not encode layout transforms as a separate
  concern
- backends receive dense tensors and can assume column-major layout
- compile-time shape reasoning is symbolic, but runtime storage layout is not

### Device transfer

tenferro does not perform implicit CPU<->GPU transfer for user-visible backend
ops. Tensors must already be on the correct device for the backend call.

The execution engine handles only two internal conveniences:

- `Constant` can be auto-uploaded through `upload_host_tensor()`
- host-only metadata/scalar operations can inspect or materialize tiny host
  values as part of execution

Unsupported backend operations must return an error rather than silently
falling back across devices.

### Placement

`CompiledGraph` and private `ExecProgram` staging are placement-agnostic.
Runtime preparation resolves placement on `Tensor` values, binds operation
endpoints in `ScheduledGraph`, and records the event-domain dependencies needed
for execution.

---

## VIII. Relation to StableHLO

StableHLO is now a reference vocabulary, not an in-process IR layer.

What remains true:

- many `StdTensorOp` / `ExecOp` names intentionally align with StableHLO
- StableHLO and JAX documentation are still useful semantic references
- future external serialization could target StableHLO if the project adds
  such a backend later

What is no longer true:

- there is no `StableHloProgram`
- there is no `StableHloOp`
- there is no `lower_to_stablehlo()` step in the live execution pipeline
- there is no `GetTupleElement`-style tuple indexing in runtime IR

---

## IX. File Ownership

The current implementation is split across:

- `crates/tenferro-runtime/src/compiler/mod.rs`
- `crates/tenferro-runtime/src/compiler/semantic_staging.rs`
- `crates/tenferro-runtime/src/shape_infer.rs`
- `crates/tenferro-runtime/src/exec.rs`
- `crates/tenferro-runtime/src/segment.rs`
- `crates/tenferro-runtime/src/runtime/preparation.rs`
- `crates/tenferro-runtime/src/runtime/schedule.rs`
- `crates/tenferro-runtime/src/runtime/execution.rs`
- `crates/tenferro-runtime/src/runtime/snapshot.rs`
- `crates/tenferro-runtime/src/graph/program.rs`
- `crates/tenferro-tensor/src/backend.rs`

Those files are the source of truth for the live backend contract. This
document is intentionally a high-level summary of that code.
