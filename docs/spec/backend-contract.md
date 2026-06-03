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
MaterializedGraph
  │
  │ compile
  ▼
CompiledProgram<StdTensorOp>
  │
  │ compile_std_to_exec()
  │   - lower StdTensorOp -> ExecOp
  │   - infer dtype and output_shapes
  │   - run DotDimensionSorter + TransposeFolding + DotDecomposer
  │   - run DeadCodeElimination
  ▼
ExecProgram
  │
  │ GraphExecutor::eval_exec_ir()
  │   - owns backend runtime cache
  │   - owns extension runtime registry/cache
  │   - routes through segmented dispatch
  ▼
internal exec dispatch
  │
  ├── unsegmented core-only path
  └── segmented path
         │
         ▼
  TensorBackend / BackendSession dispatch
```

There is no in-process `StableHloProgram` / `StableHloOp` layer. The current
runtime contract is centered on `ExecProgram`.

---

## II. Execution IR

`ExecProgram` is the single in-process execution IR for standard tensor
programs and registered extension runtimes.

```rust
pub struct ExecProgram {
    pub instructions: Vec<ExecInstruction>,
    pub input_slots: Vec<usize>,
    pub output_slots: Vec<usize>,
    pub n_slots: usize,
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

### Core guarantees

- Each instruction is SSA over slots: outputs are written once.
- `dtype` is a representative instruction dtype used by legacy single-output
  paths. For multi-output extensions, per-output slot metadata is the
  authoritative dtype source.
- `output_shapes` contains one symbolic shape per output slot.
- `output_extents` contains one extent vector per output slot.
- `last_use` is populated after lowering and is used for buffer reclamation.
- Multi-output extension instructions write directly to multiple output slots
  and may use mixed output dtypes.

### ExecOp vocabulary

The execution IR keeps StableHLO-aligned naming where it remains useful, but
the ops are the real runtime contract:

- Elementwise: `Add`, `Multiply`, `Negate`, `Conj`, `Divide`, `Abs`, `Sign`,
  `Maximum`, `Minimum`, `Compare`, `Select`, `Clamp`, `Exp`, `Log`, `Sin`,
  `Cos`, `Tanh`, `Sqrt`, `Rsqrt`, `Pow`, `Expm1`, `Log1p`
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

### StdTensorOp lowering

`compile_std_to_exec()` consumes:

- `CompiledProgram<StdTensorOp>`
- input dtypes
- input shapes as `Vec<DimExpr>`

For each computegraph instruction it:

1. infers output dtype, shape, and extent metadata; extension instructions use
   `infer_extension_output_meta()` for one `(dtype, shape)` pair per output
   slot
2. resolves output extents
3. lowers `StdTensorOp` to `ExecOp`
4. records output slot dtype/shape/extent metadata
5. runs the compiler passes on the resulting `ExecProgram`
6. populates `last_use`

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

### Backend-session instructions

These run through `BackendSession` inside `TensorBackend::with_backend_session()`.
They are the operations eligible for grouped segmented execution and, when
supported by the backend, elementwise fusion planning.

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

`Constant` uses `TensorBackend::upload_host_tensor()` so device-specific
execution still receives correctly placed tensors without implicit transfer of
user-supplied inputs.

### FFI / boundary instructions

These stay as single-instruction boundaries in segmented execution:

- `DotGeneral`
- `Extension`

`DotGeneral` dispatches through `TensorBackend`. `Extension` dispatch routes
through the registered `ExtensionRuntime` for the operation family; linalg,
einsum, and FFT register those runtimes from their owning crates.

---

## V. Segmented vs. Unsegmented Execution

`GraphExecutor::eval_exec_ir()` is the public execution entry point for an
`ExecProgram`. It carries the backend cache, extension runtime registry, and
extension runtime cache required to preserve dispatch invariants.

The segmented internal path groups fusible backend instructions:

```text
ExecProgram
  │
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
for parity checks and narrow owner-scoped extension-runtime composition. It is
not a general public execution surface. Extension instructions must run through
a registered `ExtensionRuntime`; missing runtime registration is an error, not
a fallback to `ExtensionOp::eager_execute()`.

The engine uses `last_use` metadata to reclaim buffers via
`BackendSession::reclaim_buffer()` or `TensorBackend::reclaim_buffer()`.

---

## VI. Backend Traits

### TensorBackend

`TensorBackend` is the full standard-algebra backend surface in
`crates/tenferro-tensor/src/backend.rs`.

It includes:

- elementwise arithmetic and analytic ops
- structural ops
- reductions
- `dot_general`
- indexing ops
- `with_backend_session`
- `download_to_host`
- `upload_host_tensor`
- `reclaim_buffer`

`BackendSession` is the session-local companion trait used by grouped backend
execution. Backends may override `with_backend_session()` to install one shared
execution scope, for example a CPU thread-pool context.

Custom operation families do not add a second backend trait. They lower to
`ExecOp::Extension` and dispatch through their registered `ExtensionRuntime`.
The owning extension crate is responsible for deciding whether that runtime
uses the active `TensorBackend`, a provider-specific library, or an internal
implementation.

---

## VII. Layout and Device Contract

### Layout

All runtime tensors are dense contiguous column-major tensors. The backend
contract does not expose arbitrary stride-aware dispatch in `ExecProgram`.

This means:

- `ExecProgram` does not encode layout transforms as a separate concern
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

`ExecProgram` is placement-agnostic. Device placement lives on runtime
`Tensor` values, not in the compiled IR.

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
- `crates/tenferro-runtime/src/shape_infer.rs`
- `crates/tenferro-runtime/src/exec.rs`
- `crates/tenferro-runtime/src/segment.rs`
- `crates/tenferro-runtime/src/graph/executor.rs`
- `crates/tenferro-tensor/src/backend.rs`

Those files are the source of truth for the live backend contract. This
document is intentionally a high-level summary of that code.
