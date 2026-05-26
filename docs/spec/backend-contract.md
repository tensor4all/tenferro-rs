# Backend Architecture

**Date:** 2026-04-04
**Repos:** tenferro-rs
**Related:** `../architecture/ad-pipeline.md`, `primitive-catalog.md`, `../reference/stablehlo-primitives.md`, `../reference/jax-primitives.md`

---

## I. Overview

All computation, primal and derivative, flows through the same execution
pipeline.

### Standard algebra path

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
  │   - run DotDimensionSorter + TransposeFolding
  ▼
ExecProgram
  │
  ├── eval_exec_ir_unsegmented()
  └── eval_exec_ir() / eval_exec_segmented()
         │
         ▼
  TensorBackend / TensorExec dispatch
```

### Custom algebra path

```text
MaterializedGraph
  │
  │ compile
  ▼
CompiledProgram<SemiringOp<Alg>>
  │
  │ compile_semiring_to_exec()
  ▼
ExecProgram
  │
  ▼
eval_semiring_ir()
  │
  ▼
SemiringBackend<Alg> + shared structural helpers
```

There is no in-process `StableHloProgram` / `StableHloOp` layer. The current
runtime contract is centered on `ExecProgram`.

---

## II. Execution IR

`ExecProgram` is the single in-process execution IR shared by standard and
custom algebra evaluation.

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
    pub last_use: Vec<bool>,
}
```

### Core guarantees

- Each instruction is SSA over slots: outputs are written once.
- `dtype` is the inferred output dtype for that instruction.
- `output_shapes` contains one symbolic shape per output slot.
- `last_use` is populated after lowering and is used for buffer reclamation.
- Multi-output linalg instructions write directly to multiple output slots.

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
- Contraction: `DotGeneral`, `NaryEinsum`
- Linalg: `Cholesky`, `Svd`, `Qr`, `Lu`, `Eigh`, `Eig`,
  `TriangularSolve`, `ValidateNonsingular`
- Constants: `Constant`

String `CustomCall` dispatch is gone. Structured linalg variants are first
class `ExecOp`s.

---

## III. Lowering Contract

### Standard lowering

`compile_std_to_exec()` consumes:

- `CompiledProgram<StdTensorOp>`
- input dtypes
- input shapes as `Vec<DimExpr>`

For each computegraph instruction it:

1. infers the output dtype with `infer_output_dtype()`
2. infers output shapes with `infer_output_shapes()`
3. lowers `StdTensorOp` to `ExecOp`
4. records output slot dtype/shape metadata
5. runs the compiler passes on the resulting `ExecProgram`
6. populates `last_use`

### Semiring lowering

`compile_semiring_to_exec()` consumes:

- `CompiledProgram<SemiringOp<Alg>>`
- input shapes as `Vec<DimExpr>`

The instruction dtype is derived from `Alg::Scalar`, and output shapes come
from semiring-aware shape inference. The resulting program goes through the
same pass and liveness pipeline.

### Current pass pipeline

The active optimizer passes are:

- `DotDimensionSorter`
- `TransposeFolding`

`DotDecomposer` is not part of the live compiler pipeline yet. The new
`ExecInstruction::output_shapes` metadata exists specifically to unblock that
work, which is tracked in `tensor4all/tenferro-rs#729`.

`ReductionSimplification` was deleted and is not part of the current backend
contract.

For pass algorithms and rationale, see
[`optimizer-passes.md`](optimizer-passes.md).

---

## IV. Dispatch Categories

Execution is divided into three instruction categories.

### Backend-session instructions

These run through `TensorExec` inside `TensorBackend::with_exec_session()`.
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
- `ValidateNonsingular`

`Constant` uses `TensorBackend::upload_host_tensor()` so device-specific
execution still receives correctly placed tensors without implicit transfer of
user-supplied inputs.

### FFI / boundary instructions

These stay as single-instruction boundaries in segmented execution:

- `DotGeneral`
- `NaryEinsum`
- `Cholesky`
- `Svd`
- `Qr`
- `Lu`
- `Eigh`
- `Eig`
- `TriangularSolve`

The standard path dispatches them through `TensorBackend`.

---

## V. Segmented vs. Unsegmented Execution

`eval_exec_ir_unsegmented()` evaluates one instruction at a time.

`eval_exec_ir()` delegates to segmented execution:

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

The engine uses `last_use` metadata to reclaim buffers via
`TensorExec::reclaim_buffer()` or `TensorBackend::reclaim_buffer()`.

---

## VI. Backend Traits

### TensorBackend

`TensorBackend` is the full standard-algebra backend surface in
`tenferro-internal-tensor/src/backend.rs`.

It includes:

- elementwise arithmetic and analytic ops
- structural ops
- reductions
- `dot_general`
- indexing ops
- linalg ops
- `with_exec_session`
- `download_to_host`
- `upload_host_tensor`
- `reclaim_buffer`

`TensorExec` is the session-local companion trait used by grouped backend
execution. Backends may override `with_exec_session()` to install one shared
execution scope, for example a CPU thread-pool context.

### SemiringBackend

`SemiringBackend<Alg>` is the custom-algebra execution surface.

`eval_semiring_ir()` currently accepts the semiring-compatible subset of
`ExecOp`:

- `DotGeneral`
- `Add`
- `Multiply`
- `ReduceSum`
- `Transpose`
- `Reshape`
- `BroadcastInDim`
- `ExtractDiag`
- `EmbedDiag`

Any other `ExecOp` in a semiring program is a compiler bug and currently
panics.

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

- `tenferro-runtime/src/compiler/mod.rs`
- `tenferro-runtime/src/shape_infer.rs`
- `tenferro-runtime/src/exec.rs`
- `tenferro-runtime/src/segment.rs`
- `tenferro-runtime/src/graph/executor.rs`
- `tenferro-internal-tensor/src/backend.rs`

Those files are the source of truth for the live backend contract. This
document is intentionally a high-level summary of that code.
