# tenferro-rs Internal Design

**Repo:** tenferro-rs
**Parent:** `../index.md`
**Related:** `computegraph.md`, `chainrules.md`, `tidu.md`, `../spec/backend-contract.md`, `../spec/primitive-catalog.md`

---

## I. Purpose

This document records the graph-based crate architecture that replaced the
early eager/tape split. The current workspace has no root `tenferro` facade:
runtime graph APIs live in `tenferro-runtime`, AD APIs in `tenferro-ad`,
runtime values in `tenferro-tensor`, GPU execution in `tenferro-gpu`, and
operation families in crates such as `tenferro-einsum`, `tenferro-linalg`, and
`tenferro-fft`.

---

## II. Architecture Migration

### Removed crates

The previous architecture organized around eager execution families and tape-based AD. These were replaced by the graph + fragment model:

| Previous crate | Current | Reason |
|---|---|---|
| `internal/ad-core` | deleted | Fragment replaces tape |
| `internal/ad-ops` | → `tenferro-internal-ops` PrimitiveOp impl | AD rules live on TensorOp |
| `internal/ad-linalg` | → `tenferro-linalg` | Optional extension AD rules |
| `internal/ad-surface` | → tidu-rs `differentiate`/`transpose` | External crate |
| `internal/frontend-core` | → `tenferro-runtime` TracedTensor | Lazy, not eager |
| `internal/runtime` | → `tenferro-runtime` graph compiler/executor | |
| `tenferro-dynamic-compute` | deleted | Always graph |
| `tenferro-internal-tensor-compute` | → `tenferro-internal-ops` | |
| `tenferro-linalg-prims` | → `tenferro-linalg` | Linalg is an extension family |
| `tenferro-capi` | deferred | Phase 4+ |
| `extension/*` | deferred | |

### Retained crates

| Previous crate | Current crate | Notes |
|---|---|---|
| `tenferro-internal-device` | `tenferro-internal-device` | Mostly unchanged |
| `tenferro-algebra` | `tenferro-algebra` | Mostly unchanged |
| `tenferro-internal-tensor` | `tenferro-tensor` | Public dense tensor and CPU backend crate |
| `tenferro-prims` | `tenferro-internal-ops` | Rewritten: single TensorOp enum |
| `tenferro-einsum` | `tenferro-einsum` | Rewritten: graph builder |
| `tenferro-linalg` | `tenferro-linalg` | Primal extension runtime plus optional `autodiff` linalg rules |
| `tenferro` (facade) | deleted | Direct crates are the public API boundary |

The current split is intentionally direct rather than facade-based. See
`AGENTS.md` and `REPOSITORY_RULES.md` for the authoritative dependency graph.

---

## III. Crate Dependency Graph

```text
chainrules      -> chainrules-core
tidu            -> chainrules-core

tenferro-tensor -> tenferro-internal-device
tenferro-gpu    -> tenferro-tensor

tenferro-internal-ops -> tenferro-tensor, tidu, computegraph-rs
tenferro-runtime      -> tenferro-internal-ops, tenferro-tensor

tenferro-ad           -> tenferro-runtime, tenferro-internal-ops, tidu
tenferro-einsum       -> tenferro-runtime, tenferro-internal-ops
tenferro-linalg       -> tenferro-runtime, tenferro-internal-ops, tenferro-tensor
tenferro-linalg/cuda  -> tenferro-gpu
tenferro-fft          -> tenferro-runtime, tenferro-internal-ops
```

---

## IV. Two Op Types

The fundamental design constraint is that `GraphOp::Operand` is an associated
type, so a single Op type can only serve one `Operand` type. Since standard
algebra (`Tensor`) and custom algebras (`TropicalTensor`, etc.) have
different `Operand` types, tenferro provides two Op types:

### StdTensorOp — standard algebra, full vocabulary, AD-capable

`StdTensorOp` is a **flat** enum whose core primitive variants mostly mirror
StableHLO ops 1:1, with `StdTensorOp::Extension` carrying operation-family
payloads such as linalg, einsum, and FFT. It implements `GraphOp` (only — not
`EvalGraphOp`) and `SemiringOps`. There is no `GraphOp::eval`; all execution
flows through the backend pipeline.

Canonical definition: [`spec/primitive-catalog.md`](../spec/primitive-catalog.md) (Section IV -- Tenferro IR Vocabulary).
AD trait (`PrimitiveOp`): [`spec/ad-contract.md`](../spec/ad-contract.md).

### SemiringOp\<T\> — custom algebra, semiring subset, no AD

`SemiringOp<T>` is a generic wrapper around `SemiringOpKind` that implements
`GraphOp` only (not `EvalGraphOp`). It delegates algebraic ops to free
functions in `host_ops` (dispatched through `TensorBackend`) and structural
ops to algebra-independent free functions for structural operations.
`PrimitiveOp` is **not** implemented -- no AD for custom algebras.

Canonical definition: [`spec/primitive-catalog.md`](../spec/primitive-catalog.md) (Section IV).

Users extend tenferro by implementing `SemiringBackend<Alg>` (algebraic ops +
kernel dispatch) for their algebra type, then use `SemiringOp<Alg>` as the op
type. Structural ops (`transpose`, `reshape`, `broadcast_in_dim`) are provided
automatically by the execution engine.

---

## V. SemiringOpKind — Shared Vocabulary

`SemiringOpKind` is the set of operations that all algebras must support.
It is used **only** inside `SemiringOp<T>` — the generic custom-algebra op
type. `StdTensorOp` does **not** wrap `SemiringOpKind`; it has its own flat
variants that mostly mirror StableHLO 1:1 (with documented exceptions for
composite lowerings and extension carriers).

`SemiringOpKind` is the minimal set of ops all algebras must support:
`Add`, `Mul`, `DotGeneral`, `ReduceSum`, `Transpose`, `Reshape`,
`BroadcastInDim`.

Canonical definition: [`spec/primitive-catalog.md`](../spec/primitive-catalog.md) (Section IV -- AD-closed graph core + structural ops).

`SemiringOp<T>` wraps it as a newtype. The `SemiringOps` trait bridges both
worlds: `StdTensorOp` implements it by mapping to flat variants,
`SemiringOp<T>` implements it by mapping to `SemiringOpKind` variants.

---

## VI. SemiringOps Trait — Generic Einsum

`SemiringOps` bridges both `StdTensorOp` and `SemiringOp<T>` so that einsum
Fragment construction is algebra-agnostic. Both op types implement it.

Canonical definition: [`spec/primitive-catalog.md`](../spec/primitive-catalog.md).

Einsum is algebra-agnostic:

```rust
fn build_einsum_fragment<Op: SemiringOps>(
    builder: &mut FragmentBuilder<Op>,
    path: &ContractionPath,
    inputs: &[ValRef<Op>],
) -> LocalValId {
    // Constructs DotGeneral, Transpose, Reshape, etc. nodes
    // Does not know which algebra is in use
}
```

The contraction path optimization is also algebra-agnostic (it only depends
on shapes and subscripts):

```rust
fn optimize_contraction_path(
    subscripts: &Subscripts,
    shapes: &[&[usize]],
) -> ContractionPath;
```

---

## VII. Einsum: N-ary to Graph

N-ary einsum is decomposed into a graph of binary operations:

```text
einsum("ij,jk,kl->il", A, B, C)
    |
    | optimize_contraction_path (shape-based, algebra-agnostic)
    v
ContractionPath: [(A,B) -> T, (T,C) -> result]
    |
    | build_einsum_fragment<Op: SemiringOps>
    v
Fragment<Op>:
    t0 = DotGeneral(A, B, {contract=[j]})    // "ij,jk->ik"
    t1 = DotGeneral(t0, C, {contract=[k]})   // "ik,kl->il"
```

Each binary contraction step may insert `Transpose`, `Reshape`, or
`BroadcastInDim` nodes as needed to align axes for `DotGeneral`.

For standard algebra, the resulting `Fragment<StdTensorOp>` can be
differentiated and transposed by tidu-rs. For custom algebras,
`Fragment<SemiringOp<T>>` goes directly to `materialize_merge -> compile ->
eval`.

---

## VIII. Backend Architecture — Single Execution IR

### Design principle

All execution flows through a single in-process execution IR:

```text
CompiledProgram<StdTensorOp>
    │
    │ compile_std_to_exec()
    │   - infer per-instruction dtype + output_shapes
    │   - run DotDimensionSorter + TransposeFolding
    ↓
ExecProgram / ExecOp / ExecInstruction
    │
    ├── eval_exec_ir() / segmented dispatch → TensorBackend
    └── eval_semiring_ir()                  → SemiringBackend + shared structural helpers
```

`compile_std_to_exec()` and `compile_semiring_to_exec()` lower directly from
computegraph's `CompiledProgram` into `ExecProgram`. There is no in-process
`StableHloProgram` / `StableHloOp` layer anymore. StableHLO remains a useful
semantic reference for naming and op design, but it is not part of the current
runtime pipeline.

For custom algebras (`SemiringOp<T>`), the same direct lowering applies:
`compile_semiring_to_exec()` produces `ExecProgram`, and
`eval_semiring_ir()` dispatches algebra-dependent ops through
`SemiringBackend<Alg>` while reusing the same structural helpers and pass
pipeline.

### Execution IR

`ExecInstruction` carries the `ExecOp`, slot wiring, inferred `dtype`,
inferred `output_shapes`, and liveness metadata (`last_use`). The compiler
lowers core `StdTensorOp` variants and `SemiringOpKind` almost 1:1 into
`ExecOp`; extension families lower to `ExecOp::Extension` and dispatch through
registered extension runtimes instead of stringly typed custom calls.

### Pass pipeline

The optimizing compiler now runs directly on `ExecProgram`. The active passes
are `DotDimensionSorter` and `TransposeFolding`; `DotDecomposer` is deferred
to issue `#729` now that per-instruction shape tracking is available.

Canonical `ExecOp` definition and pass list:
[`spec/backend-contract.md`](../spec/backend-contract.md).
Pass algorithms: [`spec/optimizer-passes.md`](../spec/optimizer-passes.md).

### Generic execution engine

The generic engine interprets `ExecProgram` by dispatching each instruction
to `TensorBackend` methods, host helpers, extension runtimes, or common
infrastructure, depending on the dispatch category.

*(illustrative, non-normative -- see [`spec/backend-contract.md`](../spec/backend-contract.md) for canonical definition)*

### Backend trait

`TensorBackend` (defined in `tenferro-tensor`) is the single backend trait
that encapsulates standard tensor kernel dispatch. `BackendSession` is the
session-scoped companion trait used for batches of backend ops inside one
execution context. `CpuBackend` lives in `tenferro-tensor` and implements
`TensorBackend`; the CubeCL GPU backend lives in `tenferro-gpu` and is
feature-gated.

Canonical trait signatures: [`spec/backend-contract.md`](../spec/backend-contract.md).

### Standard and custom algebra backends

The standard backend path is `CompiledProgram<StdTensorOp> ->
compile_std_to_exec() -> ExecProgram -> eval_exec_ir()`. Custom algebra
backends implement `SemiringBackend<Alg>` and follow the analogous
`compile_semiring_to_exec() -> eval_semiring_ir()` path.

`GraphCompiler` lowers traced outputs into backend-independent
`GraphProgram`s. `GraphExecutor<B: TensorBackend>` owns backend runtime state
and executes compiled programs. `TensorBackend` (in `tenferro-tensor`) is the
kernel-level trait that backend authors implement to provide kernels.

See [`spec/backend-contract.md`](../spec/backend-contract.md) for the
canonical trait signatures.

### Split graph compilation and backend execution

```rust
struct GraphCompiler {
    compile_cache: CompileCache,
    static_einsum_cache: EinsumCache,
}

struct GraphExecutor<B: TensorBackend> {
    backend: B,
    runtime_einsum_cache: EinsumCache,
    backend_cache: B::RuntimeCache,
}
```

For custom algebras, users construct their own evaluation pipeline:

```rust
let path = optimize_contraction_path(&subscripts, &shapes);
let fragment = build_einsum_fragment::<TropicalOp>(&mut builder, &path, &inputs);
let view = resolve(vec![fragment]);
let graph = materialize_merge(&view, &outputs);
let prog = compile(&graph);

// Choose backend
let mut backend = TropicalGpuBackend::new(cuda_ctx);
let result = backend.eval_program(&prog, &input_tensors);
```

---

## IX. TracedTensor, GraphCompiler, and GraphExecutor

`TracedTensor` is the user-facing lazy type for standard algebra:

```rust
struct TracedTensor {
    id: TracedTensorId,
    rank: usize,
    dtype: DType,
    fragment: Arc<Fragment<StdTensorOp>>,
    val: LocalValId,
    data: Option<Tensor>,
    // ... internal fields omitted
}
```

Key operations:

```rust
impl TracedTensor {
    /// Create from concrete data
    fn from(tensor: Tensor) -> Self;

    /// VJP: differentiate → transpose (via tidu-rs), still lazy
    fn grad(&self, wrt: &TracedTensor) -> TracedTensor;

    /// JVP: differentiate only (via tidu-rs), still lazy
    fn jvp(&self, wrt: &TracedTensor, tangent: &TracedTensor) -> TracedTensor;
}

/// Compile one or more lazy outputs into a reusable program.
fn compile_many(&mut self, outputs: &[&TracedTensor]) -> Result<GraphProgram>;

/// Run one compiled program on a backend.
fn run_many(&mut self, program: &GraphProgram) -> Result<Vec<Tensor>>;
```

`GraphCompiler::compile_many` is the recommended API when primal outputs and
their derivatives are needed together. The compiled `GraphProgram` can then be
reused through a `GraphExecutor`.

For custom algebras, users work with `Fragment<SemiringOp<T>>` and
`CompiledProgram<SemiringOp<T>>` directly through the computegraph-rs API,
without `TracedTensor`.

---

## X. User Extension Points

| Goal | What to implement |
|---|---|
| New scalar algebra for einsum (CPU) | `Semiring` (4 methods) + `SemiringBackend<Alg>::gemm` (1 method) |
| Custom GPU backend for custom algebra | `impl SemiringBackend<Alg> for MyGpuBackend` (gemm + overrides) |
| Custom standard backend | `impl TensorBackend for MyBackend` |
| AD for custom algebra | Define own Op enum, impl `PrimitiveOp` (advanced) |

The minimal extension path (CPU, e.g., tropical semiring):

1. Define algebra type, impl `Semiring` — `zero()`, `one()`, `add()`, `mul()`
2. `impl SemiringBackend<MyAlgebra> for CpuBackend` — only `gemm()` required
   (e.g., call `tropical_gemm`). `batched_gemm`, `add`, `mul`, `reduce_sum`
   have defaults using strided-kernel + `Semiring` trait.
3. Use `SemiringOp<MyAlgebra>` as the Op type — einsum + compile + eval work
   immediately via `eval_semiring_ir`.

Adding a GPU backend for custom algebra:

1. `impl SemiringBackend<MyAlgebra> for MyGpuBackend` — provide `gemm()` with
   GPU kernels. Override `batched_gemm`, `add`, `mul`, `reduce_sum` if
   optimized GPU versions exist.
2. Use the same `CompiledProgram<SemiringOp<MyAlgebra>>` — graph construction
   and compilation are backend-agnostic.

---

## XI. Backend Traits

The `Operand` trait has been **removed** from computegraph-rs entirely.

### TensorBackend -- standard algebra, full op set

`TensorBackend` (defined in `tenferro-tensor`) covers all ops for standard
algebra. Operates on `Tensor` (type-erased). `CpuBackend` implements this trait.
`tenferro-gpu::CubeclBackend` implements the same trait when the `cuda`
feature is enabled.

Canonical definition: [`spec/backend-contract.md`](../spec/backend-contract.md).

### SemiringBackend\<Alg: Semiring\> -- custom algebra, semiring ops only

`SemiringBackend<Alg>` (defined in `tenferro-tensor`) covers semiring ops for
custom algebra. Operates on `TypedTensor<Alg::Scalar>` (typed). User
provides only `gemm()` (single GEMM); `batched_gemm`, `add`, `mul`,
`reduce_sum` have default implementations using strided-kernel + `Semiring`
trait methods.

The two traits are independent (no supertrait relationship).

Canonical definition: [`spec/backend-contract.md`](../spec/backend-contract.md).

### Structural ops -- algebra-independent free functions

`transpose`, `reshape`, `broadcast_in_dim`, `extract_diagonal`,
`embed_diagonal` are free functions in `tenferro_tensor::cpu::structural`.
They are the same for all algebras. For standard algebra, `TensorBackend`
methods delegate to these. For custom algebra, `eval_semiring_ir` calls them
directly.

---

## XII. Per-Crate Contents

### tenferro-internal-device

Defines the placement vocabulary and shared runtime errors. `Placement`
contains `memory_kind` plus `resident_device`, while `ComputeDevice` remains a
separate notion for execution. Public memory kinds follow JAX/XLA-style names:
`Device`, `PinnedHost`, `UnpinnedHost`, and `Other(String)`.

### tenferro-algebra

Provides `SemiringAlgebra` trait, `StandardAlgebra`, scalar type
constraints.

### tenferro-tensor

Tensor runtime crate. No AD-related code and no GPU runtime dependency.

- `types.rs` — `TypedTensor<T>` (contiguous-only, no strides), `Tensor` enum,
  `Buffer<T>`, `DType`, `Placement`, `MemoryKind`, `ComputeDevice`
- `config.rs` — `DotGeneralConfig`, `CompareDir`, `GatherConfig`, `ScatterConfig`,
  `SliceConfig`, `PadConfig` (moved from tenferro-internal-ops to avoid dependency cycle)
- `backend.rs` — `TensorBackend` trait, `SemiringBackend<Alg>` trait
- `cpu/` — CPU backend:
  - `backend.rs` — `CpuBackend: impl TensorBackend`
  - `elementwise.rs` — strided-kernel: add, mul, neg, conj, div, abs, exp, log, ...
  - `reduction.rs` — strided-kernel: reduce\_sum, reduce\_prod, reduce\_max, reduce\_min
  - `structural.rs` — strided-kernel: transpose, broadcast\_in\_dim, extract\_diagonal;
    dedicated: reshape (metadata only), embed\_diagonal
  - `indexing.rs` — gather, scatter, slice, pad, concatenate, reverse
  - `gemm/` — `faer_gemm.rs` (cpu-faer), `blas_gemm.rs` (cpu-blas)
  - `linalg/` — `faer_linalg.rs` (cpu-faer), `lapack_linalg.rs` (cpu-blas)

**No naive CPU loop fallbacks.** All CPU kernels use strided-kernel (elementwise,
reduction, structural), faer or BLAS (GEMM), faer or LAPACK (linalg). At least
one of `cpu-faer` or `cpu-blas` must be enabled. The features are additive, and
`CpuBackend` selects the compiled provider at runtime.

### tenferro-gpu

GPU backend implementation crate. It depends on `tenferro-tensor` and exports
GPU backend/runtime types plus explicit upload/download helpers. It does not
re-export tensor types as its public API surface.

- `cubecl/` — CubeCL/CUDA backend, runtime, memory transfer, GEMM support, FFI
  handles, and kernel dispatch when the `cuda` feature is enabled
- `kernels/` — CubeCL kernel launch helpers
- `CubeclBackend` — `TensorBackend` implementation for CUDA-capable devices

ROCm remains a feature placeholder.

### tenferro-internal-ops

The core crate:

- `SemiringOpKind` enum (shared vocabulary, used only in `SemiringOp<T>`)
- `SemiringOps` trait
- `SemiringOp<T>` generic wrapper + `impl GraphOp` (graph construction only,
  no eval — execution is dispatched through `TensorBackend`)
- `StdTensorOp` enum — **flat**, with core primitive variants and an extension
  carrier for linalg, einsum, FFT, and other operation families
- `impl GraphOp for StdTensorOp` (graph construction only, no eval)
- `impl PrimitiveOp for StdTensorOp` (linearize + transpose_rule)
- `impl SemiringOps for StdTensorOp` — maps to flat variants directly
- `TensorInputKey` + `impl ADKey`

Depends on: computegraph-rs, chainrules-rs, `tenferro-tensor`.

### tenferro-einsum

Graph builder for N-ary einsum:

- `Subscripts` parsing and validation
- `ContractionPath` optimization
- `build_einsum_fragment<Op: SemiringOps>` (algebra-agnostic)

Depends on: computegraph-rs, tenferro-internal-ops.

### tenferro-runtime

Traced graph runtime and extension dispatch crate:

- `TracedTensor` (lazy graph-aware wrapper)
- `GraphCompiler` (compilation cache and static einsum planning cache)
- `GraphExecutor` (backend dispatch via `TensorBackend`, runtime extension
  cache, backend runtime cache)
- `compile_std_to_exec()` (`CompiledProgram<StdTensorOp>` -> `ExecProgram`)
- `compile_semiring_to_exec()` (`CompiledProgram<SemiringOp<T>>` -> `ExecProgram`)
- Optimizing compiler passes on `ExecProgram`
  - `DotDimensionSorter`
  - `TransposeFolding`
  - `DotDecomposer` deferred to issue `#729`
- `ExecProgram`, `ExecOp`, `ExecInstruction`
- Generic execution engine: `eval_exec_ir()` — interprets `ExecProgram`,
  dispatches to `TensorBackend` methods
- Generic semiring execution engine: `eval_semiring_ir()` — interprets
  `ExecProgram`, dispatches to `SemiringBackend<Alg>` plus shared structural
  helpers

Depends on: computegraph-rs, `tidu-rs` integration points,
`tenferro-internal-ops`, and `tenferro-tensor`.

### tenferro-ad

Explicit automatic-differentiation crate:

- traced AD extension methods such as `grad()` and `jvp()`
- eager AD runtime and `EagerTensor`
- AD metadata helpers used by standard extension companion crates

Depends on: `tenferro-runtime`, `tenferro-internal-ops` with `autodiff`, and
`tenferro-tensor`.

---

## XIII. Implementation Status

Phases 1–3 (scalar fragment AD, tensor primitives + einsum, linalg + backends)
are implemented and tested. Current work focuses on:

- Custom algebra end-to-end (tropical semiring)
- GPU backend expansion (CUDA kernels)
- C-API (FFI for Julia/Python)
- Logical-DAG-aware checkpoint scheduling
- Operator fusion in compiled IR
