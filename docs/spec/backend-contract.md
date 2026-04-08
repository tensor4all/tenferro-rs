# v2 Backend Architecture

**Date:** 2026-04-04
**Status:** Draft
**Repos:** tenferro-rs
**Related:** `../architecture/ad-pipeline.md`, `primitive-catalog.md`, `../reference/stablehlo-primitives.md`, `../reference/jax-primitives.md`

---

## I. Overview

**All computation — primal and derivatives — flows through the same pipeline:**

```
MaterializedGraph (after resolve + materialize_merge)
  │
  │ compile
  ▼
CompiledProgram (flat SSA of StableHLO IR)
  │
  │                     ┌─── single cut point ───┐
  │                     │                        │
  ├── XLA backend ──────┤  takes StableHLO IR    │
  │   (serialize to     │  directly              │
  │    MLIR, JIT)       └────────────────────────┘
  │
  └── All other backends
        │
        │ optimizing compiler (TransposeFolding, DotDecomposer,
        │                      LinalgCustomCallPassthrough)
        ▼
      Execution IR (StableHLO ops + BatchedGemm - DotGeneral)
        │               all tensors are contiguous column-major
        │
        │ generic execution engine
        │   ALL ops dispatched through TensorBackend trait
        │
        ├── CpuBackend (CPU, default; lives in tenferro-tensor)
        │     implements TensorBackend
        │     (faer GEMM is an internal optimization within CpuBackend)
        ├── CudaBackend (GPU, op-by-op, dynamic shapes; lives in tenferro-tensor)
        │     implements TensorBackend
        └── Custom algebra backends (Tropical, etc.)
              implement TensorBackend
```

AD is a graph transformation (differentiate, transpose), not a separate
execution mode. AD-side transforms may remain fragmented until a final
`materialize_merge` produces the compile-time `MaterializedGraph`. A
primal-only computation follows the same path — just without AD transforms.
Even eager execution (`apply_op`-equivalent) is internally a single-node
materialized graph compiled and evaluated through this pipeline.

### 2-level IR architecture

The **StableHLO IR** is the single cut point between graph/AD and
all backends. For **standard algebra**, it contains the full Tenferro IR
vocabulary (`StdTensorOp` variants, most mapping 1:1 to a StableHLO op;
exceptions include composite lowerings like `Conj` and multi-output linalg
ops like `Svd`). This IR is serializable to StableHLO MLIR; XLA takes it
directly. For **custom algebra**, the same `StableHloOp` types are used but
ops have semiring-specific semantics (Add=⊕, Mul=⊗) — this IR is **not**
serializable to StableHLO MLIR and always goes through the optimizing
compiler → Execution IR path. The StableHLO program is
**layout-independent** -- input layout normalization is a pure runtime
concern handled by the execution engine, not an IR transformation.

The **Execution IR** is the output of the optimizing compiler. **All tensors
are contiguous column-major** — there is no stride-aware dispatch needed.
Its op vocabulary is the **same as StableHLO, with one
substitution**: `DotGeneral` is replaced by `BatchedGemm` (produced by
DotDecomposer). All other ops -- elementwise, reductions, indexing,
structural, and `CustomCall` -- pass through from StableHLO unchanged.
The key optimization is DotGeneral decomposition; everything else is
executed as-is.

```
                  StableHLO IR
                  (the single cut point)
                          │
            ┌─────────────┴──────────────┐
            ▼                            ▼
       XLA backend                optimizing compiler
       (direct)                        │
                                       ▼
                                 Execution IR
                                       │
                              generic execution engine
                              ┌────────┼────────┐
                              ▼        ▼        ▼
                        CpuBackend CudaBackend custom algebra
```

### Backend comparison

| | CpuBackend | CudaBackend | XLA | Custom algebra |
|---|---|---|---|---|
| Input IR | Execution IR | Execution IR | StableHLO (direct) | Execution IR |
| Execution | op-by-op interpret | op-by-op interpret | JIT compile | op-by-op interpret |
| Dynamic shapes | yes | yes | no (recompile) | yes |
| Kernel fusion | none | none | yes | none |
| Tensor networks | yes | **primary target** | needs padding | yes |
| Dependencies | faer (Rust) | CUDA kernels | xla-rs (~200MB) | user kernels |

**CudaBackend** is the key backend for tensor network computations where
bond dimensions change dynamically. It interprets Execution IR op-by-op,
dispatching to CUDA kernels (reused from tenferro-rs v1). No JIT
compilation, so no recompile on shape change. Individual ops (matmul, SVD)
are large enough that fusion is not critical.

**XLA** is for workloads with stable shapes (ML training loops, fixed
batch size) where kernel fusion provides significant speedup. It takes
StableHLO IR directly (no optimizing compiler pass).

### Custom algebra backends

| Algebra | CPU | GPU |
|---------|-----|-----|
| Tropical | TensorBackend → custom kernels | TensorBackend → optimized CUDA kernels |

Custom algebras receive the same Execution IR as standard backends. They
implement the `TensorBackend` trait. This means a new algebra backend
needs only one trait implementation to support any einsum-derived program.

---

## II. Primitive Vocabulary

The Tenferro IR op vocabulary, per-op semantics, StableHLO lowering rules,
and frontend sugar are owned by [`primitive-catalog.md`](primitive-catalog.md).

Execution IR dispatch categories, the `TensorBackend` trait signature,
and the generic execution engine are owned by **this document**
(Sections V--VI below).

---

## III. StableHLO Format and Tooling

### Serialization format

StableHLO uses **MLIR bytecode** as its portable format. 5-month forward/backward
compatibility guarantee. Also available as MLIR textual assembly (`.mlir`).

### Consuming StableHLO

| Approach | Dependencies | Notes |
|----------|-------------|-------|
| Full MLIR + StableHLO C API | LLVM/MLIR + StableHLO libs | Official. Heavy build (~75k LOC C++) |
| `melior` crate (Rust MLIR bindings) | LLVM/MLIR | Extend with StableHLO C API via FFI |
| Text `.mlir` output → xla-rs | xla-rs only | Let XLA parse. Simplest for XLA backend |
| Own Rust IR (no MLIR) | None | Map ~100 ops directly. Lightest. |

**Our approach**: CompiledProgram is our own Rust IR (StableHLO IR).
For non-XLA backends (faer, custom GPU, custom algebra), the optimizing
compiler lowers to Execution IR and the generic engine interprets it — no
MLIR needed. For the XLA backend, we emit `.mlir` text or use the xla-rs
builder API. Full MLIR dependency is avoided.

StableHLO's reference interpreter (~8.7k LOC) is a useful reference for
implementing our CpuBackend.

---

## IV. StableHLO Lowering

### Principle: mostly 1:1 mapping

Most `StdTensorOp` variants map to exactly one StableHLO op. `StdTensorOp`
is flat — no `Semiring(SemiringOpKind)` wrapping — which keeps lowering
trivial. Documented exceptions include composite lowerings (e.g., `Conj` → 4
ops) and multi-output linalg ops (e.g., `Svd` → `custom_call` +
`get_tuple_element` × N).

Per-op lowering rules are owned by
[`primitive-catalog.md`](primitive-catalog.md) (Section VI).

### Type support

| Rust type | StableHLO type | Supported |
|-----------|---------------|-----------|
| `f32` | `f32` | ✅ |
| `f64` | `f64` | ✅ |
| `Complex<f32>` | `complex<f32>` | ✅ |
| `Complex<f64>` | `complex<f64>` | ✅ |
| `Tropical<f64>` | — | ❌ (StableHLO path unavailable; use custom backend on semiring-compatible subset) |

### Linalg lowering

See [`primitive-catalog.md` Section V](primitive-catalog.md) for the linalg
primitive table and StableHLO lowering rules. Key point: `Cholesky` is a
direct StableHLO op; all others (`SVD`, `QR`, `Eigh`, `Solve`) lower to
`stablehlo.custom_call`.

---

## V. Backend Dispatch

### Architecture: 2-level IR compilation

All programs first compile to a **CompiledProgram** containing
StableHLO IR. This is the single cut point between graph/AD
and all backends. From here, execution diverges:

Terminology: **Tier 1** = AD-closed graph core (`primitive-catalog.md` IV),
**Tier 2** = Standard arithmetic only (`primitive-catalog.md` V).

```
Graph → differentiate / transpose → merge → compile
    │
    ▼
CompiledProgram<Op>  (StableHLO IR — the single cut point)
    │
    ├── XLA backend (takes StableHLO IR directly)
    │     serialize to MLIR → JIT compile → LLVM/PTX → fused kernels
    │
    └── All other backends
          │
          │  optimizing compiler (algebra-agnostic passes)
          │    TransposeFolding  — fold Transpose into DotGeneral dim numbers
          │    DotDecomposer     — break multi-dim DotGeneral → BatchedGemm
          │    LinalgCustomCallPassthrough — pass linalg custom_call ops through
          │
          ▼
        Execution IR (all tensors contiguous column-major)
          │  StableHLO ops + BatchedGemm - DotGeneral
          │
          │  Engine<B: TensorBackend>
          │    ALL ops dispatched through TensorBackend trait
          │
          ├── CpuBackend (Standard CPU, default; lives in tenferro-tensor)
          │     TensorBackend impl (faer GEMM is internal optimization)
          │     Linalg custom_call → LAPACK routines
          │
          ├── CudaBackend (Standard GPU, dynamic shapes; lives in tenferro-tensor)
          │     TensorBackend impl → CUDA kernels (reused from v1)
          │
          └── Custom algebra backends (Tropical, p-adic, etc.)
                TensorBackend impl → user-provided kernels
                (semiring-compatible subset of Tier 1 only)
```

**Custom algebra programs may only use the semiring-compatible subset of
Tier 1.** Using Tier 2 ops (`Exp`, `SVD`, etc.) with a custom algebra is a
compile-time error, and even some Tier-1 ops (for example `Neg` or `Conj`)
may be unavailable when the algebra does not define them.

### Optimizing compiler

The optimizing compiler transforms StableHLO IR into Execution IR through
a sequence of algebra-agnostic passes:

| Pass | Purpose |
|------|---------|
| **TransposeFolding** | Fold chains of `Transpose` + `DotGeneral` into a single instruction with permuted dimension numbers |
| **DotDecomposer** | Break multi-contracting-dim `DotGeneral` into sequences that map to `BatchedGemm` |
| **LinalgCustomCallPassthrough** | Pass linalg `CustomCall` ops through to the Execution IR as-is |

Note: input contiguity checking is **not** a compiler pass. It happens at
eval() time as a runtime pre-processing step (see Memory layout section
below). All tensors are ensured to be contiguous column-major before
entering the Execution IR engine. No stride-aware dispatch is needed.

These passes operate on shape metadata and instruction structure, not on
element values. They are shared by all non-XLA backends (faer, custom GPU,
custom algebra).

### Execution IR

The output of the optimizing compiler is a flat sequence of Execution IR
instructions. **All tensors are contiguous column-major.** There is no
stride-aware dispatch -- the engine does not need to inspect strides or
set BLAS trans flags. This simplifies backend implementations significantly.

The Execution IR uses the **same op vocabulary as the StableHLO
IR**, with one substitution: `DotGeneral` is replaced by `BatchedGemm`
(produced by DotDecomposer). All other ops pass through from StableHLO
unchanged. The key optimization is DotGeneral decomposition; everything
else is executed as-is.

| Category | Ops | Engine dispatch |
|----------|-----|----------------|
| Contraction | `BatchedGemm` | `TensorBackend::dot_general()` |
| Reduction | `ReduceSum` | `TensorBackend::reduce_sum()` |
| Elementwise arithmetic | `Add`, `Mul`, `Neg`, `Conj` | `TensorBackend::add()`, `mul()`, `neg()`, `conj()` |
| Structural | `Reshape`, `Transpose`, `BroadcastInDim` | `TensorBackend::reshape()`, `transpose()`, `broadcast_in_dim()` |
| Diagonal | `ExtractDiagonal`, `EmbedDiagonal` | `TensorBackend::extract_diagonal()`, `embed_diagonal()` |
| Elementwise analytic | `Exp`, `Log`, `Sin`, `Cos`, `Tanh`, `Sqrt`, `Rsqrt`, `Pow`, `Expm1`, `Log1p` | TensorBackend (future methods) |
| Comparison & selection | `Compare`, `Select`, `Clamp` | TensorBackend (future methods) |
| Additional reductions | `ReduceProd`, `ReduceMax`, `ReduceMin` | TensorBackend (future methods) |
| Indexing | `Gather`, `Scatter`, `Slice`, `DynamicSlice`, `Pad`, `Concatenate`, `Reverse` | TensorBackend (future methods) |
| Linalg / extensibility | `Cholesky`, `CustomCall` | Kernel registry |

**All ops are dispatched through the single `TensorBackend` trait.** There
is no separate structural/standard/indexing dispatch -- everything goes
through the backend. Note that `Copy` is not an Execution IR instruction;
inputs are ensured to be contiguous column-major at eval() pre-processing
before IR entry.

### Backend traits

Two backend traits, both defined in `tenferro-tensor`:

#### TensorBackend — standard algebra, full op set

Operates on `Tensor` (type-erased). Covers **all** ops including
non-semiring operations (analytic, indexing, linalg). Used by the standard
algebra execution path (`eval_exec_ir<B: TensorBackend>`).

| Method | Execution IR instruction(s) covered |
|--------|----------------------------------|
| `add` | `Add` |
| `mul` | `Mul` |
| `neg` | `Neg` |
| `conj` | `Conj` |
| `reduce_sum` | `ReduceSum` |
| `dot_general` | `BatchedGemm` (lowered from `DotGeneral`) |
| `broadcast_in_dim` | `BroadcastInDim` |
| `reshape` | `Reshape` |
| `transpose` | `Transpose` / `Permute` |
| `extract_diagonal` | `ExtractDiagonal` |
| `embed_diagonal` | `EmbedDiagonal` |
| `div`, `abs`, `sign`, ... | Standard elementwise |
| `exp`, `log`, `sin`, ... | Analytic |
| `gather`, `scatter`, ... | Indexing |
| `reduce_prod`, `reduce_max`, ... | Additional reductions |
| `cholesky`, `svd`, `qr`, ... | Linalg |

#### SemiringBackend\<Alg: Semiring\> — custom algebra, semiring ops only

Operates on `TypedTensor<Alg::Scalar>` (typed). Used by the custom algebra
execution path (`eval_semiring_ir<Alg, B: SemiringBackend<Alg>>`).

**Required method (user implements):**

| Method | Description |
|--------|-------------|
| `gemm` | Single GEMM: C\[i,j\] = ⊕\_k (A\[i,k\] ⊗ B\[k,j\]) |

**Default methods (strided-kernel + Semiring trait):**

| Method | Implementation |
|--------|---------------|
| `batched_gemm` | Batch loop over `self.gemm()` |
| `add` | `strided-kernel::zip_map2_into` with `Alg::add` |
| `mul` | `strided-kernel::zip_map2_into` with `Alg::mul` |
| `reduce_sum` | `strided-kernel::reduce` with `Alg::add` |

**Structural ops** (transpose, reshape, broadcast\_in\_dim, extract\_diagonal,
embed\_diagonal) are **not** on this trait. They are algebra-independent free
functions in `tenferro-tensor::cpu::structural`, shared by both execution paths.

The two traits are **independent** (no supertrait relationship).
`TensorBackend` is not parameterized by algebra; it handles all standard
scalar types (f32/f64/c32/c64) via runtime dtype dispatch inside the `Tensor`
enum.

### Generic execution engine

The generic execution engine (`Engine<B: TensorBackend>`) is a simple
interpreter that walks the Execution IR instruction sequence and dispatches
**every** instruction through the `TensorBackend` trait. There is no
separate dispatch for structural vs elementwise vs semiring ops --
everything goes through the single backend trait.

All tensors are contiguous column-major, so the engine does not need to
inspect strides or handle layout variations.

For the full dispatch table, see the table above and
[`primitive-catalog.md`](primitive-catalog.md#iii3-execution-ir).

### Buffer lifecycle: liveness analysis + buffer pool

The Execution IR is SSA (each slot is written once), but the execution engine
must manage Rust buffer ownership efficiently. The key mechanism is
**liveness analysis**: the compiler annotates each instruction input with
whether it is the **last use** of that slot.

```rust
struct ExecInstruction {
    op: ExecOp,
    inputs: Vec<usize>,     // input slot indices
    outputs: Vec<usize>,    // output slot indices
    last_use: Vec<bool>,    // last_use[i] = true if inputs[i] is consumed here
}
```

At execution time, the engine uses this annotation to decide consume vs borrow:

- **`last_use = true`**: the engine calls `slots[i].take()` (Rust ownership
  transfer). The operation can reuse the input buffer for its output if the
  layout is compatible, or return the buffer to a **buffer pool** for later
  reuse.
- **`last_use = false`**: the engine borrows `slots[i].as_ref()`. The buffer
  stays alive for downstream consumers.

```rust
for inst in &program.instructions {
    let input = if inst.last_use[0] {
        slots[inst.inputs[0]].take().unwrap()   // consume: buffer reusable
    } else {
        slots[inst.inputs[0]].as_ref().unwrap() // borrow: buffer stays
    };
    // ... execute op, possibly reusing input buffer ...
}
```

**Buffer pool**: freed buffers are returned to a per-engine pool
(`TensorBufferPool` in v1) and recycled for future allocations. This avoids
repeated heap allocation in tight loops (e.g., einsum over many contraction
steps).

The IR semantics remain purely functional (SSA). Consume/borrow decisions are
an execution engine implementation detail, invisible to the IR and backend
traits.

### Type erasure boundary

| Layer | Scalar type T | Algebra Alg |
|-------|--------------|-------------|
| Graph + AD (tidu) | generic (via `Operand`) | generic (via `PrimitiveOp`) |
| CompiledProgram (StableHLO IR) | generic (preserved) | generic (preserved) |
| Execution IR + custom backend | generic (preserved) | custom (preserved) |
| XLA backend | erased to DType enum | Standard only |

Type erasure happens **only at the XLA/StableHLO serialization boundary**.

### CpuBackend (default for Standard CPU)

`CpuBackend` lives in `tenferro-tensor` and implements `TensorBackend`.
It interprets Execution IR instruction-by-instruction, dispatching each op
through its `TensorBackend` implementation. Internally, `CpuBackend` uses
faer GEMM as an optimization for `dot_general`, but this is an internal
detail -- not a separate trait. Linalg `CustomCall` instructions are
dispatched to LAPACK routines via a registered kernel registry. No XLA
dependency.

```
Execution IR instruction  →   CpuBackend TensorBackend dispatch
  BatchedGemm             →   dot_general() → faer::mat_mul / dgemm
  ReduceSum               →   reduce_sum()
  Add, Mul, Neg, Conj     →   add(), mul(), neg(), conj()
  Reshape, Transpose,     →   reshape(), transpose(), broadcast_in_dim()
    BroadcastInDim
  ExtractDiagonal,        →   extract_diagonal(), embed_diagonal()
    EmbedDiagonal
  Exp, Log, Sin, ...      →   TensorBackend methods (libm / faer analytic)
  Compare, Select, ...    →   TensorBackend methods
  ReduceProd, ReduceMax,  →   TensorBackend methods
    ReduceMin
  Gather, Scatter, ...    →   TensorBackend methods

CustomCall (Execution IR) →   LAPACK kernel registry dispatch
  Cholesky                →   dpotrf
  SVD                     →   dgesvd
  QR                      →   dgeqrf + dorgqr (or dungqr for complex)
  Eigh                    →   dsyevd
  Solve                   →   dgetrf + dgetrs
```

No fusion, no JIT. Op-by-op execution. Sufficient for most CPU workloads.

**Why CpuBackend (faer) is the default, not XLA CPU**: XLA's CPU backend (via
elixir-nx/xla) uses oneDNN only on x86_64 Linux. All other platforms
fall back to Eigen (unoptimized). faer has optimized kernels everywhere:

| Platform | XLA CPU backend | faer |
|----------|----------------|------|
| x86_64 Linux | oneDNN (fast) | fast (AVX/AVX-512) |
| aarch64 Linux | Eigen (slow) | fast (NEON/SVE) |
| aarch64 macOS | Eigen (slow) | fast (NEON) |
| x86_64 macOS | Eigen (slow) | fast (AVX) |

XLA is optional, primarily for GPU. Its CPU path is useful only for
fusion optimization, not raw kernel performance.

### Memory layout: contiguous column-major

**All tensors are contiguous column-major.** This is a fundamental
simplification over v1. At eval() time, input pre-processing ensures all
inputs are contiguous column-major before entering the Execution IR engine.
No stride-aware dispatch is needed -- the engine and all `TensorBackend`
implementations can assume contiguous column-major layout.

```
User Tensor:      always contiguous column-major
                          │
               eval() pre-processing:
                 ensure contiguous column-major
                          │
                   program execution
                          │
                   StableHLO IR (layout-independent)
                          │
                   optimizing compiler (TransposeFolding, DotDecomposer, etc.)
                          │
                   Execution IR → Engine<B: TensorBackend> dispatch
                          │
                   all intermediates/outputs: contiguous column-major
```

**XLA backend input contract**: XLA accepts only dense contiguous buffers
(no stride concept). Since all tensors are already contiguous column-major,
the XLA backend can upload directly with zero host-side copy.

**Contract**: the final output of any backend is always a dense contiguous
column-major `Tensor` with some runtime `Placement`. Internal intermediates
may use backend-specific layout, but this is invisible to the caller.

### Device management

CompiledProgram is **placement-agnostic** — it contains no `Placement`
information. Runtime placement is carried by `Tensor`.

```rust
struct Placement {
    memory_kind: MemoryKind,
    resident_device: Option<ComputeDevice>,
}
```

`resident_device` tells us which device owns or primarily addresses the memory.
`preferred_compute_device` is still a separate runtime execution hint on
`Tensor`.

**Phase 1: all inputs must share a resident device domain.**

The backend asserts this at `eval` time (not compile time, since
CompiledProgram has no placement info):

```rust
fn eval(&self, prog: &CompiledProgram, inputs: &[Tensor]) -> Vec<Tensor> {
    let resident = inputs[0].resident_device();
    for input in inputs {
        assert_eq!(
            input.resident_device(),
            resident,
            "all inputs must share the same resident device"
        );
    }
    // backend may normalize memory kinds internally if needed
}
```

**Cross-device execution** requires splitting into multiple programs:

```rust
let result_gpu = gpu_backend.eval(&prog1, &[a_gpu, b_gpu]);
let result_cpu = result_gpu.to_cpu();                          // explicit transfer to UnpinnedHost
let result2 = cpu_backend.eval(&prog2, &[result_cpu, c_cpu]);
```

**Phase 2 (future): Transfer op in CompiledProgram.**

CompiledProgram may include `Transfer(tensor, target_placement)` as a primitive
op. When lowering to StableHLO, Transfer ops split the program into chunks:

```
CompiledProgram:
  [op0, op1, Transfer(Device@cuda:0 → UnpinnedHost), op2, op3,
   Transfer(UnpinnedHost → Device@cuda:0), op4]

StableHLO lowering:
  chunk 1: [op0, op1]     → GPU backend
  transfer: Device@cuda:0 → UnpinnedHost
  chunk 2: [op2, op3]     → CPU backend
  transfer: UnpinnedHost → Device@cuda:0
  chunk 3: [op4]          → GPU backend
```

Transfer is NOT a StableHLO op — it exists only in CompiledProgram.
tenferro handles the insertion (manual or automatic placement planning)
and the splitting during StableHLO lowering.

tidu never knows about placement. It is purely symbolic.

### Backends live inside tenferro-tensor

`CpuBackend` and `CudaBackend` live in `tenferro-tensor`, the tensor
runtime crate. The `TensorBackend` trait is also defined there. All
backends use tenferro-tensor's placement and transfer infrastructure
(allocation, placement-aware transfer, kernel dispatch). They cannot be
separated into standalone crates without duplicating that runtime layer:

```
tenferro-tensor/       Tensor, TensorBackend trait, placement model, transfer helpers
  ├── backend_cpu/     CpuBackend (TensorBackend impl, faer GEMM internal, LAPACK dispatch)
  ├── backend_gpu/     CudaBackend (TensorBackend impl, CUDA kernels)
  └── ...

tenferro-prims/        PrimitiveOp implementations
tenferro-einsum/       High-level einsum, IR, compiler, engine
  ├── ir/
  │   ├── stablehlo/   StableHLO IR types + lowering
  │   ├── exec/        Execution IR types (StableHLO ops + BatchedGemm - DotGeneral)
  │   └── compiler/    Optimizing compiler passes
  │                      TransposeFolding, DotDecomposer, LinalgCustomCallPassthrough
  └── engine/          Engine<B: TensorBackend> (walks Execution IR)

backend_xla/           XLA backend (takes StableHLO directly, optional feature flag)
```

### Custom algebra backend (Tropical / custom algebra)

Custom algebra backends receive **Execution IR** (after the optimizing
compiler), not StableHLO IR directly. They implement `SemiringBackend<Alg>`
(canonical signature in Section V above).

The custom algebra execution engine dispatches algebra-dependent ops through
`SemiringBackend<Alg>` and structural ops through shared free functions:

```rust
fn eval_semiring_ir<Alg: Semiring, B: SemiringBackend<Alg>>(
    backend: &mut B,
    program: &ExecProgram,
    inputs: Vec<TypedTensor<Alg::Scalar>>,
) -> Vec<TypedTensor<Alg::Scalar>> {
    for inst in &program.instructions {
        let result = match &inst.op {
            // Algebra-dependent → SemiringBackend<Alg>
            ExecOp::BatchedGemm(config) => backend.batched_gemm(lhs, rhs, config),
            ExecOp::Add => backend.add(lhs, rhs),
            ExecOp::Multiply => backend.mul(lhs, rhs),
            ExecOp::ReduceSum { axes } => backend.reduce_sum(input, axes),

            // Algebra-independent → shared free functions
            ExecOp::Permute { perm } => structural::transpose(input, perm),
            ExecOp::Reshape { shape } => structural::reshape(input, shape),
            ExecOp::BroadcastInDim { shape, dims } => structural::broadcast_in_dim(input, shape, dims),
            ExecOp::ExtractDiag { axis_a, axis_b } => structural::extract_diagonal(input, axis_a, axis_b),
            ExecOp::EmbedDiag { axis_a, axis_b } => structural::embed_diagonal(input, axis_a, axis_b),

            // Standard-only ops → error (not valid in semiring IR)
            _ => panic!("non-semiring op in semiring IR"),
        };
        slots[inst.output_slots[0]] = Some(result);
    }
}
```

A custom algebra user implements `Semiring` (4 methods: zero, one, add, mul)
and `SemiringBackend<Alg>::gemm` (1 method: single GEMM). All other ops
have defaults (strided-kernel for elementwise, batch loop for batched\_gemm,
shared free functions for structural). The same compilation pipeline
(TransposeFolding, DotDecomposer, etc.) optimizes the ExecIR before
execution, so custom algebra benefits from the same optimizations as
standard algebra.

The standard algebra execution engine (`eval_exec_ir<B: TensorBackend>`)
dispatches **all** ops (including structural) through `TensorBackend`.

---

## VI. GPU Backend: XLA first, IREE later

### Strategy: XLA → IREE migration

Both XLA and IREE accept StableHLO as input. We use **XLA first** (mature,
prebuilt binaries available) and migrate to **IREE later** (future runtime).

```
CompiledProgram → StableHLO → XLA (Phase 3) → IREE (Phase 4+)
                           ↑                ↑
                           available now     future replacement
```

StableHLO is the stable interface. Backend swap is transparent.

### Phase 3: XLA via xla-rs

**Prebuilt binaries**: `elixir-nx/xla` distributes precompiled XLA shared
libraries (~200MB) for all major platforms:

| Binary | Target |
|--------|--------|
| `xla_extension-*-x86_64-linux-gnu-cpu.tar.gz` | Linux x86_64 CPU |
| `xla_extension-*-aarch64-darwin-cpu.tar.gz` | macOS Apple Silicon |
| `xla_extension-*-x86_64-linux-gnu-cuda12.tar.gz` | Linux CUDA 12 |
| `xla_extension-*-x86_64-linux-gnu-cuda13.tar.gz` | Linux CUDA 13 |

**Rust bindings**: `xla` crate (xla-rs, by Laurent Mazare / Hugging Face)
wraps XLA's PjRt C API. Uses elixir-nx/xla binaries directly.

```rust
use xla::{PjRtClient, XlaBuilder};

let client = PjRtClient::gpu()?;       // or ::cpu()
let builder = XlaBuilder::new("my_program");
// ... build HLO from CompiledProgram ...
let executable = client.compile(&computation)?;
let result = executable.execute::<xla::Literal>(&[input_buffers])?;
```

### XLA's JIT compilation design

XLA does NOT bundle precompiled CUDA kernels. It **JIT-compiles at runtime**:

```
HLO → LLVM IR → PTX → cubin (at runtime)
```

CUDA libraries (`libcudart`, `libcublas`, `libcudnn`) are loaded via `dlopen`
from the user's system installation. This keeps the XLA binary compact (~200MB
vs libtorch's ~2GB).

| | libtorch CUDA | XLA CUDA |
|---|---|---|
| Size | ~2GB | ~200MB |
| CUDA kernels | precompiled, bundled | JIT at runtime |
| CUDA libraries | partially bundled | `dlopen` from system |

### Two-level caching

Graph construction and AD transforms are expensive. Caching happens at
two levels, both outside tidu:

```
Level 1 — CompiledProgram (user / tenferro responsibility):
  Expensive:  Graph → differentiate → merge → compile → CompiledProgram
  Cheap:      retain CompiledProgram, call eval() many times
  tidu does NOT cache. Caller retains the CompiledProgram.

Level 2 — XLA executable (tenferro-xla-backend):
  Expensive:  CompiledProgram → StableHLO → XLA compile → executable
  Cheap:      cache executable, call execute() many times
  Cache key:  (program hash, input shapes, dtypes)
  Note: no layout info in key — inputs are always column-major contiguous
```

```rust
// Level 1: user retains CompiledProgram
let prog = graph.compile(...);   // expensive, do once
prog.eval(&[2.0, 3.0]);         // cheap, do many times

// Level 2: XLA backend caches compiled executable
let xla_exec = xla_cache.get_or_compile(&prog, input_shapes);
xla_exec.execute(input_buffers); // cheap
```

### XLA limitations

XLA requires **fully static shapes** at compile time. Known issues:

- **Recompilation on shape change**: every new shape combination triggers
  a full re-trace + re-compile (seconds to minutes). Cache grows with the
  number of distinct shapes.
- **Dynamic shapes are experimental**: StableHLO supports bounded dynamic
  dimensions (`tensor<? x f64>`), but XLA resolves them to static shapes
  before compilation. Truly symbolic compilation (like PyTorch's SymPy
  approach) is not planned for XLA.
- **Tensor networks**: DMRG, TCI, adaptive rank methods change bond
  dimensions at each iteration → recompilation per step → impractical.
- **Workarounds** (all with tradeoffs):
  - Padding to max size (wastes compute and memory)
  - Shape bucketing (compile a few canonical shapes)
  - Drop to eager mode (lose XLA optimization)

This is why CudaBackend exists: op-by-op execution on Execution
IR with no compilation step handles dynamic shapes natively, at the cost of
no fusion.

### Future: IREE migration

IREE is officially replacing XLA's runtime (OpenXLA project). When mature:
- Same StableHLO input — no change to our lowering code
- Additional targets: Vulkan, Metal (Apple Silicon native)
- CMake build (lighter than XLA's Bazel)
- C API for Rust FFI
- Compilation path: StableHLO → linalg → vector → LLVM IR / SPIR-V

### Memory kind compatibility

`resident_device` is tracked separately from `memory_kind`. The table below
describes only the canonical public memory kind names.

| Canonical tenferro memory kind | JAX / XLA / IREE realization | Apple Silicon notes |
|-------------------------------|------------------------------|---------------------|
| `UnpinnedHost` | `unpinned_host` or backend default host memory | `MTLBuffer` shared / host-visible allocation |
| `PinnedHost` | `pinned_host` | driver-pinned host allocation |
| `Device` | `device` | device-local or unified GPU allocation depending on backend |
| `Other("managed")` | backend-specific managed/unified memory kind | hardware-managed unified-memory realization |

---

## VII. Tensor Types and Operand

### Tensor (concrete data type)

`Tensor` is the user-facing type-erased tensor. Internally it is an enum
over concrete typed storage for each supported scalar type. **All tensors
are contiguous column-major.** No stride-aware dispatch is needed.

```rust
struct Placement {
    memory_kind: MemoryKind,
    resident_device: Option<ComputeDevice>,
}

enum MemoryKind {
    Device,
    PinnedHost,
    UnpinnedHost,
    Other(String),
}

struct TypedTensor<T: Scalar> {
    buffer: Buffer<T>,
    shape: Vec<usize>,
    // contiguous column-major: strides are implicit from shape
    placement: Placement,
}

enum Buffer<T> {
    Host(HostBuffer<T>),
    Backend(BufferHandle<T>),
}

enum Tensor {
    F32(TypedTensor<f32>),
    F64(TypedTensor<f64>),
    C32(TypedTensor<Complex<f32>>),
    C64(TypedTensor<Complex<f64>>),
}
```

### TracedTensor (graph-aware wrapper)

`TracedTensor` wraps `Tensor` with graph tracking for lazy evaluation and AD.
All operations are lazy — there is no eager mode.

```rust
struct TracedTensor {
    shape: Vec<usize>,
    dtype: DType,
    fragment: Arc<Fragment<StdTensorOp>>,  // graph info (always present)
    val: LocalValId,
    data: Option<Tensor>,               // Some for inputs / eval'd results
}
```

- `TracedTensor::from(Tensor)` creates a Fragment input node with `data = Some(...)`.
- Operations (einsum, exp, ...) build graph, return `TracedTensor` with `data = None`.
- `eval()` triggers compile (cached) + execute, fills in `data`, returns `&Tensor`.

<!-- TODO: ../examples/tensor-api-pseudocode.md was in tensor4all-meta; move to getting-started when populated -->
See `../getting-started/` for usage examples (to be added).

### Operand trait (removed)

The `Operand` trait has been removed from computegraph-rs. Compute
operations are no longer methods on the data type. Instead:

- Standard algebra: `TensorBackend` methods dispatch to optimized kernels
- Custom algebra: `SemiringBackend<Alg>` methods dispatch to user-provided
  GEMM + strided-kernel defaults for elementwise/reduction
- Structural ops: algebra-independent free functions

---

## VIII. Roadmap

### Phase 1: Minimal vertical slice

- `DotGeneral`, `Add`, `Mul`, `Neg`, `ReduceSum` (Tier 1 core)
- `SVD` primal (custom_call, LAPACK)
- 2-level IR: StableHLO IR + optimizing compiler + Execution IR
- `TensorBackend` trait in `tenferro-tensor`
- `CpuBackend` (implements `TensorBackend`, faer GEMM internal)
- Tropical: implement `TensorBackend` with existing CPU + CUDA kernels
- **Milestone**: einsum works for Standard + Tropical, SVD works

### Phase 2: Expand primitives

- Add Tier 2 ops: `Exp`, `Log`, `Sin`, `Sqrt`, `Reshape`, `Transpose`,
  `BroadcastInDim`, `Pad`, `Gather`, `Scatter`, `Slice`
- Linalg JVP rules in traced primitives (SVD, QR, Cholesky, Eigh)
- `CpuBackend` expanded for all new ops
- **Milestone**: full AD for linalg works (JVP + auto-transpose VJP)

### Phase 3: GPU backends

- `CudaBackend`: `TensorBackend` impl with CUDA kernels (reused from v1)
  - Op-by-op execution on Execution IR, no fusion, dynamic shapes
  - **Milestone**: tensor network on GPU with dynamic bond dimensions
- XLA backend: takes StableHLO IR directly (no optimizing compiler)
  - JIT compile, kernel fusion, static shapes
  - **Milestone**: ML-style workloads on GPU with fusion

### Phase 4: Optimization + IREE

- Optimizing compiler improvements (better transpose folding, memory reuse)
- Memory optimization in custom GPU engine
- IREE as future alternative to XLA (same StableHLO input)
- Tropical GPU: `TensorBackend` with hand-optimized CUDA kernels

---

## Superseded Issues (partially)

- tenferro-rs#616: Traced Tensor + StableHLO IR (AD portions → `../architecture/ad-pipeline.md`)
- tenferro-rs#618: tenferro v2 roadmap (backend portions here, AD portions → `../architecture/ad-pipeline.md`)
