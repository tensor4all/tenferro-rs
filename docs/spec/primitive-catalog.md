# Primitive Catalog

**Date:** 2026-04-04
**Parent:** `../index.md`
**Related:** `backend-contract.md`, `tensor-semantics.md`, `../reference/stablehlo-primitives.md`, `../reference/jax-primitives.md`

---

## I. Purpose

This document answers the question:

> What exactly counts as a "primitive" or "instruction" in the current design at each
> level of the IR hierarchy, and what does each op mean?

**Normative status:** this file is the **source of truth** for the primitive
and instruction-set vocabulary that tenferro is expected to implement at
all levels. If another design document has a shorter summary and the two
disagree, this file wins.

The design docs use "primitive" and "instruction" in several nearby but
different senses. For readability, this document separates them explicitly:

| Layer | Example | Meaning |
|-------|---------|---------|
| Surface API | `einsum`, `sum`, `mean`, `grad`, `svd()` | what users call |
| Tenferro IR | `DotGeneral`, `ReduceSum`, `BroadcastInDim` | what may appear as `StdTensorOp` / `SemiringOp<T>` nodes in a `Fragment`; fragment construction, AD, einsum decomposition |
| StableHLO IR | `StableHloOp` variants | serializable to StableHLO MLIR for standard algebra; the single cut point between graph/AD and backends. XLA backend takes this directly (standard algebra only) |
| Execution IR | StableHLO ops + `BatchedGemm` - `DotGeneral` | output of the optimizing compiler; input to faer / custom backends |
| Backend kernel | BLAS GEMM, cuSOLVER SVD, IREE module, faer routine | how an instruction is executed |

### The 3 IR layers

```
┌─────────────────────────────────────────┐
│ Tenferro IR                             │
│ (StdTensorOp / SemiringOp<T>)           │
│ Fragment construction, AD, einsum        │
└──────────────┬──────────────────────────┘
               │ lower_to_stablehlo()
┌──────────────▼──────────────────────────┐
│ StableHLO IR                            │
│ (StableHloOp)                           │
│ Serializable cut point                  │
│ XLA backend takes this directly         │
└──────────────┬──────────────────────────┘
               │ optimizing compiler
┌──────────────▼──────────────────────────┐
│ Execution IR                            │
│ (ExecOp)                                │
│ faer / custom backends execute this     │
│ stride-aware engine dispatch            │
└─────────────────────────────────────────┘
```

This document uses **three orthogonal classifications**:

1. **backend-facing execution architecture** (Section III): the 2-level IR
   (StableHLO IR and Execution IR), optimizing compiler, backend traits,
   and execution engine
2. **Tenferro IR vocabulary** (Section IV): what the graph / AD
   stack talks about at the Tenferro IR level
3. **standard arithmetic extensions** (Section V): ops available only for
   ordinary dense numeric types

Important distinctions:

- `differentiate`, `transpose`, `resolve`, and `compile` are **transforms**, not
  primitives.
- `einsum` is **surface syntax**, not a final persistent Tenferro IR primitive.
  It is lowered into Tenferro IR primitives such as `DotGeneral`, `Mul`,
  `Transpose`, `BroadcastInDim`, and `ReduceSum`.
- High-level linalg ops such as `SVD` and `Solve` may remain explicit
  Tenferro IR primitives because they are meaningful semantic units, even
  though their derivative rules emit lower-level primitives.
- `StdTensorOp` is **flat** (no `SemiringOpKind` wrapping). Most variants
  map 1:1 to a StableHLO op. Documented exceptions include composite
  lowerings (e.g., `Conj` -> 4 ops: `real` + `imag` + `negate` + `complex`)
  and multi-output linalg ops (e.g., `Svd` -> `custom_call` +
  `get_tuple_element` x N).
- `Tensor` allows **arbitrary strides** at the user level. Input
  pre-processing happens at **eval() time**: contiguous data (including
  permuted-contiguous views from `permute()` or `.t()`) is passed as-is
  with zero copy, preserving strides; only truly non-contiguous data
  (memory gaps from slicing) is physically copied to a contiguous buffer.
  No StableHLO ops are inserted for input normalization -- the StableHLO
  program is layout-independent. The execution engine is stride-aware and
  handles permuted inputs via BLAS trans flags at dispatch time.

Responsibility boundary:

- `chainrules-rs` owns the `PrimitiveOp` contract
- `tidu-rs` owns generic AD transforms that call `linearize` and
  `transpose_rule`
- tenferro owns the **concrete per-op derivative rules**

So this directory keeps the primitive vocabulary and cross-crate architecture,
but not a standalone per-op transpose-rule manual. Detailed formulas are a
downstream tenferro design/implementation concern.

---

## II. Reading the Tables

### Shape notation

- `x: [b, m, n]` means `x` is a rank-3 tensor with shape `(b, m, n)`.
- Scalars are rank-0 tensors and are written as `[]`.
- Batch dimensions are written explicitly; nothing is implied by position alone.

### No implicit broadcasting

Elementwise primitives such as `Add` and `Mul` do **not** silently broadcast.
If shapes differ, the graph must contain an explicit `BroadcastInDim`.

### `Transpose` vs AD transpose

`Transpose(perm)` is the tensor operation "permute axes".
It is unrelated to the AD transform `transpose(linear_fragment)`.

### Multi-output primitives

Some primitives produce multiple outputs:

- `SVD(A) -> (U, S, Vt)`
- `QR(A) -> (Q, R)`

The output ordering must be part of the primitive definition because
`GlobalValKey` includes `output_slot`.

### Column-major (Fortran) convention

Execution-produced intermediates and outputs use **column-major (Fortran)
ordering**. This is the convention for all data produced by the backend
execution pipeline. Input tensors may be contiguous with arbitrary axis
ordering; dispatch inspects strides and adjusts backend calls accordingly
(e.g., BLAS trans flags for transposed inputs).

### What tenferro is expected to implement

From this document's point of view, the implementation target is:

- implement the 2-level IR architecture and backend traits defined below
- implement the AD-closed graph core defined below
- implement the `Standard arithmetic only` primitives when tenferro claims
  standard dense numeric support
- treat control-flow primitives as future work, not part of the initial
  required set

---

## III. Relationship to Backend Execution

The backend pipeline, Execution IR dispatch categories, backend trait
signatures (`SemiringCore`, `SemiringFastPath`), generic execution engine,
buffer lifecycle, and memory layout are owned by
[`backend-contract.md`](backend-contract.md).

Key relationships:

- **StableHLO IR** uses the Tenferro IR ops from Section IV. For standard
  algebra, serializable to StableHLO MLIR. For custom algebra, same structure
  but semiring semantics (Add=⊕, Mul=⊗); XLA path not available.
- **Execution IR** = StableHLO ops + `BatchedGemm` − `DotGeneral`.
- **Add/Mul dispatch** is algebra-dependent (see backend-contract.md).
- **Custom algebra minimum**: `batched_gemm` + `reduce_sum` via `SemiringCore`.
- **Optimizing compiler**: see [`optimizer-passes.md`](optimizer-passes.md).

---

## IV. Core Traits (canonical signatures)

### GraphOp

`GraphOp` is the operation node trait. computegraph-rs is fully generic over
it and never references specific primitives.

```rust
trait GraphOp: Clone + Debug + Hash + Eq + Send + Sync + 'static {
    type Operand: Operand;
    type Context;
    type InputKey: Clone + Debug + Hash + Eq + Send + Sync + 'static;

    fn n_inputs(&self) -> usize;
    fn n_outputs(&self) -> usize;
    fn eval(&self, ctx: &mut Self::Context, inputs: &[&Self::Operand]) -> Vec<Self::Operand>;
}
```

### Operand

`Operand` is the runtime value type. Defined in
`computegraph-rs/src/traits.rs`. Contains both algebraic and structural
methods — computegraph-rs is a tensor computation graph engine, not a
fully generic DAG engine.

```rust
pub trait Operand: Clone + Send + Sync + 'static {
    fn zero(shape: &[usize]) -> Self;
    fn one(shape: &[usize]) -> Self;
    fn reshape(&self, shape: &[usize]) -> Self;
    fn broadcast_in_dim(&self, shape: &[usize], dims: &[usize]) -> Self;
    fn add(&self, other: &Self) -> Self;
    fn multiply(&self, other: &Self) -> Self;
    fn reduce_sum(&self, axes: &[usize]) -> Self;
    fn dot_general(
        &self, other: &Self,
        lhs_contracting: &[usize], rhs_contracting: &[usize],
        lhs_batch: &[usize], rhs_batch: &[usize],
    ) -> Self;
    fn conj(&self) -> Self;
}
```

`TensorData` (see [`tensor-semantics.md`](tensor-semantics.md)) provides
additional buffer access methods (`shape`, `strides`, `data`) needed by the
execution engine's common infrastructure.

---

## V. Tenferro IR Vocabulary

This section is about the graph-level vocabulary that `computegraph-rs`,
`chainrules-rs`, `tidu-rs`, and tenferro's `StdTensorOp` layer talk about.

These ops define the **Tenferro IR**. Each op maps to a StableHLO op when
lowered via `lower_to_stablehlo()`. The XLA backend takes the StableHLO IR
directly; other backends lower the StableHLO IR through the optimizing
compiler to produce Execution IR.

### AD-closed graph core

These are the tensor primitives needed to express:

- scalar and tensor JVP/VJP rules
- explicit broadcasting and reshaping
- general contractions (including repeated-index patterns like trace/diagonal)
- reverse-mode accumulation without hidden fan-out

Every op in this table is expected to implement `PrimitiveOp` directly
(for `StdTensorOp`). The set is **AD-closed**: `linearize` and
`transpose_rule` of any op in this table emit only ops from this table.

**Implementation note:** the boundary between this core set and the
"Standard arithmetic extensions" (Section V) is primarily an
**implementation priority** guide, not a formal algebraic boundary. The
full `StdTensorOp` set (core + extensions) is also AD-closed. The core
set is distinguished by two properties: (1) it is AD-closed on its own,
and (2) all ops in it are well-defined for arbitrary semiring algebras
(not just standard arithmetic), making them available to `SemiringOp<T>`.

#### Algebraic ops

| Primitive | Signature | Definition | Notes |
|-----------|-----------|------------|-------|
| `Add` | `x0: S, x1: S -> y: S` | `y[i] = x0[i] + x1[i]` | Same shape on both inputs; no hidden broadcasting. Maps to ⊕ for custom algebras. |
| `Mul` | `x0: S, x1: S -> y: S` | `y[i] = x0[i] * x1[i]` | Elementwise multiply; same-shape contract. Maps to ⊗ for custom algebras. |
| `Neg` | `x: S -> y: S` | `y[i] = -x[i]` | Unary elementwise. Standard algebra only (semirings lack additive inverse). |
| `Conj` | `x: S -> y: S` | `y[i] = conj(x[i])` | Identity on real dtypes, conjugation on complex dtypes. Standard algebra only. |
| `DotGeneral(config)` | `lhs: A, rhs: B -> out: C` | General tensor contraction over explicit batch axes and contracting axes | Canonical contraction primitive; uses ⊕ and ⊗ for custom algebras. Config defined below. |
| `ReduceSum(axes)` | `x: [d0, ..., dn-1] -> y` | `y` is formed by summing `x` over the listed axes | Uses ⊕ for custom algebras. Rank drops unless a later op restores it. |

#### Structural ops

These ops rearrange or select elements without any arithmetic. They are
well-defined for all algebras and handled by common infrastructure at the
backend level (not part of the custom backend contract).

| Primitive | Signature | Definition | Notes |
|-----------|-----------|------------|-------|
| `Transpose(perm)` | `x: [d0, ..., dn-1] -> y: [d_perm[0], ..., d_perm[n-1]]` | Reorder axes according to `perm` | Pure axis permutation |
| `Reshape(shape)` | `x: [d0, ..., dn-1] -> y: shape` | Reinterpret the element sequence with a new shape | Total element count must stay unchanged. In the IR all tensors are logically dense column-major, so there is no stride ambiguity. |
| `BroadcastInDim(shape, dims)` | `x: [a0, ..., ak-1] -> y: shape` | Place input axis `j` into output axis `dims[j]`, repeating along the others | Makes all broadcast semantics explicit |
| `Gather` | `x: S -> y: S'` | Read values from `x` at positions specified by an index tensor | Needed for repeated-index einsum patterns (trace, diagonal extraction). Pure index-based read; no arithmetic. |
| `GatherDynamicSliceSizes` | `x: S, shape_sources... -> y: S'` | Same read semantics as `Gather`, but `slice_sizes` are `DimExpr` values resolved from runtime input shapes before backend dispatch | Used when AD needs gather window sizes derived from symbolic tensor metadata. Shape-source inputs are non-differentiable. |
| `Scatter` | `updates: S, x: S' -> y: S'` | Write or accumulate values into `y` at positions specified by an index tensor | Transpose of `Gather`. Accumulation uses ⊕. Needed for AD of `Gather` and for `embed_diag`. |

### DotGeneral config

```rust
struct DotGeneralConfig {
    lhs_contracting_dims: Vec<usize>,
    rhs_contracting_dims: Vec<usize>,
    lhs_batch_dims: Vec<usize>,
    rhs_batch_dims: Vec<usize>,
}
```

Contracting dims are summed over (inner product). Batch dims are preserved
in the output. Remaining dims appear in the output in lhs-then-rhs order.
This matches StableHLO's `dot_general` dimension numbers.

### Concrete examples

| Primitive | Example |
|-----------|---------|
| `DotGeneral` | `ij,jk->ik` (ordinary matrix multiply) |
| `BroadcastInDim` | `[n] -> [b, n]` with `dims=[1]` |
| `ReduceSum` | `[b, m, n] -> [b, n]` with `axes=[1]` |
| `Transpose` | `[b, m, n] -> [m, b, n]` with `perm=[1, 0, 2]` |

---

### Trace, diagonal, and their AD helpers

**Decision:** Trace/Diag/AntiTrace/AntiDiag are **not** dedicated Tenferro IR
primitives. They are lowered to existing Tenferro IR ops (which map to
StableHLO):

| Surface op | Tenferro IR lowering |
|-----------|-------------------|
| `trace(A)` | einsum `ii->` = diagonal extraction (`Gather` pattern) + `ReduceSum` |
| `diag(A)` / `extract_diag(A)` | einsum `ii->i` = `Gather` pattern |
| `embed_diag(v)` | einsum `i->ii` = `Scatter` / `BroadcastInDim` + `Mul` with identity mask |
| AntiTrace (AD helper) | `Scatter` + `BroadcastInDim` in transpose rules |
| AntiDiag (AD helper) | `Scatter` in transpose rules |

This keeps the Tenferro IR vocabulary aligned with StableHLO (which has no
Trace/Diag ops) and avoids adding non-standard primitives. The einsum engine
already handles repeated-index patterns (`ii->i`, `ii->`) internally via
diagonal extraction in the previous (`dispatch.rs` diagonal plan).

---

## V. Standard Arithmetic Only

These primitives are available only for the ordinary dense numeric setting
(real/complex standard arithmetic). They are not assumed to exist for generic
semirings such as tropical algebra.

This section should be kept as close as practical to the official StableHLO op
set, so that tenferro's Tenferro IR primitives lower cleanly to StableHLO.
See `../reference/stablehlo-primitives.md` for the StableHLO-facing reference and
`../reference/jax-primitives.md` for the JAX-side reference point.

### Elementwise arithmetic, comparison, and selection

| Primitive | Definition | Notes |
|-----------|------------|-------|
| `Div` | `y[i] = x0[i] / x1[i]` | Canonical division op |
| `Abs` | `y[i] = abs(x[i])` | Real magnitude or complex modulus, depending on dtype contract |
| `Sign` | `y[i] = sign(x[i])` | Often used in stabilization logic |
| `Maximum` | `y[i] = max(x0[i], x1[i])` | Ordered real comparison |
| `Minimum` | `y[i] = min(x0[i], x1[i])` | Ordered real comparison |
| `Compare(dir)` | Produce a `Bool` tensor from an elementwise comparison | `dir` is things like `eq`, `lt`, `le`, `gt`, `ge` |
| `Select` | `y[i] = pred[i] ? on_true[i] : on_false[i]` | `pred` is `Bool`; value inputs determine the output dtype |
| `Clamp` | `y[i] = min(max(x[i], lower[i]), upper[i])` | Canonical clipping primitive |

### Analytic elementwise primitives

| Primitive | Definition |
|-----------|------------|
| `Exp` | `exp(x)` |
| `Log` | `log(x)` |
| `Sin` | `sin(x)` |
| `Cos` | `cos(x)` |
| `Tanh` | `tanh(x)` |
| `Sqrt` | `sqrt(x)` |
| `Rsqrt` | `1 / sqrt(x)` |
| `Pow` | `x^y` |
| `Expm1` | `exp(x) - 1` |
| `Log1p` | `log(1 + x)` |

The table above is the canonical analytic seed set. Additional analytic ops
may be added later, but they are not part of the current required list unless
this document is updated.

### Indexing and structural data movement

`Gather`, `GatherDynamicSliceSizes`, and `Scatter` are in the AD-closed graph
core (Section IV) because they are needed for repeated-index einsum patterns
and symbolic-shape AD, and are well-defined for all algebras. The remaining
indexing ops are standard-arithmetic only:

| Primitive | Definition | Notes |
|-----------|------------|-------|
| `Slice` | Read a static rectangular subregion | Start/limit/stride known in the op |
| `DynamicSlice` | Read a slice whose start index is data-dependent | Dynamic counterpart of `Slice` |
| `DynamicUpdateSlice` | Write an update tensor into an operand at data-dependent start indices | Transpose counterpart of `DynamicSlice`; start indices are adjusted with StableHLO `dynamic_update_slice` semantics. |
| `Pad` | Extend a tensor with edge/interior padding values | Needed for transpose of slicing-like ops |
| `Concatenate` | Join tensors along one axis | Rank-preserving shape change |
| `Reverse` | Reverse the order of elements along selected axes | Useful for convolutions and sequence models |

### Additional reductions

| Primitive | Definition |
|-----------|------------|
| `ReduceProd` | Multiply values over the listed axes |
| `ReduceMax` | Max over the listed axes |
| `ReduceMin` | Min over the listed axes |

`ReduceSum` stays in the AD-closed graph core because it is essential both for primal tensor code
and for transpose rules.

### Linalg primitives

| Primitive | Outputs | Definition | StableHLO lowering |
|-----------|---------|------------|--------------------|
| `Cholesky` | `(L)` or `(U)` | Cholesky factorization of a positive-definite matrix | Direct StableHLO op (`stablehlo.cholesky`) |
| `SVD` | `(U, S, Vt)` | Thin singular value decomposition `A = U diag(S) Vt` | `stablehlo.custom_call` |
| `QR` | `(Q, R)` | Thin QR factorization `A = Q R` | `stablehlo.custom_call` |
| `Eigh` | `(eigenvalues, eigenvectors)` | Hermitian / symmetric eigendecomposition | `stablehlo.custom_call` |
| `Solve` | `(X)` | Solve `A X = B` for `X` | `stablehlo.custom_call` |

`Cholesky` has a direct StableHLO op. All other linalg primitives lower to
`stablehlo.custom_call` with appropriate target names (matching JAX/XLA
conventions for LAPACK/cuSOLVER dispatch).

Regardless of lowering path, derivative rules for all linalg ops emit graph
primitives that satisfy `PrimitiveOp` closure.

### Future control-flow primitives

| Primitive | Definition |
|-----------|------------|
| `Cond` | Branch between two subcomputations based on a predicate |
| `While` | Loop while a condition remains true |
| `Scan` | Structured loop with carried state and stacked outputs |

These are intentionally future-facing and are not required for the initial
vertical slice.

---

## VI. StableHLO Alignment

When there is a choice, the Tenferro IR vocabulary should prefer the
StableHLO-style name and semantics:

| Preferred | Instead of |
|--------------|------------|
| `DotGeneral` | `einsum` or `dot` as a primitive |
| `BroadcastInDim` | implicit broadcasting or generic `broadcast` primitive |
| `Compare(dir)` + `Select` | surface names like `greater`, `greater_equal`, `where` |
| `ReduceSum` / `ReduceMax` / ... | opaque reduction primitives whose combiner is not explicit |

The goal is not to copy StableHLO mechanically. The goal is to ensure that the
`Standard arithmetic only` part of tenferro's Tenferro IR vocabulary has an
obvious, low-friction lowering path to StableHLO, because the StableHLO IR
is the single cut point for all backends.

See also `../reference/stablehlo-primitives.md` and `../reference/jax-primitives.md`.

---

## VII. Frontend Sugar and Canonical Lowering

Many familiar user-level ops are better treated as aliases or composites rather
than as distinct graph primitives.

### Constants and literals

Constants (scalar or tensor literals) are **not** Tenferro IR primitives.
They enter the graph as `Fragment` input nodes with attached data
(`TracedTensor::from_tensor_concrete_shape(Tensor::from_vec_col_major(...))`). At StableHLO lowering, these
become `stablehlo.constant` ops. Canonical lowerings that reference literal
values (e.g., `1 / n` in `mean`, `1` in `reciprocal`) construct these as
`Fragment` inputs.

### Lowering table

| Surface op | Tenferro IR form |
|------------|----------------------|
| `einsum(...)` | contraction planning + `DotGeneral`/`Mul`/`Transpose`/`Reshape`/`BroadcastInDim`/`ReduceSum` |
| `sum(x, axes)` | `ReduceSum(x, axes)` |
| `mean(x, axes)` | `ReduceSum(x, axes)` followed by `Mul(result, constant(1/n))` |
| `sub(x, y)` | `Add(x, Neg(y))` |
| `square(x)` | `Mul(x, x)` |
| `reciprocal(x)` | `Div(constant(1), x)` |
| `where(pred, a, b)` | `Select(pred, a, b)` |
| `greater(x, y)` | `Compare(dir=gt)` |
| `greater_equal(x, y)` | `Compare(dir=ge)` |
| `clamp_min(x, lo)` / `clamp_max(x, hi)` | special cases of `Clamp` |
| `trace(x)` | diagonal extraction (`Gather` pattern) + `ReduceSum` |
| `diag(x)` / `extract_diag(x)` | diagonal extraction (`Gather` pattern) |

This is useful for two reasons:

1. the Tenferro IR vocabulary stays smaller and easier to reason about
2. AD closure becomes easier to verify because fewer primitive rules are truly
   fundamental

---

## VIII. Implementation Note

This catalog defines the **semantic vocabulary at all IR levels** that the
graph, AD stack, and execution engine talk about.

- The **Tenferro IR** (Section IV) defines the graph-level vocabulary for
  fragment construction, AD, and einsum decomposition.
- The **StableHLO IR** (Section III.1) is the single cut point interface
  between the graph/AD world and execution.
- The **Execution IR** (Section III.3) is the interface between the optimizing
  compiler and backend kernels.
- **Backend traits** (Section III.4) define what a custom backend must
  implement.

The boundary is deliberate: graph-level concerns (AD closure, shape inference)
live above the StableHLO IR cut point; execution concerns (memory layout,
kernel dispatch, fast paths) live below it.
