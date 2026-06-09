# Primitive Catalog

**Date:** 2026-05-28
**Parent:** `../index.md`
**Related:** `backend-contract.md`, `tensor-semantics.md`, `../reference/stablehlo-primitives.md`, `../reference/jax-primitives.md`

---

## I. Purpose

This document answers the question:

> What exactly counts as a "primitive" or "instruction" in the current design at each
> level of the IR hierarchy, and what does each op mean?

**Current implementation note (2026-05-27):** the live source of truth for core
primitive identity and metadata is the internal `tenferro-core-ops` crate. The
graph carrier is `StdTensorOp`, extension operation families use
`StdTensorOp::Extension`, and execution lowers to `ExecOp` directly. There is no
live StableHLO IR layer. Where older StableHLO-era wording in this document
conflicts with [`backend-contract.md`](backend-contract.md) or
`tenferro-core-ops`, those current sources win.

The design docs use "primitive" and "instruction" in several nearby but
different senses. For readability, this document separates them explicitly:

| Layer | Example | Meaning |
|-------|---------|---------|
| Surface API | `einsum`, `sum`, `mean`, `grad`, `svd()` | what users call |
| Tenferro IR | `DotGeneral`, `ReduceSum`, `BroadcastInDim` | what may appear as `StdTensorOp` nodes in a `Graph`; graph construction, AD, einsum decomposition |
| Core primitive catalog | `PrimitiveOpKind`, descriptors | internal metadata in `tenferro-core-ops` used by graph, runtime, and backend dispatch |
| Execution IR | `ExecOp` variants | output of the optimizing compiler; input to runtime/backend dispatch |
| Backend kernel | BLAS GEMM, cuSOLVER SVD, IREE module, faer routine | how an instruction is executed |

### Active IR layers

```
┌─────────────────────────────────────────┐
│ Tenferro IR                             │
│ (StdTensorOp)                           │
│ Graph construction, AD, einsum          │
└──────────────┬──────────────────────────┘
               │ compile_std_to_exec()
┌──────────────▼──────────────────────────┐
│ Execution IR                            │
│ (ExecOp)                                │
│ runtime/backend dispatch                │
└─────────────────────────────────────────┘
```

This document uses **three orthogonal classifications**:

1. **backend-facing execution architecture** (Section III): the execution IR,
   optimizing compiler, backend traits, and execution engine
2. **Tenferro IR vocabulary** (Section IV): what the graph / AD
   stack talks about at the Tenferro IR level
3. **additional dense numeric primitives** (Section V): ops outside the minimal
   graph core but still in the standard dense tensor runtime surface

Important distinctions:

- `linearize`, `linear_transpose`, `resolve`, and `compile` are **transforms**, not
  primitives.
- `einsum` is **surface syntax**, not a final persistent Tenferro IR primitive.
  It is lowered into Tenferro IR primitives such as `DotGeneral`, `Mul`,
  `Transpose`, `BroadcastInDim`, and `ReduceSum`.
- High-level linalg ops such as `SVD` and `Solve` are standard extension
  operations owned by `tenferro-linalg`; they are not core primitive catalog
  entries.
- `StdTensorOp` is **flat** (no nested operation-kind wrapper). Most variants
  map to core primitive catalog entries. Documented exceptions include
  composite lowerings and `StdTensorOp::Extension` payloads for operation
  families such as linalg, einsum, and FFT.
- Dense runtime tensors use contiguous column-major storage. Layout and device
  placement are backend concerns described by the backend contract, not by the
  core primitive catalog.

Responsibility boundary:

- `tidu-rs` owns the `Primitive` contract
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

### `Transpose` vs AD linear_transpose

`Transpose(perm)` is the tensor operation "permute axes".
It is unrelated to the AD transform `linear_transpose(linear_graph)`.

### Multi-output primitives

Some primitives produce multiple outputs:

- `SVD(A) -> (U, S, Vt)`
- `QR(A) -> (Q, R)`

The output ordering must be part of the primitive definition because
`ValueKey` includes `output_slot`.

### Column-major (Fortran) convention

Runtime-owned tensors and execution-produced intermediates use **column-major
(Fortran) ordering**. Metadata-only views may describe non-contiguous access,
but `ExecProgram` itself does not expose arbitrary layout transforms as a
backend contract.

### What tenferro is expected to implement

From this document's point of view, the implementation target is:

- implement the 2-level IR architecture and backend traits defined below
- implement the AD-closed graph core defined below
- implement the additional dense numeric primitives when tenferro claims
  standard dense numeric support
- treat control-flow primitives as future work, not part of the initial
  required set

---

## III. Relationship to Backend Execution

The backend pipeline, Execution IR dispatch categories, backend trait
signatures, generic execution engine, buffer lifecycle, and memory layout are
owned by [`backend-contract.md`](backend-contract.md).

Key relationships:

- **Core primitive catalog** lives in `tenferro-core-ops` and supplies
  primitive identity plus metadata such as dtype policy and category.
- **Graph IR** uses `StdTensorOp` for core primitives and extension carriers.
- **Execution IR** uses `ExecOp`, including `ExecOp::Extension` for registered
  operation-family runtimes.
- **Optimizing compiler**: see [`optimizer-passes.md`](optimizer-passes.md).

---

## IV. Core Traits (canonical signatures)

### GraphOperation

`GraphOperation` is the operation node trait. computegraph-rs is fully generic over
it and never references specific primitives.

```rust
trait GraphOperation: Clone + Debug + Hash + Eq + Send + Sync + 'static {
    type Operand: Operand;
    type Context;
    type InputKey: Clone + Debug + Hash + Eq + Send + Sync + 'static;

    fn input_count(&self) -> usize;
    fn output_count(&self) -> usize;
    fn eval(&self, ctx: &mut Self::Context, inputs: &[&Self::Operand]) -> Vec<Self::Operand>;
}
```

### Operand

`Operand` is the runtime value type. Defined in
`computegraph-rs/src/traits.rs`. Contains both algebraic and structural
methods; computegraph-rs is a tensor computation graph engine, not a
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

Runtime tensor storage and placement are described in
[`tensor-semantics.md`](tensor-semantics.md). The graph-level `Operand`
contract is separate from backend buffer access; execution dispatch must go
through the runtime tensor and backend APIs described in
[`backend-contract.md`](backend-contract.md).

---

## V. Tenferro IR Vocabulary

This section is about the graph-level vocabulary that `computegraph-rs`,
`tidu-rs`, and tenferro's `StdTensorOp` layer talk about.

These ops define the **Tenferro IR**. Core primitives map to
`PrimitiveOpKind` descriptors in `tenferro-core-ops`; extension payloads are
kept outside the core catalog and dispatch through family-specific registries.

### AD-closed graph core

These are the tensor primitives needed to express:

- scalar and tensor JVP/VJP rules
- explicit broadcasting and reshaping
- general contractions (including repeated-index patterns like trace/diagonal)
- reverse-mode accumulation without hidden fan-out

Every op in this table is expected to implement `Primitive` directly
(for `StdTensorOp`). The set is **AD-closed**: `linearize` and
`transpose_rule` of any op in this table add only ops from this table.

**Implementation note:** the boundary between this core set and the
"Additional dense numeric primitives" (Section V) is primarily an
**implementation priority** guide, not a formal algebraic boundary. The full
`StdTensorOp` set, together with extension operations that register AD rules,
is expected to remain AD-closed. The core set is distinguished by two
properties: (1) it is AD-closed on its own, and (2) all ops in it are required
by shape, contraction, and reverse-mode accumulation machinery.

#### Algebraic ops

| Primitive | Signature | Definition | Notes |
|-----------|-----------|------------|-------|
| `Add` | `x0: S, x1: S -> y: S` | `y[i] = x0[i] + x1[i]` | Same shape on both inputs; no hidden broadcasting. |
| `Mul` | `x0: S, x1: S -> y: S` | `y[i] = x0[i] * x1[i]` | Elementwise multiply; same-shape contract. |
| `Neg` | `x: S -> y: S` | `y[i] = -x[i]` | Unary elementwise. |
| `Conj` | `x: S -> y: S` | `y[i] = conj(x[i])` | Identity on real dtypes, conjugation on complex dtypes. |
| `DotGeneral(config)` | `lhs: A, rhs: B -> out: C` | General tensor contraction over explicit batch axes and contracting axes | Canonical contraction primitive. Config defined below. |
| `ReduceSum(axes)` | `x: [d0, ..., dn-1] -> y` | `y` is formed by summing `x` over the listed axes | Rank drops unless a later op restores it. |

#### Structural ops

These ops rearrange or select elements without changing the dtype. They are
handled by the runtime/backend infrastructure described in
[`backend-contract.md`](backend-contract.md).

| Primitive | Signature | Definition | Notes |
|-----------|-----------|------------|-------|
| `Transpose(perm)` | `x: [d0, ..., dn-1] -> y: [d_perm[0], ..., d_perm[n-1]]` | Reorder axes according to `perm` | Pure axis permutation |
| `Reshape(shape)` | `x: [d0, ..., dn-1] -> y: shape` | Reinterpret the element sequence with a new shape | Total element count must stay unchanged. In the IR all tensors are logically dense column-major, so there is no stride ambiguity. |
| `BroadcastInDim(shape, dims)` | `x: [a0, ..., ak-1] -> y: shape` | Place input axis `j` into output axis `dims[j]`, repeating along the others | Makes all broadcast semantics explicit |
| `Gather` | `x: S -> y: S'` | Read values from `x` at positions specified by an index tensor | Needed for repeated-index einsum patterns (trace, diagonal extraction). Pure index-based read; no arithmetic. |
| `GatherDynamicSliceSizes` | `x: S, shape_sources... -> y: S'` | Same read semantics as `Gather`, but `slice_sizes` are `DimExpr` values resolved from runtime input shapes before backend dispatch | Used when AD needs gather window sizes derived from symbolic tensor metadata. Shape-source inputs are non-differentiable. |
| `Scatter` | `updates: S, x: S' -> y: S'` | Write or accumulate values into `y` at positions specified by an index tensor | Transpose of `Gather`. Needed for AD of `Gather` and for `embed_diag`. |

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

## V. Additional Dense Numeric Primitives

These primitives are part of the ordinary dense tensor runtime surface. They
are not in the minimal graph core above, but they are still standard
`StdTensorOp`/`ExecOp` vocabulary unless explicitly owned by an extension
crate.

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
and symbolic-shape AD. The remaining indexing ops are dense tensor primitives:

| Primitive | Definition | Notes |
|-----------|------------|-------|
| `Slice` | Read a static rectangular subregion | Start/limit/stride known in the op |
| `DynamicSlice` | Read a slice whose start index is data-dependent | Dynamic counterpart of `Slice` |
| `DynamicUpdateSlice` | Write an update tensor into an operand at data-dependent start indices | Transpose counterpart of `DynamicSlice`; start indices are adjusted with StableHLO `dynamic_update_slice` semantics. |
| `Pad` | Extend a tensor with edge/interior padding values | Needed by transpose rules for slicing-like ops |
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

### Standard linalg extension operations

These operation families are owned by `tenferro-linalg`, not by the core
primitive catalog.

| Operation | Outputs | Definition |
|-----------|---------|------------|
| `Cholesky` | `(L)` or `(U)` | Cholesky factorization of a positive-definite matrix |
| `SVD` | `(U, S, Vt)` | Thin singular value decomposition `A = U diag(S) Vt` |
| `QR` | `(Q, R)` | Thin QR factorization `A = Q R` |
| `Eigh` | `(eigenvalues, eigenvectors)` | Hermitian / symmetric eigendecomposition |
| `Solve` | `(X)` | Solve `A X = B` for `X` |

Linalg derivative rules add core graph primitives and, where needed, other
registered extension operations.

### Future control-flow primitives

| Primitive | Definition |
|-----------|------------|
| `Cond` | Branch between two subcomputations based on a predicate |
| `While` | Loop while a condition remains true |
| `Scan` | Structured loop with carried state and stacked outputs |

These are intentionally future-facing and are not required for the initial
vertical slice.

---

## VI. StableHLO Reference Vocabulary

When there is a choice, the Tenferro IR vocabulary should prefer the
StableHLO-style name and semantics:

| Preferred | Instead of |
|--------------|------------|
| `DotGeneral` | `einsum` or `dot` as a primitive |
| `BroadcastInDim` | implicit broadcasting or generic `broadcast` primitive |
| `Compare(dir)` + `Select` | surface names like `greater`, `greater_equal`, `where` |
| `ReduceSum` / `ReduceMax` / ... | opaque reduction primitives whose combiner is not explicit |

The goal is not to copy StableHLO mechanically. StableHLO remains a useful
semantic reference for primitive names and behavior, but it is not a live
in-process IR layer.

See also `../reference/stablehlo-primitives.md` and `../reference/jax-primitives.md`.

---

## VII. Frontend Sugar and Canonical Lowering

Many familiar user-level ops are better treated as aliases or composites rather
than as distinct graph primitives.

### Constants and literals

Constants (scalar or tensor literals) are represented by the `Constant` IR
primitive when they are embedded in a graph. User-supplied tensors still enter
through graph inputs, while helper-created literals such as `1 / n` in `mean`
or `1` in `reciprocal` lower to `Constant` operations with encoded dtype and
payload bytes.

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
  graph construction, AD, and einsum decomposition.
- The **core primitive catalog** (`tenferro-core-ops`) defines primitive
  identity and metadata used by graph, runtime, and backend dispatch.
- The **Execution IR** is the interface between the optimizing
  compiler and backend kernels.
- **Backend traits** define what a custom backend must implement.

The boundary is deliberate: graph-level concerns (AD closure, shape inference)
live above `ExecOp` lowering; execution concerns (memory layout, kernel
dispatch, fast paths) live below it.
