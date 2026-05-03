# Dynamic And Symbolic Shape Metadata

**Status:** canonical design note for issue #829
**Related:** `../spec/optimizer-passes.md`, `../spec/ad-contract.md`

## Purpose

This note defines the shape-metadata contract needed for dimensions that are
not plain constants or input-axis sizes.

The immediate triggers are:

- `DynamicTruncate(input, size_scalar, axis)`, whose output extent depends on a
  runtime scalar tensor value.
- `transpose_scatter`, which needs inverse gather `slice_sizes` derived from
  symbolic update-window dimensions.

Both expose the same root problem: current metadata can describe concrete
sizes and symbolic arithmetic over tensor axis sizes, but it cannot say whether
the result is exact, conservative, or derived from runtime tensor values.

This document is the current shape contract for #829 work.

## Current Model

The current system has two related expression forms:

- `DimExpr`: op-local expressions over the current op's input shapes.
- `SymDim`: value-side symbolic expressions used by traced graph construction
  and AD metadata.

This is enough for expressions such as:

```text
output_dim = input0.axis(0) * input1.axis(2)
```

It is not enough for:

```text
output_dim = clamp_runtime_scalar(input1, 0, input0.axis(axis))
```

Nor is it enough to distinguish these two claims:

```text
the output axis is exactly n
the output axis is at most n
```

The second distinction matters because compiler passes may use metadata to emit
runtime `Reshape` or `BroadcastInDim` parameters. An upper bound is useful for
some safety checks, but it is not a legal replacement for an exact dimension.

## Design Goals

- Preserve the fast static-shape path for existing concrete programs.
- Make rank metadata exact even when some extents are dynamic.
- Distinguish exact extents from upper-bound or unknown extents.
- Keep backend kernels concrete: backend configs receive resolved `usize`
  sizes, not unresolved symbolic expressions.
- Let graph and compiler layers carry symbolic config values until they can be
  resolved at execution time.
- Make unsupported dynamic-shape AD paths fail through a structured error path,
  not `panic!`.
- Avoid implementing full dynamic shape polymorphism in the first pass.

## Non-Goals

- Do not replace every shape expression user in one PR.
- Do not require all backend kernels to accept dynamic shape parameters.
- Do not introduce a constraint solver.
- Do not add scatter-only or `DynamicTruncate`-only hacks that bypass shared
  metadata invariants.
- Do not change user-facing tensor operation semantics beyond replacing
  inaccurate metadata and panics with explicit behavior.

## Recommended Approach

Use a two-layer model:

```text
ShapeMeta
  rank: exact
  extents: Vec<ShapeExtent>

ShapeExtent
  Exact(ExtentExpr)
  UpperBound(ExtentExpr)
  Unknown

ExtentExpr
  Const(usize)
  InputAxis { input_idx, axis }
  RuntimeScalar { input_idx, semantics }
  Add/Sub/Mul/FloorDiv/Min/Max(...)
```

Names are provisional. The important split is semantic:

- `ExtentExpr` says how a size would be computed.
- `ShapeExtent` says what guarantee the expression provides.

For current static programs, every extent remains `Exact(Const(...))` or
`Exact(InputAxis { ... })`. The new states are only needed where current code
already has inaccurate metadata or panic behavior.

### `Exact`

An exact extent may be used to construct runtime shape parameters. For example,
a `Reshape` target may be built from exact expressions because execution can
resolve them from the concrete input tensors.

Exact does not mean compile-time constant. It means the expression denotes the
true runtime size.

### `UpperBound`

An upper-bound extent means:

```text
actual_runtime_extent <= expression_value
```

It may be used for conservative reasoning, diagnostics, allocation guards, or
skip decisions. It must not be used as if it were the true output shape.

For the first implementation pass, `DynamicTruncate` can use:

```text
axis != truncated_axis: Exact(input.axis(axis))
axis == truncated_axis: UpperBound(input.axis(axis))
```

This immediately fixes the false exactness without requiring runtime scalar
expressions to be threaded through every compiler path.

### `Unknown`

Unknown means no useful extent expression is available. It should be rare.
Code that sees `Unknown` must either avoid shape-sensitive rewrites or return a
structured unsupported-dynamic-shape error.

## Value-Side Metadata Boundary

Shape metadata belongs to values, not to operation payloads. Operation payloads
should carry only structural identity and output requirements that are part of
the op's semantics. Input-shape snapshots used for AD, validation, or replay
belong in value metadata.

| Payload kind | Owner |
|---|---|
| Structural parameters such as axes, permutation order, or contraction dims | Op payload |
| Required output shapes supplied by the user or frontend | Op payload as exact shape expressions |
| Input shape snapshots, inferred output shape facts, and guardable metadata | Value metadata |

`ShapeGuardContext` is the normative AD-facing metadata surface. Builder and
emitter helpers may provide convenience accessors, but they must read from the
same metadata store and record the same guards. AD rules must not recover
shape facts by inspecting unrelated op payloads or assuming concrete extents
from earlier graph-building phases.

## Runtime Scalar Dimensions

The long-term exact representation for `DynamicTruncate` is a runtime scalar
dimension expression:

```text
Exact(Min(
  RuntimeScalar { input_idx: 1, semantics: DynamicTruncateSize },
  InputAxis { input_idx: 0, axis }
))
```

The `semantics` tag is required because converting a scalar tensor into a
dimension is not a generic numeric cast. `DynamicTruncate` currently accepts
specific scalar dtypes and applies operation-specific rounding and clamping
rules. Those rules must stay explicit.

This should be a second implementation stage. The first stage only needs to
stop reporting an upper bound as exact.

## Symbolic Operation Configs

Backend-facing configs should remain concrete. For example,
`tenferro_tensor::GatherConfig` can continue to carry `slice_sizes: Vec<usize>`
because backend kernels execute on concrete tensors.

Graph-facing configs need a symbolic form where shape-derived sizes can appear:

```text
GraphGatherConfig
  offset_dims: Vec<usize>
  collapsed_slice_dims: Vec<usize>
  start_index_map: Vec<usize>
  index_vector_dim: usize
  slice_sizes: Vec<ShapeExtent or ExtentExpr>
```

Lowering from graph config to backend config resolves symbolic slice sizes
against concrete runtime inputs immediately before dispatch. Resolution must
fail if the value is not exact.

This keeps the layering clean:

- AD and compiler code may express symbolic config sizes.
- Execution resolves them at the backend boundary.
- Backend kernels stay optimized for concrete sizes.

## Scatter Transpose Policy

`transpose_scatter` currently builds inverse gather `slice_sizes` from
`updates_shape[axis].constant_value()` and panics when that dimension is
symbolic.

The correct behavior is:

1. If all required update-window extents are concrete, keep emitting the
   existing concrete inverse gather.
2. If an extent is symbolic but exact and resolvable through graph metadata,
   emit a symbolic graph gather config.
3. If the needed extent is not exact or cannot be expressed by the current AD
   graph, return a structured unsupported-dynamic-shape error.

The third case is acceptable in the first implementation stage. It is still a
correctness improvement over panic because callers can report the limitation
without aborting graph construction.

## Compiler Pass Contract

Compiler passes must state what shape guarantee they require.

| Consumer | Required guarantee | Rule |
|---|---|---|
| Rank checks | exact rank | Rank is always exact metadata. |
| `Transpose` metadata | any known extents | Permute extent metadata without changing guarantees. |
| `BroadcastInDim` execution shape | exact target extents | Reject or defer if any target extent is not exact. |
| `Reshape` execution shape | exact target extents | Never use upper bounds as reshape sizes. |
| `DotDecomposer` merge reshapes | exact merged extents | Skip only if the original backend op remains correct; otherwise error. |
| DCE and last-use analysis | no extent guarantee | Shape metadata is irrelevant. |
| Diagnostics | best available | May print exact, upper-bound, or unknown metadata. |

The important invariant is that an optimization pass may become conservative,
but it may not silently reinterpret upper-bound metadata as exact metadata.

## AD Contract

AD rules must not call `constant_value().unwrap_or_else(panic)` for user-reachable
symbolic shapes.

There are two acceptable outcomes:

- emit a graph using exact symbolic metadata
- return an unsupported-dynamic-shape error

Current AD rule signatures are not uniformly `Result`-returning. The
implementation should introduce one shared error channel for AD construction
limitations instead of adding local panics or ad hoc sentinel ops.

The error should identify:

- the primitive
- the metadata field that required an exact extent
- whether the observed extent was symbolic, upper-bound, or unknown

## Alternatives Considered

### Local `DynamicTruncate` patch

Returning another static shape from `shape_infer` is not acceptable. The
pre-truncation input shape is only an upper bound, and another guessed static
extent would have the same false-exactness bug.

### Symbolic `GatherConfig` only

This would fix one panic, but it would not define how compiler passes should
treat exact versus conservative metadata. It also leaves `DynamicTruncate`
incorrect.

### Full runtime-scalar shape expressions first

This is the clean end state, but it touches shape inference, lowering,
execution, AD, and config resolution at once. The first implementation should
land the exactness contract and conservative handling before threading runtime
scalar expressions through all layers.

## Migration Plan

1. Add the metadata contract types or equivalent internal representation.
   Preserve current exact behavior for existing static programs.
2. Mark `DynamicTruncate`'s truncated axis as an upper bound instead of exact.
   Update consumers so they do not use that extent as a concrete shape
   parameter.
3. Add structured AD construction errors for dynamic or unsupported shape
   requirements.
4. Replace `transpose_scatter`'s symbolic update-window panic with either a
   symbolic graph config or the structured unsupported error.
5. Add graph-facing symbolic config support for gather-like `slice_sizes`.
6. Add exact runtime scalar extents for `DynamicTruncate` once the compiler and
   execution layers can resolve them safely.

Each step should add focused regression tests. The first tests should cover:

- `DynamicTruncate` metadata no longer being treated as exact pre-truncation
  shape.
- `transpose_scatter` with symbolic update-window dimensions no longer
  panicking.
- compiler passes refusing to build `Reshape` or `BroadcastInDim` parameters
  from upper-bound extents.

## Open Questions

- Should the new metadata type live in `tenferro-ops` beside `DimExpr` and
  `SymDim`, or in `tenferro` beside lowering metadata until it stabilizes?
- Should graph-facing symbolic gather config replace `StdTensorOp::Gather`
  directly, or should a temporary `GatherSym` variant keep migration smaller?
- Which public transform APIs should surface unsupported dynamic-shape AD
  errors first?

The preferred bias is to keep the first implementation private and narrow,
then expose public types only after the compiler and AD contracts settle.
