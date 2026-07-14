# Dynamic And Symbolic Shape Metadata

**Status:** current durable design for dynamic/symbolic metadata (#829) and
extension shape equality constraints (#1370)
**Related:** `../spec/optimizer-passes.md`, `../spec/ad-contract.md`,
`../spec/primitive-catalog.md`, `../spec/backend-contract.md`,
`../spec/extension-op.md`

## Purpose

This note defines the shape-metadata architecture needed for dimensions that
are not plain constants or input-axis sizes, and the graph/compiler lifecycle
for equality relations declared by extension operations.

The immediate metadata triggers are:

- `DynamicTruncate(input, size_scalar, axis)`, whose output extent depends on a
  runtime scalar tensor value.
- `transpose_scatter`, which needs inverse gather `slice_sizes` derived from
  symbolic update-window dimensions.

Both expose the same root problem: current metadata can describe concrete
sizes and symbolic arithmetic over tensor axis sizes, but it cannot say whether
the result is exact, conservative, or derived from runtime tensor values.

Issue #1370 adds a related requirement: extensions must be able to declare
equalities between independently sourced symbolic axes, preserve those
relations with the graph, and enforce them before backend execution.

This document is the durable architecture description for both concerns. The
normative extension API and enforcement rules are owned by the
[`ExtensionOp` shape and dtype inference contract](../spec/extension-op.md#7-shape-and-dtype-inference),
its [AD API surface](../spec/extension-op.md#10-ad-api-surface), and the
[`ExtensionOp` failure modes](../spec/extension-op.md#12-failure-modes).

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
- Avoid new AD construction panics for resolvable symbolic shape metadata.
- Avoid implementing full dynamic shape polymorphism in the first pass.

## Non-Goals

- Do not replace every shape expression user in one PR.
- Do not require all backend kernels to accept dynamic shape parameters.
- Do not introduce a general symbolic algebra solver. The implemented
  extension equality engine is deliberately limited to the rules described
  below.
- Do not add scatter-only or `DynamicTruncate`-only hacks that bypass shared
  metadata invariants.
- Do not change user-facing tensor operation semantics beyond replacing
  inaccurate metadata and panics with explicit behavior.

## Recommended Approach

Use a two-layer model:

```text
Value shape metadata
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

The concrete implementation stores `Vec<ShapeExtent<_>>` directly in
`TensorMeta` and `ExecInstruction::output_extents`; there is no separate public
shape-metadata wrapper. The important split is semantic:

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

## Extension Shape Equality Lifecycle

This section describes architecture and implementation ownership. The
normative contract remains in the linked `ExtensionOp` specification sections
above.

Extension output metadata and extension input relations share one inference
callback, but they have different owners after inference:

```text
ExtensionShapeContext declaration
  -> graph-owned constraint scope
  -> compiler proof or disproof
  -> unresolved ExecProgram guard
  -> executor metadata preflight
```

An extension reads dtype and `SymDim` input shapes through
`ExtensionShapeContext` and records equalities such as `a == b` or
`a == 2 * b`. The canonical inference driver translates those declarations
from the extension-local input namespace to `DimExpr` relations. A graph scope
then stores each relation with its ordered graph inputs and every output of the
originating operation. Constraints are not extension-payload state and do not
depend on a process-global registry.

Each `TracedTensor` carries an immutable, shared constraint-scope history.
Ordinary graph composition merges histories; a scope is skipped when it is
empty. Extension fast paths that expand directly to core operations attach the
same inferred contract to their expanded outputs, so replacing a fused node
does not weaken its shape requirements. The attachment API runs inference once
and uses the same result for output metadata and constraints.

Graph analysis discovers output metadata and local extension constraints in
one root-local walk. Already registered external values are resolved directly;
only an unregistered external value triggers an on-demand parent lookup, backed
by one lazily built key-to-parent index per analysis. This avoids replaying
ancestor inference and avoids repeated scans as graph history grows.

### Compiler proof, optimizer liveness, and guards

Compilation lowers reachable scope inputs to SSA slots and evaluates equality
relations in an order-independent pipeline:

- checked constant folding and structural normalization;
- canonical ordering for commutative expressions;
- union of bare axis symbols and binding of a bare symbol to a constant;
- evaluation when all referenced input extents are concrete.

Proven equalities disappear. Disproven equalities return a typed
`ShapeConstraintViolation`. An unresolved equality is retained as a normalized
`ShapeGuard`. Arithmetic overflow, underflow, division by zero, a missing live
input, or an invalid live axis is a typed `ShapeConstraintEvaluation`, not an
unknown result.

This is not inverse or general algebraic solving. In particular, symbolic
`a == 2 * b` remains a guard; the engine does not rearrange it to infer `b`.
The same relation is folded or rejected when its operands later become
concrete. Inequalities, divisibility reasoning, and general equation solving
remain out of scope.

Graph-scoped constraints use pre-optimizer origin slots, so a live constraint
survives elimination of an identity reshape, transpose, or its originating
extension carrier. The compiler prunes one only when none of its origin outputs
is live. Constraints freshly inferred from an extension instruction may be
pruned if that optimized instruction is dead. A graph-scoped live relation
with a broken key, slot, or axis is always an error.

The parallel symbolic analysis for `Reshape` and `BroadcastInDim` is
best-effort. If a deliberately unresolved `InputDim` cannot be resolved in that
analysis namespace, the expression stays symbolic. The separate executable
shape path remains authoritative and concrete, preserving its typed validation
instead of leaking graph-global symbolic indices into instruction-local
extent resolution.

`GraphCompiler` currently specializes placeholder descriptors to concrete
shapes, so most graph-level equalities are proved or disproved during compile.
The lower-level compiler still accepts symbolic input shapes and retains
guards. Keeping guards in `ExecProgram` is therefore part of the contract for
low-level and future polymorphic compilation, rather than dead infrastructure.

The executor checks ordered input count and metadata, then all shape guards,
before upload, deferred-zero synthesis, backend workspace/session creation, or
extension dispatch. A rejected guard cannot cause partial backend execution.

### Cache, checkpoint, and AD persistence

Normalized guard relation and operands participate in the execution-program
fingerprint and compile-cache key. Two otherwise identical instruction streams
with different shape contracts cannot share an entry. Provenance is diagnostic
rather than semantic: on a hit, the current compilation's guard vector replaces
the cached vector so family and instruction diagnostics describe the current
graph.

Checkpoint roots preserve the existing constraint history while replacing the
materialized leaf and metadata scope. JVP, VJP, and direct primal-VJP graph
construction cross the runtime/AD boundary through an opaque
`ConstraintScopeTransfer`. The transfer is a persistent `Arc`-backed chain:
cloning is constant-time, parent histories remain shared, empty scopes are not
retained, and pointer deduplication happens once on compiler materialization.
New linear, residual, and transposed graph scopes are layered over inherited
primal, tangent, or cotangent histories. Transform-cache cold construction and
hot reuse therefore preserve the same contract and produce the same typed
diagnostic.

### Current adopters and defense in depth

- Ordinary einsum records equality for every repeated label, including the
  direct extension path and core-op expanded fast path.
- Sparse matmul records payload NNZ requirements and exact primal/tangent/
  cotangent shape equalities.
- Tropical einsum records repeated-label equality and its JVP/VJP exact-shape
  contracts.

Sparse and tropical host-reference implementations also validate concrete
count, dtype, rank, and shape at their direct execution boundary. Graph guards
do not replace that host-side defense.

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

Graph-facing configs need a symbolic form where shape-derived sizes can appear.
The current implementation uses `StdTensorOp::GatherDynamicSliceSizes` and the
matching `ExecOp::GatherDynamicSliceSizes`:

```text
GatherDynamicSliceSizes
  offset_dims: Vec<usize>
  collapsed_slice_dims: Vec<usize>
  start_index_map: Vec<usize>
  index_vector_dim: usize
  slice_sizes: Vec<DimExpr>
```

Lowering from graph config to backend config resolves symbolic slice sizes
against concrete runtime inputs immediately before dispatch. Backends still see
the existing concrete `GatherConfig`.

This keeps the layering clean:

- AD and compiler code may express symbolic config sizes.
- Execution resolves them at the backend boundary.
- Backend kernels stay optimized for concrete sizes.

## Scatter Transpose Policy

`transpose_scatter` builds inverse gather `slice_sizes` from the primal updates
shape:

1. If all required update-window extents are concrete, keep emitting the
   existing concrete inverse gather.
2. If an extent is symbolic, emit `GatherDynamicSliceSizes` and add the updates
   tensor as a non-differentiable shape-source input.

The generated dynamic gather is AD-closed: its forward rule applies the same
dynamic gather to the operand tangent, and its transpose emits the same inverse
scatter as concrete `Gather` while returning `None` for indices and shape
sources.

## Compiler Pass Contract

Compiler passes must state what shape guarantee they require.

| Consumer | Required guarantee | Rule |
|---|---|---|
| Rank checks | exact rank | Rank is always exact metadata. |
| `Transpose` metadata | any known extents | Permute extent metadata without changing guarantees. |
| `BroadcastInDim` execution shape | exact target extents | Reject or defer if any target extent is not exact. |
| `Reshape` execution shape | exact target extents | Never use upper bounds as reshape sizes. |
| `DotDecomposer` merge reshapes | runtime shape inputs for execution; best extent metadata | Emit reshape parameters from actual input shapes, and propagate exact/upper-bound/unknown metadata without upgrading guarantees. |
| DCE and last-use analysis | no extent guarantee | Shape metadata is irrelevant. |
| Diagnostics | best available | May print exact, upper-bound, or unknown metadata. |

The important invariant is that an optimization pass may become conservative,
but it may not silently reinterpret upper-bound metadata as exact metadata.

## AD Contract

AD rules must not call `constant_value().unwrap_or_else(panic)` for user-reachable
symbolic shapes.

There are two acceptable outcomes for newly touched user-reachable symbolic
shape paths:

- emit a graph using exact symbolic metadata or runtime shape-source
  expressions such as `DimExpr::InputDim`
- return an unsupported-dynamic-shape error

AD rules should choose the narrowest metadata query that matches the graph they
emit. Use rank metadata for axis-count checks and runtime shape-source
expressions for broadcast, reshape, or dynamic gather parameters that should
follow the actual tensor shape. Require an exact shape only when constructing a
concrete op payload that cannot represent runtime dimensions.

Current AD rule signatures are not uniformly `Result`-returning. The
implementation should introduce one shared error channel before adding future
AD paths that cannot be expressed as graph ops.

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

1. Done: add the metadata contract types or equivalent internal representation.
   Preserve current exact behavior for existing static programs.
2. Done: mark `DynamicTruncate`'s truncated axis as an upper bound instead of exact.
   Update consumers so they do not use that extent as a concrete shape
   parameter.
3. Done: replace `transpose_scatter`'s symbolic update-window panic with a
   dynamic graph gather config.
4. Done: add graph-facing symbolic config support for gather-like `slice_sizes`.
5. Done: update rank-only AD transpose/JVP paths, including contraction,
   structural, scatter, and linalg solve rules, to use rank metadata and runtime
   shape sources instead of exact-shape metadata.
6. Deferred: add structured AD construction errors for future dynamic or
   unsupported shape requirements.
7. Deferred: add exact runtime scalar extents for `DynamicTruncate` once the compiler and
   execution layers can resolve them safely.

Each step should add focused regression tests. The first tests should cover:

- `DynamicTruncate` metadata no longer being treated as exact pre-truncation
  shape.
- `transpose_scatter` with symbolic update-window dimensions no longer
  panicking.
- compiler passes refusing to build `Reshape` or `BroadcastInDim` parameters
  from upper-bound extents.

## Open Questions

- Should graph-facing symbolic gather config eventually replace
  `StdTensorOp::Gather` directly, or should `GatherDynamicSliceSizes` remain as
  the narrow dynamic variant?
- Which public transform APIs should surface unsupported dynamic-shape AD
  errors first?

The preferred bias is to keep public surface narrow until the compiler and AD
contracts settle.
