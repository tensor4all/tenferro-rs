# Algebra And External Numeric Extensions

**Status:** current design note
**Related:** `../spec/extension-op.md`, `../spec/primitive-catalog.md`,
`../spec/ad-contract.md`

## Current Boundary

The mainline tenferro graph is not algebra-parameterized. Core traced
programs use the flat `StdTensorOp` vocabulary and execute through the standard
`ExecOp` / backend pipeline.

The following substrate is retired and must not be reintroduced as a parallel
mainline path:

- `tenferro-algebra` as an in-tree core crate
- `SemiringOp<Alg>` / `SemiringOpKind`
- `SemiringOps`
- `SemiringBackend<Alg>`
- `compile_semiring_to_exec` / `eval_semiring_ir`

Historical references to those names describe an older extension strategy.
Current extension work should use one of the patterns below.

## Supported Patterns

### Core-Op Composition

If a non-standard numeric operation can be expressed as existing core ops, the
preferred path is a wrapper that builds a `StdTensorOp` graph. This keeps the
operation inside the normal compiler, executor, and AD pipeline.

Example: max-plus tropical matrix multiply can be expressed as:

```text
BroadcastInDim(lhs) + BroadcastInDim(rhs) -> ReduceMax(contracting_axis)
```

The composition path gets AD from the underlying core ops. It is also the best
first implementation for proving user-facing semantics before introducing a
fused op.

### Fused `ExtensionOp`

If composition is too expensive or too awkward to package, external crates may
contribute a fused op via `StdTensorOp::Extension(Arc<dyn ExtensionOp>)`.

The `ExtensionOp` contract is normative in
[`../spec/extension-op.md`](../spec/extension-op.md). The important constraints
are:

- identity, hashing, and equality live on the trait object
- shape and dtype inference must be total over concrete and symbolic metadata
- eager and compiled execution both route through the extension's execution
  method
- extension AD must emit only core `StdTensorOp` values, extension families
  with AD rules before a later AD pass reaches them, or documented terminal
  first-order helper families

Example: the nested `ext/tropical` crate provides fused binary tropical
einsum over max-plus and min-plus semirings. It reuses
`tenferro-einsum` parsing and lowering plans, owns tropical arithmetic and
first-winner argmax metadata, and may dispatch matmul-shaped contractions
through the optional `tropical-gemm` feature.

### Scalar Newtypes

External crates may define scalar newtypes such as `MaxPlus<T>`,
`MinPlus<T>`, or `MaxMul<T>` to document and test algebraic scalar semantics.
Those newtypes are useful for direct scalar arithmetic and for eager helper
experiments.

They are not, by themselves, a mainline tensor execution contract. The public
tensor scalar surface is intentionally narrow and sealed by the tensor runtime.
Any traced or backend-visible behavior still needs either core-op composition
or an `ExtensionOp`.

## AD Policy

AD is defined for the standard core graph. External numeric behavior gets AD in
one of two ways:

- composition wrappers inherit the AD rules of the core ops they emit
- fused `ExtensionOp` implementations register `linearize` and
  `transpose_rule` rules. Those rules must lower to core ops or to registered
  extension helper families whose own AD rules are available before a later AD
  pass reaches them. Terminal first-order helper families may omit AD rules when
  the owning extension documents that higher-order AD through that helper is
  unsupported.

There is no separate AD system for arbitrary algebra-parameterized graphs.

## When To Add A Core Op

A fused `ExtensionOp` is intentionally narrower than a built-in primitive. A
new core op is justified only when all of the following hold:

- the pattern is repeatedly useful beyond one experiment
- composition has clear semantic or performance costs
- an out-of-tree fused extension cannot reasonably share the needed backend
  infrastructure
- the op can be specified in the core primitive catalog and kept AD-closed

Benchmark evidence for tropical matmul should live with the extension crate
that owns the fused operation.
