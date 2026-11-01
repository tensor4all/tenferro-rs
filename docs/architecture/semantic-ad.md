# Semantic AD Architecture

**Repo:** tenferro-rs
**Parent:** `../index.md`
**Related:** `computegraph.md`, `primitive-ad.md`,
`ad-pipeline.md`, `../spec/ad-contract.md`

---

## Purpose

This page records the post-U8 automatic differentiation boundary. The external
AD-engine dependency used before U8 is retired; the generic AD concepts that
used to be imported from it now live at tenferro-owned boundaries:

- `computegraph-rs` remains the generic graph substrate for tracing and local
  rule recording.
- `tenferro-internal-ops` owns the `StdTensorOp` primitive-rule vocabulary,
  shared shape metadata, support helpers, and residual-retention declarations.
  Its old core graph AD emitter and rule registry are retired.
- `tenferro-ad` owns user-facing eager/traced AD APIs, `AdContext`,
  `SemanticExtensionRuleSet`, semantic `Program -> Program` transforms, and
  the AD transform cache.
- Operation-family crates such as linalg, einsum, FFT, sparse, and tropical own
  their semantic extension AD rules and register them with `tenferro-ad`.

The architectural idea is still linearize-then-transpose: reverse mode
transposes already-linear flow instead of inventing an unrelated pullback API.
Current code should name the tenferro interfaces above.

## Current Transform Path

Traced transforms operate on a frozen semantic program:

```text
TraceContext
  -> GraphCompiler
  -> CompiledGraph / FrozenProgram
  -> tenferro-ad semantic JVP or VJP transform
  -> SemanticAdProgram
  -> GraphCompiler / Runtime execution
```

Eager functional transforms record each operation as a small graph/program and
then use the same context-owned transform machinery. Stateful eager reverse mode
adds gradient-slot accumulation on top of that execution path.

`AdContext` is the long-lived owner for:

- copy-on-write `SemanticExtensionRuleSet` values;
- the bounded AD transform cache used by traced transforms;
- the cache handle shared by `EagerRuntime::with_*_and_ad_context`.

Direct `TracedTensorAdExt` helpers remain stateless unless the caller supplies
an `AdContext`.

## Primitive Rule Boundary

Core primitive AD is implemented by `tenferro-ad/src/semantic_transform.rs`
and its helper modules as semantic `Program -> Program` transforms.

`tenferro-internal-ops/src/ad/` remains a compatibility facade and the home of
live context/support helpers, not a second core AD engine. Its existing public
paths retain `PrimitiveRuleBuilder`, `PrimitiveTransposeInput`, `ADRuleKind`,
`ADRuleError`, `ADRuleResult`, `ResidualSpec`, and `ExtensionAdDispatcher`.
The vocabulary definitions and core residual declarations have a private owner
under `tenferro-internal-ops/src/primitive_rules/`.

`primitive_residual_spec` remains retention metadata: all 52 core declarations
are preserved without the old function-pointer registry. It does not emit
or dispatch derivatives. Semantic consumers keep their existing imports.

Linalg semantic rules still use `PrimitiveRuleBuilder` to record primitive
operations without executing tensors or reading backend state.
`ShapeGuardContext` retains shape sources, active-output information, global
metadata scopes, and validation metadata. Its public context contract is not
obsolete merely because the old core graph emitter is gone.

## Extension Rule Boundary

Extension AD is semantic. An operation-family crate registers rules in a
`SemanticExtensionRuleSet`; `tenferro-ad` owns lookup and dispatch. Missing
extension rules must surface typed unsupported-AD errors rather than silently
dropping tangent or cotangent flow.

The old core graph dispatcher no longer calls extension rules.
`ExtensionAdDispatcher` remains part of the shared context vocabulary for path
compatibility; current extension AD dispatch belongs to the semantic rule set
in `tenferro-ad`, not to that retired graph path.

The linalg semantic `linearize` rule emits directly into
`SemanticProgramBuilder`. Reverse-mode linalg paths that are defined by
linearize-then-transpose still record a local linear fragment so the semantic
transpose interpreter can walk that fragment backward. That residual recording
is a local transpose-construction detail, not an active external AD-engine
dependency or documentation source of truth.

## Invariants

- `linearize` is the primitive-local derivative-producing rule.
- `transpose_rule` applies only to already-linear flow.
- Direct primal VJP rules are optional escape hatches for cases where the
  generic linearize-then-transpose path is incomplete or too slow.
- Symbolic zero, one, constant, and identity construction must use semantic
  AD helpers, not analytic tensor operations.
- AD graph emission must be deterministic for the primitive payload, input and
  output metadata, active masks, context, and extension rule set.
- Cached AD transform entries must retain graph/program artifacts only, never
  tensor buffers, backend allocations, or concrete execution outputs.

## Ownership Summary

```text
computegraph-rs
  GraphOperation, GraphBuilder, graph values, operation roles

tenferro-internal-ops
  StdTensorOp, primitive-rule vocabulary, residual declarations,
  PrimitiveRuleBuilder, PrimitiveTransposeInput, ShapeGuardContext/support

tenferro-ad
  AdContext, SemanticExtensionRuleSet, core semantic JVP/VJP transforms,
  eager/traced AD user APIs, AD transform cache

operation-family crates
  semantic extension AD rules and numerical oracle coverage
```
