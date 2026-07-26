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

- `computegraph-rs` remains the generic graph substrate used by the core sweep.
- `tenferro-internal-ops` owns the `StdTensorOp` primitive AD vocabulary:
  `PrimitiveRuleBuilder`, `PrimitiveAdRule`, `PrimitiveTransposeInput`,
  `ADRuleError`, and the core op `linearize` / `transpose_rule` registry.
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

Core primitive rules are implemented for `StdTensorOp` under
`tenferro-internal-ops/src/ad/`.

The current rule vocabulary is:

```rust
pub trait PrimitiveAdRule: Send + Sync {
    fn kind(&self) -> PrimitiveOpKind;

    fn linearize(
        &self,
        op: &StdTensorOp,
        builder: &mut dyn PrimitiveRuleBuilder,
        primal_in: &[ValueKey<StdTensorOp>],
        primal_out: &[ValueKey<StdTensorOp>],
        tangent_in: &[Option<LocalValueId>],
        ctx: &mut ShapeGuardContext,
    ) -> ADRuleResult<Vec<Option<LocalValueId>>>;

    fn transpose_rule(
        &self,
        op: &StdTensorOp,
        builder: &mut dyn PrimitiveRuleBuilder,
        cotangent_out: &[Option<LocalValueId>],
        inputs: &[TransposeInputRef<'_>],
        mode: &OperationRole,
        ctx: &mut ShapeGuardContext,
    ) -> ADRuleResult<Vec<Option<LocalValueId>>>;
}
```

Rules emit graph operations through `PrimitiveRuleBuilder`; they do not execute
tensors and must not read backend state. `ShapeGuardContext` carries shape
sources, active-output information, extension-rule dispatch, and the metadata
needed to emit validation-preserving derivative graphs.

## Extension Rule Boundary

Extension AD is semantic. An operation-family crate registers rules in a
`SemanticExtensionRuleSet`; `tenferro-ad` owns lookup and dispatch. Missing
extension rules must surface typed unsupported-AD errors rather than silently
dropping tangent or cotangent flow.

During the transition after U8, a small context-owned bridge still lets the
core graph sweep call semantic extension rules. That bridge is internal to
`tenferro-internal-ops` and implemented only by `tenferro-ad`; family crates
must not implement it directly.

Some linalg semantic AD paths still reuse legacy rule-recording helpers behind
the semantic rule surface. That is tracked by #1468. It is a cleanup of rule
construction ownership, not a reason to reintroduce retired external AD-engine
names as active dependencies or documentation sources of truth.

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
  StdTensorOp, PrimitiveAdRule registry, core linearize/transpose rules,
  PrimitiveRuleBuilder, PrimitiveTransposeInput, ShapeGuardContext

tenferro-ad
  AdContext, SemanticExtensionRuleSet, semantic JVP/VJP transforms,
  eager/traced AD user APIs, AD transform cache

operation-family crates
  semantic extension AD rules and numerical oracle coverage
```
