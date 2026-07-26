# Primitive AD Rule Structure

**Repo:** tenferro-rs
**Parent:** `../index.md`
**Related:** `semantic-ad.md`, `ad-pipeline.md`,
`../spec/ad-contract.md`

---

## Purpose

This page explains how tenferro represents primitive-local automatic
differentiation rules after U8. The canonical signatures and normative rules
live in [`../spec/ad-contract.md`](../spec/ad-contract.md).

Core primitive AD rules are tenferro-owned. They live under
`tenferro-internal-ops/src/ad/` and are registered by
`PrimitiveOpKind`. The user-facing AD surfaces and semantic extension rule set
live in `tenferro-ad`.

## Rule Kinds

tenferro keeps the linearize-then-transpose model:

- `linearize` emits the JVP graph for one primitive.
- `transpose_rule` emits cotangent-input graph fragments for already-linear
  primitive flow.

Reverse-mode support is not always a direct transpose arm on the original
primal op. Many primitives become reverse-differentiable by first applying
`linearize` and then transposing the emitted linear graph. Direct primal VJP
rules are reserved for cases where that generic route is incomplete or too
expensive.

## Core Rule Registry

The core registry maps each `PrimitiveOpKind` to a `PrimitiveAdRule`:

```rust
pub(crate) trait PrimitiveAdRule: Send + Sync {
    fn kind(&self) -> PrimitiveOpKind;
    fn linearize(...) -> ADRuleResult<Vec<Option<LocalValueId>>>;
    fn transpose_rule(...) -> ADRuleResult<Vec<Option<LocalValueId>>>;
}
```

Rules emit graph operations through `PrimitiveRuleBuilder`. They may reference
primal values, metadata values, and symbolic shape sources supplied by
`ShapeGuardContext`, but they must not execute tensors or inspect backend
runtime state.

`None` in a tangent or cotangent slot means symbolic zero flow. Rules should
keep zero flow symbolic until they must pass an actual value to another
primitive, then instantiate it through dtype-aware semantic helpers such as
`zero_like`, `one_like`, `constant_scalar`, or `identity_matrix`.

## Transpose Inputs

Transpose rules receive typed input references so they can distinguish fixed
metadata/residual operands from active linear flow:

- residual or metadata inputs may be used as fixed operands;
- linear inputs belong to tangent flow;
- a linear input's primal counterpart may be used only for metadata, runtime
  shape sources, or fixed coefficients independent of tangent flow.

Rules must not turn an active linear value into a residual operand to keep the
forward tangent sweep alive in the transposed graph.

## Extension Rules

Extension operation families register semantic rules through
`SemanticExtensionRuleSet` in `tenferro-ad`. Family crates own their formulas
and oracle coverage. The private bridge from core graph-rule dispatch to
semantic extension rules is implemented by `tenferro-ad` only; extension
families should not depend on it.

## Closure Responsibility

All operations emitted by core or extension AD rules must remain in the
supported tenferro semantic vocabulary. If a rule needs an operation outside
that vocabulary, add the operation and its validation/oracle coverage first
instead of smuggling backend-specific execution through AD rule code.
