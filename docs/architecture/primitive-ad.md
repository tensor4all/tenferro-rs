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

Core primitive AD rules and user-facing transforms live in `tenferro-ad`.
`tenferro-internal-ops/src/ad/` preserves the shared vocabulary, metadata and
support contracts used by semantic family-rule adapters; its core graph
emitter and registry are retired.

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

## Core Rules And Shared Vocabulary

Core derivatives are implemented in `tenferro-ad/src/semantic_transform.rs`
and its helper modules. There is no `PrimitiveAdRule` graph registry.

`PrimitiveRuleBuilder`, `PrimitiveTransposeInput`, rule errors and
`ResidualSpec` retain their `tenferro_ops::ad::` paths. Their private definition
owner is `tenferro-internal-ops/src/primitive_rules/`; core residual declarations
are metadata, not derivative emitters. Family-rule adapters may still record
primitive operations through the builder and use `ShapeGuardContext` metadata.
They must not execute tensors or inspect backend runtime state.

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
and oracle coverage. Dispatch belongs to semantic transforms in `tenferro-ad`,
not the retired core graph dispatcher. The shared extension callback type is
retained as vocabulary, not a second active dispatch path.

## Closure Responsibility

All operations emitted by core or extension AD rules must remain in the
supported tenferro semantic vocabulary. If a rule needs an operation outside
that vocabulary, add the operation and its validation/oracle coverage first
instead of smuggling backend-specific execution through AD rule code.
