# AD Contract

**Date:** 2026-05-28
**Parent:** [`../index.md`](../index.md)
**Related:** [`primitive-catalog.md`](primitive-catalog.md), [`../architecture/primitive-ad.md`](../architecture/primitive-ad.md), [`../architecture/tidu.md`](../architecture/tidu.md)

---

## Purpose

This document is the normative specification for the AD trait contract that
concrete primitives must satisfy. It owns the `Primitive` trait signature
and the rules that `linearize` and `transpose_rule` must follow.

For the AD pipeline architecture (linearize, linear_transpose, higher-order AD),
see [`../architecture/ad-pipeline.md`](../architecture/ad-pipeline.md).

For the AD trait design rationale, see
[`../architecture/primitive-ad.md`](../architecture/primitive-ad.md).

---

## Primitive trait (canonical signature)

Defined in `tidu-rs/src/rules/primitive_op.rs`. Extends `GraphOperation` with
the constraint `Self::InputKey: ADKey`.

```rust
pub trait Primitive: GraphOperation
where
    Self::InputKey: ADKey,
{
    type ADContext: Default;

    /// Returns the addition operation used for cotangent accumulation.
    /// tidu's `linear_transpose` emits `Op::add()` nodes when multiple cotangents
    /// flow to the same value.
    fn add() -> Self where Self: Sized;

    /// Emit the JVP rule for this primitive.
    fn jvp_rule(
        &self,
        builder: &mut impl PrimitiveBuilder<Self>,
        primal_inputs: &[ValueKey<Self>],
        primal_outputs: &[ValueKey<Self>],
        tangent_inputs: &[Option<LocalValueId>],
        ctx: &mut Self::ADContext,
    ) -> Vec<Option<LocalValueId>>
    where
        Self: Sized;

    /// Emit the transpose rule for this linear primitive.
    fn transpose_rule(
        &self,
        builder: &mut impl PrimitiveBuilder<Self>,
        cotangent_outputs: &[Option<LocalValueId>],
        inputs: &[PrimitiveValue<Self>],
        role: &OperationRole,
        ctx: &mut Self::ADContext,
    ) -> Vec<Option<LocalValueId>>
    where
        Self: Sized;
}
```

## ADKey trait (canonical signature)

Defined in `tidu-rs/src/rules/ad_key.rs`. Required bound on
`Primitive::InputKey`.

```rust
pub trait ADKey: Clone + Debug + Hash + Eq + Send + Sync + 'static {
    /// Create a tangent input key derived from this key.
    /// `pass` is a unique identifier for the `linearize` call.
    fn tangent_of(&self, pass: DiffPassId) -> Self;
}
```

`DiffPassId` is `u64`.

## LinearizedGraph (canonical definition)

Defined in `tidu-rs/src/linearized_graph.rs`. Returned by
`tidu::linearize` (which internally calls `Primitive::jvp_rule`
per operation node — note that `jvp_rule` itself returns
`Vec<Option<LocalValueId>>`, not `LinearizedGraph`; the graph is
assembled by `linearize`).

```rust
pub struct LinearizedGraph<Op: GraphOperation> {
    graph: Graph<Op>,
    tangent_inputs: Vec<(Op::InputKey, LocalValueId)>,
    tangent_outputs: Vec<Option<LocalValueId>>,
}

impl<Op: GraphOperation> LinearizedGraph<Op> {
    pub fn as_graph(&self) -> &Graph<Op>;
    pub fn tangent_inputs(&self) -> &[(Op::InputKey, LocalValueId)];
    pub fn tangent_outputs(&self) -> &[Option<LocalValueId>];
}
```

## Rules

1. **Closure**: `linearize` and `transpose_rule` must add only ops that
   themselves implement `Primitive`. This is the sole closure requirement.
   tenferro-rs is responsible for satisfying it.

2. **Cotangent accumulation**: when a value fans out to multiple consumers,
   tidu's `linear_transpose` accumulates cotangents via `Op::add()`. This means
   `Add` must implement `Primitive` and its linear_transpose rule must be the
   identity (cotangent passes through to both inputs).

3. **Linear ops**: an op whose `linearize` returns itself (identity tangent
   map) only needs a `transpose_rule`. Examples: `Transpose`, `Reshape`,
   `BroadcastInDim`.

4. **Primal reuse**: `linearize` may reference primal values via
   `External(ValueKey)` in the graph builder. These are resolved
   during `materialize_merge` so that shared primal computations are not
   duplicated.

5. **Extension AD boundary**: built-in AD is defined for `StdTensorOp`.
   `StdTensorOp::Extension` may participate in AD only when its operation
   family registers an extension AD rule. Missing extension rules must report
   unsupported AD; they must not silently drop or zero gradients.

## Owned by this document

- `Primitive` trait signature
- Closure rule
- Cotangent accumulation rule
- Linear op rule
- Primal reuse rule

Other documents link here for the AD contract; they do not re-state
these definitions.
