# AD Contract

**Date:** 2026-04-04
**Status:** Draft
**Parent:** [`../index.md`](../index.md)
**Related:** [`primitive-catalog.md`](primitive-catalog.md), [`../architecture/chainrules.md`](../architecture/chainrules.md), [`../architecture/tidu.md`](../architecture/tidu.md)

---

## Purpose

This document is the normative specification for the AD trait contract that
concrete primitives must satisfy. It owns the `PrimitiveOp` trait signature
and the rules that `linearize` and `transpose_rule` must follow.

For the AD pipeline architecture (differentiate, transpose, higher-order AD),
see [`../architecture/ad-pipeline.md`](../architecture/ad-pipeline.md).

For the AD trait design rationale, see
[`../architecture/chainrules.md`](../architecture/chainrules.md).

---

## PrimitiveOp trait (canonical signature)

Defined in `chainrules-rs/src/primitive_op.rs`. Extends `GraphOp` with
the constraint `Self::InputKey: ADKey`.

```rust
pub trait PrimitiveOp: GraphOp
where
    Self::InputKey: ADKey,
{
    /// Returns the addition operation used for cotangent accumulation.
    /// tidu's `transpose` emits `Op::add()` nodes when multiple cotangents
    /// flow to the same value.
    fn add() -> Self where Self: Sized;

    /// Emit the linear (JVP) rule for this primitive.
    ///
    /// Must be linear in tangent inputs. May reference primal inputs/outputs
    /// through `External(GlobalValKey)`. Must emit ops in `OpMode::Linear`.
    fn linearize(
        &self,
        builder: &mut FragmentBuilder<Self>,
        primal_in: &[GlobalValKey<Self>],
        primal_out: &[GlobalValKey<Self>],
        tangent_in: &[Option<LocalValId>],
    ) -> Vec<Option<LocalValId>>
    where
        Self: Sized;

    /// Emit the transpose rule for this linear primitive.
    ///
    /// Receives cotangent outputs and produces cotangent inputs.
    /// Must only emit ops that themselves implement `PrimitiveOp`.
    fn transpose_rule(
        &self,
        builder: &mut FragmentBuilder<Self>,
        cotangent_out: &[Option<LocalValId>],
        inputs: &[ValRef<Self>],
        mode: &OpMode,
    ) -> Vec<Option<LocalValId>>
    where
        Self: Sized;
}
```

## ADKey trait (canonical signature)

Defined in `chainrules-rs/src/ad_key.rs`. Required bound on
`PrimitiveOp::InputKey`.

```rust
pub trait ADKey: Clone + Debug + Hash + Eq + Send + Sync + 'static {
    /// Create a tangent input key derived from this key.
    /// `pass` is a unique identifier for the `differentiate` call.
    fn tangent_of(&self, pass: DiffPassId) -> Self;
}
```

`DiffPassId` is `u64`.

## LinearFragment (canonical definition)

Defined in `tidu-rs/src/linear_fragment.rs`. Returned by
`tidu::differentiate` (which internally calls `PrimitiveOp::linearize`
per op node — note that `linearize` itself returns
`Vec<Option<LocalValId>>`, not `LinearFragment`; the fragment is
assembled by `differentiate`).

```rust
pub struct LinearFragment<Op: GraphOp> {
    /// The fragment containing linear ops.
    pub fragment: Fragment<Op>,
    /// (primal_input_key, tangent_local_val_id) pairs.
    pub tangent_inputs: Vec<(Op::InputKey, LocalValId)>,
    /// Tangent outputs, aligned with requested outputs.
    /// None means the corresponding output is inactive.
    pub tangent_outputs: Vec<Option<LocalValId>>,
}
```

## Rules

1. **Closure**: `linearize` and `transpose_rule` must emit only ops that
   themselves implement `PrimitiveOp`. This is the sole closure requirement.
   tenferro-rs is responsible for satisfying it.

2. **Cotangent accumulation**: when a value fans out to multiple consumers,
   tidu's `transpose` accumulates cotangents via `Op::add()`. This means
   `Add` must implement `PrimitiveOp` and its transpose rule must be the
   identity (cotangent passes through to both inputs).

3. **Linear ops**: an op whose `linearize` returns itself (identity tangent
   map) only needs a `transpose_rule`. Examples: `Transpose`, `Reshape`,
   `BroadcastInDim`.

4. **Primal reuse**: `linearize` may reference primal values via
   `External(GlobalValKey)` in the fragment builder. These are resolved
   during `materialize_merge` so that shared primal computations are not
   duplicated.

5. **No AD for custom algebra**: `SemiringOp<T>` does NOT implement
   `PrimitiveOp`. AD is only available for `StdTensorOp` (standard algebra).

## Owned by this document

- `PrimitiveOp` trait signature
- Closure rule
- Cotangent accumulation rule
- Linear op rule
- Primal reuse rule

Other documents link here for the AD contract; they do not re-state
these definitions.
