# Extension Boundary In Design V3

## Summary

Out-of-tree extension support is desirable, but `v3` should not commit to a
naive `Arc<dyn ExtensionOp>` design.

> **Relation to the superseded `#740` Principle 5**: this chapter is a
> *phased counter-proposal* to the direct
> `TensorOp::Extension(Arc<dyn ExtensionOp>)` escape hatch originally
> sketched in `#740`. `v3` accepts the motivation but defers the concrete
> ABI until op identity is specified. See "Safe Phased Position" below.

The current graph substrate inherits `computegraph` requirements:

- op values must be clonable
- op values must be hashable
- op values must support equality

Any extension mechanism that ignores op identity will create instability in
interner keys, graph materialization, and caching.

## Current Recommendation

The first `v3` milestone should not include a fully generic extension ABI.

Instead:

1. external crates should be able to build traced functionality by composing
   core primitives directly
2. optional fused primitives should be considered only after a concrete
   performance case exists
3. a generic extension mechanism should be revisited only when op identity is
   fully specified

This is the safest design boundary for the initial architecture cleanup.

## Why A Raw Trait Object Is Not Enough

A design like:

```text
TensorOp::Extension(Arc<dyn ExtensionOp>)
```

looks simple, but it leaves critical questions unanswered:

- What makes two extension ops equal?
- How are extension params hashed?
- How does serialization identify the op family?
- How does the runtime decide whether two graph nodes are the same operation?
- How do caches stay stable across processes or versions?

Without explicit answers, the design is underspecified at the exact point
where the graph engine needs determinism.

## Safe Phased Position

### Phase A: Composition-Only External Extensions

External crates define new user-facing wrappers, but they lower directly to the
core op vocabulary. This already covers a large fraction of realistic
extensions, including tropical compositions.

### Phase B: Core-Owned Fused Ops

If a pattern proves important and repeatedly performance-critical, the core can
gain a dedicated fused op variant. Its AD rule should normally remain a
decomposition into core ops.

### Phase C: Revisit Generic Extension Support

Only after the previous two phases expose a real need should the repository
introduce a generic extension substrate.

At that point, the design should require explicit op identity, such as:

- a stable extension family identifier
- a hashable and equality-supporting parameter payload
- a clear serialization and compatibility story

Whether this is modeled as a trait object, a registered descriptor, or a small
extension enum is secondary to the identity contract.

## Required Properties Of Any Future Extension Mechanism

Any future design must satisfy all of these:

- explicit equality semantics
- explicit hashing semantics
- stable family identification
- AD rules emit only core ops
- eager and traced execution boundaries remain understandable
- failure behavior is explicit when a backend lacks a forward implementation

## Relation To Tropical

Tropical is a useful pressure test here.

The recommended reading is:

- tropical composition should work without a generic extension ABI
- optional fused tropical forward paths may justify a future extension design
- tropical alone is not a reason to rush an underspecified trait-object op
  boundary into the graph core

## Design Position

`v3` deliberately treats generic extensions as a deferred design problem.

This is not avoidance. It is an attempt to keep the first refactor focused on
the parts that are already well-justified by the current codebase:

- one core op vocabulary
- value-side metadata
- tropical by composition
