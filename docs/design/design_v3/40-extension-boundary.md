# Extension Boundary In Design V3

## Summary

Out-of-tree extension support is a committed part of the `v3` plan, staged
so that the op-identity contract is pinned down **before** any trait-object
mechanism is implemented.

> **Relation to the superseded `#740` Principle 5**: `#740` sketched a direct
> `TensorOp::Extension(Arc<dyn ExtensionOp>)` escape hatch. `v3` accepts the
> motivation and commits to delivering it, but staged: Stage 5 writes the
> contract, Stage 6 implements the carrier, Stage 7 ships the first
> canonical `ExtensionOp` (tropical fused dot-general) as a self-test.

The current graph substrate inherits `computegraph` requirements:

- op values must be clonable
- op values must be hashable
- op values must support equality

Any extension mechanism that ignores op identity will create instability in
interner keys, graph materialization, and caching. The Stage 5 contract
must address these explicitly.

## Phased Plan

The extension mechanism lands in three committed phases plus one optional
stage. Full stage definitions live in `90-migration-plan.md`.

### Phase A — Composition-Only External Extensions (Stage 4)

External crates define user-facing wrappers that lower directly to the core
op vocabulary. This covers a large fraction of realistic extensions,
including the tropical Phase 1 surface.

### Phase B — Generic `ExtensionOp` Substrate (Stages 5–7)

- **Stage 5 (spec)**: write the `ExtensionOp` contract — identity, hashing,
  equality, `Clone`, AD closure, serialization versioning
- **Stage 6 (impl)**: add the `TensorOp::Extension(Arc<dyn ExtensionOp>)`
  variant and a registration API; wire it through engine, compile path,
  eager emitter, and backend dispatch
- **Stage 7 (self-test)**: ship `FusedTropicalDotGeneral` in
  `tenferro-ext-tropical` as the first canonical `ExtensionOp`, with
  argmax-based AD via `Gather` / `Scatter` on core ops

### Phase C (optional) — Core-Owned Fused Primitives (Stage 8)

Only if measured performance evidence demands it, the core can also gain
dedicated fused op variants. This path is evidence-gated and remains off
the critical path.

## Why A Raw Trait Object Needs A Spec First

A declaration like:

```text
TensorOp::Extension(Arc<dyn ExtensionOp>)
```

is simple to write but underspecified on its own. Stage 5 exists precisely
to answer every one of these before Stage 6 starts:

- What makes two extension ops equal?
- How are extension parameters hashed?
- How does serialization identify the op family?
- How does the runtime decide whether two graph nodes are the same operation?
- How do caches stay stable across processes or versions?

Without explicit answers, the graph engine loses determinism at the exact
point where it needs it most. The contract-first staging is how `v3`
protects against that.

## Required Properties Of The `ExtensionOp` Contract

The Stage 5 spec must satisfy all of these:

- explicit equality semantics
- explicit hashing semantics
- stable family identification
- AD closure: `linearize` and `transpose_rule` emit only core ops
- eager and traced execution boundaries remain understandable
- failure behavior is explicit when a backend lacks a forward implementation

The contract is modality-agnostic: whether it is modeled as a trait object,
a registered descriptor, or a small extension enum is secondary to these
properties.

## Relation To Tropical

Tropical is the pressure test that drives Stages 5–7 to ship:

- tropical Phase 1 (Stage 4) proves composition-only extensions work
- tropical Phase 2 (Stage 7) requires the generic `ExtensionOp` mechanism
  for `FusedTropicalDotGeneral` with argmax-based AD
- the same crate acts as the Stage 5 contract self-test — if the spec is
  insufficient for tropical Phase 2, the spec is wrong and must be revised
  before Stage 6 continues

## Op Extension Recipes

The following recipes describe how an external author actually adds a new
primitive at each phase.

### Recipe A — Composition (Stage 4, available today)

For any operation expressible as a composition of core primitives:

1. write a Rust wrapper that takes `&TracedTensor` / `&EagerTensor<B>`
   arguments
2. call existing methods (`add`, `mul`, `reduce_max`, `dot_general`, ...) to
   build the composition
3. rely on the existing AD rules for those core primitives

No core change is required. Tropical `max-plus matmul` lowers to
`BroadcastInDim + Add + ReduceMax` this way.

### Recipe B — Fused `ExtensionOp` (Stages 5–7)

For operations that must be a single fused primitive because the
composition is measurably too expensive or because the AD path needs
information the composition loses (e.g. argmax indices):

1. implement the `ExtensionOp` trait defined in Stage 5 (identity, hash,
   eq, clone, eager execute, `linearize`, `transpose_rule` — AD rules must
   emit only core ops)
2. register the extension with the engine via the Stage 6 registration API
3. ship from an external crate; no core modification required

`FusedTropicalDotGeneral` is the canonical Stage 7 example.

### Recipe C — Core-Owned Fused Op (Stage 8, optional)

Only when an `ExtensionOp` is not enough — either because it is used
broadly across the repository or because profiling shows external dispatch
is itself the bottleneck:

1. add a new variant directly in the core `TensorOp` enum
2. implement `linearize` and `transpose_rule` against the core vocabulary
3. land in the core workspace

This path is reserved for evidence-backed cases.

## Design Position

`v3` commits to a concrete, staged extension mechanism rather than leaving
it as an open-ended deferred question. The staging is:

- one core op vocabulary (Stages 1–3)
- value-side metadata (Stage 1)
- tropical by composition first (Stage 4)
- specified-then-implemented `ExtensionOp` substrate (Stages 5–7)
- optional core-owned fused primitives, evidence-gated (Stage 8)
