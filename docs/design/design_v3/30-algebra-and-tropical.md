# Algebra And Tropical In Design V3

## Summary

`v3` separates three concerns that are currently entangled:

- eager scalar typing
- traced graph vocabulary
- tropical support

The recommendation is:

- keep scalar-level genericity where it is real
- remove semiring genericity from the traced graph and AD core
- position tropical as an external extension and composition story

## Current State

The repository currently exposes:

- `tenferro-algebra` with `HasAlgebra`, `Algebra`, `Semiring`, and `Standard<T>`
- `SemiringBackend<Alg>` for algebra-generic eager execution over typed tensors
- `SemiringOp<Alg>` plus a separate semiring compile and execute path

The critical architectural observation is that this graph-level semiring path
is not the mainline traced AD substrate. It is an auxiliary path.

## Recommended Role Of `tenferro-algebra`

`v3` does not need to delete `tenferro-algebra` immediately. However, it
should narrow the crate's architectural role and force an explicit decision
during migration.

Near-term stance (graph and AD layer):

- graph-level design must stop depending on the idea that every traced op path
  is parameterized by an algebra
- `SemiringOp`/`SemiringBackend` are demoted (see "Recommended Fate Of
  `SemiringOp`" below)

Open binary choice for the crate itself:

- **Option A — delete `tenferro-algebra`**: if `Semiring`, `Algebra`,
  `HasAlgebra`, and `Standard<T>` have zero callers once the graph and AD
  layer stops using them, the crate has no residual reason to exist.
- **Option B — reduce to eager-only convenience**: keep scalar traits and
  `Standard<T>` strictly as eager / UX sugar and remove everything
  graph-facing.

`v3` does not pre-commit to A or B. The decision gate is scheduled as part of
the migration plan (see `90-migration-plan.md`, Stage 3 exit criterion). What
matters now is that the ambiguity is closed before Stage 4 begins.

## Recommended Fate Of `SemiringOp`

`SemiringOp<Alg>` should be treated as non-mainline and eventually retired from
the central traced-graph story.

Reasons:

- it has a separate compile path
- it is not the graph type used by `TracedTensor`
- it does not integrate with mainline AD
- it makes the architecture look more generic than it really is

This does not mean algebra-aware eager experiments become impossible. It means
they should stop dictating the shape of the traced AD design.

## Tropical Design

### Principle 1: Composition First

Tropical traced operations should first be implemented as compositions of core
primitives.

Examples:

- max-plus add is `Maximum`
- max-plus multiply is standard `Add`
- max-plus reduction is `ReduceMax`

For contractions, the default traced lowering should be a composition such as:

```text
BroadcastInDim + Add + ReduceMax
```

or the appropriate variant for the chosen tropical flavor.

Full lowering of `max-plus matmul`:

```text
External crate call
───────────────────
    tropical_dot_general(a, b)         ← user-facing
            │
            │  lowered by Stage 4 wrapper
            ▼
Core primitives (inside tenferro)
───────────────────
    BroadcastInDim(a, shape_of(b))
            │
            ▼
    BroadcastInDim(b, shape_of(a))
            │
            ▼
           Add
            │
            ▼
    ReduceMax(contract_axis)
            │
            ▼
    existing core AD rules handle the backward pass
    (no new AD math required)
```

### Principle 2: Externalize The Surface

The tropical user-facing API should live outside the core workspace, or at
least outside the core workspace members, so it exercises only the public
surface.

This gives the repository two benefits:

- tropical remains an extension, not a hidden workspace privilege
- CI can use it as a contract test for whether the public extension surface is
  actually sufficient

### Principle 3: Eager And Traced Are Different Layers

Eager tropical support and traced tropical support do not need the same
mechanism.

Recommended split:

- **Eager path**: scalar newtypes `MaxPlus<T>`, `MinPlus<T>`, `MaxMul<T>`
  whose Rust arithmetic trait impls (`Add`, `Mul`, `Zero`, ...) drive the
  existing `TypedTensor<T>` T-generic kernels. This path does not need
  `SemiringBackend<Alg>` and is unaffected by the Stage 6 removal of that
  abstraction.
- **Traced path**: wrapper functions lowering to core primitives.

The traced path should not require the graph value itself to become
`Tensor<MaxPlus<T>>`.

The scalar-newtype eager story is independent of the rest of `v3` and can
proceed at any time outside the staged plan; it needs no core change.

## Fused Tropical Support

Fused tropical support is a committed part of the migration plan, not a
speculative future add-on.

- **Stage 4a (Phase 1, concrete shapes)** ships tropical support via
  composition of core primitives on concrete-shape inputs, in an external
  crate. Stage 4a is scoped to the traced `tenferro` facade only; eager
  tropical typed-tensor work is separate.
- **Stage 4b (Phase 1, symbolic shapes)** extends the same composition to
  symbolic-shape inputs, acting as the contract test for Stage 3's
  symbolic-AD correctness work.
- **Stage 7 (Phase 2)** adds `FusedTropicalDotGeneral` to that same
  external crate as an `ExtensionOp`, registered via the Stage 6
  mechanism.
- the Stage 7 AD path is argmax-based: `linearize` records argmax
  indices; `transpose_rule` emits `Gather` / `Scatter` on the core op
  vocabulary, so AD closure on the core op set is preserved.

The fused primitive therefore lives in the external crate and never
appears as a core op variant. This keeps the fast path as a packaging
decision rather than a second semantic substrate inside the core.

See `90-migration-plan.md` Stages 4a/4b and 7 for the full acceptance
criteria and `40-extension-boundary.md` Recipe B for the authoring flow.

## Design Position

The main architectural decision is:

> Tropical support is not evidence that the traced graph should become
> algebra-parameterized.

It is evidence that the core graph vocabulary should be rich enough to express
chooser-style compositions cleanly and that the extension boundary must be
usable from outside the core workspace.
