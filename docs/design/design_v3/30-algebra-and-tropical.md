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

- eager: scalar newtypes and generic kernels where the backend can operate on
  the scalar's arithmetic traits
- traced: wrapper functions lowering to core primitives

The traced path should not require the graph value itself to become
`Tensor<MaxPlus<T>>`.

## Fused Tropical Support

If performance later requires a fused tropical contraction primitive, the first
choice should be:

- keep the tropical surface in the external crate
- make the fused forward path optional
- keep AD as decomposition to core primitives unless profiling proves that
  decomposition-based AD is too expensive

This keeps the fast path as a performance detail rather than a second semantic
substrate.

## Design Position

The main architectural decision is:

> Tropical support is not evidence that the traced graph should become
> algebra-parameterized.

It is evidence that the core graph vocabulary should be rich enough to express
chooser-style compositions cleanly and that the extension boundary must be
usable from outside the core workspace.
