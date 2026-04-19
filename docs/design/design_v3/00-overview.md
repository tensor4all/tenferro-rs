# Design V3 Overview

## Purpose

`v3` is a proposal to simplify the traced graph and AD architecture without
restarting the repository from scratch.

The key observation is that the current codebase already has a strong center of
gravity (baseline = post-`#732` 1-layer IR state):

- traced execution is built around `StdTensorOp`
- the runtime already has a dynamic `Tensor` enum for concrete values
- AD rules are written against one graph vocabulary, not against
  algebra-parameterized ops
- `SemiringOp` exists, but today it is a side path rather than the main traced
  and AD substrate

The goal of `v3` is to make that reality explicit and to remove the machinery
that currently suggests a more generic graph model than the implementation
actually supports.

## Problems In The Current Shape

### 1. Mainline traced AD is `StdTensorOp`-centric

The traced frontend, lowering path, and `linearize` / `transpose_rule` logic
are all organized around `StdTensorOp`. The current architecture does not
provide a real algebra-generic AD path.

### 2. Shape data is duplicated between ops and values

Many op variants store input-shape snapshots that are derivable from the value
graph. This increases op size, weakens hashing, and creates synchronization
work for graph transforms.

### 3. The graph-level algebra abstraction is misleading

`SemiringOp<Alg>` and `SemiringBackend<Alg>` suggest a symmetric alternative
graph model. In practice they are useful for eager-style algebra experiments
and tests, but they are not the substrate that powers traced AD.

### 4. Tropical support should not force a second traced graph stack

Tropical operations such as max-plus matmul can usually be expressed as a
composition of existing core primitives. The graph layer should not gain a
parallel semiring substrate just to support this case.

### 5. Extension pressure exists, but the safe boundary is not yet defined

The desire for out-of-tree fused primitives is real. However, `computegraph`
requires `GraphOp: Clone + Hash + Eq`, so a naive `Arc<dyn ExtensionOp>` escape
hatch is not yet a sound design.

## Design Goals

### Goal A: One core graph vocabulary

The traced graph and AD pipeline should revolve around one core op enum. This
vocabulary may grow over time, but there should be one mainline graph language
for traced execution.

### Goal B: Value metadata as the source of truth

Input shape and dtype metadata should live with values and be queryable from
AD and lowering infrastructure. Op variants should keep only the parameters
that define operation identity.

### Goal C: Preserve the v2 tensor API layering

The public tensor types from `v2` — `EagerTensor<B>`, `TracedTensor`, the
dynamic `Tensor` enum, and the static `TypedTensor<T>` — all remain in place.
`v3` is not a tensor-layer redesign; it is an op-vocabulary and metadata
consolidation. None of the four types is renamed, merged, or removed by this
proposal.

### Goal D: Reduce false genericity

If the repository only has one real traced-AD graph substrate, the design
should say so clearly. Genericity should be preserved where it is real and
removed where it is only aspirational.

### Goal E: Make tropical support first-class without infecting the core

Tropical support should fit as:

- eager scalar newtypes where useful
- traced compositions over core primitives by default
- optional fused implementations only when performance justifies them

## Architectural Direction

The full tensor layering from `v2` is preserved. `v3` only changes what sits
inside the core op vocabulary and how shape metadata is attached to values.

```text
User-facing layer (unchanged from v2)
    EagerTensor<B>               TracedTensor
    eager + autograd             traced graph + AD
             │                          │
             └────────────┬─────────────┘
                          ▼
Runtime value layer (unchanged from v2)
                     Tensor enum
                          │
                          │  downcast on kernel entry
                          ▼
Kernel-facing layer (unchanged from v2)
                    TypedTensor<T>
                          │
                          ▼
Core TensorOp vocabulary     ← v3 changes concentrated here
                          │
       ┌──────────────────┼──────────────────┐
       ▼                  ▼                  ▼
 shape/dtype         AD rules emit      optional fused
 metadata            only core ops      ops only when
 from values (v3)    (v3 clarification) justified
```

This is not a proposal to introduce a completely new runtime object model.
None of the v2 tensor types is touched. The primary refactor is conceptual
and architectural, and it is scoped to:

- shrink the op model to what the traced stack really needs
- move shape metadata to the value side
- demote algebra-generic graph paths from the mainline design

## What `v3` Does Not Require

`v3` does not require:

- replacing `computegraph`
- replacing `tidu`
- replacing `PrimitiveOp`
- replacing or renaming the v2 tensor types (`EagerTensor`, `TracedTensor`,
  `Tensor`, `TypedTensor`)
- immediately deleting every algebra-related API
- introducing a generic extension ABI in the first phase

The best interpretation is an architectural consolidation, not a wholesale
runtime rewrite.

## API Compatibility For Standard Users

For users of `TracedTensor` and `EagerTensor` who operate on Standard-algebra
scalars (`f32`, `f64`, `Complex32`, `Complex64`), `v3` preserves full
source-level compatibility through the `tenferro` facade crate.

The guarantees:

- `TracedTensor` has no type parameter, and no public method gains an algebra
  type bound.
- `EagerTensor<B: TensorBackend = CpuBackend>` stays parameterized only over
  backend. No `impl` block gains an algebra bound.
- The `tenferro` facade (`pub use` in `tenferro/src/lib.rs`) does not
  re-export `Algebra`, `Semiring`, `HasAlgebra`, or `Standard<T>`. Code that
  imports from `tenferro::{...}` never names those traits.
- Helper functions (`matmul`, `cholesky`, `svd`, `qr`, `eigh`, `solve`,
  `eig`, `lu`, ...) keep their current signatures.
- `std::ops::{Add, Mul, Neg}` impls for `&EagerTensor<B>` are preserved.

The `tenferro-algebra` crate decision (Option A delete / Option B reduce, see
`30-algebra-and-tropical.md`) therefore does **not** affect any user who only
imports from the `tenferro` facade. It affects only code that directly
imports `tenferro-algebra`, `SemiringBackend`, or `SemiringOp`.

The one residual breakage surface is config structs that carry Category C
fields today. Category C removal is a deliberate migration step (see
`20-shape-metadata.md`); it already caused one shipped change in `#737` where
`DotGeneralConfig` lost `lhs_rank`/`rhs_rank`. Later steps in the same series
may remove additional Cat C fields. This is not a policy introduced by `v3` —
it is the existing cleanup direction continuing.

## Recommended Decisions

The proposal set recommends:

1. Keep one core traced op vocabulary and route AD through it.
2. Treat `SemiringOp` as non-mainline and plan to phase it out from the traced
   architecture.
3. Land value-side shape metadata cleanup before larger AD refactors.
4. Support tropical via composition first.
5. Defer a general extension mechanism until op identity is fully specified.
