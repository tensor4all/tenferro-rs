# Shape Metadata In Design V3

## Summary

`v3` adopts the direction of issue `#741`: input shape snapshots should move
off most op variants and onto value-side metadata.

This is one of the safest and most clearly incremental parts of the broader
refactor.

## The Three Categories

`v3` keeps the same useful classification:

| Category | Meaning | Examples | Keep on op? |
|---|---|---|---|
| A | Structural identity | `axes`, `perm`, `dims`, diagonal axes | Yes |
| B | Required output shape | reshape target, broadcast target | Yes |
| C | Input shape snapshot | `input_shape`, `lhs_shape`, `rhs_shape`, ranks derived from inputs | No |

The design rule is:

> Categories A and B stay on the op. Category C moves to value metadata.

### Before / After

```text
               Op payload (today / v2)
               ┌─────────────────────────┐
               │ A: structural params    │   ← kept
               │ B: required output      │   ← kept
               │ C: input-shape snapshot │   ── moves ──┐
               └─────────────────────────┘              │
                                                        │
                                                        ▼
               Value-side metadata (v3)
               ┌─────────────────────────┐
               │ shape  (DimExpr list)   │   ◀── shape_of(value)
               │ dtype                   │   ◀── dtype_of(value)
               └─────────────────────────┘
```

## Why This Matters

Moving Category C metadata off the op has immediate benefits:

- smaller op payloads
- better op hashing and interning behavior
- less duplicated state to keep synchronized
- cleaner AD rules, because they read input metadata from values instead of
  trusting construction-time snapshots

This is particularly important for reductions, linalg ops, and dot-general
variants that currently carry input-shape history on the op itself.

## Proposed Value-Side Metadata

The traced stack should expose value metadata through one small abstraction.

Minimal required fields:

```text
TensorMeta
  dtype: DType
  shape: [DimExpr]
```

The exact Rust type name is not important. What matters is that both traced
graph construction and transpose/linearization infrastructure can query:

- input dtype
- input symbolic or concrete shape

## Emitter And Builder Queries

The builder and emitter boundary should grow metadata accessors so AD rules can
stay generic over traced graph building and eager execution.

Required conceptual operations:

```text
shape_of(value) -> &[DimExpr]
dtype_of(value) -> DType
metadata_of(value) -> TensorMeta
```

The first two are sufficient for the initial migration. A richer `TensorMeta`
handle can be added later if it clarifies downstream code.

## Interaction With Existing Runtime Types

`v3` does not require inventing a new dynamic tensor value type. The runtime
already has a `Tensor` enum. The missing piece is abstract metadata attached to
graph values, not another concrete tensor container.

This is why the shape-metadata cleanup should happen before any larger op
vocabulary redesign. It gives the repository the right source of truth without
forcing unrelated API churn.

## Recommended Migration Order

Baseline: step 2 below already landed in `#737` (via `#664`). The remaining
safest order is:

1. introduce metadata queries on builder and emitter APIs
2. ~~remove `lhs_rank` and `rhs_rank` from traced `DotGeneral` variants~~ —
   **done in `#737`**
3. remove reduction `input_shape` fields
4. remove `Reshape::from_shape` and `NaryEinsum::n_inputs` where derivable
5. remove linalg input-shape snapshots
6. remove `TriangularSolve` shape snapshots

Each step should preserve behavior and keep tests focused on proving that the
shape-sensitive AD paths still produce identical results. Oracle-replay
baselines must stay green across every step.

## Design Position

This part of `v3` is straightforward normal evolution. It aligns the codebase
with what the traced graph already knows and removes historical debt rather
than adding new mechanism.
