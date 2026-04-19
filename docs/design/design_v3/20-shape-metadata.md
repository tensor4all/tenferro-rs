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

> **`TensorMeta` is a proposed new type**, not an existing one. It is the
> working name for the carrier that Stage 1 delivers. The type itself does
> not exist in the codebase today — the building blocks (`DimExpr` in
> `tenferro-ops/src/dim_expr.rs`, `SymDim` in `tenferro/src/sym_dim.rs`,
> and the internal `TracedTensor.shape_hint` field at
> `tenferro/src/traced.rs:53`) are already present. Stage 1 lifts these
> into a single public carrier and wires it through AD and lowering. The
> final Rust name may be `TensorMeta`, `ValueMeta`, or another short name
> settled during implementation.

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

The builder and emitter boundary should grow metadata accessors so AD
rules can stay generic over traced graph building and eager execution.

### Required conceptual operations

```text
shape_of(value) -> &[DimExpr]
dtype_of(value) -> DType
metadata_of(value) -> TensorMeta
```

The first two are sufficient for the initial migration. A richer
`TensorMeta` handle can be added later if it clarifies downstream code.

### Where metadata lives in the graph

The per-value metadata must be reachable from the builder, the emitter,
and from AD-rule code without requiring each rule to re-derive it.

Proposed placement:

- **Fragment-side storage**: metadata is attached to each `LocalValId`
  inside the fragment. Whether this is an extension of
  `Fragment<StdTensorOp>` itself or a parallel side table is a Stage 1
  implementation choice; either works as long as lookups are O(1) on the
  hot path.
- **Builder accessors**: `builder.shape_of(val)` and
  `builder.dtype_of(val)` read from the fragment-side storage.
- **Emitter accessors**: the eager `OpEmitter` gains the same accessors
  with the same semantics, so that eager AD paths consume the same API
  as traced AD paths.
- **AD-rule surface**: `ShapeGuardContext`
  (`tenferro-ops/src/std_tensor_op.rs:521-548`) is extended to expose
  shape/dtype queries so that `linearize` and `transpose_rule` never
  need to read op-embedded snapshots.
- **Symbolic inputs**: when a value's shape is symbolic, `shape_of`
  returns a `Vec<DimExpr>` that may contain variables; when concrete,
  all entries resolve to constants via `DimExpr::constant_value`. The
  metadata layer itself never invents a zero-length or
  placeholder-only shape — this is what Stage 1's totality acceptance
  criterion asserts.

Stage 1 must wire these accessors consistently across all three
surfaces (builder, emitter, `ShapeGuardContext`). The existing
`TracedTensor.shape_hint` (`tenferro/src/traced.rs:53`) is an internal
precursor and continues to exist during the migration as a shim; it
becomes derivable from the metadata accessors once Stage 1 is complete.

## Interaction With Existing Runtime Types

`v3` does not require inventing a new dynamic tensor value type. The runtime
already has a `Tensor` enum. The missing piece is abstract metadata attached to
graph values, not another concrete tensor container.

This is why the shape-metadata cleanup should happen before any larger op
vocabulary redesign. It gives the repository the right source of truth without
forcing unrelated API churn.

## Recommended Migration Order

The safest order is:

1. introduce metadata queries on builder and emitter APIs
2. remove `lhs_rank` and `rhs_rank` from `StdTensorOp::DotGeneral`. These
   ranks were already removed from `DotGeneralConfig` in `#737`, but they
   were relocated to the enclosing traced op variant rather than deleted —
   `tenferro-ops/src/std_tensor_op.rs:23-27` still carries them as
   Category C snapshot fields on the op
3. remove reduction `input_shape` fields
4. remove `Reshape::from_shape` and `NaryEinsum::n_inputs` where derivable
5. remove linalg input-shape snapshots
6. remove `TriangularSolve` shape snapshots

Each step should preserve behavior and keep tests focused on proving that
the shape-sensitive AD paths still produce identical results. Oracle-replay
baselines must stay green across every step.

## Relation To JAX's Shape Polymorphism

JAX handles symbolic shapes via `jax.export` (2024 GA) and internal
`_DimExpr` types. The principles `v3` aligns with are narrower than JAX's
full machinery but follow the same direction.

| Mechanism | JAX | `v3` target |
|---|---|---|
| Symbolic dim type | `_DimExpr` (variables + arithmetic + constraints) | existing `DimExpr` / `SymDim` (simpler, no constraint solver) |
| Abstract shape eval | *total* for every primitive | Stage 3 closes existing gaps |
| Shape as first-class value | `jnp.shape(x)` returns traced value | existing `shape_of(axis)` returns scalar `f64` tensor |
| Broadcast with symbolic shape | primitive-level support | Stage 4b adds a public `DimExpr`-accepting variant |
| AD under symbolic inputs | all rules total | Stage 3 fixes the zero-tangent collapse at `tenferro/src/traced.rs:820-833` |
| Opt-in polymorphism UX | `jax.export.export(polymorphic_shapes=...)` | out of scope for `v3` |
| Constraint solver | built into `_DimExpr` | out of scope for `v3` |

`v3` explicitly does not try to replicate JAX's full shape-polymorphism
UX. What it does match is the two correctness invariants that make a
symbolic system viable at all:

- abstract evaluation is total over symbolic inputs
- AD rules emit valid cotangents under symbolic shapes

Stage 3 (symbolic-AD correctness) and Stage 4b (symbolic tropical as the
contract test) close these. The constraint solver and polymorphic export
UX are not on the `v3` critical path and may be revisited as follow-up
work.

## Design Position

This part of `v3` is straightforward normal evolution. It aligns the codebase
with what the traced graph already knows and removes historical debt rather
than adding new mechanism.
