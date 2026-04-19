# AD Model In Design V3

## Summary

The `v3` AD model keeps the current `PrimitiveOp`-based differentiation flow,
but tightens the architectural contract around one core op vocabulary.

The important change is not a new AD engine. The important change is to stop
pretending that traced AD is generic over multiple graph op families when the
current implementation is not.

## Current Observation

Today:

- traced tensors carry `Fragment<StdTensorOp>`
- `linearize` and `transpose_rule` dispatch on `StdTensorOp`
- eager backward reuses the same transpose-rule logic through `OpEmitter`

This is already a clean shape. The problem is that the repository also carries
algebra-generic graph types that look more central than they really are.

## Core Rule

`v3` adopts this rule:

> The traced graph and the cotangent graph share one core operation
> vocabulary. AD rules may only emit operations from that vocabulary.

This keeps three properties simple:

- `PrimitiveOp` closure remains easy to reason about
- eager and traced AD can continue to share transpose-rule logic
- there is no mixed-op fragment problem where the primal and cotangent sides
  use unrelated graph languages

## Cotangent Semantics

For the supported dense tensor path, cotangent accumulation uses standard
linear arithmetic. This remains true even when a forward computation is exposed
through a non-standard surface such as tropical composition.

This should be documented as an invariant of the traced AD substrate:

- primal semantics may include chooser-style or semiring-inspired operations
- cotangent accumulation still lives in the ordinary additive tensor space

This is why tropical composition does not require a second AD algebra.

## Implications For Tropical

For tropical-style traced APIs, the preferred path is:

1. lower the user-facing tropical operation to core primitives
2. reuse existing AD rules for those primitives
3. only introduce a fused primitive if the decomposition is too expensive

If a fused primitive is added later, its AD implementation should still be a
decomposition into core ops unless a stronger performance case is proven.

## What Changes

### Keep

- `PrimitiveOp`
- `tidu::differentiate`
- `tidu::transpose`
- shared traced and eager transpose-rule logic via `OpEmitter`

### Change

- replace the misleading multi-op-family story with one acknowledged mainline
  op family
- remove the expectation that traced AD should work over `SemiringOp<Alg>`
- make AD rules query value metadata rather than shape snapshots stored on ops

## Recommended Core Vocabulary Strategy

The core vocabulary should be a flattened op enum covering:

- structural ops
- elementwise arithmetic and analytic ops
- reductions
- indexing ops
- linalg primitives already present in the traced stack
- optional fused performance ops only when justified

Predicate/select-style tensor ops (such as `Where`) are **intentionally out of
scope** for `v3`. Per `docs/design/supported-ops.md`, a dedicated
boolean/predicate substrate must land before they can be added cleanly, and
that substrate is not part of this proposal set.

The exact enum name is less important than the contract:

- it is the only traced AD op vocabulary
- all AD rules emit only values from this vocabulary
- any future extension mechanism must lower back into this vocabulary for AD

## What Not To Do

The following are explicitly discouraged:

- a separate semiring graph vocabulary for traced AD
- algebra-parameterized cotangent graph fragments
- a generic AD story that requires mixed primal/cotangent op families
- a new AD engine introduced solely to justify extension points

## Design Position

`v3` is a consolidation of the current successful AD path, not a replacement
for it.

The practical recommendation is:

- continue to evolve the current `PrimitiveOp` substrate
- clarify that one core op vocabulary is the traced-AD source of truth
- reduce all adjacent abstractions that imply otherwise
