# Homogeneous Tape and PyTorch-Style Scalar AD Design

**Date**: 2026-03-07

## Summary

Remove `DynTape` and the heterogeneous AD query surface, and make
`Tape<V>` the only reverse-mode tape API in `chainrules`.

The resulting AD model is:

- homogeneous graphs only: one tape carries exactly one value type `V`
- external users can still differentiate custom types via `Tape<MyType>`
- tensor scalars follow PyTorch semantics: a scalar tensor is rank-0
  (`shape=[]`), not shape `[1]`
- tensor elementwise scalar operations are convenience overloads over the
  tensor-tensor broadcast path
- implicit reverse seed creation continues to use `num_elements() == 1`

This keeps the extensibility of the generic `Differentiable` model while
removing the mixed-type graph machinery that complicates the API, tests, and
documentation.

## Current Problems

### 1. Two tape models split the AD contract

The current design exposes both `Tape<V>` and `DynTape`.

- `Tape<V>` is a monomorphic, type-safe graph
- `DynTape` is a heterogeneous graph for mixed runtime-erased value types

This duplication spreads the AD contract across two APIs, two option types,
two query families, and two documentation narratives.

### 2. `DynTape` is solving a problem we no longer need

The original justification for `DynTape` was mixed-type graphs and a fallback
for some higher-order cases where `V::Tangent != V`. The target product
direction is now narrower:

- homogeneous custom-type graphs are sufficient
- mixed graphs such as `Tensor<f64>` with user-defined scalar structs in the
  same tape are not required

Given that narrowed requirement, `DynTape` adds maintenance cost without
supporting a needed user workflow.

### 3. Scalar semantics are underspecified

The current AD and tensor narratives do not clearly separate:

- tensor-operation scalar semantics
- reverse-mode seed omission semantics

These are different in PyTorch. Tensor scalar operations use rank-0 tensors,
while implicit reverse seeds are allowed whenever the differentiated output has
exactly one element.

### 4. Current design documentation still describes the old coexistence model

`docs/design/autodiff.md`, `docs/design/index.md`, `docs/design/einsum-dyadtensor.md`,
and `docs/AD/index.md` still describe `Tape<V>` and `DynTape` as coexisting
first-class surfaces. That would become incorrect after the simplification.

## Design Goals

- Keep external custom-type differentiation via `Tape<MyType>`
- Remove heterogeneous graph support and its public API
- Align tensor scalar semantics with PyTorch
- Keep reverse-mode seed omission behavior consistent with current
  `num_elements() == 1` logic
- Reduce documentation and test surface area
- Avoid introducing a new scalar abstraction beyond the existing tensor and
  `Differentiable` interfaces

## Non-Goals

- No support for mixed-type graphs in one tape
- No attempt to preserve old `DynTape` higher-order fallback behavior
- No shape-`[1]` scalar special casing in tensor APIs
- No redesign of `Differentiable`, `ReverseRule`, or `ForwardRule`

## Chosen Architecture

### Single tape model

`chainrules` exposes only the monomorphic tape:

- `Tape<V>`
- `TrackedTensor<V>`
- `DualTensor<V>`
- `Variable<V>`

`DynTape`, `DynVariable`, `DynTangent`, `DynBackwardOptions`,
`DynHvpOptions`, `autograd::grad_dyn_tangent`, and
`autograd::grad_dyn_variable` are removed.

### Graph invariant

Each graph is homogeneous.

- a `Tape<V>` may contain only values of type `V`
- `Tape<MyType>` remains supported for downstream custom AD use cases
- mixing `Tensor<f64>` and `MyType` in the same graph is unsupported by
  construction

This is the only capability reduction. Homogeneous custom-type graphs remain
fully supported.

## Tensor Scalar Semantics

Tensor APIs follow PyTorch conventions.

### Scalar tensor definition

A tensor scalar is a rank-0 tensor:

- scalar tensor: `shape=[]`
- one-element vector: `shape=[1]`

`shape=[1]` is **not** treated as scalar. It participates in ordinary shape
and broadcast rules.

### Elementwise scalar overloads

Tensor-level scalar overloads such as `tensor + scalar` or `tensor * scalar`
remain valid convenience APIs, but they are not a distinct semantic category.

Implementation guidance:

- the canonical implementation path is tensor-tensor elementwise execution
- scalar convenience inputs are normalized to a rank-0 tensor operand or to a
  backend scalar parameter when the backend has a dedicated fast path
- the semantic model remains equivalent to the rank-0 tensor path

This matches PyTorch's `wrapped_scalar_tensor` approach for `Tensor + Scalar`
overloads.

## Reverse-Mode Seed Semantics

Reverse-mode seed omission uses the `Differentiable` contract, not tensor rank.

### Seed omission rule

Implicit seed creation is allowed when `num_elements() == 1`.

This is already the behavior of:

- `Tape::pullback`
- `Tape::hvp`
- `Variable::backward`
- `Variable::backward_hvp`
- `autograd::grad_tangent`
- `autograd::grad_variable`

This rule remains unchanged.

### Why this differs from tensor scalar semantics

For tensors, the user-facing scalar meaning is rank-0. For generic AD, the
engine only knows `Differentiable::num_elements()`. Those are intentionally
separate concepts.

Examples:

- `Tensor<f64>` with `shape=[]` is scalar for tensor operations and also has
  `num_elements() == 1`
- `Tensor<f64>` with `shape=[1]` is **not** scalar for tensor operations, but
  still has `num_elements() == 1` and may use implicit reverse seed creation
- a custom `MyType` can define `num_elements() == 1` without having any tensor
  notion of rank at all

## Higher-Order AD Contract

Removing `DynTape` simplifies the higher-order story.

- `grad_tangent` remains a detached, first-order-only API
- `grad_variable(create_graph=true)` remains supported only when
  `V::Tangent = V`
- if `V::Tangent != V`, higher-order graph-connected gradients are
  unsupported in this phase
- `backward_hvp(create_graph=true)` remains unsupported unless implemented
  separately later

The previous design text that routed `V::Tangent != V` higher-order use cases
to `DynTape` must be removed from the current documentation.

## Elementwise Operation Model

Tensor elementwise implementation should converge on one conceptual path:

- tensor-tensor broadcast kernel is the canonical execution model
- scalar overloads are sugar over that model
- no shape-`[1]` scalar special cases are introduced

This keeps tensor semantics aligned with PyTorch and avoids fragile branching
between "scalar" and "tensor" code paths.

For custom `V`, `chainrules` itself does not impose scalar-operation rules.
Those remain the responsibility of the type's `Differentiable` implementation
and the operation-specific rules built on top of it.

## Documentation Policy

### New design record

Add this design record under `docs/plans/` as the historical record for the
decision.

### Update current specification docs

Update current, non-historical docs to match the new design, including at
least:

- `docs/design/autodiff.md`
- `docs/design/index.md`
- `docs/design/einsum-dyadtensor.md`
- `docs/AD/index.md`
- `docs/api_index.md`

Other `docs/design/` pages that mention `DynTape` or heterogeneous AD should be
updated if they are part of the current architecture narrative.

### Do not rewrite historical planning records

Files under `docs/plans/` other than this new document remain historical
records and should not be edited to match the new architecture.

## Migration Scope

### Code

Remove:

- `DynTape`
- `DynVariable`
- `DynTangent`
- dyn-specific backward/HVP options
- dyn-specific autograd query functions
- heterogeneous registry and bookkeeping

Preserve:

- `Tape<V>`
- `TrackedTensor<V>`
- `DualTensor<V>`
- `Variable<V>`
- custom `Differentiable` extension points

### Tests

Delete or rewrite tests that exist only for heterogeneous `DynTape` behavior.

Keep and expand tests for:

- homogeneous `Tape<MyType>` custom-type graphs
- tensor scalar semantics using rank-0 tensors
- one-element-but-non-rank-0 outputs requiring normal tensor semantics while
  still allowing implicit reverse seeding where applicable

### Docs and examples

Public docs should show:

- `Tape<MyType>` as the custom extensibility path
- rank-0 tensor scalar examples for tensor AD APIs

They should no longer describe `DynTape` as a supported fallback or primary
surface.

## Recommended Implementation Order

1. Remove `DynTape` types, options, registry, query functions, and tests from
   `extern/chainrules`
2. Update `chainrules` crate docs and public examples
3. Update `docs/design/`, `docs/AD/`, and `docs/api_index.md`
4. Verify that remaining AD tests cover homogeneous custom-type use cases and
   tensor rank-0 scalar behavior

## Decision

Adopt a single reverse-mode tape model based on homogeneous `Tape<V>` graphs,
retain generic custom-type extensibility, and standardize tensor scalar
semantics on PyTorch-style rank-0 tensors.
