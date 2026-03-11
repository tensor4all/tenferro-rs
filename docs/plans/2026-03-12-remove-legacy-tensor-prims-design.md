# Remove Legacy TensorPrims Design

## Goal

Delete the legacy `TensorPrims<A>` execution contract, `PrimDescriptor`, and
`Extension` from the workspace in one PR, with no compatibility adapter left
behind. After this cutover, the only primitive execution contracts are the
family traits:

- `TensorSemiringCore`
- `TensorSemiringFastPath`
- `TensorScalarPrims`
- `TensorAnalyticPrims`
- `tenferro-linalg-prims` for structured linalg kernels

## Constraints

- No speculative backward compatibility.
- No internal shim that keeps `TensorPrims<A>` alive after the PR.
- `tenferro-tropical` must be cut over in the same PR so the workspace does not
  retain a split between "standard algebras use family traits" and "extension
  algebras use legacy prims".
- `tenferro-dyadtensor`, `tenferro-capi`, and `tenferro-tropical-capi` must be
  updated in the same PR so no downstream crate depends on the deleted surface.
- Production code must remain CPU/GPU generic at the API boundary. Missing GPU
  support is expressed through capability queries, not concrete backend checks.

## Current Legacy Surface

The legacy contract still exists in three forms.

1. The legacy public surface itself.
   - `tenferro-prims/src/lib.rs`
   - `TensorPrims<A>`
   - `PrimDescriptor`
   - `Extension`

2. Blanket adapters that make the new family traits defer back to the legacy
   surface.
   - `tenferro-prims/src/semiring_core.rs`
   - `tenferro-prims/src/semiring_fast_path.rs`
   - `to_legacy()` helpers

3. Downstream call sites and bounds that still require the legacy trait.
   - `tenferro-capi/src/lib.rs`
   - `extension/tenferro-dyadtensor/src/api/contracts.rs`
   - `extension/tenferro-dyadtensor/src/api/runtime_dispatch.rs`
   - `extension/tenferro-tropical/src/prims.rs`
   - `extension/tenferro-tropical-capi/src/lib.rs`
   - tests and docs across the workspace

## Target Architecture

### tenferro-prims

`tenferro-prims` becomes family-native.

- Family descriptors stay public:
  - `SemiringCoreDescriptor`
  - `SemiringFastPathDescriptor`
  - `ScalarPrimsDescriptor`
  - `AnalyticPrimsDescriptor`
- Each backend directly implements the relevant family traits.
- `PlanCache` remains, but its keys are family descriptors rather than the
  deleted monolithic descriptor.

The implementation shape should move toward:

- `semiring_core.rs` and `semiring_fast_path.rs` define contracts only
- `scalar_prims.rs` and `analytic_prims.rs` define contracts only
- backend code holds family-native planning and execution

The PR does not need to fully re-file every backend family into separate files
if that would delay the cutover, but the resulting code must not reintroduce a
monolithic legacy dispatcher.

### tenferro-tropical

`tenferro-tropical` directly implements:

- `TensorSemiringCore<MaxPlusAlgebra<_>>`
- `TensorSemiringCore<MinPlusAlgebra<_>>`
- `TensorSemiringCore<MaxMulAlgebra<_>>`

and, if useful, `TensorSemiringFastPath<_>` with truthful `false` capability
reporting for unsupported fast paths.

`TropicalPlan<T>` may remain as the tropical backend plan type.

### tenferro-einsum

`tenferro-einsum` depends only on:

- `TensorSemiringCore`
- `TensorSemiringFastPath`

It must not mention `TensorPrims`, `PrimDescriptor`, or `Extension`.

### tenferro-dyadtensor

The runtime contract aliases become family-based.

- `EinsumRuntimeValue` requires semiring families, not `TensorPrims`
- scalar and analytic runtime bounds remain on their family traits
- capability checks use `has_fast_path`, `has_scalar_support`, and
  `has_analytic_support`

### tenferro-capi

The only primitive it currently needs directly is `MakeContiguous`. That path
should become a semiring-core call.

### tenferro-tropical-capi

The tropical C API switches its generic bounds from `TensorPrims<Alg>` to
`TensorSemiringCore<Alg>`.

## Capability Model

Deleting `Extension` means every capability check must move to the family that
owns it.

- semiring core has no capability query; support is encoded by the impl
- semiring fast path uses `has_fast_path`
- scalar family uses `has_scalar_support`
- analytic family uses `has_analytic_support`
- linalg keeps `has_linalg_support`

This is cleaner and removes the last cross-family "extension registry" concept.

## Testing and Docs

Legacy-oriented tests must be rewritten, not removed without replacement.

- `tenferro-prims/tests/prims_tests.rs` should become family-oriented tests
- `tenferro-tropical` tests should validate semiring-core behavior directly
- any test asserting legacy bridge behavior should be deleted and replaced by
  direct family contract tests

Docs must stop describing the workspace as "in migration" from `TensorPrims`.
The new family traits are the design, not the transition.

At the end of the PR:

- production code has zero references to `TensorPrims`, `PrimDescriptor`, and
  `Extension`
- tests have zero references to those names
- active docs (`docs/api_index.md`, `docs/design/**`, rustdoc) have zero
  references to them except in explicitly historical notes if any remain under
  `docs/plans/`

## Non-Goals

- No custom CUDA pointwise/reduction kernel work
- No new broad feature expansion beyond what is needed for the contract cutover
- No historical plan rewrites under `docs/plans/`
