# Tropical Einsum Shared Planning Design

## Context

Issue #212 asks for an optimized tropical AD forward path with argmax capture.
The issue text refers to the old `extension/tenferro-tropical` crate and its
`tropical_einsum_forward_with_argmax` helper. That crate no longer exists on
`main`: it was renamed to `extension/tenferro-ext-tropical`, later recreated as
`ext/tropical`, and then removed by #900 when the workspace dropped the root
facade and tightened crate boundaries.

The closest current implementation base is `0e3ffa76^:ext/tropical`. That
crate implemented fused rank-2 tropical dot-general and AD rules, but it did
not use the optimized argmax path from `omeinsum-rs`.

`omeinsum-rs` provides the relevant execution pattern:

- classify binary contractions into left-only, right-only, contracted, and
  batch modes;
- lower the operands to GEMM-like column-major matrices;
- dispatch `MaxPlus`, `MinPlus`, and `MaxMul` `f32`/`f64` cases to
  `tropical-gemm` when possible;
- keep a generic fallback that also returns argmax indices.

## Decision

Restore tropical support as an extension crate, but do not fork ordinary einsum
planning logic. Instead, expose a small semiring-neutral lowering surface from
`tenferro-einsum` and let `tenferro-ext-tropical` delegate all subscript,
contraction-tree, shape, and GEMM-layout planning to that surface.

Ordinary einsum and tropical einsum should share planning. They should not
share the final arithmetic kernel, because ordinary einsum is `sum(mul(...))`
while tropical einsum is `max/min(add(...))` and AD needs argmax routing.

## Shared API Shape

Add a public lowering module to `tenferro-einsum` with read-only planning data
derived from the existing internal planner:

- pairwise step operands and output labels;
- diagonal extraction stages for repeated labels;
- pre-reduction plans for labels unique to one operand;
- mode classification: batch, left-only, right-only, contracted;
- GEMM lowering shapes: `[m, k, batch...]`, `[k, n, batch...]`,
  `[m, n, batch...]`;
- target label orders for each operand and canonical output order;
- final permutation requirement.

The surface should remain data-only. It must not expose ordinary einsum's
`ReduceSum`, `Mul`, `DotGeneral`, or graph-builder implementation as the public
extension point. This keeps the contract semiring-neutral and avoids locking
external extensions to ordinary arithmetic.

The likely public entry points are:

- `ContractionTree::step_plan(step_idx)`
- `ContractionTree::step_plans()`
- a new `tenferro_einsum::lowering` module with documented plan structs and
  accessors

The implementation may reuse the existing `planning::plan` structs internally,
but the public module should avoid exposing unnecessary mutability or builder
details.

## Tropical Extension Shape

Recreate `ext/tropical` as a workspace-excluded extension crate first, matching
the old external-extension direction. The crate depends on `tenferro-einsum`
for `Subscripts`, `ContractionTree`, and lowering plans.

The tropical crate owns:

- `TropicalKind` and tropical scalar/newtype semantics;
- public traced helpers such as tropical dot-general and tropical reduce-sum;
- runtime execution for tropical pairwise contractions;
- argmax capture for AD forward paths;
- tropical AD rules and tie-breaking semantics.

For each pairwise contraction step, tropical execution should:

1. apply diagonal extraction and tropical pre-reductions from the shared plan;
2. permute and reshape operands according to the shared GEMM lowering;
3. execute tropical GEMM with argmax when the step has contracted modes;
4. use a generic tropical fallback for unsupported dtype/kind/layout cases;
5. reshape and permute the output according to the shared plan.

## Optimized Path

The optimized path is eligible when all of the following hold:

- the contraction step lowers to GEMM with `k > 0`;
- operands can be represented as dense column-major GEMM panels without hidden
  cross-device materialization;
- dtype and tropical kind are supported by the accelerated kernel;
- argmax indices fit the selected index dtype;
- tie-breaking is deterministic and documented.

The first implementation should prefer the old supported semantics
(`MaxPlus`/`MinPlus` over `f32`/`f64`) and leave room for `MaxMul` if the API
can express it cleanly without changing the family contract.

## AD Semantics

The extension crate owns AD. Ordinary einsum AD should not be reused directly:
it differentiates `sum(mul(...))`, not a tropical semiring operation.

For tropical AD, the primal forward path records the winning contracted index
for each output element. The transpose/VJP route scatters cotangents back to
the winning inputs. Tie behavior must match the forward argmax policy and be
covered by tests.

If the current core graph vocabulary cannot express an efficient scatter path
for every case, keep a correctness-preserving fallback using the existing
indicator-mask construction from the removed `ext/tropical` implementation.
The optimized argmax path is allowed to be CPU-runtime-only at first, as long
as unsupported cases fall back explicitly.

## Error Handling

Planning errors should remain `tenferro_device::Error` or the existing
`tenferro-einsum` error shape. Tropical runtime errors should be returned as
`tenferro_tensor::Error` with operation names that identify the tropical
extension path.

Unsupported optimized-kernel cases are not user-visible errors unless no
generic fallback exists. They should fall back to the generic tropical
implementation. Shape, rank, dtype mismatch, invalid repeated-label patterns,
and unsupported AD lowering remain errors.

## Testing

Tests should be layered:

- `tenferro-einsum` tests for the new public lowering API, proving that ordinary
  planning exposes the same step plans it already uses internally;
- tropical correctness tests for values and argmax on rank-2, batched, output
  permutation, and tie cases;
- fallback tests that force unsupported optimized cases and compare values and
  argmax against the generic implementation;
- AD tests that verify cotangent routing for simple unique-winner and tie
  examples;
- doctests for every new public type and function.

Benchmarks should compare generic tropical contraction against the optimized
path for small and medium GEMM shapes. The benchmark should report both value
only and value-plus-argmax execution where possible.

## Out Of Scope

This design does not add tropical operations to the core op vocabulary and does
not make ordinary `DotGeneral` semiring-parametric. It also does not require a
GPU tropical kernel in the first implementation. GPU support can be added later
behind the extension runtime boundary.
