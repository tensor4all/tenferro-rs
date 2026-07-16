# Strided Materialization API Convergence Design

## Context

Issue #1393 shows that two public routes with the same materialization
semantics have materially different CPU performance:

- `TypedTensorView::to_contiguous` performs a context-free serial logical
  traversal in `tenferro-tensor`.
- backend structural operations such as `CpuBackend::transpose` materialize
  through the backend buffer pool and `strided-kernel::copy_into` inside the
  configured `CpuContext`.

The split is historical. The context-free method descended from a generic
host-view adapter requiring only `T: Clone`; the backend canonicalization trait
was later added for same-placement CPU/GPU behavior and delegated to that host
adapter on CPU. Subsequent `strided-kernel` parallelization improved structural
operations but did not migrate view canonicalization.

This design treats the performance divergence as an API ownership defect, not
as a request for a second optimized convenience method.

## Decision

For CPU execution, affine-strided traversal and bulk data movement are owned by
`strided-rs`. Tenferro owns tensor semantics, validation, dtype dispatch,
placement checks, `CpuContext`, buffer pooling, and error translation.

Public APIs with the same materialization semantics must converge on one
backend-aware implementation. Tenferro will not retain a context-free serial
fallback under an equivalent public name.

Einsum is the explicit exception to the general CPU delegation rule. Tenferro
keeps its optimized einsum/dot-general preparation and provider integration.
New exceptions require an accepted issue, comparative benchmark evidence, and
a recorded ownership rationale.

## API Surface

### Canonical backend operations

The backend canonicalization capability remains the public execution boundary
and provides two operations:

1. `to_contiguous`: allocate a compact column-major tensor in the same
   placement and copy a readable view into it.
2. `copy_into`: copy from a readable tensor/view into a caller-owned mutable
   tensor/view with matching shape and dtype.

`copy_into` replaces the asymmetric `copy_from_contiguous` contract. The source
and destination may both be strided. Validation occurs before allocation or
kernel dispatch.

The dtype-erased operation must be usable through runtime/backend trait objects.
Typed convenience entry points may remain on concrete backends, but they must
delegate to the same internal implementation and require `TensorScalar`, whose
`Copy + Send + Sync` bounds match `strided-kernel`.

### Removed context-free execution APIs

The following data-moving APIs are removed from the backend-neutral tensor
types:

- `TypedTensorView::to_contiguous`
- `TypedTensorViewMut::copy_from_contiguous`
- `TensorView::to_tensor`
- backend-less `to_tensor` methods on lazy owned-view/value wrappers

Metadata-only methods such as `transpose_view`, `reshape_view`, and
`slice_view` remain on view types. High-level APIs such as `EagerTensor::to_tensor`
remain available because their runtime context owns a backend; their
implementation must call the backend canonicalization operation rather than a
host fallback.

No deprecated compatibility shims are retained. Repository policy prefers
changing the canonical API directly when a public design mismatch is the root
cause.

## CPU Architecture

`tenferro-cpu` gains one pool-aware typed copy helper with this data flow:

```text
TypedTensorView / TensorView
    -> validate host placement, shape, strides, offset, and reachable range
    -> construct strided_kernel::StridedView without copying
    -> acquire compact or caller-provided destination storage
    -> construct strided_kernel::StridedViewMut
    -> strided_kernel::copy_into inside CpuContext::install
    -> wrap the output with tenferro placement and layout metadata
```

The helper supports all tenferro scalar dtypes, rank zero, empty tensors,
singleton axes, positive and negative strides, nonzero offsets, and zero-stride
read-only broadcast views. Mutable destination construction continues to rely
on tenferro's no-overlap validation.

Allocation for `to_contiguous` uses the backend `BufferPool` and does not
zero-initialize storage that the copy kernel fully overwrites. `copy_into`
performs no destination allocation.

## Structural Read Paths

The following CPU read operations consume the original view directly:

- `transpose_read`: permute the original strided metadata and copy once.
- `broadcast_in_dim_read`: apply aligned zero-stride broadcast metadata to the
  original view and copy once.
- `reshape_read`: copy the original logical order once into the required owned
  compact result and attach the requested compact shape; do not first create an
  intermediate tensor with the source shape.
- `materialize_tensor_read`: dispatch directly to the canonical copy helper.

These paths must not first invoke a generic host materializer. Owned compact
inputs may retain existing contiguous fast paths, provided they enter the same
backend/context and pool ownership model.

## Ownership Contract

The durable repository rule is:

| Responsibility | Owner |
|---|---|
| Shape/stride/offset metadata and validation | `tenferro-tensor-core` / `tenferro-tensor` |
| CPU affine-strided copy, permutation, broadcast, map, zip-map, and axis reduction kernels | `strided-rs` |
| Dtype dispatch, placement, CPU thread policy, buffer pool, and error mapping | `tenferro-cpu` |
| Multi-axis reduction orchestration and operation semantics | tenferro, delegating each supported kernel step |
| Gather/scatter and other indirect-index operations without a matching strided primitive | tenferro |
| Einsum/dot-general planning and optimized preparation | tenferro |

When a tenferro CPU operation can be expressed as metadata preparation followed
by an existing `strided-rs` primitive, the bulk traversal must use that
primitive. If the primitive is missing and the operation is generally useful,
the preferred sequence is to add it to `strided-rs` first and then consume it
from tenferro.

## Scope

This change includes:

- canonical CPU `to_contiguous` and general `copy_into` implementation;
- removal and migration of equivalent context-free materialization APIs;
- direct arbitrary-stride `transpose_read`, `broadcast_in_dim_read`, and
  `reshape_read` paths;
- runtime/eager caller migration to backend-owned materialization;
- repository-rule and user/developer documentation updates;
- correctness, source-contract, and focused performance benchmarks.

This change does not rewrite gather, scatter, pad, concatenate, triangular
masks, or diagonal embedding. Their semantics and available strided primitives
need separate focused work. It also does not change GPU kernel behavior beyond
adapting the shared canonicalization trait surface without weakening existing
same-device guarantees.

The external `tenferro-benchmark` suite should be updated separately to invoke
the canonical backend materialization API explicitly.

## Error Handling

- Shape and dtype mismatches return tenferro typed errors before dispatch.
- CPU canonicalization rejects backend/device buffers and does not download
  implicitly.
- GPU canonicalization rejects host buffers and does not upload implicitly.
- Invalid stride/offset/reachable-range metadata remains owned by tenferro's
  layout validation.
- `strided-rs` errors are translated to `Error::BackendFailure` with the public
  tenferro operation name preserved.
- Empty and rank-zero tensors are valid and must not enter unchecked allocation
  or pointer-offset paths.

## Testing And Performance Evidence

Implementation follows test-driven development. Each behavior is introduced by
a failing test before production changes.

Required correctness coverage:

- compact and permuted sources;
- the scattered 24D explicit-stride layout from #1393;
- negative strides and nonzero offsets;
- zero-stride broadcast views;
- singleton, empty, and rank-zero tensors;
- all supported tenferro dtypes;
- caller-owned strided destinations;
- shape/dtype/placement rejection;
- proof that `transpose_read` and `broadcast_in_dim_read` do not perform an
  intermediate host materialization.

Focused benchmarks compare one and four threads for compact, permuted, and
scattered high-rank sources. Small inputs are included to guard dispatch
overhead. The repository benchmark records behavior but does not use unstable
wall-clock assertions in unit tests.

Acceptance requires the full repository verification gates and a refreshed
external #1393 benchmark report. The scattered 24D case should show meaningful
thread scaling and substantially reduce the current gap to direct
`strided-rs`.

## Alternatives Rejected

### Keep both APIs and optimize both

Rejected because a context-free method cannot honor the configured
`CpuContext` and buffer-pool ownership without introducing a hidden backend or
global execution policy. Equivalent public APIs would remain free to diverge.

### Add `strided-kernel` directly to `tenferro-tensor`

Rejected because it moves CPU execution and threading policy into the
backend-neutral contract crate. It would also use ambient Rayon policy when no
backend context is available.

### Keep a generic `T: Clone` serial fallback

Rejected as a public execution API. Tenferro execution scalars already satisfy
`TensorScalar` and the strided kernel bounds. A generic fallback would preserve
the exact performance ambiguity this change removes.
