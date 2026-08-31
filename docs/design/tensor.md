# Tensor Representation

`tenferro-tensor-core` and `tenferro-tensor` split the dense tensor contract
along a backend boundary.

## Crate Split

`tenferro-tensor-core` owns backend-independent metadata and host-only
adapters:

- `TensorLayout<R>`, `Rank<N>`, `DynRank`, and `TensorRank`,
- `ShapeVec`, `StrideVec`, and checked layout validation,
- `DType`, `TensorScalar`, `HostTensor<T>`, dynamic host `Tensor`,
  `HostTensorView`, `TensorView`, and `TensorRef`.

It must not expose a public `TypedTensor` alias and must not depend on backend
buffers, CUDA, BLAS/LAPACK providers, execution traits, runtime caches, or AD.

`tenferro-tensor` owns runtime/backend-capable tensor values:

- `TypedTensor<T, R = DynRank>` for fixed scalar type and optional static rank,
- dtype-erased dynamic-rank `Tensor`,
- `TypedTensorView<'a, T, R>` and `TypedTensorViewMut<'a, T, R>` for borrowed
  strided views,
- `TensorView<'a>` and `TensorViewMut<'a>` for dtype-erased borrowed views,
- `TensorRead<'a>` and `TensorWrite<'a>` for read/write kernel dispatch over
  either owned tensors or borrowed views,
- placement metadata, backend buffer handles, `TensorBackend`, and backend
  session traits.

CPU execution, CPU kernels, provider selection, and CPU resource pools belong
to `tenferro-cpu`. GPU execution and explicit device transfer helpers belong
to `tenferro-gpu`.

## Layout

Owned runtime tensors are compact column-major. The leftmost dimension has
stride `1`, so compact strides for shape `[d0, d1, d2]` are
`[1, d0, d0 * d1]`.

Arbitrary strides, non-zero offsets, transposes, slices, and reverse layouts
belong to views or `TensorLayout` metadata. Metadata-only transformations use
the `_view` suffix, such as `transpose_view` and `slice_view`.

Owned tensors and views expose layout inspection helpers for migration and
assertion code: compact column-major checks, logical-index to physical-offset
calculation, and layout summaries that include shape, strides, and offset.
Mutable views validate that distinct logical elements do not alias the same
physical element at construction.

When a compact-only operation receives a view, it may canonicalize that view
inside the same placement. Host views can copy to host compact tensors; CUDA
views can copy to CUDA compact tensors. Canonicalization is not a CPU/GPU
transfer mechanism.

## Operation Vocabulary

Unsuffixed operation names take owned compact tensor inputs. APIs that accept
borrowed view inputs or `TensorRead` inputs use a `_read` suffix. Examples
include `add_read`, `reduce_sum_read`, and `dot_general_read`.

Preallocated-output APIs use `TensorWrite` when the output may be either an
owned tensor or a mutable view. These APIs validate output dtype and shape
before writing and do not resize the destination.

Bare `_into` methods overwrite caller-provided outputs. Read-modify-write
updates use `_add_to` for elementwise-style accumulation or `_into_accum` for
dot/GEMM accumulation with an explicit `DotGeneralAccumulation` argument. See
[Output Modes And Write Surfaces](./output-modes.md).

Dot-general accumulation keeps contraction axes and output-update semantics in
separate contracts. `DotGeneralConfig` describes only dimension roles. Output
updates such as `out = alpha * op(lhs) * op(rhs) + beta * out` use
`DotGeneralAccumulation`, including conjugation flags and the floating/complex
`ContractionScalar` coefficients. Cache ownership stays on `SessionCachedDot`
and `BackendCachedDot`; non-cached `TensorDot` methods do not take cache slots.

Metadata-only APIs that produce views use `_view`. APIs that allocate,
execute kernels, canonicalize buffers, or move data must not use `_view`.

## Validated Column-Major Host Access

`ColMajorView<'a, T, N>` and `ColMajorViewMut<'a, T, N>` retain the proof that
a `Rank<N>` tensor is host-resident and compact column-major. Construction
checks the storage and layout boundary once and exposes the shape as
`[usize; N]`. A compact `TypedTensorView<T, Rank<N>>` with a nonzero offset
borrows its exact logical slice without copying.

Safe `iter()` and `axis0_lanes()` traversal operates on slices whose valid
domain is encoded by the iterator. Mutable traversal uses `IterMut` and
`ChunksExactMut`, so safe callers cannot produce overlapping mutable element
references. Checked random access accepts `[usize; N]`; the unsafe accessor
requires every coordinate to be in bounds and repeats no rank, backend,
layout, or `Result` work.

Checked random access uses `get`/`get_mut` and returns `Option`; these views do
not implement `Index` or `IndexMut`, because invalid user coordinates must not
cross a public library boundary as a panic.

The first coordinate varies fastest. These views do not materialize storage,
transfer device data, replace arbitrary-strided kernels, or promise removal of
bounds checks for arbitrary safe random indices.
The prototype does not yet add a constructor on `TypedTensorViewMut`; mutable
validated access currently starts from an owned `TypedTensor`.

## Device Transfer

tenferro never silently transfers tensor payloads between CPU and GPU.
Callers upload CPU tensors before CUDA backend execution and download CUDA
tensors before CPU execution or host value inspection.

Result-returning backend APIs report placement mismatches with
`BackendFailure` diagnostics. Direct host-inspection methods such as
`TypedTensor::host_data()` and `host_data_mut()` return `Result` and report
backend buffers as runtime-state failures.
