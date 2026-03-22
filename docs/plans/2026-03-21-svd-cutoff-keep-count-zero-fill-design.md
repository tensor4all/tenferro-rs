# SVD Cutoff Keep-Count Zero-Fill Design

**Status:** approved in-session on 2026-03-21.

## Goal

Enable `tenferro-linalg::svd(..., cutoff)` to preserve its fixed-shape zero-fill
semantics on both CPU and CUDA without any ad hoc GPU→CPU payload transfer.

## Problem

`svd(..., cutoff)` currently computes truncation by reading singular values on
the host, deriving `actual_k`, and rebuilding `u/s/vt` with host loops. That is
not acceptable for the generic CUDA path.

The hard part is that the truncation decision lives in `s: Tensor<T::Real>`,
while the payload to be zero-filled may live in `u/vt: Tensor<T>`, where `T`
may be complex. A same-dtype numeric-mask helper is not enough.

## Design Decisions

### 1. Keep-count route, not full mask-select

We will not build a general cross-dtype `select(mask, on_true, on_false)`
surface first.

Instead:

- derive batch-local `keep_counts`
- zero-fill trailing regions after `keep_counts`
- keep tensor shapes fixed

This is the smallest reusable substrate that solves `svd(..., cutoff)` without
overdesigning the scalar family.

### 2. Helper ownership

The reusable execution primitive belongs in `tenferro-tensor`, not in
`tenferro-linalg` or `tenferro-prims`.

Reasoning:

- the operation is structural tensor materialization
- it must work for both `Tensor<T::Real>` and `Tensor<T>`
- `tenferro-linalg` should keep only thin orchestration helpers
- `tenferro-prims` should not gain a distorted cross-dtype family abstraction

`tenferro-linalg/src/backend/tensor_helpers.rs` may add a thin wrapper that
translates SVD-specific axes and shapes into the tensor-level primitive.

### 3. Helper API

The tensor-level helper is out-of-place:

```rust
pub fn zero_trailing_by_counts<R>(
    &self,
    keep_counts: &Tensor<R>,
    axis: usize,
) -> Result<Tensor<T>>
where
    T: Scalar,
    R: Scalar,
```

Contract:

- `keep_counts.dims()` must equal the batch dims of `self`
- `keep_counts` must live in the same memory space and on the same device as
  `self`
- each element of `keep_counts` must be an integer-valued count
- each count must satisfy `0 <= count <= self.dims()[axis]`
- the result keeps the same shape as `self`
- elements with coordinate `coord_axis >= count` are zero-filled

### 4. Execution layering

Execution stays layered:

- CPU path: generic tensor-side loop in `tenferro-tensor`
- CUDA path: `tenferro-tensor` delegates to a Layer 0
  `tenferro-device::cuda::runtime` kernel
- `tenferro-linalg` never touches raw device pointers

This preserves the rule that shared low-level runtime ownership lives in
`tenferro-device`.

### 5. `svd(..., cutoff)` rewrite shape

`tenferro-linalg::svd(..., cutoff)` will be rewritten as:

1. compute backend `thin_svd`
2. apply `max_rank` using existing `narrow` views
3. derive `keep_counts` from `s` and `cutoff`
4. zero trailing `u` on axis 1
5. zero trailing `s` on axis 0
6. zero trailing `vt` on axis 0

This preserves the current public semantics:

- shapes remain fixed
- trailing singular vectors/values are zeroed
- no host fallback is introduced

## Error Policy

- shape mismatch between payload and `keep_counts`: error
- non-integer `keep_counts` value: error
- negative count: error
- count larger than the selected axis length: error
- unsupported device path: error, never fallback

## Testing Strategy

The first RED/GREEN sequence should be:

1. `tenferro-tensor` CPU correctness for `zero_trailing_by_counts`
2. `tenferro-device` CUDA kernel correctness for trailing zero-fill
3. `tenferro-tensor` CUDA correctness on complex payload + real counts
4. `tenferro-linalg` `svd_cutoff_fixed_shape_zero_fill_semantics_hold`

## Non-Goals

- no public bool tensor type
- no full cross-dtype select primitive in this phase
- no ad hoc host-side rebuild path for CUDA
- no change to the public `svd` signature or `SvdOptions`
