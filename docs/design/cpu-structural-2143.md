# CPU structural materialization, view metadata, and cached-plan execution (#2143)

Status: **implementation record, in progress on branch `cpu_issue_burn`**. Not yet a
PR. Decision record for the six work items of
[#2143](https://github.com/tensor4all/tenferro-rs/issues/2143), which folds in
#2029, #2021, #2040, #1838, #2039 and #2145. The focused evidence lives in the PR
and in `tensor4all/tenferro-benchmark`; this record states the decisions and what
is verified.

Scope is the CPU structural surface that is not a numerical kernel. Elementwise
kernels (#2140), contraction planning (#2141), linalg drivers (#2142) and the
broader eager-dispatch track (#2139) stay out of scope; where an item overlaps
#2139 item 6 (the zero-metadata representation target), this record states the
shared boundary instead of restating that item.

## Decision summary

| item | issue | decision | state |
|---|---|---|---|
| 1 | #2021 | The eager implicit broadcast is a metadata-only view, on both the typed read adapter and the composite surface. | implemented, needs the #2021 reproducer measurement |
| 2 | #2029 | No code change. The pooled parallel copy is already on `main`; the item needs its paired thread and PyTorch evidence. | not measured |
| 3 | #2040 | Remove the removable per-call allocation in `slice_view` and pin the metadata-allocation count with a test. | implemented and asserted |
| 4 | #1838 | Relax the shared destination contract from compact to write-injective, and add a strided-destination path on CPU and CUDA over the pinned `strided-rs` update primitive. | code and CPU tests done, CUDA verification pending |
| 5 | #2039 | Keep the cached plan, remove the per-call lane allocation and one full pass, and record the engine difference as the reviewed reason for the residual. | implemented, timing evidence pending |
| 6 | #2145 | The tensor keeps its cached compact layout so the per-op view helper borrows it and allocates nothing. Upstream landed the same fix as #2149; this branch keeps the follow-up it did not cover. | implemented and asserted |

## 1. The eager implicit broadcast is a view (#2021)

`tenferro-runtime::typed_tensor::broadcast_to_in_read` called
`session.broadcast_in_dim_read`, which materializes the full broadcast tensor on
CPU, and the composite softmax/normalization broadcasts called it again through
`SessionComposite::broadcast_in_dim`. Every consumer reads the operand.

Decision: the broadcast is metadata on both surfaces.

- `TensorRead::broadcast_in_dim_view` (new `#[doc(hidden)]` boundary in
  `tenferro-tensor`) rebuilds a read's layout with stride 0 on expanding axes,
  keeping the same buffer, retained root and placement. It reuses
  `TensorLayout::broadcast_in_dim_view`, and it refuses a caller-owned
  (`DType::External`) payload with a typed unsupported-dtype error instead of
  reaching the view conversion that has no arm for it.
- `broadcast_to_in_read` maps **all** source axes right-aligned onto the target
  axes. It no longer runs the traced/AD `broadcast_input_plan`, whose
  singleton-removal source shape forced a materializing `reshape_read`. The
  failure vocabulary is unchanged (same `BroadcastError` variants).
- The composite surface gains `CompositeOps::broadcast_in_dim_owned`, whose
  default forwards to the existing materializing `broadcast_in_dim`.
  `SessionComposite` overrides it to return a `ReadInput::Borrowed` view for a
  borrowed read and a new `ReadInput::Value(TensorValue)` for a value it owns, so
  a broadcast of an owned intermediate is a layout change over storage that value
  already owns, not a copy. Call sites that only hold a borrow (the masked-softmax
  and `apply_affine` paths) keep the materializing method. `finish`/`materialized`
  materialize a non-compact owner at the ownership boundary.
- `broadcast_in_dim_read` keeps its owned-result contract and its documentation:
  a session read returns `Tensor`, so a caller that can consume a read-only
  operand must use the view route instead. That is the documented way to ask for
  a view.

Allocation boundary: `TensorLayout::broadcast_in_dim_view` still allocates one
rank-sized `seen` buffer per call, so this is "no tensor-sized materialization",
not "zero allocation". Removing that buffer with a bounded metadata pattern is
separate work.

## 2. Materializing reshape copy (#2029)

The pooled, thread-scaling copy for owned reshapes (gated by
`structural::reshape_uses_pool`: the token's `parallel_width` for a `Copy`-class
region, which replaced the fixed element count `SERIAL_RESHAPE_MAX_ELEMS`) is
already on `main`; the item needs its paired thread and PyTorch evidence. | not measured |
| 3 | #2040 | Remove the removable per-call allocation in `slice_view` and pin the metadata-allocation count with a test. | implemented and asserted |
| 4 | #1838 | Relax the shared destination contract from compact to write-injective, and add a strided-destination path on CPU and CUDA over the pinned `strided-rs` update primitive. | code and CPU tests done, CUDA verification pending |
| 5 | #2039 | Keep the cached plan, remove the per-call lane allocation and one full pass, and record the engine difference as the reviewed reason for the residual. | implemented, timing evidence pending |
| 6 | #2145 | The tensor keeps its cached compact layout so the per-op view helper borrows it and allocates nothing. Upstream landed the same fix as #2149; this branch keeps the follow-up it did not cover. | implemented and asserted |

## 1. The eager implicit broadcast is a view (#2021)

`tenferro-runtime::typed_tensor::broadcast_to_in_read` called
`session.broadcast_in_dim_read`, which materializes the full broadcast tensor on
CPU, and the composite softmax/normalization broadcasts called it again through
`SessionComposite::broadcast_in_dim`. Every consumer reads the operand.

Decision: the broadcast is metadata on both surfaces.

- `TensorRead::broadcast_in_dim_view` (new `#[doc(hidden)]` boundary in
  `tenferro-tensor`) rebuilds a read's layout with stride 0 on expanding axes,
  keeping the same buffer, retained root and placement. It reuses
  `TensorLayout::broadcast_in_dim_view`, and it refuses a caller-owned
  (`DType::External`) payload with a typed unsupported-dtype error instead of
  reaching the view conversion that has no arm for it.
- `broadcast_to_in_read` maps **all** source axes right-aligned onto the target
  axes. It no longer runs the traced/AD `broadcast_input_plan`, whose
  singleton-removal source shape forced a materializing `reshape_read`. The
  failure vocabulary is unchanged (same `BroadcastError` variants).
- The composite surface gains `CompositeOps::broadcast_in_dim_owned`, whose
  default forwards to the existing materializing `broadcast_in_dim`.
  `SessionComposite` overrides it to return a `ReadInput::Borrowed` view for a
  borrowed read and a new `ReadInput::Value(TensorValue)` for a value it owns, so
  a broadcast of an owned intermediate is a layout change over storage that value
  already owns, not a copy. Call sites that only hold a borrow (the masked-softmax
  and `apply_affine` paths) keep the materializing method. `finish`/`materialized`
  materialize a non-compact owner at the ownership boundary.
- `broadcast_in_dim_read` keeps its owned-result contract and its documentation:
  a session read returns `Tensor`, so a caller that can consume a read-only
  operand must use the view route instead. That is the documented way to ask for
  a view.

Allocation boundary: `TensorLayout::broadcast_in_dim_view` still allocates one
rank-sized `seen` buffer per call, so this is "no tensor-sized materialization",
not "zero allocation". Removing that buffer with a bounded metadata pattern is
separate work.

## 2. Materializing reshape copy (#2029)

The pooled, thread-scaling copy for owned reshapes whose serial time pays for
a fork
`structural::reshape_uses_pool` (the token's `parallel_width` for a `Copy`-class
region, replacing the fixed `SERIAL_RESHAPE_MAX_ELEMS` element count) is
already on `main` (`3870047bb`, `crates/tenferro-cpu/src/exec_session.rs`), and
the branch below it is deliberately serial.

Decision: no code change. This item is accepted only on both halves of the
umbrella criterion — the materialization scales with threads **and** is within
1.5x of PyTorch at 4T. Neither is measured yet on this branch; the recorded
worklog numbers are exploratory, not a paired experiment. Measuring it needs the
`tensor4all/tenferro-benchmark` MWE and the PyTorch reference.

## 3. Metadata-only view construction cost (#2040)

`transpose_view` and `reshape_view` already allocated zero heap bytes at rank
`<= 8`. `slice_view` was the outlier: `TensorValue::slice_view` built a
`Vec<StridedSliceSpec>` and `core_slice_specs` a second `Vec<CoreSliceSpec>`,
both per call.

Decision: make both inline and stop converting each axis twice.

- `TensorValue::slice_view` converts each axis straight into a
  `SmallVec<[CoreSliceSpec; 8]>`; `core_slice_specs` returns the same inline
  vector for its other callers.

The `reshape_view` duplicate `ShapeVec` derivation and the `is_compact_col_major`
recompute are timing-only (they allocate nothing), and the remaining per-axis
checked arithmetic is the validation contract of the public view constructors.
Both stay; removing them would be an unmeasured micro-optimization.

Acceptance: the `metadata_view_alloc_counter` registration is asserted in
`crates/tenferro-tensor/tests/borrowed_view_allocation_tests.rs`
(`metadata_only_view_construction_allocates_nothing`), covering
`transpose_view`, `slice_view` at rank 1 and rank 8, `reshape_view`, and the
relaxed singleton-stride reshape path, with fixtures built outside the counted
region and results retained through `black_box`. The #2040 timing rows still need
re-measurement.

## 4. Strided and offset accumulate destination (#1838)

`validate_axpby_read_into_accum` required `y` to be compact column-major, so a
consumer accumulating into a block of a larger buffer staged through a compact
destination and a second pass.

Decision: the contract relaxes compactness, not injectivity.

- `y` must stay write-injective and disjoint from `x`; compactness is no longer
  required. Injectivity is not re-checked element-wise: a `TensorWrite::View`
  already proves it at `TensorViewMut` construction
  (`validate_mutable_no_overlap`), and an owned destination is compact by
  construction. `validate_read_into_destination` keeps the alias rejection. This
  widens a public observable contract: a non-compact destination that used to be
  rejected with unchanged bytes is now mutated.
- CPU: a compact `y` keeps the dense `strided_kernel::axpby_accum` path. A
  non-compact `y` builds a `StridedViewMut` and a `StridedView` and calls the
  pinned `strided_kernel::zip_update2_into` with `|y, x| alpha * x + beta * y` —
  an existing `strided-rs` 0.5.0 primitive, so no upstream change and no pin
  bump, and no hand-rolled per-element traversal. A strided `x` therefore no
  longer needs the compact staging buffer on that path.
- CUDA: the #1832 kernel now stride-addresses `y` as well, and the vendor
  `geam_accum` path is used only when **both** operands are compact (the previous
  dispatch keyed on `x` alone, which would have updated a contiguous span for a
  strided `y`).

Claim boundary: `zip_update2_into` re-validates destination injectivity and
allocates bounded traversal metadata, so the requirement here is **no
tensor-sized staging/materialization** (`MATERIALIZATIONS == 0` on the strided
path), not zero allocated bytes.

## 5. Cached-plan FFT execution (#2039)

The plan cache was already correct: one `Arc<dyn Fft<T>>` per
`(len, forward, dtype)`, one lookup per call, no lock. The cached path allocated
a zero-filled full-length lane per job per call, moved the data through the lane
and back, and took a separate normalization pass.

Decision:

- the lane comes from the session `BufferPool` through its push-based
  `acquire_with_capacity` route and is released back after the job, under a
  documented full-overwrite invariant (the gather plus the padding or
  conjugate-mirror fill covers every element before the plan reads it), so the
  per-call lane allocation and its zero fill are gone. Buffers are per active
  job, and the RustFFT scratch keeps its zeroed allocation because a
  `&mut [Complex<T>]` over uninitialized memory is not a contract this crate can
  prove;
- the normalization is folded into the scatter, deleting a full lane pass; the
  `scale == 1` identity bypass stays, so the special-value path is unchanged;
- the plan cache and the lane/padding/truncation semantics are untouched.

Residual: the #2039 rows compare cached RustFFT execution against cached native
oneMKL DFTI descriptors. The engine difference (`rfft`/`irfft` run a full complex
plan over zero-imaginary or conjugate-mirrored data, and RustFFT in-place pow2
plans need a full-length scratch) is the reviewed reason the ratio is not closed
here. The item is accepted on no-regression plus the removed allocation and pass;
the timing rows are not yet re-measured on this branch.

## 6. The compact layout stays with the tensor (#2145)

With the strided 0.5.0 pin, `StridedView::new` borrows the caller's layout, but
the per-op view helper computed the strides locally, so it had to hand them to
`StridedView::new_owned` — metadata allocations on every eager operation.

Upstream landed this fix as **#2149** (`4a528b3a9`, "tensor: cache the compact
layout so views borrow it") on top of the branch's original base: `TypedTensor`
caches `layout: StrideVec` and `strides()` serves it, `tenferro-cpu`'s
`host_view` and `tenferro-cpu-basic`'s `typed_view` both call the borrowing
`StridedView::new`. This branch was rebased onto that state, so its own
`strides: R::Strides` field was dropped rather than kept as a second copy of the
same metadata.

What this branch adds on top, because #2149 did not cover it:

- the duplicated `host_view` in `tenferro-cpu/src/structural.rs` is deleted and
  its three call sites use the single `tenferro_cpu_basic::typed_view`, so there
  is one helper instead of two independent ones. #2149's
  `structural/tests/mod.rs` layout-borrowing check was retargeted to that helper,
  including its `strides()` pointer-identity assertion;
- `crates/tenferro-cpu-basic/tests/view_allocation.rs` asserts with a counting
  global allocator that `typed_view` allocates **zero** bytes at rank 2, rank 9
  and for a statically ranked host owner, and that the stored strides are the
  canonical column-major layout across the constructors, the rank/host/dynamic
  conversions and the derived view layout.

Size, measured on the rebased branch (the increase comes from #2149's cached
layout, not from this branch): `TypedTensor<f64, Rank<2>>` 136 → 208 B,
`TypedTensor<f64, DynRank>` 192 → 264 B, erased `Tensor` 296 B unchanged. The
cached `StrideVec` is inline up to rank 8 and spills above it.

## Measurement and evidence

Release, pinned core with its idle state recorded, effective thread count
asserted, 1T and 4T compared in the same run, median of repeated batches. The
acceptance signals are the allocation counters (items 3, 4, 6), the byte
preservation and materialization counters (item 4), and the timing rows for items
1, 2, 3 and 5. Raw paired evidence stays in `tensor4all/tenferro-benchmark`.

`tenferro-tensor --test storage_compile_contract` passes locally with and without the
`kache` wrapper; an earlier note here blamed the wrapper for a failure that does not
reproduce (#2299). Trybuild `.stderr` fixtures do drift with rustc's diagnostic rendering,
so a removed-name check is a source-contract assertion instead (#2299).

Non-goals: no `strided-rs` change, no pin bump, no new provider or
`CpuContext`/`Pool` surface, no GPU FFT.
