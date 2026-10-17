# SmallVec to TinyVec for tensor metadata (#2180)

## Decisions

- `tenferro-tensor-core` `ShapeVec` / `StrideVec` are `TinyVec` aliases over
  `INLINE_RANK = 8`. `TinyVec` is bitwise-inline and clones a fixed inline
  buffer instead of collecting element-wise, which removes the per-op clone
  floor tracked in #2180.
- Default-able metadata (`usize` / `isize` / nested `TinyVec` / `&[T]`) moved to
  `TinyVec`. `tinyvec::Array` requires `Item: Default`, so items without a
  `Default` impl keep `SmallVec` with a one-line reason:
  - `RuntimeInputReads` (`TensorRead` has no `Default`),
  - `Arc<dyn EventToken>` dependency lists,
  - `ErasedRawStridedRef` / `ErasedRawStridedPtr` operand descriptor lists,
  - `&T` admitted event-token lists.
  `smallvec` therefore stays a dependency of `tenferro-cpu`, `tenferro-gpu` and
  `tenferro-runtime`; it was removed from the crates that no longer use it.
- `tenferro-tensor-core::SliceSpec` gained a manual `Default` (`step = 1`) so
  the metadata-only `slice_view` scratch stays inline without a heap `Vec`;
  this keeps the #2040 zero-allocation slice contract.
- Retained-byte accounting translates to `TinyVec::is_heap()` /
  `capacity()`; the two accounting helpers are renamed (`tinyvec_outer_bytes`,
  `heap_backed_bytes`) so the reported quantity is still the heap-backed outer
  buffer only.
- The `DotGeneralConfig` axis lists are `TinyVec<[usize; 4]>`. Call sites that
  built them from `Vec` now use `.iter().copied().collect()` (or
  `.into_iter().collect()`), because `TinyVec` has no `From<Vec>`; this is the
  documented public-API fallout of the migration.

## Verification conclusions and constraints

- `cargo check --workspace --all-targets` is clean; CUDA and WebGPU features
  are checked separately. The existing small-op allocation budgets in
  `crates/tenferro-runtime/tests/typed_elementwise_allocations.rs` stay at zero
  for the same-shape and broadcast arms.
- The #2178 item 7 arms are zero on both CPU lanes: permuted view add, `exp`
  over a broadcast view, concatenate, a beta != 1 contraction, a rank-4
  materializing transpose, and a one-axis reduction at thread budgets 1 and 2.
  The small batched QR/SVD/packed-LU ceilings are pinned per lane in
  `crates/tenferro-linalg/tests/small_linalg_allocation.rs`; QR and SVD are
  batch independent, and BLAS packed-LU grows by one allocation from batch 1
  to 64 (recorded, not hidden). C allocations stay out of scope.
- Two residual allocations found during the first pass were tenferro's own
  pre-execution `Vec` scratch (`structural::validate_permutation`,
  `reduction::validate_axes` / `reduction_output_shape`,
  `indexing::collect_typed_inputs` / `typed_concatenate`), not strided 0.6.0.
  They are fixed rather than pinned.
