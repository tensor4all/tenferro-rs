# CPU inline metadata: the typed elementwise per-op allocation budget (#2139 item 2)

## Decisions

- `INLINE_RANK = 8` is mirrored in `tenferro-tensor-core` and drives
  `ShapeVec` / `StrideVec`; `tenferro-cpu-basic` asserts it equals
  `strided_basic::INLINE_RANK`, so tenferro metadata and strided view metadata
  spill at the same rank.
- The metadata a small-rank elementwise op builds is kept inline rather than on
  the heap, at the four sites that produced it: `broadcast_shape` /
  `broadcast_shapes` (now `ShapeVec`), `col_major_strides` (now `StrideVec`),
  `PooledUninitOutput::new` (now `impl IntoShapeVec`, storing `ShapeVec`), and
  the rank-sized `Vec<bool>` duplicate-axis scratch in
  `TensorLayout::broadcast_in_dim_view` and `validate_axis_list` (now a linear
  scan, matching `broadcast_in_dim_extent_error`).
- The duplicate-axis scratch is a linear scan instead of a bitset because the
  axis lists here are short; this is the same trade the existing
  `broadcast_in_dim_extent_error` already makes.
- `TensorRead::strides` and `TensorWrite::strides` keep their `Vec<isize>`
  return type. Neither is on the measured elementwise path, and widening them
  would churn the surface for nothing.

## Verification conclusions and constraints

- Measured with a counting global allocator
  (`crates/tenferro-runtime/tests/typed_elementwise_allocations.rs`), 1 worker
  thread, output pool warm: one typed `add` / `mul` / `sub` / rank-3 `mul` /
  rank-2 `exp` on `f64` now makes **0 heap allocations**, down from 3.
- `dot_general_output_shape` now makes **1** allocation (its own output `Vec`),
  down from 5: the four `vec![false; rank]` duplicate-axis scratches are gone.
  That number is tenferro's own and does not move with the compiled provider, so
  it is the one the test budgets.
- **Provider readings, not budgets.** A whole 2x2 typed `matmul` measures 83
  allocations on the `cpu-faer` lane and 98 on `cpu-blas`, against the 88 the
  #2139 inventory records; the spread is the provider's plan build, so the test
  does not budget the total. The stride-0 broadcast arm is no longer a provider
  reading: `strided-fused 0.6.0` builds its plan with inline vectors, so
  `crates/tenferro-runtime/tests/typed_elementwise_allocations.rs` budgets it at
  zero on both lanes. The follow-up arms are zero too: a permuted
  `TypedTensorView` add, `exp` over a broadcast view, concatenate and a
  beta != 1 contraction at thread budgets 1 and 2. The earlier residuals were
  tenferro's own pre-execution `Vec` scratch, not provider scratch:
  `structural::validate_permutation`, `reduction::validate_axes` and
  `reduction::reduction_output_shape`, and
  `indexing::collect_typed_inputs` / `indexing::typed_concatenate`. They now use
  a bitmask, `ShapeVec`, or an inline `SmallVec`.
- The typed `add` / `mul` / `sub` / rank-3 `mul` / rank-2 `exp` / broadcast arms
  stay at **0 heap allocations** with one and two worker threads.
- `ext/df64-proof`'s dispatch report asserted that its direct-call body makes at
  least one allocation as a liveness check on its counter. The body is now
  genuinely allocation-free, so the assertion was rewritten to require
  `direct_allocations == 0` and keep the liveness check on the runtime path.
- #2178 item 6 is no longer deferred: `tlinalg`/`tlinalg-blas` are pinned at
  `=0.3.1`, whose additive `*_workspace` / `*_into` entries every faer family
  now uses. Each call takes one byte region from the session pool for the
  provider's complete per-lane scratch (the query and the call share one
  `Exec`), and writes straight into pooled `_into` destinations whose recycler
  returns the storage when the caller drops the tensor. Both halves of that are
  the existing pool: no second allocator, no per-call `Vec` scratch. Results are
  bitwise identical to the wrapper route, which the provider pins to 0.3.0 on
  its own battery and which was re-checked here family by family.
- The host residual on that route is 3 allocations per call at one thread:
  the `BufferPool` in-flight-accounting node the per-operation loan reset
  re-creates, the linalg module's result `Vec`, and the CPU backend's
  dtype-erased `Vec<Tensor>`. A second lane adds one rayon `HeapJob` per spawn.
  The small QR/SVD/LU per-lane ceilings and the batch-size-independence check
  described in #2178 item 7 now record those numbers
  (`crates/tenferro-linalg/tests/small_linalg_allocation.rs`, one- and two-thread
  lanes, separate single-item and batched ceilings); the two-operation
  fused-region ceiling is in
  `crates/tenferro-cpu-fused/tests/fused_region_allocations.rs`. The wider
  per-route ceilings in `crates/tenferro-linalg/tests/cpu_linalg_allocation.rs`
  and `tests/lapack_allocation.rs` are lowered to their re-measured values.
- Not in this item: the contraction lane keeps `execute_uninit_slices`. The
  pinned `cpueinsum 0.3.1` `execute_bounded` still takes a caller-sliced
  `&mut [T]` work buffer rather than a borrowed byte region, so there is no
  `*_into`-style arena entry to move that lane onto.
