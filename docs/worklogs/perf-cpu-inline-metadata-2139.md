# CPU inline metadata: the typed elementwise per-op allocation budget (#2139 item 2)

## Decisions

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
  does not budget the total. A stride-0 broadcast operand costs 16 allocations in
  `strided_basic::kernel::build_plan_fused_small` / `fuse::compress_dims` on both
  lanes. Both remain unfixed: the eager typed surface has no cache slot to reuse
  a plan, and the fused-plan scratch is a pinned-provider change.
- `ext/df64-proof`'s dispatch report asserted that its direct-call body makes at
  least one allocation as a liveness check on its counter. The body is now
  genuinely allocation-free, so the assertion was rewritten to require
  `direct_allocations == 0` and keep the liveness check on the runtime path.
