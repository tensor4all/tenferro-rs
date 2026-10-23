# U4, part 1: the reduction family

Date: 2026-10-23. Issue: [tenferro-rs#2208](https://github.com/tensor4all/tenferro-rs/issues/2208)
unit U4, first half (reductions; the vendor operations follow).

## What landed

- `Dispatcher::reduce`: one reduction pass with the strategy the caller selects
  (`ReduceStrategy::Unit` or `Plane`), marshalling the reduction entry's arguments —
  both layouts, the axis, the output length and the plane width.
- `Dispatcher::reduce_auto`: the crate's whole-reduction policy. `plan_reduction` decides
  between a direct pass and the split policy; a split runs pass 1 into the caller's
  partials buffer (`reduction_partials_layout`) and then the combine pass with the
  crate's `combine_op`/`combine_strategy`. The returned `ReductionPlan` says which
  happened.

## Evidence

`tests/seam_reduce.rs` on the local A100 (driver 580.126.09, CUDA 12.6), against host
row sums:

- `unit_and_plane_reductions_match_the_host` — a `[37, 24]` input through the unit
  strategy and a `[37, 64]` input through the plane strategy (a plane is 32 lanes, so the
  plane strategy needs an axis at least that wide).
- `the_split_policy_runs_both_passes` — a `[3, 4096]` input splits (asserted on the
  returned plan); the two passes are compared with a relative tolerance because a split
  sums in a different order.
- `a_short_axis_reduces_directly` — a `[5, 8]` input stays a direct pass.

## Observations

- The split's pass 1 and pass 2 must be two launches on the same stream: pass 2 reads the
  partials pass 1 wrote. The dispatcher issues them in order on the runtime's stream, so
  no barrier is needed.
- `Arg` has no signed 64-bit scalar; the kernel's `long long` parameters (`splits`,
  `chunk_len`) are positive, so `Arg::U64` carries the same bits. A negative `long long`
  parameter would need a new argument kind — recorded here rather than added speculatively.
- The plane strategy's precondition is the crate's: `axis_len >= plane_width`, and the
  unit strategy's is the mirror (`axis_len <= plane_width`). A test that mixes them gets
  `StrategyMismatch` from the planner.
