# CPU elementwise fusion: declined on the scalar replay (#2140 item 3), re-enabled on `strided-fused 0.5.2` (#2161)

## Decisions

- The CPU session now returns `Ok(None)` from `execute_elementwise_fusion`
  (`SELECTS_FUSED_ELEMENTWISE_REGION` in `crates/tenferro-cpu/src/exec_session.rs`),
  so the runtime dispatches the region's instructions one by one. The planner,
  the region plan, `max_fused_region_inputs`, and the executor in
  `tenferro-cpu-fused` are untouched: re-enabling CPU fusion is one constant.
- The decline is a policy, not a threshold. The measured gap is 13x to 70x at
  every size, so no element-count floor separates a winning case from a losing
  one.
- The three `backend_misc` plan-shape tests now drive
  `tenferro_cpu_fused::elementwise_fusion_with_pool` directly instead of going
  through the session, so their plan coverage survives the policy.
- `output_affinity::dot_and_fusion_vec_outputs_use_the_selected_domain` and
  `static_replay`'s fusion test assert the decline. The fused path used to tag
  its fresh outputs with the selected resource domain, and that is no longer
  reachable through the session; the assertion records the loss rather than
  hiding it, and the design record says what to restore.

## Verification conclusions and constraints

- One `AMD EPYC 7713P` thread, release, `Runtime::prepare_compiled` +
  `run_prepared`, a `1024x64` `f32` chain of ten instructions, 300 iterations
  after five warm runs:

  | chain | fused region | declined |
  |---|---:|---:|
  | ten `tanh` (the #1990 fixture) | 23.4 ms | 1.68 ms |
  | ten `mul` | 6.0 ms | 0.11 ms |

  The decline is 13.9x and 54x faster respectively, and lands within 1.00-1.22x
  of issuing the same chain through the eager session one op at a time (1.61 ms
  and 0.083 ms), which is the acceptance #2140 item 3 states.
- Two candidate fixes were measured and rejected. Planning no region at all
  (`max_fused_region_inputs = Some(0)`) measured the same as declining at
  execution (within noise), but it removes the region from the planner's census
  and thereby guts the planning and fallback tests; the executor-level decline
  keeps them. A tenferro-side vectorized replay was rejected as the wrong home:
  `elem-simd-math` exposes slice-to-slice kernels, not vector-register
  operations, so a vectorized replay is a new executor, and the `FusedPlan` /
  `FusedOp` vocabulary and the run iteration live in `strided-fused`.
- What fusion can win on CPU is bounded by one intermediate's traffic, about
  17 us at this size, because the intermediates come from the pooled allocator
  and stay in cache. A vectorized replay would therefore recover most of its
  value on large, memory-bound chains and almost none on the analytic chains
  this measurement used, where compute dominates: ten `tanh` cost 1.6 ms of
  arithmetic and 0.17 ms of traffic.

## Reassessment after `strided-fused 0.5.1` (#2161)

- `strided-fused 0.5.1` vectorized the replay for **contiguous** blocks (a chunked
  register machine over `elem-simd-math`), so CPU fusion was re-measured rather
  than re-enabled. The decline stays: on the predeclared rule "enable only if
  fused/declined `<= 1.05` in every case", three cases fail — two clear
  regressions (below) and the outer-axis broadcast at 4T (1.07, marginal).
- The chunked replay is not taken when any operand is not unit-stride along the
  block's contiguous axis. A broadcast on the column-major innermost axis
  (`b[1, 64]` added to `x[1024, 64]`, reduced by the shared input-view path to a
  rank-1 operand) has stride `0` there, so `chunk::block_is_contiguous` fails and
  `strided-fused` runs `interpret_inner_loop_uninit`; fusion loses 16x at both 1T
  and 4T. A broadcast on the outer axis takes the chunked replay (0.83 at 1T)
  but sits just above the gate at 4T (1.07).
- The CPU adapter still compiles one `strided-fused` plan per live output
  (`single_output_strided_fused_plan`), so a two-live-output region replays its
  whole instruction list twice; fusion loses 1.85x at 1T and 1.54x at 4T.
- The winning cases are unchanged and real: ten `tanh`/`mul` f32 and f64
  `1024x64` chains win 0.34-1.00, the 16,384-element floor wins 0.75-0.89, and a
  `1<<20` chain wins 0.89-0.90. That is why the re-enable is worth doing once the
  two upstream blockers are fixed, not that the constant should be dropped.
- Measured 2026-10-16, `AMD EPYC 7713P`, release, commit `47dc94c98`, pinned with
  `taskset` to idle cores 56-63, 5 warm runs plus 300 timed runs per round, median
  of three interleaved rounds, 1 and 4 worker threads. The per-case table is
  below; `docs/design/cpu-backend-execution.md` links here.

Measured 2026-10-16 on an `AMD EPYC 7713P`, release, the prepared program's first
five runs plus 300 timed runs per round, median of three interleaved rounds,
pinned to idle cores 56-63, one worker thread:

| case | declined | fused | ratio |
|---|---:|---:|---:|
| ten `tanh` f32 `1024x64` | 2.33 ms | 2.11 ms | 0.90 |
| ten `mul` f32 `1024x64` | 0.176 ms | 0.109 ms | 0.62 |
| ten `tanh` f64 `1024x64` | 5.15 ms | 5.16 ms | 1.00 |
| ten `mul` f64 `1024x64` | 0.381 ms | 0.266 ms | 0.70 |
| broadcast on the contiguity axis f32 | 0.276 ms | 4.59 ms | **16.6** |
| broadcast on the outer axis f32 | 0.249 ms | 0.208 ms | 0.83 |
| two live outputs f32 | 0.221 ms | 0.410 ms | **1.85** |
| ten `tanh` f32 `16384` (region floor) | 0.412 ms | 0.364 ms | 0.89 |
| ten `tanh` f32 `1<<20` | 37.3 ms | 33.3 ms | 0.89 |

At four worker threads the same cases give ratios 0.34-0.93 except the
contiguity-axis broadcast (**16.4**), the outer-axis broadcast (1.07) and the two
live outputs (**1.54**). The acceptance rule was "enable only if fused/declined
`<= 1.05` in every case", so the decline stays: the contiguity-axis broadcast
(like `b[1, 64]` added to `x[1024, 64]`) forces the scalar interpreter, and the
two-live-output region pays the per-output replay.

## Re-enabled on `strided-fused 0.5.2` (#2161)

- `strided-fused 0.5.2` extends the chunked register machine to a stride-`0`
  operand: a broadcast value is filled into the chunk once instead of forcing
  `interpret_inner_loop_uninit`. `tenferro-cpu-fused` also compiles one plan with
  every live output and runs it with `ErasedFusedPlan::execute_outputs_uninit`,
  so a two-live-output region replays once. Both blockers from the 0.5.1
  reassessment are gone.
- Re-measured with the same predeclared rule ("enable only if fused/declined
  `<= 1.05` in every case") and the same cases: the rule passes. The
  contiguity-axis broadcast now wins 0.48-0.65 and the two-live-output region
  0.76-0.96, so the constant was deleted and the executor is always selected
  (subject to its own floor, arity, dtype and op filters, now the only declines).
- Measured 2026-10-17, `AMD EPYC 7713P`, release, `Runtime::prepare_compiled` +
  `run_prepared`, 5 warm runs plus 300 timed runs per round, median of three
  interleaved rounds, `taskset` on the assigned cores, 1 and 4 worker threads.
  Per-iteration ms; `ratio` is fused/declined. This is one clean repetition; the
  maximum ratio over three clean repetitions was 1.01 at 1T and 1.04 at 4T
  (ten `tanh` f64 `1024x64`).

One worker thread:

| case | declined | fused | ratio |
|---|---:|---:|---:|
| ten `tanh` f32 `1024x64` | 1.587 | 1.444 | 0.91 |
| ten `mul` f32 `1024x64` | 0.107 | 0.103 | 0.96 |
| ten `tanh` f64 `1024x64` | 3.492 | 3.530 | 1.01 |
| ten `mul` f64 `1024x64` | 0.187 | 0.171 | 0.91 |
| broadcast, contiguity axis f32 | 0.0705 | 0.0459 | 0.65 |
| broadcast, outer axis f32 | 0.0570 | 0.0421 | 0.74 |
| two live outputs f32 | 0.276 | 0.264 | 0.96 |
| ten `tanh` f32 `16384` (floor) | 0.413 | 0.371 | 0.90 |
| ten `tanh` f32 `1<<20` | 25.22 | 23.18 | 0.92 |

Four worker threads:

| case | declined | fused | ratio |
|---|---:|---:|---:|
| ten `tanh` f32 `1024x64` | 0.961 | 0.782 | 0.81 |
| ten `mul` f32 `1024x64` | 0.206 | 0.0727 | 0.35 |
| ten `tanh` f64 `1024x64` | 1.936 | 1.974 | 1.02 |
| ten `mul` f64 `1024x64` | 0.275 | 0.117 | 0.43 |
| broadcast, contiguity axis f32 | 0.100 | 0.0478 | 0.48 |
| broadcast, outer axis f32 | 0.0920 | 0.0460 | 0.50 |
| two live outputs f32 | 0.216 | 0.185 | 0.85 |
| ten `tanh` f32 `16384` (floor) | 0.495 | 0.382 | 0.77 |
| ten `tanh` f32 `1<<20` | 7.22 | 6.30 | 0.87 |
