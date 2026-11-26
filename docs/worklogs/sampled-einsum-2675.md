# Sampled einsum v0 (#2675)

Port of Feng Pan's prototype Fanerst/sampled-einsum (`c482b79a`) into
`tenferro-einsum`. Design decisions are in `docs/design/einsum.md`
§ Sampled Planning And Execution; this log records what was checked and what
was left open.

## Decisions taken while porting

- The planner (`src/sampled/plan.rs`) is the prototype's `plan.rs` with tenferro
  types; the step logic, row order, chunk rule and peak accounting are
  unchanged. Two additions: a gathered-pairs step is one instruction that the
  executors realise differently, and each contraction's subscripts and shapes
  are recorded so the concrete layer prepares its `ConcreteEinsumPlan`s once.
- The #2676 slice loop became `crate::chunked::run_chunks` (a `ChunkSource`
  per caller). Its tests (`sliced_einsum`, `sliced_memory`) pass unchanged.
- Destination windows of the concrete pairs step are rebased to offset 0 on
  host. Without it the engine's binary plan cache misses once per window
  (measured: 13 engine plan misses for 6 chunks against 10 for 3 chunks; with
  rebasing both are 10).
- The independent pre-review (gpt-6.1-sol) asked for the plan to stay private
  and `einsum_at` to return the report. Not taken: the brief asks for both
  the explicit-path plan and `einsum_at`, and for the report returned to the
  caller; `SlicedEinsumPlan` is the precedent for a public plan object.

## Post-implementation review (gpt-6.1-sol)

Fixed: a strided operand view (transposed, reversed) is gathered at full rank
when the `[prefix, extent, suffix]` relabeling does not apply; zero extents
are rejected (the prototype rejects them too; the `n = 0` dummy sample needs
index 0); the eager `einsum_at` checks the conjugation mask like the concrete
one; `Samples::value` returns `Option`; the gradient test records both routes
on one eager tape over the same variables and takes their VJPs from it.
Each fix has a test that fails without it.

Not changed: the plan stays public (see above); CUDA (below); the zero fill
of a pairs step's result before its windows are written (one `fill_scalar`
per step, kept because a fresh uninitialised device buffer has no engine
entry point); the per-slice assignment `Vec` of the sliced loop (one
allocation of `slice label count` words per slice, next to a contraction).

## Open

- CUDA: the executor is engine-generic and uploads gather indices for a
  device engine as the prototype does (which Feng Pan ran on an A100), but this
  PR did not run on a GPU.
- The prototype's tests draw from `rand::StdRng`; tenferro has no `rand`
  dev-dependency, so the same generator draws from the LCG of the sliced
  tests: same distribution, different individual networks.
