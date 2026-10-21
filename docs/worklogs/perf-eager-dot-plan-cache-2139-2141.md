# The eager contraction path reaches the session's plan cache (#2139 item 1, #2141 item 1)

## Decisions

- The eager entries of `dot_general` (`dot_general_read`,
  `dot_general_read_into_accum`, `dot_general_with_conj`) now derive a
  `cache_slot` from the request instead of passing `None`: a hash of dtype, the
  two operand shapes, the four axis lists and the conjugation flags, modulo the
  session's cache capacity. The prepared path keeps using the compiled
  instruction index, which is a better key when one exists.
- The fingerprint is a *routing hint*, not an identity. `BinaryCache` still
  compares the full plan key (both operand layouts with strides and offsets, the
  output layout, the conjugation flags) and the `DotGeneralConfig` before
  reusing, so a collision is a miss, never a wrong plan. Operand strides are
  deliberately left out of the fingerprint because reading them would cost the
  allocation the cache exists to remove; the price is hit rate, not correctness.
- `dot_general_output_shape`, `validate_dot_general_read_into`,
  `validate_dot_general_accumulation` and the internal `dot_fresh` / `fresh` /
  `allocate_dot_output` / `pooled_zero_tensor` helpers now carry `ShapeVec`
  instead of `Vec<usize>`, so the output shape no longer reaches the heap at
  small rank either.
- The eager path now *retains* plans it previously dropped after one use. The
  retention is bounded by the existing cache capacity and reported through
  `runtime_cache_stats`, which is the intended trade and is recorded in
  `docs/design/contraction-pipeline.md`.

## Verification conclusions and constraints

- One `AMD EPYC 7713P` thread (`CpuBackend::with_threads(1)`), warm output pool,
  f64 64x64 eager `matmul`:
  - heap allocations per repeated call: **84 -> 0** (the cold call makes 414-415,
    which includes warming the provider);
  - wall time per call, 4000 warm iterations: **22.4-23.6 us -> 16.5-18.5 us**
    (three runs each, non-overlapping ranges).
- Reproduced with the cache slot forced back to `None` through a temporary
  environment switch: the repeated call then made 84 allocations, which is what
  the committed test would report as a failure.
- `dot_general_output_shape` on `[2, 2] x [2, 2]` now makes 0 heap allocations,
  asserted in `crates/tenferro-runtime/tests/typed_elementwise_allocations.rs`.
- The regression test is
  `crates/tenferro-runtime/tests/eager_dot_plan_reuse.rs`: it asserts the repeat
  makes no allocation at all, so a lost slot fails it.
- The fingerprint deliberately ignores operand strides. A workload that reaches
  the same shape through two differently shaped views of one storage will see
  slot collisions and therefore misses on the eager path; the compiled path is
  unaffected.
- Not measured here: the decode-shaped (`[1,512] x [512,1024]`) case, whose time
  is dominated by the GEMM itself rather than by the plan build. The umbrella's
  `decode_proj_*` rows own that comparison.

## Follow-up: replay evidence and the 2x2 re-measure (#2139 residuals)

- The public observer for "no rebuild" is the engine's contraction plan cache:
  `CpuEngine::plan_cache().contraction_stats()`, where `misses` counts plan
  builds and `hits` counts replays. A prepared plan's own `cpu_cache_stats`
  covers only the n-ary descriptor cache, so it cannot answer this on the binary
  fast path.
- `einsum_prepared_repeat_f64_counter`
  (`crates/tenferro-einsum/tests/integration/prepared_plan_replay.rs`) prepares a
  `32x32` f64 `ij,jk->ik` plan, asserts one miss on the first `execute_into`,
  then four repeats add four hits, no miss, and no entry growth. The
  tenferro-benchmark case of the same name can report
  `binary_plan_builds_per_call` from the same delta.
- Re-measured the 2x2 typed `matmul`
  (`repeated_tiny_typed_matmul_keeps_only_the_one_time_plan_build`): with the
  provider warmed under a different key, the first call makes **41** allocations
  in the test profile on the faer lane (the one-time binary plan build; 88 at
  filing, 83/98 recorded in `perf-cpu-inline-metadata-2139.md`), and the warm
  and repeated calls make **0**. No tenferro metadata `Vec` remains on the
  repeat path, and the repeat builds no provider plan either.
- Both arms in `eager_dot_plan_reuse.rs` share one counting allocator, so their
  measured regions take a process-wide lock. The printed first-call count still
  depends on which test warmed the provider first; it is evidence, not a budget.
