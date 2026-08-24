# Final strided replay adoption

## Task

Advance all four tenferro workspace strided dependencies from P1 revision
`75fb0f70f6138bab3a5bc033d59bfc64bfd033f3` to final issue-#213 umbrella
revision `53ecd7718169e69320078f4bb2609945140450ac`.

The target includes:

- erased axis-reduction replay, PR #245 (`37ce20b7`);
- generic pad fallback replay, PR #246 (`f875cc89`);
- integer divide/remainder zero preflight, PR #248 (`53ecd771`).

Tenferro directly uses all three owners: `ErasedReducePlan` in
`tenferro-cpu/src/reduction.rs`, `ErasedPadPlan` in
`tenferro-cpu/src/indexing.rs`, and `erased_zip_into` for dynamic integer divide
in `tenferro-tensor/src/backend.rs`. No tenferro public API, feature, backend,
semantics, or architecture changes.

Baseline tenferro source: `04379f94fa87596406e325fa825c7023b497a90b`.
Selected reviewer: read-only `reviewer-flash`, high thinking, for this design and
the exact final diff. No pin or source-contract edit may start before a
Correct-to-merge design verdict.

## Design

1. Update exactly `strided-view`, `strided-traits`, `strided-perm`, and
   `strided-kernel` to `53ecd771...`; preserve versions and features.
2. Update the build-artifact source-contract test's one canonical revision.
3. Regenerate the ignored lock locally and prove all four packages resolve to
   version 0.4.0 at the exact revision.
4. Add no tenferro traversal, compatibility layer, benchmark API, or shipped
   skill change. Traversal remains owned by strided-rs.
5. Record the upstream durable evidence publication:
   `strided-rs-benchmark-suite` PR #34, merge `51c6acd1`.

## Public paired experiment

A temporary release probe outside the repository uses one reusable
`CpuBackend` and prebuilt tensors at `N = 262,144`. It runs:

- compact rank-8 `TensorReduction::reduce_sum` over axis 0, with compact rank-2
  control;
- rank-8 `TensorIndexing::pad` with axis-0 interior padding 1, with rank-1 dense
  edge-pad control;
- compact rank-8 integer `TensorElementwise::div` with all-nonzero divisors,
  with compact rank-1 divide control.

Tensor construction, backend construction, warmup, input allocation, and
correctness references stay outside timing. Each call returns a fresh tensor,
which is black-boxed. Run 3 warmups and 15 samples; report median and IQR.
Run the complete one-thread arm before the complete four-thread arm. Baseline
and candidate use identical source and arm order.

Pin each process to idle cores in one EPYC L3 domain. Before a run, selected
cores average below 2% busy over four seconds and every other core in that
L3 domain stays below 20%; otherwise rerun the complete arm. Benchmark
processes run sequentially.

Need gate: baseline rank-8 median exceeds 1.0 ms or is at least 2x its matching
control. For each family that passes need, require:

- one-thread median at least 2x faster;
- four-thread median at least 1.5x faster;
- matching control does not regress by more than 10%;
- exact reduction, pad fill/copy, and integer-division values pass.

## Verification

- build-artifact dependency-contract tests;
- focused CPU reduction, indexing/pad, and integer elementwise tests;
- `scripts/check-pr-fast.sh --coverage-reviewed` with one focused test command;
- committed repository-rules review;
- exact-diff `reviewer-flash` verdict;
- hosted CI, auto-merge, and final `origin/main` pin verification.

Coverage is metadata-only for tenferro executable source; hosted CI owns the
full dependency-resolved coverage gate.
