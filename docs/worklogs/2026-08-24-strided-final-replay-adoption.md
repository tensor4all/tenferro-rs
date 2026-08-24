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
- compact rank-8 integer `TensorElementwise::div_read_into` with all-nonzero
  divisors and a caller-owned output, with compact rank-1 `div_read_into`
  control. The owned `div` method is deliberately excluded because it routes to
  the unaffected typed zip path rather than `erased_zip_into`.

Tensor construction, backend construction, warmup, input/output allocation,
and correctness references stay outside timing. Reduction and pad return fresh
tensors, which are black-boxed; `div_read_into` overwrites and black-boxes its
preallocated output. Reduction uses exactly representable small integer-valued
`f64` inputs and checks the known axis-0 pair sums exactly. The paired candidate
must also match the baseline output bitwise. Pad and integer-division outputs
check known exact values. Run 3 warmups and 15 samples; report median and IQR.
Run the complete one-thread arm before the complete four-thread arm. Baseline
and candidate use identical source and arm order.

Precommit to L3 domain 0: CPU 1 for the one-thread arm and CPUs 1-4 for the
four-thread arm. Before a run, selected cores average below 2% busy over four
seconds and every other core in that domain stays below 20%. Permit at most two
load-gate attempts per complete arm, record every failed gate, and declare the
arm INCONCLUSIVE rather than selecting another domain. An INCONCLUSIVE arm
fails this attempt and requires a fresh sequential baseline/candidate pair;
it cannot contribute a partial result. Benchmark processes run sequentially.

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

## Design gate

Read-only `reviewer-flash` with high thinking rejected the first design because
owned `div` bypasses `erased_zip_into`. The corrected design at `dcf468d3` uses
`div_read_into`, fixes exactness and bounded load-gate semantics, and received a
fresh **Correct-to-merge** verdict. Baseline and pin work may proceed.
