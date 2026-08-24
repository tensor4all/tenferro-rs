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

## Paired public evidence

The first baseline pair was discarded in full: its one-thread arm passed on the
second gate attempt, but both four-thread gate attempts failed (first because
CPU 0 was fully occupied, then because selected-core average was 6.1%). A fresh
baseline pair then passed on the first attempts: one-thread selected 0.0% with
0.5% domain-other maximum; four-thread selected 0.0% with 0.5% domain-other
maximum.

The first candidate one-thread arm passed its load gate but showed a 26%
rank-one pad-control shift and failed the control gate. A fresh sequential
one-thread baseline/candidate arm passed its load gate (selected 0.7%,
domain-other maximum 0.8%) and replaced that arm in full. The candidate
four-thread arm passed on its first gate (selected average 1.8%, domain-other
maximum 1.3%). No partial timing from a failed arm is used below.

`N = 262,144`, medians with IQR:

| case | baseline 1T | candidate 1T | speedup | baseline 4T | candidate 4T | speedup |
|---|---:|---:|---:|---:|---:|---:|
| rank-8 reduce | 3.728041 ± 0.710904 ms | 0.867198 ± 0.090704 ms | 4.30x | 1.413155 ± 0.336191 ms | 0.251598 ± 0.273988 ms | 5.62x |
| rank-8 pad | 20.541621 ± 3.653278 ms | 2.670406 ± 0.326921 ms | 7.69x | 2.816089 ± 0.032201 ms | 0.801016 ± 0.007630 ms | 3.52x |
| rank-8 `div_read_into` | 3.923646 ± 0.243788 ms | 1.247030 ± 0.015051 ms | 3.15x | 4.489403 ± 0.134164 ms | 1.131747 ± 0.028570 ms | 3.97x |
| rank-2 reduce control | 1.733786 ± 0.011430 ms | 0.732133 ± 0.055581 ms | — | 0.730943 ± 0.050921 ms | 0.229068 ± 0.003541 ms | — |
| rank-1 pad control | 0.115534 ± 0.002150 ms | 0.122434 ± 0.001180 ms | — | 0.134544 ± 0.002679 ms | 0.134464 ± 0.000780 ms | — |
| rank-1 `div_read_into` control | 0.908729 ± 0.057132 ms | 0.944120 ± 0.154625 ms | — | 0.751604 ± 0.049551 ms | 0.583019 ± 0.004150 ms | — |

All need and candidate gates are **PASS**. Rank-8 one-thread/four-thread
improvements exceed 2x/1.5x. The largest retained control regression is 5.97%
(rank-one pad, one thread), below 10%. Both reduction arms printed identical
bitwise checksum `3e05621f1f49344d`; all known-value assertions passed for
reduction, interior pad fill/copy, and integer division.

## Pin and verification

Commit `953e9ee` updates only the four workspace pins and canonical
source-contract revision. The ignored local lock resolves all four packages at
version 0.4.0 and exact revision `53ecd771`.

- build-artifact contract tests: 9 passed
- focused `reduce_sum`: 6 passed
- focused pad: 8 passed
- focused integer divide/remainder/pow: 3 passed
- `scripts/check-pr-fast.sh --coverage-reviewed --test 'cargo test -p tenferro-cpu test_integer_div_rem_pow_contract'`: passed, including root/extension formatting, doc snippets, workspace/extension clippy, and focused test
- durable upstream results: benchmark-suite PR #34, merge `51c6acd1`

Coverage was reviewed: this adoption changes dependency metadata and its
contract test but adds no tenferro executable source line. Exact-final review,
committed repository-rules review, and hosted CI remain pending.
