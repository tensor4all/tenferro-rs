# strided generic scatter adoption

## Task

Advance all four tenferro strided dependencies from
`53ecd7718169e69320078f4bb2609945140450ac` to merged generic-scatter revision
`b40cd2f6d83c35ca23b24a8fb371ca061495729c` (strided-rs PR #252 / issue #251).
The target preserves workspace version 0.4.0 and contains all earlier #213
work. Update only the four pins, canonical source-contract revision, and this
worklog. No tenferro API, source traversal, feature, backend, or skill change.

Baseline tenferro: `5c45766c248264a7eaa9ee2deab9675692887d14`.
Selected reviewer: read-only `reviewer-flash`, high thinking, for design and
exact final diff. Pin edits start only after a Correct-to-merge design verdict.

## Public paired experiment

A temporary release probe outside the repository uses one reusable
`CpuBackend`, prebuilt F64 tensors, `N = 262,144`, 3 warmups, 15 samples, median
and IQR:

- generic rank-8 windowed additive scatter: operand/destination shape
  `[2048,2,2,2,2,2,2,2]`, indices `[2048,1]` with permutation
  `(5*i+1) mod 2048`, full update windows over axes 1-7, scatter/insert axis 0;
- compact rank-one scalar additive-scatter control with `N` identity indices.

Construction, config validation, backend creation, input allocation, and full
expected output stay outside timing. Each timed call returns and black-boxes a
fresh tensor. A correctness call compares every output value exactly before and
after timing. The generic case has unique batch destinations; separate upstream
tests own repeated-window order semantics.

Run the complete one-thread arm before four-thread, baseline before candidate,
all sequential. Precommit L3 domain 0: CPU 1 for 1T, CPUs 1-4 for 4T. Before
each arm, every selected core must be below 2% busy over four seconds and
siblings below 20%. Record failed gates; a failed arm contributes no timing and
requires a fresh complete arm.

Need gate: baseline generic rank-8 exceeds 1 ms or 2x the rank-one control.
Candidate gates when need passes:

- rank-8 1T and 4T medians >=5x faster;
- rank-one control has no >10% regression;
- full outputs exactly match the precomputed reference.

## Change and verification

1. Update `strided-view`, `strided-traits`, `strided-perm`, and
   `strided-kernel` pins together; preserve versions/features.
2. Update the source-contract test's one canonical revision.
3. Regenerate ignored lock and prove all four exact packages/version.
4. Run build-artifact tests, focused CPU scatter/indexing tests,
   `scripts/check-pr-fast.sh --coverage-reviewed` with one focused command,
   committed repository-rules review, exact-final reviewer verdict, hosted CI,
   auto-merge, and final `origin/main` pin verification.

Coverage is metadata-only: no tenferro executable source changes.

## Paired public evidence

Baseline one-thread gate attempt 1 failed because CPU 0 was 30.8% busy and
produced no timing; the accepted retry had selected CPU 1 at 0.2% and
domain-other maximum 0.3%. Baseline four-thread passed first attempt with every
domain core 0.0%. Candidate one-thread passed first attempt with every domain
core 0.0%. Five candidate four-thread gate attempts failed because CPU 0 stayed
100% busy; a fresh arm passed with CPU 0 at 0.3%, selected maximum 0.5%, and
other siblings 0.0%, then supplied the retained timing. No failed arm
contributes a timing.

| case | baseline 1T | candidate 1T | speedup | baseline 4T | candidate 4T | speedup |
|---|---:|---:|---:|---:|---:|---:|
| rank-8 windowed scatter | 7.463022 ± 0.428646 ms | 1.453812 ± 0.007650 ms | 5.13x | 8.911004 ± 0.895104 ms | 1.728380 ± 0.009851 ms | 5.16x |
| rank-1 scalar control | 1.008478 ± 0.000570 ms | 0.969094 ± 0.033301 ms | — | 1.253987 ± 0.002591 ms | 1.172441 ± 0.026412 ms | — |

The need gate and every candidate gate are **PASS**. Rank-8 public speedups
exceed 5x in both contexts; controls improve 3.9%/6.5%. Full exact-output
comparisons passed before and after timing.

## Pin and verification

Commit `629983d3` updates only the four pins and canonical source-contract
revision. The ignored lock resolves all four packages at version 0.4.0 and
exact `b40cd2f6`.

- build-artifact contracts: 9 passed
- focused CPU scatter tests: 10 passed
- `scripts/check-pr-fast.sh --coverage-reviewed --test 'cargo test -p tenferro-cpu test_scatter_clamps_negative_and_out_of_bounds_windows'`: passed, including formatting, doc snippets, workspace/extension clippy, and focused test
- upstream durable evidence: benchmark-suite PR #37, merge `b03e98db`

Coverage was reviewed: dependency metadata and its contract test changed; no
tenferro executable source line was added. Exact-final review, committed
repository-rules review, and hosted CI remain pending.

## Design gate

Read-only `reviewer-flash` with high thinking reviewed exact design `a9d35674`
and returned **Correct-to-merge**. Before pin edits, explicit ancestry and
version checks passed: `53ecd771` is an ancestor of `b40cd2f6`, and the target
workspace remains version 0.4.0.
