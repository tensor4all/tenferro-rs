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
