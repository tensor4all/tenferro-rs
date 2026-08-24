# strided P1 indexed adoption

## Task

Adopt merged strided-rs generic indexed-replay work in tenferro-rs after:

- strided-rs #237 / PR #242, merge `acdeea3f620b9a515c6915a44730022d19c4e71a`;
- strided-rs #238 / PR #244, merge `75fb0f70f6138bab3a5bc033d59bfc64bfd033f3`.

Advance all four workspace strided dependencies from
`39111bd7b397c54402d1d9370bdd27a6c04023ed` to the latest merged commit
`75fb0f70f6138bab3a5bc033d59bfc64bfd033f3`. The upstream workspace still
declares version `0.4.0`. No tenferro source, public API, feature, backend,
semantics, or architecture changes.

Baseline tenferro source: `9fdb49ef4341bc97c10c029b3b158d2a591c8f8d`.
Selected reviewer: read-only `reviewer-flash`, high thinking, for this design and
the exact final diff. The pin and source-contract test must not change before a
Correct-to-merge design verdict.

## Context reviewed

- tenferro issue #1719 closure evidence and merged PR #1722
- strided-rs #213, #237, #238, PRs #242/#244, and their worklogs
- tenferro `AGENTS.md`, `CONTRIBUTING.md`, `REPOSITORY_RULES.md`, shared
  repository/performance/Rust rules, and bug-fix workflow
- workspace strided dependency declarations and
  `scripts/ci/tests/test_build_artifact_contracts.py`
- public `TensorIndexing` and `CpuBackend` gather/dynamic slice/update paths

## Design

1. Update exactly `strided-view`, `strided-traits`, `strided-perm`, and
   `strided-kernel` to merged revision `75fb0f70...`; preserve versions and
   features.
2. Update the source-contract test's single canonical expected revision.
3. Keep the ignored workspace `Cargo.lock` out of the PR; regenerate it locally
   and verify all four packages resolve to the exact commit.
4. Add no compatibility layer or tenferro traversal. Generic indexed replay
   remains owned by strided-rs.
5. No user docs or shipped skill change is needed because API/feature/crate
   boundaries and documented idioms are unchanged.

## Public paired experiment

A temporary release probe outside the repository exercises `CpuBackend` at
`N = 262,144` for compact rank-4 generic gather, dynamic slice, and dynamic
update, plus the existing rank-one gather/dynamic controls. Setup, tensor
construction, plan caching warmup, and allocation of inputs remain outside each
timed operation; returned tensors are black-boxed. Run 3 warmups and 15 samples,
report median and IQR.

Contexts: one thread and four threads. AMD EPYC 7713P processes are pinned to
idle cores in one L3/CCD. Before each complete baseline/candidate run, every
selected core must average below 2% busy for four seconds and every sibling in
that L3 domain below 20%; otherwise the complete run is INCONCLUSIVE. Timing
runs are sequential.

Need gate: current public rank-4 operation median at N must exceed its matching
rank-one control by at least 2x or exceed 1.0 ms absolute. A family that fails
this gate retains evidence but receives no performance claim.

Predeclared candidate gates for each family that passes need:

- one-thread public median at least 3x faster;
- four-thread public median at least 2x faster;
- no rank-one control median regresses by more than 10%;
- public results exactly match known values and focused CPU indexing tests pass.

The same complete probe and cases run before and after only the pin change. The
fixed arm order is rank-4 gather, dynamic slice, dynamic update, then their
rank-one controls; run the complete one-thread arm before the complete
four-thread arm for both baseline and candidate. No case or gate changes after
baseline.

## Verification and review

- build-artifact dependency-contract tests
- focused CPU indexing/delegation/error tests
- repository `scripts/check-pr-fast.sh --coverage-reviewed` with a focused
  indexing test command
- final committed repository-rules review
- independent exact-diff `reviewer-flash` verdict
- hosted CI, auto-merge, and final `origin/main` pin verification

## Gate status

`reviewer-flash` reviewed exact design commit `d4ff55b` with high thinking and
a read-only boundary. Verdict: **Correct-to-merge**; baseline and pin work may
proceed. Before baseline, repository evidence confirmed:

- `git merge-base --is-ancestor acdeea3f 75fb0f70`: pass, so the target includes
  both #242 and #244;
- target `Cargo.toml` declares workspace version 0.4.0 and all four pinned
  packages at 0.4.0.

The fixed probe arm order is recorded above.

## Baseline evidence

A release probe built from baseline pin `39111bd7` ran the complete one-thread
arm on CPU 2 (selected 0.5%, L3 sibling maximum 11.0%) and then the complete
four-thread arm on CPUs 16,17,18,20 (selected below 2%, sibling maximum 10.8%).
Both load gates passed.

| case | 1-thread median ± IQR | 4-thread median ± IQR |
|---|---:|---:|
| rank-4 gather | 15.493624 ± 0.148721 ms | 1.694236 ± 0.052341 ms |
| rank-4 dynamic slice | 10.196747 ± 1.126156 ms | 0.826853 ± 0.005960 ms |
| rank-4 dynamic update | 2.925773 ± 0.361855 ms | 0.919375 ± 0.005470 ms |
| rank-1 gather control | 0.326885 ± 0.026871 ms | 0.276394 ± 0.107672 ms |
| rank-1 dynamic slice control | 0.078981 ± 0.001040 ms | 0.068561 ± 0.001899 ms |
| rank-1 dynamic update control | 0.130352 ± 0.011981 ms | 0.122812 ± 0.000490 ms |

The need gate is **PASS for all three families**: every one-thread generic
median exceeds 1.0 ms and is at least 22x its rank-one control. The cases and
gates are now frozen. Pin update, candidate timing, local verification, and
final review remain pending.
