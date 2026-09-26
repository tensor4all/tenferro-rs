---
name: tenferro-benchmark
description: Use when authoring, extending or running a tenferro-rs benchmark or performance campaign: adding Criterion rows for a new op or route, building an entry/route matrix, capturing or recapturing a baseline artifact, running a paired baseline/candidate comparison, or investigating whether a change regressed performance. Covers harness authoring, host-noise validity gates and this repository's campaign tooling; the rules themselves live in PERFORMANCE_TIPS.md.
---

# Tenferro Benchmark

This is a thin launcher; it carries no rule content.

1. Read `PERFORMANCE_TIPS.md`. The sections `Performance-Sensitive Tests And
   Benchmarks` (how a harness is written) and `Performance-Gated Experiment
   Protocol` (how a paired experiment is run and reported) govern this work.
2. Use the repository tooling instead of writing a new script:
   - `scripts/run-session-route-performance-gate.sh` — campaign runner: pinned
     Criterion settings, 1T thread environment, CPU affinity, recorded manifest.
     Set `CARGO_BUILD_JOBS` explicitly; its default of 1 serializes the build.
   - `scripts/capture-session-route-baseline.py` — run logs to a baseline
     artifact; fails closed on a dirty worktree, a missing target log or a
     case-count drift.
   - `scripts/compare-session-route-baseline.py` — paired comparison against a
     baseline artifact with predeclared thresholds.
3. Keep the harness identity explicit: which commit produced the baseline, which
   arms the harness registers, and which case names are paired. A changed arm
   list means a recaptured baseline, not a loosened expectation.
4. Before reporting a regression, apply the validity gates: the pinned core idle
   before and after each measurement, short per-case runs, a same-binary A/A run
   for the noise floor, and entry counts when the claim is about entry counts.
   A constant absolute delta across sizes is a per-operation cost; a uniform
   factor across unaffected cases is contention.
5. Record the experiment in `docs/worklogs/` with the manifest, the threshold
   verdict, and the cases that regressed or were inconclusive. Report negative
   and inconclusive results as such.
