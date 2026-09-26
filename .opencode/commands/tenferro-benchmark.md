---
description: Author, extend or run a tenferro-rs benchmark or performance campaign — Criterion rows, route/entry matrices, baseline capture and paired comparison — with the host-validity gates from PERFORMANCE_TIPS.md.
---

Use `$ARGUMENTS` as the scope: the benchmark or campaign to author, extend or
run, or the change whose performance is in question.

Read `PERFORMANCE_TIPS.md` in full before starting. Its
`Performance-Sensitive Tests And Benchmarks` and `Performance-Gated Experiment
Protocol` sections are the rules for writing a harness and for running a paired
experiment; this command carries no rules of its own.

Use `scripts/run-session-route-performance-gate.sh`,
`scripts/capture-session-route-baseline.py` and
`scripts/compare-session-route-baseline.py` rather than a new script. Set
`CARGO_BUILD_JOBS` explicitly (the runner defaults it to 1), keep the measurement
pinned to one core with a pinned thread count, and verify that core was idle
before and after each measurement before reporting a regression.

@PERFORMANCE_TIPS.md
