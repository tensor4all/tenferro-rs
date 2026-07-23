# Execution-engine session-entry repair

## Summary

Repaired the reconstructed Phase 2 CPU session boundary before continuing the
execution-engine phases. A Tenferro-managed multi-operation backend session now
enters its selected executor once around the session callback. Native,
dot-general, and grouped-GEMM operations reuse that compatible executor scope
while retaining an explicit logical parallel mode.

Fallible external executors intentionally retain operation-level entry. The
generic `BackendSessionHost` callback return type cannot represent an executor
admission failure before the callback starts; changing external sessions to
one entry would otherwise require a panic, hidden fallback, or public API
break.

## TDD evidence

The initial managed-session regression test ran `add`, `neg`, `mul`, and
`dot_general` in one backend session. Before the repair, its executor install
counter failed with repeated entry (`left: 3`, `right: 1`). After the repair:

- the four-operation managed session adds exactly one install;
- a following standalone `add` still adds exactly one install;
- the full CPU debug library suite passed all 511 tests;
- the focused release managed-session test passed;
- all 10 placement-bound eager tests passed in release, including typed
  external-executor failure and per-operation external entry;
- all 11 graph-executor preflight tests passed in release.

Grouped GEMM inside an already-entered managed session selects the compatible
provider-owned route instead of recursively submitting an engine-outer job.
Direct and fallible-external paths retain their previous entry behavior.

## Lightweight paired performance check

Existing Criterion benchmarks were built at the pre-repair HEAD
`1f4c92cb` and from the candidate, then the saved executables were run under
the same one-thread, warmup, sampling, and dependency-lock conditions.
No benchmark harness or raw result was added to the repository.

| Case | Pre-repair median | Candidate median | Change |
|---|---:|---:|---:|
| 32-site explicit session, `chi=4` | 881.23 µs | 286.15 µs | -67.93% |
| Compiled add-then-mul, 4096 elements | 58.78 µs | 28.54 µs | -50.69% |
| Standalone lazy eager neg, 1 element | 10.85 µs | 11.88 µs | +11.13% |

The standalone case crossed the 5% recheck threshold, so it was rerun with
twice the warmup and measurement time and 100 samples. The approximately 11%
increase reproduced, but remains well below the agreed 50% blocking threshold;
its pre-repair median is also above the microbenchmark policy's 10 µs primary
baseline range. The two multi-operation primary cases improved materially.

## Decisions and residual risk

- Session-wide entry is limited to Tenferro-managed executors, whose scoped
  install contract is synchronous and infallible for the backend-owned path.
- An entered session may select Sequential or Inner logical mode below that
  executor boundary. Recursive Outer entry is rejected.
- The small standalone eager regression is accepted under the noise-tolerant
  policy rather than expanding this repair into another benchmark campaign.
- A future generic session API may carry a typed pre-callback admission result;
  only then should fallible external executors be considered for session-wide
  entry.
