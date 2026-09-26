# Session-route baseline recapture and matched comparison (#1926 / #1929)

**Date**: 2026-09-26
**Branch**: `refactor/1929-session-route-unification` (head `af916dd87` at capture)
**Baseline worktree**: `.worktrees/issue-1929-bench` at `5c55496fe` (code `25da8d431`)

## Why

The Phase-B constraint says the baseline must be recaptured when the harness
changes. Pruning the before-only `oneshot`/`one_shot` arms is such a change — they
cannot outlive the deleted spellings — so the baseline side was re-collected with
the current harness, and the capture toolchain was exercised end to end for the
first time in this workstream.

## Environment (measured, not assumed)

| Fact | Value |
| --- | --- |
| core used | `cpu=1`; `cpu=0` (the recorded runs' core) is 100% busy for 30 s averages, held by a foreign `julia` process (109 days old, `cpus=0-63`) |
| cores >50% busy | 3 of 64 (`cpu0`, `cpu8`, `cpu32`) — the three old `julia` jobs, each saturating one core |
| load at capture | baseline `3.98 7.94 9.52`, candidate `4.88 4.88 5.93` (recorded baseline was `10.89 20.08 26.11`) |
| `nproc` in the manifest | `1`, on both the recorded and the new manifest: `nproc` honours the campaign's `OMP_NUM_THREADS=1`, so this field is an artifact of the 1T configuration and not an environment change |
| GPU target | ran here, 13 cases. The CUDA *unit tests* are hardware-gated in CI, but the campaign's GPU *benchmark* runs on this host — an earlier note claiming it needs an external GPU host was wrong |

## Recapture

Campaign run in the baseline worktree with the current harness applied
(`route_matrix.rs`, `session_chain.rs`, `route_matrix_gpu.rs` copied from the
candidate tree), `--label baseline --cpu 1`. All seven targets completed: 96 cases
in total, 0 before-only.

The capture script refused the first attempt with

```text
tenferro-cpu|route_matrix: expected 47 cases, parsed 31. The harness case list
changed; recapture the baseline instead of adjusting the expectation.
```

which is the fail-closed harness-identity guard working as designed. The recapture
is the `EXPECTED_CASES` update in `af916dd87` (47→31, 6→5, 25→13) plus
`docs/testing/session-route-baseline-recaptured.json`. The frozen
`docs/testing/session-route-baseline.json` keeps all 125 rows including the 29
before-only ones and remains the before-reference.

## Comparisons

Three comparator runs over the same logs:

| Comparison | Result |
| --- | --- |
| matched: candidate vs recaptured baseline (same core, same window) | `paired_ok=49 noisy=27 regressions=20 deleted_route=0` |
| historical: candidate vs recorded baseline | `paired_ok=67 noisy=21 regressions=8 deleted_route=29` |
| A/A: candidate pass 2 vs candidate pass 1 | `paired_ok=53 noisy=7 regressions=0` |

The A/A run covers the three targets that carry the matched-run regressions and
flags none of them, so the host's run-to-run variance is below the thresholds. The
matched deltas are therefore not explained by host noise:

* `tenferro_ad/eager_backward_shape_churn/threads_1/bond_dimension_sequence/16`
  +14.9% (7.381 ms → 8.485 ms);
* ten `eager_dispatch_baseline` small-operation cases between +5.0% and +7.2%
  (`lazy/reduce_sum_f64/{1,8}`, `materialized/reduce_sum_f64/{1,8,64}`,
  `materialized/dot_general_f64/1`, `neg_f64/{1,64}`, `lazy/neg_f64/64`);
* `route_matrix/elementwise_add_f64/session/single/65536` +5.2%.

The historical comparison is a cross-environment comparison and carries a
different, partly masking signal: the recorded baseline was collected under load
10–26 on `cpu0`, so some of its numbers are slower than today's baseline.

## Interpretation

The pattern is a small, broad cost on operation-sized work plus a large absolute
cost on one multi-millisecond backward case, which is consistent with more session
entries per unit of user work after B3/B4 (the runtime now runs maximal
session-compatible runs, and extension owner entries form the session themselves).
That is an optimization question for the umbrella's Phase C, not a Phase-B
correctness question, and it is now recorded with matched evidence rather than
left to a noisy guess.

## Residual

* Certification still needs the protocol's three alternating baseline/candidate
  pairs; this host can now run all seven targets, but `cpu0` is unavailable and the
  pairs must use one free core consistently.
* The matched regression needs an owner and a decision: reduce the extra session
  entries, or accept the cost and record it.
