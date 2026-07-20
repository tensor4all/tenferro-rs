# ResourceArbiter Fast-Path Prototype Results

## Summary

The prototype result is formally **INCONCLUSIVE**. The predeclared host-noise
rule was violated: the candidate snapshot changed the 5-minute load average by
+56.57% and the 15-minute load average by +68.68%, both beyond the 25% limit
(the 1-minute value changed by -9.43%). The plan forbids a selective retry in
the same execution, so these measurements are retained as evidence but do not
promote the implementation and do not justify a pull request.
This result does not gate architecture phases 1 or 2.

Separately from that classification, the observed primary result missed its
continuation threshold. One-thread `backend_install_empty` improved by
10.152209%, less than the required 20%, and did not reach the strong target of
2 us or less. Queue bypass therefore was not demonstrated to be a sufficient
answer to eager entry cost on this run.

## Revisions and environment

- pre-change revision: `f1fb366f4b762d74cabcb9dc17a6d0784498805d`;
- candidate revision: `480e428727276208f539afa0a941e8acbac4dbce`;
- baseline snapshot: `2026-07-20 17:30:04+09`, load average
  `3.71 / 3.73 / 3.64`;
- candidate snapshot: `2026-07-20 18:45:50.592778815+09`, load average
  `3.36 / 5.84 / 6.14`;
- CPU: AMD EPYC 7713P, 64 logical CPUs and 64 cores, one socket and one NUMA
  node;
- process placement: CPU 0 with `RAYON_NUM_THREADS=1`;
- benchmark contexts: one, two, and four CPU threads;
- toolchain: `rustc 1.96.0 (ac68faa20 2026-05-25)` and
  `cargo 1.96.0 (30a34c682 2026-05-25)`.

The named Criterion baseline was `resource-arbiter-before`. The candidate used
the same release profile, benchmark sources, CPU pinning, Rayon environment,
and Criterion configuration. All times below are nanoseconds. Confidence
intervals are 95% intervals.

## Implementation

The candidate adds direct admission under the existing arbiter mutex when the
waiter queue is empty and the active set is compatible. Contended requests
continue through the existing queue, preserving older-waiter fairness. Request
ID allocation and active insertion are shared by direct, queued, and
nonblocking paths.

Permit release notifies the condition variable only when either a normal
queued waiter or a request-ID-recovery waiter exists. The latter is a real
condition-variable sleeper outside the normal admission queue. Direct/queued
admission and acquire/release notification counters exist only in tests.

There is no public API, scheduler, pool, session, or provider change.
`BACKEND_REENTRY_PANIC` and the current backend re-entry contract are unchanged.

## Correctness verification

The candidate verification recorded:

- focused arbiter tests: 15 passed, 0 failed;
- full `tenferro-cpu` library tests: 342 passed, 0 failed;
- allocation regression test: 1 passed, 0 failed;
- `cargo fmt --all --check`: passed;
- `git diff --check`: passed.

The focused tests cover direct admission without notifications, queued
admission with release notification, normal and unwinding release, fairness,
poison recovery, and request-ID exhaustion recovery. Request IDs are reset
only after active requests and queued waiters drain; separate recovery-waiter
accounting covers an active permit and multiple simultaneous recovery callers.
Existing re-entry rejection remains covered and unchanged.

The Task 6 fast PR gate passed on the committed documentation head, including
`check-doc-snippets.py` (`doc-snippets-ok`) and `cargo test -p tenferro-cpu
--lib` (342 passed, 0 failed). The committed-head repository-rules review also
returned `pass` with no findings or unresolved items. These correctness and
policy gates do not change the formal prototype result: it remains
**INCONCLUSIVE**, and no PR is opened from this evidence branch.

## CPU-entry results

The relative intervals in the final column are explicitly **MEDIAN change
CIs**, not mean-change intervals.

| Case | Baseline median [95% CI], ns | Candidate median [95% CI], ns | Median change [95% CI] |
| --- | ---: | ---: | ---: |
| `ctx_install_empty/1` | 0.563234 [0.557364, 0.567158] | 0.602408 [0.591253, 0.617411] | +6.955292% [+4.632560%, +10.123893%] |
| `ctx_install_empty/2` | 7940.277200 [7808.717938, 8057.605213] | 8914.027576 [8912.600233, 8919.395127] | +12.263430% [+10.663338%, +14.156595%] |
| `ctx_install_empty/4` | 12238.023057 [12054.603406, 12450.120762] | 13511.310290 [13502.995655, 13517.361314] | +10.404354% [+8.512338%, +12.085136%] |
| `backend_install_empty/1` | 7398.064362 [7269.502774, 7525.140492] | 6646.997416 [6515.207051, 6781.678823] | -10.152209% [-12.651196%, -7.998460%] |
| `backend_install_empty/2` | 7012.872381 [6928.245542, 7169.629300] | 6885.368869 [6884.035724, 6889.413970] | -1.818135% [-3.965492%, -0.620308%] |
| `backend_install_empty/4` | 7413.078926 [7253.631196, 7504.045472] | 6674.482844 [6491.041456, 6763.402920] | -9.963419% [-12.894112%, -7.529109%] |

The primary baseline and candidate median CI widths are respectively 3.4555%
and 4.0089% of their medians, within the predeclared 5% primary-width check.
For the primary case only, the separately estimated **MEAN change CI** was
[-11.952077%, -9.444616%], with a -10.678372% point estimate. This is not the
median-change CI used by the gate.

All three non-gating `ctx_install_empty` cases show statistically significant
regressions because their median-change CIs are entirely positive. They are
reported rather than attributed to the arbiter optimization, since the host
noise gate already makes the run inconclusive. Both non-primary backend cases
have median-change CIs entirely below zero and therefore do not regress under
the predeclared rule.

## Public eager results

These are the eight size-one public eager gate cases. The relative intervals
are again explicitly **MEDIAN change CIs**.

| Case | Baseline median [95% CI], ns | Candidate median [95% CI], ns | Median change [95% CI] |
| --- | ---: | ---: | ---: |
| `lazy/neg_f64/1` | 9484.270762 [9355.373449, 9620.891098] | 9044.002258 [8940.210088, 9289.124241] | -4.642091% [-6.637926%, -1.615854%] |
| `lazy/add_f64/1` | 9995.162371 [9876.498951, 10125.247748] | 9258.112923 [9035.458992, 9399.683775] | -7.374062% [-9.722877%, -5.572882%] |
| `lazy/reduce_sum_f64/1` | 9795.554533 [9682.694979, 9978.377286] | 8788.070365 [8691.617496, 8846.968277] | -10.285116% [-12.233326%, -9.088055%] |
| `lazy/dot_general_f64/1` | 12416.025709 [12027.734718, 12577.091503] | 12198.725257 [12065.738773, 12265.637276] | -1.750161% [-3.408329%, +1.416497%] |
| `materialized/neg_f64/1` | 9396.248345 [9151.982185, 9502.943925] | 8950.918683 [8842.592973, 9040.582658] | -4.739441% [-6.332974%, -1.932025%] |
| `materialized/add_f64/1` | 9982.797223 [9847.981627, 10206.385805] | 10049.254270 [9982.273547, 10123.936054] | +0.665716% [-1.511991%, +2.149142%] |
| `materialized/reduce_sum_f64/1` | 9619.388810 [9507.766247, 9725.252501] | 9169.310568 [9055.292201, 9388.668357] | -4.678865% [-6.081348%, -2.144588%] |
| `materialized/dot_general_f64/1` | 11950.645533 [11767.464579, 12192.926910] | 11862.674380 [11606.690286, 12022.715835] | -0.736121% [-3.875910%, +1.672316%] |

No eager gate case has a median-change CI entirely above zero. All eight
therefore count as non-regressing under the predeclared rule, including the
three cases whose intervals cross zero.

## Decision

The formal result is **INCONCLUSIVE**, so there is no promotion and no
prototype pull request. The load-average condition alone requires a later full
paired rerun rather than a selective retry. Even ignoring host noise, the
observed primary improvement of 10.152209% missed the fixed 20% threshold and
the 6646.997416 ns candidate median missed the strong 2 us target.

This decision affects only the independent ResourceArbiter performance lane.
Architecture phases 1 and 2 do not depend on it and may continue.

## Residual risks

- Direct admission still takes the arbiter mutex and scans active requests.
- Multi-thread contexts retain Rayon pool-entry cost, which this prototype does
  not address.
- The host was noisy enough to invalidate the paired-run classification.
- A lock-free or atomic admission design is a separate experiment with its own
  fairness and overlapping-CPU-set proof obligations.
- Request-ID exhaustion recovery now retains a separate recovery-waiter
  counter and needs continued concurrency coverage.
- No architecture phase is gated by resolving these prototype risks.
