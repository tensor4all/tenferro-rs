# CPU binary copies and FFT output reuse (#1765, #1766)

## Scope and decisions

Base: `a096f280ab0dd774f92f533aa0f38b663a4c94d3`; branch:
`perf/1765-1766-fft-binary`. Work was isolated from the user's original checkout.

The maintainer requested one PR containing the necessary approved changes, not
an open-ended small-transform optimization project. This PR retains:

| Item | Classification | Resolution |
| --- | --- | --- |
| #1766 unnecessary operand copies | Auto Fix | Reuse typed borrowed/owned broadcast preparation for dynamic binary and ternary wrappers. |
| #1765 sequential CPU FFT lanes | Approved implementation | Disjoint lane jobs use the backend CPU context and its effective thread policy. |
| Final FFT output reuse | Approved ownership change | Existing pool, weak final-root recycler, separate short pool-state mutex. |
| Consuming eager c2c FFT | Approved API addition | Host C32/C64, shape-preserving, explicit ownership/error contract. |
| Managed output writes | Approved scope boundary | Keep pooled host staging and the provider's copy callback. |
| Small-transform/locality tuning | Out of scope | No tiling or experimental move-capture optimization included. |

See [the design contract](../design/cpu-fft-output-reuse.md) and
[the user guide](../guides/tenferro-fft.md). RustFFT is unchanged. There is no
new pool, global cache, provider synchronization API, or implicit device transfer.

## Contracts reviewed

Read the workspace/shared rules, AGENTS.md, REPOSITORY_RULES.md,
PERFORMANCE_TIPS.md, contribution/remediation workflows and storage ownership
contracts. Reference code was the existing typed broadcast helper, BufferPool,
PooledUninitOutput, HostAllocation, eager structural extraction and CPU context.
Issue reproducers supplied the performance use cases; no external algorithm
implementation was imported.

- Default backend read hooks still receive owned-tensor reads from dynamic
  wrappers; broadcasting temporaries remain alive through execution.
- A weak recycler neither retains the backend nor forms an ownership cycle.
  Final-root drop, not eager handle count, controls return. Explicit Vec
  extraction disarms return; recycled publication cancels replacement accounting.
- Incomplete output is never published. Parallel jobs own disjoint indices and
  join before releasing their output borrow; scratch is per job, not per lane.
- Recording/capture checks precede consuming ownership acquisition. Actual saved
  values and aliases remain protected by structural ownership.
- Regression tests exposed two necessary safety/correctness preconditions:
  extraction preserves layout, so noncompact inputs must be rejected unchanged;
  compact slices require the existing descriptor-bounded write guard rather
  than a whole-root slice. Offset, pointer identity and untouched surrounding
  elements are tested.
- Managed allocation permits uninitialized storage and its legacy write guard
  is copy-only. Direct managed writes and Apple hardware validation were
  explicitly deferred, not advertised as implemented.
- Shipped compute-skill mirrors were reviewed; none had conflicting FFT guidance.

## Verification evidence

Focused checks passed in the integrated worktree:

- Full `tenferro-fft` tests and doctests with `autodiff`: C32/C64 and real
  transforms, axes/norms, padding/truncation, 1T/8T, retained outputs, ordinary
  drop, explicit/cross-backend reclaim, backend-first drop, AD saved values,
  alias/capture rejection, nested execution and managed failure paths.
- Release-mode consuming FFT: seven tests plus the allocation regression.
- Runtime integration: 146 tests. The binary commit independently passed its
  allocation test and all 24 session_ops tests without the FFT changes.
- Recycler commit independently passed 60 kernel tests, 25 doctests and 36
  storage-filtered tests.
- Standard local gate: formatting, snippet synchronization, CI-parity clippy
  (workspace and standalone extensions), FFT and binary focused checks.
- Runnable examples for the new APIs/helpers and guide dependency smoke checks.
- Deterministic committed-head rules check; external LLM review skipped as
  prescribed by repository policy. No independent AI review is claimed.

A shared build target caused a suspected stale-artifact failure during an
independent-checkout verification. Rebuilding the three affected local packages
resolved it without code changes. Separate targets are used for performance
baseline/candidate builds. Required hosted CI and human review remain in force.

### Allocation observations

Explicit CPU backend 1T, requested allocation bytes (not retained memory):

| Path | Total calls / bytes | Payload-sized allocations |
| --- | ---: | ---: |
| F64 mul, 4096 elements: backend | 11 / 32944 | 1 |
| F64 mul, 4096 elements: public | 12 / 32952 | 1 |
| C64 FFT, 64x64: cold | 83 / 102805 | 1 |
| Second simultaneously live FFT output | 38 / 79701 | 1 |
| Ordinary-drop FFT reuse | 38 / 14357 | 0 |
| Explicit reclamation | 0 / 0 | 0 |
| FFT after explicit reclamation | 38 / 14357 | 0 |
| Consuming FFT | 22 / 6505 | 0 |

Thus payload copies/allocations are avoided where claimed; total allocation is
not zero. Reintroducing binary copies failed the allocation regression. The
layout/slice regressions failed before their fixes. Live-output separation and
reused data pointers are asserted by checked-in tests.

## Performance uncertainty and exclusions

Release comparisons used an AMD EPYC 7713P, explicit 1T/8T backends, the same
CPU affinity for baseline/candidate, five samples, 100ms warmup and 50ms batches.
The investigation's broad validity/non-regression gate was **not passed**.
For the retained implementation, pair8 was inconclusive: 32-cubed axis-2 spread
exceeded 20%, and small 8T three-axis overhead exceeded the experimental 2us
allowance. These observations are not proof of an implementation defect.
Other processes' CPU contention was observed; allocation-position correlation
was also observed. Environment and implementation effects were not fully
separated. No overall speedup or universal non-regression claim is made.

Tiling trials were rejected. A later move-capture trial also had an inconclusive
whole-suite result and was removed before this PR. No favorable trial subset
is presented as acceptance evidence. The maintainer explicitly requested that
additional small-size optimization stop and the necessary changes be submitted.
This scope decision does not relabel failed/inconclusive measurements as passes.

Focused coverage is not a repository-wide coverage pass: CPU 84.32%, eager
in-place 82.50%, backend 68.85%, eager extension 85%; lanes 93.57%, lib 90.97%.
Thresholds/exclusions were not changed. Hosted CI owns the complete coverage,
backend and docs matrix; GPU hardware tests were not run locally.

Detailed local evidence is retained outside the PR under
`/tmp/tenferro-1765-1766-*`: `pair8-*`, `pair9-*` (excluded trial),
`slice-release*`, `binary-allocation-totals.log`, `slice-coverage.json`,
`committed-gate-rebuilt.log`, and the archived `investigation-history.md`.
The checked-in regression tests are the reproducible correctness/allocation
artifacts; temporary diagnostic probes are not part of the shipped API or PR.
