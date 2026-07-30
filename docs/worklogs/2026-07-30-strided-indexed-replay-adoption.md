# Strided Indexed Replay Adoption

## Summary

Advanced every workspace strided-rs dependency from `fa312f9e` to merged
commit `4c19952f`. The upstream change adds checked contiguous rank-one replay
for dynamic slice and dynamic update while preserving the existing generic
strided path.

## Context Read

- tenferro-rs issues #1490 and #1511
- strided-rs issues #149 and #169
- strided-rs PR #182
- `AGENTS.md`, `REPOSITORY_RULES.md`, and the shared tensor4all performance,
  numerical, documentation, and test rules

## Boundary And Correctness

- This adoption changes only the merged dependency pin. Tenferro continues to
  pass its backend-owned explicit `ExecContext`; no ambient execution policy is
  introduced.
- Upstream validates the contiguous source and destination ranges before using
  safe slice copies. Non-unit, negative-stride, and higher-rank layouts retain
  the generic coordinate/offset decoder.
- Dynamic update retains copy-then-overwrite semantics. Upstream differential
  tests cover every `KernelDType`, clamped starts, empty windows, negative
  strides, and bounded execution contexts.

## Performance Evidence

The focused public API benchmark used the full publication profile with 15
measured runs after three warmups. The baseline was tenferro-rs `5ef41cc9`
with strided-rs `fa312f9e`; the candidate was this pin update. One-thread runs
were pinned to CPU 60 and four-thread runs to CPUs 60-63.

| Row | Baseline | Candidate | Change | Bootstrap 95% interval |
| --- | ---: | ---: | ---: | ---: |
| dynamic_slice direct t1 | 19.435 ms | 0.809 ms | -95.84% | [-96.03%, -95.82%] |
| dynamic_slice trace t1 | 19.490 ms | 0.809 ms | -95.85% | [-95.97%, -95.67%] |
| dynamic_update direct t1 | 10.496 ms | 1.082 ms | -89.69% | [-89.76%, -89.64%] |
| dynamic_slice direct t4 | 2.600 ms | 0.740 ms | -71.53% | [-71.81%, -70.17%] |
| dynamic_slice trace t4 | 2.320 ms | 1.195 ms | -48.49% | [-59.76%, -36.79%] |
| dynamic_update direct t4 | 2.195 ms | 1.097 ms | -50.00% | [-51.60%, -46.23%] |

All candidate-relative upper bounds are below the +20 percent stop-the-line
threshold. The former dynamic-slice one-thread anomaly is removed.

## Verification

- strided-rs PR #182 passed formatting, workspace tests, documentation,
  coverage, and Linux/macOS CI before squash merge.
- Tenferro's focused CPU indexing tests and fast-check passed. The
  repository-rules review reported no findings.
- The source-contract check continues to require one exact merged revision
  across all five strided workspace dependencies.

## Remaining Work

- The full cross-row campaign record remains under tenferro-rs #1490.
- Additive scatter and raw copy replay remain intentionally serial under the
  strided-rs #164 decision and are not changed by this adoption.
