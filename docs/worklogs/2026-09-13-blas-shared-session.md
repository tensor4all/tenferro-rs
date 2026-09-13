# BLAS shared-session executor reuse

## Decisions

- Remove the provider-default exclusion from managed session entry. It made
  BLAS sessions re-enter the executor for each operation despite holding one
  session-wide resource permit. The existing entered-context path covers both
  cached and ordinary sessions without changing provider worker policy.
- Extend the existing CPU Threading Contract rather than adding a separate
  rule. The durable contract is in
  [CPU backend execution](../design/cpu-backend-execution.md).

## Verification conclusions and constraints

- Regression coverage counts exactly one executor entry per ordinary/cached
  session at one and four threads, and checks native
  addition, numerical BLAS GEMM results, the linalg callback boundary, cached
  sessions, provider exclusion, and recovery after callback panic.
- All 543 Accelerate-enabled CPU unit tests and the local formatting/clippy
  gate pass. The previous BLAS-specific four-entry test expectation now
  enforces the same single-entry contract as other managed sessions.
- The maintainer explicitly requested implementation and merge before rerunning
  benchmarks because they are switching to mobile mode. Paired performance
  measurement and the usual performance promotion gate are deferred by that
  instruction; this change makes no measured speedup or non-regression claim.
  Existing pre-fix measurements remain in
  [tenferro-benchmark](https://github.com/tensor4all/tenferro-benchmark/blob/c299d086bdd8e9ecea0a0ab95b76e1e6d985a87d/docs/m5-shared-session-refresh-20260913.md).
