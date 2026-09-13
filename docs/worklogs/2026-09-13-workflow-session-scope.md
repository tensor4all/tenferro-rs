# Shared CPU workflow admission

## Decisions

- Extend the existing one-entry managed-session contract across high-level
  serial workflows: eager forward/derivatives/extensions and prepared runtime
  execution. The [CPU execution design](../design/cpu-backend-execution.md#workflow-execution-scopes)
  specifies the public capability and exact ownership boundaries.
- Scope state retains an owned engine and admission permit, with no borrowed
  backend pointer or persistent TLS cache. Match actual engine identity, since
  different coordinators can use the same numeric domain ID.
- Admission precedes backend/cache locks on every executing entrance, including
  ordinary calls competing with a workflow and runtime output materialization.
  This prevents a permit/backend-lock inversion reproduced by concurrent eager
  integration tests. Metadata-only inspection may retain its short lock.
- Callback backend work is serial. Active operation re-entry and managed-worker
  child entry are rejected. Arbitrary caller-created parallel tasks must not
  invoke backend work or wait for competing admission within the callback.
- External CPU executors retain operation-level fallible admission. GPU event
  semantics remain backend-owned; automatic regions end at transfer, barrier,
  collective, and executor boundaries.

## Verification conclusions and constraints

- Executor counters establish one actual entry for repeated owned, cached,
  borrowed-session, and prepared graph operations with one/four CPU workers,
  including Accelerate GEMM. All matrix elements and eager derivative values
  are checked independently of timing.
- Regression coverage includes same-ID independent engines, exclusion between
  operations, unwind recovery, external executor entry counts, recursive-entry
  rejection, and competing shared/ordinary eager calls on one runtime.
- Local formatting, CI-parity clippy and committed-head deterministic rules
  review pass. Accelerate CPU unit coverage passes 550 tests; the seven workflow
  checks also pass in release. AD/runtime execution integration passes 352/146
  tests. Public scope examples pass as doctests.
- One existing compile-fail snapshot differs only in diagnostic underline
  formatting on the local toolchain. The same import on baseline `28dfc7e3`
  produces that exact formatting; the forbidden owner remains inaccessible.
  Disabling the compiler wrapper removes a separate remapped-path discrepancy.
  Snapshot expectations are left unchanged.
- No benchmark rerun or measured speedup/non-regression claim is included.
  LAPACK batch loops, repeated factorization, and eager AD source compilation
  remain independent causes from the benchmark investigation.
- This introduces public execution-scope capability/API. Existing issue #1762
  does not preapprove new public APIs; issue intake and maintainer acceptance
  are required before an implementation PR.
