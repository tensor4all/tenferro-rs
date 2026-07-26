# Execution-Engine Session-Entry Repair

**Goal:** Remove repeated executor entry from compatible multi-operation CPU
sessions without weakening external-executor failure semantics or parallel-mode
ownership.

## Scope

- Tenferro-managed CPU sessions enter their selected executor once around the
  backend-session callback.
- Native operations reuse that entered context.
- Dot/general-contraction operations reuse the same executor boundary while
  selecting their own logical `ParallelMode`.
- Grouped GEMM inside an entered session uses the provider-owned compatible
  route instead of recursively submitting an outer executor job.
- Fallible external executors retain operation-level entry. The current
  generic `BackendSessionHost` callback return type cannot represent an
  executor admission error before the callback runs, so session-wide external
  entry would otherwise require a panic, hidden fallback, or API break.

## TDD sequence

1. Add test-only managed-executor install counting.
2. Add a three-operation backend-session test requiring one install.
3. Confirm RED reports three installs.
4. Add an optional entered context to `CpuExecSession`.
5. Enter Tenferro-managed sessions once in `run_backend_session_cached`.
6. Reuse the entered context for native and compatible provider calls.
7. Keep direct and external paths on existing operation-level entry.
8. Run focused unit, provider, graph, and full release tests.
9. Run paired lightweight session/compiled/eager benchmarks. Block only on a
   reproducible primary-case regression of at least 50 percent when the
   baseline median is at most 10 microseconds.

## Acceptance

- The RED install-count test becomes GREEN with exactly one install.
- Standalone operations still enter exactly once.
- Existing external executor, provider mode, outer scheduling, placement,
  unwind, and typed-error tests pass.
- Workspace release tests and docs pass.
- No Phase 2E campaign scripts or raw benchmark artifacts are added.
