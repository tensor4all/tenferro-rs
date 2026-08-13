# CPU Session-Open Cost Structure (eager single-op dispatch)

Status: measured 2026-08-13 on Linux x86-64, `CpuBackend::with_threads(1)`
(single worker, faer), via `TENFERRO_PROFILE_CPU_SESSION` instrumentation on
`CpuBackend::run_backend_session_cached`. Issue #1667.

This note records the per-call cost of opening a CPU backend session for one
eager op. The numbers are approximate (±30% machine noise); the *shape* of the
cost structure is the durable finding.

## Per-call breakdown (2x2 f64 solve, `no_ad`)

| Component | Cost | What it is |
|---|---:|---|
| `setup.pre_entry` | ~0.5 µs | provider-bundle clone + execution-owner fresh + `ResourceArbiter` permit + `CpuOperationEntry` construction |
| `entry.managed_session` | ~5–8 µs (wrapper only) | `enter_managed_session` machinery for a **single-worker** domain: double `with_execution_owner` guard, `install_scoped`/scoped-job indirection, `CpuExecutionContext::entered`. For `num_threads == 1` there is no Rayon pool, so the install itself is a no-op — only the wrapper runs |
| `run.resources_lock` | ~0.06 µs | engine `resources` mutex |
| `session_construct` | ~0.03 µs | `CpuExecSession` struct |
| `exec_body` | ~11–14 µs (solve) | the actual op: `with_cpu_exec_session` downcast + `to_contiguous_read` per input + faer solve |

The eager `no_ad` total for a 2x2 solve is ~40–46 µs; the eager wrapper
(validation, reads, `finish_eager_extension_outputs`, `to_tensor()`
materialization) accounts for the remainder beyond the session-open table
above.

## Key findings

1. **The `entry.managed_session` wrapper is the single biggest reducible
   component for single-worker backends.** With no Rayon pool to enter, its
   `install_scoped` indirection is pure overhead. Removing it is NOT safe in
   general: the `with_execution_owner` guards and the executor's
   `execution_scope.enter` are load-bearing for reentrancy detection and owner
   tracking (see `docs/design/cpu-backend-execution.md`), and a naive skip
   showed both an allocation-profile change and inconsistent op-level
   measurements. A safe reduction must preserve the owner/reentrancy contract
   while dropping only the pool-entry indirection.

2. **The #1662 compact-`to_contiguous_read` fast path does not apply to the
   eager path.** Eager tensor reads are borrowed views (`TensorRead::View`),
   so `to_contiguous_read` still materializes them through the session's
   native entry. This is part of `exec_body` for linalg ops.

3. **The `ResourceArbiter` is cheap (~0.5 µs) but does a futex broadcast per
   acquire even when uncontended.** Skipping the broadcast on `acquire_request`
   when the new waiter is the only one (no other thread can be blocked) is a
   safe micro-optimization (implemented in #1667). The permit-drop broadcast is
   kept unconditional: the request-id-exhaustion recovery loop parks without a
   waiter-list entry, so the waiter list cannot reveal whether a thread is
   parked, and skipping the drop broadcast would risk stranding it.

4. **`exec_body` for linalg ops is the largest remaining cost** (~12 µs for a
   2x2 solve) and is dominated by the eager view-read materialization plus the
   linalg extension dispatch, not by the faer arithmetic itself.

## Measurement method

Temporary instrumentation added sections around `run_backend_session_cached`
(`setup.pre_entry`, `entry.managed_session`, `run.resources_lock`) alongside
the sections the source still records (`session_construct`, `exec_body`),
printed via `TENFERRO_PROFILE_CPU_SESSION` +
`TENFERRO_PROFILE_CPU_SESSION_PRINT_EVERY=N`. The bench target is
`crates/tenferro-linalg/benches/eager_extension_dispatch.rs`. Re-measure with
the same instrumentation if the session-open cost is revisited.

## Remaining work (not in this PR)

- A reentrancy-preserving fast path for the single-worker
  `entry.managed_session` wrapper.
- Applying the compact-read fast path to eager view reads (or returning
  `TensorRead::Tensor` for already-compact eager values).
- Reducing the linalg extension `exec_body` overhead.

Related: #1667 (this issue), #1662 (compact `to_contiguous_read` fast path),
#1628 (Mac CPU performance), #1665 (eager extension path unification).
