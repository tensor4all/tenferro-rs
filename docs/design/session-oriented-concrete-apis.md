# Session-Oriented Concrete APIs (issue #1673)

Status: design (pre-implementation review target).

## Overview

Add a session-explicit execution surface for concrete `Tensor`/`TypedTensor`
operations (including extension-crate operations) so a caller can reuse one
borrowed `BackendSession` across a sequence of operations instead of paying a
session entry per operation. The tensor value types remain context-free and
are never owned by or lifetime-bound to a session.

The one-shot API (`a.add(&b, &mut backend)`) stays available and becomes a
thin wrapper around the session implementation where practical.

## Motivation (measured)

Session entry for managed CPU execution is ~3 µs/op (release, pinned, 1
worker; issue #1673 data comment). For trivial unary/binary ops (~3 µs total)
it is ~100% of the cost; for a 10-op trivial chain the current API pays 10
session entries (~30 µs) where one entry (~3 µs + 10 op bodies) suffices.
For compute-heavy ops (solve, ~28 µs after #1675) the session share is ~10%,
so the surface targets short chains of cheap ops; it is not a GPU
small-op-launch fix.

**Gate (from the issue):** with `T_one_shot` = 10-op trivial chain via the
current one-shot API (10 session entries) and `T_one_session` = the same
chain via one session entry, require `T_one_shot / T_one_session >= 2`
(predicted ~6). The baseline is explicit: current one-shot API, 10 session
entries.

## Verified execution model (corrects the 2026-08-14 design review)

The design review assumed `with_backend_session` runs the closure
synchronously on the calling thread and therefore proposed dropping the
`Send` bounds. That is **false for CPU**:

- `CpuBackend::with_backend_session` → `run_backend_session_cached` →
  `enter_managed_session`/`enter` → `install_scoped`
  (`crates/tenferro-cpu/src/provider.rs`) → `CpuContext::install`
  (`docs/design/exec-session.md`, "Backend Mapping / CPU") →
  `rayon::ThreadPool::install` — a **real thread handoff to a Rayon worker**,
  even for 1-thread pools (verified empirically: caller thread id differs
  from the closure thread id).
- The default (non-overriding) backend path (`default_backend_session`,
  `crates/tenferro-tensor/src/backend.rs`) runs the closure on the calling
  thread.

**Decision: keep the `Send` bounds** (`R: Send`, `f: ... + Send`) on
`with_backend_session` / `with_backend_session_cached`. They are a soundness
requirement for the CPU managed session (a non-`Send` closure would execute
on another thread). This is an inherent property of the current CPU managed
session, not a migration accident: downstream users of the session-explicit
surface must capture only `Send` state, which the one-shot API already
requires of every closure it takes.

Dropping `Send` would require redesigning CPU session execution to run
closures on the calling thread with per-op pool entry — a performance/
semantics change to the hottest segmented-execution path. That redesign is
out of scope and tracked separately if ever desired; it must NOT be assumed
as a prerequisite.

## Nested-entry prohibition: mechanism, not convention

Operations receiving a `BackendSession` must never call `with_backend_session`
internally. Enforcement by backend family:

- **CPU (release)**: `inherited_or_new_execution_owner()`
  (`crates/tenferro-cpu/src/arbiter.rs`) already panics with
  `BACKEND_REENTRY_PANIC` when a CPU session is entered while
  `EXECUTION_OWNER` is set on the thread (or an owned Rayon scope has an
  active owner). Covers the one-shot-inside-session case, including the
  pinned one-worker managed pool handoff.
- **Default adapter (debug)**: `default_backend_session` sets a thread-local
  in-session flag (Drop-guard restored, so panics in `f` still restore) and
  `debug_assert!`s it was unset at entry (implemented in PR A). Covers
  non-overriding backends.
- **CUDA / WebGPU (not yet wired)**: both override `with_backend_session` and
  call `f` directly, so neither the portable guard nor the CPU panic applies.
  Dedicated enforcement for those overrides is tracked as follow-up; the
  acceptance criterion below is scoped to CPU + default-adapter backends
  until then.

## Generic spelling

Already settled by the current signatures: entries hand out
`&mut dyn BackendSession` (no `S: BackendSession + ?Sized` generic on the
trait methods). The `_in`-surface methods take `session: &mut dyn
BackendSession` directly.

## Cache access parity

`with_backend_session_cached` is the runtime-cache-aware entry (used by the
scheduler/eager paths); `with_backend_session` is the canonical user entry
and the one-shot concrete helpers delegate to it. The session-explicit
surface uses the plain entry; extension plans (`EinsumPlan` etc.) own their
long-lived caches. The canonical user entry must not silently become the slow
entry: `with_backend_session` may forward to the cached path when a cache is
available, but the public contract is the plain entry.

## Surface design (prototype plan)

Layering (same vocabulary, no duplicate validation):

```
one-shot API: &mut TensorBackend ──enter one session──▶ session implementation
session API:  &mut dyn BackendSession ─────────────────▶ session implementation
```

1. `TensorSessionOpsExt` / `TypedTensorSessionOpsExt` with `*_in` methods
   (naming: `_in` preferred; exact names decided in the prototype).
   Implemented by calling the session's own op traits directly; broadcasting,
   dtype promotion, validation, and typed errors identical to the one-shot
   path (shared helpers — no second vocabulary).
2. One-shot methods delegate to the `_in` implementations where practical.
3. Concrete-only extensions (einsum plans first: `EinsumPlan::prepare` +
   `execute_in_session`) compose standard session ops; no
   `ExtensionOp`/`SemanticProgram`/runtime registration required at that
   tier.
4. Runtime-integrated extensions keep the semantic `ExtensionOp` + AD rules;
   runtime preparation produces an executor that runs on the borrowed
   `BackendSession` like the concrete path.

## Non-goals

- No session/runtime handle inside `Tensor`/`TypedTensor`.
- No change to the `EagerTensor` ownership/AD model.
- No unbounded or globally owned sessions; closure-scoped borrowing stays the
  default.
- No second operation vocabulary or duplicated validation.
- GPU small-op launch overhead is a separate concern (fusion/static
  execution), not addressed by session reuse.
- The one-shot API is not broken for aesthetic consistency.

## Acceptance criteria

- One session entry covers a mixed sequence of standard and extension
  concrete operations.
- Standard broadcasting, dtype promotion, validation, placement, and typed
  errors identical to the one-shot path.
- One-shot methods delegate to the same implementation where practical.
- `Tensor`/`TypedTensor` remain context-free.
- Concrete-only extension authors need no runtime/graph/AD machinery.
- Nested-entry prohibition enforced per backend family (CPU release panic +
  default-adapter debug assert; CUDA/WebGPU enforcement tracked as
  follow-up).
- `Send` bounds preserved and documented as a soundness requirement.
- The 10-op trivial-chain gate passes (≥2x, predicted ~6x).
- No material regression for large operations.
