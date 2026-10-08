# CPU child-execution ticket (borrowed) — design

Status: draft for review (not implemented).

## Problem

tenferro-rs #2004 made session entry admission-based and moved host CPU-set
arbitration into a process-global `ResourceArbiter`
(`crates/tenferro-cpu/src/arbiter.rs`, `static GLOBAL`). A request conflicts with
any *other* owner holding an overlapping CPU set, and
`arbiter.rs::acquire_request_waiting` never parks a Rayon worker: it returns
`ResourceArbiterError::Contended`, which surfaces as
`SessionEntryError::Contended { message: "a Rayon worker cannot wait for
conflicting CPU resources" }` (`crates/tenferro-cpu/src/backend.rs`,
`acquire_execution_permit`).

That rejects a pattern consumers rely on today: an outer computation enters a CPU
session, then fans work out to its own Rayon workers whose callbacks call back
into tenferro. The consumer side already joins before returning
(`hataori::map_in(…, LocalMode::Outer, …)` returns only after every callback
finished, and its API documents that nothing needs to be `'static`), i.e. the
fan-out is *scoped*. The entry is rejected even though the enclosing execution is
simply waiting for the child, so there is no deadlock risk to avoid here.

Current workaround in the consumer is to disable the feature; that is a
functional regression, not a fix.

One thing this design does **not** fix by itself: the failing consumer call site
does not currently hold an enclosing execution while it fans out
(`crates/tensor4all-partitionedtt/src/adaptive_interpolation.rs:664-696` calls
`hataori::map_in` directly), so the rejected entries may well be **siblings**
competing on the process-global default backend rather than descendants. The
consumer must first hold one execution across the fan-out and route its
numerical calls through the ticket; without that, a ticket API changes nothing
for that call site. That migration is part of the plan, not an optional extra.

## Decision

Expose the active CPU execution as a **borrowed ticket**, and let a child
execution enter with it on the current thread.

- The ticket borrows the active execution, so a child ticket cannot outlive the
  execution that produced it: the *borrow checker* enforces the lifetime
  invariant that "a child never outlives its parent execution". No owned /
  reference-counted variant is introduced in this change.
- The child's permit is acquired under the **inherited owner**, so the arbiter
  admits it as reentrant (`arbiter.rs`, `try_acquire_request_with_owner`:
  `reentrant = active.iter().any(|active| active.owner == owner)`), which never
  parks and therefore never returns `Contended` for a descendant.
- Everything about *foreign* owners is unchanged: an unrelated execution still
  conflicts and a Rayon worker still gets `Contended` rather than parking. The
  deadlock-avoidance policy of #2004 stands.

### What the reentrant path already covers, and what it does not

`crates/tenferro-cpu/src/backend.rs::with_execution_resources` already branches on
the permit:

```rust
if permit.is_reentrant() {
    let mut resources = EngineResources::new(self.shared.buffer_limit.load(Relaxed));
    return op(&mut resources);
}
let mut resources = self.engine.resources.lock()...;
```

A reentrant session builds its **own** `EngineResources` (own buffer pool and
caches) and never takes the shared `engine.resources` lock.

It is **not** sufficient on its own. N-ary contraction scratch is not part of
`EngineResources`: the store lives on `CpuContext.contraction`
(`crates/tenferro-cpu/src/context.rs:126-156`) and
`ContractionWorkspaces::with_scratch`
(`crates/tenferro-cpu/src/contraction/workspaces.rs:60-100`) takes it with
`try_lock` and documents "Resource admission normally serializes this owner.
Never wait for an unexpected concurrent or recursive workspace borrow." Two
children (or a parent and a child) that both reach a three-or-more-input einsum
(`crates/tenferro-einsum/src/cpu_concrete.rs:112-130`) therefore race and one
fails with the typed `CPU N-ary workspace` error.

So this change must also make the child's **contraction workspaces child-scoped**
(reuse the existing `ContractionWorkspaces` type; no new workspace framework),
and it must state what a child session inherits versus owns.

Resource wording that this design must not claim: the buffer limit is a
**retained-capacity ceiling for that pool**, not a live-memory or thread budget,
and the caches keep their defaults.

## API (sketch) — no new session-entry boundary

The change adds **no new session host**. It adds a way to derive an
*owner-inheriting backend handle* from the execution the caller is already
inside, and the consumer enters sessions through the **existing**
`BackendSessionHost::with_backend_session*` path:

```rust
impl CpuExecSession<'_> {
    /// A handle whose sessions are admitted under this session's execution owner.
    ///
    /// The handle borrows this session, so it cannot outlive the callback that
    /// received the session; it is cheap to clone and `Send + Sync` for workers.
    pub fn child_execution(&self) -> CpuChildExecution<'_>;
}

impl CpuChildExecution<'_> {
    /// A backend handle carrying the issuing execution's owner.
    pub fn backend(&self) -> CpuBackend;
}
```

Usage: the parent receives a session from the existing entry, derives the child
handle inside that callback, and passes `&handle` to the scoped workers it joins;
each worker calls `handle.backend().with_backend_session(…)` — the existing,
already audited entry point.

Why this shape: `scripts/audit-session-entry.py` tracks session-entry
*mechanisms* and requires every library occurrence to sit in an allowlisted
function, and `REPOSITORY_RULES.md` states the allowlist may only shrink. A new
public session API would add session-construction sites. Deriving a handle only
reads the current execution state, so the audited sites are unchanged:
`execution_admission` and `run_backend_session_cached` merely learn to honour the
owner the handle carries. Consumer-side `with_backend_session` calls are outside
the library audit.

Lifetime: the handle borrows the session (`for<'a> CpuChildExecution<'a>`), so a
ticket cannot be stored, returned, or moved into `std::thread::spawn`; the borrow
checker enforces "no child outlives the issuing execution". No `Send` bound is
added to the existing entry; the consumer imposes worker bounds where it fans out.
A `Sync` assertion plus a scoped, concurrent-use test cover the positive side, and
compile-fail fixtures cover storing the handle.

### Child entry contract

- The child acquires **its own** permit under the owner the handle carries and
  then goes through the **existing** session-construction path
  (`CpuBackend::run_backend_session_cached`), i.e. the same code that builds a
  session today. It does not reuse the issuing permit, and ordinary admission is
  never reached by installing an owner ambiently.
- Reentrant admission only means "does not wait for the issuing execution's
  reservation". It is not a claim that arbitrary numerical work runs concurrently:
  the lower executor serializes some broadcasts (tprims `exec.rs` SPMD
  broadcast), so claims stay limited to "a correctly constrained child does not
  wait for the parent's arbiter reservation".

### Ownership at child entry

| Item | Owner |
| --- | --- |
| `EngineResources` (buffer pool + caches) | the child session's `with_session` invocation (already: the `is_reentrant` branch) |
| `ContractionWorkspaces` (N-ary scratch) | the child session: a **fresh** store owned by the session's resources (move it out of `CpuContext.contraction.nary` into the per-session resources, so the reentrant branch gets it for free), obtained through `CpuExecSession::with_contraction_exec`, reused across operations inside that session and dropped with it |
| lower contraction pool / arena provider | shared, unchanged (isolate N-ary scratch only; do not rebuild `CpuContext`/`ExecutionResources` per child) |
| backend witness, domain/CPU set, pool + thread policy, allocation domain | inherited from the issuing execution |

### Public semantics to settle (proposed)

1. Recursion from the issuing thread while the callback is active stays rejected
   (`SessionEntryError::Reentered`), exactly like ordinary backend recursion.
2. Workers may hold child sessions while the issuing execution is active: that is
   the point of the ticket.
3. A child session may not issue its own ticket in this change; grandchildren use
   the same root ticket.
4. The ticket's lifetime is the issuing callback (`for<'exec>` above), so it cannot
   be stored or returned.
5. A pooled "stolen" callback that lands on a thread where an execution of the same
   owner is already active is **rejected** as `Reentered`; such a callback must use
   the session it was handed instead of entering again. Scheduler composition
   (operation-local pool install versus ticket entry) needs its own test and is
   called out as a risk rather than assumed.

## Non-goals

- No owned/reference-counted ticket for detached work. If that is ever needed it
  is a separate design (it cannot be enforced by lifetimes).
- No change to the arbiter policy, to `CpuThreadExecution`/`with_execution_scope`
  (that remains the same-thread nesting mechanism), or to the engine resource
  layout.
- No consumer-side ambient inheritance: consumers must thread the ticket
  explicitly. The legacy global-default convenience surface stays for
  non-parallel callers.

## Verification

1. New test: a Rayon worker obtains a child session from the parent's ticket and
   completes while the parent holds its own session (no parking, no deadlock).
   The decisive case is **two children admitted concurrently**, synchronized, both
   exercising the N-ary contraction workspace: assert numerical results, not just
   admission. A single child or a no-op callback would miss the shared-scratch
   defect above.
2. Deterministic scratch isolation: hold one session's N-ary scratch lease open and
   require a second ticket session to finish a numerically checked N-ary
   contraction **before** that lease is released (bounded synchronization, so a
   failure cannot hang the suite). Synchronizing two sessions before they call
   einsum is not enough: a broken implementation could still pass by running them
   sequentially.
   Also: the issuing permit's waiter stays blocked while issuance is active and
   progresses after it exits; a panic **inside N-ary scratch use** is followed by
   successful parent and fresh-child contraction; a foreign Rayon pool combined
   with a multi-threaded tenferro pool exercises operation-local pool entry; and
   the downstream interpolation regression passes once it routes through the
   ticket.
3. Existing tests: a foreign owner still gets `Contended`; the recursive-entry and
   `SharedScope` semantics are unchanged.
4. Compile-fail fixtures: a ticket moved into `std::thread::spawn`, and a ticket
   stored/returned so that it would outlive the callback while the backend stays
   alive. A passing scoped, concurrent-use fixture plus a `Sync` assertion cover
   the positive side.
5. A test that the child's own resources are accounted to the child, not the
   parent — stating the actual contract: the buffer limit is a retained-capacity
   ceiling for that pool, and the caches keep their defaults. The entirety of
   `EngineResources` is **not** bounded by the buffer limit.
6. Foreign admission keeps its meaning: a foreign owner still gets `Contended`,
   and an already-queued ordinary (non-worker) waiter progresses once the
   descendants drain. Child unwind must leave the parent able to continue and to
   admit a fresh independent execution. Grandchildren are covered by a test, not
   left implicit.
7. Repository gates: `python3 scripts/audit-session-entry.py --check` must stay
   green **with an unchanged allowlist** (that is the test that the change added
   no session-entry boundary), an audit negative test must still fail for an
   unauthorized caller, then the focused local gate from `AGENTS.md`,
   `cargo nextest run --cargo-profile ci --workspace`, clippy with the repository
   flags, `cargo test --doc`, and the panic/public-error-doc audits.

## Settled: no new session-entry boundary (decision ii)

`REPOSITORY_RULES.md` requires every library session host to be listed in
`scripts/session-entry-allowlist.json`, and the allowlist may only shrink.
Decision (taken with the maintainer): express child entry through the existing
audited entry, so the boundary count does not grow — the handle-derivation shape
in the API section above. An audit negative test for an unauthorized caller stays
part of the change.

## Deliberately not doing

- No ambient `current_ticket()` accessor and no second TLS scope registry: the
  ticket comes from the execution-bounded callback above.
- No new ticket error type; concrete admission failures reuse `SessionEntryError`.
- No `Send` bound on the synchronous child callback beyond what session entry
  already requires.
- Keep `CpuThreadExecution`, `with_execution_scope`, the same-thread operation
  guard and witness checks: they enforce distinct, existing behavior.
- No owned/reference-counted ticket for detached work (separate design).
