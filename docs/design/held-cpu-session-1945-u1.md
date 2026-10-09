# Held CPU session: ownership and entry proof (#1945 U1)

**Status:** specification half of work package U1 of
[tenferro-rs #1945](https://github.com/tensor4all/tenferro-rs/issues/1945)
("specify root/child/drop/lock graphs, access compatibility, scoped error precedence,
provider route inventory and session audit"). The **prototype** is the other half of U1
and is not delivered by this document.

Baseline: `origin/main` `763ba4c5034f952ef33601b14dd307ce2dc1ea8a`. Scope: the CPU held
root session. Held eager sessions (U2-AD), held CUDA sessions and session-bound
submission (U3), asynchronous transfer (U4) and the adoption gates (U5) are out of scope
and are not designed here. The #2004 admission-only entry and the #2044 child ticket are
the starting baseline, not deliverables of this change.

## 1. Why holding the admission is the whole problem

A CPU session callback borrows, for the callback scope only:

| Borrowed | Source | Held requirement |
| --- | --- | --- |
| `ResourcePermit` (owns the arbiter reservation; not `Clone`) | `backend.rs:1814,1822`; `arbiter.rs:381-425` | owned by the session |
| `EngineResources` (`Buffers`, `GemmAnalysisCache`, `IndexedPlanCache`, `nary: Option<ContractionWorkspaces>`, `runtime_clears`) | `MutexGuard` in `with_execution_resources`, `backend.rs:1687-1692`; `engine.rs:12-42` | checked out by value, §2 |
| `entered` context | `provider.rs:51-54,260-274` | domain facts + `ParallelMode`; no retirement resource |
| `CallerAffinityGuard` | `backend.rs:1819-1820`; `affinity.rs:130-146` | lives for the session |
| `EXECUTION_OWNER` thread-local marker | `with_execution_owner`, `arbiter.rs:30-50` | stays set on the opening thread |
| `child_backend` (owner-inheriting clone) | `with_inherited_owner`, `backend.rs:1746-1751` | derived on demand, not stored |
| `BufferPoolLoan` (in-flight accounting, replenish on unwind, clear on return) | `backend.rs:238-259,1829` | kept around **every** operation view |

Three facts shape the design:

1. A short-lived operation view is legal. `entry` borrows the permit and domain
   (`provider.rs:240-247`), `context` comes from the backend, `entered` is a `Copy`
   domain-and-mode value that `enter_managed_session` produces without
   `EngineResources` (`provider.rs:267-274`), the buffer/cache references split out of an
   owned `EngineResources`, and `nary` comes from child resources or the context store
   (`backend.rs:1839-1842`). Nothing requires a borrow longer than one operation, and
   construction stays inside `tenferro-cpu`.
2. The session-spanning `Mutex<EngineResources>` is the real obstacle. #1945 forbids a
   shared cache lock spanning a kernel or a join, and holding it would deadlock against
   the engine's own management entry points, which take the same lock (`backend.rs`
   `buffer_pool_len`/`buffer_pool_stats`/`runtime_clears` paths; `lock_engine_resources`).
3. Admission is **not** a single fresh permit. `ExecutionAdmission` has three shapes
   (`backend/execution_scope.rs:100-110,204-256`): standalone root, `Shared(Arc<ResourcePermit>,
   OperationGuard)` for an enclosing `with_execution_scope`, and a reentrant child that
   takes private `EngineResources::for_child_execution` (`backend.rs:1673-1680`). A
   design that funnels every entry through a fresh-owner root constructor breaks the
   shared-scope and child paths.

## 2. Ownership graph

Engine resources are **checked out by value**, not locked for the session's duration.
`CpuEngine` initializes `EngineResources` at construction (`engine.rs:96`), so an empty
slot means "checked out", never "not yet created".

```text
CpuEngine { domain, context: Arc<CpuContext>, resources: Mutex<Option<EngineResources>> }
      │
      │  open_session()          (root shape only; see §6 for the audit consequence)
      v
CpuHeldSession<'b>                                   // !Send + !Sync
  └── state: Option<SessionState<'b>>               // disarmable; close() takes it, Drop takes it if present
        ├── backend:    &'b CpuBackend               // immutable facts; owner-inheriting child handles derived on demand
        ├── permit:     ResourcePermit              // OWNED reservation (standalone-root shape)
        ├── resources:  ResourceCheckout            // OWNED EngineResources, taken under a short lock
        ├── affinity:   CallerAffinityGuard         // OWNED
        └── owner:      ExecutionOwnerRestore       // OWNED previous EXECUTION_OWNER value
```

- `open_session` locks `resources` **briefly**, `take()`s the value, unlocks. No lock is
  held while an operation, kernel, join or admission wait runs. The empty slot is the
  exclusion proof for this engine; `open_session` never manufactures a second
  `EngineResources`.
- Each operation is a method on `&mut CpuHeldSession` that builds a **short-lived
  `CpuExecSession<'_>` view inside the method scope**, wrapped in the existing
  `BufferPoolLoan` so a caught panic still replenishes in-flight accounting
  (`backend.rs:238-259`), runs one lower-library call, and drops the view.
- The context/mode value is derived in the view helper from the admitted domain; it is
  not stored separately, and it owns no retirement resource.
- **The child/reentrant and shared-scope shapes are unchanged.** `open_session`
  implements only the standalone-root shape. Opening a held root inside an enclosing
  shared scope is rejected with a typed error in this phase; shared-scope reuse keeps
  going through the existing callback entry, and a reentrant child keeps its private
  `EngineResources`. A held session must not escape a shared scope by merely retaining an
  `Arc<ResourcePermit>`: the scope's `OperationGuard` also protects the eager owner lock
  (`eager.rs:2974-3008`), and `current_cpu_execution` reports `SharedScope` whenever
  `operation_active` is false (`execution_scope.rs:64-75`).
- `run_backend_session_cached` keeps its current structure for the shared and child
  shapes; only the standalone-root body is re-expressed on the held session.

## 3. Unwind, close and error precedence

One unconditional cleanup function is shared by `close` and `Drop`, so neither a `?` nor
a panic can skip a step:

```text
fn release(&mut self)                                   // takes self.state
  1. return resources         short lock; put EngineResources back   (must run first: §3.1)
  2. owner                    restore previous EXECUTION_OWNER value  (infallible)
  3. affinity                 restore caller CPU mask                 (fallible, best-effort diagnostics)
  4. permit                   drop: arbiter removes the request and broadcasts
```

- **Resources return before the admission reservation is released.** Today the callback
  and the resource guard finish before admission drops (`backend.rs:1827-1862`). With a
  checkout, releasing the permit first would wake a queued independent opener that then
  finds the slot empty and gets a spurious busy error.
- The checkout is disarmed by returning its value; the returned `EngineResources` is
  moved back into the slot exactly once, on success *and* on the error path.
- **Poison policy.** Checkout and return recover a poisoned mutex exactly as the
  execution path does today (`backend.rs:1682-1691`), while management continues to
  report poison. Removing the execution-spanning mutex deliberately removes the old
  observable "engine mutex poisoned after a callback panic" behaviour; the accounting
  behaviour that `backend/tests.rs:692-729` protects is preserved by keeping
  `BufferPoolLoan` per operation instead. This is an intentional, tested change, not a
  side effect.
- **Precedence.** The backend layer keeps today's semantics: the callback's `R` is
  returned unchanged, and a close/restoration failure is returned as an outer error even
  when the callback had also failed — it can suppress an inner work result. The generic
  `with_backend_session` takes an arbitrary `R` (`tenferro-tensor/src/backend.rs:3860-3879`),
  so "attach the close error to the work error" is not implementable there and is not
  promised. A work-error-aware precedence is only possible at a typed boundary that owns
  the result type; none is introduced by U1.
- **`SessionEntryError` documentation.** Its contract says the callback has not run
  (`tenferro-tensor/src/session_entry.rs:3-7,26-31`), but the CPU path already returns an
  affinity-restoration `Executor` error *after* running it (`backend.rs:1860-1862`). The
  held entry must not widen that gap: restoration failures are reported at `close`, and
  the documentation of the entry error is corrected in the same change.
- **`Drop` never panics.** The affinity guard's diagnostics must use best-effort output
  that ignores write errors; `eprintln!` (`affinity.rs:140-144`) can panic during
  unwinding and abort. `finish` stays fallible and keeps its retry-on-drop behaviour.
- Leaking a session retains the reservation and the checkout; that is a caller error.
  No timeout and no forced reclamation are added.

## 4. Lock graph (proposed change, and today's graph)

Today's order is: admission → affinity → owner marker → engine-resource lock
(`backend.rs:1814-1820,1827-1860`; `provider.rs:260-274`). The held root changes the third
edge to a short checkout, and management/configuration paths have their own orders
(`backend.rs:370-454`: configuration → registry/build → engine resources;
`backend.rs:1500-1536`: configuration → ordered engine guards → N-ary leases → buffer
bookkeeping). The design therefore states three distinct classes rather than one line:

| Class | Contains | Spans execution? |
| --- | --- | --- |
| Exclusion / admission | arbiter reservation; the checked-out resource slot | no (taken and released inside `open_session`/`close`) |
| Short bookkeeping | buffer-pool in-flight accounting; management/configuration guards | no |
| Numerical leases | the N-ary scratch lease (`contraction/workspaces.rs:89-120`, `try_lock`, non-waiting) | yes, per N-ary call only |

Properties:

- No execution-spanning **shared plan-cache** lock exists on the held route; the caches
  are owned by the session for its duration.
- **The plain route takes no eager owner lock.** `CpuHeldSession` never touches
  `EagerRuntime` state. U2-AD's semantic adapter must be borrowed before root admission,
  or use an exclusive nonblocking borrow proved not to wait.
- **Reentry is rejected before any blocking step.** The session keeps the same
  `EXECUTION_OWNER` marker, so an operation inside a held session, a nested
  `open_session`, or any scoped entry inside it fails with the typed
  `SessionEntryError::Reentered` (`arbiter.rs:30-36`) instead of blocking; a
  `fresh_execution_owner` failure must not be re-used as a root. A Rayon worker handed
  foreign work legitimately stays `Reentered`/`Contended`; this design does not promise
  that every scheduling placement succeeds.
- **One root owner per engine.** A second opener is queued FIFO or rejected with
  `Contended` when it cannot wait (`backend.rs:1700-1703`).
- **Management does not wait on a session.** `buffer_pool_len`, `buffer_pool_stats`,
  trim and runtime-clear report a typed busy error while the slot is empty instead of
  blocking; they already return `Result`.
- **Registry scope.** The checkout alone does not freeze registry/backend mutation:
  `for_placement` can build a managed engine without consulting the slot
  (`backend.rs:1201-1209`). This record scopes #1945's "registry/backend mutation is
  unavailable during a live root" to the *held session's own engine*: another engine may
  be built concurrently, since it has its own resources. If the intent is wider, that is
  a separate decision.

## 5. Access compatibility

Three different questions that must not be merged:

| Question | Owner | Rule |
| --- | --- | --- |
| Execution-scope witness match | `execution_admission` (`backend/execution_scope.rs:204-221`) | compares `scope.identity` with `self.runtime_identity`. `CpuRuntimeIdentity` carries **no storage authority** (`backend.rs:537-541`), so this is not a storage check. |
| Storage authority / allocation domain | runtime ingress validation (`runtime_adapter.rs:159-195`), managed materialization (`lib.rs:486-515`), provider mapping contracts (`tenferro-tensor/src/types.rs:916-942`) | the real check; reused, not reimplemented. |
| Readiness | the existing pending/overlap access protocol | matching a domain is **not** readiness admission; mapping/access keeps its current checks. |

Behaviour the held route must preserve:

- Ordinary host storage crosses CPU-budget contexts without an ownership-conversion
  copy: CPU host/view adapters borrow storage and do not compare CPU execution identity
  (`tenferro-cpu-basic/src/lib.rs:123-161`), and output CPU affinity is metadata
  (`backend.rs:52-71`). This is **not** a promise that canonicalization or every
  operation is copy-free; the measured allocation/copy counters are part of the
  prototype's evidence.
- Provider-owned/device values require the receiving session's actual provider-domain
  capability; equal ordinals or equal context IDs are not proofs. Foreign-domain or
  pending inputs are rejected typed **before** any clone, no-op or single-operand fast
  path.
- AD identity is checked by the semantic adapter (U2-AD). A plain session never inspects
  it and never detaches.

## 6. Session-entry audit

`REPOSITORY_RULES.md` §"Backend Session Entry" requires every library occurrence of a
tracked entry mechanism to sit in a function listed in
`scripts/session-entry-allowlist.json`, and states the allowlist may only shrink
(`--bless` records a removal). The checker deduplicates **mechanism/function pairs**,
excludes definitions from occurrence matching, and compares the current inventory with
the current allowlist — it is not a historical no-growth ratchet.

Current census: **45 mechanism/function entries over 42 function locations in 13
mechanism groups**; four groups are empty (three retired, plus `with_execution_scope`,
which has zero allowlisted *call* locations). Neither `open_session` nor held-session
construction is a current matcher mechanism.

A complete before/after census is therefore required, not a two-row table. At minimum it
must account for: CPU execution admission in the standalone-root body of
`run_backend_session_cached`; `CpuExecSession` construction there; the
`with_backend_session`/`with_backend_session_cached` call pair; the new root constructor
and its `CpuExecSession` view construction; and the unchanged shared/child constructor
calls. The change must also extend the matcher's mechanism list and negative tests so a
held root or child constructor is *detected* rather than hidden behind an untracked
name. If the full census is not count-neutral, that requires a reviewed policy
amendment; it is not something `--bless` may paper over. #2044's "no new host, derive a
handle" precedent applies to the child path only.

## 7. Provider route inventory

Three things that must not be conflated:

1. **Pool installation.** `CpuExecutionContext::with_native_parallelism`
   (`provider.rs:69,165-181`) installs the selected pool whenever
   `uses_inner_parallelism()` (inner mode, budget > 1, pool available,
   `provider.rs:77-81`) — there is **no operation-size check**. Installed for one
   lower-library call, never for the caller's continuation (`provider.rs:162-164`).
2. **Fan-out suppression (per-family thresholds).** These decide whether a family fans
   out or a kernel goes serial; they do not by themselves prevent pool installation.
   FFT `MIN_PARALLEL_ELEMENTS = 32 Ki` (`tenferro-fft/src/cpu/lanes.rs:44`, decision
   `:96-100`); fusion `ELEMENTWISE_FUSION_MIN_ELEMENTS = 16 Ki` declines *fusion*
   (`tenferro-cpu-fused/src/lib.rs:32,337-339`); strided `MINTHREADLENGTH = 32 Ki`;
   `tprims-exec` `MIN_PARALLEL_LEN = 32 Ki`; default width model
   `serial_below_ns = 50 us`; `tlinalg` dimension cutoff `64` additionally requires
   `width > 1 && batch >= width`. A one-thread domain keeps the tenferro-controlled
   routes inline via `uses_inner_parallelism`.
3. **Provider threads.** Vendor BLAS/LAPACK threading is the provider's own and outside
   tenferro's budget (`tenferro-linalg/src/cpu/tlinalg_blas.rs:7-12`).

Production dispatch/installation sites in tenferro: FFT lane fan-out
(`tenferro-fft/src/cpu/lanes.rs:199`), FFT installation (`tenferro-fft/src/cpu.rs:162`),
faer fold installation (`tenferro-linalg/src/cpu/linalg/faer_linalg.rs:139`), native
installation (`tenferro-cpu/src/provider.rs:178`), public context installation
(`tenferro-cpu/src/context.rs:525`). Ambient-Rayon detection reads: `arbiter.rs:168` and
`tenferro-ad/src/eager.rs:2992`. Everything else is delegated to lower libraries
(tprims, cpueinsum, tlinalg, strided, faer); the "12 delegated sites" figure is a
grouped mechanism inventory whose per-site reachability was not proven and is recorded
as unproven here.

The held session lends the same immutable domain facts the scoped path lends. It adds no
install point, no ambient pool and no second executor, and it does not by itself make
small operations avoid the pool: that is a property of the thresholds above and must be
measured.

## 8. Transfer/event machinery inventory (U1 requirement)

Reused, not replaced:

| Mechanism | Contract |
| --- | --- |
| `runtime/transfer.rs:132-147` | `TransferProvider::transfer_blocking` returns an already-ready destination; must not be relabelled non-blocking. |
| `runtime/event_domain.rs:301-321` | event-domain enqueue and `drain` retiring work before return, including on error/drop. |
| `runtime/execution.rs:102-119,612-638` | execution handle, owned input package, in-flight submission ownership. |

Open gap, deliberately left to U4: how a *concrete* session publishes a source lease and
receives an owned in-flight result without a graph-oriented submission entry. Recording
that gap is the U1 inventory; no CUDA or transfer behaviour is designed or claimed here.

## 9. Prototype plan (the other half of U1)

Scope: `CpuBackend::open_session` returning a `!Send + !Sync` `CpuHeldSession` with the
checkout of §2, one concrete operation through the view (dot-general read-into-accum with
`BufferPoolLoan`), `close`, `Drop`, and the standalone-root rewrite inside
`run_backend_session_cached`. This document is not evidence that it works.

1. Compile-fail: the session cannot cross threads or be moved into a `'static` context;
   an operation view cannot escape the session.
2. Reentry: `open_session` inside a held session, and any scoped entry inside it, return
   `Reentered` before blocking, and the session stays usable.
3. Shared scope: `current_cpu_execution()` is `Active` throughout an admitted scoped
   session and `SharedScope` afterwards; a held root opened inside a shared scope is
   rejected typed; `backend/tests/execution_scope.rs:90-183` still passes.
4. Children: `tests/child_execution.rs:15-115` still passes, including a child entered
   while a held parent owns the root resources.
5. Numerical parity with the scoped path, and instrumented proof that no
   `EagerTensor`/`TracedTensor`/semantic node/gradient slot is created and no eager owner
   lock is taken.
6. Accounting and recovery: allocate, panic inside a live held session, verify in-flight
   accounting immediately, run again, close, reopen; retain the discarded-uninitialized
   output test (`backend/tests.rs:732-756`). N-ary scratch poison after a panic is
   reported, not silently promised usable.
7. Admission fairness: queue a second root opener, close the first, and assert the queued
   opener succeeds rather than seeing an empty slot; assert exactly one resource return
   on the error path.
8. Error contract: a non-`Result` callback, an application-specific callback error, and
   simultaneous work/close failures, asserting the documented observable result without
   changing existing signatures.
9. Access: host data produced under one budget, closed, then borrowed under another
   budget through `_read_into_accum` with copy counters checked; foreign-domain and
   pending provider-backed inputs rejected before shortcuts.
10. Management while held: bounded-time `buffer_pool_len`/`buffer_pool_stats`/debug and a
    successful reopen afterwards.
11. Paired release measurements (1 and multi-thread, cold/warm) across direct
    kernel/dispatch, scoped entry, held concrete execution, untracked eager and AD,
    including wrapper allocations, lock counts, plan preparation and scratch reuse. The
    small-GEMM fixed-cost target is below 1 us on the reference machine, with hardware,
    provider, affinity and versions recorded, misses reported, and no material regression
    for scoped callers.
12. `cargo fmt`, clippy with the repository flags, the affected feature combinations,
    tests, doctests and `scripts/audit-session-entry.py --check`.

## 10. Not decided here

- Lending the pool to an outer phase scheduler from a held session (#1945 "CPU pool
  lending") — a separate scope API, owned by U2-concrete.
- Held **eager** sessions: #2044 leaves parent eager-owner re-entry invalid; U2-AD owns
  that proof.
- Held CUDA sessions and session-bound submission (U3); the transfer provider/event seam
  (U4).
- Whether `EngineResources` stays a `Mutex` once management reports busy while checked
  out, or becomes a slot with a lock only for checkout/restore.
