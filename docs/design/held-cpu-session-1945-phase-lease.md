# CPU phase lease for a held session (#1945 U2-concrete, phase step)

**Status:** design record for the CPU pool-lending / phase step of U2-concrete, delivered on
top of the held session of #2053 and the public surface of #2054. Revised after an
independent pre-implementation review; the review's findings are incorporated below and the
places where this record deliberately keeps a narrower promise are marked.

## 1. What is being added, and why

#1945's CPU pool lending clause:

> Expose a narrow, scoped phase lease from `&mut` root session rather than an unrestricted
> pool handle. It parks root numerical execution, exposes joined worker execution under that
> owner's budget, and returns only after all children have closed. The downstream scheduler
> owns work queues; tenferro owns admission, executor routing and lane resources.

Downstream (tensor4all-rs #859 F4 and its interpolation fan-out) needs several workers that
run tenferro numerical work concurrently under one admission owner, without a second pool,
without per-worker eager runtimes, and without a peer being stranded when one fails. #2044
already supplies the per-worker piece — a borrowed owner-inheriting child ticket with private
scratch — but nothing drives several of them under one root session and joins them.

## 2. Surface

`phase` takes `&mut CpuHeldSession`, so the borrow — not a runtime flag — parks the caller's
own numerical execution: the phase closure receives the only handle to this engine's
execution, and no root operation can run concurrently with a phase.

```text
impl CpuHeldSession<'_> {
    pub fn phase<R>(&mut self, f: impl FnOnce(&mut CpuPhase<'_>) -> R)
        -> Result<R, CpuPhaseError>;
}

impl CpuPhase<'_> {
    /// Lanes this phase will drive: the context's thread budget, at least one.
    pub fn lanes(&self) -> usize;

    /// Run `lane` on every lane and return after all of them have finished.
    pub fn run<E: Send>(
        &mut self,
        lane: impl Fn(usize, &mut PhaseLane<'_>) -> Result<(), E> + Sync,
    ) -> Result<(), PhaseRunError<E>>;
}

pub struct PhaseLane<'lane> {
    /// The worker-local execution surface. One child session stays open for the
    /// whole lane, so its scratch and plan caches are reused across work items.
    session: &'lane mut dyn BackendSession,
    cancel: &'lane AtomicBool,
}

impl PhaseLane<'_> {
    pub fn session(&mut self) -> &mut dyn BackendSession;
    /// Cooperative stop signal. Set as soon as any lane fails, errors, or unwinds.
    pub fn cancelled(&self) -> bool;
}
```

Two shape decisions that the pre-implementation review forced, and that are deliberate:

- **The lane hands out `&mut dyn BackendSession`, not a held child session.** A worker-local
  `CpuHeldSession<'static>` is not expressible today: the held session borrows its backend
  (`held_session.rs`), `open_session` is rejected on every owner-inheriting child handle, and
  the held constructor checks out the *root* resources. The expressible route is the #2044
  one: **one child `with_backend_session` callback stays open for the whole lane**, including
  the lane's downstream queue loop, which already retains that child's private scratch across
  every operation in the callback.
- **The index is a callback argument, not also an accessor**, so there is one spelling.

## 3. Contract

| Invariant | How |
| --- | --- |
| No second pool | Lanes run through the context's own pool via `ThreadPool::broadcast`; the phase never creates a pool. |
| Driver outside the target pool | The phase is rejected with a typed error when the calling thread is a worker **of that pool** (`target_pool.current_thread_index()`), before any blocking wait. |
| One lane == one worker == one child callback | Each broadcast invocation keeps exactly one child `with_backend_session` callback open (at most one after cancellation, see §5). |
| No nested phase, no implicit root entry | `phase` takes `&mut CpuHeldSession`; the lane exposes only the operation surface, so a lane cannot open a root session or a phase directly. The guarantee this phase owns is that it joins every lane callback it started: a lane that deliberately derives an owned child handle through the native visitor and spawns further work of its own must join and cancel that work itself, and the phase does not track it. |
| Bounded **tenferro-controlled** execution | Lanes are bounded by the context's thread budget; the parked driver is not a numerical lane. Vendor BLAS/LAPACK threads are the provider's and are outside this bound. |
| One-thread context | The same API runs one lane inline on the caller under the already-held root permit; it does not call entry admission again and creates no pool. It does not fall back to an inline eager backend. |
| Join before return | `run` returns only after every lane has finished, on the success, error and panic paths. |
| Cooperative cancellation | The first lane error or unwind sets the stop signal; other lanes observe it and stop pulling work. |
| Recovery | After `run` returns a lane error, the root session is still open: the next `with_session` and a fresh `phase` both work. |
| Repeatable runs | Each `run` starts with a fresh cancellation flag, so a later `run` on the same phase does real work. |
| Output publication | The phase reports success only after joining; retaining lane outputs privately and publishing them only after a successful `run` is the downstream callback's obligation, stated in the API docs and shown in the example. |

## 4. Ownership, admission and lock order

```text
CpuHeldSession<'_>                  owns admission, the resource checkout, the owner marker, affinity
  └── phase(&mut self) -> CpuPhase<'h>
        ├── root:       &'h mut CpuHeldSession<'_>     // parked for the phase
        ├── pool:       Option<&'h rayon::ThreadPool>  // the context's own pool, borrowed; None for budget 1
        ├── budget:     NonZeroUsize                   // lanes
        ├── child:      CpuBackend                     // owner-inheriting prototype, cloned per lane
        └── cancel:     AtomicBool                     // per-run stop signal, borrowed by the lanes

`CpuPhase` is `!Send + !Sync`, like the held session: its inline lane runs under the
caller-affinity guard and owner marker the root session installed on its opening thread,
so moving a phase would run that lane outside the selected CPU set.
```

Admission and lock order, extending the U1 order by one step:

```text
1. thread-local reentry check        already held: EXECUTION_OWNER is this session's owner
2. target-pool membership check      typed error before any blocking wait
3. broadcast to the context pool     blocks the driver; the driver holds no lock
4. per worker: child session entry   joins the owner's active request (non-waiting), private resources
5. join                              every lane finished, then the phase returns
```

- The root's checkout is owned, not locked, so management entry points keep reporting it as
  checked out instead of deadlocking against the driver; nothing else is locked across the
  broadcast.
- A lane's child entry is the #2044 path: admitted reentrant under the issuing owner, never
  waiting, never taking a second root reservation.
- **Foreign-pool callers.** Only a worker *of the target pool* is rejected. A Rayon worker of
  some other pool is allowed through, and Rayon's blocking latch lets it process its own pool's
  work while it waits, so this is not the same blanket rejection the entry path applies to an
  unrelated owner; the policy difference is documented and tested rather than assumed.
- **Budget one.** The managed backend does construct a pinned pool even for one worker, but
  `uses_inner_parallelism` exposes no inner execution pool for budget one, so the phase runs a
  single inline lane. That lane runs under the **already-held root permit** with a local
  `EngineResources::for_child_execution`, because the ordinary child entry is rejected on the
  issuing thread while the root execution is active (`has_active_execution`). It gets private
  scratch and panic isolation like a worker lane, without a second admission and without
  relaxing same-thread child rejection.

## 5. Failure, cancellation and error precedence

```text
pub enum PhaseRunError<E> {
    /// A lane's callback returned an error. Returned after every lane finished.
    Lane(E),
    /// A lane could not enter or could not clean up. Replaces a lane error, as U1's
    /// restoration failures replace a callback value.
    Session(SessionEntryError),
}
```

- **Cancellation is armed before entry and observed after it.** A lane checks `cancelled()`
  before attempting child entry, and again after entering and before invoking the callback;
  the callback itself checks it while pulling work. Cancellation is set on callback `Err`, on
  child-entry or cleanup failure, and **during worker unwinding** through a guard installed
  before entry. Rayon's `broadcast` joins every worker and only then propagates one panic, so
  a peer that keeps looping on `cancelled()` can still keep the broadcast waiting: the documented
  contract is that a lane callback must return cooperatively and must not wait indefinitely for
  a peer, and that Rayon guarantees joining once every callback has terminated.
- **At most one child per worker.** After cancellation a lane may skip entry entirely; the
  phase never opens a second child on the same worker.
- **One error is returned, not the chronologically first.** `broadcast` returns lane results in
  worker order, so "first in time" is not available; the contract is "one lane error after
  joining".
- **Precedence.** A session (entry/cleanup) failure is returned as `Session`, replacing a lane
  error, mirroring U1's rule that a restoration failure replaces the callback's value. Otherwise
  the lane error is returned.
- **Panics** are not converted into errors: Rayon re-raises one after all workers joined, and
  the phase lease unwinds afterwards. The root session remains usable when it lives outside the
  caught unwind, which the tests assert.
- There is **no** "broadcast failed to start" or pool-health failure: `broadcast` returns its
  results directly, and a context without an inner pool is the legitimate sequential mode.
- `close`/`Drop` precedence is unchanged from U1.

## 6. Numerical routes inside a lane

Inner numerical work is not automatically phase-compatible, and the design does not claim it is:

- At the locked `tprims` revision, a packed route that would need a team wider than one refuses
  to run when execution is already on a target-pool worker (`ExecError::Unavailable`). That typed
  refusal is preserved, not papered over; the route review for the phase covers fresh-output
  GEMM/contraction, N-ary work and linalg, not only elementwise work.
- `yield_local` does not steal from other workers, so an idle lane spinning on a downstream queue
  does not pick up a peer's kernel partitions. Skewed work is instead handled by letting a
  finished lane **return its worker to Rayon's scheduler** rather than hold it in a spin loop.
  The phase adds no second scheduler and does not force numerical work to one thread.
- Active-thread claims are about tenferro-controlled execution on the selected pool. Provider
  BLAS/LAPACK teams are the provider's, and measurements record the provider's thread settings
  alongside the observed process-thread concurrency.

## 7. Tests and the plan behind them

Delivered in `crates/tenferro-cpu/src/backend/held_session/phase/tests.rs`:

| Property | Test |
| --- | --- |
| Every lane runs, every multi-thread lane is a worker of the context pool, and results agree with the single-threaded value | `phase_runs_every_lane_and_matches_a_single_threaded_result` |
| One-worker context: one lane, on the calling thread, no pool worker, no additional pool | `phase_on_a_one_worker_context_runs_one_lane_on_the_caller` |
| Target-pool caller rejected before any work | `phase_rejects_a_caller_inside_its_own_pool` |
| Foreign-pool caller admitted | `phase_runs_from_a_foreign_pool_worker` |
| Error cancels peers, which observe it, and one error is returned after joining | `phase_reports_one_lane_error_after_joining_and_cancels_peers` |
| Panic cancels peers while unwinding, every lane is joined first, and the session survives the caught unwind | `phase_joins_every_lane_before_a_lane_panic_surfaces` |
| Session failures take precedence over lane errors; a clean run is `Ok` | `session_failures_take_precedence_over_lane_errors` |
| The lease is repeatable: a second run on the same phase does real work | `phase_repeats_on_the_same_lease_after_a_failure` |
| Session usable after a caught lane failure | `phase_leaves_the_session_usable_after_a_caught_lane_failure` |
| Unrelated non-waiting owner rejected while inherited child numerical work succeeds | `unrelated_owner_is_rejected_while_lanes_run` |
| `!Send + !Sync` lease | `phase_lease_is_neither_send_nor_sync` |
| More lanes than work items: every lane runs, unclaimed lanes find an empty queue, and a lane 64x larger than its peers still completes | `phase_with_empty_and_skewed_lanes_completes` |
| Concurrent lane execution within the thread budget, observed inside the lanes around their numerical work, with every lane on a worker of the context pool | `phase_bounds_concurrent_lane_execution_by_the_thread_budget` |
| A lane runs contraction, transpose, reshape and reduction routes and agrees with the single-threaded values | `phase_lane_runs_concrete_route_families` |

Two items in the previous revision of this list are closed by the tests above: lane counts
larger than the task count with empty and skewed queues, and representative concrete routes
inside a lane. What the bound test does *not* claim is the fan-out of one lane into the lower
libraries: a lane bounds that only through the token this crate passes to its session, so
observing kernel-internal thread counts would need instrumentation in those crates.

Still open, and not claimed:

- A worker that arrives after cancellation, and numerical work still in flight when
  cancellation is set.
- An unwind inside private N-ary scratch, and recovery from it. The N-ary einsum and linalg
  routes that use that scratch belong to the extension crates, which depend on this one, so
  they are not reachable from this crate's tests; they are exercised in their own crates.
- `cargo fmt`, clippy `-D warnings`, the whole `tenferro-cpu` suite, doctests and
  `scripts/audit-session-entry.py --check` all pass with the new authorized caller recorded in
  the session-entry allowlist.

## 8. Out of scope

- A work-queue implementation: the downstream scheduler owns queues; the lane callback pulls
  from them.
- Rollback of caller-owned `_into` outputs or shared queue mutations. The phase promises
  "success only after joining"; publishing outputs privately only after a successful run is the
  downstream callback's discipline.
- Any second pool, detached child, external pool injection, or team-wide/SPMD barrier inside
  independently progressing lanes.
- MPI, AD/capture across lanes, and CUDA phases.
