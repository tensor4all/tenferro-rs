# Held CUDA session: lifecycle, binding and admission (#1945 U3)

This record specifies the held concrete CUDA session of [#1945](https://github.com/tensor4all/tenferro-rs/issues/1945) U3:
how it is opened, what it owns, which binding its operations use, when the host can wait, how the
reservation interacts with the entries that already exist, and which evidence establishes each
claim. It follows [held-cpu-session-1945-u1.md](./held-cpu-session-1945-u1.md),
[held-cpu-session-1945-u2-concrete.md](./held-cpu-session-1945-u2-concrete.md) and
[held-cpu-session-1945-phase-lease.md](./held-cpu-session-1945-phase-lease.md).

Downstream consumer: [tensor4all-rs #859](https://github.com/tensor4all/tensor4all-rs/issues/859)
milestone B2, which also needs U4's transfer contract. This package is independently releasable.

**Scope statement, after two independent reviews.** The issue's U3 text asks for a held session in
which "submission is not synchronization" and in which an ordinary operation or a close never
causes a host wait. Against the pinned substrate that is not implementable inside this repository:
`client.flush()` blocks on host dispatch and on the *previous* flush's device fence (§2.1), and
CubeCL additionally retires staging automatically inside ordinary kernel execution (§2.2). Removing
those waits is a change in the pinned CubeCL fork and is specified as a separate package. **What
this package delivers instead** is: a held native operation view with a captured logical-stream
binding, one settled admission rule for held versus existing entry, lifetime-visible pre-lock
protection, explicit submit/synchronize/close with *truthful* wait semantics, and counters for the
boundaries the session can actually attribute. It is an admission/ergonomics and observability
deliverable; it does not claim reduced dispatch or synchronization overhead.

## 1. What a held CUDA session owns

The CUDA execution authority is a **borrowed session object, not the owner**:

- `impl BackendSession for CudaExecSession<'_>` (`crates/tenferro-gpu/src/cubecl/exec_session.rs`)
  is the erased operation surface every concrete route goes through.
- `CudaExecSession<'a>` borrows `&'a mut CudaBackend` and carries the crate-private
  `CudaExecSessionMarker`; `CudaBackend` implements `TensorBackend` (`cubecl/mod.rs`), not
  `BackendSession`.
- `impl BackendSessionHost for CudaBackend` constructs that session for one callback;
  `with_cuda_exec_session(&mut B, f) -> Option<R>` **visits** an existing marked session and has no
  entry-error channel. Extension crates and runtime extension dispatch reach CUDA only that way.
- There is no CUDA admission arbiter, but execution ownership exists elsewhere: the runtime leases a
  `TensorBackendExecutor` behind a mutex/condvar (`crates/tenferro-runtime/src/runtime/execution.rs`),
  and eager execution refuses to take its owner lock from a thread that already holds a session or a
  CPU execution (`crates/tenferro-ad/src/eager.rs`, `EagerRuntime::lock_backend`).

A held session therefore owns the **reservation** (one settled admission rule), the **execution
binding** (a logical `StreamId` plus device ordinal), and the **lifecycle** (explicit
submit/synchronize/close with counters and a `Drop` that cannot panic). The numerical path, the
kernels and the extension visitation do not change.

## 2. The substrate this design must tell the truth about

### 2.1 `flush` already waits

```text
tenferro  cubecl/runtime.rs  client.flush()
CubeCL    cubecl-runtime/src/client.rs   device.submit_blocking(|s| s.flush(stream))  (blocking dispatch, .unwrap())
CubeCL    cubecl-cuda/src/compute/server.rs  command → current.drop_queue.flush(Fence::new(..))
CubeCL    cubecl-runtime/src/memory_management/drop_queue/queue.rs  if let Some(event) = self.fence.take() { event.sync(); }
CubeCL    cubecl-cuda/src/compute/sync/fence.rs  wait_sync → cudarc event::synchronize
```

`flush` blocks on host dispatch, and after the first flush it also waits for the previous fence that
protects the staged batch being retired. The first flush of a session has no previous fence and
therefore only dispatches. Tenferro's existing submission points are exactly the audited interop
boundaries (with_raw entry, with_cubecl entry and exit, before a raw memset, before cuBLAS/cuTENSOR
calls, before raw D2H copies, and wherever resource resolution is itself used as a queue barrier),
plus CubeCL's own host-dispatch batching.

### 2.2 Automatic retirement waits also exist

- CubeCL flushes the drop queue on its own policy thresholds during kernel execution
  (`cubecl-cuda/src/compute/command.rs` → `drop_queue.should_flush()`).
- Tenferro's `workspace_retirement` retires with an event where one can be recorded, **falls back to
  a stream barrier** at capacity or after a failed resolution, and leaks if even that fails.
- Cross-stream vendor use synchronizes before releasing retained handles; FFT performs
  `cufft_execute_synchronize`; linalg reads device status; the runtime's `Drop` synchronizes.
- Raw kernel modules and allocations remain the unsafe caller's lifetime obligation.

These are existing facts. This package neither removes them nor claims they are gone.

## 3. Specified API

```rust
impl CudaBackend {
    /// Reserve this execution domain and capture its binding.
    pub fn open_session(&mut self) -> Result<CudaHeldSession<'_>, SessionEntryError>;
}

#[derive(Debug)]
pub struct CudaHeldSession<'session> { /* §4 */ }

impl CudaHeldSession<'_> {
    /// Run one operation on the held binding through the existing CUDA session authority.
    ///
    /// Fallible: a portable session entry (any backend's scoped callback) started on this
    /// thread from inside the callback is rejected typed, exactly as it is today for a
    /// scoped CUDA callback.
    pub fn with_session<R>(
        &mut self,
        f: impl FnOnce(&mut dyn BackendSession) -> R,
    ) -> Result<R, SessionEntryError>;

    /// Explicit submission boundary: dispatch pending device work (see §5 for the waits it
    /// inherits from the substrate).
    pub fn submit(&mut self) -> Result<(), CudaSessionError>;

    /// Explicit host barrier: submit, wait for the bound stream, resolve deferred retirement.
    pub fn synchronize(&mut self) -> Result<(), CudaSessionError>;

    /// Session-observed operation and boundary counters (§5).
    pub fn stats(&self) -> CudaSessionStats;

    /// Submit pending work, release the reservation and the binding, and report the counters.
    /// The submission is an explicit boundary, not a lifetime convenience (see §5).
    pub fn close(self) -> crate::Result<CudaSessionStats>;
}

impl Drop for CudaHeldSession<'_> { /* release the reservation; never panics; never reports */ }
```

- **Reservation is held-only.** `open_session` is new, so it can define its own rule without changing
  any existing contract: at most one held session per `CudaBackendState`, and a scoped entry
  (`with_backend_session`) is rejected typed while a held session is live on that state. Concurrent
  **scoped** callbacks on different clones keep working exactly as today — the scoped path is not
  enlisted into the reservation.
- **The operation view is the existing authority.** `with_session` hands out `&mut dyn BackendSession`
  whose concrete value is a `CudaExecSession` carrying the native marker, so
  `with_cuda_exec_session` visitation keeps working inside a held session. No parallel operation
  surface and no second numerical path.
- **Borrowed owner handle.** `open_session(&mut self)`; a caller that needs the handle back keeps a
  clone (`CudaBackend` is `Clone` over one `Arc`). No second session variant.
- **`!Send + !Sync`**, asserted by an in-source compile-time probe and pinned by a compile-fail test.
- **Failure vocabulary** reuses `tenferro_tensor::SessionEntryError`, so reentry/contention reads the
  same as on the CPU and eager paths.

## 4. Reservation, marker and binding

### 4.1 Held reservation

```text
CudaBackend (Clone)                       Arc<CudaBackendState>
  ├── held_reservation: Mutex<Option<HeldOwner>>   check-and-set, never waits
  └── CudaHeldSession<'session>  borrows &'session mut CudaBackend
        ├── HeldOwner token (cleared in close/Drop, last)
        ├── HeldSessionMarker: thread-local, visible for the WHOLE held lifetime (§4.2)
        ├── binding: StreamId captured at open + device ordinal
        └── PhantomData<Rc<()>>   (!Send + !Sync)
```

1. `open_session` takes the reservation with a check-and-set before any device call: same thread as
   the live owner → `SessionEntryError::Reentered`; another thread → the contention variant. Neither
   waits. Mutex poisoning is a typed error, not a panic.
2. `with_backend_session` (and therefore the runtime's segment path) checks the reservation first and
   rejects a new **root** with a typed error while a held session is live. Scoped-vs-scoped behaviour
   is unchanged.
3. **Extension visitation is not root entry** and is never rejected: `with_cuda_exec_session` visits
   an already-admitted session (or returns `None` for a non-CUDA session). Rejecting it would break
   linalg and runtime extension dispatch.
4. The held session is **visible for its whole lifetime** through `tenferro_tensor::HeldSessionMarker`
   / `has_held_backend_session`, the new lifetime-scoped counterpart of the callback-scoped portable
   guard. `EagerRuntime::lock_backend` and the runtime executor's lease consult it in their pre-lock
   conditions, so neither blocks on an owner lock while this thread holds a held session. The
   callback-scoped portable guard keeps its current meaning and is set only around `with_session`
   callbacks, which is what makes a nested portable entry inside a held callback behave exactly as it
   does inside a scoped CUDA callback.
5. A concrete **CPU** session on the same thread stays possible (CPU admission consults its own
   execution owner, not this marker). It may itself wait for another CPU owner — that is CPU
   admission's own policy and is not claimed to be non-waiting here.

### 4.2 The binding is a logical stream

The raw-stream cache is keyed by a bounded physical slot (`stream_id.value % max_streams`), so it is
neither thread- nor stream-identifying, and `StreamId::executes` lets caller code change the ambient
logical stream on one thread. `open_session` therefore captures the logical `StreamId` and the device
ordinal — the same thing the CUDA event domain captures when it begins a run — and every
session-owned call (operations, extension callbacks, `submit`, `synchronize`, readbacks) runs inside
that captured binding, so ambient stream changes outside the session cannot redirect session work.

### 4.3 Lock facts the implementation must respect

| Lock / state | Actual span | Rule this package adds |
| --- | --- | --- |
| Held reservation | a scoped callback or a held lifetime | taken before any other lock; check-only, never waits |
| Extension cache `Mutex` | cache lookup and, through `CudaExtensionCacheGuard`/`CudaResourceGuard`, the whole vendor call that keeps the cached resource alive (linalg holds it across execution and status reads) | unchanged; the held session does not add a path that takes it before the reservation |
| Contraction / permutation plan caches, per-stream cuBLAS handle locks | plan lookup and the vendor call they guard | unchanged |
| Workspace retirement `Mutex` | retirement bookkeeping; the blocking drain waits on device events while holding it | reached from `synchronize` and from ordinary operation teardown, as today |
| Runtime executor `Mutex`/`Condvar`, eager owner lock | runtime/eager execution | the held session never takes them; their pre-lock conditions consult the lifetime marker (§4.1.4) |

A CUDA context activation is thread state, not a lock; the raw path activates it before touching the
extension cache, and the table records that rather than asserting a global order.

## 5. What the session submits, what it waits for, and what it counts

**`submit`** calls the existing flush and therefore inherits §2.1: host dispatch plus, from the second
submit onward, the previous batch's fence wait. The session does not add a flush on any operation
return.

**`close`** performs one submission — the explicit boundary at which a caller that queued work
without calling `submit` hands it to the device — then releases the reservation and the binding and
reports the counters. It adds no wait beyond that submission; cleanup afterwards belongs to the
substrate: device allocations keep their CubeCL handles (CubeCL retires them stream-ordered),
server-owned staging stays under CubeCL's drop queue, and tenferro's own scratch keeps its current
retirement contract. `Drop` releases the reservation without submitting and without reporting.
This package adds **no** new retention guarantee: raw modules, raw allocations and vendor resources
remain the unsafe caller's obligation, exactly as today.

**Documented waits that remain** (unchanged by this package): the fence wait inside `submit`
(§2.1), CubeCL's automatic staging flush (§2.2), the with_cubecl entry/exit flushes, interop
submissions before vendor calls and raw copies, the cross-stream vendor barrier, workspace-capacity
barriers, FFT's `cufft_execute_synchronize`, linalg status reads, and the runtime's `Drop`
synchronization. Each is reported as an existing boundary, not as something this package removed.

**Counters** are limited to what the session observes and can attribute without guessing:

| Counter | Meaning |
| --- | --- |
| `held_operations` | `with_session` callbacks (a callback count, labelled as such) |
| `explicit_submits` | `submit` attempts |
| `close_submits` | submission attempts performed by `close` |
| `failed_submits` | `submit`/`close` attempts that returned an error |
| `explicit_synchronizes` | `synchronize` attempts |
| `failed_synchronizes` | `synchronize` attempts that returned an error |

Readbacks are not separately attributed: they happen inside a `with_session` callback through the
`BackendSession` surface, so the session cannot distinguish them from other operations there.

Substrate-internal submissions and waits are **not** attributed to the session: the host-dispatch
channel batches work from several callers, the automatic staging flush happens inside CubeCL, and
`WorkspaceRetirementStats` is runtime-wide with no session identity. Tests establish batching
behaviour with a known chain (N operations then one explicit submit) plus the device-state check,
not by counting someone else's flush.

## 6. Errors

| Case | Behaviour |
| --- | --- |
| Second held root on a live reservation | typed `Reentered` (same thread) or the contention variant, before any device call |
| Scoped root while held | the same typed rejection |
| Portable session entry started inside a `with_session` callback | typed `Reentered`, as today for a scoped callback |
| Reservation mutex poisoned | typed error |
| `open` binding capture | infallible by construction: it captures the logical `StreamId` and ordinal and performs no server interaction. The reservation is taken before the lifetime marker, so a rejected second held root (marker conflict) releases it again |
| `submit` / `synchronize` failure | reported by that call; the failure set is the substrate's (including CubeCL's own `unwrap` on blocking dispatch, which can panic — `Drop` contains panics, and callers that must observe failures use the fallible methods) |
| Readback failure | reported; the substrate may abandon its pinned staging slot rather than reuse it (an existing behaviour, recorded here rather than hidden) |
| Unsupported numeric route | existing typed error; no implicit CPU numerical fallback and no implicit transfer |

`close` consumes the session, so it cannot report a callback result; its own submission failure is
reported, and everything else belongs to the substrate.

## 7. Evidence plan

| Claim | Evidence |
| --- | --- |
| Numerical parity | held-session chain compared against an **independent CPU** result, not only against the scoped CUDA route |
| Held reservation | second `open_session` through a clone fails typed; scoped entry while held fails typed and does not wait; extension visitation still succeeds inside a held session; the domain is reusable after close |
| Concurrent scoped callbacks are unaffected | two scoped callbacks on clones still run concurrently with no held session present |
| Lifetime marker | eager's pre-lock check and the runtime lease reject while a held session is live on this thread, before blocking; the marker is gone after close and after an unwind |
| Callback guard | a portable entry started inside a `with_session` callback is rejected typed; the guard is restored on unwind |
| Binding | operations run under the captured logical `StreamId` even when caller code changes the ambient stream outside the session |
| `!Send + !Sync` | in-source compile-time probe plus compile-fail tests in the crate's trybuild harness |
| Counters | known-chain test: N operations then one explicit submit; failing submit/synchronize counted as failures; the automatic/interop boundaries are asserted at substrate level and documented, not attributed to the session |
| Close | close with an unsubmitted partial host batch releases the reservation without waiting, and a later readback still returns correct values |
| No regression | `tenferro-gpu` CUDA tests, `tenferro-linalg`/`tenferro-fft` GPU routes, the CPU workspace route, fmt/clippy/docs/tests/doctests, and the session-entry audit extended to CUDA held construction |
| Session-entry audit | `scripts/audit-session-entry.py`: the held-entry mechanism is renamed `CPU held session entry` -> `held session entry` with its two existing sites unchanged, and two mechanisms are added (`CudaHeldSession construction`, `held session marker entry`); the allowlist grows by four sites, each with a reason, stated in the pull request |
| Feature combinations | the CUDA feature build, the CUDA+WebGPU build and the CPU-only build all compile and pass their routes |
| Real hardware | the repository's CUDA test suite on the RunPod gate, plus a local A100 run recorded in the PR with driver/toolkit versions |

## 8. Out of scope, and what it would take

- **Non-blocking submission.** Making `submit` an enqueue that does not wait on the previous fence,
  and replacing the drop-queue fence retirement with an event-owned handoff, is a change in the pinned
  CubeCL fork. It is the prerequisite for the issue's "submit is not synchronize" contract and for
  U4's real pending H2D/D2H; it is specified as its own package and tracked against #2009.
- **The optional AD adapter** for the CUDA held session. Concrete execution needs no AD owner, while
  an eager adapter must reconcile runtime-owner borrowing, the registration clones `tenferro-ad`
  installs, and admission ordering.
- **A waiting admission policy** (a CUDA arbiter with the CPU's queueing semantics), multi-device P2P,
  and the WebGPU held session.

## 9. Open questions for maintainers

1. Should the fence-wait removal in §8 be its own package in the CubeCL fork now, or stay recorded as
   a counted exception until U4?
2. Is rejecting a *new scoped root* while a held session is live acceptable, given that concurrent
   scoped callbacks (no held session) keep today's behaviour?
