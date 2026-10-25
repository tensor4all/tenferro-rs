# Execution engines, prepare and run ownership (#2206)

**Status:** implemented by
[tenferro-rs #2206](https://github.com/tensor4all/tenferro-rs/issues/2206). This note
records the tenferro side of the cross-repository decision in
[`core-boundary-and-execution-ownership.md`](https://github.com/tensor4all/tensor4all-meta/blob/main/docs/design/core-boundary-and-execution-ownership.md)
(tensor4all-meta). The meta document owns the policy; this note owns the tenferro
names, boundaries and invariants.

The model is three sentences: the caller owns the computation (graphs,
compiled programs, prepared programs, values and gradients); the caller owns the
execution resources (pools, streams, buffers, scratch, plans tied to device
handles) through an engine it keeps; and execution is one `run` of a prepared
program over borrowed engines, or an operation called directly on one borrowed
engine.

## 1. What lives where

| Part | Owned by | Lifetime | Shared how |
| --- | --- | --- | --- |
| Recorded graph, compiled program, prepared program, AD rules, values, gradients | the runtime | as long as the caller keeps the runtime | shared; concurrent runs use separate engines |
| Thread pool, buffer pool, N-ary contraction scratch, graph run slots, plan cache | the engine | as long as the caller keeps the engine | one run at a time through `&mut` |
| CUDA/WebGPU stream, allocation domain, vendor handles, device-tied plans, pinned staging, retirement queues | the engine | as long as the caller keeps the engine | one run at a time through `&mut` |

A runtime therefore stores no engine, stream, event or buffer. An engine stores
no graph, gradient or differentiation state. Prepared programs store target
*requirements*, not the engines that will run them.

## 2. The engine is the only execution surface

- `tenferro_tensor::Engine` is the backend-neutral operation trait. `CpuEngine`
  and the CUDA/WebGPU engines implement it. Every operation takes the engine as
  `&mut`, so one engine runs one computation at a time with no run-time lock and
  no admission handshake.
- `tenferro_runtime::ExecutionEngine` is the prepared-program target trait. It
  exposes `target()` (an immutable `EngineTarget` description), extension-cache
  borrowing and run retirement. `Runtime::run` takes
  `&mut [&mut dyn ExecutionEngine]`; slot `i` of a prepared program binds to the
  `i`-th supplied engine for that call only.
- `CpuEngine::serial()` has no pool and runs on the calling thread. `CpuEngine::new(pool)`
  takes a caller-built `Pool` (`tenferro_cpu::Pool`, the `strided_exec::Pool`) by
  value, so one pool can never back two engines. `Pool::builder()` reserves the
  worker stack that provider recursion needs.
- `CudaEngine::open(device)` opens the caller-selected device and owns its
  logical stream, allocation domain, vendor handles and retirement queues. Two
  opens are two independent engines; their tensors mix only through an explicit,
  self-ordering transfer.
- An operation starts on the calling thread with one placement token minted
  from the engine's pool. The token enters the pool only when an entry splits
  the work into more than one task, so a small operation pays no pool entry. The
  token is correct from a plain thread, from a worker of the engine's own pool
  and from a worker of a different pool; a serial engine never enters a pool and
  is safe inside any pool.
- There is no arbitration between engines. A caller that wants `k` concurrent
  computations holds `k` engines and runs them from `k` calling threads. tenferro
  never subdivides a pool, keeps no process-wide engine registry and does not
  choose which engine runs a task.
- A panic marks the engine broken: later operations return a typed error while
  retirement, synchronization and safe buffer returns still run. GPU work that a
  run queued but the device has not finished stays with the engine, and the
  engine bounds the memory that unfinished work retains. Some device errors break
  the whole device context, not one engine.
- `with_raw` is the only raw CubeCL/CUDA entry. It borrows the engine, pins the
  engine's context and stream for the callback, flushes queued CubeCL work on
  entry and at each CubeCL-to-CUDA switch, and restores the previous context on
  return, error and unwind. It is an execution boundary for provider calls, not a
  session: it stores no caller state and does not outlive the callback.

## 3. Compile, prepare, run

| Step | Input | Output | Frequency |
| --- | --- | --- | --- |
| compile | a recorded graph | `CompiledGraph`: hardware-independent program with recorded shape constraints | once per computation |
| prepare | compiled graph, input signature, `engine.target()` list | `Prepared`: slots, placement, transfers, fusion and target requirements | once per input signature and target set |
| run | prepared program, borrowed engines, inputs | `Vec<Output<'_>>` | as often as needed |

`Runtime::prepare(program, reads, &[engine.target()])` gives each target a
numbered slot, decides which slot runs each part, freezes the required transfer
routes and memory-domain compatibility, selects fusion, and keys specialization
by the input signature plus the target requirements. `Runtime::run` binds the
supplied engines to the slots, validates the input signature once, and launches.
`Output<'a>` is either an owned compact tensor or a borrowed view of an input.

A run whose input signature is the one the handle was prepared for skips the
specialization projection: preparation already validated that signature, and any
other signature takes the full validation path. A schedule whose every location
is the same place — one engine, one storage class, one event domain, no transfer
node — keeps each slot's value in the command array itself, so a command neither
stages its inputs nor publishes its results into a per-location list. Transfers
and several engines keep the per-location form.

On CPU, `run` returns when the work is done. On GPU, it returns once the work is
queued and it no longer reads caller-mutable host or external memory;
same-engine reads and later operations are stream-ordered, and
`engine.synchronize()` or a host read is the explicit completion point. Plans
called directly on an engine come from that engine's `PlanCache`; a lookup hands
out an immutable plan and releases the cache before the computation, so engines
that share a cache never wait for each other's work.

### Warm-up

Engines create vendor handles lazily and compile a kernel on its first launch, so
a caller that measures latency or has a first-use deadline must pay that cost at
a moment it chooses. Two explicit entries cover it, both owned by the caller:

- `CudaEngine::warm_up()` creates the backend state the engine defers (the
  cuTENSOR handle) and synchronizes. It compiles no kernel, because a kernel
  belongs to the program that uses it.
- `Runtime::warm_up(prepared, engines, reads)` performs one ordinary run of a
  prepared program and then synchronizes every engine. A prepared program is
  specialized to its input signature, so that single execution compiles exactly
  the kernels it will use — fused regions, reduction strategies, vendor plans —
  and nothing has to be guessed about the backend's dispatch.

`prepare` deliberately compiles nothing: it holds engine *targets*, not engines
(#1885). A warm-up is therefore an explicit call, and its outputs are discarded.

## 4. Threads and pools

- The caller builds the pool and hands it over. tenferro never starts a hidden
  global pool, never builds a second pool inside an operation, and never installs
  or restores a per-operation execution context.
- The width is decided before the pool is built and is fixed for the engine's
  life (`engine.num_threads()`). Providers may use fewer workers for small work
  from their own work model, up to the engine's budget; they may not derive a
  width from an ambient pool.
- The engine mints one provider bundle (`CpuEngine::exec`: placement token,
  workspace, batch policy) per operation and hands it to cpueinsum/tcontract and
  tlinalg by `&mut`; strided work takes the bundle's token by value. faer
  parallelism is the token's budget handed over as a plain width with
  `Exec::external`, chosen before any pool entry.
- A downstream direct faer call is not an engine operation: it borrows the host
  column-major slices and, when it should be parallel, takes the width from
  `engine.exec()?.into_token().external(..)` and uses `Par::rayon(width)` within
  it. tenferro hands out no
  `faer::Par`, so the `FaerParallelismExt` capability the session model needed
  was removed with the engine boundary; only the `tenferro_cpu::faer` re-export
  for build alignment remains.
- Pinning is the caller's job. tenferro does not narrow and restore CPU masks per
  operation, and does not assign one CPU per worker, because vendor teams inherit
  the creating worker's mask.

## 5. What was removed

The session model is gone, not deprecated:

| Removed | Replacement |
| --- | --- |
| `BackendSession`, `BackendSessionHost`, `with_backend_session*`, `SessionEntryError` | `Engine` operations on `&mut engine` |
| `CpuExecSession`, `CudaExecSession`, `WebGpuExecSession` as public entries | `CpuEngine`, `CudaEngine`, engine implementations |
| held CPU/CUDA session, `open_session`, `adopt_held_permit`, `CpuHeldSession`, `CudaHeldSession` | engine held by the caller |
| phase lease, child-execution ticket, shared execution scope | separate caller-held engines and caller-side fan-out |
| admission permit, reentry/contention errors, `ResourceArbiter`, CPU-affinity narrowing | `&mut` borrow; the placement token is correct from any caller thread |
| CPU domain/placement tagging | caller chooses the pool and engine |
| `run_compiled`, `run_compiled_values`, `run_prepared`, `PreparedDeferred`, `submit` | `Runtime::prepare` then `Runtime::run` |

The audit in `scripts/audit-engine-entry.py` (allowlist
`scripts/engine-entry-allowlist.json`) tracks the surviving entry surfaces:
`impl Engine for T`, `impl ExecutionEngine for T`, `CudaEngine::open` and
`with_raw`. `REPOSITORY_RULES.md` "Execution Engine Entry" states the contract.

## 6. Invariants to preserve

- No operation, runtime or prepared program stores an engine.
- No hidden pool, engine or device is constructed inside an operation; engine
  construction is a named boundary and raw scopes are allowlisted.
- A prepared program keeps requirements, never engine, stream or event
  identities.
- A plan-cache lookup completes before execution starts.
- Unfinished GPU work is bounded by what the engine retains, and a single
  operation larger than the limit is rejected before submission.
- Every operation runs on the caller's thread until a provider chooses fan-out.

## References

- [core-boundary-and-execution-ownership.md](https://github.com/tensor4all/tensor4all-meta/blob/main/docs/design/core-boundary-and-execution-ownership.md)
  (tensor4all-meta): cross-repository policy.
- [execution-engine-provider-architecture.md](./execution-engine-provider-architecture.md):
  runtime-owned provider registration and scheduled execution.
- [cpu-backend-execution.md](./cpu-backend-execution.md): CPU provider routes.
- [gpu-backend-design.md](./gpu-backend-design.md): CUDA/WebGPU kernel and
  stream rules.
- [explicit-session-boundary.md](./explicit-session-boundary.md) and
  [tensor-session-redesign-1938.md](./tensor-session-redesign-1938.md):
  historical session records; their session mechanics are superseded here.
