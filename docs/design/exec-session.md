# Execution Session Architecture

## Overview

`Engine` is the execution-time primitive surface. Ops run within a
backend-owned execution scope when the backend has one, such as a GPU runtime
or the CPU backend's reusable buffer scope. Individual ops must not re-enter
the same backend scope.

`TensorBackend::with_backend_session` creates the scope. `Runtime` owns
registered backend engines, installed extension modules, prepared-plan caches,
and extension cache state, then routes a `CompiledGraph` through segmented
execution. Consecutive backend-session instructions may run inside one backend
session.

Prepared extension operations may opt into the same scheduler-owned session by
implementing the session capability on their prepared executor. The scheduler
forms a compatible region only when every extension instruction in that region
advertises the capability; unsupported extensions remain a boundary and are
never silently retried through the ordinary per-operation path. A session-aware
executor receives `&mut dyn Engine` and must not reacquire a backend
session or let session-local state escape. Capability selection is backend-type
specific, so a CPU implementation cannot accidentally claim a CUDA or wgpu
session until that backend supplies its own mapping.

```
Runtime::run_compiled(program, inputs)
  └── runtime preparation / prepared-plan cache
        └── segmented execution
              └── fused backend segment
                    └── backend.with_backend_session(|exec| {
                            for inst in segment {
                                exec.transpose_read(...)?
                                exec.reclaim_buffer(...)
                            }
                        })??
```

## Why Sessions

Without sessions, each backend method independently prepares its execution
state and scratch-buffer access. For N-ary einsum with hundreds of small GEMM
steps, repeating that setup per instruction can dominate.

Sessions amortize that setup by creating one `Engine` for a fused
backend segment instead of one per instruction.

## Backend Mapping

### CPU (faer)

`CpuContext` stores the requested CPU thread count and owns the Rayon pool used
by tenferro-owned multi-threaded CPU work. `CpuContext::install` runs the
closure on that owned pool for multi-thread contexts and inline for one-thread
contexts — **with one exception**: `CpuContext::with_pinned_cpus` (used by the
managed engine, `CpuEngine::new_managed`) constructs a real Rayon pool even
for a single worker, so a pinned one-worker context also hands the closure off
to a Rayon worker thread rather than running it inline. Consequently the
closure given to `with_backend_session` may execute on a worker thread, and
`Send` is a soundness requirement, not a convenience bound. faer-backed
kernels use `Par::Seq` for one thread and explicit `Par::rayon(n)` otherwise,
so policy construction cannot inherit an unrelated ambient Rayon degree before
joining the `CpuContext` pool.

`CpuExecSession` implements `Engine` by calling kernel functions
directly after the session has entered `CpuContext`. Individual ops should not
re-enter the pool.

The CPU host entry is not a direct `ctx.install` + buffer swap as it once was.
`CpuBackend` owns no buffer field; admission, the `CpuOperationEntry` permit,
the engine-owned `BufferPool` loan, and provider exclusion are handled by
`CpuBackend::run_backend_session_cached`
(`crates/tenferro-cpu/src/backend.rs`), which
`BackendSessionHost::with_backend_session` calls. Read the source for the
current contract; [`cpu-backend-execution.md`](./cpu-backend-execution.md)
owns the permit and reentrancy semantics.

Which functions are allowed to reach that entry is specified in
[`explicit-session-boundary.md`](./explicit-session-boundary.md).

### CubeCL/CUDA

`CudaBackend` is the current CUDA GPU backend. It uses CubeCL/CubeCL-CUDA and
runtime-loaded CUDA libraries from `crates/tenferro-gpu/src/cubecl/`.

`CudaBackend` defines a dedicated exec-session struct, `CudaExecSession`, and
overrides `BackendSessionHost::with_backend_session` to wrap the session and
call `f` directly on the calling thread
(`crates/tenferro-gpu/src/cubecl/exec_session.rs`). WebGPU similarly overrides
with its own exec session (`crates/tenferro-gpu/src/webgpu/exec_session.rs`).
The backend session methods launch CubeCL kernels or call the relevant
cuTENSOR/cuSOLVER/cuBLAS wrapper against the backend's `CudaRuntime`.

| CPU concept | CubeCL/CUDA concept |
|---|---|
| `CpuContext` (thread count and Rayon pool) | `CudaRuntime` (CUDA device/client) |
| explicit `Par::rayon(n)` / `Par::Seq` | CubeCL launch through the stored runtime |
| `BufferPool` (host `Vec<T>`) | CubeCL device buffers plus upload/download helpers |
| faer/rayon CPU work | kernel launch on stream |
| per-step session setup overhead | per-kernel launch/runtime dispatch overhead |

GPU exec sessions run the closure on the calling thread, so `Send` is not
needed for GPU; the trait still requires it because the CPU managed path does.
Both GPU overrides call `with_session_entry_guard`
(`crates/tenferro-tensor/src/backend.rs`), so a nested entry on the same thread
is rejected with `SessionEntryError::Reentered` before its closure runs, in every
build profile.

Session entry is fallible (#1938 D6): `with_backend_session` returns
`Result<R, SessionEntryError>`, and every rejection happens before the closure
runs. CPU admission waits in FIFO order for a permit that another thread holds
and reports only states waiting cannot resolve (same-thread reentry, a busy
caller-managed domain, a scope-witness mismatch, poisoned arbiter state,
executor-entry failure). The closure's own value, including its own `Result`, is
returned unchanged inside `Ok`; see
[`tensor-session-redesign-1938.md`](tensor-session-redesign-1938.md) D6.

### Test and custom backends

A backend without resource admission wraps its closure in the portable
`with_session_entry_guard` and passes itself as the session (the pattern the
test backends use). A custom session that composes a standard one forwards the
operations it does not override; it exposes the standard session's native
services only by forwarding `native_engine()`.

## Trait Relationship

```
BackendSessionHost      — owner: admission, long-lived resources
  with_backend_session() -> Result<R, SessionEntryError>
  with_backend_session_cached()  (runtime-cache-aware, hidden)

Engine          — the only operation surface
  add_read(), dot_general_read(), reduce_sum_read(), ...
  *_into / *_read_into   — overwrite a caller-provided output
  native_engine() -> Option<NativeSessionRef<'_>>   (default None)
```

Owners no longer carry one-shot operation methods (#1929); every operation,
including standalone tensor operations, linalg multi-step logic, extension
runtimes and eager execution, runs on a borrowed `Engine`.

## Native services

Backend-leaf services that are not part of the portable operation surface
(the entered CPU execution context and buffer pool, the CubeCL client, the
WebGPU device) are reached through the leaf's safe visitor:
`tenferro_cpu::with_cpu_exec_session`, `tenferro_gpu::cuda::with_cuda_exec_session`
and the WebGPU equivalent. Each visitor asks the session for its opaque
`NativeSessionRef`, checks the leaf's crate-private marker, and performs the
single audited cast inside the leaf; the recovered reference cannot outlive
the session borrow. A token cannot be created in safe code and is not `Send`
or `Clone` (#1938 D7).

## Evaluation-wide scope (A2) is deferred

Opening one CPU execution scope for a whole evaluation or backward pass (the
`with_evaluation_scope` hook) is deliberately not implemented: a second
same-domain handle proves neither the lock order against the eager backend
owner nor single ownership of pools and caches. Operations use the existing
borrowed session at named boundaries, and the session-entry audit keeps the
hook at zero library call sites. See `explicit-session-boundary.md` and
`tensor-session-redesign-1938.md` D12.
