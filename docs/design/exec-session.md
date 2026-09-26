# Execution Session Architecture

## Overview

`BackendSession` is the execution-time primitive surface. Ops run within a
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
executor receives `&mut dyn BackendSession` and must not reacquire a backend
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
                                exec.transpose_read(...)
                                exec.reclaim_buffer(...)
                            }
                        })
```

## Why Sessions

Without sessions, each backend method independently prepares its execution
state and scratch-buffer access. For N-ary einsum with hundreds of small GEMM
steps, repeating that setup per instruction can dominate.

Sessions amortize that setup by creating one `BackendSession` for a fused
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

`CpuExecSession` implements `BackendSession` by calling kernel functions
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
(`crates/tenferro-tensor/src/backend.rs`), so nested entry is caught by the
portable in-session guard — in **debug builds only**. Release-mode nested-entry
enforcement for the GPU overrides is still open (see
`session-oriented-concrete-apis.md`).

### Default (no-op)

Backends that don't need session batching use the default implementation
which wraps the backend itself as a `BackendSession` via `BackendSessionAdapter`.

## Trait Relationship

```
TensorBackend          — factory: creates sessions, owns long-lived state
  with_backend_session()  — creates execution scope
  dot_general()        — standalone op (with per-op context entry)
  ...

BackendSession             — session surface: ops without context re-entry
  dot_general()        — op within session (no install/set_device)
  reclaim_buffer()     — return buffer to pool within session
  ...
```

`TensorBackend` methods remain for use outside `eval_exec_ir` (e.g.,
standalone tensor operations, linalg `solve` multi-step logic).
`BackendSession` is used only by the eval loop.
