# Execution Session Architecture

## Overview

`TensorExec` is the execution-time primitive surface. Ops run within a
backend-owned execution scope when the backend has one, such as the owned rayon
pool used by CPU execution. Individual ops must not re-enter the same backend
scope.

`TensorBackend::with_exec_session` creates the scope. `eval_exec_ir` runs
inside the session.

```
eval_exec_ir(backend, program, inputs)
  └── backend.with_exec_session(|exec| {
          // ALL instructions run here — one scope entry
          for inst in program {
              exec.dot_general(...) // no per-op context switch
              exec.transpose(...)
              exec.reclaim_buffer(...)
          }
      })
```

## Why Sessions

Without sessions, each backend method independently enters the execution
context (e.g., `rayon::ThreadPool::install()` for CPU). For N-ary einsum
with hundreds of small GEMM steps, this per-step overhead dominates.

Sessions amortize the context-entry cost: one `install()` call for the
entire `eval_exec_ir` loop instead of one per instruction.

## Backend Mapping

### CPU (faer)

faer's `Par::rayon(0)` uses `rayon::current_num_threads()` and
`rayon::scope()` internally. It relies on the rayon "current pool" context
set by `ThreadPool::install()` — there is no API to pass a ThreadPool
reference directly.

```rust
impl TensorBackend for CpuBackend {
    fn with_exec_session<R: Send>(
        &mut self,
        f: impl FnOnce(&mut dyn TensorExec) -> R + Send,
    ) -> R {
        let mut buffers = std::mem::take(&mut self.buffers);
        let ctx = Arc::clone(&self.ctx);
        let result = ctx.install(|| {
            // rayon context active for entire session
            let mut session = CpuExecSession { ctx: &ctx, buffers: &mut buffers };
            f(&mut session)
        });
        self.buffers = buffers;
        result
    }
}
```

`CpuExecSession` implements `TensorExec` by calling kernel functions
directly — no `install()` or `install_with_pool()` per op.

### CubeCL/CUDA

`CubeclBackend` is the current CUDA GPU backend. It uses CubeCL/CubeCL-CUDA and
runtime-loaded CUDA libraries from `tenferro-tensor/src/cubecl/`.

Today `CubeclBackend` does not define a separate exec-session struct. It uses
the default `TensorBackend::with_exec_session` adapter, so each `TensorExec`
call forwards to the backend method. The backend method launches CubeCL kernels
or calls the relevant cuTENSOR/cuSOLVER/cuBLAS wrapper against the backend's
`CubeclRuntime`.

| CPU concept | CubeCL/CUDA concept |
|---|---|
| `CpuContext` (rayon ThreadPool) | `CubeclRuntime` (CUDA device/client) |
| `ctx.install()` | CubeCL launch through the stored runtime |
| `BufferPool` (host `Vec<T>`) | CubeCL device buffers plus upload/download helpers |
| `Par::rayon(0)` | kernel launch on stream |
| per-step `install()` overhead | per-kernel launch/runtime dispatch overhead |

Future CubeCL work may introduce a dedicated GPU exec session if there is a
measurable benefit from binding temporary workspace, stream state, or device
buffer pooling across an entire compiled program. That should extend
`CubeclBackend`; it should not add a separate `CudaBackend` type.

### Default (no-op)

Backends that don't need session batching use the default implementation
which wraps the backend itself as a `TensorExec` via `BackendExecAdapter`.

## Trait Relationship

```
TensorBackend          — factory: creates sessions, owns long-lived state
  with_exec_session()  — enters execution scope
  dot_general()        — standalone op (with per-op context entry)
  ...

TensorExec             — session surface: ops without context re-entry
  dot_general()        — op within session (no install/set_device)
  reclaim_buffer()     — return buffer to pool within session
  ...
```

`TensorBackend` methods remain for use outside `eval_exec_ir` (e.g.,
standalone tensor operations, linalg `solve` multi-step logic).
`TensorExec` is used only by the eval loop.
