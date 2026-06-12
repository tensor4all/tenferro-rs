# Execution Session Architecture

## Overview

`BackendSession` is the execution-time primitive surface. Ops run within a
backend-owned execution scope when the backend has one, such as a GPU runtime
or the CPU backend's reusable buffer scope. Individual ops must not re-enter
the same backend scope.

`TensorBackend::with_backend_session` creates the scope. `GraphExecutor`
owns backend cache and extension runtime state, then routes an `ExecProgram`
through segmented execution. Consecutive backend-session instructions may run
inside one backend session.

```
GraphExecutor::eval_exec_ir(program, inputs)
  └── segmented execution
        └── fused backend segment
              └── backend.with_backend_session(|exec| {
                      for inst in segment {
                          exec.transpose(...)
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
contexts. faer-backed kernels use `Par::Seq` for one thread and
`Par::rayon(0)` otherwise, so faer joins the already-entered `CpuContext` pool.

```rust
impl TensorBackend for CpuBackend {
    fn with_backend_session<R: Send>(
        &mut self,
        f: impl FnOnce(&mut dyn BackendSession) -> R + Send,
    ) -> R {
        let mut buffers = std::mem::take(&mut self.buffers);
        let ctx = Arc::clone(&self.ctx);
        let result = ctx.install(|| {
            let mut session = CpuExecSession { ctx: &ctx, buffers: &mut buffers };
            f(&mut session)
        });
        self.buffers = buffers;
        result
    }
}
```

`CpuExecSession` implements `BackendSession` by calling kernel functions
directly after the session has entered `CpuContext`. Individual ops should not
re-enter the pool.

### CubeCL/CUDA

`CubeclBackend` is the current CUDA GPU backend. It uses CubeCL/CubeCL-CUDA and
runtime-loaded CUDA libraries from `crates/tenferro-gpu/src/cubecl/`.

Today `CubeclBackend` does not define a separate exec-session struct. It uses
the default `TensorBackend::with_backend_session` adapter, so each `BackendSession`
call forwards to the backend method. The backend method launches CubeCL kernels
or calls the relevant cuTENSOR/cuSOLVER/cuBLAS wrapper against the backend's
`CubeclRuntime`.

| CPU concept | CubeCL/CUDA concept |
|---|---|
| `CpuContext` (thread count and Rayon pool) | `CubeclRuntime` (CUDA device/client) |
| `Par::rayon(0)` / `Par::Seq` | CubeCL launch through the stored runtime |
| `BufferPool` (host `Vec<T>`) | CubeCL device buffers plus upload/download helpers |
| faer/rayon CPU work | kernel launch on stream |
| per-step session setup overhead | per-kernel launch/runtime dispatch overhead |

Future CubeCL work may introduce a dedicated GPU exec session if there is a
measurable benefit from binding temporary workspace, stream state, or device
buffer pooling across an entire compiled program. That should extend
`CubeclBackend`; it should not add a separate `CudaBackend` type.

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
