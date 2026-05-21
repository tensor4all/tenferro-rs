# Execution Session Architecture

## Overview

`TensorExec` is the execution-time primitive surface. Ops run within a
backend-owned execution scope when the backend has one, such as a GPU runtime
or the CPU backend's reusable buffer scope. Individual ops must not re-enter
the same backend scope.

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

Without sessions, each backend method independently prepares its execution
state and scratch-buffer access. For N-ary einsum with hundreds of small GEMM
steps, repeating that setup per instruction can dominate.

Sessions amortize that setup by creating one `TensorExec` for the entire
`eval_exec_ir` loop instead of one per instruction.

## Backend Mapping

### CPU (faer)

`CpuContext` stores the requested CPU thread count as a faer parallelism hint.
It does not own a Rayon thread pool and `CpuContext::install` runs the closure
on the caller thread. faer-backed kernels use `Par::Seq` for one thread and
`Par::rayon(n)` otherwise, letting faer/rayon use the current or global Rayon
pool without a tenferro-owned pool entry on each session.

```rust
impl TensorBackend for CpuBackend {
    fn with_exec_session<R: Send>(
        &mut self,
        f: impl FnOnce(&mut dyn TensorExec) -> R + Send,
    ) -> R {
        let mut buffers = std::mem::take(&mut self.buffers);
        let ctx = Arc::clone(&self.ctx);
        let mut session = CpuExecSession { ctx: &ctx, buffers: &mut buffers };
        let result = f(&mut session);
        self.buffers = buffers;
        result
    }
}
```

`CpuExecSession` implements `TensorExec` by calling kernel functions
directly, with no Rayon pool entry per op.

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
| `CpuContext` (thread-count hint) | `CubeclRuntime` (CUDA device/client) |
| `Par::rayon(n)` / `Par::Seq` | CubeCL launch through the stored runtime |
| `BufferPool` (host `Vec<T>`) | CubeCL device buffers plus upload/download helpers |
| faer/rayon CPU work | kernel launch on stream |
| per-step session setup overhead | per-kernel launch/runtime dispatch overhead |

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
  with_exec_session()  — creates execution scope
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
