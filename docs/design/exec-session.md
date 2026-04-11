# Execution Session Architecture

## Overview

`TensorExec` is the execution-time primitive surface. All ops run within a
backend-owned execution scope — rayon pool for CPU, CUDA stream for GPU.
Individual ops must NOT re-enter the backend's execution context.

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

### CUDA (future)

The same pattern maps to CUDA execution:

| CPU | CUDA |
|---|---|
| `CpuContext` (rayon ThreadPool) | `CudaContext` (device + stream) |
| `ctx.install()` | `cudaSetDevice()` + stream selection |
| `BufferPool` (host `Vec<T>`) | `DeviceBufferPool` (cudaMalloc reuse) |
| `Par::rayon(0)` | kernel launch on stream |
| per-step `install()` overhead | per-step `cudaSetDevice()` overhead |

```rust
// Future CUDA implementation (not yet implemented)
struct CudaExecSession<'a> {
    stream: &'a CudaStream,
    device_pool: &'a mut DeviceBufferPool,
}

impl TensorExec for CudaExecSession<'_> {
    fn dot_general(&mut self, lhs, rhs, config) -> Result<Tensor> {
        // cuBLAS handle bound to stream — no per-op stream switch
        cublas::gemm(self.stream, self.device_pool, lhs, rhs, config)
    }
}

impl TensorBackend for CudaBackend {
    fn with_exec_session<R: Send>(
        &mut self,
        f: impl FnOnce(&mut dyn TensorExec) -> R + Send,
    ) -> R {
        self.device.set_current();
        let mut pool = std::mem::take(&mut self.device_pool);
        let mut session = CudaExecSession {
            stream: &self.stream,
            device_pool: &mut pool,
        };
        let result = f(&mut session);
        self.device_pool = pool;
        result
    }
}
```

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
