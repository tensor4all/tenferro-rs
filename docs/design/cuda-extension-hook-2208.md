# The extension hook: cuFFT and cuSOLVER on the new CUDA path

Status: design, not implemented. This is the last unit between the current branch state and
deleting CubeCL (#2208 U6).

## Why a hook at all

`tenferro-fft` and `tenferro-linalg` execute vendor libraries (cuFFT, cuSOLVER, and a few
cuBLAS/t4a kernel calls) against the engine's stream. They are written against
`tenferro_gpu::cuda::raw`, the CubeCL raw session. The new path already owns the stream, the
submission lock and the raw scope, so the hook is a *thin owner-scoped view* onto
`CudaRuntime::with_raw_locked` — not a second execution path. The approved rule (R-api-design-90)
is that the concrete CUDA facade keeps such hooks and the generic seam stays internal; the
extension crates keep their own operation and plan types.

## Inventory of what the two crates call

Counted from `crates/tenferro-fft/src` and `crates/tenferro-linalg/src`:

| Primitive | fft | linalg | What the new path would serve |
|---|---|---|---|
| `raw.stream()` → `.raw_handle()` | 1 | 16 | the runtime's `CUstream` (already available in the raw scope) |
| `raw.alloc_output::<T>(shape)` | – | 67 | zeroed device allocation, wrapped as a typed tensor |
| `raw.tensor(t)` / `raw.tensor_mut(t)` | 2 + 2 | 64 + 66 | typed reference/mutable reference over an existing allocation |
| `raw.alloc_bytes(n, op)` | 2 | 19 | zeroed device bytes, scoped to the hook |
| `raw.copy_bytes(..)` | – | 18 | device copy inside the scope |
| `raw.retain_tensor(t, op)` | 4 | 10 | a guard keeping the allocation alive until the work's completion |
| `raw.download_tensor(t)` | – | 12 | the existing `transfer::download_tensor` |
| `raw.upload_bytes(..)` | – | 5 | the existing `CudaRuntime::upload_bytes` |
| `raw.synchronize()` | 2 | 3 | the existing `CudaRuntime::synchronize` |
| `raw.resource(..)` | – | 15 | register an allocation with the raw scope |
| `raw.len(..)` | – | 1 | allocation length |
| `load_ptx` / `compile_nvrtc` | – | RRQR kernels | the kernel-call surface (see below) |
| `with_cublas_handle` | – | few | the `Cublas` handle the new path already owns |
| `CudaRuntime::with_current_context` | 2 | – | context activation; the new raw scope already does it |
| `CudaRuntimeIdentity` | 1 | – | `CudaRuntime::identity()` already exists on the new path |

The RRQR kernels (`tenferro-linalg/src/gpu/kernels/rank_revealing_qr.rs`) also need to launch
their own PTX, so the hook eventually has to expose a *module and launch* surface. That is the
one part that must not be improvised: the new path's `launch.rs` binds and launches the tenferro
kernel catalogue's entries, and an extension's own module needs the same
`Module`/`Function`/launch-config shape the CubeCL session has.

## Ownership rules the implementation must keep

1. **Session-scoped allocations.** An allocation made inside the hook (`alloc_bytes`,
   `alloc_output`) may only be freed after the work that reads it has completed. The new
   runtime's raw scope already retains registered resources until its completion; the hook
   registers every allocation it hands out, so the scope's completion — not the hook's return —
   is the last read.
2. **Retention.** `retain_tensor` exists because a vendor call outlives the caller's handle. On
   the new path the allocation lives in `gpu::Buffer<CudaBackend>`; a guard holding a clone of
   `t4a_cuda_runtime::Buffer` (its handle is clonable) is enough, and it must also be registered
   with the scope so a failure to synchronise cannot free it early.
3. **One stream.** Every vendor call in the hook runs on the engine's own stream inside the
   runtime's submission lock, exactly like `Cublas`/`Cutensor` already do. No new thread, no
   second queue, no host barrier on the success path: the hook ends with a completion the caller
   observes, not with a wait it performs for its own convenience.
4. **No second validation source.** The hook passes pointers and sizes; the extension crate keeps
   its own library-level validation (shape, workspace size, plan compatibility), as it does today.

## Sequencing

1. `cuda_next/raw.rs`: `Session<'s>`, `StreamRef`, `DeviceBytes`, `TensorRef`, `TensorMut`,
   `Retention`, `alloc_output`, `alloc_bytes`, `copy_bytes`, `resource`, `retain_tensor`,
   `upload_bytes`, `download_tensor`, `synchronize`, `len`, `runtime_identity`. Implement this
   against `CudaRuntime::with_raw_locked` and add a `seam_raw` test target that exercises each
   primitive on the A100 (allocation, copy, retention across a dropped handle, download, and a
   failed call leaving the engine usable).
2. Port `tenferro-fft/src/cuda` onto it. Its five primitives are the small half, and its three
   CUDA test targets (`canonicalization`, `plan`, `source_contract`) are the gate. `source_contract`
   asserts the FFI pointers stay inside the hook scope, so it also guards rule 3.
3. Port `tenferro-linalg/src/gpu` onto it: the cuSOLVER/cuBLAS call sites first, then the RRQR
   kernels once the module/launch part of the hook exists. Its `extension/cuda_tests.rs` and the
   `xgesvdp` tests are the gate.
4. Only then flip `tenferro_gpu::cuda` to the new path, add `ExecutionEngine` + the registration
   the runtime's prepared-program model needs (replacing the CubeCL preparer rather than sitting
   beside it), write the `TensorBackendCapability` table against the merged surface, and delete
   `cubecl/**`.

Steps 1–3 are the bulk of the remaining work and are pure moves: each is verified by the
extension crate's own CUDA tests, which already run on the A100.
