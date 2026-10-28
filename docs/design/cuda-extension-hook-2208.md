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

## What the hook cannot do with the current upstream API

`RawScope::register` is in (t4a-gpu-backend#9, pinned at `1b23d9a`), so a hook can address a buffer it
only learns about while the scope is open — which is what "the extension picks its operand at
execution time" needed.

Allocating inside the scope is still impossible: `CudaRuntime::with_raw_locked` holds the runtime's
submission mutex for the whole closure, so any `&self` helper that takes the guard again
(`alloc_zeroed`, `upload_bytes`, `download_bytes`, `alloc_pinned`) would deadlock, and `RawScope`
exposes no allocator of its own. Both extension crates allocate inside the hook — cuFFT allocates
its workspace per execution (`raw.alloc_bytes`) and cuSOLVER allocates every output tensor
(`raw.alloc_output`) — so the hook needs one more upstream addition: `RawScope::alloc_zeroed(bytes,
align)` (and, for the extensions' uploads, `RawScope::upload(bytes)`), which allocates through the
domain the scope already owns.

Decided (route A) and landed as t4a-gpu-backend#10 (`a3e242d`, pinned): `RawScope::alloc_zeroed`
allocates through the domain the scope already owns and zero-fills on the scope's stream; the
result becomes addressable by registering it. Transfers stay outside the scope — their admission is
the runtime's own and re-entering it from a scope deadlocks, which the first version of that change
proved by hanging — so the hook's `upload`/`download` entries are built on the pre-opened runtime
rather than inside the scope.

The hook therefore has `alloc_zeroed` for scratch, `register` for any buffer it learns about
mid-scope, and the stream.

### Retention: the runtime's stream-ordered free is the guard

`retain_tensor` — the extension's way to say "this vendor call outlives my handle, keep the
allocation alive until it completes" — does not need a shareable buffer on this path, for a reason
that is stronger than "the hook waits": a dropped `Buffer` releases its device pointer with
`cuMemFreeAsync` **on the same stream** (`memory.rs::Domain::release`, whose safety comment reads
"the free is ordered after every admitted use on this stream"). A vendor call the hook enqueued on
that stream therefore completes before the allocation is reclaimed, even if the caller drops its
tensor as soon as `with_raw` returns. The CubeCL path needed a retained clone because its handles
were reference-counted and reclaimed eagerly; this runtime does not.

So `retain_tensor` becomes a membership check (the tensor is one of this runtime's allocations), and
the hook's contract states that every vendor call is enqueued on the session's stream. The one
behaviour this relies on is `Domain::release` staying stream-ordered; it is worth a test on the
hook's own path rather than an assumption (drop a buffer right after enqueuing a copy that reads it,
then read the copy back).

The measurements that make this the right call: cuFFT's hook already synchronizes inside the closure
(two sites in `tenferro-fft/src/cuda/plan.rs`), while cuSOLVER has 28 `with_raw` sites and only three
synchronise or wait inside — so a "hook waits before returning" rule would have had to be added to
25 call sites, and it is not what keeps the data safe anyway.

### What part 2 (the cuSOLVER port) still needs from the session

Part 1 shipped the session with `stream`, `alloc_bytes`, `tensor`/`tensor_mut`, `retain_tensor` and
`synchronize` — everything cuFFT uses. The cuSOLVER providers use five more entries, and each has to
respect the same rule (no re-entry into the runtime's admission):

| entry | how it has to work inside the scope |
|---|---|
| `copy_bytes(dst, src, bytes)` | `cuMemcpyDtoDAsync` on the session's stream; both pointers are registered spans. |
| `upload_bytes(bytes, op)` | device allocation from the scope plus `cuMemcpyHtoDAsync` from a **pinned** host buffer that the scope retains; `RawScope::upload` cannot be used because the runtime's transfer takes the admission the scope holds (t4a-gpu-backend#10's first version hung on exactly that). |
| `download_tensor(t)` | the mirror image: `cuMemcpyDtoHAsync` into a pinned buffer the scope retains, then a stream barrier before the host bytes are read. |
| `resource(init)` | a per-engine typed cache for vendor handles. The engine already owns the runtime's `ExtensionCacheStore`; the session needs a borrowed handle to it, which is why this one lands with the cuSOLVER port rather than before it. |
| `len()` | not a session entry: the single `raw.len()` call site is a CPU test. |

The 14 `#[cube]` kernels are the other half of the port; the catalogue's matching entries are listed
above, and each kernel swap is verified against the existing numeric tests.

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
