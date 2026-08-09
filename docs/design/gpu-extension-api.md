# GPU Kernel Extension API

This document is the public design contract for issue
[tensor4all/tenferro-rs#1597](https://github.com/tensor4all/tenferro-rs/issues/1597):
a stable extension boundary in `tenferro-gpu` so external operation crates can
launch custom GPU kernels through (1) direct CUDA APIs and (2) CubeCL kernels,
with capability-based WebGPU/ROCm handling and explicit multi-GPU.

This document freezes the design decisions that implementation must follow. It
was produced by a multi-model brainstorming session
(`.pi-meetings/2026-08-09-tenferro-rs_1597_実装詳細_検討/conclusion.md`) and
recorded here so future work continues to follow it.

## Scope

**In scope (Phase 1-4):**
- Promote the existing `#[doc(hidden)]` `CudaExecSession` + `with_cuda_exec_session`
  visitor to the single public execution-authority boundary.
- `cuda::raw` — typed stream/tensor/workspace borrowing, resource guard, and
  PTX/CUBIN/NVRTC module loading with unsafe launch.
- `cuda::cubecl` — promoted documented CubeCL client/binding/output surface.
- `webgpu::cubecl` — capability-based CubeCL session for the WebGPU backend.
- Explicit device identity (`CudaDeviceUuid`) and directional multi-GPU copy.
- Migration of `tenferro-linalg` onto the public seam and deletion of the hidden
  `cuda::interop` bridge.
- ROCm availability only on a real HIP substrate; otherwise typed `Unsupported`.

**Out of scope (see also NOT-SHIP):** universal cross-backend launch trait,
`u64`/bare `cudaStream_t` in public API, implicit CPU/GPU or GPU/GPU transfers,
implicit host staging, exposing internal CubeCL buffers or `CubeclBuffer`
fields, direct `wgpu::Device/Queue` first-version surface, NCCL/RCCL collective
integration, unbounded module caches.

## Grounding constraints (treated as satisfied)

- The storage/memory-layout contract is **frozen** (`docs/design/storage-contract-freeze.md`).
- CPU/CUDA/WebGPU layout parity is treated as satisfied; column-major owned
  tensors and frozen layout metadata are unchanged.
- `tenferro-gpu` remains the single owning crate for device identity,
  runtime/context/queue ownership, tensor placement, stream ordering,
  synchronization, capability reporting, and cross-backend validation.
- Errors reuse `tenferro_tensor::Result / Error::{Validation, RuntimeState,
  Unsupported, BackendSource}` with backend errors preserved in `source()`.

## Decisions

Adopted recommended defaults for the five contested items (2026-08-09). These
may be revised only by explicit maintainer decision; record any revision here
before implementation follows it.

| # | Topic | Decision |
|---|-------|----------|
| 1 | Stability label | provider-neutral identity/session/copy/error semantics stabilize as ordinary public API; `cuda::cubecl` and `webgpu::cubecl` bind the exact CubeCL package line and treat upgrades as a compatibility-matrix event, not version-agnostic |
| 2 | Raw view policy | `CudaTensorRef` exposes frozen layout metadata; first-version flat/vendor helpers accept compact tensors only and return `InvalidLayout`/`NonContiguous` otherwise (no hidden canonicalization); strided raw kernels pass strides explicitly under caller-`unsafe` ABI contract |
| 3 | Same-device cross-runtime copy | in #1597 Phase 3 as `copy_from_same_device`; normal copies keep strictly rejecting foreign domains |
| 4 | WebGPU/ROCm scope | WebGPU promises CubeCL-WGPU only; direct WGSL/wgpu device/queue needs a separate accepted use case. ROCm prefers CubeCL-HIP; ships typed `Unsupported` until real substrate + hardware validation |
| 5 | Fork exit gate | Phase 1-2 proceed on published `t4a-cubecl = =0.10.0`; before long-term API stabilization, complete the WGPU patch upstreaming and define the atomic switch version |

## Execution authority model

External extension is defined as **a scoped borrow of backend execution
authority**, not "a bag of raw handles".

- Only `CudaExecSession` can create CUDA extension sub-sessions
  (`with_cubecl`, `with_raw`).
- `CudaRuntime` does not publicly expose a current-context setter, native
  stream, CubeCL client, or allocation pointer.
- `raw::Session<'s>` **is** the typed context capability: it represents "the
  tenferro primary context is activated on the current thread and this
  execution stream is bound". There is no separate copyable `CudaContext` handle.
- `CudaExecSession`, both sub-sessions, and stream/context capabilities are
  non-constructible by users and are `!Send + !Sync` (thread-local capability).

### Public surface shape (names illustrative)

```rust
pub fn with_cuda_exec_session<B, R>(
    backend: &mut B,
    f: impl for<'s> FnOnce(&'s mut CudaExecSession<'s>) -> R,
) -> Option<R>
where B: BackendSession + ?Sized;

impl CudaExecSession<'_> {
    pub fn supports(&self, capability: GpuExtensionCapability) -> bool;
    pub fn device_info(&self) -> &CudaDeviceInfo;
    pub fn runtime_identity(&self) -> CudaRuntimeIdentity;
    pub fn allocation_domain(&self) -> AllocationDomainId;
    pub fn synchronize(&mut self) -> Result<()>;
    pub fn with_cubecl<R>(&mut self, op: &'static str, f: impl for<'s> FnOnce(&cuda::cubecl::Session<'s>) -> Result<R>) -> Result<R>;
    pub fn with_raw<R>(&mut self, op: &'static str, f: impl for<'s> FnOnce(&mut cuda::raw::Session<'s>) -> Result<R>) -> Result<R>;
}
```

## Identity layering

Non-interchangeable layers; each has fixed meaning and none substitutes another:

| Layer | Meaning | Example type |
|-------|---------|--------------|
| Device ordinal identity | process-visible selection, value semantics `Eq`, may change under `CUDA_VISIBLE_DEVICES` | `CudaDeviceId` |
| Stable physical/MIG identity | UUID, stable comparison, diagnostics, topology | `CudaDeviceUuid` |
| Exact runtime instance | per-executable-witness token | `CudaRuntimeIdentity` |
| Allocation ownership domain | allocator/provenance/retirement owner | `AllocationDomainId` |

`CudaDeviceEq` ordinal semantics must NOT be made UUID-dependent; `CudaDeviceUuid`
is a separate opaque token compared on its own.

## `cuda::raw` minimum surface

```rust
pub struct Session<'s> { /* opaque, !Send + !Sync */ }
pub struct StreamRef<'s> { /* opaque, non-Copy */ }
pub struct TensorRef<'s, T> { /* device span + frozen layout + placement + owner */ }
pub struct TensorMut<'s, T> { /* exclusive &mut or fresh output only */ }
pub struct DeviceBytes<'s> { /* CubeCL-owned workspace */ }
pub struct CudaResourceGuard<'s, T> { /* bounded per-runtime cache view */ }

pub struct Module { /* runtime/context owner */ }
pub struct Function { /* retains Module (Arc) */ }
pub struct LaunchConfig { pub grid: [u32; 3], pub block: [u32; 3], pub shared_mem_bytes: u32 }
pub struct KernelArg<'a> { /* scalar or checked device-address value */ }
```

- `StreamRef` is never `u64`; the native handle is extracted only through an
  explicit unsafe FFI escape with `# Safety` and must not be retained past the
  session scope.
- `TensorRef`/`TensorMut` carry validated span + frozen layout + ownership
  identity; no `Deref` to the device buffer. `TensorMut` cannot be produced
  safely from `&TypedTensor<T>`.
- `CudaResourceGuard::resource(init)` is the narrow safe view over the existing
  bounded, TypeId-keyed, per-runtime `CudaExtensionCache`.
- Module/function/workspace/tensor-root leases enter the current
  submission/event-domain retirement; never release before device completion.
  Per-launch device-wide sync is NOT an acceptable lifetime implementation.
- Any raw kernel launch is a small `unsafe fn` with `# Safety` documenting ABI,
  argument order, ranges, aliasing, initialization, and async-liveness.

### `with_raw` enter/exit protocol

1. Capture a definite CubeCL `StreamId` on the current thread; resolve its raw
   stream in the same thread.
2. Flush pending CubeCL work on that stream.
3. Save the calling thread's previous CUDA runtime device and driver current
   context; activate the tenferro-held primary context.
4. Construct the unique `raw::Session`.
5. Enqueue all raw work on the one captured stream.
6. Best-effort restore of the previous device/context on normal return,
   `Err`, and unwind (RAII guard); a restoration failure is logged to stderr
   and not returned. The success path does NOT synchronize.
7. If the implementation cannot prove both CubeCL and raw segments use the same
   captured stream, it is **NO-GO** — no extra device-wide sync to mask it.

## `cuda::cubecl` minimum surface

```rust
pub struct Session<'s> { /* exact tenferro CubeCL client */ }

impl Session<'_> {
    pub fn client(&self) -> &ComputeClient<CubeclCudaRuntime>;
    pub fn tensor_binding<T>(&self, tensor: &TypedTensor<T>) -> Result<TensorBinding<_>>;
    pub fn array_arg<T>(&self, tensor: &TypedTensor<T>) -> Result<ArrayArg<_>>;
    pub fn alloc_output<T>(&self, shape: impl IntoShapeVec) -> Result<TypedTensor<T>>;
    pub fn cube_count_1d(&self, len: usize) -> Result<CubeCount>;
    pub fn cube_dim_1d(&self) -> CubeDim;
}
```

`with_cubecl` flushes on exit (RAII). Only the narrow kernel-writing prelude is
re-exported — **no `pub use cubecl::*`**. Downstream depends on the crates.io
package line `cubecl = { package = "t4a-cubecl", version = "=0.10.0" }`.

## Capability and error contracts

```rust
#[non_exhaustive]
pub enum GpuExtensionCapability {
    CubeClKernel,
    NativeModule,
    RuntimeCompilation,
    RawStream,
    SameDeviceAsyncCopy,
    PeerCopy,
}
```

- `GpuExtensionCapability` is orthogonal to the existing primitive
  `OperationCapability`; reuse the typed/non-exhaustive/queriable pattern but do
  not merge the two enums.
- `supports()` means the capability is plausibly available; it does not
  guarantee compile/load/enqueue success.
- Provider limits (CC, max block/shared memory, WebGPU limits) stay in provider
  info structs; no mega `Option`-filled capability struct.
- Errors: typed non-exhaustive source errors under `cuda`/`webgpu` namespaces
  distinguishing unavailable, unsupported, foreign runtime/device/domain, invalid
  layout/config, compile, module load, symbol lookup, enqueue/launch, synchronize,
  and peer access. Top-level stays `tenferro_tensor::Result`.

## Multi-GPU copy semantics

- No auto-routing `copy_to_device`. Two explicit directional entry points:
  - `copy_from_same_device` — same UUID, different runtime/allocation domain;
  - `copy_from_peer` — different UUID, directional P2P available.
- Both take explicit source+destination sessions; the result allocation domain
  is always the destination.
- Ordering: source event → destination stream wait → destination enqueue.
- P2P capability is directional; re-query after swapping source/destination.
- Unsupported P2P → typed error, never auto host staging (future staging API
  must be named `*_via_host`).
- Collectives (NCCL/RCCL) are fully separate from ordinary launch/copy.

## CubeCL dependency policy

- External kernel crates depend on crates.io `t4a-cubecl = =0.10.0`; never a git
  rev. The workspace `[patch.crates-io]`/git pin is a development detail only.
- A library-level `[patch]` is rejected (cannot force downstream) and a
  type-erased CubeCL client trait is rejected (cannot solve proc-macro /
  generic-runtime crate identity).
- Source-blind downstream fixtures (workspace + `cargo package`) compile a real
  `#[cube(launch)]` kernel and prove a single `t4a-cubecl` runtime type
  (`cargo tree -d`).
- Upstreaming: the fork's WGPU exact-submission patch has not yet been merged
  upstream. Publishing `t4a-cubecl` now + parallel upstreaming is the accepted
  short-term path; switch the whole family back to official CubeCL atomically.

## Phased delivery and Definition of Done

### Phase 1 — Public seam + linalg dogfood
Public `CudaExecSession` (+ `with_raw`/`with_cubecl`), typed stream/tensor/
workspace, context-restoration guard, resource guard, `CudaDeviceUuid` +
`GpuExtensionCapability`, `t4a-cubecl = =0.10.0` policy; migrate
`tenferro-linalg` (linalg/gemm/permutation/event_domain) exclusively to the
public API; delete `cuda::interop`.
**DoD:** no `set_current_cuda_context`, `with_raw_cuda_stream`, callback-escaped
`*mut c_void`/stream, hand-written `cudaStreamSynchronize`, or
`flush_cubecl_client` in the workspace; context restores under normal/error/
panic/nested-dual-runtime/dual-thread; per-runtime cache key semantics intact;
existing numerical/zero-size/batch tests pass; no new host round-trips or
per-op device-wide sync; workspace+package fixtures resolve a unique CubeCL type.
**Go/No-Go:** NO-GO if linalg still needs a hidden runtime entry, safe
stream/address escape, per-op forced sync, or the session cannot stably capture
one stream.

### Phase 2 — External kernel vertical slices
CUDA CubeCL, direct CUDA (PTX/CUBIN/NVRTC), WebGPU CubeCL.
**DoD:** three downstream-style fixtures run upload → launch → explicit
sync/download → numerical assert on public API only; PTX+NVRTC run on CUDA
hardware CI, CUBIN on matching arch; missing symbol / CC & launch-limit
mismatch / foreign-runtime tensor / non-compact flat helper → typed error;
module/workspace/tensor leases outlive async completion without post-launch
global sync; feature/source-only CI compiles fixtures on GPU-less machines.
**Go/No-Go:** NO-GO on duplicate CubeCL type identity, module retirement not
event-domain-integrated, or safe mutable address derivation from shared tensor.

### Phase 3 — Explicit multi-GPU
Full device attributes; directional peer capability; same-UUID cross-runtime
copy; different-UUID P2P copy; event ordering.
**DoD:** four paths (same-runtime, same-UUID dual-runtime, different-UUID P2P,
P2P-unsupported) covered; swapped source/destination re-queries capability;
result always destination domain; transfer trace proves no implicit D2H/H2D;
dual-GPU hardware CI verifies concurrent streams + context restore + correct
writes + unsupported paths.
**Go/No-Go:** NO-GO without a real dual-GPU run, simulated success by weakening
identity checks, or automatic host fallback.

### Phase 4 — ROCm + stability closeout
`rocm::cubecl` on a real CubeCL-HIP substrate, or capability-only `Unsupported`
with the ROCm implementation split into an accepted follow-up issue.
**DoD:** feature combos `default`/`cuda`/`webgpu`/`cuda+webgpu`/`rocm` pass,
non-CUDA combos do not load CUDA libraries; real ROCm delivery needs hardware
run evidence; public items have runnable doctests; old bridge/aliases removed;
independent API/unsafe/lifetime/retirement/multi-GPU audit complete.

## NOT-SHIP

1. Public `u64`/bare `cudaStream_t` stream, or a public current-context setter.
2. A second execution entry bypassing `CudaExecSession` from `CudaRuntime`/
   tensor/allocation.
3. Claiming a safe callback prevents raw pointer/stream address escape while
   allowing copy.
4. Safe mutable pointer from a shared tensor, or silent materialization/miscompute
   on non-compact views.
5. Merging allocation domain by UUID/device, or per-device linalg handle cache.
6. `pub use cubecl::*`, downstream git revs, or mixing upstream/fork CubeCL
   runtime families.
7. Per-launch device-wide sync solving ordering/module lifetime.
8. Auto-routing copy or P2P symmetry assumptions.
9. WebGPU fake CUDA module/pointer API; ROCm runtime/quickstart without real
   substrate.
10. NCCL/RCCL collectives conflated with launch/copy.
11. Unbounded global module/kernel cache.
12. Retaining the hidden bridge as a compat layer after linalg migration.

## Relationship to other documents

- [gpu-backend-design.md](./gpu-backend-design.md) — backend provider model,
  runtime ownership, and cache ownership.
- `docs/design/storage-contract-freeze.md` — the frozen layout/storage contract.
- `crates/tenferro-gpu/src/cubecl/` — implementation.
- Plan: `docs/superpowers/plans/2026-08-09-gpu-extension-api.md`.
