# FFT Backend Execution

`tenferro-fft` separates FFT semantics from backend execution through the
public `FftBackend` capability. This boundary applies to direct tensor calls,
caller-owned repeated execution, and registered graph execution. A backend
that implements only `TensorBackend` is not FFT-capable.

## Validated request boundary

The tensor-facing and runtime entry points validate the operation, dtype,
axis, transform length, input shape, and real-spectrum constraints before
calling a backend. They pass the result as an immutable `FftPlanSpec`.

`FftPlanSpec` is a semantic execution request, not a vendor plan. It records:

- the FFT operation and normalization mode;
- the normalized axis and optional requested transform length;
- the validated input dtype and shape; and
- the compact column-major input requirement of the current execution path.

Backend-specific choices such as RustFFT versus cuFFT, algorithm selection,
workspace size, stream identity, and device handles do not belong in the spec.
Those choices remain private to each `FftBackend` implementation.

## Placement and fallback

An FFT backend executes on the input's existing placement. If it cannot handle
the operation, dtype, layout, or placement, it returns `Unsupported`. It must
not construct another backend, upload or download the tensor, or silently run a
host implementation.

The CPU implementation uses RustFFT. It accepts ordinary host tensors and, when
the `CpuBackend` is paired by an Apple `AppleContext`, matching managed tensors
through guarded host mappings. The WebGPU implementation uses CubeK on the
configured Metal client. Callers select either backend explicitly; neither
implementation dispatches to the other. Cross-device movement remains a
caller-visible operation before or after FFT execution.

### CUDA/cuFFT execution

#### Scoped pointer safety supersedes initial async intent

An earlier design intent allowed cached cuFFT calls to remain asynchronous
between cache-eviction events. An independent safety review superseded that
intent for #967: scoped CubeCL raw pointers do not carry a completion witness,
so a vendor call must not outlive the raw session that borrowed its
allocations. The current baseline therefore enters the credentialed raw
session (`with_raw`) and synchronizes the bound CUDA stream inside that
session, after vendor enqueue and before the session (and its tensor spans,
work area, and retention guards) return. Future asynchronous execution
requires a separately accepted event-backed external-use ownership design;
that work is outside #967.

`tenferro-gpu` owns the CUDA provider, including `CudaBackend` and
`CudaRuntime`. The CUDA FFT adapter is owned by `tenferro-fft` and uses the
dynamically loaded cuFFT library. Non-empty `TENFERRO_CUFFT_PATH` entries are
tried first, followed by `libcufft.so.11`, `libcufft.so.10`, and bare
`libcufft.so`, covering CUDA 12/cuFFT 11 and CUDA 11/cuFFT 10. It executes the
existing one-dimensional `fft`, `ifft`, `rfft`, and `irfft` operations on
tensors already resident on the exact borrowed `CudaRuntime`. It never uploads,
downloads, constructs a CPU backend, or falls back to RustFFT. The common CUDA
execution path is:

1. Move the requested transform axis to the final logical axis with the
   existing same-device CUDA transpose. If the axis is already final, skip the
   transpose. The inverse permutation restores the original logical order
   after execution.
2. For C2C and R2C, truncate or zero-pad the final axis on device to the
   validated requested length `n`. Zero-padding allocates and fills a semantic
   same-device zero tail; it never derives zero from input arithmetic, so NaN
   and infinity inputs cannot contaminate the padding. C2R does not prepare its
   input this way:
   `n` is the real output length, and the already validated half-spectrum is
   consumed unchanged.
3. Execute one out-of-place rank-one `cufftMakePlanMany64` plan when the batch
   is nonzero. For canonical shape `[..., n]`, `batch` is the checked product of
   all leading extents. The exact column-major descriptor is:

   | Transform | `inembed` | `onembed` | `istride` / `ostride` | `idist` / `odist` |
   | --- | ---: | ---: | ---: | ---: |
   | C2C | `n` | `n` | `batch` | `1` |
   | R2C | `n` | `n / 2 + 1` | `batch` | `1` |
   | C2R | `n / 2 + 1` | `n` | `batch` | `1` |

   The rank-one `n` array and both rank-one embed arrays are always non-null;
   cuFFT ignores advanced stride and distance arguments when the corresponding
   embed pointer is null. This layout represents interleaved column-major
   lanes directly and avoids in-place real-transform padding and overwrite
   constraints.
4. Apply normalization on device because cuFFT is unnormalized. For
   `FftNorm::Backward`, forward uses `1` and inverse uses `1 / n`; for
   `Forward`, forward uses `1 / n` and inverse uses `1`; `Ortho` uses
   `1 / sqrt(n)` in both directions. A real-input full `fft` first receives
   the one-sided R2C result, then completes the Hermitian spectrum on device:
   for even `n`, append the reversed conjugate of bins `1..h - 1`; for odd
   `n`, append bins `1..h`, where `h = n / 2 + 1`. DC (and the Nyquist bin for
   even `n`) is not duplicated.

If a non-transform axis is zero, the checked batch is zero. The backend
allocates the correctly shaped, correctly typed resident CUDA output and
returns before cuFFT loading, descriptor construction, cache lookup, or plan
creation. This preserves empty CPU semantics without requiring a vendor
library.

CUDA plans use the existing `FftExecutionCache` rather than a backend-global
cache. Repeated concrete calls use the caller-owned `FftExecutor` cache;
eager and traced calls use the owning runtime's extension cache. The CUDA
namespace is `cufft-plans`. Its exact structural key contains the CUDA runtime
identity, device ordinal, transform dtype/kind, operation direction, `n`,
batch, `istride`, `idist`, `ostride`, and `odist`; embed extents are derived
from the kind and `n`. Each entry retains that exact key and checks full
equality before reuse, so a cache-discriminator hash collision cannot reuse a
plan for a different request. The cuFFT work area is **not** cached: it is a
session-scoped CubeCL allocation created fresh inside each raw execution
session, so logical retained bytes charge only the entry's host-side metadata
(key, plan handle, runtime and library witnesses, workspace-size requirement).
Opaque cuFFT internal allocations and library-owned state are excluded.

The cuFFT adapter creates plans under `CudaRuntime::with_current_context`, a
scoped guard that activates the tenferro primary context and restores the
caller's previous device/context on every exit path. Execution enters the
credentialed raw session (`with_raw`): the plan stream is bound to the captured
CubeCL stream, a fresh work area is allocated, the input/output allocation
handles are retained, the vendor call is enqueued, and the bound stream is
synchronized before the session returns. If synchronization fails, the work
area and the input/output retention guards are intentionally forgotten so
allocation reclamation cannot race in-flight vendor writes (issue #967
invariant). The current `CudaRuntime` exposes one serialized CubeCL current
stream, and the mutable CUDA FFT session serializes plan rebinding and
execution on that stream. The stream is therefore omitted from the cache key,
but `cufftSetStream` is called immediately before each execution. Subsequent
CUDA normalization, Hermitian completion, and axis restoration remain
stream-managed, and an explicit download synchronizes the final output. If
selectable concurrent streams are added, stream identity must become part of
the key before plan reuse is allowed. When an entry is evicted or a cache is
cleared, retirement runs the synchronize + `cufftDestroy` sequence under the
same context-restoring guard. Cleanup failures are reported without panicking.
If context selection, synchronization, or destruction fails, the complete
plan/library/runtime witness bundle is intentionally leaked rather than
dropped while queued work may still be active.

## Apple shared execution

`AppleContext` owns one host-visible Metal runtime, one allocation domain, and
paired CPU and Metal backends. An explicitly uploaded tensor retains one
physical allocation while the paired CPU backend maps it and the paired Metal
backend launches it. New operation results receive new allocations in the same
domain. Mapping, kernel launch, and managed result writeback do not change the
context's explicit upload/download counters.

The CPU RustFFT adapter supports `F32`, `F64`, `C32`, and `C64`. Its guarded
path preserves the ordinary axis, batching, normalization, and padding rules.
The initial CubeK Metal adapter is deliberately narrower:

| Operation | Input/output | Current Metal constraints |
| --- | --- | --- |
| CFFT/IFFT | `C32` to `C32` | power-of-two length at least 2; requested length must equal the input-axis length |
| one-sided RFFT | `F32` to `C32` | power-of-two requested or input length at least 2; padding/truncation supported |
| IRFFT | `C32` to `F32` | power-of-two requested or inferred real length at least 2; padding/truncation supported |

`F64`, `C64`, full-spectrum real FFT, non-power-of-two sizes, foreign domains,
and device-local WebGPU buffers return typed errors. They never trigger CPU
fallback or an implicit transfer.

`EagerTensorFftExt` registers the same FFT runtime against `EagerBackend`.
The adapter uses the runtime owner's selected CPU, CUDA, or WebGPU capability;
it does not inspect placement to choose a backend and does not download,
upload, or select a CPU backend on behalf of a GPU operation. Traced CUDA use
likewise requires explicit engine registration and
`extension_module::<CudaBackend>(engine_id)` installation.

## Cache ownership

Every backend receives `FftExecutionCache`, which exposes one bounded typed
`ExtensionCacheStore` regardless of ownership path:

| Execution path | Cache owner | Lifetime and control |
| --- | --- | --- |
| `FftExecutor` | Caller-owned `FftPlanCache` | Reused across direct calls; caller configures capacity and uses `cache_stats` / `clear_cache` |
| Registered extension runtime | Runtime-owned `ExtensionCacheStore` | Reused across graph runs; runtime limits, selectors, aggregate stats, and clear APIs apply |
| One-shot tensor/read helper | Temporary capacity-one `FftPlanCache` | Exists only for that call; it does not create hidden long-lived state |
| Host reference | Temporary capacity-one `FftPlanCache` | Explicit reference execution only; never selected as backend fallback |

`FftPlanCache` is a backend-neutral wrapper, despite its historical name. It
does not contain RustFFT-typed fields. Each backend stores private
`Send + Sync + 'static` entries through `FftExecutionCache::store_mut` under a
stable `ExtensionCacheKey` namespace. The global entry bound and LRU order are
shared across namespaces, and the global logical retained-byte bound applies
across those same entries. A caller's limits are true upper bounds for the
whole executor rather than per-backend multipliers.

Backends must include every field that changes plan or workspace identity in
the discriminator and should retain the unhashed identity in the typed value
when a hash is used. They must provide the logical bytes retained by the cache
entry at insertion, or use dynamic retained-byte accounting when the value can
grow. `FftExecutor::cache_stats` aggregates all backend namespaces, and
`clear_cache` removes them all.

## CPU RustFFT namespace

The CPU adapter owns the private `rustfft-plans` namespace. Its exact plan key
contains transform length, forward/inverse direction, and scalar dtype (`f32`
or `f64`). The typed entry retains the exact key to reject discriminator hash
collisions before reuse. The known retained-byte estimate covers the key and
cache-owned `Arc` handle; RustFFT's opaque internal allocations cannot be
reported and are intentionally excluded.

This namespace is an implementation detail. Non-CPU backends use distinct
names for vendor plans and workspaces and do not depend on RustFFT types.

## CubeK Metal compilation cache

The CubeK adapter does not insert a synthetic plan in `FftPlanCache`: CubeK
does not expose a reusable vendor-plan object. CubeCL's configured client owns
the compiled-kernel cache used by repeated launches. The caller-owned FFT cache
therefore remains empty for this backend, while its capacity and statistics
continue to describe entries actually owned through `FftExecutionCache`.

## Extension requirements

A new FFT backend implementation must:

1. implement `FftBackend` in the crate that owns the backend integration;
2. validate that the input placement and layout are supported without moving
   the tensor;
3. consume only validated request facts from `FftPlanSpec`;
4. use a backend-specific typed cache namespace for reusable plans/workspaces;
5. supply stable keys and retained-byte accounting; and
6. test repeated caller-owned reuse, runtime-owned reuse, eviction/clear/stats,
   unsupported placement, and absence of implicit transfer.

GPU-specific kernel, stream, and device-resource rules remain governed by
[GPU Backend Design](./gpu-backend-design.md).
