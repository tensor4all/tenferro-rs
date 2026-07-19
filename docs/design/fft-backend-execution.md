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

The CPU implementation therefore accepts host tensors and uses RustFFT. A
future Metal or CUDA implementation must be registered and selected
explicitly. Cross-device movement remains a caller-visible operation before or
after FFT execution.

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
shared across namespaces, so a caller's capacity is a true upper bound for the
whole executor rather than a per-backend multiplier.

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
