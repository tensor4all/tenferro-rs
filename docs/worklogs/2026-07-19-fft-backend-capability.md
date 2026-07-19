# FFT backend capability and cache-boundary work log

Date: 2026-07-19

## Session summary

This work introduced an explicit `FftBackend` capability for concrete and
registered FFT execution. It moved CPU RustFFT execution behind that capability,
centralized validated requests in `FftPlanSpec`, retained exact bounded CPU
plan caching, and made unsupported placement return an error without transfer
or fallback.

A follow-up review found that the initial caller-owned `FftExecutionCache`
accepted a RustFFT-specific `FftPlanCache`, while only runtime-owned execution
exposed typed storage. The boundary was corrected so `FftPlanCache` wraps the
same backend-neutral `ExtensionCacheStore` used by graph execution. A recording
non-CPU backend now proves direct plan reuse, aggregate retained-byte stats, and
clear behavior.

Durable design intent is recorded in
[FFT Backend Execution](../design/fft-backend-execution.md).

## Context read

| Source | Why it was read | Decision impact |
| --- | --- | --- |
| `AGENTS.md` and `REPOSITORY_RULES.md` | Confirm repository, cache, documentation, and AI worklog requirements. | Kept cache ownership explicit and added both design and worklog records. |
| Task brief, implementation plan, and review report | Establish capability requirements and the two requested review corrections. | Preserved CPU behavior while widening only the cache storage boundary. |
| `tenferro-runtime::ExtensionCacheStore` | Inspect the existing bounded typed store, limits, selectors, LRU, and retained-byte APIs. | Reused the runtime abstraction rather than creating a second type-erased cache. |
| `tenferro-tensor::{CacheStats, RuntimeCacheControl}` | Confirm aggregate clear/stats contracts. | Kept `FftPlanCache` compatible with shared cache control. |
| Existing FFT concrete/runtime tests and CPU cache implementation | Verify exact key fields, reuse, bounds, dtype coverage, and placement behavior. | Retained RustFFT length/direction/dtype identity and collision checks. |
| Extension and cache design/worklog precedent | Match durable record scope and structure. | Separated future-facing design rules from session-level rationale. |

CodeGraph was used before direct source search to locate the FFT executor,
runtime cache API, callers, and cache-control blast radius.

## Decisions made

- **`FftBackend` is the compile-time capability boundary.** Generic
  `TensorBackend` implementations cannot enter FFT execution or register the
  FFT runtime without opting into the capability.
- **`FftPlanSpec` contains validated semantics only.** Vendor plans, handles,
  workspaces, streams, and device choices remain backend-private.
- **Caller- and runtime-owned execution expose the same typed store.**
  `FftExecutionCache::store_mut` works in both paths. Constructor-controlled
  ownership still prevents a backend from replacing or extending the cache
  lifetime.
- **One capacity bounds the full caller-owned cache.** CPU and future GPU
  namespaces share the wrapper's LRU entry limit, clear operation, and
  aggregate stats instead of each receiving an independent hidden bound.
- **RustFFT remains private.** The CPU adapter uses `rustfft-plans` with exact
  length, direction, and scalar-dtype keys. The stored exact key protects
  against discriminator hash collisions.
- **Placement remains explicit.** Unsupported backends return `Unsupported`;
  FFT execution never performs upload, download, or CPU fallback.

## Rejected or deferred alternatives

- Keeping a RustFFT LRU beside a separate generic caller store was rejected:
  two independent bounds would make capacity and aggregate eviction behavior
  misleading, and CPU entries would follow a different storage contract from
  other backends.
- Exposing RustFFT plan enums or a backend-specific cache trait was rejected:
  it would couple future Metal/CUDA integrations to the CPU provider or require
  a second type-erasure mechanism already provided by `ExtensionCacheStore`.
- GPU FFT implementation is deferred. This change supplies the capability and
  storage contract but does not add Metal, CUDA, cuFFT, streams, or device
  workspaces.

## Verification performed

The implementation was developed test-first. The new non-CPU caller-cache test
initially failed because `FftExecutionCache::store_mut` did not exist, then
passed after the shared typed-store boundary was implemented.

- `cargo fmt --all -- --check`
- `cargo test -p tenferro-fft`
- `cargo test -p tenferro-fft --features autodiff`
- `cargo clippy -p tenferro-fft --all-targets --features autodiff -- -D warnings`
- `cargo doc --workspace --no-deps`
- `python3 scripts/check-public-error-docs.py --root-dir . --changed-from origin/main`
- `python3.11 scripts/check-docs-site.py`
- `git diff --check`

All commands passed. Python 3.11 was selected explicitly because the host's
default Python 3.9 cannot parse guide dependency snippets.

## Remaining risks and follow-up

- RustFFT does not expose the memory owned by opaque plans, so retained-byte
  stats cover only the exact key and cache-owned handle.
- The workspace's mutually exclusive BLAS-provider features prevent a useful
  `--all-features` clippy invocation. The supported FFT feature matrix is used
  instead; this conflict is unrelated to the FFT cache boundary.
- Each future GPU backend must define and test its own exact key identity and
  logical retained-byte calculation.
