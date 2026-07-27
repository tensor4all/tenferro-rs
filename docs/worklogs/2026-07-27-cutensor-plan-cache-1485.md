# 2026-07-27 cuTENSOR Plan Cache (#1485)

## Scope

CUDA `dot_general` now reuses cuTENSOR contraction descriptors, plans, and
device workspace through the `CudaBackend` extension cache. The cache is owned
by the backend, uses bounded defaults, reports logical retained bytes through
the existing CUDA cache stats, and is cleared by
`CudaBackend::clear_cuda_extension_cache`.

The key is structural: dtype, extents, strides, modes, conjugation flags,
descriptor alignment requirements, and workspace preference. It intentionally
does not include allocation addresses or actual pointer-specific alignment.
Whole tensors keep the CUDA allocation alignment requirement. Borrowed views
use a conservative dtype-size alignment requirement; using `1` was rejected by
cuTENSOR for f64 view descriptors on the local A100 smoke tests.

The inner cuTENSOR plan-entry cache has separate owner-routed introspection and
bound configuration through `CudaBackend::cutensor_plan_cache_stats`,
`CudaBackend::cutensor_plan_cache_max_entries`, and
`CudaBackend::set_cutensor_plan_cache_max_entries`. The outer CUDA extension
cache still owns the typed cache entry and aggregate retained-byte bound.

## Local A100 Benchmark

Environment:

- GPU: NVIDIA A100 80GB PCIe
- Driver: 580.126.09
- cuTENSOR: `/usr/lib/x86_64-linux-gnu/libcutensor/12/libcutensor.so.2`
- CUDA toolkit: `/usr/local/cuda`

Command:

```bash
CUDA_PATH=/usr/local/cuda \
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/libcutensor/12:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-} \
CUBECL_DEBUG_LOG=0 \
CARGO_BUILD_JOBS=64 \
cargo bench -p tenferro-gpu --features cuda --bench cutensor_plan_cache -- \
  --sample-size 10 --warm-up-time 0.1 --measurement-time 0.3
```

Results compare a cold path that clears `CudaBackend` extension caches before
each iteration against the warm backend-owned cache:

| Case | Cold clear each iter | Warm cache |
| --- | ---: | ---: |
| f32 64x64x64 | 100.84 us | 60.633 us |
| f64 64x64x64 | 108.77 us | 65.224 us |
| f32 256x256x256 | 101.99 us | 64.001 us |
| f64 256x256x256 | 98.891 us | 63.395 us |

This benchmark isolates repeated same-shape cuTENSOR setup overhead. It should
not be treated as an end-to-end workload share measurement.

## Verification

Focused checks:

```bash
CARGO_BUILD_JOBS=64 cargo test -p tenferro-gpu --test integration -- --nocapture
CARGO_BUILD_JOBS=64 cargo test -p tenferro-gpu --features cuda --test integration -- --nocapture
CUDA_PATH=/usr/local/cuda LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/libcutensor/12:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-} CUBECL_DEBUG_LOG=0 CARGO_BUILD_JOBS=64 cargo test -p tenferro-gpu --features cuda test_dot_general_matmul -- --ignored --nocapture
CUDA_PATH=/usr/local/cuda LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/libcutensor/12:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-} CUBECL_DEBUG_LOG=0 CARGO_BUILD_JOBS=64 cargo test -p tenferro-gpu --features cuda test_accum_ -- --ignored --nocapture
CARGO_BUILD_JOBS=64 cargo bench -p tenferro-gpu --features cuda --bench cutensor_plan_cache --no-run
```
