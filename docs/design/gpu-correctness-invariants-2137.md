# GPU correctness invariants recorded by #2137

The umbrella [#2137](https://github.com/tensor4all/tenferro-rs/issues/2137) tracks
the CUDA defects whose symptom is a wrong answer, a refused compile, or a caller
told everything succeeded. It names two invariants - I2 restated for the current
pins, and I5 - and requires them to be recorded in `docs/design/` rather than left
in code comments. This is that record. The measurements below are from `main` at
the 0.11 fork pins (`t4a-cubecl 0.11.0` / `t4a-cubek 0.3.0`, A100 80GB, CUDA 12.6,
driver 580.126.09).

`docs/design/gpu-performance-invariants-2112.md` is the companion record for the
performance invariants (I1, I3, I4).

## I2 - resource ownership: a refused allocation is a synchronous typed error

**The rule.** An allocated resource is released by its owner; when the owner is
gone the memory returns to the driver, or the caller gets a typed `Err`. Never a
silent retention, and never an `Ok` for work that did not happen.

**Current contract.** `CudaEngine::alloc_output` calls `alloc_tensor`, which submits
`alloc_zeroed` on the engine's runtime. The runtime asks its memory pool with
`cuMemAllocFromPoolAsync`; a driver out-of-memory result is translated to
`t4a_cuda_runtime::Error::OutOfMemory { bytes }` before any `Buffer` or tensor is
built. Tenferro wraps that source in `Error::RuntimeStateSource`, so the caller
receives `Err` with the typed cause still reachable through the source chain.
There is no failed-buffer handle to download later, and no later operation has to
observe a deferred allocation failure.

**Evidence.** `cuda_refused_allocation::a_refused_device_allocation_returns_out_of_memory_without_a_buffer`
requests 200 GiB (214748364800 bytes) through `CudaEngine::alloc_output`, asserts
the source chain downcasts to `OutOfMemory { bytes: 214748364800 }`, and asserts
the runtime's live allocation count and bytes are unchanged. It then pins
recovery: a small zero-filled allocation, an upload/download round trip, and an
empty download all succeed with the expected values.
`cuda_refused_allocation_observer::a_refused_device_allocation_requests_no_huge_host_reservation`
observes every Rust global-allocator request (not native CUDA-library
allocations) while the same request is refused and asserts the largest request
stays below 1 GiB, so the refusal cannot have reserved a 200 GiB host vector.
Both tests are `#[ignore]`-gated and classified in the CUDA GPU lane.

A valid device-to-host download still reserves its host slot inside
`t4a_cuda_runtime` (`DownloadSlot::new`); that reservation's fallibility is owned
by the runtime dependency and is not covered by these tests.

## I5 - a device-side failure is observable at the boundary that caused it

**The rule.** A failure on CubeCL's device thread must reach the tenferro call
that caused it, either as an `Err` or as a claim on the buffer that call produced.
A panic that only becomes a `log::warn`, or a buffer handed back as valid after the
work that was to fill it never ran, violates I5.

**Evidence: the compile-failure path.** With `CUDA_PATH` pointing at a header-less
directory, a complex elementwise op fails to compile. The op returns `Ok` (its
launch is queued) and the *read* reports the chain:

```text
The bytes were never written (failure #2, still claiming 1 buffer(s))
  -> A launch error happened
  -> A compilation error happened during launch
  -> [Compilation Error] catastrophic error: cannot open source file "cuComplex.h"
```

**Not verified, and therefore not claimed.** The no-toolkit-at-all variant, where
`cubecl_cuda::install::include_path()` panics (`cuda_path().expect(...)`), needs a
host with no toolkit, no `CUDA_PATH`, and none of `/usr/local/cuda`, `/opt/cuda` or
`/usr/bin/nvcc`. No such host was available for this record.

## Item status on the current pins

| item | status | evidence |
| --- | --- | --- |
| [#2133](https://github.com/tensor4all/tenferro-rs/issues/2133) complex full `reduce_sum` refused by NVRTC | **fixed by the 0.11 pin move** | `tensor4all/cubecl` `d3de21b2d feat(plane): lower complex plane reductions for CUDA` is an ancestor of the 0.11 release branch and not of the previous pin. `test_cubecl_complex_full_reduction_compiles_and_matches_cpu` pins it: c32/c64 at len 1, 1e3 and 1e6, rank-0 result, CPU parity at 1e-4, and `reduce_sum(sub(x, x))` exactly zero |
| [#2091](https://github.com/tensor4all/tenferro-rs/issues/2091) `with_cubecl` returned `Ok` and the output kept its old bytes | **the silent-return half is fixed** | the compile-failure evidence above |
| [#2081](https://github.com/tensor4all/tenferro-rs/issues/2081) a failed reserve panicked while `alloc_output` returned `Ok` | **the synchronous half is fixed** | `cuda_refused_allocation::a_refused_device_allocation_returns_out_of_memory_without_a_buffer` pins the typed `Err` and the absence of a buffer; the retention half is #2112 item 3 |
| [#2020](https://github.com/tensor4all/tenferro-rs/issues/2020) `scale_real` / `scale_complex` fail on CUDA | **fixed by #2138** | the eager factor is uploaded with `constant_from_host`, the traced constant broadcasts through `broadcast_binary`; `test_gpu_scale_real_and_scale_complex_match_cpu` covers both surfaces and both dtypes with CPU parity |

## What this record does not claim

- The **retention** half of #2081 (device memory freed by dropped tensors not
  returning to the driver) is not fixed here. It is #2112 item 3 and keeps its own
  acceptance arm (`pool_retention_f64_*`).
- The no-toolkit-at-all panic variant of #2091 is unverified, as stated above.
- I2 covers a refused device allocation only. A valid download's host
  reservation is owned by `t4a_cuda_runtime` (`DownloadSlot::new`) and its
  fallibility is not covered here.
