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

## I2 - resource ownership: the buffer carries the failure

**The rule.** An allocated resource is released by its owner; when the owner is
gone the memory returns to the driver, or the caller gets a typed `Err`. Never a
silent retention, and never an `Ok` for work that did not happen.

**Restated for the current pins.** CubeCL's allocator lives on the device thread,
so an allocation cannot report synchronously without a round trip.
`Client::do_empty` (`cubecl-runtime/src/client.rs`) queues the reservation, and the
CUDA server attaches a failure to the buffer (`fail_unallocated`,
`cubecl-cuda/src/compute/server.rs`). `Client::empty` therefore returns a `Handle`,
and the failure is reported at that buffer's next use: `check`, a read,
`sync_buffers`, or a download. Tenferro's `alloc_output` returns `Ok` meaning *the
request was accepted*, and every caller surfaces the failure through the tensor it
produced.

**Why the literal version is not affordable.** Measured on the A100, `--release`,
one test thread:

| operation | cost |
| --- | --- |
| `Client::check`, no handles (the bare device-thread round trip) | 6.94 us |
| `Client::check`, one handle | 8.19 us |
| `Client::flush()` | 6.83 us |
| `Client::empty(4096)` - enqueue only, today's `alloc_output` | 1.06 us |

#2112 item 7 targets a per-operation host cost within 1-2 us of a 5.36 us
launch-only floor (the eager `add` measures 9.11 us). One `check` per output
allocation costs 8.19 us, more than the entire current per-operation cost. A
fallible *upstream* entry point does not help either: the allocator state is on the
device thread, so any synchronous `Result` is the same round trip. **Do not put a
blocking check on the per-allocation path.**

**Residual, recorded rather than hidden.** A write-only output that nobody reads,
inside a session that nobody synchronizes, will not surface a refused allocation.
Closing that needs an opt-in strict path or one observation at an execution
boundary - never a per-allocation check, for the reason measured above.

**Evidence.** With a 200 GiB request, which the device refuses outright, on the
current pins:

| step | result |
| --- | --- |
| `client.empty(200 GiB)` then `check` | `Err: Several failures at once` - no panic |
| a later small allocation, then `check` | `Ok` - the device is not poisoned |
| `alloc_output(200 GiB)` | `Ok` - the deferred contract |
| `download` of that allocation | `Err: backend failure: Several failures at once` |
| an unrelated op on an existing tensor afterwards | `Ok`, values correct |

`cubecl_refused_allocation_is_reported_on_the_buffer`
(`crates/tenferro-gpu/src/cubecl/tests/cubecl_session_tests.rs`) pins all five
rows, so a future pin cannot go back to panicking on the device thread while the
caller is told the allocation succeeded.

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
| [#2081](https://github.com/tensor4all/tenferro-rs/issues/2081) a failed reserve panicked while `alloc_output` returned `Ok` | **fixed by the 0.11 pin move** | the forced-OOM table above |
| [#2020](https://github.com/tensor4all/tenferro-rs/issues/2020) `scale_real` / `scale_complex` fail on CUDA | **fixed by #2138** | the eager factor is uploaded with `constant_from_host`, the traced constant broadcasts through `broadcast_binary`; `test_gpu_scale_real_and_scale_complex_match_cpu` covers both surfaces and both dtypes with CPU parity |

## What this record does not claim

- The **retention** half of #2081 (device memory freed by dropped tensors not
  returning to the driver) is not fixed here. It is #2112 item 3 and keeps its own
  acceptance arm (`pool_retention_f64_*`).
- The no-toolkit-at-all panic variant of #2091 is unverified, as stated above.
- I2's residual - a never-read output in a never-synchronized session - is recorded
  above, not closed.
