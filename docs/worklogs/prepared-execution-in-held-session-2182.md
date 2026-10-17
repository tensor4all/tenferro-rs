# Prepared execution in a caller-borrowed session, caller-buffer readback, SVD driver evidence (#2182)

## Decisions
- `Runtime::run_prepared_in` runs a prepared single-engine graph on a session the caller already entered (scoped or
  `CudaHeldSession::with_session`). Every node, fusion and output collection uses that session; the executor state is taken once
  per call without waiting. Retirement still happens before the call returns (stage 1). Removing the per-call stream barrier
  needs an ownership protocol for the run's inputs and outputs and is left as an open maintainer decision (stage 2), see
  [held-cuda-session-1945-u3.md](../design/held-cuda-session-1945-u3.md) §8.
- Session ownership is checked by the backend (`BackendSessionHost::owns_session`, default `false`) against its exact
  runtime identity, not by backend type: two CUDA backends on one device share a type but not allocations.
- Readback into caller memory takes an owned, reusable `Vec<T>` (`download_tensor_into`), not `&mut [T]`: after an unproven copy
  failure the device may still write the destination, so it is retained instead of returned, as `download_pending` does.
- `SvdDriver::Auto` is unchanged (JAX-compatible since #971). The comparison below is evidence for a maintainer decision.

## Verification conclusions and constraints
- Two existing CUDA defects surfaced and are fixed: the event domain recorded its completion event before queued CubeCL launches
  were dispatched to the stream, and an identity graph could not collect a borrowed device input. Both have tests that fail
  without the fix.
- `download_tensor_into` with a reused buffer: 256 MiB 18.5 ms vs 176.7 ms for `download_tensor` (fresh `Vec`), A100; the copy
  itself is unchanged at 8 MiB. The gap is the per-call host allocation and first touch.
- CUDA SVD driver comparison (`cargo bench -p tenferro-linalg --features cuda --bench cuda_svd_driver`, A100 80GB PCIe,
  2026-10-17, shared host, load average 2-14 during the run; thin SVD with vectors; median time / relative reconstruction error):

  | dtype | n | Auto (= gesvdj) | gesvd | xgesvdp |
  |---|---:|---:|---:|---:|
  | f32 | 64 | 2.37 ms / 8.7e-06 | 3.85 ms / 6.0e-07 | 2.75 ms / 1.1e-06 |
  | f32 | 256 | 17.3 ms / 6.7e-05 | 15.0 ms / 9.7e-07 | 6.34 ms / 1.4e-06 |
  | f32 | 1024 | 124.5 ms / 1.9e-04 | 75.0 ms / 1.0e-06 | 34.5 ms / 4.1e-06 |
  | f64 | 64 | 5.61 ms / 2.6e-14 | 4.06 ms / 3.6e-15 | 3.09 ms / 2.0e-15 |
  | f64 | 256 | 49.2 ms / 2.6e-13 | 14.0 ms / 3.1e-15 | 7.02 ms / 2.3e-15 |
  | f64 | 1024 | 416.0 ms / 8.0e-13 | 65.5 ms / 4.3e-15 | 41.1 ms / 6.8e-15 |
  | c32 | 64 | 3.33 ms / 2.0e-05 | 4.85 ms / 9.4e-07 | 3.13 ms / 1.1e-06 |
  | c32 | 256 | 34.3 ms / 1.2e-04 | 18.7 ms / 1.0e-06 | 7.57 ms / 1.3e-06 |
  | c32 | 1024 | 288.9 ms / 2.6e-04 | 116.6 ms / 2.5e-06 | 45.5 ms / 3.0e-06 |
  | c64 | 64 | 8.99 ms / 5.4e-14 | 5.43 ms / 2.9e-15 | 3.53 ms / 2.0e-15 |
  | c64 | 256 | 122.2 ms / 4.9e-13 | 17.4 ms / 3.3e-15 | 9.04 ms / 2.9e-15 |
  | c64 | 1024 | 1127 ms / 1.4e-12 | 79.2 ms / 5.3e-15 | 60.3 ms / 4.9e-15 |

  On this machine xgesvdp is fastest in every row except f32 n=64 and is never less accurate than gesvdj; these are shared-host
  single runs, so ratios below about 1.2x are not conclusions. Square inputs only; tall/wide and batched shapes are not measured.
- `run_prepared_in` measures ~23 µs vs ~25 µs for `run_prepared` on the small fused route-matrix graph: the per-call barrier,
  not session entry, dominates small CUDA prepared calls, which is what stage 2 would address.
