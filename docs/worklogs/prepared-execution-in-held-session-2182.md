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
  2026-10-17, shared host, load average 3-8 during the run; dense pseudo-random square input, thin SVD with vectors; median time /
  relative reconstruction error):

  | dtype | n | Auto (= gesvdj) | gesvd | xgesvdp |
  |---|---:|---:|---:|---:|
  | f32 | 64 | 1.70 ms / 9.0e-06 | 4.37 ms / 1.3e-06 | 2.74 ms / 9.8e-07 |
  | f32 | 256 | 9.85 ms / 6.4e-05 | 20.3 ms / 2.4e-06 | 6.57 ms / 1.3e-06 |
  | f32 | 1024 | 77.8 ms / 3.1e-04 | 133 ms / 5.5e-06 | 35.7 ms / 2.0e-06 |
  | f64 | 64 | 2.39 ms / 2.1e-14 | 4.83 ms / 2.6e-15 | 3.00 ms / 1.8e-15 |
  | f64 | 256 | 16.1 ms / 7.6e-14 | 22.7 ms / 5.5e-15 | 7.84 ms / 3.2e-15 |
  | f64 | 1024 | 144 ms / 4.6e-13 | 162 ms / 1.0e-14 | 42.7 ms / 5.1e-15 |
  | c32 | 64 | 2.60 ms / 2.1e-05 | 4.70 ms / 1.3e-06 | 3.05 ms / 1.1e-06 |
  | c32 | 256 | 18.8 ms / 1.4e-04 | 22.3 ms / 2.5e-06 | 7.71 ms / 1.8e-06 |
  | c32 | 1024 | 170 ms / 5.5e-04 | 151 ms / 5.0e-06 | 46.7 ms / 2.7e-06 |
  | c64 | 64 | 4.96 ms / 3.8e-14 | 5.43 ms / 3.2e-15 | 3.42 ms / 2.2e-15 |
  | c64 | 256 | 39.3 ms / 3.2e-13 | 27.6 ms / 5.3e-15 | 9.92 ms / 4.1e-15 |
  | c64 | 1024 | 400 ms / 1.9e-12 | 220 ms / 9.8e-15 | 61.8 ms / 6.2e-15 |

  On this input family xgesvdp is fastest from n=256 up in every dtype and at n=64 for c64, gesvdj is fastest at n=64 for
  f32/f64/c32, and xgesvdp has the smallest reconstruction error in every row. Shared-host single runs: ratios below about 1.2x are
  not conclusions. Square inputs only; tall/wide and batched shapes are not measured. A first run used a low-rank, clustered input
  and was discarded because it inflated gesvdj times.
- `run_prepared_in` measures ~23 µs vs ~25 µs for `run_prepared` on the small fused route-matrix graph: the per-call barrier,
  not session entry, dominates small CUDA prepared calls, which is what stage 2 would address.
- The dispatch fence that fixes the early completion event costs about 5 µs per scheduled CUDA node (8-node matmul chain
  163 -> 204 µs per call). Recording one event per single-stream run instead of one per node would remove it, but changes the
  event-domain model and belongs with the stage-2 decision.
- Stage 2a (single-domain scheduler coalescing, #2182) removes that per-node cost: an eligible single-stream prepared run records
  zero per-node scheduler events and one scheduler dispatch fence per call, with the same retirement barrier after output
  collection. A100, 2026-10-17, shared host, short Criterion rows (`--warm-up-time 1 --measurement-time 2 prepared_`), before ->
  after: `prepared_fused` 32.7/30.6 -> 24.6/27.4 µs and `prepared_matmul_chain` 288/382 -> 162/156 µs (standalone/in_held). A
  multi-domain or transfer schedule keeps the per-node path.
