# cuBLASLt batched GEMM for small standard batched matmul

## Decisions
- Small standard batched matmuls (`batch_count >= 2`, `m`, `n`, `k <= 64`) use a
  cuBLASLt plan. The plan asks the cuBLASLt heuristic for one batched algorithm,
  so the kernel that serves the whole product is chosen by the plan instead of
  being left to whatever kernel cuBLAS `gemmStridedBatchedEx` picks internally.
- Plan state is owned by the runtime: one cuBLASLt handle and a geometry-keyed
  plan cache per CubeCL stream slot, analogous to the cached cuBLAS handles.
  Descriptors, the heuristic algorithm, and the heuristic-selected workspace are
  cached together, so a warm call is one `cublasLtMatmul` on the existing
  CubeCL stream with no added synchronization.
- Larger slices keep the cuBLAS strided-batched path and non-standard geometry
  keeps cuTENSOR. A failed descriptor or heuristic query falls back to cuBLAS
  instead of failing the contraction; an enqueue failure stays an error.
- Workspaces retire through the runtime's existing deferred-retirement queue.
  Runtime teardown retires every stream before clearing plans, so cached
  workspaces and descriptors are released without adding a per-call barrier, and
  an unconfirmed retirement leaks them with the other raw library resources.
- The plan cache is bounded at 64 geometries per stream slot and a heuristic may
  request at most 4 MiB of cached workspace for this path.

## Verification conclusions and constraints
- nsys on the focused 16-by-32x32 f64 test (one call, commit `7b6ec931e`) shows
  exactly one GEMM kernel for the cuBLASLt route:
  `cutlass_sm80_tensorop_d884gett_f64_iden_f64_iden_f64_f64_f64_32x32x16_2x2x1_4x4x8_16_ttt_align1`.
- With the routing temporarily forced off (`LT_MAX_DIM = 0`, not committed), the
  same call reached cuBLAS `gemmStridedBatchedEx` as **one** kernel
  (`cutlass::Kernel2<cutlass_80_tensorop_d884gemm_32x32_16x5_nn_align1>`) on this
  host's cuBLAS 12.6.4.1. The per-batch kernel count from the motivating census
  (16 kernels per call) did not reproduce in isolation here, so this change is
  not a kernel-count reduction on every driver: it makes the batched plan
  explicit and cacheable instead of depending on cuBLAS' internal choice for the
  same shape.
- The `dense_batched_matmul_f64_b16_32` harness case is served by cuTENSOR in
  this tree (nsys on the harness backend shows cuTENSOR's
  `cutlass_80_tensorop_d884gemm` and `permute_coop` kernels and no cuBLASLt
  kernel), so its host-visible wall time cannot move with this routing. Measured
  at `7b6ec931e` over three runs: 0.045 / 0.043 / 0.044 ms trace and 0.024 /
  0.025 / 0.025 ms eager, against the recorded baseline 0.045 / 0.025 ms.
- `dense_batched_matmul_f64_b1024_256` stays on the strided path (`256 > 64`) and
  is unchanged: 2.120 ms trace / 2.097 ms eager.
- Device parity covers f32, f64, c32, and c64 with batch 2 and 16, square and
  rectangular slices, both transpose combinations, a two-batch-axis case, and a
  `65`-extent case that must stay strided. The cuBLASLt and strided launch
  counters prove which path served each case, so a numerical match alone cannot
  hide an unintended route.
