# CUDA fused softplus (#2530)

## Decisions

- `CudaEngine` answers `Engine::fused_activation_read` for `softplus` (F32/F64)
  with one region from the `t4a-cuda-kernels` fusion emitter, not a new
  catalogue entry or a tenferro-local kernel source. The catalogue has no
  softplus unary, and the emitter is the catalogue's own route to a one-pass
  kernel that reuses the `abi.cuh` helpers the per-operation entries use.
- The region is `|x|`, `-`, `exp`, `log1p`, `x + tail`, `max(x + tail, tail)`.
  The fusion vocabulary has no compare/select and no constants; `max` makes the
  composite's `x > 0 ? x + tail : tail` choice, so results are bitwise equal to
  the composite (NaN class only for NaN inputs).
- The prepared region is kept on the engine, one slot per real dtype, beside the
  prepared dispatchers and under the same policy (fixed capacity, dropped with
  the engine). Re-emitting per call would rebuild and re-key a ~42 KB source.
- Launch: the emitter's flat geometry (`FLAT_BLOCK` = 256 threads, one element
  per thread, grid = ceil(n / 256)). Vector width and tile shape are #2529's
  question; this region inherits whatever that settles.
- Output allocation and the standalone `prove_launch` follow the existing
  element-wise unary path. Other activations still decline.

## Verification conclusions and constraints

- Red first: the CUDA test
  `gpu_nn_composites::test_gpu_softplus_is_one_fused_launch_matching_the_composite`
  failed at `ad281831f` with "CUDA declined the fused softplus" and passes at
  the candidate. It pins exactly one kernel launch on the concrete surface (the
  runtime's admitted-launch counter, not the wait counter) and, on the untracked
  eager surface, the launches and waits of one eager unary op, and bitwise equality with the composite on per-dtype boundary
  values: +-0, finite extremes, smallest normal and subnormal magnitudes, inputs
  with a nonzero subnormal tail (-744 in f64, -100 in f32), +-inf, +-NaN and a
  multi-block ramp.
- Independent review found that the eager fused-activation fast path skipped
  the runtime check, so a tensor from another `EagerRuntime` was accepted
  (pre-existing on CPU, newly reachable on CUDA). It now refuses it with
  `ContextMismatch` like the composite path; the CPU test
  `nn_composites::fused_activations_refuse_a_tensor_from_another_runtime`
  failed before that fix and passes after it, for all five activations.
- The eager surface already pays two kernel launches and two standalone
  completions for a single `exp` (a direct `exp` pays one of each); that per-op
  eager cost is outside this change.
- Traced `TracedTensor::softplus` still lowers to the composite graph; the
  fused hook covers the concrete and untracked eager surfaces only (the two
  benchmark arms).
- A100 80GB PCIe (this host), tenferro-benchmark `gpu/nn` harness at
  `918046a`, cases `softplus_{f32,f64}_n{1024,1048576}`, release build,
  15 runs / 3 warmups, order A B A B per arm, process pinned to core 30.
  Before every run: no CUDA compute process and 0% GPU utilization; core 30
  idle. Another user's CPU-only jobs (load about 16, pinned to other core
  pairs) ran throughout, so the GPU was exclusive and the host CPU was not.
  Per-operation median (IQR), f64 n=1048576, rounds 1 | 2:

  **Measured at `079caefc6`; the submitted head `49b33c8d` was not re-measured.**
  Between the two, this path changes only in comments in `activation.rs` (the
  emitted fusion source, hence the compiled kernel, is the same) and by one
  host-side `Arc::ptr_eq` runtime check added to the eager fast path; the numbers
  below are from `079caefc6` and were not re-run at `49b33c8d`.

  | arm | before `ad281831f` | after `079caefc6` |
  |---|---|---|
  | direct | 214.88 (2.31) / 223.22 (3.40) us | 40.38 (0.72) / 44.20 (4.04) us |
  | eager | 359.13 (4.41) / 368.88 (9.71) us | 70.99 (0.66) / 72.72 (0.97) us |

  Same-run PyTorch: 25.99 (0.08) us. Every row verified `ok` by the harness.
  The before rounds differ by 4-10% (A/A noise floor).
- Remaining gap to PyTorch (direct about 1.6x, eager about 2.7x) is not
  attributed here. Countable candidates for #2205: the output zero fill, the
  per-call synchronize, the extra eager completion.
