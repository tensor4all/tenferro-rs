# CUDA reduction per-call fixed cost (#2406)

## Decisions

- The per-call fixed cost was named from an `nsys` trace of
  `reduce_max_full_f32_n1048576`, not from the timing inference in the issue's
  root-cause comment. That inference was wrong in one place: every standalone
  native op synchronizes, not only the reduction. The extra cost was that each
  standalone reduction made 3 launches, 3 pool allocations with 3 zero fills, and
  2 waits. Two of the three launches are the kernel crate's split plan. The third
  was a `copy_exact` that squeezed the keepdims `[1]` result to `[]`, and it
  carried its own allocation, zero fill and second wait.
- The fix is host-side in `tenferro-gpu`, with no kernel ABI change:
  - the last pass writes the squeezed shape directly;
  - the output and the split partials come from `alloc_uninit`;
  - one wait follows the last pass.
  Each call is now 2 launches, 2 allocations, 0 zero fills and 1 wait. The
  kernels and their arguments are unchanged, so results are bit-identical.
- The remaining wait stays. It is the documented completion contract of a
  standalone call (`prove_launch`), and deferring it changes public semantics,
  which #2406 already declined.
- No persistent partials scratch was added. With the zero fill gone, a pool
  allocation plus its stream-ordered free is about 1.7 µs of traced host time
  per call. An engine-owned scratch would save at most that, but it would need
  the bound, clear and stats surface that PERFORMANCE_TIPS "Cache Ownership"
  requires.

## Verification conclusions and constraints

- Results are bit-identical to `baa1cf4b` for sum, prod, max, min and
  sum-of-squares, in `f32` and `f64`, at every plan boundary (n = 1 to 4M) and on
  multi-axis shapes. The comparison was a scratch dump on both trees, 396
  results.
- In the paired `gpu/isolated_ops` run, all 52 timed reduction cells of the
  direct arm improve (0.47 to 0.66, median 0.57). The full-reduce cells at 1M
  go from 47 to 55 µs down to 27 to 35 µs. The per-call floor of a reduction now
  equals that of an elementwise op. The evidence and commands are in the PR.
- The profile-based prediction for the primary cell was 34 ± 3 µs. It measured
  30.3 to 30.5 µs, better than the band. The prediction scaled traced host time
  by one traced-to-untraced factor, which underweights the removed API-heavy
  segment.
- Still open, outside this repository's reduction path:
  - GPU time: 2 launches of 16.35 µs in total, against PyTorch's single
    7.87 µs `reduce_kernel` plus a 1.09 µs memset. Closing that needs a
    one-kernel reduction with a different geometry, which reassociates sums, so
    it is a design decision for `t4a-cuda-kernels`.
  - The wait contract: PyTorch does not wait per op.
  - `t4a-cuda-runtime`'s per-wait event gate and per-call `cuCtxSetCurrent`
    calls (t4a-gpu-backend#13).
  - `ncu` could not collect on this host (`ERR_NVGPUCTRPERM`). The launch
    geometry comes from the CUPTI records in `nsys`.
- The rank > `MAX_RANK` general reduction path (`reduce_axes_wide`) keeps its
  zeroed allocations and its per-pass wait. The benchmark does not reach it.
