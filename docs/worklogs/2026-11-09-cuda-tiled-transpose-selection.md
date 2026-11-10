# CUDA tiled-transpose selection for materialized views

Issue: tensor4all/tenferro-rs#2528. Stage 1 is #2551; stage 2 is the section at the end.

## Decisions

- The catalogue already ships `t4a_cuda_tiled_transpose` (32-, 16- and 8-wide
  shared-memory tiles with a `TransposeConfig` family, tuned where the kernel
  lives). Stage 1 wires that entry into the materialization boundary instead of
  adding a once-per-repository tile kernel: `tenferro-gpu` owns dispatch, not
  CUDA sources, and a second tiled transpose in this repository would be a
  kernel this crate cannot benchmark.
- The entry is selected from one pure derivation,
  `cuda/structural.rs::tiled_transpose_geometry`, which is a *selection*, not a
  fallback tier: the generic `MaterializeStrided` path stays the implementation
  for every other view, and a launch the catalogue's planner refuses also keeps
  it. The catalogue's permutation implementation for this backend is the kernel
  catalogue, not cuTENSOR, so no vendor-library fallback policy is involved.
- The accepted geometry is a view whose destination-fastest axis reads the
  source at the second axis' extent, whose second axis is the source's fastest
  at stride one, and whose remaining axes form one compact run in the source.
  That is exactly "the source's two fastest axes swapped, batched" — the shape
  the kernel's shared-memory tile is built for. A general N-D permutation is out
  of scope, and the derivation says so rather than approximating one.
- The tile configuration is the catalogue's `TransposeConfig::DEFAULT` (tile
  16, 8 block rows, padding 1, vector width 1). The reason to keep it is stated
  in the kernel's own source: its `__launch_bounds__` cap exists because, without
  it, ptxas costs "the default 16x8 tile 8-12 % of bandwidth on the A100"
  (`t4a-cuda-kernels` at the pinned revision, `cuda/src/movement.cu:565-570`).
  No member of the family has been measured for the layouts this repository
  sends, so choosing another would be a guess and no benchmark artifact here
  supports one.
- `CudaEngine::tiled_transposes` is a production field, incremented by
  `note_tiled_transpose` on the materialization path (the same shape as
  `staged_reads`); only its getter is `#[cfg(test)]`, so the selection and its
  negative control are observable from a device test.

## Verification conclusions and constraints

- A review of the first head found that the dispatcher built every launch from
  `plan.grid[0]` and `plan.block[0]` alone, dropping the tiled entry's tile rows
  (`y`), batches (`z`), its `block.y` and its dynamic shared memory. The mapping
  now lives in `cuda/dispatch.rs::launch_config` and carries all three axes and
  the shared bytes; the raw-scope path that also derives a config from a plan
  uses the same function. A CPU test pins a real tiled plan of a `17 x 19 x 7`
  pair swap — grid `[2, 2, 7]`, so a mapping that kept only `grid[0]` and `grid[2]`
  would not pass by accident — against the mapping, with the plan's `block.y`
  and its non-zero shared-memory request, and fails on the truncating form, so
  the defect class is caught without a device; a second one takes the same
  `17 x 19 x 7` view the device test materializes through the selector and the
  planner and pins its grid `[2, 2, 7]`, its `block` `[16, 8, 1]` and the
  2176-byte shared tile. That second test pins the derivation-to-plan seam and
  nothing more: it never names a launch argument, so a mis-wired `src_offset`
  or a grid axis dropped at the launch call site is outside it. The mapping
  test above is the one that sees the mapping.
- The derivation is pinned by CPU tests: an asymmetric 3-D pair swap and a
  rank-two view at a nonzero offset address exactly the elements the view names
  (reproducing the kernel's own `dst[b*M*N + i + j*M] = src[o + b*M*N + i*N + j]`
  walk against a host reference), and a `[2, 0, 1]` rotation, a non-compact
  batch stride, reversed and broadcast strides, rank one and a negative offset
  all stay on the generic path. Those tests reimplement the walk on the host;
  the `src_offset` argument the launch actually carries is covered by a device
  test instead (below).
- Three device tests assert that a pair swap launches the tiled entry once, that
  a three-axis rotation launches it not at all, and that every output equals the
  host reference. They skip on a host without a usable device and fail under
  `TENFERRO_REQUIRE_GPU=1`. The pair-swap fixture is the multi-tile
  `17 x 19 x 7` view (`[19, 17, 7]` contiguous storage, strides
  `[19, 1, 323]`, offset zero), whose plan is grid `[2, 2, 7]`; the third is the
  same view at offset one over a `2262`-element allocation with a dummy leading
  element, so a launch that carried `src_offset` as zero reads every element one
  place off. Both pair-swap comparisons cover the whole destination.
- Device results are bound to the head they were run at and are never inherited
  by a later one: `scripts/ci/local_gpu_gate.py run` names its SHA in the
  `CI GPU gate` evidence comment for that head on PR #2551, and a head that
  changes needs its own run. At the head that contains this file the lane
  reports 361 device tests passed, both pair-swap tests among them, and two
  source mutations were executed there and reverted: forcing `grid.1` to one in
  `cuda/dispatch.rs::launch_config` fails the multi-tile test with the second
  tile row unwritten (357 of 2261 elements), and replacing the launch's
  `Arg::U64(geometry.src_offset as u64)` with `Arg::U64(0)` fails the offset test
  on all 2261 elements while leaving the offset-zero test green.
- Earlier heads do not cover what came after them. `5e547ffe` passed this lane
  with 360 device tests and only the offset-zero fixtures, so it does not cover
  the offset test, and the RunPod run `42629754236` executed the single-tile
  `5 x 3 x 7` fixture of these tests, so it establishes neither multi-tile nor
  nonzero-offset device coverage.
- The seven added tests are classified in `scripts/ci/cuda_test_partition.tsv`
  (four `host`, three `gpu`), which the lane inventory check requires before it
  executes anything.
- No timing was taken, so stage 1 claims no speedup. Which rows it should move
  is a hypothesis for the GPU host: `transpose_3d_1024_1024_512_102` (`[1, 0, 2]`),
  `transpose_2d_32768_16384` and any batched two-dimensional transpose
  materialized through `transpose`/`to_contiguous`. The issue's largest row,
  `transpose_3d_1024_1024_512_201` at 81.09 ms, is a three-axis rotation and is
  **not** affected by stage 1.
- `copy_read_into` keeps `StridedToStrided` for every non-flat pair, including
  the benchmark's destination-reuse arm: its destination is a strided region,
  while the tiled entry writes a compact destination. Covering that arm needs a
  kernel that tiles a strided destination as well.

## Stage 2: merged axes, reordered destinations, no zero fill

### Decisions

- Stage 2 was decided from an `nsys` profile of `020d6911c` (stage 1 merged) on
  the A100, not from the issue's hypothesis. The 2-D transpose arm was already
  on the tiled kernel (5.58 ms), but each call first zero-filled its 4 GiB
  destination (a 2.4 ms `cuMemsetD8Async`). `[2, 0, 1]` and `cyclic_18d_3` ran
  the generic `materialize_strided`, and every destination-reuse arm ran
  `strided_to_strided`.
- Axis merging, not a new kernel. Once the axes the source continues through
  are merged, `[2, 0, 1]` of a compact source is the pair swap stage 1 already
  serves (`512 x 2^20`), and so is the cyclic shift (`3^17 x 3`). The stage-1
  comment's options A (a two-tiled-axis catalogue kernel) and A' (two passes)
  are therefore not needed for these rows.
- The merged `[2, 0, 1]` run has `2^20 / 16 = 65536` tiles on `grid.y`, one
  more than the hardware allows, so the planner refuses the default tile. The
  catalogue's own `32 x 8` member is tried only after that refusal. Every
  geometry stage 1 served keeps `TransposeConfig::DEFAULT`, so the reasoning
  above for that default still holds.
- `copy_read_into`: a destination that is a permuted compact span at offset 0
  is written in its own order. That turns the copy into the same materialization
  instead of needing a kernel that tiles a strided destination (the constraint
  stage 1 recorded). Offset destinations keep `StridedToStrided`. The catalogue's
  entries take no destination offset, and rebasing through a raw scope is not
  needed by any measured row.
- cuTENSOR (`cutensorPermute`) was not chosen. `cuda/cutensor.rs` binds
  contraction only, so this route would need new FFI, a plan cache and the typed
  vendor-absent error policy. It would also make permutation depend on a vendor
  library being loadable. After the merged-axis route reached the cuTENSOR
  figure on the issue's headline row, nothing measured justified that cost.

### Verification conclusions and constraints

- Paired A100 experiment, baseline `020d6911c` against candidate `09089d0ec`:
  every Rust arm of every `gpu/permutation` pattern, five interleaved processes
  each, and every cell passing the runner's sampled check. The commands, the
  host, all per-process samples and the criterion are in the execution record
  below.
  - `transpose_3d_1024_1024_512_201`: 29.77 to 5.21 ms (transpose) and 84.98 to
    5.18 ms (destination reuse), against cuTENSOR at 5.33 ms.
  - No cell regressed.
- Not reached: `reverse_18d_3` and `reverse_23d_128x2` are unchanged (about
  30, 42, 61 and 85 ms against cuTENSOR at about 5 ms). No axes merge, the views
  stay above the by-value rank, and the wide path keeps both its generic kernel
  and its zero fill. Two more rows remain above cuTENSOR:
  - `cyclic_18d_3` is about 2.1x cuTENSOR, because a 3-wide source run fills
    3 of the 16 tile lanes.
  - `rotation_6d` is about 1.35x cuTENSOR, on the generic kernel.
  Closing these needs an N-D tiled kernel in `t4a-cuda-kernels`.
- A device test for each new route fails on `020d6911c`, because the tiled
  counter stays at 0, and passes here. The offset-destination control passes on
  both. The `[2, 0, 1]` negative control of stage 1 now selects the tiled entry
  by design, so it was replaced by a `[2, 1, 0]` reversal.

### Execution record

What the numbers above rest on, and what a reader can check from it. The
protocol was recorded before the candidate's first timed run (2026-11-11
04:23 +09:00; first process 04:25:05).

| item | value |
|---|---|
| host / GPU | `primerose`, NVIDIA A100 80GB PCIe, device 0 |
| driver | 580.178.04, driver CUDA 13.0 |
| toolkit | `/usr/local/cuda` -> CUDA 12.6 (V12.6.85), used through `LD_LIBRARY_PATH` |
| rustc | 1.99.1; both binaries embed `/rustc/58cf8f9c3`. The repository pins no toolchain. |
| baseline | tenferro-rs `020d6911c`, `origin/main` at the time |
| candidate | tenferro-rs `09089d0ec`; the branch's two later commits (`2b6da075f`, `fe2656940`) change docs and one doc comment |
| benchmark | tenferro-benchmark `41d25b1`, `src/bin/benchmark_gpu_permutation.rs`, `extern/tenferro-rs` pointing at each revision's worktree |
| processes | 5 per binary, interleaved B, C, B, C, ... ; `taskset -c 8-15`; runner defaults (3 warmups, 15 timed iterations) |

The two binaries were built in the benchmark worktree with that script's own
build command, `cargo build --release --features cuda --bin
benchmark_gpu_permutation` (`scripts/run_gpu_permutation.sh:176`). That build
invocation and its log were not saved. The run loop, as executed:

```text
cd $HOME/tensor4all/.worktrees/tbench-2528
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-} RUST_MIN_STACK=67108864 \
  GPU_BENCH_DEVICE=0 \
  TENFERRO_CUTENSOR_PATH=/usr/lib/x86_64-linux-gnu/libcutensor/12/libcutensor.so.2
for rep in 1 2 3 4 5; do
  for arm in base_020d6911c cand_09089d0ec; do
    apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
    echo "rep=$rep arm=$arm gpu_apps_before=$apps $(date -Is) $(nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader)" >> runs/log.txt
    BENCH_OUTPUT=runs/${arm}_r$rep.jsonl taskset -c 8-15 bin/bench_$arm > runs/${arm}_r$rep.out 2>&1
    echo "rep=$rep arm=$arm rc=$? $(date -Is)" >> runs/log.txt
    sleep 2
  done
done
```

`CUDA_VISIBLE_DEVICES` was not set in any process; the ordinal came from
`GPU_BENCH_DEVICE=0`. `PERMUTATION_FULL_VERIFY`, `BENCH_RUNS` and
`BENCH_WARMUPS` were not set either, so every process ran the defaults. All ten
processes logged `gpu_apps_before=0`, exit code 0, 42-59 C and 210-375 MHz SM
clock.

Samples: one value per process, that process's `median_ms`, in run order
r1..r5. Baseline `020d6911c`:

| problem_id | arm | r1 | r2 | r3 | r4 | r5 | median (IQR) |
|---|---|---|---|---|---|---|---|
| `cyclic_18d_3` | cuTENSOR | 4.744 | 4.747 | 4.784 | 4.776 | 4.733 | **4.747** (0.033) |
| `cyclic_18d_3` | destination-reuse | 38.550 | 38.554 | 38.554 | 38.556 | 38.554 | **38.554** (0.001) |
| `cyclic_18d_3` | to-contiguous | 22.168 | 22.175 | 22.170 | 22.175 | 22.172 | **22.172** (0.005) |
| `cyclic_18d_3` | transpose | 22.177 | 22.179 | 22.176 | 22.181 | 22.178 | **22.178** (0.003) |
| `reverse_18d_3` | cuTENSOR | 5.023 | 5.025 | 5.028 | 5.027 | 5.023 | **5.025** (0.004) |
| `reverse_18d_3` | destination-reuse | 61.072 | 61.201 | 61.070 | 61.139 | 61.196 | **61.139** (0.124) |
| `reverse_18d_3` | to-contiguous | 30.168 | 30.106 | 30.084 | 30.083 | 30.107 | **30.106** (0.022) |
| `reverse_18d_3` | transpose | 30.214 | 30.128 | 30.130 | 30.108 | 30.105 | **30.128** (0.022) |
| `reverse_23d_128x2` | cuTENSOR | 5.654 | 5.660 | 5.664 | 5.659 | 5.664 | **5.660** (0.004) |
| `reverse_23d_128x2` | destination-reuse | 84.792 | 84.808 | 84.881 | 84.889 | 84.822 | **84.822** (0.074) |
| `reverse_23d_128x2` | to-contiguous | 41.944 | 42.018 | 42.022 | 42.022 | 42.026 | **42.022** (0.004) |
| `reverse_23d_128x2` | transpose | 41.963 | 41.967 | 42.033 | 42.035 | 42.041 | **42.033** (0.068) |
| `rotation_6d_64_32_32_32_16_16` | cuTENSOR | 5.400 | 5.401 | 5.403 | 5.401 | 5.401 | **5.401** (0.001) |
| `rotation_6d_64_32_32_32_16_16` | destination-reuse | 65.228 | 65.262 | 64.987 | 65.205 | 64.884 | **65.205** (0.241) |
| `rotation_6d_64_32_32_32_16_16` | to-contiguous | 9.494 | 9.579 | 9.622 | 9.608 | 9.644 | **9.608** (0.043) |
| `rotation_6d_64_32_32_32_16_16` | transpose | 9.516 | 9.576 | 9.579 | 9.588 | 9.527 | **9.576** (0.052) |
| `tn_light_415_24d_contiguous_same_perm_gpu` | cuTENSOR | 5.205 | 5.204 | 5.203 | 5.203 | 5.207 | **5.204** (0.002) |
| `tn_light_415_24d_contiguous_same_perm_gpu` | destination-reuse | 70.011 | 70.029 | 70.025 | 70.025 | 70.021 | **70.025** (0.004) |
| `tn_light_415_24d_contiguous_same_perm_gpu` | to-contiguous | 38.861 | 38.869 | 38.867 | 38.870 | 38.869 | **38.869** (0.003) |
| `tn_light_415_24d_contiguous_same_perm_gpu` | transpose | 38.886 | 38.887 | 38.885 | 38.888 | 38.887 | **38.887** (0.001) |
| `tn_light_415_24d_scattered_to_colmajor_gpu` | cuTENSOR | 5.477 | 5.470 | 5.474 | 5.476 | 5.474 | **5.474** (0.003) |
| `tn_light_415_24d_scattered_to_colmajor_gpu` | to-contiguous | 38.862 | 38.866 | 38.869 | 38.870 | 38.880 | **38.869** (0.004) |
| `transpose_2d_32768_16384` | cuTENSOR | 5.401 | 5.406 | 5.396 | 5.400 | 5.397 | **5.400** (0.004) |
| `transpose_2d_32768_16384` | destination-reuse | 25.582 | 25.589 | 25.582 | 25.584 | 25.584 | **25.584** (0.001) |
| `transpose_2d_32768_16384` | to-contiguous | 8.173 | 8.173 | 8.163 | 8.173 | 8.150 | **8.173** (0.010) |
| `transpose_2d_32768_16384` | transpose | 8.016 | 8.015 | 8.016 | 8.011 | 8.011 | **8.015** (0.004) |
| `transpose_3d_1024_1024_512_102` | cuTENSOR | 5.308 | 5.308 | 5.307 | 5.306 | 5.307 | **5.307** (0.001) |
| `transpose_3d_1024_1024_512_102` | destination-reuse | 17.785 | 17.773 | 17.768 | 17.792 | 17.764 | **17.773** (0.017) |
| `transpose_3d_1024_1024_512_102` | to-contiguous | 7.883 | 7.883 | 7.880 | 7.881 | 7.882 | **7.882** (0.002) |
| `transpose_3d_1024_1024_512_102` | transpose | 7.734 | 7.733 | 7.732 | 7.734 | 7.734 | **7.734** (0.001) |
| `transpose_3d_1024_1024_512_201` | cuTENSOR | 5.334 | 5.335 | 5.333 | 5.331 | 5.337 | **5.334** (0.002) |
| `transpose_3d_1024_1024_512_201` | destination-reuse | 84.966 | 85.065 | 84.981 | 84.991 | 84.954 | **84.981** (0.025) |
| `transpose_3d_1024_1024_512_201` | to-contiguous | 29.751 | 29.751 | 29.750 | 29.753 | 29.751 | **29.751** (0.000) |
| `transpose_3d_1024_1024_512_201` | transpose | 29.772 | 29.773 | 29.770 | 29.775 | 29.772 | **29.772** (0.001) |

The `median` and `IQR` columns were computed from the runner's unrounded `median_ms` values, which the
archive does not keep; only the three decimals shown are. Recomputing the IQR from the rounded samples above
therefore disagrees in the last digit on 19 of the 68 rows, in both directions (`cyclic_18d_3`
destination-reuse gives 0.000 against the stated 0.001). The differences are at most 0.001 ms. **Treat the
medians as the reproducible quantity**; the IQR is here to show the spread was under a hundredth of the
median, not as a digit-exact statistic.

Candidate `09089d0ec`:

| problem_id | arm | r1 | r2 | r3 | r4 | r5 | median (IQR) |
|---|---|---|---|---|---|---|---|
| `cyclic_18d_3` | cuTENSOR | 4.742 | 4.777 | 4.804 | 4.780 | 4.780 | **4.780** (0.004) |
| `cyclic_18d_3` | destination-reuse | 9.946 | 9.895 | 9.895 | 9.895 | 9.895 | **9.895** (0.000) |
| `cyclic_18d_3` | to-contiguous | 9.898 | 9.898 | 10.053 | 9.968 | 10.004 | **9.968** (0.106) |
| `cyclic_18d_3` | transpose | 9.898 | 9.897 | 9.897 | 9.897 | 9.899 | **9.897** (0.001) |
| `reverse_18d_3` | cuTENSOR | 5.026 | 5.027 | 5.025 | 5.025 | 5.027 | **5.026** (0.002) |
| `reverse_18d_3` | destination-reuse | 61.247 | 61.163 | 61.111 | 61.191 | 61.189 | **61.189** (0.028) |
| `reverse_18d_3` | to-contiguous | 30.110 | 30.082 | 30.079 | 30.079 | 30.089 | **30.082** (0.009) |
| `reverse_18d_3` | transpose | 30.134 | 30.132 | 30.134 | 30.131 | 30.108 | **30.132** (0.003) |
| `reverse_23d_128x2` | cuTENSOR | 5.653 | 5.661 | 5.665 | 5.663 | 5.661 | **5.661** (0.002) |
| `reverse_23d_128x2` | destination-reuse | 84.829 | 84.865 | 84.885 | 84.864 | 84.857 | **84.864** (0.007) |
| `reverse_23d_128x2` | to-contiguous | 41.949 | 42.020 | 42.017 | 42.036 | 42.034 | **42.020** (0.017) |
| `reverse_23d_128x2` | transpose | 41.965 | 41.997 | 42.064 | 42.040 | 42.023 | **42.023** (0.043) |
| `rotation_6d_64_32_32_32_16_16` | cuTENSOR | 5.401 | 5.401 | 5.400 | 5.398 | 5.400 | **5.400** (0.001) |
| `rotation_6d_64_32_32_32_16_16` | destination-reuse | 7.193 | 7.246 | 7.297 | 7.305 | 7.326 | **7.297** (0.059) |
| `rotation_6d_64_32_32_32_16_16` | to-contiguous | 7.246 | 7.319 | 7.317 | 7.308 | 7.335 | **7.317** (0.012) |
| `rotation_6d_64_32_32_32_16_16` | transpose | 7.198 | 7.310 | 7.312 | 7.298 | 7.307 | **7.307** (0.013) |
| `tn_light_415_24d_contiguous_same_perm_gpu` | cuTENSOR | 5.207 | 5.206 | 5.205 | 5.208 | 5.208 | **5.207** (0.002) |
| `tn_light_415_24d_contiguous_same_perm_gpu` | destination-reuse | 6.133 | 6.136 | 6.182 | 6.136 | 6.139 | **6.136** (0.003) |
| `tn_light_415_24d_contiguous_same_perm_gpu` | to-contiguous | 6.193 | 6.208 | 6.266 | 6.197 | 6.221 | **6.208** (0.024) |
| `tn_light_415_24d_contiguous_same_perm_gpu` | transpose | 6.136 | 6.141 | 6.187 | 6.189 | 6.152 | **6.152** (0.046) |
| `tn_light_415_24d_scattered_to_colmajor_gpu` | cuTENSOR | 5.474 | 5.475 | 5.478 | 5.475 | 5.471 | **5.475** (0.002) |
| `tn_light_415_24d_scattered_to_colmajor_gpu` | to-contiguous | 8.266 | 8.352 | 8.352 | 8.352 | 8.352 | **8.352** (0.000) |
| `transpose_2d_32768_16384` | cuTENSOR | 5.400 | 5.398 | 5.403 | 5.396 | 5.403 | **5.400** (0.005) |
| `transpose_2d_32768_16384` | destination-reuse | 5.574 | 5.575 | 5.575 | 5.573 | 5.572 | **5.574** (0.002) |
| `transpose_2d_32768_16384` | to-contiguous | 5.724 | 5.726 | 5.725 | 5.727 | 5.728 | **5.726** (0.002) |
| `transpose_2d_32768_16384` | transpose | 5.582 | 5.583 | 5.581 | 5.579 | 5.583 | **5.582** (0.002) |
| `transpose_3d_1024_1024_512_102` | cuTENSOR | 5.307 | 5.309 | 5.309 | 5.305 | 5.307 | **5.307** (0.002) |
| `transpose_3d_1024_1024_512_102` | destination-reuse | 5.286 | 5.285 | 5.284 | 5.286 | 5.285 | **5.285** (0.001) |
| `transpose_3d_1024_1024_512_102` | to-contiguous | 5.428 | 5.428 | 5.427 | 5.433 | 5.426 | **5.428** (0.001) |
| `transpose_3d_1024_1024_512_102` | transpose | 5.300 | 5.303 | 5.299 | 5.301 | 5.303 | **5.301** (0.003) |
| `transpose_3d_1024_1024_512_201` | cuTENSOR | 5.331 | 5.334 | 5.333 | 5.331 | 5.336 | **5.333** (0.002) |
| `transpose_3d_1024_1024_512_201` | destination-reuse | 5.177 | 5.176 | 5.175 | 5.176 | 5.176 | **5.176** (0.001) |
| `transpose_3d_1024_1024_512_201` | to-contiguous | 5.310 | 5.313 | 5.309 | 5.308 | 5.307 | **5.309** (0.002) |
| `transpose_3d_1024_1024_512_201` | transpose | 5.207 | 5.206 | 5.205 | 5.203 | 5.206 | **5.206** (0.001) |

What these records establish:

- A cell counts as verified when its record reports `status: "ok"` and
  `correctness: "passed"`. All 68 cells (34 pattern/arm pairs on both binaries)
  do, in all five processes: 340 records, 0 non-ok, 0 `verification_failed`.
- The check is the runner's default sampled gate, not a whole-output compare:
  exact equality (`!=`) of sampled ranges — per-axis corners plus 1000 seeded
  blocks, about 1M of up to 2^29 elements — against precomputed expected values,
  run outside any timed region before timing
  (`src/bin/benchmark_gpu_permutation.rs:22-30, 459-476`). Every record reads
  `verification_mode: "sampled"`; `PERMUTATION_FULL_VERIFY=1` was not run, so
  these records do not exclude a defect outside the sampled ranges.
- "Improved" is the protocol's test, candidate max < baseline min; the 19
  improved cells meet it. The 6 unchanged cells are `reverse_18d_3` and
  `reverse_23d_128x2` on all three arms, and no cell's median is more than 5 %
  above its baseline. The 9 cuTENSOR cells are the control: the same binary
  pair, unchanged.
- cuTENSOR numbers are the same process's cuTENSOR arm, through the library
  `TENFERRO_CUTENSOR_PATH` names above.
- The kernel medians in the profile section come from separate `nsys` runs at
  the same two revisions, not from these wall-clock samples.
- Raw artifacts — the 10 JSON Lines files, `log.txt`, the protocol and the
  scripts — are kept at `~/work/t4a-2528-paired/` on the measurement host;
  `analyze.py` there regenerates every median and IQR in this section.
