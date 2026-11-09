# CUDA tiled-transpose selection for materialized views

Issue: tensor4all/tenferro-rs#2528, stage 1 of two.

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
  `[19, 1, 323]`, offset zero), whose plan is grid `[2, 2, 7]`; a third one is the
  same view at offset one over a `2262`-element allocation with a dummy leading
  element, so a launch that passed `src_offset` as zero reads every element one
  place off. Both pair-swap comparisons cover the whole destination. All three
  ran green on this branch's local A100 lane (`scripts/ci/local_gpu_gate.py run`
  on `primerose`, published as the `CI GPU gate` evidence comment on PR #2551,
  which records the tested SHA). Two mutations were executed there: forcing
  `grid.1` to one in `cuda/dispatch.rs::launch_config` fails the multi-tile test
  with the second tile row unwritten (357 of 2261 elements), and replacing the
  launch's `Arg::U64(geometry.src_offset as u64)` with `Arg::U64(0)` fails the
  offset test on all 2261 elements while leaving the offset-zero test green.
  Reverting either mutation makes the test pass again. The earlier RunPod run
  `42629754236` executed the previous single-tile `5 x 3 x 7` fixture of these
  tests, so it does not establish multi-tile or nonzero-offset device coverage;
  that evidence is the local run above.
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
