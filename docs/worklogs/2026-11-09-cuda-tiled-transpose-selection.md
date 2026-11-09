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
  `5 x 3 x 7` view the device test materializes through the selector and the
  planner and pins its seven batches, its tile rows and the 2176-byte
  shared tile, so a mis-wired derivation cannot reach the planner unnoticed
  either.
- The derivation is pinned by CPU tests: an asymmetric 3-D pair swap and a
  rank-two view at a nonzero offset address exactly the elements the view names
  (reproducing the kernel's own `dst[b*M*N + i + j*M] = src[o + b*M*N + i*N + j]`
  walk against a host reference), and a `[2, 0, 1]` rotation, a non-compact
  batch stride, reversed and broadcast strides, rank one and a negative offset
  all stay on the generic path.
- Two device tests assert that an adjacent pair swap launches the tiled entry
  once, that a three-axis rotation launches it not at all, and that both outputs
  equal the host reference. They skip on a host without a usable device and fail
  under `TENFERRO_REQUIRE_GPU=1`. Both ran green on the RunPod lane, run
  `42629754236`: its log records `PASS tenferro-gpu
  cuda::structural::tests::a_three_axis_rotation_stays_on_the_generic_materialization`
  and `PASS ...::an_adjacent_pair_swap_reads_through_the_tiled_transpose`, and the
  run's tested merge commit `fbbd8eb8` (tree `ff7a79f8`) merges this branch at
  `72a4ac20`, whose tree is the same `ff7a79f8`. The device fixture stays the
  `5 x 3 x 7` view those tests ran with; the multi-tile `17 x 19 x 7` coverage is
  host-side (the launch-mapping test), because a test-only change cancels
  `CI_gpu`.
- The six added tests are classified in `scripts/ci/cuda_test_partition.tsv`
  (four `host`, two `gpu`), which the lane inventory check requires before it
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
