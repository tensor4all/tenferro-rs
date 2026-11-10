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
  each, and every cell verified. The full table, all samples and the protocol
  are in the PR description.
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
