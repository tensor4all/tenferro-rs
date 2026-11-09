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
- The tile configuration is the catalogue's `TransposeConfig::DEFAULT`. Its
  `__launch_bounds__` cap was measured against that configuration there, and no
  member of the family has been measured for the layouts this repository sends.
  Choosing another member without that measurement would be a guess.
- `CudaEngine::tiled_transposes` is a `#[cfg(test)]` counter, the same shape as
  the existing `staged_reads`, so the selection and its negative control are
  observable from a device test.

## Verification conclusions and constraints

- The derivation is pinned by CPU tests: an asymmetric 3-D pair swap and a
  rank-two view at a nonzero offset address exactly the elements the view names
  (reproducing the kernel's own `dst[b*M*N + i + j*M] = src[o + b*M*N + i*N + j]`
  walk against a host reference), and a `[2, 0, 1]` rotation, a non-compact
  batch stride, reversed and broadcast strides, rank one and a negative offset
  all stay on the generic path.
- Two device tests assert that an adjacent pair swap launches the tiled entry
  once, that a three-axis rotation launches it not at all, and that both outputs
  equal the host reference. They skip on a host without a usable device and fail
  under `TENFERRO_REQUIRE_GPU=1`; **they have not been executed** — this branch
  was developed on a machine with no CUDA toolkit, GPU or driver.
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
