# Batched Tiled Transpose Design

## Context

The issue #1507 native permutation implementation tiles fused rank-two
transposes. The Metal profile shows that `mac_transpose_3d_102`
(`256x256x240`, permutation `[1,0,2]`) remains about 1.7 times slower than
PyTorch MPS. Its fused layout is 240 independent compact `256x256`
transposes, but it currently uses the generic rank-three affine kernel.

## Decision

Extend the existing `TiledTranspose` specialization to compact batched
two-dimensional transposes. A plan is eligible when:

- the first two fused axes already satisfy the rank-two transpose predicate;
- every remaining fused axis is contiguous in both source and destination;
- source and destination use the same matrix-sized stride for the first batch
  axis, so every batch is an independent compact matrix;
- the batch product and launch grid are representable without overflow.

The initial implementation accepts exactly one fused batch axis. This covers
the measured `[1,0,2]` case without introducing a general permutation
classifier. Higher-rank layouts remain on the generic path.

The tiled kernel uses `CUBE_POS_Z` as the batch index. It adds a
compile-time `batch_stride = rows * columns` to both source and destination
base addresses. The two-dimensional shared-memory load, barrier, and store are
otherwise unchanged. Rank-two transposes use a batch count of one.

If any X, Y, or Z dispatch dimension exceeds 65,535, launch selection returns
to the generic kernel. CUDA and wgpu consume the same planner decision and
CubeCL kernel definition.

## Alternatives

1. Flatten batch into the Y grid dimension. This avoids the Z limit but adds
   division and remainder work to every workgroup and complicates the existing
   two-dimensional coordinate contract.
2. Optimize the generic rank-three decoder. This benefits more layouts but
   cannot provide coalesced access on both sides of the transpose and does not
   address the measured shared-memory opportunity.

The Z-grid design is the smallest extension with a direct performance
hypothesis.

## Validation

- Planner tests prove `[1,0,2]` is tiled, a non-compact batch remains generic,
  and an oversized batch grid falls back.
- Existing rank-two and boundary-tile tests must remain green.
- Metal runtime correctness covers the full `mac_transpose_3d_102` pattern.
- The optimization is retained only if the stable Metal median improves
  without causing any wgpu row to exceed the +20% regression gate.
- CUDA plus WebGPU compile together; Linux A100 CUDA and wgpu/Vulkan runtime
  execution remains the required pre-merge validation.

