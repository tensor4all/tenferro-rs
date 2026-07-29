# CubeCL Native Permutation Optimization for WebGPU/Metal

Date: 2026-07-29

Issue: [tensor4all/tenferro-rs#1507](https://github.com/tensor4all/tenferro-rs/issues/1507)

## Objective

Optimize tenferro's native CubeCL permutation and materialization path for
WebGPU/Metal without weakening the CUDA vendor-library policy established by
issue #1506.

The work is profile-first:

1. Make the existing, unoptimized native structural kernels measurable on
   WebGPU without changing their algorithms.
2. Add and run a `mac-gpu` permutation benchmark profile.
3. Record the Metal baseline.
4. Only then change native kernel algorithms.
5. Unify transpose and view materialization, share bilateral dimension-fusion
   planning with `strided-perm`, and add a shared-memory tiled transpose whose
   tile dimensions are compile-time parameters.
6. Re-run the Metal profile and investigate PyTorch or JAX implementation
   details if tenferro remains extremely slower on comparable rows.

## Current State

- CUDA `F32`, `F64`, `C32`, and `C64` structural permutation uses cuTENSOR.
  That route remains unchanged.
- CUDA vendor-unsupported dtypes or layouts use native CubeCL structural
  kernels.
- `WebGpuBackend::transpose`, `WebGpuBackend::to_contiguous_read`, and
  `WebGpuBackend::copy_read_into` currently return `Unsupported`.
- `transpose_kernel` decodes both output and input coordinates for every
  element.
- `view_to_contiguous_kernel` decodes a full-rank coordinate for every element.
- Neither native path fuses dimensions or tiles stride-hostile transposes.
- The existing GPU permutation benchmark is CUDA-only and uses A100-sized
  `F64` patterns.
- The development Mac is an Apple M5 Max. The user explicitly approved the
  recorded M5 bounded tile sweep as the final Apple Silicon sweep even though
  the issue originally named M4. No additional M4 rerun is required.

## Entry Gate

Kernel optimization is forbidden until a Metal baseline has been written to
the benchmark repository.

Step 0 may include only the provider wiring required to measure the existing
kernel algorithms:

- compile the existing structural kernel module for both CUDA and WebGPU;
- add WebGPU launch glue for the existing `transpose_kernel` and
  `view_to_contiguous_kernel`;
- implement the WebGPU tensor/view dispatch necessary for the benchmark;
- preserve the current kernel bodies, launch geometry, rank-deep decoding, and
  lack of tiling;
- add correctness coverage for the newly exposed provider route.

The wiring commit and the benchmark baseline commit must precede every
dimension-fusion or tiled-kernel commit. The benchmark report records the exact
tenferro commit used for the baseline.

## Repository Boundaries

### tenferro-benchmark

Owns:

- the `mac-gpu` target profile;
- Metal-appropriate permutation patterns;
- the tenferro WebGPU/Metal runner;
- PyTorch MPS and best-effort JAX Metal reference runners;
- a device-to-device Metal copy ceiling;
- run metadata, validation, formatting, raw records, and latest reports.

### strided-rs

Owns device-independent bilateral fusion planning. It exposes a small planning
API from `strided-perm`; it does not expose the HPTT execution tree or CPU
micro-kernel details.

### tenferro-rs

Owns:

- WebGPU structural tensor integration;
- common native CubeCL permutation launch planning;
- the generic fused-rank materialization kernel;
- the shared-memory tiled transpose kernel;
- runtime-specific launch wrappers and specialization selection;
- placement, aliasing, dtype, shape, and shared-memory validation;
- GPU design documentation and the implementation work log.

## Benchmark Design

### Target Profile

Add `mac-gpu` anywhere target profiles are validated or documented. Keep
`suite_id: gpu/permutation`; hardware belongs in `target_profile`, not the
suite identity.

Latest report:

```text
result/mac-gpu/gpu/permutation.md
```

Raw run:

```text
data/results/mac-gpu/gpu/permutation/<timestamp>/
```

### Dtype and Sizes

The `mac-gpu` profile uses `F32`. WGSL/Metal WebGPU does not provide a portable
`F64` compute path, while PyTorch MPS and the experimental JAX Metal path can
express `F32`.

Preserve the semantic pattern classes from the NVIDIA profile but scale element
counts so steady-state rows are approximately 10 ms on Apple Silicon:

- device copy;
- 2D transpose;
- two 3D permutations;
- rotation;
- high-rank reverse;
- high-rank cyclic permutation;
- TN-representative scattered-to-column-major materialization;
- TN-representative contiguous permutation that should collapse after fusion.

The suite keeps deterministic index-derived values and exact correctness
comparison. Pattern sizes may differ from the A100 profile, but pattern class
and permutation semantics remain explicit.

### Participants

- `tenferro-webgpu-transpose-baseline` before path unification;
- `tenferro-webgpu-to-contiguous`;
- `pytorch-mps`;
- `jax-metal` when a compatible isolated environment works;
- `memcpy-metal-d2d`.

JAX Metal is best-effort. The repository's normal JAX installation is CPU-only,
and current upstream guidance conflicts: Apple's experimental plugin page
remains available while JAX maintainers have described the old plugin as no
longer maintained. A missing or incompatible plugin produces a
`not_configured` row with the reason; it must not silently run on CPU.

### Timing Contract

- Run participants sequentially.
- Warm up before collecting samples.
- Time host dispatch plus backend-native device synchronization.
- Do not download inside timed regions.
- Verify output before timing.
- Report median, p25, p75, effective read-plus-write bandwidth, allocation
  behavior, synchronization method, device name, runtime, dtype, and exact
  tenferro revision.
- Treat the device-copy row as the measured bandwidth ceiling.

## Shared Dimension-Fusion Planner

Expose a small `strided-perm` API conceptually shaped as:

```rust
pub struct BilateralFusionPlan {
    pub dims: Vec<usize>,
    pub src_strides: Vec<isize>,
    pub dst_strides: Vec<isize>,
}

pub fn plan_bilateral_fusion(
    dims: &[usize],
    src_strides: &[isize],
    dst_strides: &[isize],
) -> Result<BilateralFusionPlan, FusionPlanError>;
```

The final naming follows `strided-perm` conventions. The API:

- validates equal metadata lengths;
- removes size-one axes;
- fuses adjacent axes only when both source and destination remain contiguous
  across the boundary;
- checks dimension-product overflow;
- preserves signed strides;
- has no dependency on a device runtime, element type, tile size, or HPTT
  execution mode.

`tenferro-gpu` consumes this plan at the host launch boundary. It does not copy
or independently reimplement the fusion algorithm.

## Native Materialization Architecture

### Unified Route

Delete the dedicated native transpose algorithm after the entry gate.

Native transpose becomes:

1. validate the permutation;
2. form logical output shape and permuted source strides;
3. construct the same host-side materialization plan used by
   `to_contiguous`;
4. launch one of the native materialization specializations.

CUDA cuTENSOR-supported cases continue to bypass this native route. Native
CUDA fallbacks and WebGPU use the same CubeCL kernel definitions with
runtime-specific launch types.

### Plan Classification

The launch plan classifies fused metadata as:

- `LinearCopy`: scalar or fused rank zero/one with contiguous source and
  destination;
- `GenericStrided`: arbitrary fused-rank materialization;
- `TiledTranspose`: a two-dimensional transpose class where source and
  destination stride-one axes differ and both tile axes are representable by
  affine strides.

The generic kernel performs coordinate decoding only across fused dimensions.
The TN contiguous case should reduce from rank 24 to a low-rank or linear
launch.

### Tiled Transpose

The tiled kernel:

- maps a two-dimensional output tile to one workgroup;
- cooperatively reads a coalesced source tile into `SharedMemory`;
- synchronizes the workgroup;
- writes the transposed tile with coalesced destination access;
- checks source and destination edges for partial tiles;
- uses padding where needed to avoid shared-memory bank conflicts;
- never aliases source and destination.

Tile width, tile height, shared-memory leading dimension/padding, workgroup
dimensions, and vector width are compile-time parameters. They are not fixed
inside the kernel body.

The host selects from a small, tested candidate set and rejects candidates
whose shared-memory requirement exceeds
`hardware.max_shared_memory_size`. If no tiled candidate is valid, the
operation uses `GenericStrided`; it never falls back to CPU or transfers data
between providers.

## Validation and Errors

Validate once before launch:

- permutation rank, range, and uniqueness;
- shape/stride metadata length;
- shape products and byte counts;
- view reachable range and base offset;
- source and destination placement and runtime identity;
- destination internal overlap;
- source/destination allocation overlap;
- launch cube-count limits;
- shared-memory requirement for the selected specialization.

Kernels are out-of-place. Source/destination overlap remains a typed boundary
error. Permutation is pure data movement; it introduces no accumulation-order
or determinism policy.

Provider or kernel errors retain their typed source. Unsupported dtype or
layout conditions remain explicit and must not trigger a hidden CPU path.

## Test Strategy

### Planner Tests

Cover:

- scalar and empty metadata;
- size-one axes;
- identity layout;
- partially fused layouts;
- non-fusible layouts;
- 2D transpose;
- high-rank reverse and cyclic patterns;
- TN rank-24 collapse;
- negative strides;
- mismatched metadata lengths;
- multiplication overflow.

### tenferro GPU Tests

Cover:

- transpose and view materialization share the same native planner;
- generic and tiled paths match a host reference;
- boundary tiles;
- scalar, zero-size, size-one, high-rank, and noncompact views;
- all supported WebGPU structural dtypes;
- native CUDA fallback dtypes/layouts;
- alias rejection and foreign-runtime rejection;
- shared-memory-limit fallback;
- simultaneous `cuda,webgpu` feature compilation.

### Cross-Runtime Verification

Development and most algorithmic verification run on Linux A100 using both:

- CubeCL CUDA runtime;
- CubeCL wgpu runtime backed by Vulkan.

The same kernel source must compile and produce matching results on both
runtimes. CUDA numbers are diagnostic because cuTENSOR remains the production
CUDA route for supported dtypes.

Improvement decisions are based on the Step 0 Metal baseline, not on native
CUDA performance. The final Apple Silicon pass sweeps the bounded compile-time
tile candidates and reruns the complete `mac-gpu` profile. The recorded M5 Max
execution is the approved final pass and records the device truthfully; an M4
rerun is not a completion requirement.

## Performance Gates

The campaign's stop-the-line rule applies after the baseline exists:

- any comparable WebGPU row that regresses by at least 20% stops the
  optimization series until explained or fixed;
- correctness, timing scope, dtype, pattern, allocation behavior, and device
  must match before ratios are reported.

Treat tenferro as extremely behind a comparable PyTorch or JAX row when either:

- tenferro median time exceeds the reference by more than 2x; or
- the reference reaches at least 70% of the copy ceiling while tenferro remains
  below 35%.

For every such row:

1. identify the reference dispatch path;
2. inspect the relevant upstream source or generated kernel;
3. compare dimension collapse, copy specialization, launch geometry,
   vectorization, coalescing, tiling, padding, and synchronization;
4. implement applicable changes that fit tenferro's runtime-generic CubeCL
   architecture;
5. rerun correctness and performance measurements.

## Provenance

`strided-perm` remains the owner of the bilateral fusion algorithm and its
existing HPTT provenance. Consuming its public planning API does not duplicate
the implementation.

If PyTorch, JAX, Apple MPS, or another third-party implementation is read while
writing code, add a source comment at writing time that identifies the project,
file, and function and states whether the code is ported, derived, follows a
convention, or was only validated against it. A close translation also carries
the original license and copyright requirements.

Scientific-credit changes to repository citation policy require separate user
confirmation before they are added.

## Commit and Delivery Order

1. Commit this design.
2. Add WebGPU measurement wiring without changing kernel algorithms.
3. Add `mac-gpu` benchmark support and record the baseline.
4. Publish the shared `strided-perm` fusion-planning change and update the
   tenferro dependency revision.
5. Unify the native materialization path and consume the fusion plan.
6. Add the compile-time tiled transpose specializations.
7. Run Linux CUDA plus wgpu/Vulkan verification.
8. Run Apple Silicon tile sweep and final Metal profile.
9. Investigate and address extreme PyTorch/JAX gaps where comparable reference
   rows exist.
10. Update GPU design documentation, work log, and benchmark reports.

Each stage has focused correctness tests and a coherent commit. Benchmark
evidence identifies the exact code revision used.

## Completion Criteria

The issue is complete only when all of the following hold:

- `mac-gpu` profile and Metal baseline exist;
- baseline predates kernel optimization;
- WebGPU transpose and materialization are supported without hidden transfer;
- dedicated native transpose is unified with view materialization;
- bilateral fusion planning is shared with `strided-perm`;
- tiled transpose uses compile-time tile parameters;
- correctness passes on Metal and both Linux runtime paths;
- final Apple Silicon tile sweep and profile are recorded;
- the 20% regression gate passes or every exception is explained;
- every extreme comparable PyTorch/JAX gap has been investigated and applicable
  optimizations have been implemented;
- documentation, provenance, work log, raw records, and latest reports are
  current.
