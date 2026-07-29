# Issue #1507: CubeCL native permutation on wgpu/Metal

Date: 2026-07-29

## Entry gate

Kernel work started only after the `mac-gpu` benchmark profile produced a
correctness-checked Metal baseline. The baseline was collected on Apple M5 Max
as the approved development substitute. JAX Metal was not configured and was
recorded as such; CPU fallback was not used.

The baseline 2D transpose median was 1.472 ms for the direct WebGPU path and
1.481 ms for view materialization. PyTorch MPS measured 0.664 ms, while the
Metal device-copy reference was 0.469 ms.

## Implementation

- `strided-perm` now exposes the validated bilateral dimension-fusion planner
  used by HPTT.
- CUDA-native and WebGPU transpose/materialization routes consume one
  `NativePermutationPlan`.
- The planner validates shape products, allocation bounds, destination
  non-overlap, source offsets, and compact destination layout before launch.
- Generic materialization uses one signed affine kernel. Its logical length is
  compile-time metadata because raw `Array` runtime length metadata was
  observed to alias a logical dimension on Metal for a 3D case.
- Exact compact 2D transpose uses shared memory. Tile size, block rows, padding,
  and vector width are all compile-time kernel parameters.
- `TENFERRO_NATIVE_TRANSPOSE_TILE=generic` forces the generic path. Bounded
  sweep values include 8, 16, and 32-wide tiles with vector widths 1, 2, or 4.

## M5 development sweep

The `mac_transpose_2d` transpose medians were:

| configuration | median (ms) |
| --- | ---: |
| `generic` | 1.469 |
| `8x8-p1-v1` | 1.481 |
| `16x8-p1-v1` | 1.463 |
| `16x8-p1-v2` | 1.465 |
| `32x8-p1-v1` | 1.471 |
| `32x8-p1-v2` | 1.479 |
| `32x8-p1-v4` | 1.483 |

All values are within 0.02 ms because the profile includes fresh output
allocation and explicit synchronization. `16x8-p1-v1` is the development
default; it is not a substitute for the final M4 sweep.

## Framework-gap inspection

The PyTorch comparison exceeded the 2x inspection threshold for the baseline
2D case. PyTorch's current MPS `Copy.mm` routes strided views through its unary
copy machinery. `OperationUtils.mm` uses a 2D strided dispatch, detects
inner-contiguous layouts, and can move aligned 16-byte chunks. The benchmark
also reuses a PyTorch destination while tenferro allocates a fresh output per
call. This explains why the reported framework gap is not an isolated
transpose-kernel comparison. The tenferro kernel was independently implemented;
no PyTorch source was copied.

## Verification status

- Metal/WebGPU integration: 84 tests passed.
- `tenferro-gpu` WebGPU clippy with warnings denied: passed.
- Combined `cuda,webgpu,cpu-faer` compile: passed.
- M5 profile correctness: all participating tenferro patterns passed.
- Linux A100 CUDA plus wgpu/Vulkan runtime execution: pending. Configured A100
  SSH endpoints were unreachable from the development session because the
  internal network/VPN and DNS were unavailable.
- Final Apple M4 tile sweep: pending and required for the release judgment.
