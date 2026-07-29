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

An allocation/synchronization diagnostic explains why the per-call sweep
looks flat. Raw output allocation averaged 0.002 ms. When 51 dispatches shared
one final synchronization, the selected tile averaged 1.153 ms per transpose
versus 1.431 ms for the generic kernel, a roughly 20% throughput improvement.
The profile intentionally keeps per-call synchronization for comparable
public-API latency, where the fixed flush/synchronization cost masks most of
that kernel gain.

## Rejected batched-tile experiment

The remaining `mac_transpose_3d_102` framework gap motivated an experiment
that classified `[1,0,2]` as a batch of compact 2D transposes and mapped the
batch index to the CubeCL dispatch Z dimension. A partial-edge, multi-batch
Metal correctness test passed, and CUDA plus WebGPU compiled from the same
kernel.

With 20 warmups and 101 measured iterations, however, the selected tiled path
measured 1.726 ms for transpose and 1.731 ms for view materialization. The
generic path measured 1.729 ms and 1.733 ms, respectively. This is not a
material improvement under the profile's public-API allocation and
synchronization contract, so commit `585c28dd` was reverted by `5586394f`.
The two-dimensional tiled specialization remains unchanged.

## Rejected native-vector experiment

The configured tile vector width originally unrolled scalar lanes. An
experimental kernel instead bound source and destination arrays as CubeCL
`Vector<E, N>` values, performing native vector loads and stores while keeping
the same shared-memory transpose. Metal correctness passed for vector widths
1, 2, and 4.

With 10 warmups and 51 measured iterations on `mac_transpose_2d`, the
`32x8-p1` transpose medians for vector widths 1, 2, and 4 were 1.740 ms,
1.727 ms, and 1.726 ms. View materialization measured 1.733 ms, 1.735 ms, and
1.722 ms. These differences are within run-to-run noise, so the native-vector
experiment was not retained.

## Framework-gap inspection

The PyTorch comparison exceeded the 2x inspection threshold for the baseline
2D case. PyTorch's current MPS `Copy.mm` routes strided views through its unary
copy machinery. `OperationUtils.mm` uses a 2D strided dispatch, detects
inner-contiguous layouts, and can move aligned 16-byte chunks. The benchmark
was then corrected so PyTorch, like tenferro, allocates a fresh destination on
every timed call. Focused 51-iteration medians under the matched allocation
contract were 0.552 ms for the 2D case and 1.037 ms for `mac_transpose_3d_102`.
The remaining framework gap is an end-to-end host-API comparison, not an
isolated transpose-kernel comparison. The tenferro kernel was independently
implemented; no PyTorch source was copied.

## Verification status

- Metal/WebGPU integration: 84 tests passed.
- `tenferro-gpu` WebGPU clippy with warnings denied: passed.
- Combined `cuda,webgpu,cpu-faer` compile: passed.
- M5 profile correctness: all participating tenferro patterns passed.
- Linux A100 CUDA plus wgpu/Vulkan runtime execution: pending. Configured A100
  SSH endpoints were unreachable from the development session because the
  internal network/VPN and DNS were unavailable.
- Final Apple M4 tile sweep: pending and required for the release judgment.
