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
default and the selected final tile. The user approved this recorded M5
bounded sweep as the final Apple Silicon gate; no additional M4 rerun is
required.

Allocation/submission diagnostics explain why the per-call sweep looks flat.
Raw output allocation averaged 0.002 ms and an idle synchronization averaged
0.017 ms. In three 101-iteration passes where warmed dispatches shared one
final synchronization, the median per-transpose times were 0.427 ms for
`16x8-p1-v1` and 0.516 ms for the generic kernel, a 17% throughput
improvement. The selected tile was also the fastest of the bounded candidates;
the next-best `32x8-p1-v1` measured 0.432 ms.

The selected kernel's queued throughput is effectively the same as the
0.422 ms Metal device-copy reference and is faster than the allocation-matched
PyTorch MPS end-to-end median. The remaining synchronized single-call latency
comes from encoding and submitting one CubeCL/wgpu command buffer per call,
not from output allocation or an inferior transpose kernel. The profile keeps
per-call synchronization to preserve its public-API latency contract.

## Batched-tile follow-up

The remaining `mac_transpose_3d_102` framework gap motivated an experiment
that classified `[1,0,2]` as a batch of compact 2D transposes and mapped the
batch index to the CubeCL dispatch Z dimension. A partial-edge, multi-batch
Metal correctness test passed, and CUDA plus WebGPU compiled from the same
kernel.

The first per-call measurement showed no material difference because command
submission dominated: tile measured 1.726 ms and generic measured 1.729 ms.
After the queue-throughput diagnostic exposed that masking effect, three
warmed 101-iteration comparisons were repeated with one final synchronization.
The batched tile measured 0.414, 0.401, and 0.405 ms per transpose; generic
measured 0.495, 0.471, and 0.490 ms. The 0.405 versus 0.490 ms medians are a
17% kernel-throughput improvement, so the earlier revert was superseded and
the batched compact-transpose specialization was reinstated.

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

- Metal/WebGPU integration: 86 tests passed.
- `tenferro-gpu` WebGPU clippy with warnings denied: passed.
- Combined `cuda,webgpu,cpu-faer` compile: passed.
- M5 profile correctness: all participating tenferro patterns passed.
- Linux A100 CUDA native structural runtime: 27 passed, 0 failed. A direct
  native `I32` batched partial-tile diagnostic for shape `[17, 19, 3]` and
  permutation `[1, 0, 2]` matched the complete column-major reference.
- Linux A100 wgpu/Vulkan structural runtime: 6 passed, 0 failed. The Vulkan
  inventory contained only the NVIDIA A100 80GB PCIe through the NVIDIA ICD,
  excluding software and CPU adapter fallback.
- Final Apple Silicon bounded tile sweep: passed on the approved M5 Max;
  `16x8-p1-v1` remains selected and no M4 rerun is required.

The A100 environment, commands, and full results are recorded in
[issue comment 5112907691](https://github.com/tensor4all/tenferro-rs/issues/1507#issuecomment-5112907691).
