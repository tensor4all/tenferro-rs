# Issue #1891: exact complex CUDA permutations through the real view

## Summary

The CUDA erased `copy_read_into` routed `F32`/`F64`/`C32`/`C64` through the
cuTENSOR permutation executor (#1522), which computes `alpha * permute(A)` with
`alpha = 1`. For a complex dtype that multiply is
`re = 1*re - 0*im`, `im = 1*im + 0*re`, so `0 * inf = NaN`: copying
`(inf, 2.0)` produced `(inf, NaN)` and a finite component was lost. The typed
`copy_into` entry, which uses a plain assignment kernel, was exact, so the two
surfaces disagreed.

A first attempt routed `C32`/`C64` to the exact native copy. That fixed the
value loss but left a multi-axis permutation destination 24x slower than the
vendor path, because that layout is outside the repository's tiled transpose
kernel.

## Decisions

1. **Plan a complex operand through its real view.** A complex scalar is two
   adjacent real scalars, so a `Complex32`/`Complex64` operand is exactly the
   real tensor of shape `[...shape, 2]` with doubled leading strides and a
   unit-stride trailing axis. `real_view_operand` appends that axis with a fresh
   mode id for both operands, `DATA_TYPE` becomes the corresponding real type,
   and `execute_permutation` passes a real `alpha = 1`. The permutation moves
   the same bytes, but the scaling multiply is a real multiply, which is exact
   for every value. This keeps the vendor path for complex, so no layout pays
   the native kernel's uncoalesced cost.

2. **Select the tiled transpose kernel in the native copy.** Independently of
   the above, `copy_view_to_view_typed` fell back to
   `contiguous_to_view_kernel` / `strided_to_strided_kernel` for a transposing
   destination, which are uncoalesced. A copy from a compact source into a
   row-major compact destination view is physically the same operation as
   materializing the transposed source into a compact column-major destination,
   so `transpose_copy_plan` builds a `NativePermutationPlan::for_transpose` plan
   for that narrow layout (rank 2/3, destination offset zero, destination
   strides exactly row-major compact) and reuses `launch_native_materialization`.
   That is the path the typed `copy_into` entry and every reversed or broadcast
   source takes.

Integer and `Bool` copies are unaffected: `CutensorPermutationScalar` is
implemented for the four float and complex types only, so an integer operand
never reaches the vendor path.

## Measured results (release profile)

A100 80GB PCIe, CUDA 12.6, cuTENSOR from the system path, warm plans, median of
7. Release, because the copy entry point has non-trivial host-side work and a
dev-profile build misreports these paths by roughly an order of magnitude.

| case | complex descriptor (value-corrupting) | native | real-view vendor |
|---|---:|---:|---:|
| `C64` `[8192, 8192]` transposing | 1.35 ms | 1.34 ms (tiled) | **1.34 ms** |
| `C64` `[1024, 1024, 512]` permuted `[1, 2, 0]` | 10.6 ms | 254 ms | **31.2 ms** |
| `F64` `[1024, 1024, 512]` permuted `[1, 2, 0]` | 5.36 ms | 84.4 ms | 5.36 ms |

The reference bound is measurable: the vendor path reaches 1.62 TB/s on the
`C64` permutation and 1.60 TB/s on the `F64` one, i.e. the A100 HBM limit,
while the native kernel reaches 66 GB/s. So the gap was never "cuTENSOR is
unusually fast"; it is the native kernel's uncoalesced access pattern.

The tiled transpose plan drops the native transposing copy from 3.40 ms to
1.34 ms (`C64`) and from 2.77 ms to 0.73 ms (`F64`), which is what the typed
`copy_into` and native fallbacks now use. PyTorch 2.9.1 `copy_` on the same
device is 0.98 ms (`C64` transposing) and 0.97 ms (`F64`).

Exactness (`C64`, identical and transposing destinations): `(inf, 2.0)`,
`(-0.0, 3.0)`, `(5.0, inf)` and a `NaN` component are preserved bit for bit and
match the CPU reference. On the unmodified parent commit the same test fails
with `(inf, NaN)`.

## Rejected alternatives

- **Routing complex copies to the exact native copy.** Correct, but a multi-axis
  permutation destination costs 254 ms against the vendor path's 10.6 ms,
  because the tiled layout does not cover it. Replaced by the real-view plan.
- **Removing the vendor path for `F32`/`F64`.** Those dtypes never lost a value
  (`alpha = 1` is exact for reals; only a `NaN` payload can change), and the
  native permutation kernel is 16x slower, so there is nothing to gain.
- **A layout- or size-dependent crossover between the native and vendor copy.**
  The crossover is not machine-independent, and choosing a kernel per size is
  the machine-specific guess that issue #1887 rejects for zero fill.
- **A real-view descriptor that doubles one axis extent instead of adding an
  axis.** It avoids the extra mode but only when the permutation maps the
  logical stride-1 axis onto the destination's stride-1 axis, which excludes
  exactly the transposing copies this change exists for.

## Residual

The real-view plan costs 2.9x on the `C64` multi-axis permutation (10.6 ms to
31.2 ms) even though it is 8x faster than the native kernel there. Accepted:
correctness first, and the vendor path also reaches the bandwidth limit on the
transposing layouts that motivated #1522. A tiled native permutation kernel
would remove both gaps and is the natural follow-up.

## Native permutation kernel: implemented, measured, not shipped

A native replacement for the vendor permutation was attempted and abandoned on
measurement. Recorded here so it is not repeated blindly.

What was built:

- `tiled_transpose_kernel` took the source's slow-axis and batch strides as
  explicit parameters instead of assuming a compact source, and
  `tiled_transpose_eligible` was relaxed to "the plan is a transpose between the
  two unit-stride axes".
- The tile axes were moved into a one-dimensional grid ordered with the
  destination tiles varying fastest, so that a source-fast extent larger than
  the 65535 CUDA allows for `gridDim.y` is representable and adjacent blocks
  write adjacent destination tiles.
- `copy_view_to_view_typed` built the plan for any destination view that is a
  compact layout under some axis order, not only a row-major compact one.

The generalized path is correct (its `C64` `[4, 3, 2]` permuted case matches the
CPU reference) but slow, and it is slower than the kernel it replaces:

| case | previous native | generalized native | real-view vendor |
|---|---:|---:|---:|
| `C64` `[1024, 1024, 512]` permuted `[1, 2, 0]` | 84.8 ms | 229 ms | 31.2 ms |
| `F64` same | 84.3 ms | 115 ms | 5.37 ms |

Diagnosis, from `C64` `[rows, 512]` transposed into a row-major compact view
(destination plan `[512, rows]`), release, A100:

| rows | bytes moved | throughput |
|---:|---:|---:|
| 32768 | 1.07 GB | 1435 GB/s |
| 262144 | 8.6 GB | 74 GB/s |
| 1048576 | 8.6 GB | 76 GB/s |

The same access pattern is at bandwidth for a small allocation and collapses by
19x once the working set outgrows a cache/TLB capacity. The plan is a transpose
whose two axes have strides `1` and `rows`, so one direction of the tile is
always a 256-byte burst at a 16 MiB stride: sweeping the destination compactly
necessarily touches one 2 MiB page per source row. cuTENSOR reaches 1.6 TB/s on
the same operation, so it must order its work to keep the resident footprint
small; a native replacement needs that ordering (for example a walk over the
batch axis that keeps both operands inside a window) rather than a tile-shape
change, since every `TENFERRO_NATIVE_TRANSPOSE_TILE` configuration measured the
same, as did the generic strided-materialization kernel.

## Follow-up: the exact complex permutation reaches the native ceiling

The fused plan for `C64 [1024, 1024, 512]` permuted `[1, 2, 0]` is
`dims = [1048576, 512]` with a unit-stride source axis and a unit-stride
destination axis, so it is exactly one 2D transpose. Two things kept it off the
tiled kernel:

- routing: every numeric dtype went to the vendor permutation, and for a complex
  operand that plan describes the buffer through its real view, whose unit-stride
  run is the 16-byte real/imaginary pair. Measured on the A100, that costs 2.8x:
  the same copy runs at 5.51 ms in `f64` (1.56 TB/s) but 30.96 ms in `C64`
  (0.555 TB/s), and keeping the complex unit-stride axis inside the plan instead
  of moving it measures the same 31.37 ms, so no descriptor reordering recovers
  it.
- the grid limit: the source-fast extent needs 65536 blocks at the default
  16-wide tile, one past the per-dimension launch limit, so the launch fell back
  to the flat pass instead of the tiled kernel (84.9 ms, which is the earlier
  "native is slower than the vendor" measurement).

The fix routes a copy whose fused plan is one row-major-source/column-major-
destination matrix to the tiled kernel (complex operands prefer it over the
real-view vendor plan), widens the tile to 32 when the grid limit needs it (and
falls back when the shared-memory budget does not allow it), and takes the grid
extents from the kernel's own destination-fast/source-fast roles rather than
from the plan's axis order, which the mirrored orientation reverses.

Measured on the same A100, `C64 [1024, 1024, 512]`, 17.18 GB of traffic:

| permutation | before | after | bandwidth |
|---|---:|---:|---:|
| `[1, 2, 0]` | 31.37 ms | **10.61 ms** | 1.62 TB/s |
| `[2, 0, 1]` (mirrored) | — | **10.59 ms** | 1.62 TB/s |
| `[1, 0, 2]` (three-axis plan, vendor) | 30.95 ms | 30.98 ms | 0.555 TB/s |
| `[8192, 8192]` transpose | 1.34 ms | 1.43 ms | 1.64 TB/s |
| `F64 [8192, 8192]` (vendor, control) | 0.73 ms | 0.77 ms | 1.60 TB/s |

10.61 ms is the HBM ceiling at this traffic, so the exact path now matches the
value-corrupting complex descriptor's speed. Permutations whose fusion keeps
three or more axes (`[0, 2, 1]`, `[1, 0, 2]`, `[1, 0, 3, 2]`) still take the
vendor real-view plan; a genuinely multi-axis shuffle would need the windowed
multi-axis schedule sketched below.

## Verification

- New CUDA test
  `cuda_runtime_copy_read_into_preserves_non_finite_complex_components` compares
  bits for `(inf, x)`, `(-0.0, x)`, `(x, inf)` and a `NaN` component on identical
  and transposing destinations, and matches the CPU reference. It fails on the
  parent commit.
- All 21 CUDA copy tests pass on an A100, including the #1522 destination-reuse
  benchmark and the offset strided region cases for `F32`, `F64`, `C32`, `C64`,
  `I32` and `I64`.
- Not measured: any device other than the A100 above, and HIP/WebGPU.
