# CUDA write-side views (#2223 item 3)

## Decisions

- One crate-private write boundary, `CudaEngine::write_region`, mirrors the read boundary
  `read_region`: an owned destination is the compact, offset-zero region of its own allocation,
  and a borrowed `TensorWrite::View` is the region it names. Write injectivity is not re-proved
  there — every `TypedTensorViewMut` constructor validates the layout with
  `TensorLayout::validate_mutable_no_overlap`, which rejects two logical elements aliasing one
  physical element — so the boundary documents that dependency instead of adding a second check.
- `copy_read_into` reads the source through its own layout and writes the destination region in
  place, staging neither side. It picks the catalogue entry from the two regions: `CopyExact`
  when both are one contiguous span at their allocation base, `StridedToStrided` otherwise. That
  keeps the eager einsum output copy (compact result into a compact or flat-view destination) on
  the exact copy it already used, and only a genuinely strided side pays the layout parameters.
- `elementwise_read_into` cannot address a destination layout at all: `plan_elementwise`,
  `plan_unary` and `plan_binary` take only the element count. A view destination therefore
  computes into a compact scratch of the destination's own shape and `ContiguousToView` scatters
  it once. The scratch is the operation's required output, not operand staging, and the
  destination region and every operand are validated first, so a refused call allocates no
  destination-sized scratch and stages no operand.
- `dot_general_read_into` validates dtype and shape against the destination with the shared
  `validate_dot_general_read_into`, contracts into a compact scratch, and scatters it with
  `ContiguousToView`; cuTENSOR is handed one destination buffer.
- `dot_general_read_into_accum` materializes the destination region into the scratch
  (`MaterializeStrided`) only when `beta != 0`, then runs the existing contraction with that beta
  and scatters back. `beta == 0` never reads the destination, so a NaN there cannot survive as
  `beta * NaN`; an empty contraction keeps the same beta semantics because `submit_dot` scales or
  zeroes the scratch exactly as it does for an owned destination.
- `scale_tensor_write` scales a view in place with `CombinedStrided`'s `MUL` over the region's own
  layout, with the factor as one element broadcast over the destination extents (zero strides).
  The ABI permits `out` to alias an input whose layout is identical, so the region is both the
  operand and the destination and there is no copy-out/copy-back. A unit factor is skipped inside
  the two scale kernels, after the write boundary has accepted the target, so a host or foreign
  view is refused whatever the factor.
- Rank ceiling: a writable **view** wider than `t4a_cuda_kernels::MAX_RANK` is a typed refusal,
  because none of the three write entries wires an `_any` symbol with device-resident geometry.
  `write_region` states it once and marks it `// ponytail:`. Owned destinations keep their
  any-rank paths, and a non-flat borrowed *source* above that rank keeps the read boundary's
  general-path materialization plus the exact copy.

## Verification conclusions and constraints

- Device coverage: `seam_structural` drives `copy_read_into`, `elementwise_read_into` (binary and
  unary arms) and `scale_tensor_write` on transposed, offset and negative-stride destination
  regions of one `6 x 5` sentinel root, plus a compact-offset destination with a reversed offset
  source, a flat destination and an empty destination; `seam_dot` drives `dot_general_read_into`,
  `dot_general_read_into_accum` with `alpha = 0.5, beta = 2`, a zero-beta accumulation over a
  destination pre-filled with NaN, and an empty contraction with `beta = 3`, `beta = 0` and the
  overwrite form. Each test compares the whole root against a host reference computed from the
  region's layout, so the 24 entries outside the 6-entry region must be unchanged.
- A100 results after the review follow-up: `tenferro-gpu` lib 11, `seam_copy` 7, `seam_dot` 12,
  `seam_structural` 32 pass; the earlier `seam_contraction` 9 and `tenferro-linalg` integration
  64-test run cover entries this follow-up does not touch. Clippy `--all-targets -D warnings`
  clean. The CUDA partition matches the nextest inventory (1253).
- No benchmark row. The view destinations were typed refusals before this change, so there is no
  before/after timing to publish for them, and the owned forms keep `CopyExact`, the flat
  elementwise entries, the direct cuTENSOR destination and `binary`'s `MUL`. One previously
  served call does change route: a borrowed source copied into an owned destination used to be
  materialized into a compact operand and now copies layout-aware through `StridedToStrided`.
  The one added device cost is bounded by the destination's own element count (a scratch and one
  scatter), which is the required output write, not a temporary of a larger shape.
