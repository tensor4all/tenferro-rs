# CUDA strided data movement and destination reset (issues #1832, #1834)

## Decisions

- `copy_into` / `copy_read_into` describe both operands by their own extents,
  strides, and offsets. On the cuTENSOR path the source now uses its view
  strides with identity modes, the same descriptor shape the allocating
  materialization path already used; the destination keeps its physical
  stride-order descriptor. The native path keeps the tuned compact-source
  kernel and adds one strided-to-strided kernel for every other layout, so a
  block region inside a larger allocation moves in one pass instead of
  requiring a canonicalizing copy first.
- Reversed (negative-stride) and broadcast (stride < 1) sources stay on the
  native kernel. cuTENSOR 2.x descriptors require positive strides, so this is
  a layout the vendor path cannot represent, not a missing-library fallback.
- cuTENSOR operand descriptors now advertise the alignment the *resolved*
  pointer guarantees instead of the 256-byte allocation alignment. A region
  view folds its element offset into the pointer, and claiming more alignment
  than that address satisfies makes cuTENSOR pick a vectorized kernel that
  fails the launch with `misaligned address`. This was already reachable for an
  offset destination view before this change.
- `axpby_read_into_accum` keeps the documented compact-destination contract and
  serves an arbitrary-stride or offset `x` with a native strided-source kernel
  rather than the previous silent `to_contiguous_read` of `x`. A compact `x`
  still runs the single cuBLAS `geam` call.
  - Rejected: a cuTENSOR elementwise-binary path (`alpha*A + gamma*C -> C`).
    It would be the vendor-library shape, but it needs a new plan/descriptor
    FFI surface for one BLAS-1 layout the native kernel covers in one pass.
  - Deferred: a strided *destination*. The shared
    `validate_axpby_read_into_accum` contract requires a compact injective
    destination, and the CPU path consumes dense slices
    (`strided-basic::axpby_accum`), so relaxing it is a cross-backend contract
    change rather than a CUDA-side addition.
  - The coefficients reach the kernel as a two-element device array, not as
    kernel scalars: the pinned CubeCL CUDA dialect panics while sizing a
    complex scalar kernel parameter (`SizedInfoField::padded_size` divides by a
    zero packing size). The array form is the same explicit device-constant
    boundary the in-place scaling kernels already use.
- `Session::fill_zero_write` fills a compact destination (including an offset
  region view) with one stream-ordered `cuMemsetD8Async` and a strided region
  with one native fill kernel. The memset needs no kernel compilation, is
  dtype-agnostic, and writes exact `+0.0` bits without reading the previous
  contents, which is the property `0 * y` and `scale_tensor_write(y, 0.0)`
  cannot provide. `alloc_zero_output` now shares that primitive.
- A strided `Bool` destination returns a typed `Unsupported` error, matching
  the existing CUDA `Bool` materialization and `copy_into` gaps, instead of
  binding a `u8` fill kernel for a dtype whose other data movement is also
  absent.

## Verification conclusions and constraints

- A100 (CUDA 12.6, cuTENSOR 2.5.0): full `tenferro-gpu --features cuda`
  `--ignored` suite passes, including new coverage for offset strided region
  copies (F32/F64/I32/I64/C32/C64), a rank-3 permuted cuTENSOR destination,
  offset strided materialization, strided-source `axpby` against the CPU
  backend, and `fill_zero_write` (NaN/Inf/-0.0 to `+0.0` bit patterns, strided
  regions, untouched surroundings, allocation accounting).
- The strided-source `axpby` data movement is pinned by a `cfg(test)` pass
  counter, not by device allocation accounting: a materialized `x` is allocated
  and released inside the call, so an active-allocation probe cannot see it.
  `fill_zero_write` does use the allocation probe, because a zero-source or
  staging buffer would have to stay alive across the call.
- `tenferro-fft`'s `cuda_traced_missing_extension_module_is_an_explicit_error`
  fails on this branch and on unmodified `origin/main`; it is unrelated to this
  change.
