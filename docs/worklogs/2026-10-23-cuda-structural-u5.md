# U5, first slice: the tensor surface on the new engine

Date: 2026-10-23. Issue: [tenferro-rs#2208](https://github.com/tensor4all/tenferro-rs/issues/2208)
unit U5, first slice.

## What landed

- `cuda_next::CudaEngine` now implements `TensorStructural`, so the tensor layer addresses
  the new engine through the backend-neutral trait rather than a private typed call:
  - `to_contiguous_read` — a device tensor of this runtime becomes a fresh compact device
    allocation; a host tensor takes the trait's own host canonicalization (no transfer).
  - `copy_read_into` — a same-placement compact copy in stream order.
  - `cast` — dtype conversion on the device through `t4a_cuda_convert`.
- `Dispatcher::copy_exact` — the movement entry as a dispatcher method, alongside the
  unary/binary/convert/gather/reduce family.
- Every other structural operation (`transpose_read`, `reshape_read`,
  `broadcast_in_dim_read`, `extract_diagonal`, `embed_diagonal`, `tril`, `triu`) answers a
  typed `Unsupported` naming itself. None of them falls back to a host round trip, so a
  caller cannot accidentally pay an implicit transfer where the old path did not.
- `alloc_tensor` gives the structural slice a zeroed device allocation with the runtime's
  allocation-domain identity, which is what every later slice of the surface needs.

The slice serves `F32` and `F64`, matching the dtype scope U1 recorded as an accepted,
documented gap.

## Evidence

`tests/seam_structural.rs` on the local A100 (driver 580.126.09, CUDA 12.6):

- `a_device_tensor_materializes_into_fresh_storage` — a 64-element f64 tensor materializes
  into fresh device storage, both tensors stay on the device, and the values read back.
- `a_host_tensor_is_duplicated_without_device_work` — a host tensor takes the host
  canonicalization and stays host-resident.
- `copy_read_into_overwrites_the_destination` — a 32-element copy reaches the destination.
- `a_cast_converts_on_the_device` — f64 to f32 conversion matches the host cast.
- `the_unported_families_refuse_instead_of_transferring` — all six unported operations
  return their typed refusal.

## Why this ordering

cuSOLVER and cuFFT — the rest of U4 — live in `tenferro-linalg` and `tenferro-fft` and are
driven through the old engine's `ExecutionEngine` surface, so they cannot move before the
tensor surface they are called through exists. The structural slice is the first piece of
that surface, which is why it precedes them.

## Second slice (this change): the movement and broadcast reads

- `transpose_read` — the output shape is the input's permuted, and the strided view over the
  source reads each output axis from the axis the permutation names; one
  `t4a_cuda_materialize_strided` gathers it into a compact destination. This is the same
  fold the ABI rev 10 work chose for `reverse`: a permutation is a strided movement, not a
  second kernel family.
- `reshape_read` — a compact copy taken **at the new shape** (a compact copy does not care
  about shape, and a metadata-only `reshape_view` cannot become an owned tensor), with the
  equal-element-count check before any allocation.
- `broadcast_in_dim_read` — `t4a_cuda_broadcast_in_dim` with the destination compact and the
  input's own extents and strides.
- `Dispatcher::copy_exact` gained an optional destination shape through
  `CudaEngine::copy_to_fresh`.

Evidence, same host-reference style on the A100:

- `a_transposed_read_matches_the_host_transpose` — a 3x4 matrix read as its 4x3 transpose.
- `a_reshape_read_relabels_the_same_elements` — 3x4 to 2x6 keeps the elements in order, and
  a 5x5 reshape is refused.
- `a_broadcast_in_dim_read_matches_the_host` — a 3-vector broadcast over the second axis of
  a 3x2 output.

Two facts worth recording:

- `TensorValue::reshape_view` produces a view, and `into_tensor` refuses a metadata-only
  view, so an owned tensor with new metadata must be a fresh allocation, not a relabel.
- `Tensor::strides()` is not on the dtype-erased tensor; strides come from the typed payload.

## Third slice (this change): the diagonal and triangular families

- `extract_diagonal` — the output is the input's shape with the second axis removed and the
  first axis shortened to the smaller of the two; one `t4a_cuda_extract_diagonal` into the
  compact destination, which the planner requires.
- `embed_diagonal` — the input's axis is re-inserted at the requested position, and the
  destination is the zeroed allocation, because only the diagonal positions are written.
- `tril`/`triu` — `t4a_cuda_tril`/`t4a_cuda_triu` over a rank-2-or-higher compact tensor,
  with higher axes batching the mask.

One ABI fact: `Arg` still has no signed 64-bit scalar, and the mask offset `k` may be
negative, so it travels as raw bytes of type `i64` (`pod_arg(&k)`) rather than `Arg::U64`,
which is only bit-identical for positive values.

Evidence, same host-reference style on the A100:

- `a_diagonal_extract_and_embed_round_trip` — a 3x3 matrix's diagonal extracts to `[1, 5,
  9]`, re-embedding gives the zero-filled matrix with only those positions written, and two
  equal axes are refused.
- `the_triangular_masks_match_the_host` — `tril` and `triu` over a 3x4 matrix matched element
  by element, and a rank-1 tensor is refused.

With this, every `TensorStructural` method is implemented on the new path for F32/F64; the
only remaining refusals are outside that dtype scope and for borrowed views.

## Fourth slice (this change): the reduction family

`TensorReduction` is implemented for all five operations (`sum`, `prod`, `max`, `min`,
`sum_squares`), served by the U4 reduction dispatcher:

- Axes are deduplicated and sorted, then reduced **one axis at a time, largest first**, so
  each pass sees the shape the previous one produced.
- The reduction kernel is **keepdims**: the destination has the input's rank with extent one
  on the reduced axis. The unit axes are therefore squeezed off by a final compact copy,
  which is the only extra work a multi-axis reduction pays.
- No axes is the identity copy; an axis outside the rank is a typed refusal.
- The strategy (unit/plane/split) is the crate's own policy — `reduce_auto` — so a split
  reduction's two passes run on the runtime's stream in order.
- The partials workspace is one allocation of the input's element count, which is the
  crate's own cap, shared by every pass.

Evidence: `the_reductions_match_the_host` on the A100 — axis-1 sum of a 3x4 matrix, a
two-axis sum to a scalar, axis-0 max, axis-1 sum of squares, the no-axes identity, and an
out-of-range axis refused.

## Fifth slice (this change): the elementwise family

`TensorElementwise` for the eleven operations the arithmetic and unary kernels already
serve:

- `add`, `sub`, `mul`, `div` — the `t4a_cuda_add`/`sub`/`mul`/`div` entries (division only
  for F32/F64, which is the ABI's own dtype list for that entry).
- `neg`, `conj`, `abs`, `sign` — `t4a_cuda_unary` with the matching operation.
- `maximum`, `minimum` — `t4a_cuda_binary` with `Max`/`Min` and both operands as buffers.
- `clamp` — `t4a_cuda_clamp` with **per-element** lower and upper operands, which is the ABI's
  shape (the scalar form is the caller's broadcast).
- `compare_read`, `select_read` (boolean destinations/predicates), `rem`/`rem_read` (checked
  remainder carries a status word) and `elementwise_read_into` (session-shaped) answer typed
  refusals. Those four are the remaining pieces of the family.
- New dispatcher methods: `arithmetic` (one elementwise entry over three operands) and
  `clamp_buffers` (four operand buffers).
- The binary entry always takes a status word for its checked-integer route; a comparison of
  two full buffers uploads a zeroed one and never reads it back.

Evidence: `the_elementwise_family_matches_the_host` on the A100 covers `add`, `sub`, `mul`,
`div`, `neg`, `abs`, `sign`, `maximum`, `clamp` with per-element bounds, and the refusal of a
shape mismatch.

One convention worth recording: the kernel's `sign` is the XLA convention — `sign(0) = 0` —
while `f64::signum(0)` is 1. The first version of the test used `signum` and caught it.

## Sixth slice (this change): comparisons, selection and boolean storage

- `compare_read` writes the boolean result of `t4a_cuda_compare`; `CompareDir` maps one to
  one onto the kernel's `CompareOp`.
- `select_read` runs `t4a_cuda_select` with a boolean predicate and two same-shaped,
  same-dtype branches; a non-boolean predicate is a typed refusal.
- The transfer route now serves `Bool` in both directions, and the structural slice's scalar
  macro covers it, because a boolean tensor can now exist on the device at all. Before this,
  a comparison's result could not even be read back.

Evidence: `compare_and_select_round_trip_through_a_boolean_tensor` on the A100 compares eight
f64 values (predicate matched element by element), selects between a +10 and a -10 branch, and
refuses a non-boolean predicate.

The remaining elementwise pieces are `rem`/`rem_read` (the checked integer remainder carries a
status word and needs a read-back) and `elementwise_read_into` (session-shaped).

## Seventh slice (this change): the last elementwise pieces

- `rem`/`rem_read` — floats use the plain `Rem` entry; integers use `RemChecked`, whose
  divide-by-zero flag is a device word that is read back and reported as a typed error rather
  than silently producing a zero. Complex remainder is refused (it is not defined).
- `elementwise_read_into` — the session-shaped route, for the six operations the enum names:
  `Add`/`Subtract`/`Multiply` through the arithmetic entries, `Divide` through `t4a_cuda_div`,
  `Negate`/`Conj` through `t4a_cuda_unary`. It writes straight into the caller's allocation,
  so no intermediate tensor is created.
- The scalar macro and the dtype mapping now also carry `I32`/`I64`, which the integer
  remainder needs.

Evidence: `remainder_and_read_into_match_the_host` on the A100 covers the floating remainder,
the integer remainder (`[7,-7,7,7] % [2,2,-2,3] = [1,-1,1,1]`), the checked divide-by-zero
refusal, reading into an existing allocation for both `Add` and `Negate`, and the
operand-count refusal.

With this, every `TensorElementwise` method is implemented on the new path. One limitation is
recorded: `elementwise_read_into` requires every operand to share the destination's shape, so
the scalar-operand forms are refused rather than broadcast.

## Next slices of U5

`TensorIndexing` (needs the windowing family — `slice`, `dynamic_slice`, `pad` — that U2 left
out), `TensorDot`, and the `ExecutionEngine` session wiring that eager/traced/AD drive.
