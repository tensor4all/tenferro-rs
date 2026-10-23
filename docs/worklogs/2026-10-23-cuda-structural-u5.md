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

## Next slices of U5

`transpose_read` (a permuted `materialize_strided`), `reshape_read` (compact copy plus
metadata), `broadcast_in_dim_read` (`t4a_cuda_broadcast_in_dim`), the diagonal and
triangular families, then `TensorElementwise`, `TensorReduction`, `TensorIndexing`, and the
`ExecutionEngine` session wiring that eager/traced/AD drive.
