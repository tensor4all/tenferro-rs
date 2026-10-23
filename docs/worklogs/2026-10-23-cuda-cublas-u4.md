# U4, part 2: cuBLAS GEMM on the runtime's stream

Date: 2026-10-23. Issue: [tenferro-rs#2208](https://github.com/tensor4all/tenferro-rs/issues/2208)
unit U4, vendor operations (cuBLAS first; cuTENSOR, cuSOLVER and cuFFT follow).

## What landed

- `cuda_next::Cublas`: one cuBLAS handle per runtime, created inside the runtime's raw
  scope with host pointer mode and the runtime's own stream. `Cublas::gemm` serves a
  plain GEMM (`batches = 1`) or a strided-batched GEMM, for F32/F64/C32/C64, with the
  operand pointers taken from buffers registered in that raw scope — so a vendor call is
  ordered with the kernels that produced or will consume its operands and the buffers
  cannot be freed underneath it.
- `GemmRequest` is column-major throughout, like every tenferro operand, and carries the
  leading dimensions and batch strides; the adapter refuses a non-positive extent or a
  leading dimension below the operand's row count before calling the library.
- Complex coefficients are passed through a `#[repr(C, align(16))]` pair, never a
  `num_complex` value: `cuDoubleComplex` is `double2`, which CUDA declares 16-byte
  aligned, and the crate denies that cast class at its FFI boundary (issue #1870).

## Evidence

`tests/seam_vendor.rs` on the local A100 (driver 580.126.09, CUDA 12.6), against host
column-major references:

- `a_plain_gemm_matches_the_host` — a 37x41 times 41x29 f64 product.
- `a_transposed_operand_matches_the_host` — the same product with `a` stored `(k, m)` and
  `CUBLAS_OP_T`.
- `a_strided_batched_gemm_matches_the_host` — three 8x5 times 5x6 batches with element
  strides.
- `an_impossible_leading_dimension_is_refused` — a leading dimension below the operand's
  row count is a typed refusal before the library is called.

## Observations

- The seam's `VendorOps` trait stays declared and unimplemented. A vendor call needs the
  backend's device handle, while `gpu::Buffer` carries only the allocation identity today;
  the parent design leaves that handle to the private enum behind `Buffer`, which is the
  storage decision U5's tensor surface takes. Until then the CUDA adapter exposes typed
  methods rather than a second dispatch path, and the trait's placeholder `serve` signature
  is deleted in this change instead of being kept alive unused.
- The handle is destroyed in `Drop` after a best-effort synchronisation, inside a raw
  scope so the owning context is current.

## Not done (rest of U4)

cuTENSOR contraction, cuSOLVER linalg and cuFFT on the runtime stream. Each needs its own
handle and plan cache; the cuBLAS adapter establishes the pattern (handle per runtime,
per-call geometry validated, calls inside the runtime's raw scope).
