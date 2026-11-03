# Thin Complex64 on the vendor-BLAS provider (tensor4all-rs#1099)

## Decisions

- **Route thin c64 products away from Accelerate's Complex64 GEMM kernel, at
  the plan choice in `tenferro-cpu`, not at the GEMM call site.** The vendor
  GEMM is `cpueinsum-blas`'s; tenferro owns which prepared plan a contraction
  runs, so the only tenferro-side lever is *which plan*, not *how the kernel is
  called*. A `cblas_zgemv` formulation (17x on the same buffers) or a transposed
  `zgemm` argument order would have to be made where the CBLAS call is built,
  i.e. in `cpueinsum-blas`, and neither is implemented here.
- **Gate: Complex64, GEMM row role <= 2, tcontract would fuse the product onto
  faer, and the link is Accelerate.** Each part is load-bearing:
  - f32/f64/c32 are never routed: faer measured 1.8x to 3x slower than the
    vendor kernel for them on this host.
  - A real row role keeps Accelerate, which is 4.1x faster at `256^3`; the
    vendor's advantage grows with the row role, and no measurement here locates
    the crossover, so `m = 8` (1.4x faster on faer) deliberately stays on the
    vendor kernel.
  - Requiring tcontract's own report to say `Faer` keeps a product tcontract
    declines (or sends to its packed driver above the c64 `FaerLimit`) on the
    kernel it runs today.
  - The Accelerate condition keeps a reference-BLAS build out of a change
    measured only against Accelerate.

## Verification conclusions and constraints

- **The routed product does not reproduce the vendor kernel bit-for-bit.** It
  cannot: no alternative formulation on this host does. Measured on the same
  buffers and inputs, `cblas_zgemv` (5.9 us against 103 us) differs from
  `cblas_zgemm` by 1.4e-14, a scalar Rust loop (36.8 us) by 1.4e-14, and a
  transposed `zgemm` argument order is bit-identical (0 difference) but not
  faster (80 us against 66 us). Accelerate's Complex64 GEMM does not sum `k` in
  plain order, so the expectation that a BLAS-2 formulation is bit-exact does
  not hold for it (it does for a reference BLAS).
- **What the change does move is exactly the routed cells and nothing else**
  (`DIGEST` of complete outputs, raw `to_bits`, before against after on an M5
  Max, one thread):

  | cell | bitwise equal | max abs diff | relative to max abs value | worst component |
  |---|---|---|---|---|
  | c64 `1x141 * 141x141` | no | 1.4e-14 | 2.2e-16 | 1408 ULP |
  | c64 `1x141 * 141x282` | no | 2.1e-14 | 3.1e-16 | 1408 ULP |
  | c64 `2x141 * 141x141`, `2x141 * 141x282` | no | 3.6e-14 | 1.3e-15 | 364 ULP |
  | c64 `8x141 * 141x282`, `256^3`, `1x1024 * 1024x282` | yes | 0 | 0 | 0 |
  | every f64 cell | yes | 0 | 0 | 0 |

  The large per-component ULP counts are cancellation in a 141-term complex dot
  product against a small component, the effect the forward-error test in
  `contraction/tests.rs` already documents; normwise the change is 1 to 6 eps of
  the output scale.
- **Throughput** (median of 300 interleaved calls per cell in one process, two
  runs, one thread): `1x141 * 141x282` c64 61.0-61.3 us -> 17.8 us,
  `2x141 * 141x282` c64 81.0-81.3 us -> 19.3 us, `1x141 * 141x141` c64
  32.2 us -> 8.5 us, `2x141 * 141x141` c64 39.2-42.8 us -> 10.9 us. The cells
  outside the gate measured unchanged (c64 `8x141 * 141x282` 78.6-79.7 ->
  79.8-80.2, c64 `256^3` 540-545 -> 533-536, c64 above the faer bound
  427 -> 417-427, every f64 cell within noise).
- **Not checked:** the same measurement on another macOS version, an Intel
  macOS host, or a second Apple CPU; Complex32; batched products; the faer
  route's behaviour on a reference-BLAS build (the gate is off there, so
  behavior is unchanged); and whether Accelerate's kernel is also degenerate
  for thin products at thread budgets above one.
- **Remaining gap:** the `cblas_zgemv` formulation measured 17x against
  Accelerate's `zgemm` (5.9 us against 103 us raw, 3.5x against the routed
  faer plan's 17.8 us) and is not reachable from this crate. It belongs in
  `cpueinsum-blas`, which owns the CBLAS call and its layout inference.
