# Explicit polar-based CUDA SVD driver (#1849)

## Decisions

- Add `SvdDriver::Xgesvdp` as an explicit alternative, retaining the #1839
  default and CPU behavior. The CUDA integration shares vector and values-only
  workspace handling; wide matrices use the vendor's native support.
- Choose issue option A: discard `h_err_sigma` and document its meaning on the
  variant. Returning diagnostics would expand the output contract beyond this
  feature. No logging dependency or automatic driver policy is introduced.
- Preserve the existing full-SVD public surface, which has no driver option.
  Test its internal full/economy implementation directly rather than add an
  unrelated options API. See [GPU backend design](../design/gpu-backend-design.md).
- Bindings follow the NVIDIA cuSOLVER API reference, cross-checked against
  exAClior's independently written standalone harness supplied with #1849.
  This implementation is offered under `MIT OR Apache-2.0`; it neither copies
  the harness nor derives from NVIDIA's BSD-3 `CUDALibrarySamples/Xgesvdp` sample.

## Verification conclusions and constraints

- Tests cover explicit selection, the unchanged Auto boundary, operation
  identity and values-only pruning, and ignored CPU driver selection.
- Hardware-gated tests cover rectangular and batched ComplexF64, Float64,
  eager/traced execution, values-only execution, reconstruction, orthogonality,
  full factors, and agreement with Gesvd at the issue's acceptance sizes.
- The pending GPU validation was carried out during intake on an NVIDIA A100
  80GB PCIe (driver 580.126.09, CUDA 12.6). All four `#[ignore]` CUDA tests
  pass. Measured through tenferro's public API on a ten-decade ComplexF64
  spectrum (median of 5 after one warm-up):

  | m | driver | median ms | vs `gesvd` | `max\|Δs\|/s₁` |
  |---|---|---|---|---|
  | 400 | `Gesvd` | 44.37 | 1.00x | 4.00e-15 |
  | 400 | `Gesvdj` | 118.08 | 2.66x | 1.15e-13 |
  | 400 | `Xgesvdp` | 20.63 | 0.46x | 1.55e-15 |
  | 800 | `Gesvd` | 122.58 | 1.00x | 9.88e-15 |
  | 800 | `Gesvdj` | 384.95 | 3.14x | 1.59e-13 |
  | 800 | `Xgesvdp` | 51.23 | 0.42x | 1.78e-15 |
  | 1024 | `Gesvd` | 195.15 | 1.00x | 1.04e-14 |
  | 1024 | `Gesvdj` | 540.25 | 2.77x | 1.40e-13 |
  | 1024 | `Xgesvdp` | 75.96 | 0.39x | 1.55e-15 |

  The `Xgesvdp` column reproduces the issue's standalone cuSOLVER harness
  (20.3 / 51.0 / 75.8 ms) through the public API, so the integration adds no
  measurable overhead. `h_err_sigma` is discarded by design (option A), so it
  is not in the table.

- **Polar SVD accuracy is input-dependent, not just spectrum-dependent.** The
  acceptance test originally built its ten-decade input from a circulant-like
  closed form to stay O(m²). On that matrix `Xgesvdp` lost about two digits
  against `gesvd` (4.1e-13 versus 4.4e-15 at m = 400), so the 1e-13 tolerance
  — measured in the issue on `U diag(s) Vᴴ` — did not hold and the test failed
  on first GPU execution. The construction now uses two Householder
  reflectors, which keeps the O(m²) cost, makes the unitary factors exact to
  machine precision, and reproduces the issue's family: `Xgesvdp` is then the
  most accurate of the three drivers. The caveat is recorded on the
  `SvdDriver::Xgesvdp` doc comment, because a caller choosing this driver for
  accuracy on small singular values cannot assume it transfers between input
  families.

## #2182: single-precision rank-deficient failure and the Auto policy

The stage-2 Auto switch first made `Xgesvdp` the default for every dtype at
`min(m, n) >= 256`. Rank-deficient validation on an A100 (256x256, rank 64)
found that cuSOLVER's polar driver returns an impossible spectrum for single
precision: `f32` reported `smax = 9.03e2` and a tail of `8.09e2` for a matrix
whose true `smax` is `1.42e2` (device `gesvd` and the CPU reference agree), and
`c32` reported a tail of `16.2` against `1.2e-4`. The `pinv` residuals were
`0.996` (`f32`) and `1.48` (`c32`).

The integration is not at fault. The FFI signature and the
`dataTypeA`/`dataTypeS`/`dataTypeU`/`dataTypeV`/`computeType` pairings match
`cusolverDnXgesvdp` in CUDA 12.6, and cuSOLVER reports the inaccuracy itself
through `h_err_sigma`, its absolute perturbation bound on the computed singular
values. Measured `h_err_sigma`: `808.69` for the failing `f32` matrix (equal to
the bogus tail), `1.5e-8` for `f64` rank-deficient, `0` for full rank and for
the ten-decade spectrum, and `2.5e-13` for the near-cutoff c64 case.

Resolution (final):

- `SvdDriver::Auto` stays the accuracy-preserving JAX-compatible rule
  (`gesvdj` when both dimensions are at most 1024, else `gesvd`) for every
  dtype; `Xgesvdp` is opt-in only. The stage-2 speed proposal (`Auto =
  Xgesvdp` at `min(m, n) >= 256`) was dropped: `Xgesvdp`'s own `h_err_sigma`
  bound only guarantees the singular values to about `1e-6` of the largest, so
  any `eps`-scale truncation is unsafe, not just `pinv`. The speed table below
  stays as evidence for the opt-in choice.
- `svd_polar_typed` still reads `h_err_sigma` and returns a typed
  `BackendFailure` when it exceeds `1e-6` of the largest singular value, so an
  explicit `Xgesvdp` request cannot return silently wrong factors. The check is
  per matrix (a batch member is paired with its own spectrum) and reports the
  failing batch index, so a large well-conditioned member cannot mask a bad one.
- The pinv measurement that drove the decision: with `Auto = Xgesvdp` and the
  default `eps * max(m, n)` cutoff, the f64 rank-deficient case retained 253 of
  256 singular values against the CPU reference's 64, and
  `||P - P_cpu|| / ||P_cpu||` was `2.1e12`. Reverting `Auto` restores the
  reference rank and drops the error below `1e-13`; the CUDA test
  `cuda_svd_auto_default_pinv_rank_deficient_double_precision` pins that.
- The circulant-like family is no longer a correctness test; the
  `cuda_svd_driver` bench prints its accuracy alongside the other drivers.
