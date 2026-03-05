# AD Formula Notes

Mathematical derivations for the automatic differentiation rules (rrule/frule)
implemented in `tenferro-linalg`.

## Purpose

These notes contain the step-by-step mathematics behind each AD rule:
derivations from first principles, intermediate matrix identities, and
verification procedures (reconstruction checks, finite-difference gradient checks).

## Role distinction

| Location | Content |
|----------|---------|
| [`docs/design/autodiff.md`](../design/autodiff.md) | Architecture and API: crate split, explicit tape engine, torch-like wrapper layer, `BackwardOptions`, `DynTape` coexistence |
| [`docs/design/linalg.md`](../design/linalg.md) | Linalg API: function signatures, result types, cotangent types, rrule/frule tables |
| `docs/AD/` (this directory) | Mathematical details: derivations, formulas, verification for each operation |

## Notes

| File | Operation | Description |
|------|-----------|-------------|
| [svd.md](./svd.md) | SVD (`svd_rrule`) | Reverse-mode rule for `A = U diag(S) Vt`; F-matrix, non-square corrections, complex gauge |
| [qr.md](./qr.md) | QR / LQ (`qr_rrule`) | Reverse-mode rule for `A = QR` and `A = LQ`; `copyltu` helper, full-rank and wide/tall cases |
| [lu.md](./lu.md) | LU (`lu_rrule`, `lu_frule`) | Reverse-mode and forward-mode rules for `PA = LU`; square, wide, and tall cases |
| [cholesky.md](./cholesky.md) | Cholesky (`cholesky_rrule`) | Reverse-mode rule for `A = LLH` |
| [eigen.md](./eigen.md) | Symmetric eigen (`eigen_rrule`) | Reverse-mode rule for symmetric/Hermitian eigendecomposition |
| [eig.md](./eig.md) | General eigen (`eig_rrule`) | Reverse-mode rule for general (non-symmetric) eigendecomposition |
| [inv.md](./inv.md) | Matrix inverse (`inv_rrule`) | AD rules for `inv(A)`; formula `dA = -A^{-H} cotangent A^{-H}` |
| [det.md](./det.md) | Determinant and slogdet | AD rules for `det(A)` and `slogdet(A)` |
| [solve.md](./solve.md) | Linear solve (`solve_rrule`, `solve_triangular_frule`, `solve_triangular_rrule`) | AD rules for `Ax = b` and triangular solve |
| [lstsq.md](./lstsq.md) | Least squares (`lstsq_rrule`) | Reverse-mode rule for `argmin ||Ax - b||^2` |
| [pinv.md](./pinv.md) | Pseudoinverse (`pinv_rrule`) | AD rules for Moore-Penrose pseudoinverse |
| [matrix_exp.md](./matrix_exp.md) | Matrix exponential (`matrix_exp_rrule`) | AD rules for `exp(A)` |
| [norm.md](./norm.md) | Norm (`norm_rrule`) | AD rules for matrix and vector norms |
| [scalar_ops.md](./scalar_ops.md) | Scalar ops (`conj`, `sqrt`, `powf`, `powi`) | PyTorch-aligned scalar rrule/frule conventions and `handle_r_to_c` projection note |
| [dyadtensor_reverse.md](./dyadtensor_reverse.md) | Dyadtensor reverse wiring | `.run()` pullback registration coverage, including mixed-type `eig` bridge pullback |
