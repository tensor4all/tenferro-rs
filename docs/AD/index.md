# AD Formula Notes

Mathematical derivations for the automatic differentiation rules (rrule/frule)
implemented across the `tenferro-rs` workspace.

## Purpose

These notes contain the step-by-step mathematics behind each AD rule:
derivations from first principles, intermediate matrix identities, and
verification procedures (reconstruction checks, finite-difference gradient checks).

## Role distinction

| Location | Content |
|----------|---------|
| [`docs/design/autodiff.md`](../design/autodiff.md) | Architecture and API: crate split, homogeneous `Tape<V>` engine, `backward`/`grad` frontend APIs, and rank-0 tensor scalar semantics |
| [`docs/design/einsum-dyadtensor.md`](../design/einsum-dyadtensor.md) | Einsum/frontend AD integration details on top of homogeneous `Tape<V>`, including retain/create semantics and tensor scalar conventions |
| [`docs/design/linalg.md`](../design/linalg.md) | Linalg API: function signatures, result types, cotangent types, rrule/frule tables |
| `docs/AD/` (this directory) | Mathematical details: derivations, formulas, verification for each operation |

## Notes

## Implemented AD Rules by Crate

This section is the quick inventory; the per-operation notes below remain the
source of formula detail.

### `chainrules`

- Scalar arithmetic: `add`, `sub`, `mul`, `div` plus matching `*_rrule` / `*_frule`
- Unary scalar rules: `conj`, `sqrt`, `exp`, `log`
- Binary scalar rules: `atan2`
- Power helpers: `powf`, `powi`

`chainrules` intentionally stays small. Tensor-level wrappers such as
`sin`, `cos`, `tanh`, `asin`, `pow`, `hypot`, `var`, and `std` are implemented
one layer up in `tenferro` by composing runtime-generic tensor prims.

### `tidu`

`tidu` owns the execution engine:

- reverse mode: `Tape<V>` and `TrackedValue<V>`
- forward mode: `DualValue<V>`
- HVP: `Tape::hvp`

### `tenferro-einsum`

- `einsum_rrule`
- `einsum_frule`
- `einsum_hvp`

### `tenferro-linalg`

- `svd_rrule`, `svd_frule`
- `qr_rrule`, `qr_frule`
- `lu_rrule`, `lu_frule`
- `eigen_rrule`, `eigen_frule`
- `lstsq_rrule`, `lstsq_frule`
- `cholesky_rrule`, `cholesky_frule`
- `solve_rrule`, `solve_frule`
- `solve_triangular_rrule`, `solve_triangular_frule`
- `inv_rrule`, `inv_frule`
- `det_rrule`, `det_frule`
- `slogdet_rrule`, `slogdet_frule`
- `eig_rrule`, `eig_frule`
- `pinv_rrule`, `pinv_frule`
- `matrix_exp_rrule`, `matrix_exp_frule`
- `norm_rrule`, `norm_frule`

### `tenferro`

- Dynamic eager tensor methods on `Tensor`:
  - tensor/reduction: `einsum`, `sum`, `mean`, `var`, `std`
  - scalar/analytic: `add`, `atan2`, `pow`, `hypot`, `sqrt`, `exp`, `expm1`, `log`, `log1p`, `sin`, `cos`, `tanh`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `asinh`, `acosh`, `atanh`
  - linalg: `svd`, `qr`, `lu`, `eigen`, `eig`, `lstsq`, `cholesky`, `solve`, `inv`, `det`, `slogdet`, `pinv`, `matrix_exp`, `solve_triangular`, `norm`
- Internal builder plumbing still lives in `*_ad(...).run()` builders, but the preferred public surface is now the `Tensor` method API
  - `eig` reverse mode is same-domain only; real-input reverse mode is currently unsupported in the `tenferro` frontend

For a broader crate-by-crate support view, including primal coverage and
runtime status, see [Supported Operations by Crate](../design/supported-ops.md).

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
| [scalar_ops.md](./scalar_ops.md) | Scalar and tensor pointwise/reduction ops | Scalar basis rules (`add`, `conj`, `sqrt`, `exp`, `log`, `atan2`, powers) plus tensor wrappers (`mean`, `var`, `std`, `sin`, `cos`, `tanh`, ...) |
| [dyadtensor_reverse.md](./dyadtensor_reverse.md) | Frontend reverse wiring | `.run()` pullback registration coverage, including mixed-type `eig` bridge pullback |
