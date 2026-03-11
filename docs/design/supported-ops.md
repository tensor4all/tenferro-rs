# Supported Operations by Crate

This page is the deployable reference for what each operation-bearing crate
currently supports. It is intentionally operational, not aspirational: items
listed here are implemented today, while unsupported families are called out
explicitly.

Crates such as `tenferro-algebra` and `tenferro-device` define types and
protocols rather than user-visible operation families, so they are omitted
here.

## `tenferro-tensor`

### Structural tensor operations

- Shape/view: `reshape`, `reshape_owned`, `permute`, `broadcast_to`
- Matrix helpers: `transpose_last2`, `diagonal`
- Indexing/view selection: `slice_axis`, `select_axis`
- Materialization helpers: `to_tensor`, `to_owned`, contiguous conversion APIs

## `tenferro-prims`

### Semiring families

- `TensorSemiringCore`: `BatchedGemm`, `ReduceAdd`, `Trace`, `AntiTrace`, `AntiDiag`, `MakeContiguous`
- `TensorSemiringFastPath`: `Contract`, `ElementwiseBinary::{Add, Mul}`

### Scalar family

`TensorScalarPrims` currently exposes:

- Unary: `Neg`, `Conj`, `Abs`, `Reciprocal`, `Real`, `Imag`, `Square`
- Binary: `Add`, `Sub`, `Mul`, `Div`, `Maximum`, `Minimum`, `ClampMin`, `ClampMax`
- Reductions: `Sum`, `Prod`, `Mean`, `Max`, `Min`

### Analytic family

`TensorAnalyticPrims` currently exposes:

- Unary: `Sqrt`, `Rsqrt`, `Exp`, `Expm1`, `Log`, `Log1p`, `Sin`, `Cos`, `Tan`, `Tanh`, `Asin`, `Acos`, `Atan`, `Sinh`, `Cosh`, `Asinh`, `Acosh`, `Atanh`
- Binary: `Pow`, `Atan2`, `Hypot`, `Xlogy`
- Reductions: `Var`, `Std`

### Runtime status

- CPU: semiring/scalar/analytic families are implemented.
- CUDA and ROCm: semiring runtime hooks exist, but `TensorScalarPrims` and `TensorAnalyticPrims` currently advertise `has_*_support(...) == false` in phase 1.

## `tenferro-linalg-prims`

### Backend-facing kernel families

`TensorLinalgPrims` currently covers the low-level factorization and solve
contracts used by `tenferro-linalg`:

- `qr`
- `svd`
- `lu`
- `eigen`
- `eig`
- `lstsq`
- `cholesky`
- `solve`
- `solve_triangular`

## `tenferro-einsum`

### Primal

- `einsum`
- `einsum_with_subscripts`
- `einsum_with_plan`
- corresponding `_into`, `_owned`, and plan-based variants

### AD

- `einsum_rrule`
- `einsum_frule`
- `einsum_hvp`

## `tenferro-linalg`

### Primal

- Decompositions: `svd`, `qr`, `lu`, `lu_factor`, `lu_factor_ex`, `cholesky`, `cholesky_ex`, `eigen`, `eig`
- Solvers: `solve`, `solve_ex`, `solve_triangular`, `lu_solve`, `lstsq`
- Matrix utilities: `inv`, `inv_ex`, `det`, `slogdet`, `pinv`, `matrix_exp`, `matrix_power`, `norm`, `cond`
- Tensorized helpers: `cross`, `householder_product`, `vander`, `tensorinv`, `tensorsolve`

### Stateless AD rules

`_rrule` and `_frule` are implemented for:

- `svd`
- `qr`
- `lu`
- `eigen`
- `lstsq`
- `cholesky`
- `solve`
- `solve_triangular`
- `inv`
- `det`
- `slogdet`
- `eig`
- `pinv`
- `matrix_exp`
- `norm`

### AD gaps

The following public primal ops do not yet have stateless linalg AD rules:

- `*_ex` result-struct variants
- `lu_factor`
- `lu_factor_ex`
- `lu_solve`
- `matrix_power`
- `cond`
- `cross`
- `householder_product`
- `vander`
- `tensorinv`
- `tensorsolve`

## `tenferro-dyadtensor`

### Eager AD tensor entrypoints

`tenferro_dyadtensor::ad::*` currently includes:

- Einsum: `einsum`
- Reductions: `sum`, `mean`, `var`, `std`
- Scalar pointwise: `add`, `atan2`, `sqrt`, `exp`, `expm1`, `log`, `log1p`, `sin`, `cos`, `tanh`
- Linalg: `svd`, `qr`, `lu`, `eigen`, `lstsq`, `cholesky`, `solve`, `inv`, `det`, `slogdet`, `eig`, `pinv`, `matrix_exp`, `solve_triangular`, `norm`

### Builder-based AD surface

Builder APIs are implemented for:

- Einsum: `einsum_ad`
- Scalar/reduction: `add_ad`, `atan2_ad`, `sqrt_ad`, `exp_ad`, `expm1_ad`, `log_ad`, `log1p_ad`, `sin_ad`, `cos_ad`, `tanh_ad`, `mean_ad`, `var_ad`, `std_ad`
- Linalg: `svd_ad`, `qr_ad`, `lu_ad`, `eigen_ad`, `lstsq_ad`, `cholesky_ad`, `solve_ad`, `inv_ad`, `det_ad`, `slogdet_ad`, `eig_ad`, `pinv_ad`, `matrix_exp_ad`, `solve_triangular_ad`, `norm_ad`

### Runtime status

- API contract: runtime-generic across CPU, CUDA, and ROCm
- Actual execution today:
  - CPU paths are implemented for the operations listed above
  - CUDA and ROCm dispatch report unsupported capability for scalar/analytic and most linalg families rather than assuming CPU-only execution

## `tenferro-capi`

### C-API surface

The exported C surface currently focuses on a narrow, stable subset:

- Einsum entrypoints
- SVD entrypoints
- AD rule entrypoints for the supported FFI tensor/value wrappers

## `chainrules-scalarops`

This external crate provides the scalar formula basis reused by
`tenferro-dyadtensor` tensor-level wrappers.

- Arithmetic: `add`, `add_rrule`, `add_frule`
- Unary analytic/scalar: `conj`, `sqrt`, `exp`, `expm1`, `log`, `log1p`, `sin`, `cos`, `tanh`
- Binary analytic: `atan2`
- Power helpers: `powf`, `powi`

For formula details, see [AD Formula Notes](../AD/index.md).
