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

Predicate/select-style tensor ops are intentionally absent here today. `Where`
and the AD surface for branch-select families need a dedicated boolean/predicate
substrate before they can be added cleanly.

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

The backend-facing dtype contract is `KernelLinalgScalar`, with LAPACK-specific
eig helpers isolated behind `LapackEigScalar`.

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

## `tenferro`

### Structured tensor helpers

- `StructuredTensor::to_dense`
- `StructuredTensor::einsum_with_subscripts`

### Eager AD tensor entrypoints

`tenferro::Tensor` currently exposes these eager methods:

- Einsum: `einsum`
- Reductions: `sum`, `mean`, `var`, `std`
- Scalar pointwise:
  `add`, `atan2`, `pow`, `hypot`,
  `sqrt`, `exp`, `expm1`, `log`, `log1p`,
  `sin`, `cos`, `tanh`,
  `asin`, `acos`, `atan`,
  `sinh`, `cosh`, `asinh`, `acosh`, `atanh`
- Linalg: `svd`, `qr`, `lu`, `eigen`, `lstsq`, `cholesky`, `solve`, `inv`, `det`, `slogdet`, `eig`, `pinv`, `matrix_exp`, `solve_triangular`, `norm`

### Builder-based AD surface

Internal builder APIs are implemented for:

- Einsum: `einsum_ad`
- Scalar/reduction:
  `add_ad`, `atan2_ad`, `pow_ad`, `hypot_ad`,
  `sqrt_ad`, `exp_ad`, `expm1_ad`, `log_ad`, `log1p_ad`,
  `sin_ad`, `cos_ad`, `tanh_ad`,
  `asin_ad`, `acos_ad`, `atan_ad`,
  `sinh_ad`, `cosh_ad`, `asinh_ad`, `acosh_ad`, `atanh_ad`,
  `sum_ad`, `mean_ad`, `var_ad`, `std_ad`
- Linalg: `svd_ad`, `qr_ad`, `lu_ad`, `eigen_ad`, `lstsq_ad`, `cholesky_ad`, `solve_ad`, `inv_ad`, `det_ad`, `slogdet_ad`, `eig_ad`, `pinv_ad`, `matrix_exp_ad`, `solve_triangular_ad`, `norm_ad`
  - `eig_ad` reverse mode is same-domain only in `tenferro`; real-input reverse mode is intentionally rejected to keep the tape homogeneous

### Runtime status

- API contract: runtime-generic across CPU, CUDA, and ROCm
- internal chainrules-backed einsum helpers remain backend-parametric over `tenferro-einsum::EinsumBackend`; runtime selection still happens through the frontend runtime context type
- Structured tensor materialization and compressed einsum reuse the same einsum runtime-dispatch layer rather than maintaining separate CPU/CUDA/ROCm builder paths
- Builder execution uses an explicit default-runtime holder, and reverse-mode bookkeeping stays on one homogeneous runtime-typed graph
- `Tensor` is the canonical public payload for downstream tensor algebra; implicit result-type promotion happens inside mixed-dtype tensor ops (`complex` beats `real`, 64-bit beats 32-bit), explicit numeric casts use `to_scalar_type(...)`, and `detach()` drops AD metadata without switching to a second public tensor type
- Actual execution today:
  - CPU paths are implemented for the operations listed above
  - CUDA and ROCm dispatch report unsupported capability for scalar/analytic and most linalg families rather than assuming CPU-only execution
  - mixed-dtype reverse propagation is supported when operands share one reverse graph; pullbacks cast gradients back to each input dtype
  - linalg entry points are dense-only at the `tenferro` frontend layer; non-dense structured inputs return a runtime error instead of silently materializing dense fallbacks

## `tenferro-capi`

### C-API surface

The exported C surface currently focuses on a narrow, stable subset:

- Einsum entrypoints
- SVD entrypoints
- AD rule entrypoints for the supported FFI tensor/value wrappers

## `tenferro-burn`

### Burn bridge surface

- Checked helpers: `try_einsum`, `try_burn_to_tenferro`, `try_tenferro_to_burn`
- Convenience wrappers: `einsum`, `burn_to_tenferro`, `tenferro_to_burn`
- Backend extension trait: `TensorNetworkOps`

### Runtime status

- Current execution lowers through CPU tenferro tensors after checked conversion.
- `NdArray<f64>` forward execution and `Autodiff<B, C>` backward execution are implemented.
- Invalid subscripts, malformed nested einsum trees, and conversion failures now flow through checked helpers before any public panic-wrapper boundary.

## `tenferro-mdarray`

### mdarray bridge surface

- Checked helpers: `try_mdarray_to_tensor`, `try_tensor_to_mdarray`
- Convenience wrappers: `mdarray_to_tensor`, `tensor_to_mdarray`

### Runtime status

- Both conversion directions are eager copy paths.
- Zero-copy interoperability is intentionally unsupported.
- Conversion helpers are CPU-buffer oriented and reject non-owned/non-CPU materialization through checked errors.

## `chainrules-scalarops`

This external crate provides the scalar formula basis reused by
`tenferro` tensor-level wrappers.

- Arithmetic: `add`, `sub`, `mul`, `div` with matching `*_rrule` / `*_frule`
- Unary analytic/scalar: `conj`, `sqrt`, `exp`, `log`
- Binary analytic: `atan2`
- Power helpers: `powf`, `powi`

The broader tensor-level analytic families in `tenferro` are built
from these scalar formulas plus runtime-generic tensor primitives. They are not
all exported directly from `chainrules-scalarops`.

For formula details, see [AD Formula Notes](../AD/index.md).
