# tenferro-prims

`tenferro-prims` is the execution-layer crate for tensor operations that sit
below `tenferro-einsum` and above raw tensor storage/views.

It owns the public primitive families used across the workspace:

- `TensorSemiringCore`
- `TensorSemiringFastPath`
- `TensorScalarPrims`
- `TensorAnalyticPrims`

Structural view operations such as `permute`, `reshape`, `broadcast`,
`diagonal`, `narrow`, and `select` stay in `tenferro-tensor`. Factorizations
and solve kernels stay in `tenferro-linalg-prims`.

## Public Operation Vocabulary

### `TensorSemiringCore`

- `BatchedGemm`
- `ReduceAdd`
- `Trace`
- `AntiTrace`
- `AntiDiag`
- `MakeContiguous`

### `TensorSemiringFastPath`

- `Contract`
- `ElementwiseBinary::{Add, Mul}`

### `TensorScalarPrims`

- Unary: `Neg`, `Conj`, `Abs`, `Reciprocal`, `Real`, `Imag`, `Square`
- Binary: `Add`, `Sub`, `Mul`, `Div`, `Maximum`, `Minimum`, `ClampMin`, `ClampMax`
- Reductions: `Sum`, `Prod`, `Mean`, `Max`, `Min`

### `TensorAnalyticPrims`

- Unary: `Sqrt`, `Rsqrt`, `Exp`, `Expm1`, `Log`, `Log1p`, `Sin`, `Cos`, `Tan`, `Tanh`, `Asin`, `Acos`, `Atan`, `Sinh`, `Cosh`, `Asinh`, `Acosh`, `Atanh`
- Binary: `Pow`, `Atan2`, `Hypot`, `Xlogy`
- Reductions: `Var`, `Std`

## Backend Status

| Backend | Status |
| --- | --- |
| CPU | Semiring core/fast path, scalar, and analytic families are implemented. |
| CUDA | Semiring core/fast path are implemented. Scalar and analytic families execute GPU-resident on real `f32`/`f64` tensors, plus the supported complex subset on `Complex32`/`Complex64`. |
| ROCm | Stub only. Support predicates remain false and planning/execution return unsupported errors. |

CUDA support is opt-in via `--features cuda`. GPU support is truthful rather
than aspirational:

- no silent CPU fallback during CUDA execution
- unsupported dtypes and ops must report unsupported capability
- intermediate tensors, scratch, and workspaces stay GPU-resident

## CUDA Notes

The CUDA backend uses two execution paths:

- cuTENSOR for contractions, reductions, layout materialization, and supported
  pointwise cases
- custom CUDA C++ kernels for phase-1 scalar/analytic gaps, compiled with
  `NVRTC` and cached on disk as PTX artifacts

CUDA scalar-family support today is:

- real `f32`/`f64`: full scalar inventory
- complex `Complex32`/`Complex64`: unary `Neg`, `Conj`, `Abs`, `Reciprocal`, `Real`, `Imag`, `Square`; binary `Add`, `Sub`, `Mul`, `Div`; reductions `Sum`, `Prod`, `Mean`
- real-only on CUDA: `Maximum`, `Minimum`, `ClampMin`, `ClampMax`, `Max`, `Min`

CUDA analytic-family support today is:

- real `f32`/`f64`: full analytic inventory
- complex `Complex32`/`Complex64`: all analytic unary ops plus binary `Pow` and `Xlogy`
- real-only on CUDA: `Atan2`, `Hypot`, `Var`, `Std`

The cache root is selected in this order:

1. `TENFERRO_CACHE_DIR`
2. `XDG_CACHE_HOME/tenferro/cuda`
3. `$HOME/.cache/tenferro/cuda`

## Features

- `cuda`: enables the real CUDA backend and pulls in `tenferro-tensor/cuda`
- `gemm-faer`: default CPU GEMM backend
- `gemm-blas` and provider/source features: optional BLAS-backed CPU GEMM

## Design References

- [`docs/design/tensor-prims.md`](../docs/design/tensor-prims.md)
- [`docs/design/supported-ops.md`](../docs/design/supported-ops.md)
