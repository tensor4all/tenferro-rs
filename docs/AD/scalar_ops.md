# Scalar AD Rules (`conj`, `sqrt`, `powf`, `powi`, `exp`, `log`, `atan2`)

This note records the scalar AD formulas we align with before implementing
`chainrules-scalarops` and exposing `AdScalar` APIs in `tenferro-dyadtensor`.

## Scope

Target operations:

- `conj`
- `sqrt`
- `powf` (scalar exponent)
- `powi` (integer exponent; a restricted `pow` case)
- `exp`
- `log`
- `atan2`

Target scalar domains:

- `f32`, `f64`
- `Complex32`, `Complex64`

## PyTorch Baseline (Local Snapshot)

Reference repository: `../pytorch`  
Commit used for this note: `8dd3b7637abd04433bafe77765de59df6388f9f9`

Primary source files:

- `tools/autograd/derivatives.yaml`
- `torch/csrc/autograd/FunctionsManual.cpp`
- `docs/source/notes/autograd.rst`

Key lines:

- `_conj` backward: `derivatives.yaml:477-479`
- `sqrt` backward: `derivatives.yaml:1622-1624`
- `exp` backward: `derivatives.yaml:652-654`
- `log` backward: `derivatives.yaml:966-968`
- `atan2` backward: `derivatives.yaml:260-263`
- `pow.Tensor_Scalar` / `pow.Tensor_Tensor`: `derivatives.yaml:1385-1392`
- `pow_backward*` implementation: `FunctionsManual.cpp:473-557`
- `handle_r_to_c`: `FunctionsManual.cpp:169-183`
- Complex AD convention statement: `notes/autograd.rst:601-607`

## Complex Gradient Convention

We follow PyTorch's convention for real-valued losses:

- gradients are conjugate-Wirtinger (`dL/dz*`) style
- VJP formulas include complex conjugation where required
- if input is real and an intermediate gradient is complex, project back to real
  (`handle_r_to_c` behavior)

## Rule Summary

Let `g` be output cotangent, `x` input primal, `y = f(x)` output primal.

### `conj`

- Primal: `y = conj(x)`
- rrule: `dx = conj(g)`
- frule: `dy = conj(dx)`

### `sqrt`

- Primal: `y = sqrt(x)`
- rrule: `dx = g / (2 * conj(y))`
- frule: `dy = dx / (2 * conj(y))`

### `powf` (fixed scalar exponent `a`)

- Primal: `y = x^a`
- rrule (self gradient): `dx = g * conj(a * x^(a - 1))`
- frule (self tangent): `dy = dx * conj(a * x^(a - 1))`

### `powi` (fixed integer exponent `n`)

This is `powf` with integer exponent semantics.

- Primal: `y = x^n`
- rrule: `dx = g * conj(n * x^(n - 1))`
- frule: `dy = dx * conj(n * x^(n - 1))`

### `exp`

- Primal: `y = exp(x)`
- rrule: `dx = g * conj(y)`
- frule: `dy = dx * conj(y)`

### `log`

- Primal: `y = log(x)`
- rrule: `dx = g / conj(x)`
- frule: `dy = dx / conj(x)`

### `atan2` (real-valued inputs)

Let `y = atan2(a, b)` with `a` as numerator-like input and `b` as
denominator-like input.

- rrule:
  - `da = g * b / (a^2 + b^2)`
  - `db = g * (-a) / (a^2 + b^2)`
- frule:
  - `dy = da * b / (a^2 + b^2) + db * (-a) / (a^2 + b^2)`

## Edge Cases

Aligned with PyTorch:

- `pow` with exponent `0` gives zero self-gradient.
- Real-input/complex-intermediate gradients are projected back to real
  (`handle_r_to_c` equivalent).

## API Placement

Implementation placement:

- formulas and helper projection (`handle_r_to_c` equivalent):
  `extern/chainrules-scalarops`
- user-facing scalar AD API:
  `extension/tenferro-dyadtensor::AdScalar`
  and tensor-level generic unary/binary/reduction wrappers such as
  `exp_ad`, `add_ad`, and `mean_ad`
