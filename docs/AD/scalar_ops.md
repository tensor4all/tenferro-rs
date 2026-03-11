# Scalar and Reduction AD Rules

This note records the scalar AD formulas implemented in
`chainrules-scalarops` and the tensor-level wrappers exposed through
`tenferro-dyadtensor`.

## Scope

Implemented scalar basis:

- `add`
- `conj`
- `sqrt`
- `powf` (scalar exponent)
- `powi` (integer exponent; a restricted `pow` case)
- `exp`
- `expm1`
- `log`
- `log1p`
- `sin`
- `cos`
- `tanh`
- `atan2`

Tensor-level wrappers built on top of those formulas:

- pointwise: `add_ad`, `atan2_ad`, `sqrt_ad`, `exp_ad`, `expm1_ad`, `log_ad`, `log1p_ad`, `sin_ad`, `cos_ad`, `tanh_ad`
- reductions: `mean_ad`, `var_ad`, `std_ad`

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

### `add`

- Primal: `y = x1 + x2`
- rrule: `(dx1, dx2) = (g, g)`
- frule: `dy = dx1 + dx2`

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

### `expm1`

- Primal: `y = exp(x) - 1`
- rrule: `dx = g * conj(exp(x))`
- frule: `dy = dx * conj(exp(x))`

### `log`

- Primal: `y = log(x)`
- rrule: `dx = g / conj(x)`
- frule: `dy = dx / conj(x)`

### `log1p`

- Primal: `y = log(1 + x)`
- rrule: `dx = g / conj(1 + x)`
- frule: `dy = dx / conj(1 + x)`

### `sin`

- Primal: `y = sin(x)`
- rrule: `dx = g * conj(cos(x))`
- frule: `dy = dx * conj(cos(x))`

### `cos`

- Primal: `y = cos(x)`
- rrule: `dx = g * conj(-sin(x))`
- frule: `dy = dx * conj(-sin(x))`

### `tanh`

- Primal: `y = tanh(x)`
- rrule: `dx = g * conj(1 - y^2)`
- frule: `dy = dx * conj(1 - y^2)`

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

## Tensor Reduction Wrappers

The tensor-level reduction builders in `tenferro-dyadtensor` reuse the scalar
rules above plus runtime-generic tensor primitives.

### `mean_ad`

For `y = mean(x)` over all elements (`N = x.len()`):

- rrule: every element receives `g / N`
- frule: `dy = mean(dx)`

### `var_ad`

For `y = var(x)` with population normalization:

- center: `c = x - mean(x)`
- rrule: `dx = g * 2c / N`
- frule: `dy = mean(2c * dx)`

### `std_ad`

For `y = std(x)` with `y = sqrt(var(x))`:

- rrule: `dx = g * c / (N * y)`
- frule: `dy = dvar / (2y)`

## API Placement

Implementation placement:

- scalar formulas and helper projection (`handle_r_to_c` equivalent):
  `extern/chainrules-scalarops`
- tensor-level generic unary/binary/reduction wrappers:
  `extension/tenferro-dyadtensor::scalar_ad_builders`
- eager AD entrypoints:
  `extension/tenferro-dyadtensor::ad`
