# Scalar AD Rules (`conj`, `sqrt`, `powf`, `powi`)

This note records the scalar AD formulas we align with before implementing
`chainrules-scalarops` and exposing `AdScalar` APIs in `tenferro-dyadtensor`.

## Scope

Target operations:

- `conj`
- `sqrt`
- `powf` (scalar exponent)
- `powi` (integer exponent; a restricted `pow` case)

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

