# Frontend Reverse Gradient Coverage

This note summarizes how reverse-mode gradients are wired in the
`tenferro` eager API.

## Goal

Keep graph internals hidden from end-user tensor code while still allowing:

- `out = x.exp()?.sum()?`
- `grads = grad(&[&out], &[&x], None, GradOptions::default())?`
- `out.backward(None, &[&x], BackwardOptions::default())?`

without passing any explicit tape handle through user APIs.

## Registered `.run()` pullbacks

For reverse-mode outputs, these builders now register a local pullback on the
tensor-local tape node:

| Builder | Pullback implementation |
|---|---|
| `einsum_ad(...).run()` | `tenferro_einsum::einsum_rrule` |
| `svd_ad(...).run()` | `tenferro_linalg::svd_rrule` (per-output: `u`, `s`, `vt`) |
| `qr_ad(...).run()` | `tenferro_linalg::qr_rrule` (per-output: `q`, `r`) |
| `lu_ad(...).run()` | `tenferro_linalg::lu_rrule` (per-output: `l`, `u`) |
| `eigen_ad(...).run()` | `tenferro_linalg::eigen_rrule` (per-output: `values`, `vectors`) |
| `lstsq_ad(...).run()` | `x`: `tenferro_linalg::lstsq_rrule`; `residual`: zero pullback |
| `solve_triangular_ad(...).run()` | `tenferro_linalg::solve_triangular_rrule` |
| `cholesky_ad(...).run()` | `tenferro_linalg::cholesky_rrule` |
| `solve_ad(...).run()` | `tenferro_linalg::solve_rrule` |
| `inv_ad(...).run()` | `tenferro_linalg::inv_rrule` |
| `det_ad(...).run()` | `tenferro_linalg::det_rrule` |
| `slogdet_ad(...).run()` | `logabsdet`: `tenferro_linalg::slogdet_rrule`; `sign`: zero pullback |
| `eig_ad(...).run()` | reverse-mode currently unsupported for real inputs; forward/primal remain available |
| `pinv_ad(...).run()` | `tenferro_linalg::pinv_rrule` |
| `matrix_exp_ad(...).run()` | `tenferro_linalg::matrix_exp_rrule` |
| `norm_ad(...).run()` | `tenferro_linalg::norm_rrule` |

Public reverse entrypoints:

- `Tensor::set_requires_grad(...)`
- `Tensor::grad()`
- `Tensor::backward(...)`
- free `grad(...)`
- free `backward(...)`

These keep graph symbols internal to `tenferro` while preserving a homogeneous
reverse graph over runtime-typed tensor payloads.

## Current limits

- Reverse-mode graphs are homogeneous: scalar AD values are represented as
  rank-0 tensors, and cross-dtype reverse bridges are not part of the graph.
- `eig_ad` remains available for primal/forward paths, but reverse-mode for
  real inputs is currently rejected rather than maintaining a mixed-type bridge.

## Related AD-rule status updates

- `NormKind::L1` and `NormKind::Inf` are implemented in `tenferro-linalg` for:
  - primal `norm`
  - `norm_frule`
  - `norm_rrule`
- For L1/Inf ties (multiple maximizing columns/rows), gradients are averaged
  uniformly over active maximizers.
- `lu(..., LuPivot::NoPivot)` forward path is implemented (returns `p: None`);
  LU AD rules remain available through existing `lu_frule` / `lu_rrule`.
