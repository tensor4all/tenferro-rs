# Dyadtensor Reverse Pullback Coverage

This note summarizes how reverse-mode pullbacks are wired in
`tenferro-dyadtensor` builder/eager APIs.

## Goal

Keep tape internals hidden from end-user tensor code while still allowing:

- `out = op_ad(...).run()?`
- `grads = ad::pullback(&out, &cotangent)?`

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
| `eig_ad(...).run()` | `tenferro_linalg::eig_rrule` via mixed-type bridge (`Complex -> Real`) |
| `pinv_ad(...).run()` | `tenferro_linalg::pinv_rrule` |
| `matrix_exp_ad(...).run()` | `tenferro_linalg::matrix_exp_rrule` |
| `norm_ad(...).run()` | `tenferro_linalg::norm_rrule` |

APIs:

- Same scalar domain (`output` and `wrt` share dtype):
  - `ad::pullback`
  - `ad::pullback_wrt`
- Mixed scalar domain (e.g. `eig_ad` complex outputs, real inputs):
  - `ad::pullback_wrt_mixed`

All of these keep tape symbols internal to dyadtensor.

## Current limits

- Mixed-type pullback is currently bridge-based (`register_bridge_rule`) and is
  implemented for operators that explicitly register a cross-domain reverse
  bridge (`eig_ad` currently).
- `ad::pullback` remains same-domain by design. Use `ad::pullback_wrt_mixed`
  when `output` and `wrt` dtypes differ.

## Related AD-rule status updates

- `NormKind::L1` and `NormKind::Inf` are implemented in `tenferro-linalg` for:
  - primal `norm`
  - `norm_frule`
  - `norm_rrule`
- For L1/Inf ties (multiple maximizing columns/rows), gradients are averaged
  uniformly over active maximizers.
- `lu(..., LuPivot::NoPivot)` forward path is implemented (returns `p: None`);
  LU AD rules remain available through existing `lu_frule` / `lu_rrule`.
