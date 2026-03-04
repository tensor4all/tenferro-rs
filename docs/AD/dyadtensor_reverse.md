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
| `solve_triangular_ad(...).run()` | `tenferro_linalg::solve_triangular_rrule` |
| `cholesky_ad(...).run()` | `tenferro_linalg::cholesky_rrule` |
| `solve_ad(...).run()` | `tenferro_linalg::solve_rrule` |
| `inv_ad(...).run()` | `tenferro_linalg::inv_rrule` |
| `det_ad(...).run()` | `tenferro_linalg::det_rrule` |
| `pinv_ad(...).run()` | `tenferro_linalg::pinv_rrule` |
| `matrix_exp_ad(...).run()` | `tenferro_linalg::matrix_exp_rrule` |
| `norm_ad(...).run()` | `tenferro_linalg::norm_rrule` |

`ad::pullback` / `ad::pullback_wrt` can therefore consume these outputs
directly, with no explicit tape symbol in user code.

## Current limits

Multi-output builders (`svd_ad`, `qr_ad`, `lu_ad`, `eigen_ad`, `lstsq_ad`,
`slogdet_ad`, `eig_ad`) still expose reverse metadata on outputs, but do not
yet auto-register composed pullbacks in `.run()`.

For those operators, use stateless `*_rrule` entry points directly when
pullback execution is required.

## Related AD-rule status updates

- `NormKind::L1` and `NormKind::Inf` are implemented in `tenferro-linalg` for:
  - primal `norm`
  - `norm_frule`
  - `norm_rrule`
- For L1/Inf ties (multiple maximizing columns/rows), gradients are averaged
  uniformly over active maximizers.
- `lu(..., LuPivot::NoPivot)` forward path is implemented (returns `p: None`);
  LU AD rules remain available through existing `lu_frule` / `lu_rrule`.
