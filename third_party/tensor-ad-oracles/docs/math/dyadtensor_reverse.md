# Dyadtensor Reverse Wiring Notes

## Purpose

This note records how higher-level dyadtensor APIs attach reverse-mode pullbacks
to tensor values without exposing tape internals.

## Registered `.run()` pullbacks

For reverse-mode outputs, builders such as

- `einsum_ad(...).run()`
- `svd_ad(...).run()`
- `qr_ad(...).run()`
- `lu_ad(...).run()`
- `eigen_ad(...).run()`
- `lstsq_ad(...).run()`
- `solve_triangular_ad(...).run()`
- `cholesky_ad(...).run()`
- `solve_ad(...).run()`
- `inv_ad(...).run()`
- `det_ad(...).run()`
- `slogdet_ad(...).run()`
- `eig_ad(...).run()`
- `pinv_ad(...).run()`
- `matrix_exp_ad(...).run()`
- `norm_ad(...).run()`

register a local pullback on the tensor-local tape node.

## Pullback APIs

- same scalar domain:
  - `ad::pullback`
  - `ad::pullback_wrt`
- mixed scalar domain:
  - `ad::pullback_wrt_mixed`

The mixed-domain entrypoint is needed for operators such as `eig_ad(...).run()`
whose outputs are complex while the primal input may be real.

## Why the bridge exists

The reverse rule corpus needs one place that explains how raw operator rules
become tensor-facing reverse APIs. This note bridges

- raw operator notes such as [eig.md](./eig.md)
- shared scalar wrappers in [scalar_ops.md](./scalar_ops.md)
- higher-level DB replay and frontend documentation

## Current limits

- Mixed-type pullback is bridge-based and only applies to operators that
  explicitly register a cross-domain reverse bridge, currently through
  `register_bridge_rule`.
- Same-domain `ad::pullback` remains intentionally strict; use
  `ad::pullback_wrt_mixed` when output and input scalar domains differ.

## DB Status

This note documents implementation wiring rather than a published `(op, family)`
oracle family, so it does not currently appear in `docs/math/registry.json`.
