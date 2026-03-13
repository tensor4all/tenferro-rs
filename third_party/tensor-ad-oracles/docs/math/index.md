# Math Notes

`tensor-ad-oracles` publishes two first-class artifacts:

- mathematical AD notes
- the machine-readable oracle database

The mathematical notes under `docs/math/` are the human-facing source of truth
for known AD rules in this repository. They are maintained to preserve the full
derivation detail migrated from `tenferro-rs/docs/AD/` while adding explicit
correspondence to PyTorch's manual autograd formulas where relevant.

Standalone linalg operations are documented as one note per operation, while
shared scalar and wrapper formulas are grouped where that keeps the corpus
easier to maintain.

Published DB families are linked to the note corpus through
`docs/math/registry.json`, which maps each materialized `(op, family)` pair to a
stable note anchor.

For the human-facing explanation of that linkage, see
[math-registry.md](../math-registry.md).

## Core Linalg Notes

- [svd.md](./svd.md)
- [solve.md](./solve.md)
- [qr.md](./qr.md)
- [lu.md](./lu.md)
- [cholesky.md](./cholesky.md)
- [inv.md](./inv.md)
- [det.md](./det.md)
- [eig.md](./eig.md)
- [eigen.md](./eigen.md)
- [pinv.md](./pinv.md)
- [lstsq.md](./lstsq.md)
- [norm.md](./norm.md)

## Shared And Cross-Cutting Notes

- [scalar_ops.md](./scalar_ops.md)
- [matrix_exp.md](./matrix_exp.md)
- [dyadtensor_reverse.md](./dyadtensor_reverse.md)
