# Tropical Einsum AD Notes

## Scope

This note records the oracle family for the max-plus tropical matrix product
used by `tenferro-ext-tropical` AD coverage.

## PyTorch Baseline

The reference is a PyTorch composition for `ij,jk->ik`:

```python
torch.max(a.unsqueeze(2) + b.unsqueeze(0), dim=1).values
```

Published cases use unique winning contraction coordinates only. At such
points, the operation is locally affine, so central finite differences are
meaningful for JVP checks and for scalarized VJP adjoint checks.

## Published DB Families Using This Note

- <a id="op-tropical_einsum_maxplus"></a>`tropical_einsum_maxplus`
