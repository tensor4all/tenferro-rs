# Cholesky AD Notes

## Forward Definition

$$
A = L L^{\mathsf{H}},
\qquad
A \in \mathbb{C}^{N \times N},
\qquad
A = A^{\mathsf{H}},
\qquad
L \text{ lower triangular}
$$

with $A$ Hermitian positive-definite.

## Auxiliary Operator

Define

$$
\varphi(X) = \mathrm{tril}(X) - \tfrac{1}{2}\mathrm{diag}(X),
$$

which extracts the lower triangle and halves the diagonal. Its adjoint is

$$
\varphi^*(X) = \tfrac{1}{2}(X + X^{\mathsf{H}} - \mathrm{diag}(X)).
$$

## Forward Rule

Given a Hermitian tangent $\dot{A}$:

$$
\dot{L} = L \, \varphi\!\bigl(L^{-1}\dot{A}\,L^{-\mathsf{H}}\bigr).
$$

Differentiate $A = L L^{\mathsf{H}}$:

$$
\dot{A} = \dot{L} L^{\mathsf{H}} + L \dot{L}^{\mathsf{H}}.
$$

Left-multiplying by $L^{-1}$ and right-multiplying by $L^{-\mathsf{H}}$ gives

$$
L^{-1}\dot{A}\,L^{-\mathsf{H}} =
L^{-1}\dot{L} + (L^{-1}\dot{L})^{\mathsf{H}}.
$$

Since $L^{-1}\dot{L}$ is lower triangular, $\varphi$ inverts this
symmetrization.

## Reverse Rule

Given a cotangent $\bar{L}$:

$$
\bar{A} =
L^{-\mathsf{H}} \,
\varphi^*\!\bigl(\mathrm{tril}(L^{\mathsf{H}}\bar{L})\bigr)
\, L^{-1}.
$$

This is the adjoint of the JVP map and keeps $\bar{A}$ Hermitian.

## Implementation Correspondence

- `tenferro-rs/docs/AD/cholesky.md` uses the same $\varphi / \varphi^*$ pair to
  express both JVP and VJP.
- PyTorch's `cholesky_jvp` and `cholesky_backward` implement the same
  triangular-solve sandwich rather than explicit inverses.
- Never form $L^{-1}$ explicitly; use triangular solves on the left and right.

## Verification

### Forward reconstruction

$$
\|A - L L^{\mathsf{H}}\|_F < \varepsilon.
$$

### Backward checks

- compare JVP/VJP against finite differences on Hermitian perturbations
- confirm failure outside the positive-definite domain

## References

1. S. P. Smith, "Differentiation of the Cholesky Algorithm," 1995.
2. I. Murray, "Differentiation of the Cholesky decomposition," 2016.

## DB Families

<a id="family-cholesky-identity"></a>
### `cholesky/identity`

The DB publishes the differentiable Cholesky factor.

<a id="family-cholesky-ex-identity"></a>
### `cholesky_ex/identity`

The DB validates the factor output while treating auxiliary status outputs as
metadata.
