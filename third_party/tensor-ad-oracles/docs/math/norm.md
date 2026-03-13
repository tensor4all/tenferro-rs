# Norm AD Notes

## 1. Vector $p$-norm

$$
\|x\|_p = \left(\sum_i |x_i|^p\right)^{1/p},
\qquad
x \in \mathbb{C}^N
$$

### Forward Rule

$$
\dot{n} =
\frac{\sum_i |x_i|^{p-2}\operatorname{Re}(\bar{x}_i \dot{x}_i)}
{\|x\|_p^{p-1}}.
$$

### Reverse Rule

$$
\bar{x}_i =
\bar{n} \cdot \frac{x_i |x_i|^{p-2}}{\|x\|_p^{p-1}}.
$$

### Special cases

| $p$ | Reverse rule | Notes |
|---|---|---|
| $0$ | $0$ | Piecewise constant |
| $1$ | $\bar{n}\,\operatorname{sgn}(x)$ | Subgradient at $x_i = 0$ |
| $2$ | $\bar{n}\,x / \|x\|_2$ | Masked at $\|x\| = 0$ |
| $\infty$ | uniform average over active maximizers | Tie-sensitive |

## 2. Frobenius norm

$$
\|A\|_F = \sqrt{\operatorname{tr}(A^\dagger A)}.
$$

### Forward Rule

$$
\dot{n} = \frac{\operatorname{Re}\operatorname{tr}(A^\dagger \dot{A})}{\|A\|_F}.
$$

### Reverse Rule

$$
\bar{A} = \bar{n} \cdot \frac{A}{\|A\|_F}.
$$

## 3. Nuclear norm

$$
\|A\|_* = \sum_i \sigma_i(A).
$$

If $A = U S V^\dagger$ is an SVD, then

### Forward Rule

$$
\dot{n} = \operatorname{Re}\operatorname{tr}(U^\dagger \dot{A} V).
$$

### Reverse Rule

$$
\bar{A} = \bar{n} \cdot U V^\dagger.
$$

This norm inherits the same smoothness caveats as the SVD.

## 4. Spectral norm

$$
\|A\|_2 = \sigma_{\max}(A).
$$

For a simple top singular value:

### Forward Rule

$$
\dot{n} = \operatorname{Re}(u_1^\dagger \dot{A} v_1).
$$

### Reverse Rule

$$
\bar{A} = \bar{n} \cdot u_1 v_1^\dagger.
$$

For multiplicity $k > 1$, the subgradient is the average over the active
singular-vector dyads.

## Implementation Correspondence

- `tenferro-rs/docs/AD/norm.md` separates vector norms, Frobenius norm, nuclear
  norm, and spectral norm explicitly. This note preserves that structure.
- PyTorch's `norm_backward` and `norm_jvp` implement the scalar/vector $p$-norm
  cases directly, including the tie-handling for $p = \infty$.
- `linalg_vector_norm_backward` is a thin wrapper around the same formulas.
- Matrix nuclear and spectral norms are implemented in PyTorch by decomposition
  into SVD-derived primitives rather than a dedicated manual formula.

## Numerical Notes

- Nonsmooth points, especially zero inputs and repeated top singular values,
  require subgradient conventions.
- The DB excludes upstream norm families that are classified as unsupported
  subgradient cases.

## Verification

- compare primal norm values against direct evaluation
- compare JVP/VJP against finite differences away from nonsmooth points
- for nuclear and spectral norms, cross-check against SVD-based observables

## References

1. M. B. Giles, "An extended collection of matrix derivative results for
   forward and reverse mode automatic differentiation," 2008.
2. G. A. Watson, "Characterization of the subdifferential of some matrix norms,"
   1992.

## DB Families

<a id="family-norm-identity"></a>
### `norm/identity`

The DB publishes the chosen norm value directly.

<a id="family-matrix-norm-identity"></a>
### `matrix_norm/identity`

The DB publishes the matrix-norm observable directly.

<a id="family-vector-norm-identity"></a>
### `vector_norm/identity`

The DB publishes the vector-norm observable directly.

<a id="family-cond-identity"></a>
### `cond/identity`

The DB treats condition-number families as scalar spectral observables derived
from the same singular-value sensitivities.
