# Determinant AD Notes

## 1. Determinant

### Forward Definition

For

$$
d = \det(A),
\qquad
A \in \mathbb{C}^{N \times N},
$$

Jacobi's formula gives

$$
\dot{d} = \det(A) \cdot \operatorname{tr}(A^{-1}\dot{A}).
$$

### Reverse Rule

Given a cotangent $\bar{d}$:

- real case:

$$
\bar{A} = \bar{d} \cdot \det(A) \cdot A^{-\mathsf{T}}
$$

- complex case:

$$
\bar{A} = \overline{\bar{d} \cdot \det(A)} \cdot A^{-\mathsf{H}}.
$$

## Singular matrix handling

The inverse formula fails at singular matrices, but the adjugate interpretation
still makes sense:

- rank $N-1$: the adjugate is rank 1 and can be reconstructed from an SVD
- rank $\le N-2$: the adjugate vanishes

PyTorch's `linalg_det_backward` handles this regime by reconstructing the
leave-one-out singular-value products together with the orientation/phase factor
coming from $U$ and $V^{\mathsf{H}}$.

## 2. `slogdet`

### Forward Definition

$$
(\operatorname{sign}, \operatorname{logabsdet}) = \operatorname{slogdet}(A).
$$

If $w = \operatorname{tr}(A^{-1}\dot{A})$, then in the complex case

$$
\dot{\operatorname{logabsdet}} = \operatorname{Re}(w),
\qquad
\dot{\operatorname{sign}} = i \operatorname{Im}(w)\operatorname{sign}.
$$

### Reverse Rule

For the differentiable log-magnitude path:

- real case:

$$
\bar{A} = \overline{\operatorname{logabsdet}} \cdot A^{-\mathsf{T}}
$$

- complex case:

$$
\bar{A} = g \cdot A^{-\mathsf{H}},
\qquad
g = \overline{\operatorname{logabsdet}}
- i \operatorname{Im}(\overline{\operatorname{sign}}^* \operatorname{sign}).
$$

`slogdet` is not differentiable at singular matrices because
$\operatorname{logabsdet} = -\infty$ there.

## Implementation Correspondence

- `tenferro-rs/docs/AD/det.md` keeps both `det` and `slogdet` in one note and
  discusses the singular adjugate path explicitly.
- PyTorch's `linalg_det_jvp`, `linalg_det_backward`, `slogdet_jvp`, and
  `slogdet_backward` implement the same split and use solves rather than
  explicit inverses.

## Verification

- compare primal `det(A)` and `slogdet(A)` with direct evaluation
- compare JVP/VJP against finite differences away from singularity

## References

1. C. G. J. Jacobi, "De formatione et proprietatibus determinantium," 1841.
2. M. B. Giles, "An extended collection of matrix derivative results for
   forward and reverse mode AD," 2008.

## DB Families

<a id="family-det-identity"></a>
### `det/identity`

The DB publishes the determinant value directly.

<a id="family-slogdet-identity"></a>
### `slogdet/identity`

The DB publishes the differentiable `slogdet` observable directly.
