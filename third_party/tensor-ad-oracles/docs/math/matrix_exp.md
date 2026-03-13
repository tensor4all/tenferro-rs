# Matrix Exponential AD Notes

## Forward Definition

$$
B = \exp(A),
\qquad
A \in \mathbb{C}^{N \times N}
$$

The Fr\'echet derivative in direction $E$ is

$$
L(A, E) =
\int_0^1 \exp(sA)\,E\,\exp((1-s)A)\,ds.
$$

## Block Matrix Formula (Mathias 1996)

Both JVP and VJP can be written through a single exponential of a
$2N \times 2N$ block upper-triangular matrix:

$$
\exp\!\begin{pmatrix} A & E \\ 0 & A \end{pmatrix}
= \begin{pmatrix} \exp(A) & L(A, E) \\ 0 & \exp(A) \end{pmatrix}.
$$

The upper-right block is the Fr\'echet derivative.

## Forward Rule

Given $\dot{A}$:

$$
\dot{B} = L(A, \dot{A}),
$$

which is the upper-right block of the block exponential above.

## Reverse Rule

Given a cotangent $\bar{B}$:

$$
\bar{A} = L(A^{\mathsf{H}}, \bar{B}),
$$

which is the adjoint of the Fr\'echet derivative map under the Frobenius inner
product.

## Generality

The same block-matrix technique works for any analytic matrix function
$f$, not just the exponential:

$$
f\!\begin{pmatrix} A & E \\ 0 & A \end{pmatrix}
= \begin{pmatrix} f(A) & L_f(A, E) \\ 0 & f(A) \end{pmatrix}.
$$

PyTorch factors this pattern through the helper
`differential_analytic_matrix_function`.

## Computational cost

| Method | Cost relative to $\exp(A)$ |
|---|---|
| Block matrix ($2N \times 2N$) | about $8\times$ |
| Dedicated Fr\'echet scaling-and-squaring | about $3\times$ |
| Eigendecomposition shortcut | cheaper on paper, but unstable for non-normal $A$ |

## Implementation Correspondence

- `tenferro-rs/docs/AD/matrix_exp.md` uses the block-exponential construction
  as the main derivation.
- PyTorch's `differential_analytic_matrix_function` and
  `linalg_matrix_exp_differential` implement the same Mathias 1996 identity.
- The block matrix approach is simple but more expensive than a dedicated
  scaling-and-squaring Fr\'echet implementation.

## Verification

- compare the block-matrix Fr\'echet derivative against finite differences
- check JVP/VJP agreement on scalar losses of `matrix_exp(A)`

## References

1. R. Mathias, "A Chain Rule for Matrix Functions and Applications," 1996.
2. A. H. Al-Mohy and N. J. Higham, "Computing the Frechet Derivative of the
   Matrix Exponential," 2009.

## DB Status

`matrix_exp` is documented here as a known rule, but it is **not yet materialized**
in the current published `cases/` tree.
