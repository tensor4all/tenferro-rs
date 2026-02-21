# Matrix Exponential AD Rules (`matrix_exp`)

## Forward

$$
B = \exp(A) = \sum_{k=0}^{\infty} \frac{A^k}{k!}, \quad A \in \mathbb{C}^{N \times N}
$$

## Frechet derivative

The Frechet derivative of the matrix exponential at $A$ in direction $E$ is:

$$
L(A, E) = \frac{d}{dt}\exp(A + tE)\Big|_{t=0}
= \int_0^1 \exp(sA)\,E\,\exp((1-s)A)\,ds
$$

## Block matrix formula (Mathias 1996)

Both JVP and VJP reduce to computing a single matrix exponential of a
$2N \times 2N$ block upper-triangular matrix:

$$
\exp\!\begin{pmatrix} A & E \\ 0 & A \end{pmatrix}
= \begin{pmatrix} \exp(A) & L(A,E) \\ 0 & \exp(A) \end{pmatrix}
$$

The Frechet derivative $L(A, E)$ is the upper-right $N \times N$ block.

**Proof sketch.** Let $M = \begin{pmatrix}A & E \\ 0 & A\end{pmatrix}$.
Then $M^k = \begin{pmatrix}A^k & S_k \\ 0 & A^k\end{pmatrix}$ where
$S_k = \sum_{j=0}^{k-1} A^j E A^{k-1-j}$.
Summing: $\exp(M) = \begin{pmatrix}\exp(A) & L(A,E) \\ 0 & \exp(A)\end{pmatrix}$.

## Forward rule (JVP)

Given tangent $\dot{A}$:

$$
\dot{B} = L(A, \dot{A})
= \text{upper-right block of } \exp\!\begin{pmatrix} A & \dot{A} \\ 0 & A \end{pmatrix}
$$

## Reverse rule (VJP)

Given cotangent $\bar{B}$:

$$
\bar{A} = L(A^{\mathsf{H}}, \bar{B})
= \text{upper-right block of } \exp\!\begin{pmatrix} A^{\mathsf{H}} & \bar{B} \\ 0 & A^{\mathsf{H}} \end{pmatrix}
$$

**Derivation.** The adjoint of the linear map $E \mapsto L(A, E)$ is:

$$
\langle \bar{B},\, L(A, E) \rangle
= \int_0^1 \operatorname{tr}\!\bigl(\bar{B}^{\mathsf{H}}\,\exp(sA)\,E\,\exp((1{-}s)A)\bigr)\,ds
= \operatorname{tr}\!\Bigl(\bigl[\int_0^1 \exp(sA^{\mathsf{H}})\,\bar{B}\,\exp((1{-}s)A^{\mathsf{H}})\,ds\bigr]^{\mathsf{H}} E\Bigr)
= \langle L(A^{\mathsf{H}}, \bar{B}),\, E \rangle
$$

The only difference between JVP and VJP is whether the diagonal blocks
contain $A$ or $A^{\mathsf{H}}$.

## Generality of the block matrix approach

The block matrix technique (Mathias 1996) works for **any analytic matrix
function** $f$, not just $\exp$:

$$
f\!\begin{pmatrix} A & E \\ 0 & A \end{pmatrix}
= \begin{pmatrix} f(A) & L_f(A,E) \\ 0 & f(A) \end{pmatrix}
$$

PyTorch uses a single generic function `differential_analytic_matrix_function`
for both forward and reverse modes of any analytic matrix function.

## Computational cost

| Method | Cost (relative to $\exp(A)$) |
|--------|------------------------------|
| Block matrix ($2N \times 2N$) | $\sim 8\times$ |
| Al-Mohy & Higham SPS (2009) | $\sim 3\times$ |
| Eigendecomposition | $\sim 1.5{-}2\times$ (poor accuracy for non-normal $A$) |

## Implementation notes

- The block matrix approach is simple but $8\times$ slower. For production,
  the Al-Mohy & Higham (2009) scaling-and-squaring pushforward is preferred.
- For real $A$: $A^{\mathsf{H}} = A^{\mathsf{T}}$.

## References

1. Mathias, R. (1996). "A Chain Rule for Matrix Functions and
   Applications." *SIAM J. Matrix Anal. Appl.*, 17(3), 610-620.
2. Al-Mohy, A. H. and Higham, N. J. (2009). ["Computing the Frechet
   Derivative of the Matrix Exponential, with an Application to
   Condition Number Estimation."](https://epubs.siam.org/doi/10.1137/080716426)
   *SIAM J. Matrix Anal. Appl.*, 30(4), 1639-1657.
3. Najfeld, I. and Havel, T. F. (1995). "Derivatives of the Matrix
   Exponential and Their Computation." *Adv. Appl. Math.*, 16(3), 321-375.
4. Higham, N. J. (2008). *Functions of Matrices: Theory and Computation.* SIAM.
5. Van Loan, C. F. (1978). "Computing Integrals Involving the Matrix
   Exponential." *IEEE Trans. Automat. Control*, AC-23, 395-404.
6. PyTorch `FunctionsManual.cpp`:
   `differential_analytic_matrix_function` (L4199),
   `linalg_matrix_exp_differential` (L4247).
7. JAX `jax/_src/scipy/linalg.py`: `expm_frechet` via `jax.jvp(expm, ...)`.
