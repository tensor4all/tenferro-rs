# Cholesky Decomposition AD Rules

## Forward

$$
A = L L^{\mathsf{H}}, \quad A \in \mathbb{C}^{N \times N} \text{ (Hermitian positive-definite)},\; L \text{ lower triangular}
$$

## Auxiliary operator

Define $\varphi(X) = \mathrm{tril}(X) - \tfrac{1}{2}\mathrm{diag}(X)$
(extract lower triangle, halve the diagonal).

Its adjoint is
$\varphi^*(X) = \tfrac{1}{2}(X + X^{\mathsf{H}} - \mathrm{diag}(X))$.

## Forward rule (JVP)

Given tangent $\dot{A}$ (Hermitian):

$$
\dot{L} = L \,\varphi\!\bigl(L^{-1}\,\dot{A}\,L^{-\mathsf{H}}\bigr)
$$

**Derivation.** Differentiating $A = L L^{\mathsf{H}}$:

$$
\dot{A} = \dot{L}\,L^{\mathsf{H}} + L\,\dot{L}^{\mathsf{H}}
$$

Left-multiply by $L^{-1}$, right-multiply by $L^{-\mathsf{H}}$:

$$
L^{-1}\dot{A}\,L^{-\mathsf{H}} = L^{-1}\dot{L} + (L^{-1}\dot{L})^{\mathsf{H}}
$$

Since $L^{-1}\dot{L}$ is lower triangular, inverting the symmetrization gives
$L^{-1}\dot{L} = \varphi(L^{-1}\dot{A}\,L^{-\mathsf{H}})$, hence $\dot{L} = L\,\varphi(\cdots)$.

**Algorithm:**

1. Solve $T \leftarrow L^{-1}\,\dot{A}$ (triangular solve, left)
2. Solve $T \leftarrow T\,L^{-\mathsf{H}}$ (triangular solve, right)
3. $T \leftarrow \mathrm{tril}(T) - \tfrac{1}{2}\mathrm{diag}(T)$
4. $\dot{L} \leftarrow L\,T$

## Reverse rule (VJP)

Given cotangent $\bar{L}$:

$$
\bar{A} = L^{-\mathsf{H}}\,\varphi^*\!\bigl(\mathrm{tril}(L^{\mathsf{H}}\bar{L})\bigr)\,L^{-1}
$$

**Derivation.** Taking the adjoint of the JVP linear map $\dot{A} \mapsto \dot{L}$:

$$
\delta\ell = \langle \bar{L},\,\dot{L}\rangle
= \mathrm{tr}\!\bigl(\bar{L}^{\mathsf{H}}\,L\,\varphi(L^{-1}\dot{A}\,L^{-\mathsf{H}})\bigr)
$$

Working through the adjoint chain:

1. Adjoint of $\varphi$: $\varphi^*$
2. Adjoint of left-multiply by $L$: left-multiply by $L^{\mathsf{H}}$, then project to lower triangle
3. Adjoint of $L^{-1}(\cdot)L^{-\mathsf{H}}$: $L^{-\mathsf{H}}(\cdot)L^{-1}$

This yields $\bar{A} = L^{-\mathsf{H}}\,\varphi^*(\mathrm{tril}(L^{\mathsf{H}}\bar{L}))\,L^{-1}$.

The output $\bar{A}$ is symmetric (Hermitian), consistent with the constraint on $A$.

**Algorithm:**

1. $S \leftarrow \mathrm{tril}(L^{\mathsf{H}}\bar{L})$
2. $S \leftarrow \tfrac{1}{2}(S + S^{\mathsf{H}} - \mathrm{diag}(S))$
   (equivalently: $S \leftarrow \tfrac{1}{2}(S + \mathrm{tril}(S,-1)^{\mathsf{H}})$)
3. Solve $\bar{A} \leftarrow L^{-\mathsf{H}}\,S$ (triangular solve, left)
4. Solve $\bar{A} \leftarrow \bar{A}\,L^{-1}$ (triangular solve, right)

## Implementation notes

- All operations are $O(N^3)$, same as the forward Cholesky.
- Use blocked algorithms (Murray 2016) for large matrices to exploit Level-3 BLAS.
- Never form $L^{-1}$ explicitly; always use triangular solves.
- For complex types, ensure diagonal of $\bar{A}$ is real ($A$ is Hermitian).

## References

1. Smith, S. P. (1995). "Differentiation of the Cholesky Algorithm."
   *J. Comput. Graph. Stat.*, 4(2), 134-147.
2. Giles, M. B. (2008). ["An extended collection of matrix derivative
   results for forward and reverse mode AD."](https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf)
3. Murray, I. (2016). ["Differentiation of the Cholesky decomposition."](https://arxiv.org/abs/1602.07527)
   arXiv:1602.07527.
4. Seeger, M. et al. (2017). ["Auto-Differentiating Linear Algebra."](https://arxiv.org/abs/1710.08717)
   arXiv:1710.08717.
5. PyTorch `FunctionsManual.cpp`: `cholesky_jvp` (L1962),
   `cholesky_backward` (L1983).
6. JAX `jax/_src/lax/linalg.py`: `_cholesky_jvp_rule`.
7. ChainRules.jl `src/rulesets/LinearAlgebra/factorization.jl`:
   `_cholesky_pullback_shared_code`.
