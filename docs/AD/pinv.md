# Pseudoinverse AD Rules (`pinv`)

## Forward

$$
A^+ = \operatorname{pinv}(A), \quad A \in \mathbb{C}^{M \times N}
$$

where $A^+$ is the Moore-Penrose pseudoinverse satisfying $A A^+ A = A$,
$A^+ A A^+ = A^+$, $(A A^+)^{\mathsf{H}} = A A^+$, $(A^+ A)^{\mathsf{H}} = A^+ A$.

**Assumption:** $A(t)$ has constant rank in a neighborhood of the
evaluation point. The pseudoinverse is not continuous at rank-changing points.

## Notation

- $P_{\mathrm{col}} = A A^+$: orthogonal projector onto $\operatorname{col}(A)$
- $P_{\mathrm{row}} = A^+ A$: orthogonal projector onto $\operatorname{row}(A)$

## Forward rule (JVP)

Given tangent $\dot{A}$ (Golub & Pereyra, 1973):

$$
\dot{A}^+ = -A^+\,\dot{A}\,A^+
  + (I - A^+ A)\,\dot{A}^{\mathsf{H}}\,(A^+)^{\mathsf{H}} A^+
  + A^+\,(A^+)^{\mathsf{H}}\,\dot{A}^{\mathsf{H}}\,(I - A A^+)
$$

**Three-term interpretation:**

1. $-A^+\dot{A}\,A^+$: analogous to $d(A^{-1}) = -A^{-1}\,dA\,A^{-1}$
2. $(I - P_{\mathrm{row}})\,\dot{A}^{\mathsf{H}}\,(A^+)^{\mathsf{H}} A^+$:
   correction from the null-space of $A$ (row-space projection perturbation)
3. $A^+(A^+)^{\mathsf{H}}\,\dot{A}^{\mathsf{H}}(I - P_{\mathrm{col}})$:
   correction from the left null-space of $A$ (column-space projection perturbation)

For full-rank square $A$: $P_{\mathrm{row}} = P_{\mathrm{col}} = I$,
Terms 2 and 3 vanish, recovering the standard inverse derivative.

### Derivation sketch

1. Differentiate $A^+ A A^+ = A^+$ to get an expression involving $\dot{A}^+$
   on both sides.
2. Differentiate $(A A^+ A)^{\mathsf{H}} = A^{\mathsf{H}}$ (MP condition 2)
   to isolate $A^+ A\,\dot{A}^+$ and $\dot{A}^+ A A^+$.
3. Substitute back to eliminate $\dot{A}^+$ from the RHS factors.

## Reverse rule (VJP)

Given cotangent $\bar{A}^+$ (same shape as $A^+$, i.e., $N \times M$):

$$
\bar{A} = -(A^+)^{\mathsf{H}}\,\bar{A}^+\,(A^+)^{\mathsf{H}}
  + (I - A A^+)\,(\bar{A}^+)^{\mathsf{H}}\,A^+\,(A^+)^{\mathsf{H}}
  + (A^+)^{\mathsf{H}}\,A^+\,(\bar{A}^+)^{\mathsf{H}}\,(I - A^+ A)
$$

This is structurally identical to the JVP formula with $\dot{A}$ replaced
by $\bar{A}^+$ and conjugate-transposition applied to the first term.

## Implementation notes

- Branch on $M \leq N$ vs $M > N$ to minimize intermediate matrix sizes
  (PyTorch optimization).
- Requires both $A$ and $A^+$ to be saved from the forward pass.
- The SVD-based alternative ($A = U \Sigma V^{\mathsf{H}}$,
  $A^+ = V \Sigma^{-1} U^{\mathsf{H}}$, differentiate through SVD) is
  less efficient than the direct Golub-Pereyra formula.
- The `atol`/`rtol` singular-value thresholding is not differentiated through.

## References

1. Golub, G. H. and Pereyra, V. (1973). ["The Differentiation of
   Pseudo-Inverses and Nonlinear Least Squares Problems Whose Variables
   Separate."](https://epubs.siam.org/doi/10.1137/0710036)
   *SIAM J. Numer. Anal.*, 10(2), 413-432.
2. Giles, M. B. (2008). ["An extended collection of matrix derivative
   results for forward and reverse mode AD."](https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf)
3. PyTorch `FunctionsManual.cpp`: `pinv_jvp` (L2091), `pinv_backward` (L2110).
4. JAX `jax/_src/numpy/linalg.py`: `_pinv_jvp`
   ([PR #2794](https://github.com/jax-ml/jax/pull/2794)).
