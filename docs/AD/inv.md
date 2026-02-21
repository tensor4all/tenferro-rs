# Matrix Inverse AD Rules (`inv`)

## Forward

$$
B = A^{-1}, \quad A \in \mathbb{C}^{N \times N}
$$

## Forward rule (JVP)

Differentiating $A B = I$:

$$
\dot{A}\,B + A\,\dot{B} = 0
$$

$$
\dot{B} = -B\,\dot{A}\,B
$$

## Reverse rule (VJP)

Given cotangent $\bar{B}$:

$$
\bar{A} = -B^{\mathsf{H}}\,\bar{B}\,B^{\mathsf{H}}
$$

**Derivation.** From the JVP:

$$
\delta\ell = \langle \bar{B},\,\dot{B}\rangle
= \mathrm{tr}(\bar{B}^{\mathsf{H}}(-B\,\dot{A}\,B))
= -\mathrm{tr}(B\,\bar{B}^{\mathsf{H}}\,B\,\dot{A})
= \langle -B^{\mathsf{H}}\bar{B}\,B^{\mathsf{H}},\,\dot{A}\rangle
$$

## Relationship to solve

`inv(A)` is the special case of `solve(A, I)` with $B = I$, $\dot{B} = 0$:

- JVP: $\dot{X} = A^{-1}(0 - \dot{A}\,A^{-1}) = -B\,\dot{A}\,B$
- VJP: $G = A^{-\mathsf{H}}\bar{X}$, $\bar{A} = -G\,(A^{-1})^{\mathsf{H}} = -B^{\mathsf{H}}\bar{X}\,B^{\mathsf{H}}$

## Implementation notes

- Prefer implementing via `solve` when possible, to avoid redundant code.
- For higher-order AD, use `solve` rather than explicit multiplication
  with the cached inverse.

## References

1. Giles, M. B. (2008). ["An extended collection of matrix derivative
   results for forward and reverse mode AD."](https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf)
2. Dwyer, P. S. and Macphail, M. S. (1948). "Symbolic Matrix
   Derivatives." *Ann. Math. Stat.*, 19(4), 517-534.
3. PyTorch `derivatives.yaml`: `linalg_inv_ex` (L906-909).
4. JAX `jax/_src/numpy/linalg.py`: `inv` delegates to `solve(A, I)`.
