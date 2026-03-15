# Linear Solve AD Rules (`solve`, `solve_triangular`)

## Forward

$$
AX = B, \quad A \in \mathbb{C}^{N \times N},\; B \in \mathbb{C}^{N \times K}
$$

Solution: $X = A^{-1}B$.

## Forward rule (JVP)

Differentiating $AX = B$:

$$
\dot{A}\,X + A\,\dot{X} = \dot{B}
$$

$$
\dot{X} = A^{-1}(\dot{B} - \dot{A}\,X)
$$

i.e., solve $A\,\dot{X} = \dot{B} - \dot{A}\,X$ reusing the LU factorization of $A$.

## Reverse rule (VJP)

Given cotangent $\bar{X}$:

$$
\delta\ell = \langle \bar{X},\,\dot{X}\rangle
= \langle \bar{X},\, A^{-1}(\dot{B} - \dot{A}\,X)\rangle
= \langle A^{-\mathsf{H}}\bar{X},\, \dot{B}\rangle
- \langle A^{-\mathsf{H}}\bar{X},\, \dot{A}\,X\rangle
$$

Define $G = A^{-\mathsf{H}}\bar{X}$ (solve $A^{\mathsf{H}} G = \bar{X}$). Then:

$$
\bar{B} = G, \qquad \bar{A} = -G\,X^{\mathsf{H}}
$$

**Derivation.** The second term:
$\langle G,\, \dot{A}\,X\rangle = \mathrm{tr}(G^{\mathsf{H}}\,\dot{A}\,X)
= \mathrm{tr}(X\,G^{\mathsf{H}}\,\dot{A})
= \langle G\,X^{\mathsf{H}},\, \dot{A}\rangle$,
so $\bar{A} = -G\,X^{\mathsf{H}}$.

## Triangular solve

When $A$ is lower (or upper) triangular, the same formulas apply with
triangular solves instead of general LU solves. Additionally, the
cotangent $\bar{A}$ must be projected onto the triangular structure:

$$
\bar{A} = \mathrm{tril}(-G\,X^{\mathsf{H}}) \quad \text{(lower triangular case)}
$$

$$
\bar{A} = \mathrm{triu}(-G\,X^{\mathsf{H}}) \quad \text{(upper triangular case)}
$$

For unit-triangular matrices, the diagonal of $\bar{A}$ is additionally
zeroed out.

## PyTorch alignment (reference)

Reference implementation in PyTorch:

- `torch/csrc/autograd/FunctionsManual.cpp`
  - `triangular_solve_jvp`
  - `linalg_solve_triangular_forward_AD`
  - `linalg_solve_triangular_backward`

Equivalent formulas used there:

- Forward/JVP:
  - $dX = A^{-1}(dB - dAX)$
  - with projection of $dA$ to triangular tangent space (`triu`/`tril`)
- Backward/VJP:
  - $G_B = A^{-H} G_X$
  - $G_A = -G_B X^H$
  - then triangular projection of $G_A$

These match the formulas above and are the compatibility target for tenferro.

## tenferro implementation mapping

- `solve_frule`: implemented (general solve)
- `solve_rrule`: implemented (general solve)
- `solve_triangular_frule`: implemented (triangular JVP with triangular projection)
- `solve_triangular_rrule`: implemented (triangular VJP with triangular projection;
  real and complex scalars use adjoint-based formulas)
- `tenferro::solve_ad(...).run()`: reverse node now registers a
  local pullback backed by `solve_rrule`
- `tenferro::ad::solve_triangular_rrule`: implemented as a
  stateless `tenferro`-frontend wrapper for integration code
- `tenferro::solve_triangular_ad(...).run()`: reverse node now
  registers a local pullback, so `tenferro::ad::pullback` can
  execute VJP without exposing tape internals

## Right-side solve ($XA = B$)

By transposition symmetry:

| | Left ($AX = B$) | Right ($XA = B$) |
|---|---|---|
| JVP | $A\,\dot{X} = \dot{B} - \dot{A}\,X$ | $\dot{X}\,A = \dot{B} - X\,\dot{A}$ |
| $\bar{B}$ | $A^{-\mathsf{H}}\bar{X}$ | $\bar{X}\,A^{-\mathsf{H}}$ |
| $\bar{A}$ | $-G\,X^{\mathsf{H}}$ | $-X^{\mathsf{H}}\,G$ |

## Implementation notes

- Reuse the LU (or triangular) factorization from the forward pass for
  both JVP and VJP solves.
- Never form $A^{-1}$ explicitly.
- For higher-order AD, call `solve(A.mH(), ...)` instead of
  `lu_solve(LU, pivots, ...)` so the solve itself is differentiable
  (PyTorch convention).

## References

1. Giles, M. B. (2008). ["An extended collection of matrix derivative
   results for forward and reverse mode AD."](https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf)
2. PyTorch `FunctionsManual.cpp`: `linalg_solve_jvp` (L6052),
   `linalg_solve_backward` (L6084), triangular solve backward (L4541).
3. ChainRules.jl `src/rulesets/LinearAlgebra/structured.jl`:
   `rrule` for `\` and `/`.
4. JAX `jax/_src/lax/linalg.py`: `triangular_solve` JVP and transpose rules.
