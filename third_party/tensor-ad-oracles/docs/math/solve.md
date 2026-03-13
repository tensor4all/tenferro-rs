# Solve AD Notes

## Forward Definition

For the left solve

$$
A X = B,
\qquad
A \in \mathbb{C}^{N \times N},
\qquad
B \in \mathbb{C}^{N \times K},
$$

the primal solution is

$$
X = A^{-1} B.
$$

## Forward Rule

Differentiate the defining equation:

$$
\dot{A} X + A \dot{X} = \dot{B}.
$$

Therefore

$$
\dot{X} = A^{-1}(\dot{B} - \dot{A} X).
$$

This is exactly the JVP implemented by PyTorch's `linalg_solve_jvp`.

## Reverse Rule

Given a cotangent $\bar{X}$:

$$
\delta \ell
= \langle \bar{X}, \dot{X} \rangle
= \langle A^{-\mathsf{H}} \bar{X}, \dot{B} \rangle
- \langle A^{-\mathsf{H}} \bar{X} X^{\mathsf{H}}, \dot{A} \rangle.
$$

Define

$$
G = A^{-\mathsf{H}} \bar{X}.
$$

Then

$$
\bar{B} = G,
\qquad
\bar{A} = -G X^{\mathsf{H}}.
$$

This is the same adjoint implemented by PyTorch's `linalg_solve_backward`.

## Triangular Solve

When $A$ is triangular, the same formulas apply with triangular solves replacing
the generic solve.

For lower-triangular $A$:

$$
\bar{A} = \mathrm{tril}(-G X^{\mathsf{H}}).
$$

For upper-triangular $A$:

$$
\bar{A} = \mathrm{triu}(-G X^{\mathsf{H}}).
$$

For unit-triangular matrices, the diagonal of $\bar{A}$ is additionally zeroed.
This matches PyTorch's `triangular_solve_jvp` and
`linalg_solve_triangular_backward`.

## Right-side solve

By transposition symmetry, the right solve $X A = B$ obeys

$$
\dot{X} A = \dot{B} - X \dot{A},
$$

$$
\bar{B} = \bar{X} A^{-\mathsf{H}},
\qquad
\bar{A} = -X^{\mathsf{H}} \bar{B}.
$$

## Structured Variants

- `solve_ex` shares the same derivative on the solution output; status outputs
  are nondifferentiable metadata.
- `solve_triangular` uses the same formulas with triangular projection.
- `lu_solve` reuses the solve cotangent while taking LU factors and pivots as
  primal inputs.
- `tensorsolve` is the indexed tensor analogue of the same implicit-system
  rule.

## Implementation Correspondence

- `tenferro-rs/docs/AD/solve.md` writes both the left/right solve identities and
  the triangular projection rules explicitly.
- PyTorch's `linalg_solve_jvp` and `linalg_solve_backward` implement the same
  two equations $dX = A^{-1}(dB - dA X)$ and $gA = -gB X^H$.
- Higher-order AD should solve against $A^\dagger$ directly rather than expose
  saved LU factors as differentiable objects.

## Verification

### Forward residual

$$
A X \approx B.
$$

### Backward checks

- perturb $A$ and compare the VJP against finite differences
- perturb $B$ and compare the VJP against finite differences
- for triangular solves, confirm the cotangent respects the triangular and
  unit-triangular structure

## References

1. M. B. Giles, "An extended collection of matrix derivative results for
   forward and reverse mode AD," 2008.
2. PyTorch `FunctionsManual.cpp`: `linalg_solve_jvp`,
   `linalg_solve_backward`, `triangular_solve_jvp`,
   `linalg_solve_triangular_backward`.

## DB Families

<a id="family-solve-identity"></a>
### `solve/identity`

The DB publishes the solution tensor directly.

<a id="family-solve-ex-identity"></a>
### `solve_ex/identity`

The DB validates the differentiable solution output; auxiliary execution-status
fields are treated as metadata.

<a id="family-solve-triangular-identity"></a>
### `solve_triangular/identity`

The DB applies the same solve differential with the triangular structure
enforced by the primal operator.

<a id="family-lu-solve-identity"></a>
### `lu_solve/identity`

The DB uses the solution observable for factor-backed solves as well.

<a id="family-tensorsolve-identity"></a>
### `tensorsolve/identity`

The DB treats `tensorsolve` as the indexed tensor analogue of linear solve.
