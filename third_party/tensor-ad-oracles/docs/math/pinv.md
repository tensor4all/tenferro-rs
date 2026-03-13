# Pseudoinverse AD Notes

## Forward Definition

$$
A^+ = \operatorname{pinv}(A),
\qquad
A \in \mathbb{C}^{M \times N}
$$

where $A^+$ satisfies the Moore-Penrose identities. We assume the rank of $A$
is locally constant; the pseudoinverse is not continuous at rank-changing
points.

## Notation

- $P_{\mathrm{col}} = A A^+$: projector onto the column space of $A$
- $P_{\mathrm{row}} = A^+ A$: projector onto the row space of $A$

## Forward Rule

Given a tangent $\dot{A}$:

$$
\dot{A}^+ =
-A^+ \dot{A} A^+
+ (I - A^+ A)\dot{A}^\dagger (A^+)^\dagger A^+
+ A^+ (A^+)^\dagger \dot{A}^\dagger (I - A A^+).
$$

### Three-term interpretation

1. $-A^+ \dot{A} A^+$ is the inverse-like core term.
2. $(I - P_{\mathrm{row}})\dot{A}^\dagger (A^+)^\dagger A^+$ corrects the row
   space.
3. $A^+ (A^+)^\dagger \dot{A}^\dagger (I - P_{\mathrm{col}})$ corrects the
   column space.

For full-rank square $A$, the projector corrections vanish and the rule reduces
to the usual inverse derivative.

## Reverse Rule

Given a cotangent $\bar{A}^+$:

$$
\bar{A} =
-(A^+)^\dagger \bar{A}^+ (A^+)^\dagger
+ (I - A A^+) (\bar{A}^+)^\dagger A^+ (A^+)^\dagger
+ (A^+)^\dagger A^+ (\bar{A}^+)^\dagger (I - A^+ A).
$$

This is the adjoint counterpart of the same three-term structure.

## Implementation Correspondence

- `tenferro-rs/docs/AD/pinv.md` follows the classical Golub-Pereyra formulas and
  makes the projector interpretation explicit.
- PyTorch's `pinv_jvp` and `pinv_backward` implement algebraically equivalent
  forms but branch on $M \leq N$ versus $M > N$ to reduce intermediate matrix
  sizes.
- The `atol` / `rtol` thresholding used to define the primal pseudoinverse is
  treated as fixed metadata, not as a differentiable branch.

## Verification

### Moore-Penrose identities

Check the standard projector equalities:

$$
A A^+ A \approx A,
\qquad
A^+ A A^+ \approx A^+.
$$

### Backward checks

Compare JVP/VJP against finite differences away from rank changes.

## References

1. G. H. Golub and V. Pereyra, "The Differentiation of Pseudo-Inverses and
   Nonlinear Least Squares Problems Whose Variables Separate," 1973.
2. M. B. Giles, "An extended collection of matrix derivative results for
   forward and reverse mode automatic differentiation," 2008.

## DB Families

<a id="family-pinv-identity"></a>
### `pinv/identity`

The DB publishes the pseudoinverse tensor directly.

<a id="family-pinv-hermitian-identity"></a>
### `pinv_hermitian/identity`

The DB uses the Hermitian pseudoinverse convention for the primal operator but
the same projector-based derivative structure.

<a id="family-pinv-singular-identity"></a>
### `pinv_singular/identity`

This family captures the singular-input regime explicitly while keeping the same
observable shape.
