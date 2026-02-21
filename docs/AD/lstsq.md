# Least Squares Reverse-Mode Rule (`lstsq_rrule`)

## Forward

$$
x = \arg\min_x \|Ax - b\|_2^2, \quad A \in \mathbb{C}^{M \times N},\ b \in \mathbb{C}^M,\ M \geq N
$$

The solution satisfies the normal equations $A^\dagger A x = A^\dagger b$.
Via thin QR ($A = QR$): $x = R^{-1} Q^\dagger b$.

## Reverse rule

**Given:** cotangent $\bar{x} \in \mathbb{C}^N$ of a real scalar loss $\ell$.

**Compute:** $\bar{A} \in \mathbb{C}^{M \times N}$ and $\bar{b} \in \mathbb{C}^M$.

### Step 1: QR decompose $A$

$$
A = QR
$$

where $Q \in \mathbb{C}^{M \times N}$ ($Q^\dagger Q = I_N$) and $R \in \mathbb{C}^{N \times N}$ (upper triangular).

### Step 2: Solve two triangular systems

$$
y = R^{-\dagger} \bar{x}, \qquad z = R^{-1} y
$$

Note that $z = (R^\dagger R)^{-1} \bar{x} = (A^\dagger A)^{-1} \bar{x}$.

### Step 3: Compute cotangents

$$
\bar{b} = Q y
$$

$$
\bar{A} = r \, z^\dagger - \bar{b} \, x^\dagger
$$

where $r = b - Ax$ is the residual.

### Complete formulas

$$
\bar{b} = Q R^{-\dagger} \bar{x}
$$

$$
\bar{A} = (b - Ax)(R^{-1} R^{-\dagger} \bar{x})^\dagger - (Q R^{-\dagger} \bar{x}) x^\dagger
$$

### Derivation

The optimality condition is $A^\dagger(Ax - b) = 0$, i.e. $A^\dagger r = 0$ where $r = b - Ax$.

Differentiating the normal equations $A^\dagger A x = A^\dagger b$:

$$
dA^\dagger A x + A^\dagger dA \, x + A^\dagger A \, dx = dA^\dagger b + A^\dagger db
$$

Rearranging:

$$
A^\dagger A \, dx = A^\dagger db + dA^\dagger r - A^\dagger dA \, x
$$

$$
dx = (A^\dagger A)^{-1}(A^\dagger db + dA^\dagger r - A^\dagger dA \, x)
$$

For the pullback, let $z = (A^\dagger A)^{-1} \bar{x}$:

$$
\delta\ell = \langle \bar{x}, dx \rangle = \langle z, A^\dagger db + dA^\dagger r - A^\dagger dA \, x \rangle
$$

$$
= \langle Az, db \rangle + \langle r z^\dagger, dA \rangle - \langle Az \, x^\dagger, dA \rangle
$$

Reading off the cotangents:

$$
\bar{b} = Az = A (A^\dagger A)^{-1} \bar{x} = Q R R^{-1} R^{-\dagger} \bar{x} = Q R^{-\dagger} \bar{x} = Qy
$$

$$
\bar{A} = r z^\dagger - \bar{b} x^\dagger
$$

## Implementation notes

- Compute QR once and reuse for both triangular solves.
- Never form $(A^\dagger A)^{-1}$ explicitly; always use triangular solves.
- The residual $r = b - Ax$ may already be available from the forward pass.

## Verification

### Forward check

$$
\|Ax - b\|_2 \text{ is minimized}, \quad A^\dagger(Ax - b) \approx 0
$$

### Gradient check (backward)

Scalar test function (from BackwardsLinalg.jl):

$$
f(A, b) = x^\dagger \operatorname{op} \, x, \quad x = A \backslash b
$$

where $\operatorname{op}$ is a random Hermitian matrix independent of $A$ and $b$.

Two separate gradient checks:
- **$\bar{A}$:** fix $b$, perturb $A$
- **$\bar{b}$:** fix $A$, perturb $b$

## References

1. BackwardsLinalg.jl (GiggleLiu), `src/lstsq.jl`.
2. M. B. Giles, "An extended collection of matrix derivative results
   for forward and reverse mode automatic differentiation," 2008.
