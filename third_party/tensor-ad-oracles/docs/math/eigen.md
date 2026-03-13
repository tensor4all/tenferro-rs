# Hermitian Eigen AD Notes

## Forward Definition

$$
A = U \operatorname{diag}(E) U^\dagger,
\qquad
A \in \mathbb{C}^{N \times N},
\qquad
A = A^\dagger
$$

- $U \in \mathbb{C}^{N \times N}$ is unitary
- $E \in \mathbb{R}^N$ contains the real eigenvalues

## Reverse Rule

Given cotangents $\bar{E}$ and $\bar{U}$, compute a Hermitian cotangent
$\bar{A}$.

### Step 1: Build the $F$ matrix

$$
F_{ij} =
\begin{cases}
\dfrac{E_i - E_j}{(E_i - E_j)^2 + \eta}
\approx \dfrac{1}{E_i - E_j}, & i \neq j, \\
0, & i = j.
\end{cases}
$$

Regularization $\eta > 0$ avoids division by zero for degenerate eigenvalues.

### Step 2: Build the inner matrix $D$

$$
D =
\frac{1}{2}
\left(
F \odot (\bar{U}^\dagger U)
+ (F \odot (\bar{U}^\dagger U))^\dagger
\right)
+ \operatorname{diag}(\bar{E}).
$$

The symmetrization ensures that $D$ is Hermitian.

### Step 3: Conjugate back to the input basis

$$
\bar{A} = U D U^\dagger.
$$

The diagonal imaginary gauge of $U^\dagger dU$ drops out after symmetrization,
so the final cotangent stays inside the Hermitian tangent space.

## Derivation Sketch

Differentiate

$$
A U = U \operatorname{diag}(E)
$$

to obtain

$$
dA \, U + A \, dU = dU \, \operatorname{diag}(E) + U \, \operatorname{diag}(dE).
$$

Left-multiplying by $U^\dagger$ yields

$$
U^\dagger dA \, U =
U^\dagger dU \, \operatorname{diag}(E)
- \operatorname{diag}(E) U^\dagger dU
+ \operatorname{diag}(dE).
$$

If $\Omega = U^\dagger dU$, then $\Omega$ is skew-Hermitian, the diagonal of
$U^\dagger dA U$ gives $dE$, and the off-diagonal entries are divided by
$E_j - E_i$. Applying the adjoint of that split yields the formula above.

## Forward Rule

The Hermitian forward rule is the simplification of the general eigendecomposition
JVP:

$$
dE = \operatorname{diag}(U^\dagger dA U),
$$

$$
dU = U \left(F \odot (U^\dagger dA U - \operatorname{diag}(dE))\right),
$$

with the understanding that the skew-Hermitian gauge is projected away.

## Implementation Correspondence

- `tenferro-rs/docs/AD/eigen.md` writes the reverse rule through the explicit
  Hermitian inner matrix $D$; this note keeps that structure.
- PyTorch does not have a separate Hermitian kernel. It calls
  `linalg_eig_backward(..., is_hermitian=true)` and
  `linalg_eig_jvp(..., is_hermitian=true)`, which reduce to the same formulas
  with $V^{-1} = V^\dagger$.

## Verification

### Forward reconstruction

$$
\|A - U \operatorname{diag}(E) U^\dagger\|_F < \varepsilon,
\qquad
U^\dagger U \approx I.
$$

### Backward checks

- eigenvalues only: $f(A) = \sum_i E_i$
- eigenvectors only: scalar losses on a column of $U$
- compare JVP/VJP against finite differences on Hermitian perturbations

## References

1. M. Seeger et al., "Auto-Differentiating Linear Algebra," 2018.
2. M. B. Giles, "An extended collection of matrix derivative results for
   forward and reverse mode automatic differentiation," 2008.

## DB Families

<a id="family-values-vectors-abs"></a>
### `values_vectors_abs`

The DB publishes the real eigenvalues together with `vectors.abs()` to remove
the sign or phase ambiguity of the Hermitian eigenbasis.

<a id="family-eigvalsh-identity"></a>
### `eigvalsh/identity`

The eigenvalue-only Hermitian family reuses the diagonal part of the same rule.

<a id="family-gauge-ill-defined"></a>
### `gauge_ill_defined`

This family records expected failures for losses that are not invariant under
the eigenvector gauge freedom.
