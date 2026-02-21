# Symmetric Eigendecomposition Reverse-Mode Rule (`eigen_rrule`)

## Forward

$$
A = U \mathrm{diag}(E) \, U^\dagger, \quad A \in \mathbb{C}^{N \times N}, \quad A = A^\dagger
$$

- $U \in \mathbb{C}^{N \times N}$, $U^\dagger U = I_N$ (unitary eigenvector matrix)
- $E \in \mathbb{R}^N$: real eigenvalues (since $A$ is Hermitian)

## Reverse rule

**Given:** cotangents $\bar{E}, \bar{U}$.

**Compute:** $\bar{A}$ (which must be Hermitian since $A$ is Hermitian and $\ell$ is real).

### Step 1: Build the $F$ matrix

$$
F_{ij} = \frac{E_i - E_j}{(E_i - E_j)^2 + \eta}
\approx \frac{1}{E_i - E_j}, \quad i \neq j
$$

$F_{ii} = 0$ (in the limit $\eta \to 0$).
Regularization $\eta > 0$ (default $10^{-40}$) prevents division by zero
for degenerate eigenvalues.

Note: for SVD the $F$ matrix uses $\sigma_j^2 - \sigma_i^2$; for eigendecomposition
it uses $E_i - E_j$ directly, since eigenvalues can be negative.

### Step 2: Build the inner matrix $D$

$$
D = \frac{1}{2}\left(F \odot (\bar{U}^\dagger U) + (F \odot (\bar{U}^\dagger U))^\dagger\right) + \mathrm{diag}(\bar{E})
$$

The symmetrization $\frac{1}{2}(X + X^\dagger)$ ensures $D$ is Hermitian, which
guarantees $\bar{A}$ is Hermitian.

When $\bar{U} = 0$: $D = \mathrm{diag}(\bar{E})$.

### Step 3: Apply the basis transformation

$$
\bar{A} = U D U^\dagger
$$

### Derivation

From $AU = U \mathrm{diag}(E)$, differentiating:

$$
dA \cdot U + A \cdot dU = dU \cdot \mathrm{diag}(E) + U \cdot \mathrm{diag}(dE)
$$

Left-multiply by $U^\dagger$ and use $U^\dagger A = \mathrm{diag}(E) U^\dagger$:

$$
U^\dagger dA \cdot U = U^\dagger dU \cdot \mathrm{diag}(E) - \mathrm{diag}(E) \cdot U^\dagger dU + \mathrm{diag}(dE)
$$

Define $\Omega = U^\dagger dU$ (skew-Hermitian from $U^\dagger U = I$).

**Off-diagonal** ($i \neq j$):

$$
(U^\dagger dA \, U)_{ij} = \Omega_{ij} (E_j - E_i)
\quad \Longrightarrow \quad
\Omega_{ij} = \frac{(U^\dagger dA \, U)_{ij}}{E_j - E_i}
$$

**Diagonal:**

$$
(U^\dagger dA \, U)_{ii} = dE_i
$$

The diagonal of $\Omega$ is purely imaginary (gauge freedom from the phase
of eigenvectors). Since $A$ is Hermitian and $\ell$ is real, this gauge freedom
does not affect $\bar{A}$, and the symmetrization in Step 2 projects it out.

For the pullback, applying the chain rule to $\delta\ell = \langle \bar{E}, dE \rangle + \langle \bar{U}, dU \rangle$
and using the separation above yields the formula in Steps 2-3.

## Verification

### Reconstruction check (forward)

$$
\|A - U \mathrm{diag}(E) U^\dagger\|_F < \varepsilon, \quad U^\dagger U \approx I
$$

### Gradient check (backward)

Scalar test functions (see [`docs/design/testing.md`](../design/testing.md)):

- **dE only:** $f(A) = \sum_i E_i$
- **dU only:** $f(A) = \mathrm{Re}(v^\dagger \mathrm{op} \, v)$, $v = U_{:,1}$

where $\mathrm{op}$ is a random Hermitian (or symmetric) matrix independent of $A$.

**Input matrix:** $A$ must be Hermitian. Generate as $A = B + B^\dagger$
where $B$ has entries from uniform $[-1, 1]$ (real and imaginary parts).

## References

1. M. Seeger *et al.*, "Auto-Differentiating Linear Algebra," 2018.
2. M. B. Giles, "An extended collection of matrix derivative results
   for forward and reverse mode automatic differentiation," 2008.
3. BackwardsLinalg.jl (GiggleLiu), `src/symeigen.jl`.
