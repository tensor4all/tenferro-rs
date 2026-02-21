# SVD Reverse-Mode Rule (`svd_rrule`)

## Forward

$$
A = U \Sigma V^\dagger, \quad A \in \mathbb{C}^{M \times N}, \quad K = \min(M, N)
$$

- $U \in \mathbb{C}^{M \times K}$, $U^\dagger U = I_K$
- $\Sigma = \operatorname{diag}(\sigma_1, \ldots, \sigma_K)$, $\sigma_i > 0$, descending
- $V \in \mathbb{C}^{N \times K}$, $V^\dagger V = I_K$

## Reverse rule

**Given:** cotangents $\bar{U}, \bar{S}, \bar{V}$ of a real scalar loss $\ell$, i.e. $\bar{U}_{ij} = \partial \ell / \partial U_{ij}^*$.

**Compute:** $\bar{A} = \partial \ell / \partial A^*$.

### Step 1: Build the $F$ matrix

$$
F_{ij} = \frac{\sigma_j^2 - \sigma_i^2}{(\sigma_j^2 - \sigma_i^2)^2 + \eta}
\approx \frac{1}{\sigma_j^2 - \sigma_i^2}, \quad i \neq j
$$

$F_{ii} = 0$ (in the limit $\eta \to 0$).
The regularization $\eta > 0$ (default $10^{-40}$)
prevents division by zero when singular values are degenerate.

Also define $S_{\text{inv},i} = \sigma_i / (\sigma_i^2 + \eta) \approx 1/\sigma_i$.

### Step 2: Accumulate the inner matrix

Compute the $K \times K$ inner matrix
$\Gamma = \Gamma_{\bar{U}} + \Gamma_{\bar{V}} + \Gamma_{\bar{S}}$
from whichever cotangents are nonzero:

#### From $\bar{U}$ (dU path)

$$
J = F \odot (U^\dagger \bar{U})
$$

$$
\Gamma_{\bar{U}} = (J + J^\dagger) \Sigma
  + \operatorname{diag}(i \cdot \operatorname{Im}(\operatorname{diag}(U^\dagger \bar{U})) \cdot S_\text{inv})
$$

**Derivation:**
Differentiating $U^\dagger U = I$ gives $U^\dagger dU$ skew-Hermitian.
The off-diagonal part of $U^\dagger dU$ is determined by $F$ and the SVD
differential equation. The diagonal of $U^\dagger dU$ is purely imaginary
(gauge freedom in the complex case), requiring the second term.
For real SVD, the diagonal term vanishes since $\operatorname{Im}(\operatorname{diag}(U^T \bar{U})) = 0$.

#### From $\bar{V}$ (dV path)

$$
K = F \odot (V^\dagger \bar{V})
$$

$$
\Gamma_{\bar{V}} = \Sigma (K + K^\dagger)
$$

Analogous to the $\bar{U}$ path but with $\Sigma$ on the left.
No imaginary-diagonal correction is needed because the gauge freedom
is already absorbed by the $\bar{U}$ term.

#### From $\bar{S}$ (dS path)

$$
\Gamma_{\bar{S}} = \operatorname{diag}(\bar{S})
$$

This is the simplest cotangent path: $\sigma_i$ are independent real parameters.

### Step 3: Core formula

$$
\bar{A}_\text{core} = U \Gamma V^\dagger
$$

### Step 4: Non-square corrections

When $A$ is not square, the thin SVD has $U$ or $V$ with fewer columns than rows.
The core formula only accounts for perturbations within the column space.
Perturbations in the orthogonal complement require additional terms.

**When $M > K$ (tall $A$, thin $U$):**

$$
\bar{A} \mathrel{+}= (\bar{U} - U U^\dagger \bar{U}) \operatorname{diag}(S_\text{inv}) V^\dagger
$$

The projector $(I_M - U U^\dagger)$ extracts the component of $\bar{U}$ in the
orthogonal complement of the column space of $U$.

**When $N > K$ (wide $A$, thin $V$):**

$$
\bar{A} \mathrel{+}= U \operatorname{diag}(S_\text{inv}) (\bar{V}^\dagger - \bar{V}^\dagger V V^\dagger)
$$

Analogous correction for the orthogonal complement of $V$.

### Complete formula

For general $M \times N$ with $K = \min(M, N)$:

$$
\bar{A} = U \Gamma V^\dagger
  + \mathbf{1}_{M > K} (I_M - U U^\dagger) \bar{U} \operatorname{diag}(S_\text{inv}) V^\dagger
  + \mathbf{1}_{N > K} U \operatorname{diag}(S_\text{inv}) (I_N - V V^\dagger) \bar{V}^\dagger
$$

where $\mathbf{1}$ denotes the indicator function and $\Gamma$ is defined in Step 2.

## Verification

### Reconstruction check (forward)

$$
\|A - U \operatorname{diag}(S) V^\dagger\|_F < \varepsilon
$$

$U^\dagger U \approx I$, $V^\dagger V \approx I$, $S \geq 0$ descending.

### Gradient check (backward)

Finite-difference gradient check with scalar test functions
(see [`docs/design/testing.md`](../design/testing.md) for details):

- **dU only:** $f(A) = \operatorname{Re}(\psi^\dagger H \psi)$, $\psi = U_{:,1}$
- **dV only:** $f(A) = \operatorname{Re}(\psi^\dagger H \psi)$, $\psi = V_{:,1}$
- **dS only:** $f(A) = \sum_i \sigma_i$
- **joint dU+dV:** $f(A) = \operatorname{Re}(U_{1,1}^* V_{1,1})$

where $H$ is a random Hermitian matrix independent of $A$.

## References

1. J. Townsend, "Differentiating the Singular Value Decomposition," 2016.
   <https://j-towns.github.io/papers/svd-derivative.pdf>
2. J.-G. Liu, "Einsum backward," 2019.
   <https://giggleliu.github.io/2019/04/02/einsumbp.html>
3. M. B. Giles, "An extended collection of matrix derivative results
   for forward and reverse mode automatic differentiation," 2008.
4. M. Seeger *et al.*, "Auto-Differentiating Linear Algebra," 2018.
