# SVD AD Notes

## Forward Definition

For a real or complex matrix

$$
A = U \Sigma V^\dagger, \qquad A \in \mathbb{C}^{M \times N}, \qquad K = \min(M, N),
$$

the thin SVD uses

- $U \in \mathbb{C}^{M \times K}$ with $U^\dagger U = I_K$
- $\Sigma = \operatorname{diag}(\sigma_1, \ldots, \sigma_K)$ with $\sigma_i > 0$
- $V \in \mathbb{C}^{N \times K}$ with $V^\dagger V = I_K$

PyTorch's `_linalg_svd` may be called with `full_matrices=True`, but its AD
formulas narrow back to the leading $K$ singular vectors before applying the
differential rules. The thin factors are therefore the mathematical source of
truth for the note and for the oracle DB.

## Reverse Rule

Given cotangents $\bar{U}$, $\bar{S}$, and $\bar{V}$ of a real scalar loss
$\ell$, compute the cotangent $\bar{A} = \partial \ell / \partial A^*$.
If an implementation returns `Vh = V^\dagger` instead of $V$, translate its
cotangent back via $\bar{V} = (\bar{Vh})^\dagger$ before substituting into the
formulas below.

### Step 1: Spectral-gap helpers

Define

$$
E_{ij} =
\begin{cases}
\sigma_j^2 - \sigma_i^2, & i \neq j, \\
1, & i = j,
\end{cases}
$$

and, equivalently, the stabilized inverse-gap matrix

$$
F_{ij} =
\begin{cases}
\dfrac{\sigma_j^2 - \sigma_i^2}{(\sigma_j^2 - \sigma_i^2)^2 + \eta}
\approx \dfrac{1}{\sigma_j^2 - \sigma_i^2}, & i \neq j, \\
0, & i = j,
\end{cases}
$$

with a small $\eta > 0$ for repeated or nearly repeated singular values.
Also define

$$
S_{\text{inv},i} = \frac{\sigma_i}{\sigma_i^2 + \eta} \approx \frac{1}{\sigma_i}.
$$

PyTorch writes the formulas using $E$, while `tenferro-rs` writes them using
$F$. They are the same off the diagonal.

### Step 2: Inner matrix split

Introduce

$$
\Gamma = \Gamma_{\bar{U}} + \Gamma_{\bar{V}} + \Gamma_{\bar{S}}.
$$

The three contributions are:

#### From $\bar{U}$

$$
J = F \odot (U^\dagger \bar{U})
$$

$$
\Gamma_{\bar{U}} =
(J + J^\dagger)\Sigma
+ \operatorname{diag}\!\left(i \, \operatorname{Im}(\operatorname{diag}(U^\dagger \bar{U})) \odot S_{\text{inv}}\right).
$$

The off-diagonal part reconstructs the skew-Hermitian variation allowed by the
constraint $U^\dagger U = I$. In the complex case, the diagonal imaginary term
captures the phase gauge freedom of the singular vectors. In the real case this
term vanishes.

#### From $\bar{V}$

$$
K = F \odot (V^\dagger \bar{V})
$$

$$
\Gamma_{\bar{V}} = \Sigma (K + K^\dagger).
$$

This is the right-singular-vector analogue of the $\bar{U}$ path. PyTorch's
`svd_backward` combines the same information through
$S ((V^\dagger \bar{V}) / E)$ inside its skew formulation.

#### From $\bar{S}$

$$
\Gamma_{\bar{S}} = \operatorname{diag}(\bar{S}).
$$

This is the direct singular-value path.

### Step 3: Core square-case formula

For the square-thin part,

$$
\bar{A}_{\text{core}} = U \Gamma V^\dagger.
$$

Equivalently, PyTorch writes the same expression as

$$
\bar{A}_{\text{core}} =
U \left[
\left(\frac{\operatorname{skew}(U^\dagger \bar{U})}{E}\right)\Sigma
+ \Sigma \left(\frac{\operatorname{skew}(V^\dagger \bar{V})}{E}\right)
+ \operatorname{diag}(\bar{S})
+ \operatorname{diag}\!\left(i \, \operatorname{Im}(\operatorname{diag}(U^\dagger \bar{U})) \odot S_{\text{inv}}\right)
\right] V^\dagger,
$$

where the diagonal imaginary term is grouped into the $U$ side using the gauge
constraint discussed below.

### Step 4: Non-square corrections

The thin factors only span a $K$-dimensional subspace. When $M \neq N$, the
parts of $\bar{U}$ or $\bar{V}$ orthogonal to that subspace contribute extra
terms.

#### Tall case: $M > K$

$$
\bar{A} \mathrel{+}= (I_M - U U^\dagger)\bar{U} \operatorname{diag}(S_{\text{inv}}) V^\dagger.
$$

#### Wide case: $N > K$

$$
\bar{A} \mathrel{+}= U \operatorname{diag}(S_{\text{inv}}) \bar{V}^\dagger (I_N - V V^\dagger).
$$

### Complete reverse rule

$$
\bar{A} = U \Gamma V^\dagger
+ \mathbf{1}_{M > K}(I_M - U U^\dagger)\bar{U}\operatorname{diag}(S_{\text{inv}})V^\dagger
+ \mathbf{1}_{N > K}U\operatorname{diag}(S_{\text{inv}})\bar{V}^\dagger(I_N - V V^\dagger).
$$

### Gauge condition and ill-defined losses

For complex SVD, $(U, V)$ is only defined up to a diagonal phase action

$$
(U, V) \mapsto (U L, V L), \qquad L = \operatorname{diag}(e^{i \theta_k}).
$$

The loss must therefore be invariant along this fibre. A necessary condition is

$$
\operatorname{Im}(\operatorname{diag}(U^\dagger \bar{U} + V^\dagger \bar{V})) = 0.
$$

PyTorch's `svd_backward` checks this numerically and raises an error when the
loss depends on the singular-vector phase. The DB's `gauge_ill_defined` family
records those expected failures.

## Forward Rule

The forward rule solves for $(dU, dS, dV)$ using the same spectral-gap
machinery. Define

$$
dP = U^\dagger (dA) V,
\qquad
dS = \operatorname{Re}(\operatorname{diag}(dP)),
\qquad
dX = dP - \operatorname{diag}(dS),
$$

and let $\operatorname{sym}(X) = X + X^\dagger$. Then:

### Square-thin part

$$
dU = U \left(\frac{\operatorname{sym}(dX \Sigma)}{E}
+ \operatorname{diag}\!\left(i \, \operatorname{Im}(\operatorname{diag}(dX)) \oslash (2 S)\right)\right)
$$

$$
dV = V \left(\frac{\operatorname{sym}(\Sigma dX)}{E}
- \operatorname{diag}\!\left(i \, \operatorname{Im}(\operatorname{diag}(dX)) \oslash (2 S)\right)\right)
$$

$$
dS = \operatorname{Re}(\operatorname{diag}(dP)).
$$

In the real case the diagonal imaginary terms vanish.

### Non-square forward corrections

For $M > K$,

$$
dU \mathrel{+}= (I_M - U U^\dagger)(dA) V \operatorname{diag}(S_{\text{inv}}).
$$

For $N > K$,

$$
dV \mathrel{+}= (I_N - V V^\dagger)(dA)^\dagger U \operatorname{diag}(S_{\text{inv}}).
$$

This is the form implemented in PyTorch's `linalg_svd_jvp`, up to the
convention that PyTorch returns `Vh = V^\dagger` and thus reports
$dVh = (dV)^\dagger$ directly.

## Numerical and Domain Notes

- The formulas assume distinct singular values. Repeated singular values make
  the inverse spectral-gap matrix unstable.
- If $A$ is rectangular, the reverse rule also assumes the active singular
  values are nonzero so that $S_{\text{inv}}$ is well defined.
- `full_matrices=True` does not make the extra singular-vector columns
  differentiable; implementations narrow to the thin factors before applying AD.
- Raw singular vectors are gauge-dependent, so the DB does not publish raw
  `U` or `Vh`.

## Verification

### Forward reconstruction

Check

$$
\|A - U \operatorname{diag}(S) V^\dagger\|_F < \varepsilon,
$$

together with $U^\dagger U \approx I$, $V^\dagger V \approx I$, and descending
nonnegative singular values.

### Backward checks

Representative scalar test functions:

- $dU$ only: $f(A) = \operatorname{Re}(\psi^\dagger H \psi)$ with $\psi = U_{:,1}$
- $dV$ only: $f(A) = \operatorname{Re}(\psi^\dagger H \psi)$ with $\psi = V_{:,1}$
- $dS$ only: $f(A) = \sum_i \sigma_i$
- mixed: $f(A) = \operatorname{Re}(U_{1,1}^* V_{1,1})$

where $H$ is a random Hermitian matrix independent of $A$.

## Implementation Correspondence

- `tenferro-rs/docs/AD/svd.md` writes the reverse rule by splitting
  $\Gamma_{\bar{U}}$, $\Gamma_{\bar{V}}$, and $\Gamma_{\bar{S}}`, and by making
  the `F` and `S_inv` helpers explicit. This note keeps that structure.
- PyTorch's `svd_backward` uses the equivalent $E$-matrix formulation together
  with skew/sym operators and an explicit gauge check in the complex case.
- PyTorch's `linalg_svd_jvp` uses the same thin-factor formulas and only pads
  zeros back out when it must return `full_matrices=True` shaped tangents.

## References

1. J. Townsend, "Differentiating the Singular Value Decomposition," 2016.
2. M. B. Giles, "An extended collection of matrix derivative results for
   forward and reverse mode automatic differentiation," 2008.
3. M. Seeger et al., "Auto-Differentiating Linear Algebra," 2018.

## DB Families

<a id="family-u-abs"></a>
### `u_abs`

The DB publishes `U.abs()` rather than raw `U` to remove sign and phase gauge
ambiguity.

<a id="family-s"></a>
### `s`

The DB publishes the singular values directly.

<a id="family-vh-abs"></a>
### `vh_abs`

The DB publishes the pair `(S, Vh.abs())` so that singular values remain paired
with a gauge-stable right singular-vector observable.

<a id="family-uvh-product"></a>
### `uvh_product`

The DB publishes `(U @ Vh, S)`, which preserves the gauge-invariant subspace
information while keeping the singular values explicit.

<a id="family-svdvals-identity"></a>
### `svdvals/identity`

The `svdvals` family is the singular-value-only projection of the same spectral
rule. It reuses the singular-value part of the SVD differential.

<a id="family-gauge-ill-defined"></a>
### `gauge_ill_defined`

This family records expected failure cases where the chosen loss is not
gauge-invariant and derivatives through the decomposition are intentionally ill
defined.
