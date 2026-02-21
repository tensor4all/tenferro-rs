// SVD Reverse-Mode AD Rule (rrule)
// Reference: BackwardsLinalg.jl (GiggleLiu), Townsend (2016), Giles (2008)

#set page(margin: 2cm)
#set text(size: 11pt)

= SVD Reverse-Mode Rule (`svd_rrule`)

== Forward

$
A = U Sigma V^dagger, quad
A in CC^(M times N), quad K = min(M, N)
$

- $U in CC^(M times K)$, $U^dagger U = I_K$
- $Sigma = op("diag")(sigma_1, dots, sigma_K)$, $sigma_i > 0$, descending
- $V in CC^(N times K)$, $V^dagger V = I_K$

== Reverse rule

*Given:* cotangents $overline(U), overline(S), overline(V)$ of a real scalar
loss $ell$, i.e.
$overline(U)_(i j) = (partial ell) / (partial U_(i j)^*)$.

*Compute:* $overline(A) = (partial ell) / (partial A^*)$.

=== Step 1: Build the $F$ matrix

$
F_(i j) = (sigma_j^2 - sigma_i^2) / ((sigma_j^2 - sigma_i^2)^2 + eta)
approx 1 / (sigma_j^2 - sigma_i^2), quad i eq.not j
$

$F_(i i) = 0$ (in the limit $eta -> 0$).
The regularization $eta > 0$ (default $10^(-40)$)
prevents division by zero when singular values are degenerate.

Also define $S_(op("inv"), i) = sigma_i / (sigma_i^2 + eta) approx 1 / sigma_i$.

=== Step 2: Accumulate the inner matrix

Compute the $K times K$ inner matrix
$Gamma = Gamma_(overline(U)) + Gamma_(overline(V)) + Gamma_(overline(S))$
from whichever cotangents are nonzero:

==== From $overline(U)$ (dU path)

$
J = F circle.small (U^dagger overline(U))
$

$
Gamma_(overline(U)) = (J + J^dagger) Sigma
  + op("diag")(i dot op("Im")(op("diag")(U^dagger overline(U))) dot.c S_(op("inv")))
$

*Derivation sketch:*
Differentiating $U^dagger U = I$ gives $U^dagger d U$ skew-Hermitian.
The off-diagonal part of $U^dagger d U$ is determined by $F$ and the SVD
differential equation. The diagonal of $U^dagger d U$ is purely imaginary
(gauge freedom in the complex case), requiring the second term.
For real SVD, the diagonal term vanishes since $op("Im")(op("diag")(U^T overline(U))) = 0$.

==== From $overline(V)$ (dV path)

$
K = F circle.small (V^dagger overline(V))
$

$
Gamma_(overline(V)) = Sigma (K + K^dagger)
$

Analogous to the $overline(U)$ path but with $Sigma$ on the left.
No imaginary-diagonal correction is needed because the gauge freedom
is already absorbed by the $overline(U)$ term.

==== From $overline(S)$ (dS path)

$
Gamma_(overline(S)) = op("diag")(overline(S))
$

This is the simplest cotangent path: $sigma_i$ are independent real parameters.

=== Step 3: Core formula

$
overline(A)_("core") = U Gamma V^dagger
$

=== Step 4: Non-square corrections

When $A$ is not square, the thin SVD has $U$ or $V$ with fewer columns than rows.
The core formula only accounts for perturbations within the column space.
Perturbations in the orthogonal complement require additional terms.

*When $M > K$ (tall $A$, thin $U$):*

$
overline(A) <- overline(A)_("core")
  + (overline(U) - U U^dagger overline(U)) op("diag")(S_(op("inv"))) V^dagger
$

The projector $(I_M - U U^dagger)$ extracts the component of $overline(U)$ in the
orthogonal complement of the column space of $U$.

*When $N > K$ (wide $A$, thin $V$):*

$
overline(A) <- overline(A)
  + U op("diag")(S_(op("inv"))) (overline(V)^dagger - overline(V)^dagger V V^dagger)
$

Analogous correction for the orthogonal complement of $V$.

=== Complete formula (combined)

For general $M times N$ with $K = min(M, N)$:

$
overline(A) = U Gamma V^dagger
  + bb(1)_(M > K) (I_M - U U^dagger) overline(U) op("diag")(S_(op("inv"))) V^dagger
  + bb(1)_(N > K) U op("diag")(S_(op("inv"))) (I_N - V V^dagger) overline(V)^dagger
$

where $bb(1)$ denotes the indicator function and $Gamma$ is defined in Step 2.

== Verification

=== Reconstruction check (forward)

$
norm(A - U op("diag")(S) V^dagger)_F < epsilon
$

$U^dagger U approx I$, $V^dagger V approx I$, $S >= 0$ descending.

=== Gradient check (backward)

Finite-difference gradient check with scalar test functions
(see `docs/design/testing.md` for details):

- *dU only:* $f(A) = op("Re")(psi^dagger H psi)$, $psi = U_(: , 1)$
- *dV only:* $f(A) = op("Re")(psi^dagger H psi)$, $psi = V_(: , 1)$
- *dS only:* $f(A) = sum_i sigma_i$
- *joint dU+dV:* $f(A) = op("Re")(U_(1,1)^* V_(1,1))$

where $H$ is a random Hermitian matrix independent of $A$.

== References

+ J. Townsend, "Differentiating the Singular Value Decomposition," 2016.
  https://j-towns.github.io/papers/svd-derivative.pdf
+ J.-G. Liu, "Erta-einsum backward," 2019.
  https://giggleliu.github.io/2019/04/02/einsumbp.html
+ M. B. Giles, "An extended collection of matrix derivative results
  for forward and reverse mode automatic differentiation," 2008.
+ M. Seeger _et al._, "Auto-Differentiating Linear Algebra," 2018.
