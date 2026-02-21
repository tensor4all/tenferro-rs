// QR and LQ Reverse-Mode AD Rules (rrule)
// Reference: BackwardsLinalg.jl (GiggleLiu), Seeger et al. (2018), Liao et al.

#set page(margin: 2cm)
#set text(size: 11pt)

= QR Reverse-Mode Rule (`qr_rrule`)

== Forward

$
A = Q R, quad A in CC^(M times N)
$

- $Q in CC^(M times K)$, $Q^dagger Q = I_K$, $K = min(M, N)$
- $R in CC^(K times N)$, upper triangular (with positive real diagonal by convention)

== Helper: `copyltu` (Hermitianize from lower triangle)

$
op("copyltu")(M)_(i j) = cases(
  M_(i j) & "if" i > j,
  op("Re")(M_(i i)) & "if" i = j,
  overline(M_(j i)) & "if" i < j,
)
$

This constructs a Hermitian matrix from the lower triangular part of $M$.
For real matrices, this is equivalent to $op("tril")(M) + op("tril")(M)^T - op("diag")(op("diag")(M))$.

== Case 1: Full-rank ($M >= N$, square $R$)

When $K = N$ (i.e. $R$ is $N times N$ square upper triangular):

*Given:* cotangents $overline(Q) in CC^(M times N)$, $overline(R) in CC^(N times N)$.

=== Step 1: Build auxiliary matrix

$
W = R overline(R)^dagger - overline(Q)^dagger Q
$

Both terms are $N times N$. $R overline(R)^dagger$ carries information from $overline(R)$,
and $overline(Q)^dagger Q$ carries information from $overline(Q)$.

=== Step 2: Hermitianize

$
H = op("copyltu")(W)
$

*Why:* The constraint $Q^dagger Q = I$ means $Q^dagger d Q$ is skew-Hermitian.
In the QR backward, only the lower triangular part of $W$ contributes
(the upper triangular part is determined by the triangularity constraint of $R$).
Hermitianizing via `copyltu` correctly combines both constraints.

=== Step 3: Form the right-hand side

$
B = overline(Q) + Q H
$

=== Step 4: Triangular solve

$
overline(A) = B R^(-dagger)
$

where $R^(-dagger) = (R^dagger)^(-1) = (R^(-1))^dagger$.

*Implementation:* Solve $R X = B^dagger$ via forward substitution
(`trtrs!('U', 'N', 'N', R, B†)`), then $overline(A) = X^dagger$.

=== Complete formula (full-rank)

$
overline(A) = [overline(Q) + Q dot op("copyltu")(R overline(R)^dagger - overline(Q)^dagger Q)] R^(-dagger)
$

=== Derivation sketch

From $A = Q R$, differentiating: $d A = d Q dot R + Q dot d R$.

Left-multiply by $Q^dagger$: $Q^dagger d A = Q^dagger d Q dot R + d R$.

Since $Q^dagger d Q$ is skew-Hermitian and $d R$ is upper triangular,
we can separate the strictly lower triangular part (from $Q^dagger d Q$)
and the upper triangular part (from $d R$). This separation, combined
with the chain rule for cotangents, yields the formula above.

The key insight is that $R overline(R)^dagger - overline(Q)^dagger Q$ encodes
the cotangent information, and `copyltu` extracts the Hermitian part needed
for the $Q$ constraint.

== Case 2: Wide $R$ ($M < N$, $K = M$)

When $R$ has more columns than rows, partition:
$
R = [U | D], quad U in CC^(K times K) "upper triangular", quad D in CC^(K times (N - K))
$

and correspondingly:
$
A = Q [U | D] = [Q U | Q D]
$

Let $A_1 = A_(: , 1 : K)$ and $A_2 = A_(: , K + 1 : N)$.
Note $A_2 = Q D$, so $D = Q^dagger A_2$.

=== Backward

Partition $overline(R) = [overline(U) | overline(D)]$.

*From $D = Q^dagger A_2$:*
- $overline(Q) <- overline(Q) + A_2 overline(D)^dagger$
  #h(1em) (chain rule for $Q$ through $D = Q^dagger A_2$)
- $overline(A_2) = Q overline(D)$
  #h(1em) (chain rule for $A_2$ through $D = Q^dagger A_2$)

*For $A_1 = Q U$:*
Apply the full-rank QR backward with the augmented cotangent:
$
overline(A_1) = op("qr\\_back\\_fullrank")(Q, U, overline(Q) + A_2 overline(D)^dagger, overline(U))
$

*Combine:*
$
overline(A) = [overline(A_1) | overline(A_2)]
$

== Verification

=== Reconstruction check (forward)

$
norm(A - Q R)_F < epsilon, quad Q^dagger Q approx I, quad R "is upper triangular"
$

=== Gradient check (backward)

Scalar test function (exercises both $overline(Q)$ and $overline(R)$ jointly):

$
f(A) = op("Re")(v^dagger op("op") v + v_2^dagger op("op")_2 v_2), quad v = Q_(: , 1), quad v_2 = R_(2, :)
$

where $op("op"), op("op")_2$ are random Hermitian matrices independent of $A$.

#pagebreak()

= LQ Reverse-Mode Rule (`lq_rrule`)

== Forward

$
A = L Q, quad A in CC^(M times N)
$

- $L in CC^(M times K)$, lower triangular, $K = min(M, N)$
- $Q in CC^(K times N)$, $Q Q^dagger = I_K$

LQ is the transpose dual of QR: if $A = L Q$ then $A^dagger = Q^dagger L^dagger$
is a QR decomposition.

== Case 1: Full-rank ($N >= M$, square $L$)

When $K = M$ (i.e. $L$ is $M times M$ square lower triangular):

=== Step 1: Build auxiliary matrix

$
W = L^dagger overline(L) - overline(Q) Q^dagger
$

Both terms are $M times M$.

=== Step 2: Hermitianize

$
H = op("copyltu")(W)
$

=== Step 3: Form the right-hand side

$
C = H Q + overline(Q)
$

=== Step 4: Triangular solve

$
overline(A) = L^(-dagger) C
$

*Implementation:* Solve $L^dagger X = C$
(`trtrs!('L', 'C', 'N', L, C)`).

=== Complete formula (full-rank)

$
overline(A) = L^(-dagger) [op("copyltu")(L^dagger overline(L) - overline(Q) Q^dagger) Q + overline(Q)]
$

== Case 2: Tall $L$ ($M > N$, $K = N$)

Partition:
$
L = mat(U; D), quad U in CC^(K times K) "lower triangular", quad D in CC^((M - K) times K)
$

and $A = mat(U; D) Q$, so $A_1 = U Q$ and $A_2 = D Q$.

=== Backward

Partition $overline(L) = mat(overline(U); overline(D))$.

$
overline(A_1) = op("lq\\_back\\_fullrank")(U, Q, overline(U), overline(Q) + overline(D)^dagger A_2)
$

$
overline(A_2) = overline(D) Q
$

$
overline(A) = mat(overline(A_1); overline(A_2))
$

== Verification

Same gradient check method as QR, with scalar test function:

$
f(A) = op("Re")(v^dagger op("op") v + v_2^dagger op("op")_2 v_2), quad v = L_(: , 1), quad v_2 = Q_(2, :)
$

== References

+ M. Seeger, A. Hetzel, Z. Dai, E. Meissner, N. D. Lawrence,
  "Auto-Differentiating Linear Algebra," 2018.
+ H.-J. Liao, J.-G. Liu, L. Wang, T. Xiang,
  "Differentiable Programming Tensor Networks," 2019.
