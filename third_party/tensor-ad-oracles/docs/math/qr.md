# QR AD Notes

This note covers the reduced QR rule that is materialized in the DB and keeps
the transpose-dual LQ formulas from `tenferro-rs/docs/AD/qr.md` so that no
derivation detail is lost in the migration.

## QR Forward Definition

For

$$
A = Q R, \qquad A \in \mathbb{C}^{M \times N}, \qquad K = \min(M, N),
$$

the reduced QR factorization uses

- $Q \in \mathbb{C}^{M \times K}$ with $Q^\dagger Q = I_K$
- $R \in \mathbb{C}^{K \times N}$ upper triangular in its leading $K \times K$
  block

The differential identity is

$$
dA = dQ \, R + Q \, dR,
$$

with $Q^\dagger dQ$ skew-Hermitian.

## Helper Operators

### `copyltu`

$$
\operatorname{copyltu}(M)_{ij} =
\begin{cases}
M_{ij}, & i > j, \\
\operatorname{Re}(M_{ii}), & i = j, \\
\overline{M_{ji}}, & i < j.
\end{cases}
$$

This constructs the Hermitian matrix determined by the lower-triangular part of
$M$. In the real case it is the lower-triangular copy plus the mirrored strictly
upper part.

### `trilImInvAdjSkew`

For the wide reduced-QR backward we use

$$
\operatorname{trilImInvAdjSkew}(X) =
\begin{cases}
\operatorname{tril}(X - X^\top), & \text{real case}, \\
\operatorname{tril}(X - X^\dagger) \text{ with imaginary diagonal halved},
& \text{complex case}.
\end{cases}
$$

This is the adjoint helper appearing in PyTorch's `linalg_qr_backward` for the
$M < N$ case.

## Reverse Rule

Given cotangents $\bar{Q}$ and $\bar{R}$ of a real scalar loss $\ell$,
compute $\bar{A}$.

### QR Case 1: Full-rank ($M \geq N$)

Here $K = N$ and $R \in \mathbb{C}^{N \times N}$ is square upper triangular.

#### Step 1: Auxiliary matrix

$$
W = R \bar{R}^\dagger - \bar{Q}^\dagger Q.
$$

#### Step 2: Hermitian projection

$$
H = \operatorname{copyltu}(W).
$$

#### Step 3: Assemble the right-hand side

$$
B = \bar{Q} + Q H.
$$

#### Step 4: Triangular solve

$$
\bar{A} = B R^{-\dagger}.
$$

Implementation-wise this is a right solve with $R^\dagger$. PyTorch expresses
the same step as

$$
\bar{A} =
\left(\bar{Q} + Q \, \operatorname{syminvadj}(\operatorname{triu}(R \bar{R}^\dagger - Q^\dagger \bar{Q}))\right) R^{-\dagger},
$$

which is equivalent to the `copyltu` formulation.

#### Complete formula

$$
\bar{A} =
\left[\bar{Q} + Q \cdot \operatorname{copyltu}(R \bar{R}^\dagger - \bar{Q}^\dagger Q)\right] R^{-\dagger}.
$$

### QR Case 2: Wide Reduced QR ($M < N$)

Here $K = M$ and the leading square block

$$
R_1 = R_{:, 1:K} \in \mathbb{C}^{K \times K}
$$

controls the orthogonality-constrained part of the backward pass.

#### Step 1: Square auxiliary matrix

$$
X = Q^\dagger \bar{Q} - \bar{R} R^\dagger.
$$

#### Step 2: Leading-block cotangent

$$
\bar{A}_{\mathrm{lead}} =
Q \, \operatorname{trilImInvAdjSkew}(X) \, R_1^{-\dagger}.
$$

#### Step 3: Embed into the full width

Let $\pi^\*(Y) = [Y \mid 0]$ pad trailing zero columns so that
$\pi^\*(Y) \in \mathbb{C}^{K \times N}$.

#### Step 4: Add the direct $R$ path

$$
\bar{A} = \pi^\*(\bar{A}_{\mathrm{lead}}) + Q \bar{R}.
$$

PyTorch's `linalg_qr_backward` implements the same case as

$$
\bar{A} = Q \bar{R} + \pi^\*\!\left(
Q \, \operatorname{trilImInvAdjSkew}(Q^\dagger \bar{Q} - \bar{R} R^\dagger)
R_1^{-\dagger}\right).
$$

## Forward Rule

PyTorch's `linalg_qr_jvp` uses the same case split.

### Case $M \geq N$

Define $\operatorname{sym}(X) = X + X^\dagger$ and

$$
\operatorname{syminv}(X) = \operatorname{triu}(X)
- \tfrac{1}{2}\operatorname{diag}(X),
$$

the inverse of `sym` on upper-triangular matrices with real diagonal.
Then

$$
dR = \operatorname{syminv}\!\left(\operatorname{sym}(Q^\dagger (dA) R^{-1})\right) R,
$$

$$
dQ = (dA) R^{-1} - Q \left(dR R^{-1}\right).
$$

### Case $M < N$

Let $A_1$ be the leading $M \times M$ block of $A$, and define

$$
\operatorname{trilIm}(X) =
\begin{cases}
\operatorname{tril}(X, -1), & \text{real case}, \\
\operatorname{tril}(X) \text{ with real diagonal zeroed}, & \text{complex case}.
\end{cases}
$$

Its inverse on skew-Hermitian inputs is

$$
\operatorname{trilImInv}(X) =
\begin{cases}
X - X^\top, & \text{real case}, \\
X - X^\dagger \text{ with diagonal halved}, & \text{complex case}.
\end{cases}
$$

Then

$$
dQ =
Q \, \operatorname{trilImInv}\!\left(
\operatorname{trilIm}(Q^\dagger (dA)_1 R_1^{-1})\right),
$$

$$
dR = Q^\dagger (dA) - Q^\dagger dQ \, R.
$$

## LQ Reverse Rule

The transpose-dual LQ formulas are retained here because the original
`tenferro-rs` note grouped QR and LQ together.

### LQ Forward Definition

$$
A = L Q, \qquad A \in \mathbb{C}^{M \times N}.
$$

- $L \in \mathbb{C}^{M \times K}$ is lower triangular in its leading block
- $Q \in \mathbb{C}^{K \times N}$ satisfies $Q Q^\dagger = I_K$

### LQ Case 1: Full-rank ($N \geq M$)

With $K = M$, define

$$
W = L^\dagger \bar{L} - \bar{Q} Q^\dagger,
$$

$$
H = \operatorname{copyltu}(W),
$$

$$
C = H Q + \bar{Q},
$$

$$
\bar{A} = L^{-\dagger} C.
$$

### LQ Case 2: Tall $L$ ($M > N$)

Partition

$$
L =
\begin{pmatrix}
U \\
D
\end{pmatrix},
\qquad
U \in \mathbb{C}^{K \times K},
\qquad
D \in \mathbb{C}^{(M-K) \times K}.
$$

With the matching partition

$$
\bar{L} =
\begin{pmatrix}
\bar{U} \\
\bar{D}
\end{pmatrix},
$$

the backward pass is

$$
\bar{A}_1 =
\operatorname{lq\_back\_fullrank}\!\left(U, Q, \bar{U}, \bar{Q} + \bar{D}^\dagger A_2\right),
$$

$$
\bar{A}_2 = \bar{D} Q,
$$

$$
\bar{A} =
\begin{pmatrix}
\bar{A}_1 \\
\bar{A}_2
\end{pmatrix}.
$$

## Verification

### Forward reconstruction

Check

$$
\|A - Q R\|_F < \varepsilon,
\qquad
Q^\dagger Q \approx I,
\qquad
R \text{ upper triangular}.
$$

### Backward checks

A representative scalar functional that couples the $\bar{Q}$ and $\bar{R}$
paths is

$$
f(A) =
\operatorname{Re}(v^\dagger \operatorname{op} \, v
+ v_2^\dagger \operatorname{op}_2 \, v_2),
\qquad
v = Q_{:,1},
\qquad
v_2 = R_{2,:},
$$

with random Hermitian operators independent of $A$.

## Implementation Correspondence

- `tenferro-rs/docs/AD/qr.md` writes the rule in terms of `copyltu`,
  `trilImInvAdjSkew`, and the QR/LQ duality. This note keeps those helpers.
- PyTorch's `linalg_qr_backward` uses the same two reduced-QR cases:
  full-rank via `syminvadj(... ) R^{-H}` and wide reduced QR via the
  `pi*`-embedded `trilImInvAdjSkew` formula.
- PyTorch's `linalg_qr_jvp` mirrors the same case split in forward mode.

## References

1. M. Seeger, A. Hetzel, Z. Dai, E. Meissner, N. D. Lawrence,
   "Auto-Differentiating Linear Algebra," 2018.
2. H.-J. Liao, J.-G. Liu, L. Wang, T. Xiang,
   "Differentiable Programming Tensor Networks," 2019.

## DB Families

<a id="family-identity"></a>
### `identity`

The DB publishes the differentiable reduced-QR outputs directly.
