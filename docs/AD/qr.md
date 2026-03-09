# QR / LQ Reverse-Mode Rules

## QR Forward

$$
A = QR, \quad A \in \mathbb{C}^{M \times N}
$$

- $Q \in \mathbb{C}^{M \times K}$, $Q^\dagger Q = I_K$, $K = \min(M, N)$
- $R \in \mathbb{C}^{K \times N}$, upper triangular

## Helper: `copyltu` (Hermitianize from lower triangle)

$$
\mathrm{copyltu}(M)_{ij} = \begin{cases}
  M_{ij} & \text{if } i > j \\
  \mathrm{Re}(M_{ii}) & \text{if } i = j \\
  \overline{M_{ji}} & \text{if } i < j
\end{cases}
$$

This constructs a Hermitian matrix from the lower triangular part of $M$.
For real matrices: $\mathrm{tril}(M) + \mathrm{tril}(M)^T - \mathrm{diag}(\mathrm{diag}(M))$.

## QR Case 1: Full-rank ($M \geq N$, square $R$)

When $K = N$ (i.e. $R$ is $N \times N$ square upper triangular):

**Given:** cotangents $\bar{Q} \in \mathbb{C}^{M \times N}$, $\bar{R} \in \mathbb{C}^{N \times N}$.

### Step 1: Build auxiliary matrix

$$
W = R \bar{R}^\dagger - \bar{Q}^\dagger Q
$$

Both terms are $N \times N$.

### Step 2: Hermitianize

$$
H = \mathrm{copyltu}(W)
$$

**Why:** The constraint $Q^\dagger Q = I$ means $Q^\dagger dQ$ is skew-Hermitian.
In the QR backward, only the lower triangular part of $W$ contributes
(the upper triangular part is determined by the triangularity constraint of $R$).
Hermitianizing via `copyltu` correctly combines both constraints.

### Step 3: Form the right-hand side

$$
B = \bar{Q} + Q H
$$

### Step 4: Triangular solve

$$
\bar{A} = B R^{-\dagger}
$$

where $R^{-\dagger} = (R^\dagger)^{-1} = (R^{-1})^\dagger$.

**Implementation:** Solve $R X = B^\dagger$ via `trtrs('U', 'N', 'N', R, B†)`, then $\bar{A} = X^\dagger$.

### Complete formula (full-rank)

$$
\bar{A} = \left[\bar{Q} + Q \cdot \mathrm{copyltu}(R \bar{R}^\dagger - \bar{Q}^\dagger Q)\right] R^{-\dagger}
$$

### Derivation

From $A = QR$, differentiating: $dA = dQ \cdot R + Q \cdot dR$.

Left-multiply by $Q^\dagger$: $Q^\dagger dA = Q^\dagger dQ \cdot R + dR$.

Since $Q^\dagger dQ$ is skew-Hermitian and $dR$ is upper triangular,
we can uniquely separate the strictly lower triangular part (from $Q^\dagger dQ$)
and the upper triangular part (from $dR$). Applying the chain rule for
cotangents to this decomposition yields the formula above.

The key insight is that $R\bar{R}^\dagger - \bar{Q}^\dagger Q$ encodes
the cotangent information, and `copyltu` extracts the Hermitian part
consistent with the $Q$ orthogonality constraint.

## QR Case 2: Wide Reduced QR ($M < N$, $K = M$)

When $M < N$, the reduced QR still has

$$
Q \in \mathbb{C}^{K \times K}, \qquad R \in \mathbb{C}^{K \times N},
$$

with the leading square block

$$
R_1 = R_{:, 1:K} \in \mathbb{C}^{K \times K}.
$$

The reverse rule is most convenient to write directly in terms of
$Q$, $R$, $\bar{Q}$, and $\bar{R}$, without repartitioning the primal
matrix.

### Step 1: Build the square auxiliary matrix

$$
X = Q^\dagger \bar{Q} - \bar{R} R^\dagger \in \mathbb{C}^{K \times K}
$$

### Step 2: Project to the lower-triangular skew part

Define the adjoint helper used in the wide reduced-QR backward:

$$
\mathrm{trilImInvAdjSkew}(X) =
\begin{cases}
\mathrm{tril}(X - X^\top) & \text{real case}, \\
\mathrm{tril}(X - X^\dagger)\ \text{with imaginary diagonal halved} & \text{complex case}.
\end{cases}
$$

For the current `tenferro-linalg` implementation, the real-valued
specialization is the one in use.

### Step 3: Solve for the leading block

$$
\bar{A}_{\mathrm{lead}} = Q \, \mathrm{trilImInvAdjSkew}(X)\, R_1^{-\dagger}
$$

This produces the cotangent for the leading $K$ columns only.

### Step 4: Embed and add the direct $R$ path

Let $\pi^\*(Y)$ embed a $K \times K$ block back into the full-width
$K \times N$ matrix by padding trailing zero columns:

$$
\pi^\*(Y) = [Y \mid 0].
$$

Then the full reverse rule is

$$
\bar{A} = \pi^\*(\bar{A}_{\mathrm{lead}}) + Q \bar{R}.
$$

The first term accounts for the orthogonality constraint on $Q$ together
with the triangular structure of the leading block of $R$. The second term
is the direct cotangent flow through the explicit $Q \bar{R}$ contribution.

### Implementation note

This is the formula used by the current `tenferro-linalg::qr_rrule`
implementation, and it matches the reduced-QR backward written in
PyTorch's manual autograd formulas for the real case.

---

## LQ Forward

$$
A = LQ, \quad A \in \mathbb{C}^{M \times N}
$$

- $L \in \mathbb{C}^{M \times K}$, lower triangular, $K = \min(M, N)$
- $Q \in \mathbb{C}^{K \times N}$, $QQ^\dagger = I_K$

LQ is the transpose dual of QR: if $A = LQ$ then $A^\dagger = Q^\dagger L^\dagger$
is a QR decomposition.

## LQ Case 1: Full-rank ($N \geq M$, square $L$)

When $K = M$ (i.e. $L$ is $M \times M$ square lower triangular):

### Steps

$$
W = L^\dagger \bar{L} - \bar{Q} Q^\dagger
$$

$$
H = \mathrm{copyltu}(W)
$$

$$
C = HQ + \bar{Q}
$$

$$
\bar{A} = L^{-\dagger} C
$$

**Implementation:** Solve $L^\dagger X = C$ via `trtrs('L', 'C', 'N', L, C)`.

### Complete formula (full-rank)

$$
\bar{A} = L^{-\dagger} \left[\mathrm{copyltu}(L^\dagger \bar{L} - \bar{Q} Q^\dagger) Q + \bar{Q}\right]
$$

## LQ Case 2: Tall $L$ ($M > N$, $K = N$)

Partition:

$$
L = \begin{pmatrix} U \\ D \end{pmatrix}, \quad
U \in \mathbb{C}^{K \times K} \text{ lower triangular}, \quad
D \in \mathbb{C}^{(M-K) \times K}
$$

and $A = \begin{pmatrix} U \\ D \end{pmatrix} Q$, so $A_1 = UQ$ and $A_2 = DQ$.

### Backward

Partition $\bar{L} = \begin{pmatrix} \bar{U} \\ \bar{D} \end{pmatrix}$.

$$
\bar{A}_1 = \mathrm{lq\_back\_fullrank}(U, Q, \bar{U}, \bar{Q} + \bar{D}^\dagger A_2)
$$

$$
\bar{A}_2 = \bar{D} Q
$$

$$
\bar{A} = \begin{pmatrix} \bar{A}_1 \\ \bar{A}_2 \end{pmatrix}
$$

## Verification

### Reconstruction check (forward)

$$
\|A - QR\|_F < \varepsilon, \quad Q^\dagger Q \approx I, \quad R \text{ is upper triangular}
$$

### Gradient check (backward)

Scalar test function (exercises both $\bar{Q}$ and $\bar{R}$ jointly):

$$
f(A) = \mathrm{Re}(v^\dagger \mathrm{op} \, v + v_2^\dagger \mathrm{op}_2 \, v_2), \quad v = Q_{:,1}, \quad v_2 = R_{2,:}
$$

where $\mathrm{op}, \mathrm{op}_2$ are random Hermitian matrices independent of $A$.

## References

1. M. Seeger, A. Hetzel, Z. Dai, E. Meissner, N. D. Lawrence,
   "Auto-Differentiating Linear Algebra," 2018.
2. H.-J. Liao, J.-G. Liu, L. Wang, T. Xiang,
   "Differentiable Programming Tensor Networks," 2019.
