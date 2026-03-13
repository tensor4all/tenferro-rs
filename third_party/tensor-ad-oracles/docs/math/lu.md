# LU AD Notes

## Forward Definition

$$
P A = L U, \qquad A \in \mathbb{C}^{M \times N}, \qquad K = \min(M, N)
$$

- $P \in \mathbb{R}^{M \times M}$ is a permutation matrix
- $L \in \mathbb{C}^{M \times K}$ is unit lower triangular
- $U \in \mathbb{C}^{K \times N}$ is upper triangular

The permutation is discrete metadata and is not differentiated.

## Notation

- $\mathrm{tril}_-(X)$: strictly lower-triangular part of $X$
- $\mathrm{triu}(X)$: upper-triangular part of $X$, including the diagonal

Since $L$ is unit lower triangular, its tangent and cotangent only use the
strictly lower-triangular part.

## Forward Rule

### Square case ($M = N$)

Given a tangent $\dot{A}$, define

$$
\dot{F} = L^{-1} P \dot{A} U^{-1}.
$$

Then

$$
\dot{L} = L \, \mathrm{tril}_-(\dot{F}),
\qquad
\dot{U} = \mathrm{triu}(\dot{F}) \, U.
$$

Differentiating $P A = L U$ gives

$$
P \dot{A} = \dot{L} U + L \dot{U},
$$

and after left-multiplying by $L^{-1}$ and right-multiplying by $U^{-1}$:

$$
\dot{F} = L^{-1} \dot{L} + \dot{U} U^{-1}.
$$

The two terms separate uniquely into strictly lower-triangular and
upper-triangular parts.

### Wide case ($M < N$)

Partition

$$
A = [A_1 \mid A_2],
\qquad
U = [U_1 \mid U_2],
$$

where $A_1, U_1 \in \mathbb{C}^{M \times M}$.
Define

$$
\dot{F} = L^{-1} P \dot{A}_1 U_1^{-1}.
$$

Then

$$
\dot{L} = L \, \mathrm{tril}_-(\dot{F}),
\qquad
\dot{U}_1 = \mathrm{triu}(\dot{F}) \, U_1,
$$

$$
\dot{U}_2 = L^{-1} P \dot{A}_2 - \mathrm{tril}_-(\dot{F}) U_2.
$$

### Tall case ($M > N$)

Partition

$$
L =
\begin{pmatrix}
L_1 \\
L_2
\end{pmatrix},
\qquad
P =
\begin{pmatrix}
P_1 \\
P_2
\end{pmatrix},
$$

with $L_1 \in \mathbb{C}^{N \times N}$ unit lower triangular and
$P_1 A = L_1 U$.
Define

$$
\dot{F} = L_1^{-1} P_1 \dot{A} U^{-1}.
$$

Then

$$
\dot{L}_1 = L_1 \, \mathrm{tril}_-(\dot{F}),
\qquad
\dot{U} = \mathrm{triu}(\dot{F}) \, U,
$$

$$
\dot{L}_2 = P_2 \dot{A} U^{-1} - L_2 \, \mathrm{triu}(\dot{F}).
$$

## Reverse Rule

Given cotangents $\bar{L}$ and $\bar{U}$ of a real scalar loss $\ell$:

### Square case ($M = N$)

Define

$$
\bar{F} = \mathrm{tril}_-(L^\dagger \bar{L}) + \mathrm{triu}(\bar{U} U^\dagger).
$$

Then

$$
\bar{A} = P^T L^{-\dagger} \bar{F} U^{-\dagger}.
$$

This is the adjoint of the triangular split in the forward rule.

### Wide case ($M < N$)

Partition $\bar{U} = [\bar{U}_1 \mid \bar{U}_2]$ and define

$$
\bar{H}_1 =
\left(
\mathrm{tril}_-(L^\dagger \bar{L} - \bar{U}_2 U_2^\dagger)
+ \mathrm{triu}(\bar{U}_1 U_1^\dagger)
\right) U_1^{-\dagger},
$$

$$
\bar{H}_2 = \bar{U}_2.
$$

Then

$$
\bar{A} = P^T L^{-\dagger} [\bar{H}_1 \mid \bar{H}_2].
$$

### Tall case ($M > N$)

Partition

$$
\bar{L} =
\begin{pmatrix}
\bar{L}_1 \\
\bar{L}_2
\end{pmatrix}
$$

and define

$$
\bar{H}_1 =
L_1^{-\dagger}
\left(
\mathrm{tril}_-(L_1^\dagger \bar{L}_1)
+ \mathrm{triu}(\bar{U} U^\dagger - L_2^\dagger \bar{L}_2)
\right),
$$

$$
\bar{H}_2 = \bar{L}_2.
$$

Then

$$
\bar{A} =
P^T
\begin{pmatrix}
\bar{H}_1 \\
\bar{H}_2
\end{pmatrix}
U^{-\dagger}.
$$

## Implementation Correspondence

- `tenferro-rs/docs/AD/lu.md` writes the rule in exactly this block-structured
  way, with separate square, wide, and tall cases.
- PyTorch's `linalg_lu_backward` and `linalg_lu_jvp` implement the same three
  cases using `tril(-1)` / `triu()` projections and triangular solves rather
  than explicit inverses.
- All $L^{-1} X$ and $X U^{-1}$ operations should be implemented as triangular
  solves.

## Verification

### Forward reconstruction

$$
\|P A - L U\|_F < \varepsilon
$$

with $L$ unit lower triangular and $U$ upper triangular.

### Backward checks

Representative scalar tests:

- $dL$ only: $f(A) = \operatorname{Re}(v^\dagger \operatorname{op} \, v)$ with
  $v = L_{:,1}$
- $dU$ only: $f(A) = \operatorname{Re}(v^\dagger \operatorname{op} \, v)$ with
  $v = U_{1,:}$
- mixed: $f(A) = \operatorname{Re}(L_{1,1}^* U_{1,1})$

where $\operatorname{op}$ is a random Hermitian matrix independent of $A$.

## References

1. S. Axen, "Differentiating the LU decomposition," 2021.
2. M. Seeger et al., "Auto-Differentiating Linear Algebra," 2018.

## DB Families

<a id="family-lu-identity"></a>
### `lu/identity`

The DB publishes the differentiable $(P, L, U)$ decomposition, with the
permutation treated as nondifferentiable metadata.

<a id="family-lu-factor-identity"></a>
### `lu_factor/identity`

The DB validates the packed factor tensor while treating pivots as
nondifferentiable metadata.

<a id="family-lu-factor-ex-identity"></a>
### `lu_factor_ex/identity`

The extended factorization uses the same derivative contract on the factor
tensor; status outputs remain metadata.
