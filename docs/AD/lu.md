# LU Reverse-Mode Rule (`lu_rrule`)

## Forward

$$
PA = LU, \quad A \in \mathbb{C}^{M \times N}
$$

- $P \in \mathbb{R}^{M \times M}$: permutation matrix ($PP^T = I$, discrete, not differentiable)
- $L \in \mathbb{C}^{M \times K}$: **unit** lower triangular ($L_{ii} = 1$), $K = \min(M, N)$
- $U \in \mathbb{C}^{K \times N}$: upper triangular

## Notation

- $\operatorname{tril}_-(X)$: **strictly** lower triangular part of $X$ (zero diagonal)
- $\operatorname{triu}(X)$: upper triangular part of $X$ (including diagonal)

Since $L$ is *unit* lower triangular, $dL$ has zero diagonal, so:
- $\dot{L}$ (tangent) is strictly lower triangular
- $\bar{L}$ (cotangent) contributes only through its strictly lower triangular part

## Pushforward (frule)

**Given:** tangent $\dot{A}$.

### Square case ($M = N$)

Define:

$$
\dot{F} = L^{-1} P \dot{A} U^{-1}
$$

This intermediate separates uniquely into strictly-lower and upper-triangular parts:
- $L^{-1} \dot{L}$ is strictly lower triangular (since $L^{-1}$ is unit lower triangular and $\dot{L}$ is strictly lower triangular)
- $\dot{U} U^{-1}$ is upper triangular

Therefore:

$$
\dot{L} = L \operatorname{tril}_-(\dot{F}), \qquad
\dot{U} = \operatorname{triu}(\dot{F}) \, U
$$

**Derivation:** From $PA = LU$, differentiating ($P$ constant):

$$
P \dot{A} = \dot{L} U + L \dot{U}
$$

Left-multiply by $L^{-1}$, right-multiply by $U^{-1}$:

$$
\dot{F} = L^{-1} P \dot{A} U^{-1} = L^{-1} \dot{L} + \dot{U} U^{-1}
$$

The first term is strictly lower triangular, the second is upper triangular.
Since these triangular parts are disjoint, they separate uniquely.

### Wide case ($M < N$)

Partition $A = [A_1 \mid A_2]$, $U = [U_1 \mid U_2]$ where $A_1, U_1$ are $M \times M$ and $A_2, U_2$ are $M \times (N-M)$.

From $PA_1 = LU_1$ (square case) and $PA_2 = LU_2$:

$$
\dot{F} = L^{-1} P \dot{A}_1 U_1^{-1}
$$

$$
\dot{L} = L \operatorname{tril}_-(\dot{F}), \qquad
\dot{U}_1 = \operatorname{triu}(\dot{F}) \, U_1
$$

$$
\dot{U}_2 = L^{-1} P \dot{A}_2 - \operatorname{tril}_-(\dot{F}) \, U_2
$$

The last equation follows from differentiating $U_2 = L^{-1} P A_2$:
$\dot{U}_2 = L^{-1}(P \dot{A}_2 - \dot{L} U_2) = L^{-1} P \dot{A}_2 - \operatorname{tril}_-(\dot{F}) U_2$.

### Tall case ($M > N$)

Partition $L = \begin{pmatrix} L_1 \\ L_2 \end{pmatrix}$ where $L_1$ is $N \times N$ (unit lower triangular) and $L_2$ is $(M-N) \times N$.

Correspondingly partition $P = \begin{pmatrix} P_1 \\ P_2 \end{pmatrix}$ so that $P_1 A = L_1 U$ and $P_2 A = L_2 U$.

$$
\dot{F} = L_1^{-1} P_1 \dot{A} \, U^{-1}
$$

$$
\dot{L}_1 = L_1 \operatorname{tril}_-(\dot{F}), \qquad
\dot{U} = \operatorname{triu}(\dot{F}) \, U
$$

$$
\dot{L}_2 = P_2 \dot{A} \, U^{-1} - L_2 \operatorname{triu}(\dot{F})
$$

## Pullback (rrule)

**Given:** cotangents $\bar{L}, \bar{U}$ of a real scalar loss $\ell$.

### Square case ($M = N$)

Define:

$$
\bar{F} = \operatorname{tril}_-(L^\dagger \bar{L}) + \operatorname{triu}(\bar{U} U^\dagger)
$$

Then:

$$
\bar{A} = P^T L^{-\dagger} \bar{F} \, U^{-\dagger}
$$

**Derivation:** From the pushforward, $\delta\ell = \langle \bar{L}, \dot{L} \rangle + \langle \bar{U}, \dot{U} \rangle$ (Frobenius inner product).

Substituting $\dot{L} = L \operatorname{tril}_-(\dot{F})$ and $\dot{U} = \operatorname{triu}(\dot{F}) U$:

$$
\delta\ell = \operatorname{Re}\operatorname{tr}\!\left(
  \operatorname{tril}_-(L^\dagger \bar{L})^\dagger \dot{F}
  + \operatorname{triu}(\bar{U} U^\dagger)^\dagger \dot{F}
\right)
= \operatorname{Re}\operatorname{tr}(\bar{F}^\dagger \dot{F})
$$

where we used the identities:
- $\langle X, \operatorname{tril}_-(Y) \rangle = \langle \operatorname{tril}_-(X), Y \rangle$
  (strictly-lower projection is self-adjoint)
- $\langle X, \operatorname{triu}(Y) \rangle = \langle \operatorname{triu}(X), Y \rangle$
  (upper-triangular projection is self-adjoint)

Substituting $\dot{F} = L^{-1} P \dot{A} U^{-1}$:

$$
\delta\ell = \operatorname{Re}\operatorname{tr}\!\left(
  (P^T L^{-\dagger} \bar{F} U^{-\dagger})^\dagger \dot{A}
\right)
$$

So $\bar{A} = P^T L^{-\dagger} \bar{F} \, U^{-\dagger}$.

### Wide case ($M < N$)

Partition $\bar{U} = [\bar{U}_1 \mid \bar{U}_2]$.

Define:

$$
\bar{H}_1 = \left(
  \operatorname{tril}_-(L^\dagger \bar{L} - \bar{U}_2 U_2^\dagger)
  + \operatorname{triu}(\bar{U}_1 U_1^\dagger)
\right) U_1^{-\dagger}
$$

$$
\bar{H}_2 = \bar{U}_2
$$

Then:

$$
\bar{A} = P^T L^{-\dagger} [\bar{H}_1 \mid \bar{H}_2]
$$

**Derivation:** From $U_2 = L^{-1} P A_2$, the cotangent of $A_2$ through $U_2$ is
$\bar{A}_2 = P^T L^{-\dagger} \bar{U}_2$. The cotangent of $Q$ (here $L$) receives an additional
contribution $-\bar{U}_2 U_2^\dagger$ from $\dot{U}_2$'s dependence on $\operatorname{tril}_-(\dot{F})$.

### Tall case ($M > N$)

Partition $\bar{L} = \begin{pmatrix} \bar{L}_1 \\ \bar{L}_2 \end{pmatrix}$.

Define:

$$
\bar{H}_1 = L_1^{-\dagger} \left(
  \operatorname{tril}_-(L_1^\dagger \bar{L}_1)
  + \operatorname{triu}(\bar{U} U^\dagger - L_2^\dagger \bar{L}_2)
\right)
$$

$$
\bar{H}_2 = \bar{L}_2
$$

Then:

$$
\bar{A} = P^T \begin{pmatrix} \bar{H}_1 \\ \bar{H}_2 \end{pmatrix} U^{-\dagger}
$$

**Derivation:** From $L_2 = P_2 A U^{-1}$, the cotangent of $A$ through $L_2$ is
$\bar{A} \mathrel{+}= P_2^T \bar{L}_2 U^{-\dagger}$. The $\bar{U}$ term receives an additional
contribution $-L_2^\dagger \bar{L}_2$ from $\dot{L}_2$'s dependence on $\operatorname{triu}(\dot{F})$.

## Implementation notes

All $L^{-1} X$ and $X U^{-1}$ operations should be implemented as triangular solves
(forward/back substitution), never as explicit inversions.

## Verification

### Reconstruction check (forward)

$$
\|PA - LU\|_F < \varepsilon
$$

$L$ is unit lower triangular, $U$ is upper triangular.

### Gradient check (backward)

Scalar test functions (see [`docs/design/testing.md`](../design/testing.md)):

- **dL only:** $f(A) = \operatorname{Re}(v^\dagger \operatorname{op} \, v)$, $v = L_{:,1}$
- **dU only:** $f(A) = \operatorname{Re}(v^\dagger \operatorname{op} \, v)$, $v = U_{1,:}$
- **joint dL+dU:** $f(A) = \operatorname{Re}(L_{1,1}^* \, U_{1,1})$

where $\operatorname{op}$ is a random Hermitian matrix independent of $A$.

## References

1. S. Axen, "Differentiating the LU decomposition," 2021.
   <https://sethaxen.com/blog/2021/02/differentiating-the-lu-decomposition/>
2. M. Seeger *et al.*, "Auto-Differentiating Linear Algebra," 2018.
