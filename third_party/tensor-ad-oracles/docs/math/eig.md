# General Eigen AD Notes

## Forward Definition

$$
A V = V \operatorname{diag}(\lambda),
\qquad
A \in \mathbb{C}^{N \times N}
$$

where $V$ contains right eigenvectors and $\lambda$ contains the eigenvalues.
We assume the eigenvalues are simple. The rule is undefined at repeated
eigenvalues.

## Auxiliary Matrix

$$
E_{ij} =
\begin{cases}
\lambda_j - \lambda_i, & i \neq j, \\
1, & i = j.
\end{cases}
$$

## Forward Rule

Differentiate the eigen equation and left-multiply by $V^{-1}$:

$$
V^{-1}\dot{A}\,V =
V^{-1}\dot{V}\,\operatorname{diag}(\lambda)
- \operatorname{diag}(\lambda) V^{-1}\dot{V}
+ \operatorname{diag}(\dot{\lambda}).
$$

Define

$$
\Delta P = V^{-1}\dot{A}\,V.
$$

Then

$$
\dot{\lambda}_i = (\Delta P)_{ii}.
$$

For the eigenvector tangents,

$$
Q_{ij} = \frac{(\Delta P)_{ij}}{E_{ij}} \quad (i \neq j),
\qquad
Q_{ii} = 0,
$$

$$
\dot{V}_{\mathrm{raw}} = V Q.
$$

### Normalization correction

PyTorch and `tenferro-rs` both normalize eigenvectors to unit norm. Therefore
the raw tangent must be projected back onto that gauge:

$$
\dot{V} =
\dot{V}_{\mathrm{raw}}
- V \, \operatorname{diag}\!\left(\operatorname{Re}(V^\dagger \dot{V}_{\mathrm{raw}})\right).
$$

## Reverse Rule

Given cotangents $\bar{\lambda}$ and $\bar{V}$:

### Step 1: Normalization adjoint

$$
\bar{V}_{\mathrm{adj}} =
\bar{V}
- V \, \operatorname{diag}\!\left(\operatorname{Re}(V^\dagger \bar{V})\right).
$$

### Step 2: Inner matrix

$$
G = V^\dagger \bar{V}_{\mathrm{adj}}
$$

$$
G_{ij} \leftarrow \frac{G_{ij}}{\overline{\lambda_j - \lambda_i}}
\quad (i \neq j),
\qquad
G_{ii} = \bar{\lambda}_i.
$$

### Step 3: Conjugation back to the input basis

$$
\bar{A} = V^{-\dagger} G V^\dagger.
$$

For real $A$, the final cotangent is projected back to the real domain:

$$
\bar{A} \leftarrow \operatorname{Re}(\bar{A}).
$$

### Gauge invariance check

Eigenvectors are only defined up to per-column complex phase. The loss must be
invariant under $V \mapsto V \operatorname{diag}(e^{i\phi_k})$, which implies

$$
\operatorname{Im}(\operatorname{diag}(V^\dagger \bar{V})) = 0.
$$

PyTorch's `linalg_eig_backward` checks this condition numerically and raises for
ill-defined losses.

## Relationship to the Hermitian Case

When $A$ is Hermitian, $V$ is unitary, $V^{-1} = V^\dagger$, and eigenvalues are
real. The formulas simplify to the structured rule documented in
[`eigen.md`](./eigen.md).

## Implementation Correspondence

- `tenferro-rs/docs/AD/eig.md` uses the $V^{-1}\dot{A}V$ and
  $V^{-\dagger} G V^\dagger$ formulation with an explicit normalization
  correction.
- PyTorch's `linalg_eig_jvp` and `linalg_eig_backward` implement the same rule.
  Their comments explicitly note that the uncorrected textbook formulas are
  missing the normalization term.
- For real inputs with complex outputs, PyTorch applies the usual
  `handle_r_to_c` projection back to the real cotangent domain.

## Verification

### Forward reconstruction

$$
A V \approx V \operatorname{diag}(\lambda).
$$

### Backward checks

- eigenvalues only: scalar losses of $\lambda$
- eigenvectors only: scalar losses of a normalized column of $V$
- mixed losses: compare JVP/VJP against finite differences away from repeated
  eigenvalues

## References

1. M. B. Giles, "An extended collection of matrix derivative results for
   forward and reverse mode AD," 2008.
2. C. Boeddeker et al., "On the Computation of Complex-valued Gradients," 2019.

## DB Families

<a id="family-values-vectors-abs"></a>
### `values_vectors_abs`

The DB publishes the eigenvalues together with `vectors.abs()` to remove the
column-wise phase ambiguity of raw eigenvectors.

<a id="family-eigvals-identity"></a>
### `eigvals/identity`

The eigenvalue-only family reuses the diagonal part of the same differential.
