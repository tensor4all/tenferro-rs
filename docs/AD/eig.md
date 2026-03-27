# General Eigendecomposition AD Rules (`eig`)

## Forward

$$
A V = V \mathrm{diag}(\lambda), \quad A \in \mathbb{C}^{N \times N}
$$

where $V$ is the matrix of right eigenvectors (columns) and $\lambda$ the
eigenvalue vector. Eigenvalues may be complex even for real $A$.

**Assumption:** All eigenvalues are distinct (simple). The derivative is
undefined for repeated eigenvalues.

## Auxiliary matrix

$$
E_{ij} = \begin{cases}
\lambda_j - \lambda_i & i \neq j \\
1 & i = j
\end{cases}
$$

## Forward rule (JVP)

Differentiating $AV = V\mathrm{diag}(\lambda)$ and left-multiplying by $V^{-1}$:

$$
V^{-1}\dot{A}\,V = V^{-1}\dot{V}\,\mathrm{diag}(\lambda)
- \mathrm{diag}(\lambda)\,V^{-1}\dot{V}
+ \mathrm{diag}(\dot{\lambda})
$$

Define $\Delta P = V^{-1}\dot{A}\,V$.

**Eigenvalue tangents:**

$$
\dot{\lambda}_i = (\Delta P)_{ii} = (V^{-1}\dot{A}\,V)_{ii}
$$

**Eigenvector tangents (raw):**

$$
Q_{ij} = \frac{(\Delta P)_{ij}}{E_{ij}} \quad (i \neq j), \qquad Q_{ii} = 0
$$

$$
\dot{V}_{\mathrm{raw}} = V\,Q
$$

**Normalization correction** (for $\|v_i\| = 1$ convention):

$$
\dot{V} = \dot{V}_{\mathrm{raw}} - V\,\mathrm{diag}\!\bigl(\mathrm{Re}(V^{\mathsf{H}}\dot{V}_{\mathrm{raw}})\bigr)
$$

This projects out the component that would change the eigenvector norms.

## Reverse rule (VJP)

Given cotangents $\bar{\lambda}$ (eigenvalue) and $\bar{V}$ (eigenvector):

**Step 1: Normalization adjoint**

$$
\bar{V}_{\mathrm{adj}} = \bar{V} - V\,\mathrm{diag}\!\bigl(\mathrm{Re}(V^{\mathsf{H}}\bar{V})\bigr)
$$

**Step 2: Inner gradient matrix**

$$
G = V^{\mathsf{H}}\bar{V}_{\mathrm{adj}}
$$

$$
G_{ij} \leftarrow \frac{G_{ij}}{\overline{\lambda_j - \lambda_i}} \quad (i \neq j), \qquad
G_{ii} = \bar{\lambda}_i
$$

**Step 3: Conjugation**

$$
\bar{A} = V^{-\mathsf{H}}\,G\,V^{\mathsf{H}}
$$

For real $A$: $\bar{A} = \mathrm{Re}(\bar{A})$.

### Gauge invariance check

Eigenvectors are defined up to a complex phase $e^{i\phi}$. The loss function
must be gauge-invariant, verified by $\mathrm{Im}(\mathrm{diag}(V^{\mathsf{H}}\bar{V})) \approx 0$.

## Relationship to symmetric case

For symmetric/Hermitian $A$: $V$ is unitary ($V^{-1} = V^{\mathsf{H}}$),
eigenvalues are real, and $V^{-\mathsf{H}} = V$, simplifying all formulas.
See `eigen.md` for the symmetric case.

## Dyadtensor integration

`tenferro` keeps reverse-mode graphs homogeneous: the graph carries
one runtime-typed tensor payload, and scalar AD values are rank-0 tensors
rather than a separate scalar graph type.

That means the public `tenferro::Tensor::eig()` wrapper is currently split as
follows:

- real inputs in primal mode: supported
- real inputs in forward mode: supported
- reverse mode: currently unsupported in the `tenferro` frontend
- complex inputs: currently rejected at the public frontend

The frontend keeps reverse-mode tapes homogeneous, but `eig()` turns real
inputs into complex outputs. The old mixed-type reverse bridge was removed, so
the public wrapper cannot currently register a reverse pullback. Dense
`eig_rrule` / `eig_frule` still exist in `tenferro-linalg` for real typed
inputs; the restriction is in the frontend wrapper layer.

## References

1. Giles, M. B. (2008). ["An extended collection of matrix derivative
   results for forward and reverse mode AD."](https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf)
   Section 3.1.
2. Boeddeker, C. et al. (2019). ["On the Computation of Complex-valued
   Gradients with Application to Statistically Optimum
   Beamforming."](https://arxiv.org/abs/1701.00392) Eqs. 4.60, 4.63, 4.77.
3. Magnus, J. R. (1985). "On Differentiating Eigenvalues and
   Eigenvectors." *Econometric Theory*, 1, 179-191.
4. Greenbaum, A., Li, R.-C., Overton, M. L. (2019).
   ["First-order Perturbation Theory for Eigenvalues and
   Eigenvectors."](https://arxiv.org/abs/1903.00785)
5. PyTorch `FunctionsManual.cpp`: `linalg_eig_backward` (L3755),
   `linalg_eig_jvp` (L3858).
6. ChainRules.jl `src/rulesets/LinearAlgebra/factorization.jl`:
   `frule`/`rrule` for `eigen`.
7. JAX `jax/_src/lax/linalg.py`: `eig_jvp_rule` (eigenvalue tangents
   only; eigenvector derivatives raise `NotImplementedError`).
