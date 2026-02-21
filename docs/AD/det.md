# Determinant and Sign-Log-Determinant Reverse-Mode Rules

## 1. Determinant (`det`)

### Forward

$$
d = \det(A), \quad A \in \mathbb{C}^{N \times N}
$$

### Forward mode (JVP)

By Jacobi's formula (1841):

$$
\dot{d} = \det(A) \cdot \mathrm{tr}(A^{-1} \dot{A})
$$

### Reverse mode (VJP)

Given cotangent $\bar{d} \in \mathbb{C}$:

$$
\bar{A} = \bar{d} \cdot \det(A) \cdot A^{-\mathsf{T}}
$$

Equivalently, $\bar{A} = \bar{d} \cdot \mathrm{adj}(A)^{\mathsf{T}}$, where
$\mathrm{adj}(A) = \det(A) \cdot A^{-1}$ is the classical adjugate.

**Derivation.** From the JVP:

$$
\delta\ell = \langle \bar{d},\, \dot{d} \rangle
= \bar{d} \cdot \det(A) \cdot \mathrm{tr}(A^{-1} \dot{A})
= \mathrm{tr}\!\bigl((\bar{d} \cdot \det(A) \cdot A^{-\mathsf{T}})^{\mathsf{T}} \dot{A}\bigr)
$$

Reading off: $\bar{A} = \bar{d} \cdot \det(A) \cdot A^{-\mathsf{T}}$.

### Singular matrix handling

When $A$ is singular, $A^{-1}$ does not exist, but
$\mathrm{adj}(A)^{\mathsf{T}}$ is well-defined:

- $\mathrm{rank}(A) = N-1$: $\mathrm{adj}(A)$ is rank 1,
  computable via SVD: $A = U \Sigma V^{\mathsf{H}}$ gives
  $\mathrm{adj}(A) = V \mathrm{diag}(d) U^{\mathsf{H}}$
  where $d_k = \prod_{i \neq k} \sigma_i$.
- $\mathrm{rank}(A) \leq N-2$: $\mathrm{adj}(A) = 0$.

PyTorch uses `prod_safe_zeros_backward` (leave-one-out product via
exclusive cumulative product) for the SVD-based adjugate.

---

## 2. Sign-Log-Determinant (`slogdet`)

### Forward

$$
(\mathrm{sign}, \mathrm{logabsdet}) = \mathrm{slogdet}(A)
$$

where $\det(A) = \mathrm{sign} \cdot \exp(\mathrm{logabsdet})$.

- Real: $\mathrm{sign} \in \{-1, 0, +1\}$,
  $\mathrm{logabsdet} = \log|\det(A)|$.
- Complex: $\mathrm{sign} = \det(A)/|\det(A)|$ (unit complex),
  $\mathrm{logabsdet} = \log|\det(A)|$.

### Forward mode (JVP)

Let $w = \mathrm{tr}(A^{-1} \dot{A})$.

**Real case:**

$$
\dot{\mathrm{logabsdet}} = w, \qquad
\dot{\mathrm{sign}} = 0
$$

($\mathrm{sign}$ is piecewise constant.)

**Complex case:**

$$
\dot{\mathrm{logabsdet}} = \mathrm{Re}(w), \qquad
\dot{\mathrm{sign}} = i \cdot \mathrm{Im}(w) \cdot \mathrm{sign}
$$

**Derivation.** From $\log\det(A) = \mathrm{logabsdet} + i\arg(\det(A))$
and Jacobi's formula $d(\log\det(A)) = \mathrm{tr}(A^{-1} dA)$,
the real part gives the log-magnitude derivative and the imaginary part
gives the argument (phase) derivative. Since
$\mathrm{sign} = e^{i\arg(\det(A))}$, we get
$d(\mathrm{sign}) = i \cdot d(\arg) \cdot \mathrm{sign}$.

### Reverse mode (VJP)

Given cotangents $(\overline{\mathrm{sign}},\, \overline{\mathrm{logabsdet}})$:

**Real case** ($\overline{\mathrm{sign}}$ has no contribution):

$$
\bar{A} = \overline{\mathrm{logabsdet}} \cdot A^{-\mathsf{T}}
$$

**Complex case:**

$$
\bar{A} = g \cdot A^{-\mathsf{H}}
$$

where

$$
g = \overline{\mathrm{logabsdet}}
  - i \cdot \mathrm{Im}(\overline{\mathrm{sign}}^* \cdot \mathrm{sign})
$$

**Derivation.** Taking the adjoint of the JVP for each output:

- logabsdet cotangent: $\langle \bar{g}_{\mathrm{abs}},\, \mathrm{Re}(w) \rangle
  = \mathrm{Re}(\bar{g}_{\mathrm{abs}} \cdot w)
  = \langle \bar{g}_{\mathrm{abs}} \cdot A^{-\mathsf{H}},\, \dot{A} \rangle$.

- sign cotangent: Using $\mathrm{Re}(z \cdot \mathrm{Im}(w))
  = \mathrm{Re}(-\mathrm{Re}(z) \cdot i \cdot w)$, we get
  contribution $-i \cdot \mathrm{Im}(\bar{g}_{\mathrm{sign}}^* \cdot \mathrm{sign}) \cdot A^{-\mathsf{H}}$.

Combining yields the formula above.

### Note on singularity

`slogdet` is **not differentiable** at singular matrices
($\mathrm{logabsdet} = -\infty$), unlike `det`.

## Implementation notes

- Compute $A^{-1}$ via LU factorization (never form $A^{-1}$ explicitly);
  solve $A X = I$ or $A^{\mathsf{H}} X = g \cdot I$.
- Cost: $O(N^3)$, same as forward evaluation.

## References

1. Jacobi, C. G. J. (1841). "De formatione et proprietatibus
   determinantium." *J. Reine Angew. Math.*, 22, 285-318.
2. Giles, M. B. (2008). ["An extended collection of matrix derivative
   results for forward and reverse mode algorithmic differentiation."](https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf)
3. PyTorch `FunctionsManual.cpp`: `linalg_det_backward` (L4308),
   `linalg_det_jvp` (L4290), `slogdet_backward` (L4396),
   `slogdet_jvp` (L4376).
4. JAX `jax/_src/numpy/linalg.py`: `_slogdet_jvp`, `_det_jvp`,
   `_cofactor_solve`.
5. ChainRules.jl `src/rulesets/LinearAlgebra/dense.jl`:
   `frule`/`rrule` for `det`, `logdet`, `logabsdet`.
