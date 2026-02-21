# Determinant and Sign-Log-Determinant Reverse-Mode Rules

## 1. Determinant (`det`)

### Forward

$$
d = \det(A), \quad A \in \mathbb{C}^{N \times N}
$$

### Forward mode (JVP)

By Jacobi's formula (1841):

$$
\dot{d} = \det(A) \cdot \operatorname{tr}(A^{-1} \dot{A})
$$

### Reverse mode (VJP)

Given cotangent $\bar{d} \in \mathbb{C}$:

$$
\bar{A} = \bar{d} \cdot \det(A) \cdot A^{-\mathsf{T}}
$$

Equivalently, $\bar{A} = \bar{d} \cdot \operatorname{adj}(A)^{\mathsf{T}}$, where
$\operatorname{adj}(A) = \det(A) \cdot A^{-1}$ is the classical adjugate.

**Derivation.** From the JVP:

$$
\delta\ell = \langle \bar{d},\, \dot{d} \rangle
= \bar{d} \cdot \det(A) \cdot \operatorname{tr}(A^{-1} \dot{A})
= \operatorname{tr}\!\bigl((\bar{d} \cdot \det(A) \cdot A^{-\mathsf{T}})^{\mathsf{T}} \dot{A}\bigr)
$$

Reading off: $\bar{A} = \bar{d} \cdot \det(A) \cdot A^{-\mathsf{T}}$.

### Singular matrix handling

When $A$ is singular, $A^{-1}$ does not exist, but
$\operatorname{adj}(A)^{\mathsf{T}}$ is well-defined:

- $\operatorname{rank}(A) = N-1$: $\operatorname{adj}(A)$ is rank 1,
  computable via SVD: $A = U \Sigma V^{\mathsf{H}}$ gives
  $\operatorname{adj}(A) = V \operatorname{diag}(d) U^{\mathsf{H}}$
  where $d_k = \prod_{i \neq k} \sigma_i$.
- $\operatorname{rank}(A) \leq N-2$: $\operatorname{adj}(A) = 0$.

PyTorch uses `prod_safe_zeros_backward` (leave-one-out product via
exclusive cumulative product) for the SVD-based adjugate.

---

## 2. Sign-Log-Determinant (`slogdet`)

### Forward

$$
(\operatorname{sign}, \operatorname{logabsdet}) = \operatorname{slogdet}(A)
$$

where $\det(A) = \operatorname{sign} \cdot \exp(\operatorname{logabsdet})$.

- Real: $\operatorname{sign} \in \{-1, 0, +1\}$,
  $\operatorname{logabsdet} = \log|\det(A)|$.
- Complex: $\operatorname{sign} = \det(A)/|\det(A)|$ (unit complex),
  $\operatorname{logabsdet} = \log|\det(A)|$.

### Forward mode (JVP)

Let $w = \operatorname{tr}(A^{-1} \dot{A})$.

**Real case:**

$$
\dot{\operatorname{logabsdet}} = w, \qquad
\dot{\operatorname{sign}} = 0
$$

($\operatorname{sign}$ is piecewise constant.)

**Complex case:**

$$
\dot{\operatorname{logabsdet}} = \operatorname{Re}(w), \qquad
\dot{\operatorname{sign}} = i \cdot \operatorname{Im}(w) \cdot \operatorname{sign}
$$

**Derivation.** From $\log\det(A) = \operatorname{logabsdet} + i\arg(\det(A))$
and Jacobi's formula $d(\log\det(A)) = \operatorname{tr}(A^{-1} dA)$,
the real part gives the log-magnitude derivative and the imaginary part
gives the argument (phase) derivative. Since
$\operatorname{sign} = e^{i\arg(\det(A))}$, we get
$d(\operatorname{sign}) = i \cdot d(\arg) \cdot \operatorname{sign}$.

### Reverse mode (VJP)

Given cotangents $(\overline{\operatorname{sign}},\, \overline{\operatorname{logabsdet}})$:

**Real case** ($\overline{\operatorname{sign}}$ has no contribution):

$$
\bar{A} = \overline{\operatorname{logabsdet}} \cdot A^{-\mathsf{T}}
$$

**Complex case:**

$$
\bar{A} = g \cdot A^{-\mathsf{H}}
$$

where

$$
g = \overline{\operatorname{logabsdet}}
  - i \cdot \operatorname{Im}(\overline{\operatorname{sign}}^* \cdot \operatorname{sign})
$$

**Derivation.** Taking the adjoint of the JVP for each output:

- logabsdet cotangent: $\langle \bar{g}_{\mathrm{abs}},\, \operatorname{Re}(w) \rangle
  = \operatorname{Re}(\bar{g}_{\mathrm{abs}} \cdot w)
  = \langle \bar{g}_{\mathrm{abs}} \cdot A^{-\mathsf{H}},\, \dot{A} \rangle$.

- sign cotangent: Using $\operatorname{Re}(z \cdot \operatorname{Im}(w))
  = \operatorname{Re}(-\operatorname{Re}(z) \cdot i \cdot w)$, we get
  contribution $-i \cdot \operatorname{Im}(\bar{g}_{\mathrm{sign}}^* \cdot \operatorname{sign}) \cdot A^{-\mathsf{H}}$.

Combining yields the formula above.

### Note on singularity

`slogdet` is **not differentiable** at singular matrices
($\operatorname{logabsdet} = -\infty$), unlike `det`.

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
