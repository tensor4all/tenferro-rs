# Norm AD Rules (`norm`)

## 1. Vector $p$-norm

$$
\|x\|_p = \Bigl(\sum_i |x_i|^p\Bigr)^{1/p}, \quad x \in \mathbb{C}^N
$$

### Forward rule (JVP)

$$
\dot{n} = \frac{\sum_i |x_i|^{p-2}\,\mathrm{Re}(\bar{x}_i\,\dot{x}_i)}{\|x\|_p^{p-1}}
$$

### Reverse rule (VJP)

$$
\bar{x}_i = \bar{n} \cdot \frac{x_i\,|x_i|^{p-2}}{\|x\|_p^{p-1}}
$$

### Special cases

| $p$ | $\bar{x}$ (VJP) | Notes |
|-----|-----------------|-------|
| $0$ | $0$ | $\ell^0$ "norm" is piecewise constant |
| $1$ | $\bar{n}\,\mathrm{sgn}(x)$ | Subgradient at $x_i = 0$ |
| $2$ | $\bar{n}\,x / \|x\|_2$ | Masked at $\|x\| = 0$ |
| $\infty$ | $\bar{n}\,\mathrm{sgn}(x) \cdot \mathbb{1}_{|x| = \|x\|_\infty} / k$ | $k$ = multiplicity of max |

---

## 2. Frobenius norm

$$
\|A\|_F = \sqrt{\mathrm{tr}(A^{\mathsf{H}}A)}
$$

Equivalent to the vector 2-norm of the flattened matrix.

### Forward rule (JVP)

$$
\dot{n} = \frac{\mathrm{Re}\!\mathrm{tr}(A^{\mathsf{H}}\dot{A})}{\|A\|_F}
$$

### Reverse rule (VJP)

$$
\bar{A} = \bar{n} \cdot \frac{A}{\|A\|_F}
$$

---

## 3. Nuclear norm (trace norm)

$$
\|A\|_* = \sum_i \sigma_i(A) = \mathrm{tr}(S)
$$

where $A = U S V^{\mathsf{H}}$ is the SVD.

### Forward rule (JVP)

$$
\dot{n} = \mathrm{Re}\!\mathrm{tr}(U^{\mathsf{H}}\,\dot{A}\,V)
$$

### Reverse rule (VJP)

$$
\bar{A} = \bar{n} \cdot U V^{\mathsf{H}}
$$

**Derivation.** Since $\|A\|_* = \sum_i \sigma_i$ and
$\dot{\sigma}_i = \mathrm{Re}(u_i^{\mathsf{H}}\,\dot{A}\,v_i)$,
summing gives $\dot{n} = \mathrm{Re}\!\mathrm{tr}(U^{\mathsf{H}}\dot{A}\,V)$.
The adjoint is $\bar{A} = \bar{n}\,U V^{\mathsf{H}}$.

**Non-smooth case** ($A$ rank-deficient): the subdifferential is
$\{UV^{\mathsf{H}} + W : P_U^{\perp} W P_V^{\perp} = W,\, \|W\|_2 \leq 1\}$
(Watson 1992).

---

## 4. Spectral norm (operator 2-norm)

$$
\|A\|_2 = \sigma_{\max}(A)
$$

### Forward rule (JVP)

For simple $\sigma_{\max}$ (multiplicity 1):

$$
\dot{n} = \mathrm{Re}(u_1^{\mathsf{H}}\,\dot{A}\,v_1)
$$

where $u_1, v_1$ are the leading singular vectors.

### Reverse rule (VJP)

$$
\bar{A} = \bar{n} \cdot u_1\,v_1^{\mathsf{H}}
$$

For multiplicity $k$:

$$
\bar{A} = \bar{n} \cdot \frac{1}{k}\sum_{i:\,\sigma_i = \sigma_{\max}} u_i\,v_i^{\mathsf{H}}
$$

**Non-differentiable** when $\sigma_{\max}$ has multiplicity $> 1$.

---

## Implementation notes

- **Frobenius**: PyTorch decomposes to `linalg_vector_norm(A, 2, dims)`.
- **Nuclear**: PyTorch decomposes to `svdvals(A).sum()` — no dedicated backward.
- **Spectral**: PyTorch decomposes to `amax(svdvals(A))` — no dedicated backward.
- Nuclear and spectral norms inherit AD rules from SVD backward.

## References

1. Giles, M. B. (2008). ["An extended collection of matrix derivative
   results for forward and reverse mode AD."](https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf)
2. Watson, G. A. (1992). "Characterization of the subdifferential of
   some matrix norms." *Linear Algebra Appl.*, 170, 33-45.
3. Petersen, K. B. and Pedersen, M. S. (2012). *The Matrix Cookbook.*
   Section 10.6.
4. PyTorch `FunctionsManual.cpp`: `norm_backward` (L250),
   `norm_jvp` (L304), `linalg_vector_norm_backward` (L459).
5. PyTorch `LinearAlgebra.cpp`: `linalg_matrix_norm` decomposition
   (Frobenius→vector_norm, nuclear→svdvals.sum, spectral→amax(svdvals)).
