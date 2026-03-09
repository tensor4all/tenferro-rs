# Linear Algebra

Batched matrix decompositions and solvers with stateless AD rules.

**Implementation status**: CPU backend fully implemented via
[faer](https://crates.io/crates/faer). GPU backends (cuSOLVER/hipTensor) are
planned but not yet implemented.

---

## Position in Workspace Architecture

`tenferro-linalg` sits at Layer 4, alongside `tenferro-einsum`:

```
Layer 4: tenferro-einsum   — N-ary einsum + einsum AD
         tenferro-linalg   — SVD/QR/LU/eigen + linalg AD ← this crate
Layer 3: tenferro-prims    — TensorPrims<A> trait
Layer 2: tenferro-tensor   — Tensor<T>, view ops
```

Dependencies: `tenferro-device`, `tenferro-algebra`, `tenferro-prims`,
`tenferro-tensor`, `chainrules-core`.

Note: depends on `chainrules-core` (traits only), **not** full `chainrules`
(engine). This crate provides stateless AD rules — it never creates tapes
or tracked tensors.

---

## Dimension Convention

**First 2 dimensions = matrix, remaining = batch.**

```
Input shape: (m, n, *)
             ├──┘  └──── batch dimensions (independent problems)
             matrix
```

This is the column-major counterpart to PyTorch's `(*, m, n)`: in col-major
layout the first dimensions are contiguous, so placing the matrix there
lets LAPACK/cuSOLVER operate without transposition.

Linalg APIs internally normalize inputs to column-major contiguous layout.
If an input is not already contiguous, an internal copy is performed.

For arbitrary tensor decomposition (e.g., SVD along legs [0,1] vs [2,3]),
the caller still performs `permute` + `reshape` before calling linalg.
Calling `contiguous(ColumnMajor)` explicitly is optional but useful
when the caller wants to control exactly where memory copies occur.

---

## Operations

### Decompositions

| Function | Input shape | Result type | Description |
|----------|-------------|-------------|-------------|
| `svd` | `(m, n, *)` | `SvdResult { u, s, vt }` | $A = U \operatorname{diag}(S) V^\top$, optional truncation via `SvdOptions` |
| `qr` | `(m, n, *)` | `QrResult { q, r }` | $A = QR$, thin QR |
| `lu` | `(m, n, *)` | `LuResult { p, l, u }` | $A = PLU$, with `LuPivot` strategy |
| `cholesky` | `(n, n, *)` | `Tensor<T>` | $A = LL^\dagger$, returns lower triangular L |
| `eigen` | `(n, n, *)` | `EigenResult { values, vectors }` | Symmetric/Hermitian eigendecomposition (validated) |
| `eig` | `(n, n, *)` | `EigResult { values, vectors }` | General (non-symmetric) eigendecomposition; output is always `Complex<T>` |

### Solvers

| Function | Inputs | Output | Description |
|----------|--------|--------|-------------|
| `solve` | `A: (n,n,*)`, `b: (n,*)` or `(n,k,*)` | `Tensor<T>` | General square system $Ax = b$ |
| `solve_triangular` | `A: (n,n,*)`, `b: (n,*)` or `(n,k,*)`, `upper: bool` | `Tensor<T>` | Triangular system |
| `lstsq` | `A: (m,n,*)`, `b: (m,*)` | `LstsqResult { x, residual }` | Least squares `argmin \|\|Ax-b\|\|²` (m >= n), vector RHS in current implementation |

### Utilities

| Function | Input shape | Output | Description |
|----------|-------------|--------|-------------|
| `inv` | `(n, n, *)` | `Tensor<T>` | Matrix inverse |
| `det` | `(n, n, *)` | `Tensor<T>` shape `(*)` | Determinant |
| `slogdet` | `(n, n, *)` | `SlogdetResult { sign, logabsdet }` | Numerically stable log-determinant |
| `pinv` | `(m, n, *)` | `Tensor<T>` | Moore-Penrose pseudoinverse (SVD-based) |
| `matrix_exp` | `(n, n, *)` | `Tensor<T>` | Matrix exponential via Pade[13/13] scaling-and-squaring |
| `norm` | `(m, n, *)` | `Tensor<T>` shape `(*)` | Matrix norm (`Fro`, `Nuclear`, `Spectral`) |

### Notes

- `lu(..., LuPivot::NoPivot)` is implemented for the supported CPU paths and
  returns an LU factorization with `p: None`.
- `eig()` always returns `EigResult` with `Complex<T>` eigenvalues and eigenvectors,
  even for real input. This avoids branching on whether eigenvalues happen to be real.
- `norm(...)` implements `Fro`, `Nuclear`, and `Spectral` only; other variants
  return `Error::InvalidArgument`.
- `solve_triangular` has public reverse- and forward-mode AD rules via
  `solve_triangular_rrule` and `solve_triangular_frule`.

---

## Result Types

Structured result types avoid positional return confusion:

```rust
pub struct SvdResult<T: Scalar> {
    pub u: Tensor<T>,   // (m, k, *)
    pub s: Tensor<T>,   // (k, *)
    pub vt: Tensor<T>,  // (k, n, *)
}

pub struct QrResult<T: Scalar> {
    pub q: Tensor<T>,   // (m, k, *)
    pub r: Tensor<T>,   // (k, n, *)
}

pub struct LuResult<T: Scalar> {
    pub p: Option<Vec<usize>>,  // None when NoPivot
    pub l: Tensor<T>,           // (m, k, *)
    pub u: Tensor<T>,           // (k, n, *)
}

pub struct EigenResult<T: Scalar> {
    pub values: Tensor<T>,   // (n, *)
    pub vectors: Tensor<T>,  // (n, n, *)
}

/// General (non-symmetric) eigendecomposition.
/// Output is always Complex<T>, even for real input T.
pub struct EigResult<R: LinalgScalar + Float> {
    pub values: Tensor<Complex<R>>,   // (n, *)
    pub vectors: Tensor<Complex<R>>,  // (n, n, *)
}

pub struct SlogdetResult<T: Scalar> {
    pub sign: Tensor<T>,       // (*)
    pub logabsdet: Tensor<T>,  // (*)
}

pub struct LstsqResult<T: Scalar> {
    pub x: Tensor<T>,        // (n, *)
    pub residual: Tensor<T>, // (m, *)
}
```

Where `k = min(m, n)`.

---

## Options and Configuration

### SVD Truncation

```rust
pub struct SvdOptions {
    pub max_rank: Option<usize>,  // cap number of singular values
    pub cutoff: Option<f64>,      // discard below threshold
}
```

When both are set, the more restrictive constraint applies.

### LU Pivoting

```rust
pub enum LuPivot {
    Partial,   // row pivoting (default, stable)
    NoPivot,   // no pivoting (faster, unstable)
}
```

### Norm Kind

```rust
pub enum NormKind {
    Fro,         // Frobenius / L2
    Nuclear,     // sum of singular values
    Spectral,    // largest singular value
    L1,          // max abs column sum / sum abs
    Inf,         // max abs row sum / max abs
    Lp(f64),     // general Lp (vectors only)
}
```

---

## AD Rules

All AD rules are **stateless free functions** — no `tracked_*` / `dual_*`
wrappers. The chainrules tape engine composes `permute_backward` +
`reshape_backward` + `svd_rrule` etc. via the standard chain rule
automatically.

`tenferro-linalg` depends on **`chainrules-core` only** (not the full
`chainrules` engine). It uses `AdResult` and the `Differentiable` trait from
`chainrules-core`; it never creates tapes or `TrackedTensor` values. See
[autodiff.md](./autodiff.md) for the AD crate split and
the algebra
interaction model. For how the algebra type `A` (e.g., `Standard<T>`) affects
which backend primitives are dispatched during the AD formulas, see
[algebra.md](./algebra.md).

For the step-by-step mathematical derivations of each rule, see the
[AD Formula Notes](../AD/index.md).

**Status**: All 14 rrule and 14 frule functions are implemented. AD formulas
are sourced from PyTorch's autograd formulas and Mathieu (2019). Each rule
is verified by finite-difference (FD) checks (see the Testing section below).

`solve_triangular` is the only forward function without AD rules; it is used
as a utility within other AD formulas.

### Cotangent Types

Structured cotangent types with `Option` fields allow partial gradient
computation (e.g., gradient only through singular values):

```rust
pub struct SvdCotangent<T: Scalar> {
    pub u: Option<Tensor<T>>,
    pub s: Option<Tensor<T>>,
    pub vt: Option<Tensor<T>>,
}

pub struct QrCotangent<T: Scalar> {
    pub q: Option<Tensor<T>>,
    pub r: Option<Tensor<T>>,
}

pub struct LuCotangent<T: Scalar> {
    pub l: Option<Tensor<T>>,
    pub u: Option<Tensor<T>>,
}

pub struct EigenCotangent<T: Scalar> {
    pub values: Option<Tensor<T>>,
    pub vectors: Option<Tensor<T>>,
}

pub struct SlogdetCotangent<T: Scalar> {
    pub logabsdet: Option<Tensor<T>>,
    // sign is piecewise constant, not differentiable
}
```

Gradient types for multi-input operations:

```rust
pub struct SolveGrad<T: Scalar> {
    pub a: Tensor<T>,
    pub b: Tensor<T>,
}

pub struct LstsqGrad<T: Scalar> {
    pub a: Tensor<T>,
    pub b: Tensor<T>,
}
```

### rrule Functions (Reverse-Mode / VJP)

| Function | Signature (abbreviated) |
|----------|------------------------|
| `svd_rrule` | `(tensor, cotangent: &SvdCotangent, options) -> AdResult<Tensor>` |
| `qr_rrule` | `(tensor, cotangent: &QrCotangent) -> AdResult<Tensor>` |
| `lu_rrule` | `(tensor, cotangent: &LuCotangent, pivot) -> AdResult<Tensor>` |
| `eigen_rrule` | `(tensor, cotangent: &EigenCotangent) -> AdResult<Tensor>` |
| `eig_rrule` | `(tensor, cotangent: &EigCotangent) -> AdResult<Tensor>` |
| `lstsq_rrule` | `(a, b, cotangent) -> AdResult<LstsqGrad>` |
| `cholesky_rrule` | `(tensor, cotangent) -> AdResult<Tensor>` |
| `solve_rrule` | `(a, b, cotangent) -> AdResult<SolveGrad>` |
| `inv_rrule` | `(tensor, cotangent) -> AdResult<Tensor>` |
| `det_rrule` | `(tensor, cotangent) -> AdResult<Tensor>` |
| `slogdet_rrule` | `(tensor, cotangent: &SlogdetCotangent) -> AdResult<Tensor>` |
| `pinv_rrule` | `(tensor, cotangent, rcond) -> AdResult<Tensor>` |
| `matrix_exp_rrule` | `(tensor, cotangent) -> AdResult<Tensor>` |
| `norm_rrule` | `(tensor, cotangent, kind) -> AdResult<Tensor>` |

### frule Functions (Forward-Mode / JVP)

| Function | Signature (abbreviated) |
|----------|------------------------|
| `svd_frule` | `(tensor, tangent, options) -> AdResult<(SvdResult, SvdResult)>` |
| `qr_frule` | `(tensor, tangent) -> AdResult<(QrResult, QrResult)>` |
| `lu_frule` | `(tensor, tangent, pivot) -> AdResult<(LuResult, LuResult)>` |
| `eigen_frule` | `(tensor, tangent) -> AdResult<(EigenResult, EigenResult)>` |
| `eig_frule` | `(tensor, tangent) -> AdResult<(EigResult, EigResult)>` |
| `lstsq_frule` | `(a, b, tangent_a, tangent_b) -> AdResult<(LstsqResult, LstsqResult)>` |
| `cholesky_frule` | `(tensor, tangent) -> AdResult<(Tensor, Tensor)>` |
| `solve_frule` | `(a, b, tangent_a, tangent_b) -> AdResult<(Tensor, Tensor)>` |
| `inv_frule` | `(tensor, tangent) -> AdResult<(Tensor, Tensor)>` |
| `det_frule` | `(tensor, tangent) -> AdResult<(Tensor, Tensor)>` |
| `slogdet_frule` | `(tensor, tangent) -> AdResult<(SlogdetResult, SlogdetResult)>` |
| `pinv_frule` | `(tensor, tangent, rcond) -> AdResult<(Tensor, Tensor)>` |
| `matrix_exp_frule` | `(tensor, tangent) -> AdResult<(Tensor, Tensor)>` |
| `norm_frule` | `(tensor, tangent, kind) -> AdResult<(Tensor, Tensor)>` |

frule return type convention: `(primal_result, tangent_result)` with the
same type for both.

### Key AD Formulas

**SVD rrule** (Mathieu 2019): see [autodiff.md](./autodiff.md) for the
8-step algorithm. All steps use `tenferro-prims` operations (BatchedGemm,
ElementwiseMul, ElementwiseUnary, Permute, AntiTrace) plus tensor-level
additions (eye, tril/triu).

**inv rrule**: $\bar{A} = -A^{-\mathsf{H}} \bar{C}\, A^{-\mathsf{H}}$

**det rrule**: $\bar{A} = \det(A)\, \bar{c}\, A^{-\top}$

**slogdet rrule**: $\bar{A} = \bar{s}\, A^{-\top}$

**solve rrule** ($Ax = b$): $\bar{b} = A^{-\top} \bar{x}$, $\bar{A} = -\bar{b}\, x^\top$

---

## Backend Mapping

```
tenferro-linalg functions
    │
    ├── CPU (implemented): matricize → faer → unmatricize
    │     faer provides: SVD, QR, LU, Cholesky, eigen (symmetric + general),
    │     solve, lstsq, triangular solve, and LU-based det/inv
    │
    └── GPU (planned): matricize → cuSOLVER → unmatricize
```

The `TensorLinalgBackend<T>` trait abstracts the backend.
`CpuTensorLinalgBackend` is the CPU implementation (backed by faer under
the `linalg-faer` feature). It maps each operation to faer's thin-matrix
API, handles column-major layout, and converts between `Tensor<T>` data
slices and faer's `MatRef`/`MatMut`. Execution state is held in
`tenferro_prims::CpuContext`, shared with the prims layer and passed
explicitly to all operations.

The linalg crate calls `tenferro-prims` operations for its AD formulas
(BatchedGemm, ElementwiseMul, etc.) but calls the backend directly for
forward decompositions (not through TensorPrims).

---

## Prims-First Boundary

The project adopts a **prims-first** implementation strategy where possible,
with explicit boundaries:

1. **Primal forward for heavy decompositions is backend-native.**
   `svd/qr/lu/eig/cholesky/solve/lstsq` primal implementations use dedicated
   CPU/GPU solver backends (faer/LAPACK/cuSOLVER family), not a decomposition
   synthesized only from `TensorPrims`.

2. **AD rules should be expressed in TensorPrims as far as practical.**
   Reverse/forward rules are written with `BatchedGemm`, `Permute`, `Reduce`,
   `Trace`, `AntiTrace`, `ElementwiseMul`, and `ElementwiseUnary` so the same
   rule logic can run across CPU/GPU backends.

3. **Why this split exists.**
   Reconstructing full linalg primal kernels from generic prims is possible in
   principle but is not practical for numerical stability, performance, and
   implementation complexity.

This keeps rule logic backend-agnostic while preserving production-grade
numerics for primal evaluation.

---

## Design Record Workflow

For linalg/prims integration changes, design records are managed as follows:

1. **Source of truth lives in docs.**
   Keep design details in existing files under `docs/design/` (this file and
   `tensor-prims.md`) and formula-level notes in `docs/AD/`.

2. **Issues track execution, not full specs.**
   GitHub issues should capture scope, acceptance criteria, and task breakdown,
   and must link back to the canonical design doc section.

3. **One topic, one design anchor.**
   For each major change, update one primary design location first, then create
   a parent issue that references it and spawns implementation sub-issues.

This reduces drift between discussion threads and technical decisions.

---

## Design Decisions

1. **Matrix-only API.** Higher-level tensor decomposition (e.g., SVD along
   arbitrary legs) belongs in application layers (TensorKit equivalent).
   Linalg takes `(m, n, *)` matrices.

2. **Stateless AD only.** No `tracked_*` wrappers — the chainrules tape
   engine composes linalg AD rules with permute/reshape backward passes.
   This avoids coupling linalg to the AD engine implementation.

3. **Cotangent types with Option fields.** Enables partial gradient
   computation (e.g., SVD gradient through `s` only is always numerically
   stable, avoiding the F-matrix singularity issues that arise when
   differentiating through `U`/`Vt` with degenerate singular values).

4. **Column-major internal normalization.** Linalg operations normalize inputs
   to column-major contiguous internally, potentially with copies. Callers may
   still make layout explicit up front to control allocation points.

5. **chainrules-core dependency, not chainrules.** Linalg only uses
   `AdResult` from chainrules-core. It never creates tapes or tracked
   tensors, so the full engine is unnecessary.

6. **tenferro-prims dependency.** AD formulas use TensorPrims operations
   (BatchedGemm, ElementwiseMul, ElementwiseUnary, Permute, AntiTrace).
   The `UnaryOp` enum (`Reciprocal`, `Sqrt`, etc.) was added to
   tenferro-prims specifically for linalg AD needs.

---

## Testing

### Forward operations

All forward operations are tested via **reconstruction and property checks**
rather than comparing decomposition outputs directly (LAPACK/faer do not
guarantee sign/phase conventions). See [testing.md](./testing.md) for the
per-operation reconstruction identities.

### AD: Finite-Difference Verification

All rrule and frule functions are verified against central finite differences:

```
FD tangent:  df/dx_i ≈ (f(x + eps*e_i) - f(x - eps*e_i)) / (2*eps)
```

Parameters: `eps = 1e-6`, `atol = 1e-4`.

Test utilities (`check_rrule_fd`, `check_frule_fd`) live alongside the test
file and compare the analytic AD output element-wise against the FD
approximation.

### Current FD status

All current `rrule` and `frule` implementations in `tenferro-linalg`
pass the finite-difference checks in `tenferro-linalg/tests/linalg_tests.rs`
with the documented tolerances (`eps = 1e-6`, `atol = 1e-4`).

The previously problematic `lu_rrule`, `lstsq_rrule`, and `qr_frule`
paths were aligned with the formulas used in PyTorch's manual autograd
implementation and are now covered by normal (non-ignored) tests.

### Coverage thresholds

Enforced via `scripts/check-coverage.py` with per-file thresholds:

| File | Threshold |
|------|-----------|
| `tenferro-linalg/src/lib.rs` | 95% |
| `tenferro-linalg/src/backend/faer_backend.rs` | 97% |
