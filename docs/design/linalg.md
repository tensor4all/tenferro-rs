# Linear Algebra

Batched matrix decompositions and solvers with stateless AD rules.

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
| `lu` | `(m, n, *)` | `LuResult { p, l, u }` | $A = PLU$, with `LuPivot` strategy (`NoPivot` currently returns error) |
| `cholesky` | `(n, n, *)` | `Tensor<T>` | $A = LL^\dagger$, returns lower triangular L |
| `eigen` | `(n, n, *)` | `EigenResult { values, vectors }` | Symmetric/Hermitian eigendecomposition (validated) |
| `eig` | `(n, n, *)` | `EigenResult { values, vectors }` | General (non-symmetric) eigendecomposition (currently returns error) |

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
| `matrix_exp` | `(n, n, *)` | `Tensor<T>` | Matrix exponential exp(A) (currently returns error) |
| `norm` | `(m, n, *)` | `Tensor<T>` shape `(*)` | Matrix norm (`Fro`, `Nuclear`, `Spectral`) |

### Current Availability Notes

- `lu(..., LuPivot::NoPivot)` currently returns `Error::InvalidArgument`.
- `eig(...)` currently returns `Error::InvalidArgument` (complex-valued path deferred).
- `matrix_exp(...)` currently returns `Error::InvalidArgument`.
- `norm(...)` currently implements `Fro`, `Nuclear`, and `Spectral` only.

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
[autodiff.md](./autodiff.md) for the overall AD crate split and the algebra
interaction model. For how the algebra type `A` (e.g., `Standard<T>`) affects
which backend primitives are dispatched during the AD formulas, see
[algebra.md](./algebra.md).

For the step-by-step mathematical derivations of each rule, see the
[AD Formula Notes](../AD/index.md).

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
| `eig_rrule` | `(tensor, cotangent: &EigenCotangent) -> AdResult<Tensor>` |
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
| `eig_frule` | `(tensor, tangent) -> AdResult<(EigenResult, EigenResult)>` |
| `lstsq_frule` | `(a, b, tangent_a, tangent_b) -> AdResult<(LstsqResult, LstsqResult)>` |
| `cholesky_frule` | `(tensor, tangent) -> AdResult<(Tensor, Tensor)>` |
| `solve_frule` | `(a, b, tangent_a, tangent_b) -> AdResult<(Tensor, Tensor)>` |
| `inv_frule` | `(tensor, tangent) -> AdResult<(Tensor, Tensor)>` |
| `det_frule` | `(tensor, tangent) -> AdResult<(Tensor, Tensor)>` |
| `slogdet_frule` | `(tensor, tangent) -> AdResult<(SlogdetResult, SlogdetResult)>` |
| `eig_frule` | `(tensor, tangent) -> AdResult<(EigenResult, EigenResult)>` |
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
    ├── CPU: matricize → faer (dsyev/dgesdd/dgeqrf/dgetrf/dpotrf) → unmatricize
    │
    └── GPU: matricize → cuSOLVER (Xgesvd/Xgeqrf/Xgetrf/Xpotrf) → unmatricize
```

The linalg crate calls `tenferro-prims` operations for its AD formulas
(BatchedGemm, ElementwiseMul, etc.) but calls external LAPACK/cuSOLVER
directly for the forward decompositions (not through TensorPrims).

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
