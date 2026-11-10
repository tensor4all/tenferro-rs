# Public operation registry (generated)

This page is generated from the operation registries (`api_ops!` in
`crates/<family>/src/api/registry.rs`; core in `tenferro-runtime`). Do not
edit it by hand: each family's section is rewritten by
`UPDATE_API_DOCS=1 cargo test -p <family> api_docs_are_current`, and the same
test fails when the committed section differs from the registry.

## Call grammar

```text
recv.op(operands…, exec)
recv.op_with(operands…, &opts, exec)
recv.op_into(operands…, &opts, exec, dest)

session.op(&x, operands…)          // eager: the session is the receiver
traced.op_with(operands…, &opts)   // traced: no exec, run through prepare → run
```

## Suffixes

| Suffix | Meaning |
|---|---|
| (none) | Default options; returns a new result. |
| `_with` | Explicit `XxxOptions`. |
| `_into` | Writes into a caller-provided `Dest` (`Overwrite`, `Accumulate` or `Scaled`), only where the registry declares it. |
| `_in_place` | Consuming destructive update; exception E01 (in-place FFT). |

Every other spelling of input kind, parameters, layer or output mode is not a
suffix: it is a receiver type, an options field or a `Dest` value.

<!-- api-ops:linalg:begin -->
## linalg

| Op | concrete | traced | eager | Suffixes |
|---|---|---|---|---|
| `svd` | LinalgOps | TracedLinalgOps | EagerLinalgOps | `svd`, `svd_with` |
| `svdvals` | LinalgOps | TracedLinalgOps | EagerLinalgOps | `svdvals`, `svdvals_with` |
| `qr` | LinalgOps | TracedLinalgOps | EagerLinalgOps | `qr`, `qr_with` |
| `householder_qr` | LinalgOps | TracedLinalgOps | EagerLinalgOps | `householder_qr`, `householder_qr_with` |
| `rank_revealing_qr` | LinalgOps | TracedLinalgOps | EagerLinalgOps | `rank_revealing_qr`, `rank_revealing_qr_with` |
| `lu` | LinalgOps | TracedLinalgOps | EagerLinalgOps | `lu`, `lu_with` |
| `full_piv_lu` | LinalgOps | TracedLinalgOps | EagerLinalgOps | `full_piv_lu`, `full_piv_lu_with` |
| `full_piv_lu_solve` | LinalgOps | TracedLinalgOps | EagerLinalgOps | `full_piv_lu_solve`, `full_piv_lu_solve_with` |
| `solve` | LinalgOps | TracedLinalgOps | EagerLinalgOps | `solve`, `solve_with`, `solve_into` (overwrite) |
| `lstsq` | LinalgOps | TracedLinalgOps | EagerLinalgOps | `lstsq`, `lstsq_with` |
| `triangular_solve` | LinalgOps | TracedLinalgOps | EagerLinalgOps | `triangular_solve`, `triangular_solve_with` |
| `cholesky` | LinalgOps | TracedLinalgOps | EagerLinalgOps | `cholesky`, `cholesky_with` |
| `eig` | LinalgOps | TracedLinalgOps | EagerLinalgOps | `eig`, `eig_with` |
| `eigvals` | LinalgOps | TracedLinalgOps | EagerLinalgOps | `eigvals`, `eigvals_with` |
| `eigh` | LinalgOps | TracedLinalgOps | EagerLinalgOps | `eigh`, `eigh_with` |
| `eigvalsh` | LinalgOps | TracedLinalgOps | EagerLinalgOps | `eigvalsh`, `eigvalsh_with` |
| `slogdet` | LinalgOps | TracedLinalgOps | EagerLinalgOps | `slogdet`, `slogdet_with` |
| `det` | LinalgOps | TracedLinalgOps | EagerLinalgOps | `det`, `det_with` |
| `inv` | LinalgOps | TracedLinalgOps | EagerLinalgOps | `inv`, `inv_with` |
| `pinv` | LinalgOps | TracedLinalgOps | EagerLinalgOps | `pinv`, `pinv_with` |
| `norm` | LinalgOps | TracedLinalgOps | EagerLinalgOps | `norm`, `norm_with` |

### svd

Singular value decomposition `a = u diag(s) vh`; `SvdShape::Full` returns the full-matrices factors.

- Options: `SvdOptions`
- Output: `Svd<V, R>`
- Dtypes: `a`: f32, f64, Complex32 or Complex64. `u` and `vh` have the dtype of `a`; `s` is real.
- Layout: On the native CPU provider a rank-2 matrix with non-negative strides (compact, transposed or any strided view) is read in place; a negative stride, or a batched input (rank >= 3), is gathered once into a compact buffer of the same placement and dtype before the computation. The BLAS provider gathers every operand that is not already compact column-major. Operands stay on their device; nothing is transferred. In a lane (`CpuEngine::with_lanes`), a negative-stride input is gathered once into the lane's pool; zero steady-state allocation is promised only for non-negative strides. A batched operand is read in place in a lane.
- Errors: `Validation`, `Unsupported`, `NumericalFailure(NonConvergence)`
- Omissions:
  - `SvdShape::Full` combined with a non-default `gauge`, `derivative_eps` or `driver` returns `Unsupported`: the full-matrices route has no gauge or derivative regularization yet
  - `SvdShape::Full` has no derivative: differentiating through it is unsupported
  - lanes (`CpuEngine::with_lanes`): `SvdShape::Thin`, `SvdShape::Full` and the gauge are available

### svdvals

Singular values only, in descending order.

- Options: `SvdvalsOptions`
- Output: `R`
- Dtypes: `a`: f32, f64, Complex32 or Complex64. The result is real.
- Layout: On the native CPU provider a rank-2 matrix with non-negative strides (compact, transposed or any strided view) is read in place; a negative stride, or a batched input (rank >= 3), is gathered once into a compact buffer of the same placement and dtype before the computation. The BLAS provider gathers every operand that is not already compact column-major. Operands stay on their device; nothing is transferred. In a lane (`CpuEngine::with_lanes`), a negative-stride input is gathered once into the lane's pool; zero steady-state allocation is promised only for non-negative strides. A batched operand is read in place in a lane.
- Errors: `Validation`, `Unsupported`, `NumericalFailure(NonConvergence)`
- Omissions:
  - `driver` is a hint for providers that can choose a driver; the current CPU provider ignores it

### qr

Thin QR factorization `a = q r`.

- Options: `QrOptions`
- Output: `Qr<V>`
- Dtypes: `a`: f32, f64, Complex32 or Complex64. `q` and `r` have the dtype of `a`.
- Layout: On the native CPU provider a rank-2 matrix with non-negative strides (compact, transposed or any strided view) is read in place; a negative stride, or a batched input (rank >= 3), is gathered once into a compact buffer of the same placement and dtype before the computation. The BLAS provider gathers every operand that is not already compact column-major. Operands stay on their device; nothing is transferred. In a lane (`CpuEngine::with_lanes`), a negative-stride input is gathered once into the lane's pool; zero steady-state allocation is promised only for non-negative strides. A batched operand is read in place in a lane.
- Errors: `Validation`, `Unsupported`
- Omissions:
  - lanes (`CpuEngine::with_lanes`): only the default `QrOptions` run in a lane; a non-default gauge returns `Unsupported`

### householder_qr

Compact Householder QR: an opaque factorization state from which `r` and `q_columns` are extracted.

- Options: `HouseholderQrOptions`
- Output: `HouseholderQr<V>`
- Dtypes: `a`: f32, f64, Complex32 or Complex64.
- Layout: The compact factorization takes a compact operand: an owned tensor is read in place, any other operand (a view or a typed tensor) is first made compact by the backend (same placement and dtype). Operands stay on their device; nothing is transferred.
- Errors: `Validation`, `Unsupported`
- Omissions:
  - lanes (`CpuEngine::with_lanes`): not available; returns `Unsupported`

### rank_revealing_qr

Column-pivoted QR with a numerical rank: `q`, `r`, `column_permutation` and `rank`.

- Options: `RankRevealingQrOptions`
- Output: `RankRevealingQrResult<V, I>`
- Dtypes: `a`: f32, f64, Complex32 or Complex64. `q` and `r` have the dtype of `a`; `column_permutation` and `rank` are i64.
- Layout: On the native CPU provider a rank-2 matrix with non-negative strides (compact, transposed or any strided view) is read in place; a negative stride, or a batched input (rank >= 3), is gathered once into a compact buffer of the same placement and dtype before the computation. The BLAS provider gathers every operand that is not already compact column-major. Operands stay on their device; nothing is transferred.
- Errors: `Validation`, `Unsupported`
- Omissions:
  - lanes (`CpuEngine::with_lanes`): not available; returns `Unsupported`

### lu

Partial-pivot LU `a = p l u`; `p` is a permutation matrix and `parity` is the real sign of the permutation.

- Options: `LuOptions`
- Output: `Lu<V>`
- Dtypes: `a`: f32, f64, Complex32 or Complex64. `p`, `l`, `u` and `parity` (+1 or -1) have the dtype of `a`.
- Layout: On the native CPU provider a rank-2 matrix with non-negative strides (compact, transposed or any strided view) is read in place; a negative stride, or a batched input (rank >= 3), is gathered once into a compact buffer of the same placement and dtype before the computation. The BLAS provider gathers every operand that is not already compact column-major. Operands stay on their device; nothing is transferred.
- Errors: `Validation`, `Unsupported`
- Omissions:
  - lanes (`CpuEngine::with_lanes`): not available; returns `Unsupported`

### full_piv_lu

Complete-pivot LU `a = p l u q`; `p` and `q` are permutation matrices and `parity` is the real sign of the permutations.

- Options: `FullPivLuOptions`
- Output: `FullPivLu<V, R>`
- Dtypes: `a`: f32, f64, Complex32 or Complex64. `p`, `l`, `u` and `q` have the dtype of `a`; `parity` (+1 or -1) is real for an owned `Tensor` and has the dtype of `a` on the borrowed route (#2552 Q21).
- Layout: A rank-2 matrix with non-negative strides is read in place by the faer route (every CPU build); a negative stride or a batched input (rank >= 3) is gathered once into a compact buffer of the same placement and dtype. Operands stay on their device; nothing is transferred.
- Errors: `Validation`, `Unsupported`
- Omissions:
  - lanes (`CpuEngine::with_lanes`): not available; returns `Unsupported`
  - typed Complex32/Complex64 input: returns `Unsupported`; the borrowed route returns `parity` in the input dtype while the owned route and the typed result type have it real (#2552 Q21); erased inputs keep the dtype their route returns

### full_piv_lu_solve

Solves `a x = b` through the complete-pivot LU of `a`; a rank-deficient system is rejected.

- Options: `FullPivLuSolveOptions`
- Output: `V`
- Dtypes: `a` and `b`: the same dtype, one of f32, f64, Complex32 or Complex64. `x` has that dtype.
- Layout: `a` and `b` that are not owned compact tensors (views, typed tensors) are first made compact (same placement and dtype); owned tensors are read in place. Operands stay on their device; nothing is transferred.
- Errors: `Validation`, `Unsupported`, `NumericalFailure(Singular)`
- Omissions:
  - lanes (`CpuEngine::with_lanes`): not available; returns `Unsupported`

### solve

Solves `a x = b` with a partial-pivot LU of `a`; `solve_into` overwrites a destination.

- Options: `SolveOptions`
- Output: `V`
- Dtypes: `a` and `b`: the same dtype, one of f32, f64, Complex32 or Complex64. `x` has that dtype.
- Layout: A rank-2 `a` with a rank-1 or rank-2 non-empty `b` is read in place (`b` is gathered straight into the right-hand side); any other shape or layout is gathered once into compact buffers of the same placement and dtype. `solve_into` writes the solution into `dest` directly when `dest` is a column-major matrix of the shape of `b`; otherwise it computes the solution and copies it in. Operands and `dest` must be on the same device; nothing is transferred.
- Errors: `Validation`, `Unsupported`, `NumericalFailure(Singular)`
- Omissions:
  - `solve_into` accepts only `Dest::Overwrite`; `Accumulate` and `Scaled` return `Unsupported` before `dest` is touched
  - lanes (`CpuEngine::with_lanes`): not available; returns `Unsupported`

### lstsq

Least-squares solution of `a x = b` for a tall or square, full-column-rank `a` (thin QR, then a triangular solve).

- Options: `LstsqOptions`
- Output: `V`
- Dtypes: `a` and `b`: the same dtype, one of f32, f64, Complex32 or Complex64. `x` has that dtype.
- Layout: `a` is read like `qr` reads it (rank-2 with non-negative strides in place, otherwise gathered once into a compact buffer of the same placement and dtype); `b` is read in place by the contraction. Operands stay on their device; nothing is transferred.
- Errors: `Validation`, `Unsupported`
- Omissions:
  - an underdetermined (wide, `rows < cols`) system returns a `Validation` error
  - lanes (`CpuEngine::with_lanes`): not available; returns `Unsupported`

### triangular_solve

Solves a triangular system with `a` (or its transpose) on the chosen side; only the selected triangle of `a` is read.

- Options: `TriangularSolveOptions`
- Output: `V`
- Dtypes: `a` and `b`: the same dtype, one of f32, f64, Complex32 or Complex64. `x` has that dtype.
- Layout: A rank-2 `a` with non-negative strides is read in place and a rank-2 `b` is gathered straight into the right-hand side; any other shape or layout is gathered once into compact buffers of the same placement and dtype. Operands stay on their device; nothing is transferred.
- Errors: `Validation`, `Unsupported`, `NumericalFailure(Singular)`
- Omissions:
  - `MatOp::Adjoint` on a complex matrix returns `Unsupported`; for a real matrix it equals `MatOp::Transpose`
  - lanes (`CpuEngine::with_lanes`): not available; returns `Unsupported`

### cholesky

Lower Cholesky factor `l` of a Hermitian positive-definite matrix, `a = l l^H`.

- Options: `CholeskyOptions`
- Output: `V`
- Dtypes: `a`: f32, f64, Complex32 or Complex64. `l` has the dtype of `a`.
- Layout: On the native CPU provider a rank-2 matrix with non-negative strides (compact, transposed or any strided view) is read in place; a negative stride, or a batched input (rank >= 3), is gathered once into a compact buffer of the same placement and dtype before the computation. The BLAS provider gathers every operand that is not already compact column-major. Operands stay on their device; nothing is transferred.
- Errors: `Validation`, `Unsupported`, `NumericalFailure(NonConvergence)`
- Omissions:
  - `Uplo::Upper` returns `Unsupported`: only the lower factor is computed
  - lanes (`CpuEngine::with_lanes`): not available; returns `Unsupported`

### eig

General eigendecomposition `a v = v diag(values)`; eigenvectors have unit 2-norm.

- Options: `EigOptions`
- Output: `Eig<C>`
- Dtypes: `a`: f32, f64, Complex32 or Complex64. `values` and `vectors` are complex (Complex32 for f32 and Complex32 input, Complex64 otherwise).
- Layout: On the native CPU provider a rank-2 matrix with non-negative strides (compact, transposed or any strided view) is read in place; a negative stride, or a batched input (rank >= 3), is gathered once into a compact buffer of the same placement and dtype before the computation. The BLAS provider gathers every operand that is not already compact column-major. Operands stay on their device; nothing is transferred.
- Errors: `Validation`, `Unsupported`, `NumericalFailure(NonConvergence)`, `NumericalFailure(NonFinite)`
- Omissions:
  - lanes (`CpuEngine::with_lanes`): not available; returns `Unsupported`

### eigvals

Eigenvalues of a general matrix, as complex numbers.

- Options: `EigvalsOptions`
- Output: `C`
- Dtypes: `a`: f32, f64, Complex32 or Complex64. The result is complex.
- Layout: On the native CPU provider a rank-2 matrix with non-negative strides (compact, transposed or any strided view) is read in place; a negative stride, or a batched input (rank >= 3), is gathered once into a compact buffer of the same placement and dtype before the computation. The BLAS provider gathers every operand that is not already compact column-major. Operands stay on their device; nothing is transferred. In a lane (`CpuEngine::with_lanes`), a negative-stride input is gathered once into the lane's pool; zero steady-state allocation is promised only for non-negative strides. A batched operand is read in place in a lane.
- Errors: `Validation`, `Unsupported`, `NumericalFailure(NonConvergence)`, `NumericalFailure(NonFinite)`
- Omissions: none

### eigh

Eigendecomposition of a Hermitian matrix (the lower triangle is read); `values` are real and ascending.

- Options: `EighOptions`
- Output: `Eigh<V, R>`
- Dtypes: `a`: f32, f64, Complex32 or Complex64. `vectors` has the dtype of `a`; `values` is real.
- Layout: On the native CPU provider a rank-2 matrix with non-negative strides (compact, transposed or any strided view) is read in place; a negative stride, or a batched input (rank >= 3), is gathered once into a compact buffer of the same placement and dtype before the computation. The BLAS provider gathers every operand that is not already compact column-major. Operands stay on their device; nothing is transferred. In a lane (`CpuEngine::with_lanes`), a negative-stride input is gathered once into the lane's pool; zero steady-state allocation is promised only for non-negative strides. A batched operand is read in place in a lane.
- Errors: `Validation`, `Unsupported`, `NumericalFailure(NonConvergence)`
- Omissions:
  - `driver` is a hint for providers that can choose a driver; the current CPU provider ignores it

### eigvalsh

Eigenvalues of a Hermitian matrix (the lower triangle is read), real and ascending.

- Options: `EigvalshOptions`
- Output: `R`
- Dtypes: `a`: f32, f64, Complex32 or Complex64. The result is real.
- Layout: On the native CPU provider a rank-2 matrix with non-negative strides (compact, transposed or any strided view) is read in place; a negative stride, or a batched input (rank >= 3), is gathered once into a compact buffer of the same placement and dtype before the computation. The BLAS provider gathers every operand that is not already compact column-major. Operands stay on their device; nothing is transferred. In a lane (`CpuEngine::with_lanes`), a negative-stride input is gathered once into the lane's pool; zero steady-state allocation is promised only for non-negative strides. A batched operand is read in place in a lane.
- Errors: `Validation`, `Unsupported`, `NumericalFailure(NonConvergence)`
- Omissions:
  - `driver` is a hint for providers that can choose a driver; the current CPU provider ignores it

### slogdet

Sign and natural log of the absolute determinant of a square matrix.

- Options: `SlogdetOptions`
- Output: `Slogdet<V, R>`
- Dtypes: `a`: f32, f64, Complex32 or Complex64. `sign` has the dtype of `a`; `log_abs_det` is real.
- Layout: On the native CPU provider a rank-2 matrix with non-negative strides (compact, transposed or any strided view) is read in place; a negative stride, or a batched input (rank >= 3), is gathered once into a compact buffer of the same placement and dtype before the computation. The BLAS provider gathers every operand that is not already compact column-major. Operands stay on their device; nothing is transferred.
- Errors: `Validation`, `Unsupported`
- Omissions:
  - lanes (`CpuEngine::with_lanes`): not available; returns `Unsupported`

### det

Determinant of a square matrix, computed as `sign * exp(log_abs_det)`.

- Options: `DetOptions`
- Output: `V`
- Dtypes: `a`: f32, f64, Complex32 or Complex64. The result has the dtype of `a`.
- Layout: On the native CPU provider a rank-2 matrix with non-negative strides (compact, transposed or any strided view) is read in place; a negative stride, or a batched input (rank >= 3), is gathered once into a compact buffer of the same placement and dtype before the computation. The BLAS provider gathers every operand that is not already compact column-major. Operands stay on their device; nothing is transferred.
- Errors: `Validation`, `Unsupported`
- Omissions:
  - lanes (`CpuEngine::with_lanes`): not available; returns `Unsupported`

### inv

Inverse of a square matrix (a solve against the identity).

- Options: `InvOptions`
- Output: `V`
- Dtypes: `a`: f32, f64, Complex32 or Complex64. The result has the dtype of `a`.
- Layout: A rank-2 `a` is read in place by the solve (the identity right-hand side is allocated compact); any other shape or layout is gathered once into a compact buffer of the same placement and dtype. Operands stay on their device; nothing is transferred.
- Errors: `Validation`, `Unsupported`, `NumericalFailure(Singular)`
- Omissions:
  - lanes (`CpuEngine::with_lanes`): not available; returns `Unsupported`

### pinv

Moore-Penrose pseudoinverse through the SVD; singular values below `rtol * max(s)` are dropped.

- Options: `PinvOptions`
- Output: `V`
- Dtypes: `a`: f32, f64, Complex32 or Complex64. The result has the dtype of `a`.
- Layout: On the native CPU provider a rank-2 matrix with non-negative strides (compact, transposed or any strided view) is read in place; a negative stride, or a batched input (rank >= 3), is gathered once into a compact buffer of the same placement and dtype before the SVD. The BLAS provider gathers every operand that is not already compact column-major. Operands stay on their device; nothing is transferred.
- Errors: `Validation`, `Unsupported`, `NumericalFailure(NonConvergence)`
- Omissions:
  - lanes (`CpuEngine::with_lanes`): not available; returns `Unsupported`

### norm

Vector or matrix norm over the chosen axes, selected by `NormOrd`.

- Options: `NormOptions`
- Output: `R`
- Dtypes: `a`: f32, f64, Complex32 or Complex64. The result is real.
- Layout: `NormOrd::Frobenius` over all axes of a strided operand is one fused pass that reads it in place; the other orders read the operand through strided reductions, and a matrix norm over axes that are not the leading two first moves them to the front (a layout change of the same placement and dtype). Operands stay on their device; nothing is transferred.
- Errors: `Validation`, `Unsupported`
- Omissions:
  - lanes (`CpuEngine::with_lanes`): not available; returns `Unsupported`

### Exceptions

| ID | Status | Symbols | Scope | Category | Owner | Reason | Verification | Exit condition |
|---|---|---|---|---|---|---|---|---|
| E08 | retired | `tenferro_linalg::svd_with_exec`<br>`tenferro_linalg::svd_values_with_exec`<br>`tenferro_linalg::qr_with_exec`<br>`tenferro_linalg::eigh_with_exec`<br>`tenferro_linalg::eigh_values_with_exec`<br>`tenferro_linalg::eig_values_with_exec` | free functions that run a factorization on a borrowed execution bundle and buffer pool | legacy \| free function | `tenferro_linalg::with_exec` | retired: replaced by CpuEngine::with_lanes, which hands each lane an engine that runs the ordinary grammar (view.svd_with(&opts, lane)); the six entries were added for #2462 to split threads into lanes | retired: the P7 prototype verified lifetime, buffer reuse, nesting and concurrency, and no retention after the call; the lane route keeps the steady-state allocation counts of the former entries | met: the six functions, their module and the Borrowed-Exec section of the crate docs were removed (#2552 S3); lanes and the grammar replace them |

<!-- api-ops:linalg:end -->

<!-- api-ops:core:begin -->
## core

| Op | concrete | traced | eager | Suffixes |
|---|---|---|---|---|
| `exp` | CoreOps | described | described | `exp` |
| `neg` | CoreOps | described | described | `neg` |
| `sign` | CoreOps | described | described | `sign` |
| `conj` | CoreOps | described | described | `conj` |
| `log` | CoreOps | described | described | `log` |
| `expm1` | CoreOps | described | described | `expm1` |
| `log1p` | CoreOps | described | described | `log1p` |
| `erf` | CoreOps | described | described | `erf` |
| `sin` | CoreOps | described | described | `sin` |
| `cos` | CoreOps | described | described | `cos` |
| `tanh` | CoreOps | described | described | `tanh` |
| `sqrt` | CoreOps | described | described | `sqrt` |
| `rsqrt` | CoreOps | described | described | `rsqrt` |
| `abs` | CoreOps | described | described | `abs` |
| `sigmoid` | CoreOps | described | described | `sigmoid` |
| `silu` | CoreOps | described | described | `silu` |
| `softplus` | CoreOps | described | described | `softplus` |
| `gelu` | CoreOps | described | described | `gelu` |
| `gelu_tanh` | CoreOps | described | described | `gelu_tanh` |
| `mul` | CoreOps | described | described | `mul` |
| `sub` | CoreOps | described | described | `sub` |
| `div` | CoreOps | described | described | `div` |
| `rem` | CoreOps | described | described | `rem` |
| `pow` | CoreOps | described | described | `pow` |
| `maximum` | CoreOps | described | described | `maximum` |
| `minimum` | CoreOps | described | described | `minimum` |
| `add` | CoreOps | described | described | `add`, `add_with`, `add_into` (overwrite) |
| `matmul` | CoreOps | described | described | `matmul` |
| `compare` | CoreOps | described | described | `compare` |
| `clamp` | CoreOps | described | described | `clamp` |
| `where_select` | CoreOps | described | described | `where_select` |
| `dot_general` | CoreOps | described | described | `dot_general`, `dot_general_with` |
| `reduce_sum` | CoreOps | described | described | `reduce_sum` |
| `reduce_max` | CoreOps | described | described | `reduce_max` |
| `reduce_min` | CoreOps | described | described | `reduce_min` |
| `reduce_prod` | CoreOps | described | described | `reduce_prod` |
| `reduce_sum_squares` | CoreOps | described | described | `reduce_sum_squares` |
| `reduce_mean` | CoreOps | described | described | `reduce_mean` |
| `reshape` | CoreOps | described | described | `reshape` |
| `broadcast_in_dim` | CoreOps | described | described | `broadcast_in_dim` |
| `transpose` | CoreOps | described | described | `transpose` |
| `reverse` | CoreOps | described | described | `reverse` |
| `tril` | CoreOps | described | described | `tril` |
| `triu` | CoreOps | described | described | `triu` |
| `extract_diag` | CoreOps | described | described | `extract_diag` |
| `embed_diag` | CoreOps | described | described | `embed_diag` |
| `slice` | CoreOps | described | described | `slice` |
| `pad` | CoreOps | described | described | `pad` |
| `gather` | CoreOps | described | described | `gather` |
| `scatter` | CoreOps | described | described | `scatter` |
| `dynamic_slice` | CoreOps | described | described | `dynamic_slice` |
| `dynamic_update_slice` | CoreOps | — | — | `dynamic_update_slice` |
| `concatenate` | CoreOps | described | described | `concatenate` |
| `convert` | CoreOps | described | described | `convert` |
| `cast` | CoreOps | described | described | `cast` |
| `scale_real` | CoreOps | described | described | `scale_real` |
| `scale_complex` | CoreOps | described | described | `scale_complex` |
| `softmax` | CoreOps | described | described | `softmax` |
| `log_softmax` | CoreOps | described | described | `log_softmax` |
| `masked_softmax` | CoreOps | described | described | `masked_softmax` |
| `masked_log_softmax` | CoreOps | described | described | `masked_log_softmax` |
| `take_along_axis` | CoreOps | described | described | `take_along_axis` |
| `layer_norm` | CoreOps | described | described | `layer_norm` |
| `rms_norm` | CoreOps | described | described | `rms_norm` |
| `select` | — | described | described | `select` |
| `stack` | — | described | described | `stack` |
| `index_select` | — | described | described | `index_select` |
| `slice_axis` | — | described | described | `slice_axis` |
| `take_axis` | — | described | described | `take_axis` |
| `take_rows` | — | — | described | `take_rows` |
| `take_cols` | — | — | described | `take_cols` |
| `take_block` | — | — | described | `take_block` |
| `dynamic_truncate` | — | described | — | `dynamic_truncate` |
| `pad_to_match` | — | described | — | `pad_to_match` |

### exp

Elementwise exponential.

- Options: none
- Output: `V`
- Dtypes: f32, f64, Complex32 or Complex64; the result has the input dtype.
- Layout: Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.
- Errors: `Validation`, `Unsupported`
- Omissions: none

### neg

Elementwise negation.

- Options: none
- Output: `V`
- Dtypes: Any numeric dtype the engine supports for the operation; the result has the input dtype.
- Layout: Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.
- Errors: `Validation`, `Unsupported`
- Omissions: none

### sign

Elementwise sign.

- Options: none
- Output: `V`
- Dtypes: Any numeric dtype the engine supports for the operation; the result has the input dtype.
- Layout: Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.
- Errors: `Validation`, `Unsupported`
- Omissions: none

### conj

Elementwise complex conjugate (identity on real dtypes).

- Options: none
- Output: `V`
- Dtypes: f32, f64, Complex32 or Complex64; the result has the input dtype.
- Layout: Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.
- Errors: `Validation`, `Unsupported`
- Omissions: none

### log

Elementwise natural logarithm.

- Options: none
- Output: `V`
- Dtypes: f32, f64, Complex32 or Complex64; the result has the input dtype.
- Layout: Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.
- Errors: `Validation`, `Unsupported`
- Omissions: none

### expm1

Elementwise `exp(x) - 1`.

- Options: none
- Output: `V`
- Dtypes: f32, f64, Complex32 or Complex64; the result has the input dtype.
- Layout: Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.
- Errors: `Validation`, `Unsupported`
- Omissions: none

### log1p

Elementwise `log(1 + x)`.

- Options: none
- Output: `V`
- Dtypes: f32, f64, Complex32 or Complex64; the result has the input dtype.
- Layout: Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.
- Errors: `Validation`, `Unsupported`
- Omissions: none

### erf

Elementwise error function.

- Options: none
- Output: `V`
- Dtypes: f32, f64, Complex32 or Complex64; the result has the input dtype.
- Layout: Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.
- Errors: `Validation`, `Unsupported`
- Omissions: none

### sin

Elementwise sine.

- Options: none
- Output: `V`
- Dtypes: f32, f64, Complex32 or Complex64; the result has the input dtype.
- Layout: Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.
- Errors: `Validation`, `Unsupported`
- Omissions: none

### cos

Elementwise cosine.

- Options: none
- Output: `V`
- Dtypes: f32, f64, Complex32 or Complex64; the result has the input dtype.
- Layout: Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.
- Errors: `Validation`, `Unsupported`
- Omissions: none

### tanh

Elementwise hyperbolic tangent.

- Options: none
- Output: `V`
- Dtypes: f32, f64, Complex32 or Complex64; the result has the input dtype.
- Layout: Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.
- Errors: `Validation`, `Unsupported`
- Omissions: none

### sqrt

Elementwise square root.

- Options: none
- Output: `V`
- Dtypes: f32, f64, Complex32 or Complex64; the result has the input dtype.
- Layout: Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.
- Errors: `Validation`, `Unsupported`
- Omissions: none

### rsqrt

Elementwise reciprocal square root.

- Options: none
- Output: `V`
- Dtypes: f32, f64, Complex32 or Complex64; the result has the input dtype.
- Layout: Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.
- Errors: `Validation`, `Unsupported`
- Omissions: none

### abs

Elementwise absolute value; complex inputs return their modulus.

- Options: none
- Output: `R`
- Dtypes: Any numeric dtype; complex inputs return the real dtype of the same precision.
- Layout: Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.
- Errors: `Validation`, `Unsupported`
- Omissions: none

### sigmoid

Logistic sigmoid.

- Options: none
- Output: `V`
- Dtypes: f32 or f64.
- Layout: Composed from core primitives inside the engine; operands are read through `TensorRead` and intermediates are engine allocations. A fused engine kernel is used when the engine offers one.
- Errors: `Validation`, `Unsupported`
- Omissions: none

### silu

SiLU (`x * sigmoid(x)`).

- Options: none
- Output: `V`
- Dtypes: f32 or f64.
- Layout: Composed from core primitives inside the engine; operands are read through `TensorRead` and intermediates are engine allocations. A fused engine kernel is used when the engine offers one.
- Errors: `Validation`, `Unsupported`
- Omissions: none

### softplus

Softplus `log(1 + exp(x))`.

- Options: none
- Output: `V`
- Dtypes: f32 or f64.
- Layout: Composed from core primitives inside the engine; operands are read through `TensorRead` and intermediates are engine allocations. A fused engine kernel is used when the engine offers one.
- Errors: `Validation`, `Unsupported`
- Omissions: none

### gelu

GELU with the exact error function.

- Options: none
- Output: `V`
- Dtypes: f32 or f64.
- Layout: Composed from core primitives inside the engine; operands are read through `TensorRead` and intermediates are engine allocations. A fused engine kernel is used when the engine offers one.
- Errors: `Validation`, `Unsupported`
- Omissions: none

### gelu_tanh

GELU with the tanh approximation.

- Options: none
- Output: `V`
- Dtypes: f32 or f64.
- Layout: Composed from core primitives inside the engine; operands are read through `TensorRead` and intermediates are engine allocations. A fused engine kernel is used when the engine offers one.
- Errors: `Validation`, `Unsupported`
- Omissions: none

### mul

Elementwise product with broadcasting.

- Options: none
- Output: `V`
- Dtypes: Both operands share a numeric dtype; the result has that dtype.
- Layout: Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.
- Errors: `Validation`, `Unsupported`
- Omissions: none

### sub

Elementwise difference with broadcasting.

- Options: none
- Output: `V`
- Dtypes: Both operands share a numeric dtype; the result has that dtype.
- Layout: Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.
- Errors: `Validation`, `Unsupported`
- Omissions: none

### div

Elementwise quotient with broadcasting.

- Options: none
- Output: `V`
- Dtypes: Both operands share a numeric dtype; the result has that dtype.
- Layout: Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.
- Errors: `Validation`, `Unsupported`
- Omissions: none

### rem

Elementwise remainder with broadcasting.

- Options: none
- Output: `V`
- Dtypes: Both operands share a numeric dtype; the result has that dtype.
- Layout: Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.
- Errors: `Validation`, `Unsupported`
- Omissions: none

### pow

Elementwise power with broadcasting.

- Options: none
- Output: `V`
- Dtypes: Both operands share a numeric dtype; the result has that dtype.
- Layout: Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.
- Errors: `Validation`, `Unsupported`
- Omissions: none

### maximum

Elementwise maximum with broadcasting.

- Options: none
- Output: `V`
- Dtypes: Both operands share a numeric dtype; the result has that dtype.
- Layout: Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.
- Errors: `Validation`, `Unsupported`
- Omissions: none

### minimum

Elementwise minimum with broadcasting.

- Options: none
- Output: `V`
- Dtypes: Both operands share a numeric dtype; the result has that dtype.
- Layout: Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.
- Errors: `Validation`, `Unsupported`
- Omissions: none

### add

Elementwise sum with broadcasting.

- Options: `AddOptions`
- Output: `V`
- Dtypes: Both operands share a numeric dtype; the result has that dtype.
- Layout: Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel. `add_into` writes the result into the destination's storage.
- Errors: `Validation`, `Unsupported`
- Omissions:
  - traced and eager: only `add`; `add_with` and `add_into` exist on the concrete layer

### matmul

Matrix product of the two leading axes, batched over trailing axes.

- Options: none
- Output: `V`
- Dtypes: Both operands share a numeric dtype; the result has that dtype.
- Layout: Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.
- Errors: `Validation`, `Unsupported`
- Omissions: none

### compare

Elementwise comparison with broadcasting; returns a `bool` tensor.

- Options: none
- Output: `B`
- Dtypes: Both operands share a dtype; the result is `bool`.
- Layout: Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.
- Errors: `Validation`, `Unsupported`
- Omissions: none

### clamp

Elementwise clamp to `[lower, upper]` with broadcasting.

- Options: none
- Output: `V`
- Dtypes: All operands share a real dtype.
- Layout: Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.
- Errors: `Validation`, `Unsupported`
- Omissions: none

### where_select

Elementwise selection: `on_true` where the receiver is true, `on_false` elsewhere, with broadcasting. The receiver is the `bool` condition; a typed receiver must be `TypedTensor<bool>`.

- Options: none
- Output: `<Y::Repr as tenferro_tensor::Repr>::Value`
- Dtypes: Receiver `bool`; the branches share a dtype, which is the result dtype.
- Layout: Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.
- Errors: `Validation`, `Unsupported`
- Omissions: none

### dot_general

General contraction with batch dimensions (`DotGeneralConfig`); `DotGeneralOptions` conjugates an operand.

- Options: `tenferro_tensor::DotGeneralOptions`
- Output: `V`
- Dtypes: Both operands share a numeric dtype; the result has that dtype.
- Layout: Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.
- Errors: `Validation`, `Unsupported`
- Omissions:
  - traced: no `dot_general_with`; conjugate an operand with `conj()` first

### reduce_sum

Sum over `axes` (`None`: every axis); axes are signed, `-1` is the last axis.

- Options: none
- Output: `V`
- Dtypes: Any numeric dtype the engine supports for the operation; the result has the input dtype.
- Layout: Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.
- Errors: `Validation`, `Unsupported`
- Omissions: none

### reduce_max

Maximum over `axes` (`None`: every axis); axes are signed, `-1` is the last axis.

- Options: none
- Output: `V`
- Dtypes: Any numeric dtype the engine supports for the operation; the result has the input dtype.
- Layout: Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.
- Errors: `Validation`, `Unsupported`
- Omissions: none

### reduce_min

Minimum over `axes` (`None`: every axis); axes are signed, `-1` is the last axis.

- Options: none
- Output: `V`
- Dtypes: Any numeric dtype the engine supports for the operation; the result has the input dtype.
- Layout: Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.
- Errors: `Validation`, `Unsupported`
- Omissions: none

### reduce_prod

Product over `axes` (`None`: every axis); axes are signed, `-1` is the last axis.

- Options: none
- Output: `V`
- Dtypes: Any numeric dtype the engine supports for the operation; the result has the input dtype.
- Layout: Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.
- Errors: `Validation`, `Unsupported`
- Omissions: none

### reduce_sum_squares

Sum of squared magnitudes over `axes` (`None`: every axis); axes are signed, `-1` is the last axis.

- Options: none
- Output: `V`
- Dtypes: Any numeric dtype the engine supports for the operation; the result has the input dtype.
- Layout: Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.
- Errors: `Validation`, `Unsupported`
- Omissions: none

### reduce_mean

Mean over `axes` (`None`: every axis); axes are signed, `-1` is the last axis.

- Options: none
- Output: `V`
- Dtypes: Any numeric dtype the engine supports for the operation; the result has the input dtype.
- Layout: Composed from core primitives inside the engine; operands are read through `TensorRead` and intermediates are engine allocations. A fused engine kernel is used when the engine offers one.
- Errors: `Validation`, `Unsupported`
- Omissions: none

### reshape

Reshape to `shape` in column-major element order.

- Options: none
- Output: `V`
- Dtypes: Any numeric dtype the engine supports for the operation; the result has the input dtype.
- Layout: Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel. A view that is not compact in column-major order is gathered by the engine; reshape is a data operation, not a metadata query.
- Errors: `Validation`, `Unsupported`
- Omissions: none

### broadcast_in_dim

Broadcast to `shape`; `dims[i]` is the output axis of input axis `i` (an IR-level dimension map, unsigned like the `*Config` structs).

- Options: none
- Output: `V`
- Dtypes: Any numeric dtype the engine supports for the operation; the result has the input dtype.
- Layout: Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.
- Errors: `Validation`, `Unsupported`
- Omissions: none

### transpose

Permute axes; `perm` is signed (`-1` is the last axis).

- Options: none
- Output: `V`
- Dtypes: Any numeric dtype the engine supports for the operation; the result has the input dtype.
- Layout: Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.
- Errors: `Validation`, `Unsupported`
- Omissions: none

### reverse

Reverse the order of elements along `axes` (signed).

- Options: none
- Output: `V`
- Dtypes: Any numeric dtype the engine supports for the operation; the result has the input dtype.
- Layout: Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.
- Errors: `Validation`, `Unsupported`
- Omissions: none

### tril

Lower triangle of the two leading axes relative to diagonal `k`.

- Options: none
- Output: `V`
- Dtypes: Any numeric dtype the engine supports for the operation; the result has the input dtype.
- Layout: Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.
- Errors: `Validation`, `Unsupported`
- Omissions: none

### triu

Upper triangle of the two leading axes relative to diagonal `k`.

- Options: none
- Output: `V`
- Dtypes: Any numeric dtype the engine supports for the operation; the result has the input dtype.
- Layout: Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.
- Errors: `Validation`, `Unsupported`
- Omissions: none

### extract_diag

Extract the diagonal of axes `axis_a` and `axis_b` (signed).

- Options: none
- Output: `V`
- Dtypes: Any numeric dtype the engine supports for the operation; the result has the input dtype.
- Layout: Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.
- Errors: `Validation`, `Unsupported`
- Omissions: none

### embed_diag

Embed axis `axis_a` of the receiver as the diagonal of a new axis pair; `axis_b` (signed, against the output rank) is the position of the inserted axis.

- Options: none
- Output: `V`
- Dtypes: Any numeric dtype the engine supports for the operation; the result has the input dtype.
- Layout: Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.
- Errors: `Validation`, `Unsupported`
- Omissions: none

### slice

Static strided slice (`SliceConfig`).

- Options: none
- Output: `V`
- Dtypes: Any numeric dtype the engine supports for the operation; the result has the input dtype.
- Layout: Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.
- Errors: `Validation`, `Unsupported`
- Omissions: none

### pad

Pad with edge and interior padding (`PadConfig`).

- Options: none
- Output: `V`
- Dtypes: Any numeric dtype the engine supports for the operation; the result has the input dtype.
- Layout: Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.
- Errors: `Validation`, `Unsupported`
- Omissions: none

### gather

XLA-style gather (`GatherConfig`); `indices` is an integer tensor of any representation.

- Options: none
- Output: `V`
- Dtypes: Operand: any dtype; indices: an integer dtype.
- Layout: Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.
- Errors: `Validation`, `Unsupported`
- Omissions: none

### scatter

XLA-style scatter (`ScatterConfig`); `indices` and `updates` may have any representation.

- Options: none
- Output: `V`
- Dtypes: Operand and updates share a dtype; indices: an integer dtype.
- Layout: Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.
- Errors: `Validation`, `Unsupported`
- Omissions: none

### dynamic_slice

Slice of `sizes` starting at the run-time `starts` (an integer vector).

- Options: none
- Output: `V`
- Dtypes: Operand: any dtype; starts: an integer dtype.
- Layout: Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.
- Errors: `Validation`, `Unsupported`
- Omissions: none

### dynamic_update_slice

Copy of the receiver with `update` written at the run-time `starts`.

- Options: none
- Output: `V`
- Dtypes: Operand and update share a dtype; starts: an integer dtype.
- Layout: Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.
- Errors: `Validation`, `Unsupported`
- Omissions:
  - traced and eager: not available

### concatenate

Concatenate `inputs` along `axis` (signed). Called as `CoreOps::concatenate(&[&a, &b], axis, exec)`.

- Options: none
- Output: `V`
- Dtypes: All inputs share a dtype.
- Layout: Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.
- Errors: `Validation`, `Unsupported`
- Omissions: none

### convert

Checked dtype conversion along tenferro's promotion lattice. The result is a dtype-erased `Tensor` because its dtype is a run-time value.

- Options: none
- Output: `tenferro_tensor::Tensor`
- Dtypes: Any source dtype the conversion supports; the result has dtype `to`.
- Layout: An owned `Tensor` receiver is converted directly. Any other receiver (a borrowed view or a typed tensor) is first gathered once into a compact tensor of the same placement and dtype, because the engine conversion reads an owned tensor.
- Errors: `Validation`, `Unsupported`
- Omissions: none

### cast

Explicit, possibly lossy dtype projection. The result is a dtype-erased `Tensor` because its dtype is a run-time value.

- Options: none
- Output: `tenferro_tensor::Tensor`
- Dtypes: Any source dtype the conversion supports; the result has dtype `to`.
- Layout: An owned `Tensor` receiver is converted directly. Any other receiver (a borrowed view or a typed tensor) is first gathered once into a compact tensor of the same placement and dtype, because the engine conversion reads an owned tensor.
- Errors: `Validation`, `Unsupported`
- Omissions: none

### scale_real

Multiply by a real scalar.

- Options: none
- Output: `V`
- Dtypes: f32, f64, Complex32 or Complex64; the result has the input dtype.
- Layout: Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.
- Errors: `Validation`, `Unsupported`
- Omissions: none

### scale_complex

Multiply by a complex scalar.

- Options: none
- Output: `V`
- Dtypes: Complex32 or Complex64.
- Layout: Every operand is read through `TensorRead` by the engine's `*_read` entry: owned, borrowed, typed and strided inputs reach the same body. Implicit broadcasting is a zero-stride view, never a copy. Whether a strided or negative-stride view is read in place or gathered once (same placement and dtype) is the engine's documented choice for that kernel.
- Errors: `Validation`, `Unsupported`
- Omissions: none

### softmax

Softmax along `axis` (signed).

- Options: none
- Output: `V`
- Dtypes: f32 or f64.
- Layout: Composed from core primitives inside the engine; operands are read through `TensorRead` and intermediates are engine allocations. A fused engine kernel is used when the engine offers one.
- Errors: `Validation`, `Unsupported`
- Omissions: none

### log_softmax

Log-softmax along `axis` (signed).

- Options: none
- Output: `V`
- Dtypes: f32 or f64.
- Layout: Composed from core primitives inside the engine; operands are read through `TensorRead` and intermediates are engine allocations. A fused engine kernel is used when the engine offers one.
- Errors: `Validation`, `Unsupported`
- Omissions: none

### masked_softmax

Softmax along `axis` over the entries where `mask` is true.

- Options: none
- Output: `V`
- Dtypes: Input f32 or f64; mask `bool` of any representation.
- Layout: Composed from core primitives inside the engine; operands are read through `TensorRead` and intermediates are engine allocations. A fused engine kernel is used when the engine offers one.
- Errors: `Validation`, `Unsupported`
- Omissions: none

### masked_log_softmax

Log-softmax along `axis` over the entries where `mask` is true.

- Options: none
- Output: `V`
- Dtypes: Input f32 or f64; mask `bool` of any representation.
- Layout: Composed from core primitives inside the engine; operands are read through `TensorRead` and intermediates are engine allocations. A fused engine kernel is used when the engine offers one.
- Errors: `Validation`, `Unsupported`
- Omissions: none

### take_along_axis

Gather along `axis` (signed) with per-position integer `indices`.

- Options: none
- Output: `V`
- Dtypes: Input: any dtype; indices: an integer dtype.
- Layout: Composed from core primitives inside the engine; operands are read through `TensorRead` and intermediates are engine allocations. A fused engine kernel is used when the engine offers one.
- Errors: `Validation`, `Unsupported`
- Omissions: none

### layer_norm

Layer normalization over `axis` (signed) with optional affine `weight` and `bias`.

- Options: none
- Output: `V`
- Dtypes: f32 or f64; weight and bias share the input representation.
- Layout: Composed from core primitives inside the engine; operands are read through `TensorRead` and intermediates are engine allocations. A fused engine kernel is used when the engine offers one.
- Errors: `Validation`, `Unsupported`
- Omissions: none

### rms_norm

RMS normalization over `axis` (signed) with optional affine `weight` and `bias`.

- Options: none
- Output: `V`
- Dtypes: f32 or f64; weight and bias share the input representation.
- Layout: Composed from core primitives inside the engine; operands are read through `TensorRead` and intermediates are engine allocations. A fused engine kernel is used when the engine offers one.
- Errors: `Validation`, `Unsupported`
- Omissions: none

### select

Alias of `where_select` with the condition as the first argument.

- Options: none
- Output: `V`
- Dtypes: See the method documentation.
- Layout: Recorded as graph operations (traced) or executed by the eager engine.
- Errors: `Validation`, `Unsupported`
- Omissions:
  - concrete: use `where_select`

### stack

Stack equally shaped tensors along a new signed axis.

- Options: none
- Output: `V`
- Dtypes: See the method documentation.
- Layout: Recorded as graph operations (traced) or executed by the eager engine.
- Errors: `Validation`, `Unsupported`
- Omissions:
  - concrete: not available; use `reshape` and `concatenate`

### index_select

Select positions along a signed axis.

- Options: none
- Output: `V`
- Dtypes: See the method documentation.
- Layout: Recorded as graph operations (traced) or executed by the eager engine.
- Errors: `Validation`, `Unsupported`
- Omissions:
  - concrete: not available; use `gather`

### slice_axis

Slice a range along a signed axis.

- Options: none
- Output: `V`
- Dtypes: See the method documentation.
- Layout: Recorded as graph operations (traced) or executed by the eager engine.
- Errors: `Validation`, `Unsupported`
- Omissions:
  - concrete: not available; use `slice`

### take_axis

Take the listed positions along a signed axis.

- Options: none
- Output: `V`
- Dtypes: See the method documentation.
- Layout: Recorded as graph operations (traced) or executed by the eager engine.
- Errors: `Validation`, `Unsupported`
- Omissions:
  - concrete: not available; use `gather`

### take_rows

Take the listed rows of a matrix.

- Options: none
- Output: `V`
- Dtypes: See the method documentation.
- Layout: Recorded as graph operations (traced) or executed by the eager engine.
- Errors: `Validation`, `Unsupported`
- Omissions:
  - concrete and traced: not available

### take_cols

Take the listed columns of a matrix.

- Options: none
- Output: `V`
- Dtypes: See the method documentation.
- Layout: Recorded as graph operations (traced) or executed by the eager engine.
- Errors: `Validation`, `Unsupported`
- Omissions:
  - concrete and traced: not available

### take_block

Take a block given row and column positions.

- Options: none
- Output: `V`
- Dtypes: See the method documentation.
- Layout: Recorded as graph operations (traced) or executed by the eager engine.
- Errors: `Validation`, `Unsupported`
- Omissions:
  - concrete and traced: not available

### dynamic_truncate

Truncate a signed axis to a run-time size.

- Options: none
- Output: `V`
- Dtypes: See the method documentation.
- Layout: Recorded as graph operations (traced) or executed by the eager engine.
- Errors: `Validation`, `Unsupported`
- Omissions:
  - concrete and eager: not available

### pad_to_match

Zero-pad a signed axis to the size of a reference tensor.

- Options: none
- Output: `V`
- Dtypes: See the method documentation.
- Layout: Recorded as graph operations (traced) or executed by the eager engine.
- Errors: `Validation`, `Unsupported`
- Omissions:
  - concrete and eager: not available

### Exceptions

| ID | Status | Symbols | Scope | Category | Owner | Reason | Verification | Exit condition |
|---|---|---|---|---|---|---|---|---|
| E03 | retained | `tenferro_einsum::ConcreteEinsumPlan::clear_cpu_cache`<br>`tenferro_einsum::ConcreteEinsumPlan::cpu_cache_stats` | plan-cache owner methods of the concrete einsum plan | cache owner \| inherent method | `tenferro_einsum::concrete` | a cache owner exposes its cache; these are not tensor operations | the einsum registry tests and the cache tests of tenferro-einsum | kept while ConcreteEinsumPlan owns a CPU cache |
| E04 | retained | `tenferro_runtime::TraceContext`<br>`tenferro_runtime::GraphCompiler::compile`<br>`tenferro_runtime::GraphCompiler::compile_many`<br>`tenferro_runtime::Runtime::prepare`<br>`tenferro_runtime::Runtime::run`<br>`tenferro_runtime::TraceContext::finish` | trace, compile, prepare and run lifecycle | lifecycle \| builder | `tenferro_runtime::{trace, graph, runtime}` | program construction and execution are not tensor operations; their shapes are fixed by the public API snapshot (api/) | scripts/api-snapshot.sh --check and the runtime lifecycle tests | kept |
| E05 | retained | `tenferro_tensor::Tensor::shape`<br>`tenferro_tensor::Tensor::dtype`<br>`tenferro_tensor::Tensor::into_dynamic`<br>`tenferro_tensor::TypedTensor::as_view`<br>`tenferro_tensor::TypedTensorView::transpose_view`<br>`tenferro_tensor::TypedTensorView::slice_view`<br>`tenferro_runtime::TracedTensor::rank`<br>`tenferro_runtime::TracedTensor::sym_size`<br>`tenferro_runtime::TracedTensor::axis_sym_dim`<br>`tenferro_runtime::TracedTensor::shape_of` | descriptor-only operations and metadata queries | metadata \| view | `tenferro_tensor, tenferro_runtime::traced` | they never touch element data, so they take no engine and are not operations of the grammar | tenferro-runtime `metadata_queries_and_views_do_not_copy` (pointer identity) | kept |
| E06 | retained | `tenferro_cpu::with_borrowed_cpu_engine`<br>`tenferro_tensor::Dest::into_accumulation`<br>`num_complex::Complex` | approved external types and owner-scoped doc-hidden bridges | bridge \| external type | `tenferro_cpu, tenferro_tensor::dest` | the lane bridge lets tenferro-linalg reach the lane engine (#2552 S2b); into_accumulation maps a Dest to the backend accumulation descriptor (#2552 Q3); num_complex is the public complex scalar | listed symbol by symbol in api/allowlist.txt; the bridges are #[doc(hidden)] | kept while the owners need the bridge |
| E09 | retained | `tenferro_einsum::Error`<br>`tenferro_einsum::PlanningError`<br>`tenferro_einsum::parse_einsum_subscripts`<br>`tenferro_einsum::parse_einsum_notation`<br>`tenferro_linalg::Error`<br>`tenferro_ad::EagerFftInPlaceError` | domain errors of parsers and planners, and failures that return ownership | error \| domain | `the owning family crates` | parsers and planners report domain errors; the operations wrap them as the `source()` of the layer error | the error-cause tests of each family | kept |
| E10 | retained | `tenferro_tensor::Engine`<br>`tenferro_tensor::TensorDot`<br>`tenferro_tensor::TensorDot::dot_general_read_into_accum`<br>`tenferro_tensor::TensorElementwise::axpby_read_into_accum`<br>`tenferro_linalg::LinalgBackend`<br>`tenferro_fft::FftBackend` | backend and provider traits and their read/accumulate hooks | provider contract \| trait | `tenferro_tensor::backend and the family backends` | a provider-facing contract, not the user grammar; `*_read` and `*_read_into_accum` name backend hooks | the backend contract tests of each crate | kept |

<!-- api-ops:core:end -->

<!-- api-ops:einsum:begin -->
## einsum

| Op | concrete | concrete_tensordot | traced | traced_tensordot | eager | Suffixes |
|---|---|---|---|---|---|---|
| `einsum` | EinsumOps | — | TracedEinsumOps | — | EagerEinsumOps | `einsum`, `einsum_with`, `einsum_into` (overwrite, accumulate, scaled) |
| `tensordot` | — | TensordotOps | — | TracedTensordotOps | EagerEinsumOps | `tensordot`, `tensordot_with` |

### einsum

Einstein summation over N inputs: contract, trace, transpose, reduce and broadcast as the spec says.

- Options: `EinsumOptions`
- Output: `V`
- Dtypes: Every input has the same dtype (any numeric dtype the backend supports); the result has it too.
- Layout: Strided inputs are read in place by the contraction kernels (any strides, including transposed and negative-stride views); an operand that a kernel cannot read is packed once into a compact buffer of the same placement and dtype. A two-input contraction that maps to one dot_general writes straight into the destination. Inputs stay on their device. `_into` checks the destination's dtype and shape before any write; `Accumulate` and `Scaled` read the destination, `Overwrite` and `Scaled` with `beta = 0` never read it.
- Errors: `Validation`, `Unsupported`, `RuntimeState`
- Omissions:
  - `_into` is concrete only: traced and eager results are new values
  - lanes (`CpuEngine::with_lanes`): einsum is unsupported in a lane
  - `Scaled` needs `alpha` and `beta` of the input dtype

### tensordot

Contract `a` with `b` over the given axis pairs (or the last/first `n` axes); free axes of `a` come first.

- Options: `TensordotOptions`
- Output: `V`
- Dtypes: `a` and `b` have the same dtype; the result has it too.
- Layout: Lowered to one dot_general, which reads strided operands in place; an operand that the provider cannot read is packed once into a compact buffer of the same placement and dtype.
- Errors: `Validation`, `Unsupported`
- Omissions:
  - lanes (`CpuEngine::with_lanes`): tensordot is unsupported in a lane

### Exceptions

| ID | Status | Symbols | Scope | Category | Owner | Reason | Verification | Exit condition |
|---|---|---|---|---|---|---|---|---|
| E03 | retained | `tenferro_einsum::ConcreteEinsumPlan::clear_cpu_cache`<br>`tenferro_einsum::ConcreteEinsumPlan::cpu_cache_stats` | cache maintenance of a prepared concrete einsum plan | cache owner \| inherent method | `tenferro_einsum::concrete` | the plan owns its CPU numerical descriptor cache; clearing and inspecting it is not a tensor operation and has no options, inputs or destination | doctests of both methods; the plan's execute/execute_into follow the grammar (inputs, exec, dest) | retained while ConcreteEinsumPlan owns a CPU cache |
| E07 | retired | `tenferro_einsum::TraceContextEinsumExt::einsum`<br>`tenferro_einsum::TraceContextEinsumExt::einsum_notation`<br>`tenferro_einsum::TraceContextEinsumExt::einsum_subscripts`<br>`tenferro_einsum::TraceContextEinsumExt::einsum_with`<br>`tenferro_einsum::TraceContextEinsumExt::einsum_notation_with`<br>`tenferro_einsum::TraceContextEinsumExt::einsum_subscripts_with` | einsum entries that took and returned TraceValue on a `&mut TraceContext` | legacy \| trait | `tenferro_einsum::traced` | retired: replaced by TracedEinsumOps for TraceContext (`trace.einsum(&[&t1, &t2], spec)`), which takes and returns TracedTensor (#2552 decision 4) | retired: a temporary old-vs-new test recorded the same values before removal (#2552 S4) | met: the trait was removed in #2552 S4 |
| E09 | retained | `tenferro_einsum::Error`<br>`tenferro_einsum::PlanningError`<br>`tenferro_einsum::parse_einsum_notation`<br>`tenferro_einsum::parse_einsum_subscripts`<br>`tenferro_einsum::ContractionTree::optimize`<br>`tenferro_einsum::ContractionTree::optimize_with_options`<br>`tenferro_einsum::ContractionTree::from_pairs` | the einsum parser and contraction planner | domain error \| free function | `tenferro_einsum::{error, subscripts, planning}` | parsing and planning are einsum-domain operations whose errors are einsum-domain; the grammar's operations wrap them (concrete: tenferro_tensor::Error::Extension, traced/eager: tenferro_runtime::Error::Extension) with the einsum error as source | error-cause tests reach tenferro_einsum::Error::InvalidSubscripts through source() on all three layers | retained |

<!-- api-ops:einsum:end -->

<!-- api-ops:fft:begin -->
## fft

| Op | concrete | traced | eager | Suffixes |
|---|---|---|---|---|
| `fft` | FftOps | TracedFftOps | EagerFftOps | `fft`, `fft_with` |
| `ifft` | FftOps | TracedFftOps | EagerFftOps | `ifft`, `ifft_with` |
| `rfft` | FftOps | TracedFftOps | EagerFftOps | `rfft`, `rfft_with` |
| `irfft` | FftOps | TracedFftOps | EagerFftOps | `irfft`, `irfft_with` |
| `fftn` | FftOps | TracedFftOps | EagerFftOps | `fftn`, `fftn_with` |
| `ifftn` | FftOps | TracedFftOps | EagerFftOps | `ifftn`, `ifftn_with` |
| `rfftn` | FftOps | TracedFftOps | EagerFftOps | `rfftn`, `rfftn_with` |
| `irfftn` | FftOps | TracedFftOps | EagerFftOps | `irfftn`, `irfftn_with` |

### fft

One-axis discrete Fourier transform along `axis` (default the last axis).

- Options: `FftOptions`
- Output: `C`
- Dtypes: `a`: f32, f64, Complex32 or Complex64. The result is complex (Complex32 for f32/Complex32 inputs, Complex64 otherwise); a real input is transformed as a complex one with zero imaginary part.
- Layout: On the CPU provider a compact column-major operand is read in place; any other strided view (transposed, sliced, negative strides) is gathered once into a compact buffer of the same placement and dtype before the transform. The CUDA provider canonicalizes on the device. Operands stay on their device; nothing is transferred.
- Errors: `Validation`, `Unsupported`
- Omissions:
  - lanes (`CpuEngine::with_lanes`): an FFT on a lane engine returns `Unsupported`

### ifft

One-axis inverse discrete Fourier transform along `axis`.

- Options: `FftOptions`
- Output: `C`
- Dtypes: `a`: Complex32 or Complex64. The result has the dtype of `a`.
- Layout: On the CPU provider a compact column-major operand is read in place; any other strided view (transposed, sliced, negative strides) is gathered once into a compact buffer of the same placement and dtype before the transform. The CUDA provider canonicalizes on the device. Operands stay on their device; nothing is transferred.
- Errors: `Validation`, `Unsupported`
- Omissions:
  - lanes (`CpuEngine::with_lanes`): an FFT on a lane engine returns `Unsupported`

### rfft

One-axis transform of a real signal, returning the one-sided spectrum.

- Options: `FftOptions`
- Output: `C`
- Dtypes: `a`: f32 or f64. The result is the one-sided complex spectrum (Complex32 or Complex64) of length `n / 2 + 1` along `axis`.
- Layout: On the CPU provider a compact column-major operand is read in place; any other strided view (transposed, sliced, negative strides) is gathered once into a compact buffer of the same placement and dtype before the transform. The CUDA provider canonicalizes on the device. Operands stay on their device; nothing is transferred.
- Errors: `Validation`, `Unsupported`
- Omissions:
  - lanes (`CpuEngine::with_lanes`): an FFT on a lane engine returns `Unsupported`

### irfft

Inverse of `rfft`: a one-sided spectrum back to a real signal.

- Options: `FftOptions`
- Output: `R`
- Dtypes: `a`: Complex32 or Complex64 (a one-sided spectrum). The result is real (f32 or f64); its length along `axis` is `n`, by default `2 * (m - 1)`.
- Layout: On the CPU provider a compact column-major operand is read in place; any other strided view (transposed, sliced, negative strides) is gathered once into a compact buffer of the same placement and dtype before the transform. The CUDA provider canonicalizes on the device. Operands stay on their device; nothing is transferred.
- Errors: `Validation`, `Unsupported`
- Omissions:
  - lanes (`CpuEngine::with_lanes`): an FFT on a lane engine returns `Unsupported`

### fftn

N-dimensional discrete Fourier transform over `axes` (default every axis).

- Options: `FftnOptions`
- Output: `C`
- Dtypes: `a`: f32, f64, Complex32 or Complex64. The result is complex, as for `fft`.
- Layout: On the CPU provider a compact column-major operand is read in place; any other strided view (transposed, sliced, negative strides) is gathered once into a compact buffer of the same placement and dtype before the transform. The CUDA provider canonicalizes on the device. Operands stay on their device; nothing is transferred. The transform runs one one-axis transform per axis; the intermediate spectra stay on the device and the normalization is applied once to the whole transform.
- Errors: `Validation`, `Unsupported`
- Omissions:
  - lanes (`CpuEngine::with_lanes`): an FFT on a lane engine returns `Unsupported`

### ifftn

N-dimensional inverse discrete Fourier transform over `axes`.

- Options: `FftnOptions`
- Output: `C`
- Dtypes: `a`: Complex32 or Complex64. The result has the dtype of `a`.
- Layout: On the CPU provider a compact column-major operand is read in place; any other strided view (transposed, sliced, negative strides) is gathered once into a compact buffer of the same placement and dtype before the transform. The CUDA provider canonicalizes on the device. Operands stay on their device; nothing is transferred. The transform runs one one-axis transform per axis; the intermediate spectra stay on the device and the normalization is applied once to the whole transform.
- Errors: `Validation`, `Unsupported`
- Omissions:
  - lanes (`CpuEngine::with_lanes`): an FFT on a lane engine returns `Unsupported`

### rfftn

N-dimensional transform of a real signal; the last transformed axis is one-sided.

- Options: `FftnOptions`
- Output: `C`
- Dtypes: `a`: f32 or f64. The result is complex; the last transformed axis is one-sided (`n / 2 + 1`).
- Layout: On the CPU provider a compact column-major operand is read in place; any other strided view (transposed, sliced, negative strides) is gathered once into a compact buffer of the same placement and dtype before the transform. The CUDA provider canonicalizes on the device. Operands stay on their device; nothing is transferred. The transform runs one one-axis transform per axis; the intermediate spectra stay on the device and the normalization is applied once to the whole transform.
- Errors: `Validation`, `Unsupported`
- Omissions:
  - lanes (`CpuEngine::with_lanes`): an FFT on a lane engine returns `Unsupported`

### irfftn

Inverse of `rfftn`.

- Options: `FftnOptions`
- Output: `R`
- Dtypes: `a`: Complex32 or Complex64. The result is real (f32 or f64); the last transformed axis has length `2 * (m - 1)` unless `shape` says otherwise.
- Layout: On the CPU provider a compact column-major operand is read in place; any other strided view (transposed, sliced, negative strides) is gathered once into a compact buffer of the same placement and dtype before the transform. The CUDA provider canonicalizes on the device. Operands stay on their device; nothing is transferred. The transform runs one one-axis transform per axis; the intermediate spectra stay on the device and the normalization is applied once to the whole transform.
- Errors: `Validation`, `Unsupported`
- Omissions:
  - lanes (`CpuEngine::with_lanes`): an FFT on a lane engine returns `Unsupported`

### Exceptions

| ID | Status | Symbols | Scope | Category | Owner | Reason | Verification | Exit condition |
|---|---|---|---|---|---|---|---|---|
| E01 | retained | `tenferro_fft::EagerFftOps::fft_in_place`<br>`tenferro_fft::EagerFftOps::ifft_in_place`<br>`tenferro_fft::EagerFftInPlaceError` | consuming eager FFTs that reuse the input storage | in-place \| consuming | `tenferro_fft::eager_in_place` | an FFT of a large compact complex tensor can reuse its storage; the grammar's `_into` writes to a separate destination and cannot express a consumed input | src/api/tests/registry.rs: a length change and a real input are Rejected with the input returned unchanged; an accepted call equals the out-of-place fft bit for bit | a consuming `Dest` form in the grammar |
| E02 | retained | `tenferro_fft::FftExecutor::fft_with`<br>`tenferro_fft::FftExecutor::ifft_with`<br>`tenferro_fft::FftExecutor::rfft_with`<br>`tenferro_fft::FftExecutor::irfft_with`<br>`tenferro_fft::FftExecutor::fftn_with`<br>`tenferro_fft::FftExecutor::ifftn_with`<br>`tenferro_fft::FftExecutor::rfftn_with`<br>`tenferro_fft::FftExecutor::irfftn_with`<br>`tenferro_fft::FftExecutor::new`<br>`tenferro_fft::FftExecutor::plan_cache`<br>`tenferro_fft::FftExecutor::plan_cache_mut`<br>`tenferro_fft::FftExecutor::cache_stats`<br>`tenferro_fft::FftExecutor::clear_cache` | the caller-owned FFT plan cache and its transforms | cache owner | `tenferro_fft` | the receiver is the plan cache, not the input; the transforms take the input, the options and the engine in grammar order | src/api/tests/registry.rs: every `_with` equals the `FftOps` result bit for bit and fills the caller's plan cache | plan caches owned by the engine for every caller |

<!-- api-ops:fft:end -->
