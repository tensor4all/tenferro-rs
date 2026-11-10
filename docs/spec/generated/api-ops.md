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
- Output: `FullPivLu<V>`
- Dtypes: `a`: f32, f64, Complex32 or Complex64. `p`, `l`, `u`, `q` and `parity` (+1 or -1) have the dtype of `a`.
- Layout: A rank-2 matrix with non-negative strides is read in place by the faer route (every CPU build); a negative stride or a batched input (rank >= 3) is gathered once into a compact buffer of the same placement and dtype. Operands stay on their device; nothing is transferred.
- Errors: `Validation`, `Unsupported`
- Omissions:
  - lanes (`CpuEngine::with_lanes`): not available; returns `Unsupported`

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
