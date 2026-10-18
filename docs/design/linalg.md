# Linear Algebra

`tenferro-linalg` is the public tensor linalg extension of the workspace. Its
role is to validate contracts, build traced and eager linalg operations,
register linalg execution with the runtime, and provide optional linalg AD
rules behind the `autodiff` feature. It is not the backend execution contract
itself.

## Position in the Workspace

```text
tenferro-tensor
    TensorBackend, CPU kernels, linalg execution
        |
        v
tenferro-runtime
    traced graph runtime and extension dispatch
        |
        v
tenferro-linalg
    linalg extension API, runtime registration, and optional AD rules
```

## Responsibilities

`tenferro-linalg` owns:

- shape and option validation
- `TensorLinalgExt`, `TensorReadLinalgExt`, and `TypedTensorLinalgExt` for
  backend-explicit primal execution
- fixed-arity result tuples and typed real/complex output mappings
- composite lowering
- traced extension op payloads and runtime registration
- optional eager helpers and linalg AD rules behind `autodiff`

`tenferro-linalg` does not own:

- backend-specific kernel interfaces
- CPU/GPU direct dispatch branches
- standalone structural view operations
- mandatory AD dependencies for primal-only users

## Kernel Basis vs Composite API

The public API is larger than the backend kernel basis.

### Kernel-oriented operations

These lower directly to `TensorBackend` linalg methods:

- `solve`
- `solve_triangular`
- `qr`
- `svd`
- `lu_factor`
- `cholesky`
- `eigen`
- `eig`

### Composite operations

These remain in `tenferro-linalg` and lower through structural ops, core tensor
ops, scalar/analytic ops, and the kernel basis above:

- `matrix_power`
- `cond`
- `tensorinv`
- `tensorsolve`
- `multi_dot`
- `vecdot`
- `vander`
- `inv` and structured `*_ex` wrappers built from solve/factorization kernels

The key rule is that public API breadth does not imply backend kernel breadth.

### CPU kernel ownership

The CPU linalg kernels are not injectable. `tenferro-linalg`'s CPU route calls
the extracted `tlinalg` (faer) provider and, in a `blas` build, `tlinalg-blas`
directly, through the adapter in `src/cpu/tlinalg.rs`. The former runtime
kernel slot and `install_linalg_kernels` hook were removed by #2004; there is no
per-primitive override of the CPU linalg basis.

compact Householder QR state compositions (append, from-factors, `R`,
`Q` columns) built on the providers' reflector kernels.

Batch lanes. Every batched CPU linalg family hands one parallelism token to
`tlinalg` - `Sequential`, or the selected pool with the thread budget - and the
provider owns lane selection, vendor batching, and scratch within that budget.
The former host lane plan and `CpuBatchPolicy` plumbing were removed by #2004.

Scratch ownership. The faer lane drives every family through the provider's
borrowed-workspace pair: `<family>_workspace(shape, batch, exec)` reports the
bytes one call needs at the route the *same* `Exec` resolves, and
`<family>_into(..., workspace, exec)` carves its complete per-lane scratch
(worst-case alignment padding included) from that one region. The region is a
byte view over the session buffer pool, and fresh outputs are pooled
`MaybeUninit` leases the provider writes in full, so the storage returns to the
pool when the caller drops the tensor. A warm pool therefore leaves only host
bookkeeping allocations per call (#2178 item 6). A new family uses that pair,
not the provider's `Vec` wrappers, and maps the provider's
`Error::InsufficientWorkspace` as an internal-contract failure because the
host's own query sized the call.

## Concrete, Read, And Typed Boundary

Owned dynamic tensors call the matching `LinalgBackend` owned hook. Borrowed
tensor reads use the `_read` surface and preserve strided layouts until the
selected provider performs documented same-placement canonicalization. Typed
tensors erase only the borrowed input through `TensorScalar::tensor_read`, then
validate each dynamic backend output while downcasting it to the fixed typed
tuple.

`LinalgScalar` is sealed to `f32`, `f64`, `Complex32`, and `Complex64`. It uses
`TensorScalar::Real` for singular values, Hermitian eigenvalues, determinant
log-magnitudes, and norms, and supplies a `Complex` associated type for general
eigendecomposition. These adapters never perform an implicit device transfer;
unsupported layout or placement combinations remain typed backend errors.

## Shape Convention

All matrix APIs use column-major tensor layout with:

- first two dimensions = matrix dimensions
- remaining dimensions = batch dimensions

This is the column-major counterpart to PyTorch's trailing matrix convention.

## AD Boundary

AD formulas live in `tenferro-linalg` behind the `autodiff` feature.

That feature boundary is deliberate:

- primal-only users can depend on `tenferro-linalg` without AD dependencies
- AD users enable `tenferro-linalg/autodiff` and pass the owned rule set into an
  explicit `tenferro_ad::AdContext`
- the process-global registration API is retained only as a compatibility
  bridge

Some public APIs are naturally primal-only, especially structured status/result
surfaces such as factorization contracts with pivots or `info` metadata.

## QR Solver Status Reporting

Ordinary QR (`LinalgBackend::qr` and `qr_with_options`, including the traced and
eager surfaces) does not inspect the CUDA cuSOLVER `geqrf`/`orgqr` device status
flags. Reading them is a blocking device-to-host copy that waits for the
factorization kernels, and the flags report neither a reachable illegal-argument
case nor a rank-deficient factor. Synchronous provider return codes are still
checked by the cuSOLVER wrappers; the doc-hidden compact Householder hooks keep
their own `geqrf` status read.

- Caller mistakes are rejected up front, before any vendor call: rank, shape,
  dtype, dimensions, and leading dimensions are validated by
  `tenferro-linalg`, so the vendor's illegal-argument report is unreachable and
  need not be observed.
- Rank deficiency is not an error. An exactly zero diagonal of `R` still yields
  a valid factorization; cuSOLVER does not set its QR flags for it (the A100
  regression in `tests/integration/gpu_linalg.rs` pins this), and the CPU
  providers and `torch.linalg.qr` behave the same. The caller's own
  verification decides whether the factors are acceptable.

Other decompositions keep their status checks because their flags report
non-convergence that invalidates the result.

## Solve Lowering

Traced `solve` and tracked eager `solve` emit one `LuFactorSolve` extension op
with outputs `(x, packed_lu, pivots)`. The CPU backend implements it as one
fused kernel: each matrix is factored in a pooled scratch buffer and solved in
place, and that scratch buffer becomes the packed LU output, so a primal only
program does the same work as the plain `Solve` kernel. The backend default,
used by CUDA, composes `lu_factor` with `lu_solve_prepared`.

Linearization emits `LuSolvePrepared` on the saved factors, and the transpose
rule solves the adjoint system with the same factors. Reverse mode therefore
factors A exactly once.

`LuFactorSolve` is never pruned to `Solve`. Traced AD prunes unused extension
outputs of the source program before differentiating it; a prune to `Solve`
would drop the saved factors and make the adjoint refactor A. The untracked
eager surface, which never differentiates, still constructs `Solve` directly.

Batched CPU LAPACK kernels and the faer packed LU kernels follow one loop
discipline: one workspace query per call, scratch reused across the batch,
results written directly into the batched output, and no per matrix tensor or
`Vec` allocation. `lu_solve_prepared` applies the pivots inside the provider solve
(one `getrs` per matrix for every transpose and conjugate flag).

## Current Implementation Status

The architectural boundary is now active rather than transitional:

- `tenferro-tensor` owns backend-facing structured linalg kernels through
  `TensorBackend`
- `tenferro-linalg` owns extension APIs, runtime registration, eager helpers,
  and feature-gated linalg differentiation rules

Current debt is mainly about capability breadth and composite coverage:

- some composite families still bottom out in CPU-only kernels because GPU
  capability is not implemented yet
- public primal parity is broader than VJP/JVP/HVP parity for several newer
  families
- some structured results are intentionally primal-only

## Non-Goals

`tenferro-linalg` is not trying to be a literal mirror of `torch.linalg` at the
backend boundary. PyTorch-style API families may exist publicly, but backend
contracts are intentionally smaller and more Rust-structured.

For the broader family-level parity and backlog view, see
[reference/pytorch-dense-cpu-parity.md](../reference/pytorch-dense-cpu-parity.md).
