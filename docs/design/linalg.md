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
