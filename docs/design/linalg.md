# Linear Algebra

`tenferro-linalg` is the public primal tensor linalg extension of the
workspace. Its role is to validate contracts, build traced linalg operations,
and register linalg execution with the runtime. It is not the backend execution
contract itself, and AD support is intentionally owned by `tenferro-linalg-ad`.

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
    primal linalg extension API and runtime registration
        |
        v
tenferro-linalg-ad
    explicit AD companion for linalg extension ops
```

## Responsibilities

`tenferro-linalg` owns:

- shape and option validation
- public result structs and ergonomic APIs
- composite lowering
- traced extension op payloads and runtime registration

`tenferro-linalg` does not own:

- backend-specific kernel interfaces
- CPU/GPU direct dispatch branches
- standalone structural view operations
- AD registration or differentiation formulas

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

## Shape Convention

All matrix APIs use column-major tensor layout with:

- first two dimensions = matrix dimensions
- remaining dimensions = batch dimensions

This is the column-major counterpart to PyTorch's trailing matrix convention.

## AD Boundary

AD formulas live in `tenferro-linalg-ad`, not `tenferro-linalg`.

That split is deliberate:

- `tenferro-linalg` stays primal-only and can be used without AD dependencies
- `tenferro-linalg-ad` depends on both `tenferro-ad` and `tenferro-linalg` to
  register mathematical differentiation rules over linalg extension ops

Some public APIs are naturally primal-only, especially structured status/result
surfaces such as factorization contracts with pivots or `info` metadata.

## Current Implementation Status

The architectural boundary is now active rather than transitional:

- `tenferro-tensor` owns backend-facing structured linalg kernels through
  `TensorBackend`
- `tenferro-linalg` owns primal extension APIs and runtime registration
- `tenferro-linalg-ad` owns AD registration and linalg differentiation rules

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
