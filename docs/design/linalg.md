# Linear Algebra

`tenferro-linalg` is the public tensor linalg layer of the workspace. Its role
is to validate contracts, lower composite operations, and expose structured
results and AD-facing APIs. It is not the backend execution contract itself.

## Position in the Workspace

```
tenferro-internal-tensor
    structural views
        │
        ├── tenferro-prims
        │      semiring/scalar/analytic execution families
        │
        └── tenferro-linalg-prims
               factorization and solve kernel contracts
                    │
                    ▼
              tenferro-linalg
              public tensor linalg APIs
```

## Responsibilities

`tenferro-linalg` owns:

- shape and option validation
- public result structs and ergonomic APIs
- composite lowering
- oracle and tenferro frontend integration
- stateless AD rules for supported operations

`tenferro-linalg` does not own:

- backend-specific kernel interfaces
- CPU/GPU direct dispatch branches
- standalone structural view operations

## Kernel Basis vs Composite API

The public API is larger than the backend kernel basis.

### Kernel-oriented operations

These lower directly to `tenferro-linalg-prims`:

- `solve`
- `solve_triangular`
- `qr`
- `svd`
- `lu_factor`
- `cholesky`
- `eigen`
- `eig`

### Composite operations

These remain in `tenferro-linalg` and lower through structural ops, semiring
prims, scalar/analytic prims, and the kernel basis above:

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

AD formulas remain part of `tenferro-linalg`, not `tenferro-linalg-prims`.

That split is deliberate:

- `tenferro-linalg-prims` describes execution contracts
- `tenferro-linalg` owns mathematical differentiation rules over public ops

Some public APIs are naturally primal-only, especially structured status/result
surfaces such as factorization contracts with pivots or `info` metadata.

## Current Implementation Status

The architectural boundary is now active rather than transitional:

- `tenferro-linalg` owns public/composite logic and AD formulas
- `tenferro-prims` owns semiring/scalar/analytic execution
- `tenferro-linalg-prims` owns backend-facing structured linalg kernels

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
