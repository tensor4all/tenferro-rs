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
- public result structs and ergonomic APIs
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
