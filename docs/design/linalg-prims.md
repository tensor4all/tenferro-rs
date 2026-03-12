# Linalg Prims

`tenferro-linalg-prims` defines the backend-facing tensor linalg contracts used
by `tenferro-linalg`. It is intentionally narrower than a full public linalg
API surface.

## Why This Crate Exists

The redesign separates two concerns that were previously coupled:

- public/composite tensor linalg APIs
- backend-facing structured kernel contracts

`tenferro-linalg` owns the first. `tenferro-linalg-prims` owns the second.

This keeps `tenferro-prims` focused on semiring/scalar execution substrate and
prevents `einsum-only` or tropical backends from inheriting linalg-specific
requirements.

## What Belongs Here

Only operations that naturally map to structured backend kernels belong in
`tenferro-linalg-prims`.

Current kernel basis:

- `solve`
- `solve_triangular`
- `qr`
- `thin_svd`
- `lu_factor`
- `cholesky`
- `eigen_sym`
- `eig`

Associated structured result types also live here:

- `QrTensorResult`
- `SvdTensorResult`
- `LuTensorResult`
- `EigenTensorResult`
- `EigTensorResult`

## What Does Not Belong Here

Composite public APIs stay in `tenferro-linalg`.

Examples:

- `matrix_power`
- `cond`
- `tensorinv`
- `tensorsolve`
- `multi_dot`
- `vecdot`
- `vander`

These are public linalg operations, but they are not backend kernel contracts.
They should lower through tensor structural ops, semiring/scalar prims, and
the smaller kernel basis above.

## Relation to `tenferro-linalg`

`tenferro-linalg` is expected to:

1. validate public API contracts
2. normalize shape/axis options
3. lower composites to prims or linalg-prims
4. expose structured public results

`tenferro-linalg` should not directly branch on backend types or contain
backend-specific execution kernels.

## Relation to `tenferro-prims`

The two crates are peers, not parent/child abstractions.

- `tenferro-prims` covers semiring, scalar, and analytic execution families
- `tenferro-linalg-prims` covers structured factorization and solve kernels

High-level linalg code may depend on both families:

- semiring/scalar prims for composites
- linalg-prims for factorization kernels

## Current Status

The crate exists and is wired into backend implementations as the canonical
backend-facing linalg contract. Some concrete backends still use local helper
modules internally, but those helpers now sit behind `tenferro-linalg-prims`
instead of acting as a competing public abstraction.

One important current debt is that `LinalgScalar` still carries
LAPACK-oriented eigendecomposition helper requirements. That helper surface is
more specific than the true backend-generic scalar contract and should be
isolated into a narrower CPU-oriented trait.
