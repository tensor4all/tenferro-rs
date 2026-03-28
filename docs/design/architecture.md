# Architecture

This document describes the current high-level layering of the tenferro
workspace after the prims/linalg protocol split and introduces the public /
internal crate taxonomy used by the workspace.

## Workspace Crate Taxonomy

The workspace now uses a public / internal crate taxonomy with three naming buckets:

- `end-user public` crates are the recommended starting points for users
- `protocol public` crates expose execution contracts and foundation APIs
- `internal` crates are implementation-only and will use the `tenferro-internal-` prefix

This naming rule is intentionally simple: package names should make the public
vs internal boundary obvious, while README and API docs explain which public
crate is best for a given use case.

Internal implementation crates will live under `internal/` and set
`publish = false` explicitly so they are easy to spot in both the workspace
layout and the manifest metadata.

## Layered Architecture

```
Layer 5: tenferro-capi
    FFI entry points for tensor, einsum, and linalg functionality

Layer 4: tenferro-einsum
    High-level contraction planning and execution
         tenferro-linalg
    Public linalg APIs, composite lowering, AD-facing result shaping

Layer 3a: tenferro-prims
    TensorSemiringCore
    TensorSemiringFastPath
    TensorScalarPrims
    TensorAnalyticPrims

Layer 3b: tenferro-linalg-prims
    Backend-facing linalg kernel contracts
    (solve, factorization, eigensolvers, SVD, QR, Cholesky)

Layer 2: tenferro-tensor
    Tensor storage, shape/stride metadata, structural views

Shared: tenferro-algebra
    Semiring/algebra vocabulary and scalar typing
        tenferro-device
    workspace error and device abstractions
        chainrules-core / chainrules / tidu
    AD traits, scalar rules, and engine

Layer 1: CPU/GPU backend implementations
    faer / BLAS / LAPACK / cuTENSOR / future GPU linalg providers
```

## Core Design Rules

The protocol split is driven by four rules.

1. `tenferro-einsum` depends only on the semiring core and optional semiring
   fast paths.
2. Structural view operations belong to `tenferro-tensor`, not to prims.
3. `tenferro-linalg` is a public/composite layer and does not own
   backend-specific execution contracts.
4. Backend-facing linalg kernels live in `tenferro-linalg-prims`, not in
   `tenferro-prims`.

This keeps `einsum-only` and tropical backends lightweight while still giving
standard arithmetic backends access to scalar, analytic, and linalg
capabilities.

## Execution Boundaries

### Structural Layer

`tenferro-tensor` owns zero-copy views:

- `permute`
- `reshape`
- `broadcast`
- `diagonal`
- related indexing/view transforms

These operations do not imply execution and therefore are not prims.

### Semiring Execution Layer

`tenferro-prims` exposes four protocol families:

- `TensorSemiringCore<Alg>`
- `TensorSemiringFastPath<Alg>`
- `TensorScalarPrims<Alg>`
- `TensorAnalyticPrims<Alg>`

This is the execution substrate for `einsum`, tropical algebra, and
non-factorization scalar/tensor composites.

### Linalg Execution Layer

`tenferro-linalg-prims` exposes backend-facing structured linalg contracts such
as:

- `solve`
- `solve_triangular`
- `qr`
- `thin_svd`
- `lu_factor`
- `cholesky`
- `eigen_sym`
- `eig`

These contracts are kernel-oriented. They are not a one-to-one mirror of
`torch.linalg` or the public `tenferro-linalg` API.

### Public Linalg Layer

`tenferro-linalg` validates shapes/options, lowers composite operations, and
formats structured results. Examples:

- `matrix_power` lowers to repeated multiplication and inverse paths
- `cond` lowers to norms and singular-value-based building blocks
- `tensorinv` and `tensorsolve` lower through reshape/permute plus `inv`/`solve`

When a dedicated factorization kernel is needed, `tenferro-linalg` routes
through `tenferro-linalg-prims`.

## Current Implementation Status

The family traits and `tenferro-linalg-prims` crate are now the active
execution contracts for the workspace.

Current status by layer:

- `TensorSemiringCore` and `TensorSemiringFastPath` are the sole semiring
  execution contracts for CPU/CUDA/ROCm backends.
- `TensorScalarPrims` and `TensorAnalyticPrims` have real CPU execution for the
  current inventory, while CUDA/ROCm expose truthful unsupported capabilities
  where custom kernels do not yet exist.
- `tenferro-linalg` and tenferro runtime entrypoints route through
  capability-driven backend contracts instead of direct
  `ensure_cpu_backend(...)` or `with_cpu_runtime(...)` production paths.

## Dependency Direction

```
tenferro-algebra ───────┐
tenferro-device ────────┤
chainrules-core ────────┤
                        ▼
                  tenferro-tensor
                        │
        ┌───────────────┴───────────────┐
        ▼                               ▼
  tenferro-prims                 tenferro-linalg-prims
        │                               │
        ├───────────────┐               │
        ▼               ▼               ▼
  tenferro-einsum   extensions     tenferro-linalg
        └───────────────┬───────────────┘
                        ▼
                  tenferro-capi
```

## Performance Principles

The redesign is constrained by three performance principles.

1. Keep BLAS/cuTENSOR-style `alpha * op(inputs) + beta * output` execution
   contracts.
2. Preserve the `einsum` lowering shape:
   `permute view -> MakeContiguous -> BatchedGemm`, with `Contract` as an
   optional fast path.
3. Generalize public protocol descriptors only when backends can still
   specialize at plan time and keep hot loops free of per-element dynamic
   dispatch.

## Backlog Categories

The dense parity audit separates follow-up work into two main buckets.

- substrate gaps: scalar/analytic execution vocabulary breadth and semiring
  compatibility cleanup
- layer gaps: CPU-only runtime assumptions in tenferro and composite linalg

Those categories intentionally stay separate so the workspace does not confuse
missing family coverage with abstraction drift.
