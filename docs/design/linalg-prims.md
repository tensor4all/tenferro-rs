# Linalg Prims

This file keeps the historical `linalg-prims.md` name, but the current
workspace no longer has a separate `tenferro-linalg-prims` crate. The
backend-facing tensor linalg contracts are owned by
`tenferro-linalg::backend::LinalgBackend` and implemented by backend crates
such as `tenferro-cpu` and `tenferro-gpu`.

## Why This Crate Exists

The redesign separates two concerns that were previously coupled:

- public/composite tensor linalg APIs
- backend-facing structured kernel contracts

`tenferro-linalg` owns both the public linalg API and the linalg extension
runtime, while `LinalgBackend` defines the narrower backend kernel surface.
Core scalar, analytic, structural, and contraction vocabulary lives in
`tenferro-internal-ops::StdTensorOp`, with primitive metadata supplied by
`tenferro-core-ops`, and is executed through `tenferro-runtime::ExecOp` and
`tenferro-tensor::TensorBackend`.
This prevents generic tensor backends and operation-family crates from
inheriting linalg-specific requirements.

## What Belongs Here

Only operations that naturally map to structured backend kernels belong in the
`LinalgBackend` contract.

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
They should lower through tensor structural ops, core `StdTensorOp` families,
operation-family runtimes, and the smaller kernel basis above.

## Relation to `tenferro-linalg`

`tenferro-linalg` is expected to:

1. validate public API contracts
2. normalize shape/axis options
3. lower composites to core tensor ops or `LinalgBackend` kernels
4. expose structured public results

`tenferro-linalg` should not directly branch on backend types or contain
backend-specific execution kernels.

## Relation to Core Tensor Ops

The linalg backend contract is peer to the core tensor backend contract, not a
parent/child abstraction.

- `StdTensorOp` / `ExecOp` cover scalar, analytic, structural, reduction, and
  contraction execution vocabulary
- `LinalgBackend` covers structured factorization and solve kernels

High-level linalg code may depend on both families:

- core tensor ops for composites
- `LinalgBackend` for factorization kernels

## Current Status

`LinalgBackend` exists and is wired into backend implementations as the
canonical backend-facing linalg contract. Some concrete backends still use
local helper modules internally, but those helpers sit behind `LinalgBackend`
instead of acting as competing public abstractions.

The scalar side is intentionally split:

- `LinalgScalar` describes the general scalar behavior shared by linalg code
- `KernelLinalgScalar` marks the dtypes that backend kernel contracts currently
  support
- `LapackEigScalar` isolates the narrower CPU eig buffer conversion helpers

That separation keeps public/high-level linalg APIs backend-generic without
leaking CPU-specific scalar trait names.

## Prepared Factorization Contract

Prepared factorization is a separate public expert path; it does not change
the convenient owned-output methods on `LinalgBackend`. An immutable plan
binds the operation, shape, dtype, options, provider, placement, and execution
context. A caller explicitly allocates one opaque mutable workspace per
concurrency lane and supplies caller-owned destinations to `execute_into`.

The first implementation is compact SVD through Faer. For an `m x n` input and
`k = min(m, n)`, its exact outputs are `U: [m, k]`, `S: [k]`, and
`Vt: [k, n]`. The workspace retains Faer scratch, signed-stride input packing,
singular-value staging where required by the scalar ABI, and the `V` staging
needed to produce public conjugate-transposed `Vt`.

Execution follows these invariants:

- validate backend/workspace identity, all metadata, layout, placement, and
  conservative alias regions before the first output write;
- return structured capability errors for unsupported provider, dtype,
  placement, or layout instead of calling the owned API;
- pass compatible borrowed inputs and compact destinations directly to the
  provider, using only workspace-owned staging at documented boundaries;
- apply gauge normalization in place; and
- perform no Rust global-allocator calls after plan/workspace/output warm-up.

`CpuLinalgBinding` is a narrow opaque interop contract owned by
`tenferro-cpu`. It retains coordinator and context allocations so backend
identity cannot suffer pointer reuse, but application code must not inspect or
construct it. Provider-specific resources remain private to the implementation
and never appear in the backend-neutral prepared API.

The workspace is deliberately not `Clone`. Multiple workspaces may be created
from one plan, while Rust's exclusive `&mut SvdWorkspace` prevents ordinary
concurrent reuse of a single workspace. Validation failures are atomic for all
destinations; a provider numerical failure after writes begin may leave partial
output and is reported without an allocating rollback.
