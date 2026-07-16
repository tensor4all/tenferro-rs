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

Callers that factorize many independent matrices may enter an opaque
`PreparedFactorizationSession` once and call operation-specific leaf methods
such as `PreparedSvd::execute_into_session`. The session is operation-neutral
so compact QR and EIGH can share the same lifecycle later. It is neither
constructible nor cloneable, and its callback-scoped lifetime prevents it from
escaping the backend execution domain. Standalone `execute_into` is defined as
a one-leaf session adapter rather than a separate execution path.

`TensorRead` and `TensorWrite` descriptors belong to caller setup and are
evaluated before an execution method is entered. Dynamic-rank view descriptors
own shape and stride metadata, so constructing or cloning them may allocate.
The prepared-leaf contract begins with preconstructed descriptors; it does not
hide descriptor construction inside execution. A separate rank-2 descriptor
API is deliberately not introduced because it would duplicate the canonical
tensor view contract.

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
- reuse all tenferro-controlled output, input-pack, and provider-workspace
  storage without growing it after plan/workspace/output warm-up.

Strict Rust global-allocator zero is evidence for the measured small sequential
Faer regime, not a guarantee for every shape. Faer's numerical kernels may
allocate their own internal metadata even with one worker, and its parallel
Rayon/Spindle path may allocate scheduler storage. Those provider-internal
allocations are explicitly outside the retained tenferro-storage contract and
must be reported by shape and provider rather than hidden behind an unsupported
fixed upper bound. The first implementation documents this distinction instead
of adding a public capability enum before another provider needs one.

On CPU, session entry acquires the resource permit and installs the managed
execution owner exactly once. Leaves validate the retained backend, plan, and
workspace bindings and invoke the provider directly; they do not call
`CpuBackend::install`, broadcast worker state, or reacquire arbitration. The
CPU implementation uses static session construction plus a private enum at the
operation leaf. A tensor `BackendSession` trait object was rejected because
factorization leaves neither need its buffer pool nor its dynamic operation
surface.

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

Prepared-resource `Debug` output distinguishes ownership from sizing. The plan's
`plan_retained_bytes` counts provider-private heap buffers owned by the plan
(zero for Faer's inline `StackReq` metadata), while `workspace_required_bytes`
is the logical byte requirement computed at preparation. A workspace reports
that logical requirement separately from `workspace_retained_bytes`, which
counts its actual scratch length and vector capacities. Shared execution-context
storage, backend binding and identity-token metadata, and inline Rust object
size are outside these provider-resource counts.

`PreparedSvd::retained_bytes` and `SvdWorkspace::retained_bytes` expose those
currently retained provider-private heap bytes without parsing `Debug`. They are
read-only, allocation-free snapshots, not estimates of total process memory.
Provider representation determines the value; callers must not infer monotonic
ordering across shapes, dtypes, options, or providers.
