# Tenferro PyTorch-Aligned AD Surface Design

## Goal

Align the public `tenferro::Tensor` linalg AD surface with PyTorch for the remaining complex-sensitive operators while keeping the internal implementation on tenferro's runtime-dispatched `primal + frule/rrule + LinearizedOp::jvp/vjp` stack.

## Scope

This design covers three public-surface changes:

1. `lstsq` result contract
2. `norm` API split
3. `eig` / `eigh` public contract split

It does not change the `tidu`-style internal AD architecture. The internal execution model remains:

- public `Tensor` facade
- runtime-dispatched primal kernels
- `frule` / `rrule` as mathematical reference rules
- `LinearizedOp::jvp/vjp` as the runtime seam

## Why PyTorch Alignment

The remaining blockers are not just missing dtype gates. They are public-contract mismatches.

- `lstsq` needs multiple outputs with distinct differentiability.
- `norm` needs separate vector and matrix semantics.
- `eig` and `eigh` need different output dtype and gauge contracts.

PyTorch solves these problems at the public API layer instead of hiding them behind a single overloaded surface. Tenferro should do the same.

## Architecture

### Internal Model

Internal implementation remains generic across CPU / CUDA / ROCm through the existing runtime-dispatch layer.

- primal execution dispatches through `with_linalg_runtime(...)`
- `frule` / `rrule` define operator-local math
- `LinearizedOp::jvp/vjp` packages that math into the AD runtime seam

This means public API changes should not introduce backend-specific AD paths. New support should be expressed as:

1. runtime-generic primal
2. runtime-generic `frule`
3. runtime-generic `rrule`
4. runtime-generic `LinearizedOp::jvp/vjp`
5. public `Tensor` seam exposure

### Public Surface Direction

Tenferro should explicitly model the same distinctions that PyTorch models:

- `lstsq` exposes solution plus extra outputs
- `norm` is split into `norm`, `vector_norm`, and `matrix_norm`
- `eig` and `eigh` are distinct APIs

This avoids ad hoc special-casing in AD rules and keeps public semantics aligned with oracle expectations.

## Design Decisions

### 1. `lstsq`

Replace the current two-field result contract with a PyTorch-aligned structured result.

Public result:

- `solution`
- `residuals`
- `rank`
- `singular_values`

Differentiability:

- differentiable: `solution`, `residuals`
- non-differentiable / auxiliary: `rank`, `singular_values`

Important semantic change:

- `residuals` are squared residual summaries, not raw `b - A x`
- `residuals` are real-valued
- `residuals` may be empty, matching oracle and PyTorch semantics

This removes the current mismatch where tenferro stores a raw residual tensor but the oracle and PyTorch both treat residuals as a summarized output.

### 2. `norm`

Split the current single public `norm(NormKind)` surface into:

- `vector_norm`
- `matrix_norm`
- `norm` as a convenience wrapper

The important semantic rule is:

- complex input is allowed
- output is always real-valued

This matches both oracle families and PyTorch's public contract. It also removes the need to overload one enum with incompatible vector and matrix meanings.

Recommended public types:

- `VectorNormOrd`
- `MatrixNormOrd`

The existing `NormKind` should be removed rather than preserved as a legacy compatibility layer.

### 3. `eig` and `eigh`

Publicly split the APIs.

- `eig()` for general square matrices
- `eigh()` for Hermitian / symmetric matrices

`eigen()` should be renamed to `eigh()` in the public surface.

Public dtype/result rules:

- `eig`
  - eigenvalues: always complex
  - eigenvectors: always complex
- `eigh`
  - eigenvalues: always real
  - eigenvectors: same dtype as input

Internal helper sharing is allowed, but public contracts must remain split.

For `eig`, real-input backward may still need a `handle_r_to_c`-style projection internally, just as PyTorch does.

## Backend and Runtime Requirements

These public changes do not require separate CPU-only AD implementations.

They require that each supported operator be expressible against the runtime capability layer already used by tenferro:

- CPU path
- CUDA path
- ROCm path

If a backend lacks capability, the correct behavior is to report unsupported runtime capability through the existing dispatch boundary rather than introducing a special public API fork.

## Testing Strategy

Each operator family should continue to pass through three layers:

1. oracle replay
2. internal linearized seam tests
3. public `Tensor` integration tests

Repository rule:

- if `LinearizedOp::jvp/vjp` is not a thin `frule/rrule` delegation, add focused seam tests

This is especially important for operators whose packaged outputs differ from the raw mathematical objects, such as `lstsq` and `eig/eigh`.

## Rollout Order

Recommended order:

1. `norm`
2. `eig` / `eigh`
3. `lstsq`

Rationale:

- `norm` has the clearest surface mismatch and smallest internal contract shock
- `eig/eigh` benefits from early public naming and contract cleanup
- `lstsq` has the heaviest result-contract rewrite and should come after the surface patterns are established

## Non-Goals

- no HVP work
- no public dual-number API
- no compatibility shims for removed names
- no legacy preservation of `NormKind` / `eigen()` / old `lstsq` result layout

## Summary

Tenferro should align its public linalg AD surface with PyTorch while keeping the existing internal runtime-dispatched AD architecture.

The key principle is:

- PyTorch-aligned public contracts
- tenferro-native generic implementation

That gives clear semantics for users, preserves backend-generic AD, and avoids ad hoc rule-layer workarounds.
