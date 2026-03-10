# PyTorch Dense CPU Parity Audit

This document audits tenferro's dense tensor coverage against the subset of
PyTorch relevant to the current workspace design effort:

- dense tensor primal execution
- VJP / JVP support
- oracle-backed HVP coverage where `tensor-ad-oracles` publishes a family
- layer cleanliness and CPU/GPU-generic abstraction boundaries

It is intentionally family-first rather than a literal one-row-per-PyTorch-op
inventory.

## Scope

This audit covers dense tensor functionality only. Sparse tensors, random
factories, FFT, sorting, indexing-heavy APIs, and neural-network higher-level
surfaces are out of scope.

## Audit Method

The audit groups APIs by tenferro family and then maps relevant PyTorch dense
CPU operations into those families. Coverage is tracked separately for:

- primal execution
- VJP
- JVP
- oracle-backed HVP
- CPU/GPU-generic abstraction cleanliness
- layer cleanliness

Status labels in the matrix use:

- `Yes`: implemented and aligned with the intended layer boundary
- `Partial`: some important coverage exists, but either family coverage or
  abstraction cleanliness is incomplete
- `No`: the family is materially absent for the audit target

## Coverage Matrix

| Family | Primal | VJP | JVP | Oracle-HVP | CPU/GPU generic | Layer-clean | Notes |
|--------|--------|-----|-----|------------|-----------------|-------------|-------|
| Structural (`tenferro-tensor`) | Yes | Partial | Partial | No | Yes | Yes | `permute`, `reshape`, `broadcast`, and `diagonal` exist as tensor views; AD coverage is not yet documented as a first-class family surface |
| Semiring core / fast path (`tenferro-prims`) | Yes | Partial | Partial | Yes | Partial | Partial | `einsum` is strong, but the public family traits still route through legacy `TensorPrims<A>` adapters and legacy `Permute` remains in the crate |
| Scalar (`TensorScalarPrims`) | Partial | No | No | No | Partial | Partial | Vocabulary exists, but only `Neg`, `Conj`, `Abs`, `Reciprocal`, `Mul`, `Sum`, `Max`, and `Min` are actually wired to legacy prims |
| Analytic (`TensorAnalyticPrims`) | Partial | No | No | No | Partial | Partial | Vocabulary exists, but only `Sqrt` is wired today |
| Linalg kernel (`tenferro-linalg-prims`) | Yes | Partial | Partial | Partial | Partial | Partial | Solve/factorization kernels exist, but CPU eig helpers still leak through `LinalgScalar` and some execution still routes through CPU-local helpers |
| Linalg composite (`tenferro-linalg`) | Yes | Partial | Partial | Partial | Partial | Partial | Public coverage is broad, but many composite paths still rely on `ensure_cpu_backend(...)` and CPU-only helper stacks |
| Dyadtensor / AD surface | Partial | Partial | Partial | Partial | No | No | Eager builders cover `einsum` and many linalg ops, but dense pointwise families are missing and runtime still hard-codes CPU paths |

### Matrix Interpretation

- The biggest parity gap is not tensor linalg primal surface. It is the missing
  dense pointwise / reduction substrate.
- `primal` and `AD` coverage are uneven in different ways:
  `tenferro-linalg` is broad on primal surface but uneven on VJP/JVP/HVP, while
  `tenferro-prims` is still missing much of the scalar/analytic substrate.
- `CPU/GPU generic` and `layer-clean` are separate axes on purpose.
  Several families already work on CPU but still leak CPU-only runtime choices
  into public or mid-level APIs.

## PyTorch-to-tenferro Mapping

### Structural family

Owned by `tenferro-tensor`:

- `permute`, `transpose`, `reshape`, `view`, `expand`
- `diagonal`, `select`, `narrow`
- `view_as_real`, `view_as_complex`

These are tensor metadata or view operations and should not be execution prims.

### Semiring family

Owned by `tenferro-prims`:

- `einsum`, `matmul`, `bmm`, `tensordot`
- semiring-valid `trace`
- semiring-valid elementwise add/mul fast paths

This is the minimal substrate that must stay usable by `einsum-only` and
tropical backends.

### Scalar family

Owned by `TensorScalarPrims`:

- pointwise arithmetic such as `add`, `sub`, `mul`, `div`
- pointwise scalar ops such as `neg`, `conj`, `real`, `imag`, `abs`,
  `reciprocal`, `square`
- scalar reductions such as `sum`, `prod`, `mean`, `max`, `min`
- ordered-real helpers such as `maximum`, `minimum`, `clamp*`, `where`

This family is the largest missing substrate relative to PyTorch dense CPU.

### Analytic family

Owned by `TensorAnalyticPrims`:

- `sqrt`, `rsqrt`, `exp`, `expm1`, `log`, `log1p`
- trigonometric / hyperbolic families
- `pow`, `atan2`, `hypot`, `xlogy`
- analytic reductions such as `var` and `std`

### Linalg kernel family

Owned by `tenferro-linalg-prims`:

- `solve`, `solve_triangular`
- `qr`, `svd`, `lu_factor`, `cholesky`
- `eigen_sym`, `eig`
- the structured tensor result types that travel with those kernels

These are backend contracts, not the full public linalg surface.

### Linalg composite family

Owned by `tenferro-linalg`:

- `inv`, `det`, `slogdet`, `pinv`
- `matrix_exp`, `matrix_power`, `cond`
- `tensorinv`, `tensorsolve`
- `multi_dot`, `vecdot`, `cross`, `vander`
- shape-normalized families such as `svdvals`, `eigvals`, `eigvalsh`,
  `matrix_norm`, and `vector_norm`

These are public APIs that should lower through structural ops, semiring/scalar
prims, and the smaller linalg kernel basis.

### Dyadtensor / AD surface

Owned by `extension/tenferro-dyadtensor` and `chainrules`:

- reverse / forward / HVP entry points over `einsum`
- eager builder APIs for linalg results
- graph-connected wrappers over supported VJP/JVP families

Today this surface is linalg-heavy and does not yet provide a dense generic
pointwise family comparable to PyTorch eager tensor math.

## Layer Findings

### 1. Dense scalar and analytic substrate is still migration-only

`TensorScalarPrims` and `TensorAnalyticPrims` exist as public traits, but the
implementations still lower through blanket adapters over legacy
`TensorPrims<A>`. That is acceptable as a migration step, but it means the
family boundary exists before the substrate does.

### 2. `Permute` remains legacy debt inside `tenferro-prims`

The current design wants `permute` to live in `tenferro-tensor` as a view and
`MakeContiguous` to be the execution boundary. The legacy
`PrimDescriptor::Permute` still exists in `tenferro-prims`, so the crate
surface is not fully aligned with the intended semiring-core design yet.

This debt is tracked as follow-up substrate work under `#441`, not by this
bundle.

### 3. `tenferro-linalg` is public/composite in design but still carries CPU-only debt

The crate is now structurally split, but many composite or structured paths
still guard through `ensure_cpu_backend(...)`. That is an improvement over
hard-coded `CpuContext` signatures, but it is still an explicit marker that the
current implementation is not backend-generic enough.

### 4. `tenferro-linalg-prims` still mixes generic scalar semantics with LAPACK-specific helpers

`LinalgScalar` currently carries eigendecomposition buffer conversion helpers
that only make sense for the current CPU LAPACK-style path. That makes the
trait broader than the true cross-backend contract. This is the concrete layer
problem addressed by `#445`.

### 5. Dyadtensor runtime is still CPU-first

`extension/tenferro-dyadtensor` exposes a runtime enum, but the eager AD
surface still leans on `with_cpu_runtime(...)`, `CpuContext`, and
`CpuBackend`-specific bounds. That means the public AD story is not yet
CPU/GPU generic even where the math itself is backend-parametric.

### 6. Oracle-HVP coverage is meaningful but still selective

`tensor-ad-oracles` replay now covers many Batch A and Batch B families such as
`cholesky_ex`, `solve_ex`, `lu_factor(_ex)`, `lu_solve`, `cond`,
`matrix_power`, `cross`, `householder_product`, `tensorinv`, `tensorsolve`,
and `vander`. However, several scalar-output and solver-family oracle rows are
still unsupported, including `det`, `eig`, `eigvals`, `eigvalsh`,
`lstsq_grad_oriented`, `lu`, `matrix_norm`, `norm`, `pinv`, `slogdet`,
`solve_triangular`, `svdvals`, and `vector_norm`.

## Follow-up Backlog

### Substrate gaps

- Expand `TensorScalarPrims` beyond the currently wired unary/binary/reduction
  subset
- Expand `TensorAnalyticPrims` beyond `Sqrt`
- Remove legacy `Permute` from the prim execution surface and complete the
  structural/materialization split

### Layer gaps

- Finish removing CPU-only runtime assumptions from dyadtensor eager AD
- Continue reducing `ensure_cpu_backend(...)` reliance in composite linalg paths
- Split LAPACK eig helpers out of `LinalgScalar`

### Public API and family gaps

- Add dense pointwise builder and AD families so dyadtensor is not linalg-only
- Audit PyTorch dense CPU public families that still have no tenferro family
  owner
- Continue replay support for currently unsupported scalar-output oracle rows

### Verification gaps

- Add family-level parity tracking to docs/design rather than issue text only
- Keep the audit updated as substrate work lands

## Issue Traceability

- `#443`: the workspace architecture references must reflect the split among
  `tenferro-prims`, `tenferro-linalg-prims`, and `tenferro-linalg`
- `#444`: the new scalar and analytic family traits need rustdoc that explains
  current support and reserved vocabulary
- `#445`: the LAPACK-specific eig helper split belongs in
  `tenferro-linalg-prims`, not in the generic scalar contract
- `#446`: this audit document is the durable repo artifact that records the
  family matrix, layer findings, and backlog
- `#441`: remains open because the substrate redesign is larger than this audit
  bundle; in particular, legacy `Permute` removal and broad scalar/analytic
  execution are follow-up implementation work
