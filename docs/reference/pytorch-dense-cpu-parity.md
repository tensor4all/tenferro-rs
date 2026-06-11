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
| Structural (`tenferro-tensor`) | Yes | Partial | Partial | No | Yes | Yes | tenferro exposes metadata-only view APIs such as `transpose_view`, `slice_view`, `reshape_view`, broadcast views, and diagonal views; AD coverage is not yet documented as a first-class family surface |
| Semiring core / fast path (`tenferro-core-ops` + `tenferro-einsum`) | Yes | Partial | Partial | Yes | Partial | Yes | `einsum` is strong, `Permute` is gone from the core op surface, and semiring execution now routes through `StdTensorOp` / `ExecOp` plus operation-family runtimes; the remaining gap is GPU capability breadth, not legacy layering |
| Scalar (`StdTensorOp` elementwise/reduction ops) | Partial | Partial | Partial | No | Partial | Partial | CPU phase 1 now executes unary `Neg/Conj/Abs/Reciprocal/Real/Imag/Square`, binary `Add/Sub/Mul/Div/Maximum/Minimum/Clamp*`, and reductions `Sum/Prod/Mean/Max/Min`; predicate/select tensor ops such as `where` are still absent |
| Analytic (`StdTensorOp` analytic ops) | Partial | Partial | Partial | No | Partial | Partial | CPU phase 1 now executes unary `Sqrt/Rsqrt/Exp/Expm1/Log/Log1p/Sin/Cos/Tan/Tanh/Asin/Acos/Atan/Sinh/Cosh/Asinh/Acosh/Atanh`, binary `Pow/Atan2/Hypot/Xlogy`, and reductions `Var/Std`; GPU custom-kernel coverage is still absent |
| Linalg kernel (`tenferro-linalg::backend::LinalgBackend`) | Yes | Partial | Partial | Partial | Partial | Partial | Solve/factorization kernels exist and now gate through backend-generic linalg/runtime contracts; the remaining gap is backend breadth, not CPU-named scalar contracts |
| Linalg composite (`tenferro-linalg`) | Yes | Partial | Partial | Partial | Partial | Partial | Public coverage is broad and production entrypoints are capability-driven; the remaining gap is that several composites still bottom out in CPU-only kernels because broader backend support is missing |
| Dyadtensor / AD surface | Partial | Partial | Partial | Partial | Partial | Partial | Eager builders now cover scalar/analytic unary, binary, and reduction families including `add`, `atan2`, `pow`, `hypot`, `exp`, `log`, `sin`, `cos`, `tanh`, `asin`, `acos`, `atan`, hyperbolic families, `sum`, `mean`, `var`, and `std`; predicate/select families are still missing and some AD families remain CPU-complete only |

### Matrix Interpretation

- The biggest parity gap is not tensor linalg primal surface. It is the missing
  dense pointwise / reduction substrate.
- `primal` and `AD` coverage are uneven in different ways:
  `tenferro-linalg` is broad on primal surface but uneven on VJP/JVP/HVP, while
  the current `StdTensorOp` / backend surface is still missing parts of the
  scalar/analytic substrate.
- `CPU/GPU generic` and `layer-clean` are separate axes on purpose.
  Several families already work on CPU but still leak CPU-only runtime choices
  into public or mid-level APIs.

## PyTorch-to-tenferro Mapping

### Structural family

Owned by `tenferro-tensor`:

- PyTorch-style `permute`/`transpose` map to tenferro metadata-only
  `transpose_view`
- `reshape_view`, `slice_view`, view construction, and broadcast/expand-style
  views
- `diagonal`, `select`, `narrow`
- `view_as_real`, `view_as_complex`

These are tensor metadata or view operations and should not be execution prims.

### Semiring family

Owned by `tenferro-core-ops` / `tenferro-einsum` and backend dispatch:

- `einsum`, `matmul`, `bmm`, `tensordot`
- semiring-valid `trace`
- semiring-valid elementwise add/mul fast paths

This is the minimal substrate that must stay usable by `einsum-only` and
tropical backends.

### Scalar family

Owned by `StdTensorOp` elementwise/reduction operations:

- pointwise arithmetic such as `add`, `sub`, `mul`, `div`
- pointwise scalar ops such as `neg`, `conj`, `real`, `imag`, `abs`,
  `reciprocal`, `square`
- scalar reductions such as `sum`, `prod`, `mean`, `max`, `min`
- ordered-real helpers such as `maximum`, `minimum`, `clamp*`
- predicate/select helpers such as `where`

This family is the largest missing substrate relative to PyTorch dense CPU.
The main blocker is no dedicated boolean/predicate tensor substrate yet, which
prevents a clean `where` family and branch-select AD rules.

### Analytic family

Owned by `StdTensorOp` analytic operations:

- `sqrt`, `rsqrt`, `exp`, `expm1`, `log`, `log1p`
- trigonometric / hyperbolic families
- `pow`, `atan2`, `hypot`, `xlogy`
- analytic reductions such as `var` and `std`

### Linalg kernel family

Owned by `tenferro-linalg::backend::LinalgBackend`:

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

These are public APIs that should lower through structural ops, core
`StdTensorOp` families, operation-family runtimes, and the smaller linalg
kernel basis.

### Dyadtensor / AD surface

Owned by `tenferro-ad`, `tenferro-runtime`, and operation-family AD rules:

- reverse / forward / HVP entry points over `einsum`
- eager builder APIs for linalg results
- graph-connected wrappers over supported VJP/JVP families

Today this surface is linalg-heavy and does not yet provide a dense generic
pointwise family comparable to PyTorch eager tensor math.

## Layer Findings

### 1. Dense scalar and analytic substrate is real on CPU, but still incomplete in breadth

`StdTensorOp` scalar and analytic families now have dedicated CPU planning
and execution. The remaining gap is breadth, not existence:

- predicate/select tensor ops such as `where` are still absent
- GPU pointwise/reduction custom kernels are still absent
- GPU capability is still narrower than the CPU implementation set

### 2. Structural reorder now lives in `tenferro-tensor`

The current design keeps axis reordering in `tenferro-tensor` as metadata-only
`transpose_view` operations and uses explicit contiguous/canonicalization
boundaries for execution. The old materializing `Permute` primitive has now
been removed from the core op surface, which aligns the public substrate with the
intended semiring-core design.

### 3. `tenferro-linalg` is public/composite in design and production, but backend breadth is uneven

The crate is structurally split and production runtime entrypoints are now
capability-driven. The remaining issue is not legacy layering; it is that some
composite paths still bottom out in CPU-only kernels because broader backend
coverage has not landed yet.

### 4. `tenferro-linalg` now separates backend-generic kernel scalars from LAPACK-specific helpers

The backend contract now distinguishes:

- `KernelLinalgScalar` for dtypes supported by backend kernel implementations
- `LapackEigScalar` for the narrower CPU eig helper path

That keeps public/high-level linalg and tenferro frontend bounds free of CPU-specific
names while still allowing the CPU backend to own its concrete eig buffer
conversion details.

### 5. Dyadtensor runtime is generic at the API boundary, but backend coverage is still mixed

`tenferro-ad` and operation-family crates now route high-level primal and AD entrypoints
through runtime-dispatch helpers rather than CPU-specific production shortcuts.
The remaining gap is that several families are only implemented deeply enough
for CPU today, so unsupported backends still fail truthfully on capability.

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

- Expand `StdTensorOp` scalar coverage beyond the phase-1 unary/binary/reduction subset
- Add a dedicated predicate/select substrate so `where` and branch-select AD
  families can land without smuggling boolean semantics into scalar core traits
- Broaden `StdTensorOp` analytic coverage and expose the remaining user-facing
  analytic wrappers such as `xlogy`
- Keep the structural/materialization split stable as additional semiring fast
  paths land

### Layer gaps

- Continue broadening backend capability coverage behind the now-generic
  tenferro runtime surface
- Keep composite linalg paths lowering through capability-driven contracts as
  more backends land
- Keep `KernelLinalgScalar` and `LapackEigScalar` separate as backend breadth grows

### Public API and family gaps

- Add dense pointwise builder and AD families so tenferro is not linalg-only
- Audit PyTorch dense CPU public families that still have no tenferro family
  owner
- Continue replay support for currently unsupported scalar-output oracle rows

### Verification gaps

- Add family-level parity tracking to docs/design rather than issue text only
- Keep the audit updated as substrate work lands

## Issue Traceability

- `#443`: the workspace architecture references must reflect the split among
  `tenferro-core-ops`, `tenferro-einsum`, and `tenferro-linalg`
- `#444`: the new scalar and analytic family traits need rustdoc that explains
  current support and reserved vocabulary
- `#445`: the LAPACK-specific eig helper split belongs in
  `tenferro-linalg`, not in the generic scalar contract
- `#446`: this audit document is the durable repo artifact that records the
  family matrix, layer findings, and backlog
- `#441`: remains open because the substrate redesign is larger than this audit
  bundle; the remaining work is broad scalar/analytic expansion, predicate
  substrate design, and continued generic execution cleanup
