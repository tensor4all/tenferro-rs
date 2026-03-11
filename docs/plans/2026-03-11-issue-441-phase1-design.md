# Issue #441 Phase 1 CPU/GPU-Generic Scalar and Analytic Substrate Design

## Summary

Issue `#441` is larger than a single implementation PR. The first PR should
not try to deliver broad CUDA parity. It should instead make the scalar and
analytic tensor substrate real on CPU while preserving a strictly
CPU/GPU-generic public contract for prims, AD, and eager tensor wrappers.

This document defines that first PR.

## Goals

- make `TensorScalarPrims` and `TensorAnalyticPrims` usable as real protocol
  families rather than migration-only vocabulary
- expand CPU primal coverage for the phase-1 scalar and analytic family set
- keep AD formulas backend-independent
- keep dyadtensor eager entry points backend-generic in contract
- ensure future GPU work can plug into the same descriptors and capability
  queries without public API churn

## Non-Goals

- broad CUDA parity for scalar and analytic ops
- custom CUDA pointwise or reduction kernels
- full removal of legacy `PrimDescriptor::Permute`
- full PyTorch dense parity in one PR
- all oracle-backed scalar HVP coverage in the same change

## Constraints

The top-level constraint is stricter than “CPU first”:

- CPU implementation may land first
- CPU-specific public contracts may not

Concretely:

- no new public or mid-level APIs may require `CpuContext`
- no new public or mid-level APIs may branch directly on `CpuBackend` /
  `CudaBackend`
- no new `with_cpu_runtime(...)` or `ensure_cpu_backend(...)` dependencies may
  be introduced for scalar/analytic families
- capability queries must be the way higher layers discover backend support

## Approaches Considered

### 1. Recommended: CPU substrate first, GPU protocol-ready

Implement the new scalar and analytic family surface broadly on CPU. Keep GPU
support descriptor-compatible and capability-query-compatible, but only
advertise the currently available cuTENSOR-backed subset.

This keeps the architecture clean while avoiding a large custom CUDA kernel
project in the same PR.

### 2. CPU plus minimal custom CUDA kernels

Implement a small hand-picked set such as `exp`, `log`, and `sin` on CUDA in
the same PR.

This was rejected for phase 1 because tenferro does not yet have a general
custom CUDA pointwise/reduction substrate. The current CUDA backend in
`tenferro-prims` is cuTENSOR-plan-centric and lacks an NVRTC/PTX/module cache
layer for generic pointwise work.

### 3. AD first, primal later

Extend scalar rule coverage and dyadtensor wrappers before widening the primal
execution substrate.

This was rejected because it would likely reintroduce CPU-specific plumbing and
repeat the current “wrapper exists before substrate exists” mismatch.

## Phase-1 Family Inventory

### Scalar family

- unary: `Neg`, `Conj`, `Abs`, `Reciprocal`, `Real`, `Imag`, `Square`
- binary: `Add`, `Sub`, `Mul`, `Div`, `Maximum`, `Minimum`, `ClampMin`,
  `ClampMax`
- reduction: `Sum`, `Prod`, `Mean`, `Max`, `Min`

### Analytic family

- unary: `Sqrt`, `Rsqrt`, `Exp`, `Expm1`, `Log`, `Log1p`, `Sin`, `Cos`, `Tan`,
  `Tanh`
- binary: `Pow`, `Atan2`, `Hypot`, `Xlogy`
- reduction vocabulary keeps `Var` and `Std`, but phase 1 does not require
  CPU implementation for those two if the capability contract is in place

The criterion is “enough to make the substrate real and useful” rather than
“match the whole PyTorch inventory immediately”.

## Layering and Ownership

### `tenferro-prims`

Owns the execution substrate:

- `TensorScalarPrims`
- `TensorAnalyticPrims`
- CPU planning and execution for the phase-1 set
- backend capability queries

The public descriptors remain family-level and backend-generic. CPU-specific
specialization happens at planning time, not at the API boundary.

### `extern/chainrules-scalarops`

Owns backend-independent scalar math rules:

- primal helper functions when useful
- VJP rules
- JVP rules

No backend branching belongs here.

### `extension/tenferro-dyadtensor`

Owns eager tensor-level AD wiring:

- generic unary/binary/reduction builder APIs
- eager `ad::*` entry points
- runtime/capability-driven execution selection

It must not hard-code CPU runtime assumptions for the new scalar and analytic
families.

### `tenferro-linalg`

Not a primary implementation site for this PR. It may consume the new scalar
and analytic substrate later, but phase 1 should not grow new scalar-specific
CPU-only helpers there.

## CPU Execution Strategy

The public descriptor surface stays generalized:

- `ScalarPrimsDescriptor`
- `AnalyticPrimsDescriptor`

CPU lowers them through plan-time specialization:

- the planner resolves the descriptor to a CPU-specific plan variant
- the executor runs specialized loops or `strided-rs` kernels without
  per-element dynamic dispatch

This preserves performance while keeping the protocol surface clean.

Reductions should also resolve accumulation policy and identity values during
planning, rather than inside hot loops.

## GPU Strategy for Phase 1

Phase 1 is intentionally not a custom-CUDA PR.

GPU behavior is:

- keep the same public descriptors and family traits
- implement capability queries honestly
- advertise only the subset backed by current cuTENSOR support
- reject unsupported descriptors through planning, not through CPU-only API
  guards

This means GPU coverage will remain narrow, but the protocol and higher-level
APIs will already be correct.

### Why custom CUDA kernels are deferred

General pointwise and reduction CUDA support is substantially harder than the
CPU half:

- pointwise needs rank-generic broadcasting and strided indexing
- reductions need block reductions, accumulation policy, and often multi-pass
  logic
- tenferro currently has no NVRTC/PTX/module-cache execution layer in
  `tenferro-prims`

The sibling repository `tropical-gemm` is useful as a reference for:

- NVRTC compilation
- PTX loading
- thin Rust-side kernel launch abstractions

But it is still a GEMM-specific design, not a reusable pointwise/reduction
substrate.

## AD Strategy

The AD rule math must remain backend-independent.

That means:

- scalar math rules live in `chainrules-scalarops`
- dyadtensor only wires tensors, tangents, and cotangents to those rules and to
  the prim family execution substrate
- backend choice only affects primal/tangent/cotangent execution, not the
  formulas

Phase 1 should add generic dyadtensor surfaces for:

- unary pointwise maps
- binary pointwise maps
- scalar reductions

The PR does not need to close all scalar HVP oracle rows, but it must not
create any structural obstacle to doing so later.

## Testing Strategy

### `tenferro-prims`

- add deterministic CPU primal coverage for each new family group
- test capability queries explicitly
- keep CUDA coverage focused on surface truthfulness and current fast-path
  smoke tests

### `chainrules-scalarops`

- add rule tests for the new unary/binary families
- prefer small hand-checked values and centralized tolerance helpers

### `tenferro-dyadtensor`

- add primal, VJP, and JVP tests for generic unary/binary/reduction wrappers
- test the backend-generic runtime path rather than CPU-only entry points

### Workspace verification

The full PR gate remains:

- `cargo fmt --all --check`
- `cargo test --workspace --release`
- `cargo llvm-cov --workspace --release --json --output-path coverage.json`
- `python3 scripts/check-coverage.py coverage.json`
- `cargo doc --workspace --no-deps`
- `python3 scripts/check-docs-site.py`

Performance benchmarking is not part of the phase-1 exit criteria, but the PR
must not disturb `einsum` lowering or its existing checks.

## Documentation Updates

The PR should update:

- `docs/design/tensor-prims.md`
- `docs/design/architecture.md`
- `docs/design/reference/pytorch-dense-cpu-parity.md`

The docs must clearly say:

- what phase 1 implemented
- what remains GPU-only debt
- why custom CUDA kernels are deferred

## Risks

### 1. Reintroducing CPU-specific public contracts

This is the main failure mode. It is easier to write CPU-backed eager helpers
than to keep capability-driven APIs, so reviews should explicitly reject new
`CpuContext`-shaped public surfaces.

### 2. Overscoping reductions

`Var` and `Std` are qualitatively harder than `Sum`, `Prod`, `Mean`, `Max`,
and `Min`. The phase-1 PR should not hold itself hostage to implementing them
if the descriptor and capability model are already in place.

### 3. Hidden performance regressions from generic descriptors

The descriptor layer must specialize during planning, not during execution.
Otherwise scalar broadening would quietly slow CPU hot paths.

## Exit Criteria

Phase 1 is complete when all of the following are true:

- CPU executes the phase-1 scalar and analytic set through the family traits
- dyadtensor has generic unary/binary/reduction AD entry points for the same
  family surface
- AD formulas remain backend-independent
- higher layers rely on capability queries rather than CPU-only types
- workspace verification passes
- docs reflect the new phase-1 substrate and remaining GPU debt
