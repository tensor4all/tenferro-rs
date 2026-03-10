# Prims and Linalg Protocol Redesign Design

## Goal

Redesign the execution protocol stack around a minimal semiring core, explicit
performance fast paths, and a separate linalg kernel substrate, while keeping
`einsum` performance at least at the current level and making the overall
architecture easier to understand from `docs/design/`.

## Context

The current protocol stack mixes several concerns inside
`tenferro-prims::TensorPrims`:

- semiring-generic execution needed by `einsum`
- Standard-arithmetic pointwise and reduction vocabulary
- linalg-adjacent needs such as conjugation support
- a single `Extension` enum that mixes semantic scope with optimization scope

At the same time, `tenferro-linalg` still carries backend-facing assumptions
that should live below the public/composite linalg layer, and the current
design docs split that story across several partially overlapping documents.

This redesign is constrained by four hard requirements:

1. preserve `einsum` performance, especially the current CPU lowering
2. keep the minimal substrate sufficient for `einsum` and tropical algebra + AD
3. make `tenferro-linalg` execute via prim protocols rather than backend-local
   direct code
4. land as one integrated branch/PR without backward-compatibility shims

## Design Principles

### Minimal Core Principle

An operation belongs in the semiring core only if it is needed for:

- `einsum` primal lowering, or
- tropical algebra + AD minimal support

Being definable for a semiring is not sufficient by itself.

### Prims-First Execution

`tenferro-linalg` is a public/composite lowering layer, not a kernel provider.
All execution must go through `tenferro-prims` or the new
`tenferro-linalg-prims` crate.

### No Hot-Path Dynamic Dispatch

The public protocol may use generalized descriptors, but backend `plan()`
implementations must lower them immediately into specialized internal plans.
The hot execution path must keep the current dedicated loops / BLAS / cuTENSOR /
provider-native kernels.

### No Compatibility Layer

The refactor is a one-shot replacement:

- no deprecated parallel surface
- no temporary old/new adapter layer
- no long-lived compatibility shims

## Final Layering

### TensorStructural

Structural, zero-copy tensor transforms remain in `tenferro-tensor` rather than
the prim layer:

- `permute`
- `reshape`
- `broadcast`
- `diagonal`
- `narrow` / `select`

This is the main reason `Permute` should be removed from the prim protocol.
`Tensor::permute()` is already a metadata-only view, while the current prim
`Permute` is an eager materialization path.

### TensorSemiringCore

`TensorSemiringCore<Alg: Semiring>` is the minimum execution contract that
`tenferro-einsum` may depend on.

Required operations:

- `BatchedGemm`
- `ReduceAdd`
- `Trace`
- `AntiTrace`
- `AntiDiag`
- `MakeContiguous`

These operations are enough to support:

- current `einsum` lowering
- tropical algebra contraction
- the diagonal/scatter operations needed by AD

`ReduceMul` is semiring-valid but intentionally excluded from the required core
because it is not part of the `einsum + tropical + AD` minimum.

### TensorSemiringFastPath

`TensorSemiringFastPath<Alg: Semiring>` contains optional performance paths that
are semiring-valid but not required for correctness:

- `Contract`
- `ElementwiseBinary { Add, Mul }`

These are optional because they can be expressed in terms of the core
operations, but optimized backends should be able to provide them directly.

The name `FastPath` is deliberate: this layer is about optimization rather than
semantic necessity.

### TensorScalarPrims

`TensorScalarPrims<Alg>` contains Standard-like pointwise and scalar reduction
operations that are not semiring-generic:

- pointwise unary
- pointwise binary
- pointwise ternary
- scalar reductions

This layer owns things like `Div`, `Abs`, `Max`, `Min`, `Mean`, `Real`,
`Imag`, and similar arithmetic vocabulary.

### TensorAnalyticPrims

`TensorAnalyticPrims<Alg>` contains transcendental and analytic operations:

- `Exp`, `Log`, `Log1p`, `Expm1`
- `Sin`, `Cos`, `Tan`, `Tanh`
- inverse trig / hyperbolic variants
- `Pow`, `Atan2`, `Hypot`, `Xlogy`
- reduction families such as `Var`, `Std`

This keeps the scalar arithmetic substrate separate from the larger analytic
surface described in issue `#441`.

### TensorLinalgPrims

Linalg kernel contracts move to a new crate, `tenferro-linalg-prims`, so
`tenferro-prims` stays focused on semiring/scalar substrate.

`tenferro-linalg-prims` contains only backend-facing kernel contracts, not the
entire `torch.linalg`-shaped public API.

## `tenferro-linalg-prims` Scope

### Kernel Basis, Not Public API Mirror

`tenferro-linalg-prims` should contain only operations that naturally map to
provider-native factorization / solve kernels or require kernel-level metadata
such as pivots, workspaces, and `info/status` values.

Recommended minimum kernel basis:

- `CholeskyFactor`
- `LuFactor`
- `LuSolve`
- `TriangularSolve`
- `QrFactor`
- `HouseholderProduct`
- `SvdFactor`
- `EigenGeneral`
- `EigenHermitian`
- `LeastSquares`

These are execution contracts. They are the kernel substrate used to implement
public linalg APIs.

### Not Linalg Prims

The following stay in `tenferro-linalg` as composites or public API utilities:

- `matrix_power`
- `cond`
- `tensorinv`
- `tensorsolve`
- `multi_dot`
- `vecdot`
- `vander`
- `solve` / `solve_ex`
- `inv` / `inv_ex`
- `lu`
- `qr`

For example:

- `matrix_power` lowers to repeated `matmul` plus `inv` when needed
- `cond` lowers to `norm`, `inv`, and/or `svdvals`
- `tensorinv` lowers to structural view transforms plus `inv`

So these are linalg APIs, but they are not kernel substrate.

## `tenferro-linalg` Contract

After the redesign, `tenferro-linalg` is defined as:

- public API layer
- shape/axis/options validation layer
- composite lowering layer
- result-struct assembly layer

It is explicitly not:

- a backend implementation crate
- a direct CPU slice kernel crate
- a place for backend name checks such as `CpuBackend` / `CudaBackend`

Every execution path in `tenferro-linalg` must go through:

- `tenferro-prims`, or
- `tenferro-linalg-prims`

This is the core meaning of "prims-first" in this redesign.

## Execution Contract

All execution families keep the current BLAS/cuTENSOR-style contract:

`output <- alpha * op(inputs) + beta * output`

This applies to:

- semiring core
- semiring fast path
- scalar prims
- analytic prims
- linalg prims

Keeping this contract avoids extra temporaries and preserves current optimized
kernel usage for GEMM, contraction, and scatter-add style operations.

## Capability Model

Capability queries should be split by family instead of reusing one global
`Extension` enum.

Examples:

- `has_semiring_fast_path(op)`
- `has_scalar_unary(op)`
- `has_scalar_binary(op)`
- `has_scalar_reduction(op)`
- `has_analytic_unary(op)`
- `has_linalg_kernel(op)`

This avoids mixing unrelated vocabularies and lets `tenferro-linalg` remain
backend-generic: it should only look at trait bounds and capability queries,
never backend names.

## Performance Constraints

### Einsum

`einsum` must preserve the current lowering strategy:

- CPU: `permute view -> MakeContiguous -> BatchedGemm`
- GPU: prefer `Contract` when the fast path is available

`Permute` should not remain part of the prim protocol just to support this,
because the tensor-layer view already exists and the eager materialization
should be represented by `MakeContiguous`.

### Elementwise and Scalar Paths

Generalized public descriptors must still specialize in `plan()` to preserve
the current fast paths. For example, a public
`ElementwiseBinary { Add, Mul }` descriptor may lower to:

- `CpuPlan::ElementwiseAdd`
- `CpuPlan::ElementwiseMul`
- provider-native GPU binary kernels

No per-element dynamic dispatch should remain in the hot path.

### Benchmark Gates

Performance validation should have two layers:

1. `einsum` external regression gate using sibling repository
   `../tenferro-einsum-benchmark`
2. in-tree microbenchmarks for scalar/elementwise/linalg-kernel-sensitive paths

`../tenferro-einsum-benchmark` is appropriate because it directly benchmarks
`tenferro-einsum` over the standardized einsum benchmark corpus and already
serves as a realistic external performance harness.

## Documentation Reorganization

This redesign should also clean up `docs/design/` so the final architecture is
readable as one coherent story.

Required documentation work:

- update `docs/design/architecture.md` to show the new crate/layer split
- rewrite `docs/design/tensor-prims.md` around the new family-based protocol
- add `docs/design/linalg-prims.md` for the new crate
- rewrite `docs/design/linalg.md` so it describes `tenferro-linalg` as a
  public/composite lowering layer
- update `docs/design/index.md` to present the new document map
- update `docs/design/testing.md` to include benchmark/verification expectations

The older design notes:

- `docs/design/linalg-backend-api.md`
- `docs/design/linalg-gemm-prims.md`

should either be absorbed into the new canonical documents or reduced to clear
"superseded by" notes. The goal is that a new reader can understand the final
stack from `index.md` and the canonical docs without reconstructing the design
history manually.

## Integrated Migration Strategy

The work lands in one integrated branch/PR with no compatibility layer.

Recommended internal order:

1. add new crate/protocol surfaces
2. port backend plan/execute implementations
3. migrate `tenferro-einsum`
4. restore tropical + AD coverage on the new substrate
5. add scalar / analytic families
6. add `tenferro-linalg-prims`
7. migrate `tenferro-linalg` lowering
8. migrate dyadtensor and oracle replay
9. remove the old protocol surface
10. finalize docs/design organization and verification gates

This is externally a big-bang change, but internally it preserves a sequence
that protects performance-sensitive paths first.

## Verification Strategy

### Correctness

Required verification includes:

- existing `tenferro-einsum` tests
- tropical algebra minimum AD coverage
- `tenferro-linalg` tests and oracle replay
- `extension/tenferro-dyadtensor` primal/AD coverage
- workspace formatting, docs, and coverage gates

### Performance

Required performance checks include:

- representative `tenferro-einsum` internal tests/benches
- sibling `../tenferro-einsum-benchmark` runs for before/after comparison
- in-tree microbenchmarks for elementwise/scalar and linalg-kernel-sensitive
  paths

Performance regressions in the current `einsum` lowering are release blockers.

## Risks

### Scope

This is a wide refactor touching crate boundaries, docs, protocol traits, and
AD/linalg lowering simultaneously.

### Benchmark Drift

The sibling benchmark repo is useful but not CI-owned by this workspace. It is
an excellent local release gate, but the in-tree verification story still needs
its own benchmark coverage for scalar/elementwise and linalg hot paths.

### Documentation Divergence

If the old design docs remain half-updated, the architectural cleanup fails
even if the code is correct. Documentation reorganization is part of the
deliverable, not a follow-up polish task.

## Success Criteria

The redesign is complete when:

1. the old monolithic prim protocol is replaced by the new family-based stack
2. `tenferro-linalg-prims` exists and owns the linalg kernel substrate
3. `tenferro-linalg` executes only through prim protocols
4. `einsum` preserves the current lowering and does not regress on benchmark
   gates
5. tropical + AD still work on the reduced semiring core
6. `docs/design/` presents the final architecture clearly and canonically
