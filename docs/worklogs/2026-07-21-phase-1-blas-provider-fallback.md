# Phase 1 BLAS provider fallback repair

## Session summary

The Phase 1 provider seam made an initial `Unsupported` result from the
configured GEMM provider terminal. That changed the prior BLAS behavior for
valid contractions whose direct descriptors could not express an operand
layout or conjugation. This repair restores a bounded canonical-materialization
retry without selecting behavior by provider kind.

The successful `Executed` path remains a direct return. It introduces no
allocation, string construction, or downcast. The broader Phase 1 performance
record remains in
[`2026-07-20-phase-1-cpu-provider-seams.md`](./2026-07-20-phase-1-cpu-provider-seams.md).

## Decisions made

- Retry only `Unsupported(Layout(Lhs))`, `Unsupported(Layout(Rhs))`, and
  `Unsupported(Conjugation)` through canonical operand materialization.
  `Layout(Output)` and all other unsupported reasons remain terminal after the
  first provider call.
- Retry the same configured GEMM provider. The runtime seam does not inspect or
  special-case BLAS, faer, or any future provider implementation.
- Fuse conjugation into the canonical materialization pass, then clear the two
  conjugation flags on the retry while preserving `alpha` and `beta`.
- Treat every provider error as terminal and preserve the original typed error.
- Treat a second `Unsupported` result from the canonical retry as terminal,
  regardless of its reason. Canonical fallback never chains.
- Hold the canonical retry result until both temporary operands have been
  returned to the buffer pool. Success, repeated `Unsupported`, and typed error
  exits all follow the same reclaim path.
- Run strided layout and conjugation preprocessing through
  `CpuExecutionContext::with_native_parallelism`, never ambient Rayon.
  `Sequential`, an `Outer` child, and an `Inner` operation backed by external
  workers all install a sequential strided policy. An `Inner` operation whose
  selected executor advertises Rayon installs a policy capped by the validated
  operation thread budget.
- Advance every workspace `strided-rs` dependency and the local lock resolution
  to reviewed revision `6b0b4a46b7dd9a9ea1677a0d596c0b4adab1acbc`, which
  supplies scoped `ExecutionPolicy`. This library workspace intentionally
  ignores `Cargo.lock`, so the manifest revision is the committed pin.
- Normalize matrix strides for unit extents while constructing the generic
  provider descriptor. This preserves valid vector and singleton-dimension
  contractions for all providers.
- Resolve empty BLAS contractions before descriptor capability checks, retaining
  the established zero-size and `beta` scaling semantics.

## Test-first evidence

Routing tests were added for `Layout(Lhs)`, `Layout(Rhs)`, and `Conjugation`.
Before the production change, the focused tests failed with the configured GEMM
provider's first unsupported result. The final tests require exactly one
canonical retry, verify that caller output is unchanged before that retry, and
cover complex conjugation, retained `alpha`/`beta`, and cleared retry flags.

Terminal-contract tests cover `Layout(Output)` with one provider call, a typed
provider error with one call and the original error preserved, and a canonical
retry returning `Unsupported` with exactly two calls. Each fixture obeys the
provider contract by leaving output untouched, and each test verifies that
output remains unchanged.

Seeded-pool tests cover successful retry, repeated unsupported, and typed retry
failure. Each path restores the exact retained-buffer statistics after both F64
materializations are reclaimed; the typed-error case also preserves the
original error and caller output.

Deterministic native-policy tests enter from an unrelated ambient four-thread
pool. The selected two-thread managed executor observes exactly its own two
participants. Sequential, external-worker, and engine-outer child contexts
observe one participant, and error or panic unwinding restores the prior
strided policy. Direct and backend-session native operations use the same
selected executor and policy.

The layout provider has a direct complex test proving that conjugation is fused
into materialization. Existing tests continue to cover terminal required
general contraction, cache identity, owned and viewed tensors, accumulation,
and unsupported output non-mutation.

The build-artifact contract initially rejected all five updated `strided-rs`
dependencies because it still expected the old revision. After updating the
contract, all six tests pass. A temporary manifest mutation back to the old
revision failed all five dependency subtests and passed again after restoration,
proving that pin drift cannot silently pass.

## Verification performed

- Real OpenBLAS vector-dot reproducer: passed.
- Runtime-injected BLAS normal, left-singleton, and right-singleton GEMM tests:
  three passed.
- Focused canonical routing/reclaim tests: eight passed.
- Focused native-policy tests: nine passed.
- Complete `tenferro-cpu` faer unit suite: 459 passed.
- Complete `tenferro-cpu` faer plus real OpenBLAS unit suite: 468 passed.
- The historically aborting `cpu-faer,cpu-blas,provider-inject --lib` command
  now completes 268 tests without calling unregistered FFI. Provider-inject
  call-through remains owned by its serialized integration fixture, which
  registers every symbol before use.
- Build-artifact contracts: six passed.
- `tenferro-linalg` faer, real OpenBLAS, and combined unit/integration/doctest
  matrices passed: respectively 108/113/59, 106/114/59, and 110/114/59.
- CPU public documentation examples passed 180/180 for both faer and combined
  real OpenBLAS feature sets.
- Warm provider-dispatch and fixed-main allocation probes passed 2/2; the warm
  empty-install allocation probe passed 1/1.
- Strict all-target Clippy passed for CPU/linalg faer and combined real BLAS;
  provider-inject library Clippy also passed. Formatting and all documentation
  consistency/site checks passed.

The linalg-only `cpu-blas,provider-inject` integration fixture is not in the
repository's formal provider-inject CI matrix. Its standalone test binary has
an inherited environment limitation: both candidate and isolated base
`f5e0f8fd` fail to link because the injection shim does not provide
`ssyevd_`/`dsyevd_`. Supplying the machine's OpenBLAS/LAPACK link flags resolves
those unrelated symbols and all six intended registered injection tests pass.
This is baseline-identical feature-combination evidence, not a candidate
regression or a claim that the standalone linalg-inject matrix is supported.

## Remaining risk

Canonical fallback intentionally allocates pooled operand materializations when
the first provider rejects input layout or conjugation. It does not broaden
fallback to output layout, dtype, runtime availability, output policy, or
required-general failures. Provider count/placement capability classification
remains future Task 7 work.
