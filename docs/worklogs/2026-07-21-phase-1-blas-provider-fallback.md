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
- Run strided layout and conjugation preprocessing under an explicit execution
  policy rather than ambient Rayon. `Sequential` remains on the calling thread;
  `Inner` installs into the selected `CpuContext` and applies a Rayon policy
  capped by its validated thread budget. The policy is independent of provider
  kind, and preprocessing finishes before the GEMM provider is called.
- Store `CpuContext`'s validated budget as `NonZeroUsize` and expose it only
  through a crate-private provider accessor when constructing the strided
  execution policy. This removes the production `expect()` without adding a
  fallback or another panic boundary for an invariant already enforced by every
  context constructor.
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
Before the production change, the first test failed with the configured GEMM
provider's `Layout(Lhs)` result. The final tests require exactly one canonical
retry, verify that caller output is unchanged before that retry, and cover
complex conjugation, retained `alpha`/`beta`, and normalized retry flags.
A follow-up specification test reproduced `Layout(Output)` incorrectly making
two calls before the retry predicate was narrowed; it now makes one terminal
call.

Terminal-contract tests cover `Layout(Output)` with one provider call, a typed
provider error with one call and the original error preserved, and a canonical
retry returning layout, conjugation, or dtype `Unsupported` with exactly two
calls. Each fixture obeys the provider contract by leaving output untouched,
and each test verifies that output remains unchanged.

A seeded-pool test reproduced typed retry failure dropping both F64
materializations: retained stats changed from two buffers / 64 bytes to zero.
The repaired path preserves the exact seed stats and original error while
leaving output unchanged. The successful and repeated-unsupported paths use the
same seeded assertion.

Deterministic tracking-operation tests run preprocessing inside a four-thread
ambient pool. `Sequential` observes one participant on the calling thread.
`Inner` observes exactly two participants, all owned by the selected two-thread
`CpuContext`, both when entered from the unrelated ambient pool and when already
nested in the selected pool.

The layout provider also has a direct complex test proving that conjugation is
fused into materialization. Existing tests continue to cover terminal required
general contraction, cache identity, owned and viewed tensors, accumulation,
and unsupported output non-mutation.

The build-artifact contract initially rejected all five updated `strided-rs`
dependencies because it still expected the old revision. After updating the
contract, all six tests pass. A temporary-manifest mutation back to the old
revision makes the five dependency subtests fail, confirming that the contract
detects pin drift.

## Verification performed

- Real OpenBLAS vector-dot reproducer: passed.
- Runtime-injected BLAS normal, left-singleton, and right-singleton GEMM tests:
  three passed.
- `workspace-blas`, excluding the independently pre-existing fixed-baseline
  allocation gate: 2,416 passed, one skipped.
- `workspace-blas` doctests: passed.
- Default/faer workspace tests: 2,416 passed.
- Default/faer workspace doctests: passed.
- Focused provider/policy tests: 14 passed; canonical routing/reclaim tests:
  eight passed; `CpuContext` construction/execution tests: 14 passed.
- Build-artifact contracts: six passed; the old-revision mutation failed all
  five strided dependency checks as intended.
- `tenferro-cpu` public documentation examples: 170 passed.
- Formatting, documentation consistency, staged/unstaged diff checks, and
  workspace/all-target Clippy: passed. Clippy reported only the 11 warnings
  already present at the starting revision.

The excluded allocation gate reports exactly the same values at source revision
`f5e0f8fd` and with this repair: elementwise `1202 / 57440`, reduction
`5004 / 111072`, slice `602 / 39840`, and dot `1000 / 37600` (allocations /
bytes). It therefore predates and is independent of this routing change; its
fixed constants were not modified.

## Remaining risk

Canonical fallback intentionally allocates pooled operand materializations when
the first provider rejects input layout or conjugation. It does not broaden
fallback to output layout, dtype, runtime availability, output-policy, or
required-general failures.
The repository-wide `clippy -D warnings` gate also has existing warnings at the
starting revision; this change does not attempt to repair that unrelated debt.
