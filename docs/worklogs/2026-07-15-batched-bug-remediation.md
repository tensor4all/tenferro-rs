# Batched Bug Remediation Work Log

## Session Summary

This work log records the single-PR remediation of issues #1275, #1276, #1368,
#1375, #1381, and #1385. All six reports were reconciled against
`origin/main` at `50c6623d`; none had become stale or gained a narrowing comment
before implementation. The final changes are split into issue-scoped commits,
one downstream integration follow-up, two independent-review follow-ups, one
hardware-CI follow-up, and one local-gate metadata-ownership follow-up, so each
contract can be reviewed or reverted independently.

The batch deliberately treats the repository's API and numerical conventions
as authoritative. In particular, #1276 is a clean break to the owned/read/write
vocabulary rather than a compatibility shim, and #1275 removes the second CPU
execution model instead of introducing a process-global default backend.

## Context And Rules Read

- The repository and subdirectory `AGENTS.md` files, `REPOSITORY_RULES.md`,
  `CONTRIBUTING.md`, and the bug-fix/remediation contribution workflows.
- `docs/design/api-and-convention-freeze.md`, `docs/design/einsum.md`, the
  active API specifications and guides, and the CPU execution/pool code.
- The current issue bodies and comments for #1275, #1276, #1368, #1375,
  #1381, and #1385 immediately before implementation and again before final
  verification.
- CodeGraph neighborhoods for the affected linalg gauge, CubeCL numeric,
  typed einsum, CPU public operation, execution-resource, and GEMM paths.
- The approved design at
  `docs/superpowers/specs/2026-07-15-batched-bug-remediation-design.md`.

The repository root occupied about 44.8 GB at the start of work, below the
100 GB cleanup threshold. Implementation used an isolated worktree based on
the recorded `origin/main` commit. When work resumed after the first hosted GPU
run, the root occupied about 108 GB; cleanup of old build artifacts was proposed
but no user files or build outputs were removed during this remediation.

## Final Classification Ledger

| Issue | Final classification | Implemented result | Close condition |
| --- | --- | --- | --- |
| #1375 | Auto Fix | `763b1246` prepares one checked SVD gauge layout containing batch count, per-batch spans, and total storage spans. Gauge traversal validates storage before indexing and uses checked ranges for f32/f64/c32/c64. | Overflow, malformed-storage, batched, zero-batch, source-contract, and workspace checks pass. Ready to close when the PR merges. |
| #1368 | Auto Fix | `61f2cffa` routes CubeCL elementwise and unit/plane reductions through shared NaN-propagating max/min helpers. Independent review then found that fusion code generation still emitted native extrema directly; `889fbe46` applies the same contract to fused max/min with explicit `IsNan`/`Select` IR. The first RunPod run exposed an invalid generic NaN literal in plane reductions and an over-strict fused signed-zero assertion; `d91fa95e` propagates an actual NaN lane value through `plane_sum` and compares non-NaN fused values numerically. The implementation retains native non-NaN signed-zero and infinity behavior. | Source contracts and host checks pass; the corrected ignored CUDA parity tests must pass in the rerun of trusted RunPod CI before merge. |
| #1381 | Verify First, then Auto Fix | `61f2cffa` confines the bare CubeCL integer IR operators to named wrapping helpers with `INVARIANT` markers. Elementwise, broadcast multiply, negation, power/remainder internals, and unit/plane reductions call those helpers. | Source inventory and existing host checks pass; the added CUDA overflow tests must pass in trusted RunPod CI before merge. |
| #1276 | Auto Fix / intentional API correction | `3f57b1cd` adds `TypedTensorWrite`, splits typed view inputs into `TypedTensorReadEinsumExt` and `TypedTensorReadEinsumIntoExt`, uses `_read` method names, and accepts owned or mutable-view outputs through `Into<TypedTensorWrite>`. Active design, spec, and guide text is migrated. `cefa3313` adds runnable examples to the new public traits, methods, and write adapters after independent review. | Public-surface, owned/view/output, error, doctest, and workspace checks pass. Ready to close when the PR merges. |
| #1275 | Auto Fix / intentional API removal | `2b793176` removes production crate-root CPU operation reexports and per-call `with_local_pool` wrappers. Supported use goes through `CpuBackend` and backend traits; crate-private adapters remain only under `cfg(test)`. `bfa2db37` removes the downstream linalg test module's now-invalid, unused imports found by the release workspace gate. | Public-surface inventory, pool ownership, downstream compilation, docs, and workspace checks pass. Ready to close when the PR merges. |
| #1385 | Auto Fix | `028e2668` borrows the installed pool directly, caches the environment-derived default with `OnceLock`, and borrows internal Faer/BLAS/TBLIS GEMM descriptors instead of cloning them in backend/session dispatch. | Pool restoration/panic, provider-feature, structural, benchmark, and workspace checks pass. Ready to close when the PR merges. |

## Decisions And Alternatives

- **One PR, issue-scoped commits.** These reports share repository-contract and
  public-surface concerns, while their commits remain independently reviewable.
- **No compatibility aliases for #1276.** Retaining unsuffixed typed-view
  methods would preserve the inconsistency the canonical API design forbids.
- **No process-global CPU backend for #1275.** A hidden singleton would make
  threading, placement, provider selection, and resource ownership implicit.
  Requiring an explicit `CpuBackend` preserves one execution model.
- **Named CubeCL wrapping helpers for #1381.** The current CubeCL generic
  integer layer does not expose Rust-style `wrapping_*` intrinsics. Bare IR
  expressions therefore remain only inside small helpers whose names,
  invariant comments, source contracts, and CUDA boundary tests state the
  verified two's-complement lowering contract.
- **Direct pool borrowing for #1385.** A scratch/default pool or resettable
  global cache would retain avoidable construction. A mutable loan preserves
  the existing in-flight panic accounting without unsafe aliasing.
- **One-shot environment parsing.** `BufferPool::new()` uses the process
  default cached by `OnceLock`; explicit limits and runtime setters remain
  dynamic. Parser tests do not mutate global process state.
- **Acquire only borrowed metadata keys.** Child graph analysis may read
  metadata owned by a parent or overlapping scope. Acquiring those specific
  keys into the child scope keeps their reference counts balanced without
  cloning the global metadata map or serializing otherwise independent tests.

## Regression Boundaries

- #1375 adds executable overflow tests without attempting impossible
  near-`usize::MAX` allocations and a source contract forbidding raw gauge
  batch products/offsets.
- #1368/#1381 add CUDA value tests for both float precisions, fused extrema,
  and integer overflow boundaries, plus a host-runnable source inventory
  covering every affected kernel family. The fused tests compare NaN in both
  operand orders, infinity, and ordinary values with CPU results. They exercise
  signed zero without requiring CPU/GPU sign-bit parity because #1368 defines
  NaN propagation while native CubeCL extrema retain their non-NaN zero-sign
  behavior.
- #1276 adds public-surface tests, typed owned/read execution, owned output,
  strided mutable-view output, and `TypedTensorWrite` conversion coverage.
- #1275 prevents production free-function reexports and `with_local_pool`
  helpers from returning, while existing backend/pool tests cover reuse.
- #1385 tests parser edge cases, direct pool ownership, panic replenishment,
  descriptor borrowing, Faer behavior, and BLAS/TBLIS feature compilation.
- The final workspace gate also fixes a pre-existing metadata-scope ownership
  race exposed by the larger batch: a deterministic parent/child scope test
  proves that borrowed metadata survives the parent scope and is released when
  the final child scope drops.

## #1385 Benchmark Evidence

The `grouped_gemm` Criterion target now has a steady-state case matching the
reported profile: one-thread Faer, 2,000 explicit warm-up iterations, six
grouped calls per measured iteration, eight 4x4x4 jobs per call, persistent
backend/cache/output, and reusable `TensorView` descriptors.

The benchmark-only change was applied to a detached worktree at parent commit
`2b793176` and built in a separate target directory so Cargo could not reuse
the fixed library artifact. Both sides used:

```bash
cargo bench -p tenferro-cpu --bench grouped_gemm -- \
  'grouped_gemm/steady_state/six_calls_8x4x4' \
  --sample-size 20 --measurement-time 2 --warm-up-time 1 --noplot
```

| Revision | Distribution | Median estimate |
| --- | --- | --- |
| Before, `2b793176` plus benchmark only | 3.1435–3.1604 µs | 3.1524 µs |
| After, `028e2668` | 2.0563–2.0754 µs | 2.0673 µs |

That is a 34.4% reduction in the measured steady-state dispatch time on this
Apple arm64 host. The percentage is evidence for this workload, not a portable
performance threshold; the durable acceptance conditions are the structural
absence of placeholder pool construction, repeated environment lookup, and
internal descriptor clones.

## Verification Performed

Targeted verification completed during implementation:

- `cargo test -p tenferro-cpu`: 284 unit tests, 9 capability-contract tests,
  5 provider-contract tests, 25 runtime-error tests, and 105 doctests passed.
- `cargo check -p tenferro-cpu --no-default-features --features cpu-blas
  --lib`: passed.
- `cargo check -p tenferro-cpu --features cpu-tblis --lib`: passed.
- `cargo clippy -p tenferro-cpu --all-targets -- -D warnings`: passed.
- The #1385 Criterion before/after run above completed successfully.
- Issue-focused linalg, GPU source-contract, tensor, einsum, CPU, public
  surface, doctest, and clippy checks passed before their commits.
- Independent review found the missing fused #1368 path. Its regression was
  first observed as a failing source-contract test, then fixed through shared
  `IsNan`/`Select` code generation. The final GPU source-contract suite passed
  7/7, and `cargo check -p tenferro-gpu --features cuda --tests` passed.
- The independent-review documentation follow-up passed 73 einsum doctests
  and 275 tensor doctests as part of the final release workspace gate.
- The first trusted RunPod CUDA run executed 842 tests: 839 passed and 3
  failed. Both fused f32/f64 failures had already passed all NaN cases and
  failed only because the test compared `+0.0` and `-0.0` by bits. The plane
  reduction failure was an NVRTC compile error because `F::new(f32::NAN)`
  lowered to the undefined CUDA identifier `NaN`. The integer-overflow and
  elementwise NaN tests passed in that run.
- The CUDA-codegen regression contract was observed RED before `d91fa95e`
  because `plane_propagate_nan` was absent, then GREEN after the fix. The full
  `tenferro-gpu` release suite, `cargo check -p tenferro-gpu --features cuda
  --tests`, formatting, and diff checks passed locally. A CUDA-enabled clippy
  probe remains unsuitable as an extra gate because the feature combination
  exposes pre-existing CubeCL macro and FFI lints outside this diff; the
  repository's required workspace clippy profile remains the authoritative
  clippy check.
- The first full workspace rerun after `d91fa95e` exposed an intermittent
  `qr_sum_grad_optimized_graph_is_structurally_compact` failure: a derived pad
  tensor had lost its metadata while parallel structural tests were dropping
  overlapping scopes. The exact test passed 20/20 alone, the parallel test
  binary reproduced the failure, and `--test-threads=1` passed. A deterministic
  parent/child scope regression was then observed RED before `bbf9b7e8` and
  GREEN after graph analysis began acquiring only the global metadata keys it
  actually borrowed. The complete runtime release suite (232 unit tests and
  224 doctests), the original 27-test parallel linalg suite, and 20 consecutive
  repetitions of that suite passed after the fix.

Repository-wide verification on the complete code batch also passed:

- `cargo fmt --all --check` and `git diff --check`: passed.
- `cargo test --workspace --release`: passed. Its first run exposed unused
  imports of the removed #1275 helpers in the linalg test module; `bfa2db37`
  removed those imports, and the complete release workspace command then
  passed on the corrected tree. The post-RunPod rerun later exposed the
  metadata ownership race described above; after `bbf9b7e8`, the complete
  release workspace command passed again.
- `cargo llvm-cov --workspace --release --json --output-path coverage.json`
  and `python3 scripts/check-coverage.py coverage.json`: passed, with 159/159
  measured files meeting their thresholds and 3 files excluded by policy.
- `cargo doc --workspace --no-deps`: passed.
- `python3.11 scripts/check-docs-site.py`: passed, including 4 dependency
  guide smoke tests and all 13 workspace library crates. The macOS system
  `python3` is 3.9.6, so the installed Python 3.11 interpreter was used for
  the script's documented TOML parser requirement.
- `cargo clippy --workspace --all-targets -- -D warnings`: passed.
- `cargo clippy --manifest-path ext/tropical/Cargo.toml --all-targets -- -D
  warnings`: passed.
- `python3 scripts/repository-rules-review.py --base origin/main --head HEAD
  --output-json /tmp/repository-rules-review-final-code.json`: verdict `pass`,
  with no findings on the final code batch. The same review is run again on
  the final work-log commit before PR creation.

The corrected trusted RunPod CUDA value tests and the repeated repository-wide
checks are the remaining pre-merge gates rather than a local CUDA claim.

## GPU Environment

The implementation host is Apple arm64/macOS and has no NVIDIA runtime, so no
local CUDA value test is claimed. The new CUDA tests remain `#[ignore]` in the
repository's established hardware-test suite. The host-runnable source
contracts prove that all affected entry points route through the new helpers.
The first trusted RunPod run confirmed #1381 and the elementwise #1368 path,
then exposed the two narrower failures recorded above; the corrected plane and
fusion paths still require the rerun before merge.

## Residual Risks

- #1275 and #1276 intentionally remove or rename public APIs. Workspace code
  and active docs are migrated, but downstream users must move to explicit
  `CpuBackend` ownership and typed `_read` traits/output adapters.
- CubeCL lowering is guarded by source contracts and CUDA tests, but future
  CubeCL/CUDA changes could alter NaN, signed-zero, or integer-overflow
  behavior; retain the hardware regressions.
- The `OnceLock` pool default intentionally observes the environment only on
  its first use in a process. Callers requiring later changes must use the
  explicit constructor or runtime limit setter.
- The benchmark isolates the reported small grouped-GEMM workload. It does not
  imply the same percentage for larger GEMMs, other providers, or other hosts.
- Scoped metadata references now add one refcount operation per borrowed
  global key. The regression proves balanced final release, and the targeted
  acquisition avoids a full-map clone or a global serialization workaround.
