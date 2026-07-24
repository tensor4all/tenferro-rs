# Phase 4 runtime preparation substrate

This worklog records the Phase 4 implementation of immutable runtime snapshots,
runtime preparation metadata, bounded prepared-plan caching, CPU registration,
and the eager runtime snapshot bridge. It closes the bounded design in
[`2026-07-24-phase-4-runtime-preparation-design.md`](../superpowers/specs/2026-07-24-phase-4-runtime-preparation-design.md)
without starting the Phase 5 execution migration or the Phase 6 operation-family
derived-cache work.

## Implemented boundary

- `tenferro-runtime` now owns runtime identities, immutable configuration
  snapshots, epochs, transactional engine/extension-module reconfiguration,
  direct core capability slots, runtime cache owners, input signatures,
  finite specialization requirements/projections, and normalized
  `PrepareOptions`.
- `Runtime::prepare_for` prepares a crate-private `PreparedProgram` from a
  frozen semantic program, concrete input signature, and current runtime
  configuration. It remains binding-free and routes through the single private
  `lower_semantic_to_exec_staging` adapter until Phase 5 deletes that adapter.
- Prepared-plan caching is runtime-owned, bounded by entry and retained-byte
  limits, single-flight for same-key preparation, negative-caches deterministic
  failures within one epoch, and rejects recursive distinct-key preparation
  before capacity waits.
- `tenferro-cpu` implements the approved CPU-to-runtime dependency edge only
  for the CPU direct core preparation adapter. The adapter is metadata-only and
  attaches the CPU backend as a runtime cache owner. `tenferro-runtime` has no
  production dependency on `tenferro-cpu`.
- `EagerRuntime` constructs one private `Runtime` for CPU-backed eager contexts.
  `CpuPlacementBoundEager` preserves its Phase 2 CPU coordinator/provider
  snapshot while refreshing private runtime registration metadata by epoch
  comparison.

Phase 4 deliberately leaves public prepared execution, extension-family runtime
execution migration, XLA/subgraph compilation, and operation-family derived
caches to later phases.

## Commit sequence

- `5176ed27` — `feat(runtime): add A0 runtime identities`
- `1f2453a1` — `feat(runtime): add A0 policy and normalized keys`
- `602da8b8` — `feat(runtime): add A0 signatures and specialization`
- `244ad108` — `feat(runtime): define immutable preparation capabilities`
- `6540a65b` — `feat(runtime): add immutable snapshots and reconfiguration`
- `271fabec` — `feat(runtime): add transactional extension modules`
- `cdf8647e` — `feat(runtime): add bounded prepared plan cache`
- `f5222ee2` — `refactor(runtime): centralize semantic staging ownership`
- `c5f5199e` — `feat(runtime): integrate prepared program preparation`
- `e6b4ffb4` — `feat(cpu): adapt providers into runtime preparation`
- `1c9da42d` — `feat(ad): bridge eager runtime to immutable snapshots`

## Verification evidence

The final D1 checkpoint was verified with:

```text
cargo test -p tenferro-ad eager::tests::runtime_snapshot --lib
cargo test -p tenferro-ad eager::tests::placement_bound --lib
cargo test -p tenferro-ad --test integration placement_bound_eager
cargo test -p tenferro-ad --test integration runtime_snapshot_bridge
cargo test -p tenferro-ad
cargo clippy -p tenferro-ad --all-targets -- -D warnings
cargo fmt --all --check
git diff --check
```

Observed final local `tenferro-ad` package counts:

- unit tests: 71 passed;
- integration tests: 328 passed;
- doctests: 134 passed.

Repository-rules review for the D1 worktree and D1 commit could not use the
external LLM reviewer because it returned HTTP 400. The deterministic dry-run
review passed with no findings:

- `/tmp/repository-rules-review-p4-d1-worktree-dry-run.json`;
- `/tmp/repository-rules-review-p4-d1-commit-dry-run.json`.

Earlier Phase 4 nodes were committed with their focused runtime/CPU tests,
formatting, clippy, and deterministic repository-rules dry-run evidence. The
remaining closeout commit updates documentation and doc-consistency checks so
the public docs no longer describe Phase 4 as merely accepted-but-unimplemented.

## Later-phase ownership

- Phase 5 owns deleting `lower_semantic_to_exec_staging` and moving execution
  onto prepared runtime artifacts.
- Phase 6 owns concrete operation-family derived caches and their
  retained-byte accounting.
- Phase 8 owns XLA/subgraph integration through `SubgraphCompiler`.
