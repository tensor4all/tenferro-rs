# Phase 6 extension resolution and CPU einsum checkpoint

This worklog records the Phase 6 local checkpoint for extension lowering,
extension cache observability, and CPU einsum runtime evidence. It follows the
Phase 5 scheduled-graph boundary recorded in
[`2026-07-25-phase-5-common-scheduled-graph.md`](2026-07-25-phase-5-common-scheduled-graph.md).

## Session summary

Phase 6 was implemented locally on
`codex/execution-engine-phase9-restart`. Per maintainer direction, no PR is
created until Phase 8 and the AMD CPU/CUDA benchmark gate are complete.

The implemented slice is intentionally bounded:

- typed standard-op extension lowering outcome;
- XLA migration to that typed lowering outcome;
- shared cache event stats for extension caches;
- CPU einsum runtime changing-shape cache evidence through the registered
  `EinsumRuntime`.

## Context read

- Workspace and repository rules: `AGENTS.md`, `REPOSITORY_RULES.md`,
  workspace `CODING_RULES.md`, and shared tensor4all rules.
- Restart handoff:
  `/home/shinaoka/tensor4all/HANDOFF-2026-07-24-tenferro-phase9-restart.md`
  with SHA-256
  `9a3274751103e38182bd5a2b62fedec150f09b9bb5521527ffe7fdee1846baf5`.
- Design authority:
  `docs/design/execution-engine-provider-architecture.md`.
- Phase 5 checkpoint:
  `docs/worklogs/2026-07-25-phase-5-common-scheduled-graph.md`.

## Implementation decisions

1. `ExtensionStandardLowering` now represents standard-op lowering as
   `Lowered(outputs)` or `Unsupported`. The legacy `lower_to_standard_ops`
   compatibility hook remains, and its `Option` result is adapted into the
   typed outcome.
2. XLA lowering consumes `lower_to_standard_ops_typed`, so unsupported lowering
   is no longer encoded as the peer-lowerer protocol.
3. The shared `CacheStats` shape now includes cache events:
   `hits`, `misses`, `evictions`, and `clears`.
4. `ExtensionCacheStore::stats(selector)` scopes both retained entries and
   event counters to `All`, `Family`, or `Cache`. This avoids reporting a
   family-specific miss count polluted by another extension family.
5. CPU einsum changing-shape evidence uses the existing registered
   `EinsumRuntime` through `ExtensionExecutor<CpuBackend>`. Three distinct
   concrete matrix-chain shapes populate three native plan-cache entries; a
   repeated first shape records one cache hit and preserves numerical behavior.

## TDD evidence

The following RED checks were observed before the corresponding implementation:

```text
cargo test -p tenferro-internal-ops typed_lowering --lib
  -> compile failure: ExtensionStandardLowering/lower_to_standard_ops_typed absent

cargo test -p tenferro-xla extension_lowering_uses_typed_outcome_not_option_protocol --lib
  -> assertion failure before XLA matched ExtensionStandardLowering

cargo test -p tenferro-runtime store_stats_track_hits_misses_evictions_and_clears --lib
  -> compile/assertion failure before CacheStats event fields were recorded

cargo test -p tenferro-einsum runtime_einsum_changing_shapes_track_native_plan_cache_stats --lib
  -> assertion failure: misses stayed at 0 before runtime cache event tracking

cargo test -p tenferro-runtime store_stats_scope_events_to_selected_family_and_cache --lib
  -> assertion failure: selector stats included events from another family
```

The corresponding focused GREEN checks passed:

```text
cargo test -p tenferro-internal-ops typed_lowering --lib
cargo test -p tenferro-xla extension_lowering_uses_typed_outcome_not_option_protocol --lib
cargo test -p tenferro-runtime store_stats_track_hits_misses_evictions_and_clears --lib
cargo test -p tenferro-runtime store_stats_scope_events_to_selected_family_and_cache --lib
cargo test -p tenferro-runtime extension_cache::tests --lib
cargo test -p tenferro-tensor cache_stats_empty_reports_zeroes --lib
cargo test -p tenferro-einsum runtime_einsum_changing_shapes_track_native_plan_cache_stats --lib
python3 scripts/check-doc-snippets.py
python3 scripts/check-public-error-docs.py
python3 scripts/test-doc-consistency.py
python3 scripts/check-guide-dependency-snippets.py
cargo fmt --all --check
git diff --check
```

## Commit sequence

- `dc787788` — `feat(ops): add typed extension lowering outcome`
- `7f47a15a` — `refactor(xla): consume typed extension lowering outcomes`
- `3cf20b43` — `feat(runtime): track extension cache events`
- `9a93071e` — `fix(runtime): scope extension cache event stats`
- closeout docs commit — record the Phase 6 checkpoint and spec updates

## Open-decision ledger

These items are intentionally not implemented in this Phase 6 production slice:

- linalg #1377 lifecycle mapping;
- FFT migration inventory;
- sparse operation owner;
- permutation/data-movement provider ownership;
- compatibility bridge removal policy;
- GPU native einsum, owned by Phase 7 unless later narrowed;
- XLA/subgraph integration and executor-shaped portable artifact retirement,
  owned by Phase 8.

No production code should guess these decisions without a new accepted slice.

## Phase 7/8 handoff

Phase 7 starts from the typed extension-lowering contract and the selector-safe
extension cache stats. Phase 8 remains the owner of XLA/subgraph integration.
PR creation, CI babysitting, and merge are deferred until Phase 8 and the AMD
CPU/CUDA benchmark gate are complete.
