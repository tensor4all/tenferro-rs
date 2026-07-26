# Post-U8 stabilization worklog

Date: 2026-07-26

This worklog records the single-PR post-U8 stabilization sequence tracked by
#1477. The branch keeps cache cleanup, runtime dispatch cleanup, AD cleanup,
performance triage, and crate-split evidence in one coherent PR rather than
splitting the follow-up work into unrelated pull requests.

## Completed checkpoints so far

- #1476 docs/doc-pin cleanup: removed stale Phase 3/Phase 5 references that
  could send later agents toward the retired `GraphExecutor` or pre-semantic-AD
  contracts.
- #1478 cache lifetime and ownership cleanup: added explicit retained-byte
  bounds to runtime extension caches, preserved cache clear/hit/miss/eviction
  statistics, and documented cache owner and limit surfaces.
- #1456 production dispatch checkpoint: prepared roots now retain a
  crate-private `ScheduledGraph`; `Runtime::run_compiled*` walks that schedule
  synchronously; same-storage semantic operations execute through their
  selected engine bridge. Transfer-requiring placements are rejected until
  #1471 supplies transfer execution.

## Current runtime boundary

- `CompiledGraph` remains backend-neutral.
- Runtime preparation owns the semantic-to-prepared binding, cache ownership,
  selected engine bindings, and schedule construction.
- The synchronous scheduled executor is deliberately same-storage only.
  `ScheduledTransfer`, event-domain bridging, async submit, and admission logic
  remain #1471 scope.
- The extension execution hook is still the public
  `PreparedOperation::execute` trait method. This is compatible with the
  current extension architecture; narrowing it to an internal operation object
  would be a separate public-boundary cleanup.
- `run_compiled_values` preserves the metadata-only terminal lazy value path
  for layout/view-like outputs before falling back to normal scheduled
  execution.

## Verification evidence

```text
python3 scripts/test-doc-consistency.py
python3 scripts/check-doc-snippets.py --root-dir . --check
cargo test -p tenferro-runtime per_operation_placement_can_mix_same_storage_core_and_extension_engines -- --nocapture
cargo test -p tenferro-runtime --test integration runtime_run_compiled_dispatches_same_storage_extension_on_selected_engine -- --nocapture
cargo test -p tenferro-runtime prepared_program_is_binding_free_and_shares_staged_root -- --nocapture
cargo test -p tenferro-runtime preparation -- --nocapture
cargo test -p tenferro-runtime --test integration runtime_execution -- --nocapture
cargo test -p tenferro-ad --test integration runtime_execution -- --nocapture
git diff --check
```

## Residual risks to carry forward

- The scheduled executor currently leases a backend state per instruction and
  uses the unsegmented execution path. This is correct for #1456; #1464 and
  #1473 own evidence-backed fusion/performance restoration.
- CPU buffer-pool and GEMM analysis caches already expose ownership and limit
  controls, but their event counters remain coarse. This is not a blocker for
  #1478 because the generic runtime extension caches now have bounded retained
  bytes and observable statistics.
- Cross-storage placement and transfer nodes must stay rejected until #1471.
