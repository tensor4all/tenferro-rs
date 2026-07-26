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
  selected engine bridge.
- #1471 U3 substrate checkpoint: `Runtime::submit` returns an
  `ExecutionHandle`, engine snapshots expose runtime-allocated event domains,
  transfer providers are registered by storage-class pair, and cross-storage
  linear execution calls the transfer provider from the production scheduled
  loop before dispatching the downstream operation.
- #1468 linalg semantic AD cleanup: semantic linalg `linearize` now emits JVP
  fragments directly into `SemanticProgramBuilder` instead of first recording
  and replaying a legacy fragment. The remaining linalg recorded-fragment usage
  is limited to reverse-mode construction paths that need a local linear
  fragment to transpose.
- #1464 CPU elementwise fusion classifier: replaced the hand-written
  add/multiply pattern arms with an explicit two-input, two-operation,
  one-output, identity-view binary-tail classifier. The classifier recognizes
  the benchmark-motivated `Add`/`Multiply` family only, including reversed
  commutative input order for the first op, and still rejects broadcast views,
  repeated inputs, multi-output plans, unsupported ops, and longer chains.
- #1473 tactical performance triage: rechecked the remaining #1426 findings
  against the post-U8 code, closed #1473 with one-line dispositions, and split
  the true positives into focused follow-ups: #1479 strided `dot_general`
  accumulation fallback, #1480 host identity copies, #1481 multi-axis CPU
  reductions, #1482 CPU indexing hot loops, #1483 eager AD accumulation
  allocation, #1484 FFT plan/scratch reuse, and #1485 cuTENSOR
  descriptor/plan/workspace caching.

## Current runtime boundary

- `CompiledGraph` remains backend-neutral.
- Runtime preparation owns the semantic-to-prepared binding, cache ownership,
  selected engine bindings, selected operation placements, and schedule
  construction.
- The synchronous scheduled executor tracks each slot's current storage class.
  Cross-storage handoff is supported only when a registered transfer provider
  can materialize the slot for the downstream operation's storage class.
- `ScheduledTransfer`, device-native event-domain bridging, pending-output
  composition, and full admission logic remain later #1471 follow-up scope.
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
cargo test -p tenferro-runtime --test integration runtime_run_compiled_transfers_between_storage_classes_on_scheduled_path -- --nocapture
cargo test -p tenferro-runtime --test integration runtime_run_compiled_reports_missing_transfer_provider_for_cross_storage -- --nocapture
cargo test -p tenferro-runtime --test integration runtime_submit_wait_uses_prepared_execution_path -- --nocapture
cargo test -p tenferro-runtime prepared_program_is_binding_free_and_shares_staged_root -- --nocapture
cargo test -p tenferro-runtime preparation -- --nocapture
cargo test -p tenferro-runtime --test integration runtime_execution -- --nocapture
cargo test -p tenferro-ad --test integration runtime_execution -- --nocapture
cargo test -p tenferro-linalg --test integration --features autodiff linalg_internal_path_contract::semantic_linalg_linearize_does_not_replay_recorded_legacy_fragments -- --nocapture
cargo test -p tenferro-linalg --test integration --features autodiff ad_support_manifest:: -- --nocapture
cargo test -p tenferro-linalg --test integration --features autodiff traced_ad_explicit:: -- --nocapture
cargo test -p tenferro-linalg --test integration --features autodiff oracle_replay:: -- --nocapture
RUN_ORACLE_REPLAY=1 ORACLE_REPLAY_JOBS=64 cargo test -p tenferro-linalg --test integration --features autodiff oracle_replay::oracle_replays_supported_db_cases_when_requested -- --nocapture
cargo test -p tenferro-cpu two_input_binary_tail_classifier --lib -- --nocapture
cargo test -p tenferro-cpu binary_tail_specialization --lib -- --nocapture
cargo test -p tenferro-cpu elementwise_fusion --lib -- --nocapture
cargo test -p tenferro-runtime --test integration runtime_public_api -- --nocapture
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 RAYON_NUM_THREADS=1 CARGO_BUILD_JOBS=64 cargo bench -p tenferro-runtime --features __bench_unification_run_compiled_api --bench elementwise_fusion -- --sample-size 10 --warm-up-time 0.1 --measurement-time 0.3
gh issue close 1473 --reason completed
git diff --check
```

The full local oracle replay reported:

```text
ReplayRunSummary { total_records: 9585, supported_success_records: 2090, expected_error_records: 2, unsupported_records: 7493, skipped_by_filter_records: 0, replayed_success_records: 2090, replayed_expected_error_records: 2, parallel_jobs: 64 }
```

The #1464 Criterion run rebuilt the release bench target in `6m09s`, exposing
the same `tenferro-cpu` release-optimization long pole tracked by #1472. Median
times from the focused run were:

```text
runtime_elementwise_chain/f64/add_mul/segmented_graph/4096       43.776 us
runtime_elementwise_chain/f64/add_mul/segmented_graph/65536      125.64 us
runtime_elementwise_chain/f64/add_mul/segmented_graph/1048576    1.2320 ms
runtime_elementwise_chain/f64/broadcast_mul/segmented_graph/256x256       112.53 us
runtime_elementwise_chain/f64/broadcast_mul/segmented_graph/1024x1024     1.9468 ms
runtime_elementwise_chain/f64/broadcast_mul_add/segmented_graph/256x256   146.97 us
runtime_elementwise_chain/f64/broadcast_mul_add/segmented_graph/1024x1024 2.7071 ms
```

## Residual risks to carry forward

- The scheduled executor currently leases a backend state per instruction and
  uses the unsegmented execution path. This is correct for #1456; #1464 and
  #1473 own evidence-backed fusion/performance restoration.
- Cross-storage transfer currently rewrites the slot in place immediately
  before the downstream operation. This covers the linear fake two-device
  substrate; split-use buffer lifetime and explicit `ScheduledTransfer` nodes
  remain future scheduler work.
- CPU buffer-pool and GEMM analysis caches already expose ownership and limit
  controls, but their event counters remain coarse. This is not a blocker for
  #1478 because the generic runtime extension caches now have bounded retained
  bytes and observable statistics.
- Linalg reverse-mode paths that rely on `linearize` followed by semantic
  transposition still record a local linear fragment so the transpose
  interpreter can traverse it backward. This is no longer used by semantic JVP
  emission, and the numerical/oracle coverage above passed.
- CPU elementwise fusion remains a legacy segment-executor specialization.
  This PR records and slightly generalizes the current `Add`/`Multiply`
  classifier, but does not introduce a broad symbolic optimizer or tune
  `Divide`, `Pow`, ordered ops, broadcast views, or longer chains without
  separate benchmark evidence.
- Cache consolidation is complete for the current PR's runtime/AD/extension
  cache owners (#1478), but performance follow-ups #1484 and #1485 will add or
  reuse FFT/cuTENSOR caches only if they inherit the same owner, bounded
  default, retained-byte accounting, clear/configure API, and stats contract.
