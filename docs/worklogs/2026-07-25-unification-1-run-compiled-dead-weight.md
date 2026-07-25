# Unification 1 run_compiled dead-weight removal

## Scope

This worklog records the #1455 cleanup pass after the Phase 8 local work. The
goal is to remove old execution paths and obvious per-run dead weight before
running the broader #1454 benchmark gate.

## Context read

- Issue #1455, including the accepted item list from the PR #1453 review.
- `crates/tenferro-runtime/src/runtime/execution.rs`
- `crates/tenferro-runtime/src/runtime/preparation.rs`
- `crates/tenferro-runtime/src/runtime/cache.rs`
- `crates/tenferro-runtime/src/graph/compiler.rs`
- `crates/tenferro-runtime/src/compiler/semantic_staging.rs`
- Current public docs, design docs, tutorials, and guide-snippet checker.

## Attribution table

| #1455 item | Local delta | Status |
|---|---|---|
| Per-run `ScheduledGraph` construction and validation in `run_compiled` | `Runtime::run_compiled` now prepares once through `Runtime::prepare_compiled_for` and directly executes the prepared root `ExecProgram`; no production `ScheduledGraph` is constructed in `runtime/execution.rs`. | Removed before this checkpoint. |
| Unreachable specialization comparison in `runtime/execution.rs` | The execution bridge no longer performs a separate specialization comparison. Specialization remains part of the prepared-cache key and is checked by `PreparedEntryKey::exact_eq`. | Removed before this checkpoint. |
| Triple shape-guard validation per run | Production execution no longer runs the old staged `exec::validate_shape_guards` pass. `resolve_input_tensors` still validates public ordered-input count/dtype/rank/exact/bound errors, while preparation validates semantic shape guards once for cache identity and provider planning. | Production duplicate removed; public input contract retained. |
| Discarded compile staging / write-only compile cache | `GraphCompiler::compile_frozen` now returns `CompiledGraph::new(frozen.clone(), compiler_options)` and does not lower to execution staging. Runtime preparation owns staging via `stage_semantic_program`. | Removed before this checkpoint. |
| Prepared-plan cache probe cloning on hit path | `PreparedPlanCache::probe` no longer allocates a candidate-key list. It snapshots bucket entry IDs, borrows one candidate key at a time, evaluates `exact_eq` outside the cache lock, and validates the generation before returning a hit. | Removed in this checkpoint. |
| `Runtime::snapshot()` global mutex | `Runtime` publishes snapshots through an `RwLock` and an atomic epoch. The exact lock-call counter test is not present. | Removed before this checkpoint. |
| `legacy_stage_compiled_graph` test divergence | The `graph/executor.rs` file and GraphExecutor-only integration tests are deleted; source-contract tests assert the retired facade is not re-exported. | Removed before this checkpoint. |

## TDD evidence for this checkpoint

The cache-probe regression guard was added before the implementation change:

```console
cargo test -p tenferro-runtime runtime::tests::cache::probe_path_does_not_collect_arc_cloned_candidate_keys --lib -- --nocapture
```

The RED run failed because `probe` still referenced `ProbeCandidate`. After the
implementation change, the same command passed. The existing lock-boundary
regression test also passed:

```console
cargo test -p tenferro-runtime runtime::tests::cache::digest_collision_uses_exact_equality_outside_lock --lib -- --nocapture
```

## Residual risk

This pass records structural cleanup, not final performance attribution. The
next performance evidence should be the #1454 gate: requires_grad true/false,
warm/cold, extent-churn, topology-churn, operation-size tiers, PyTorch
reference values as non-gating context, and the gating comparison against
`main`.
