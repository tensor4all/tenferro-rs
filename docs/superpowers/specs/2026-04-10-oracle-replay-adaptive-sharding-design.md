# Oracle Replay Adaptive Sharding Design

## Goal

Reduce `cargo nextest run --workspace --release --no-fail-fast` wall time further by replacing the current fixed 4-way oracle replay split with an adaptive scheme that chooses shard count from the execution environment and keeps shard runtimes more even.

## Current State

`tenferro/tests/oracle_replay/main.rs` currently exposes four replay shard tests plus an ignored legacy `oracle_replay_all`. The fixed split reduced workspace wall time from about 80.8 seconds to about 30.5 seconds, but the slowest shard still ran for about 30.2 seconds while the others finished earlier. The remaining tail comes from uneven case cost and from treating shard count as a compile-time constant.

## Requirements

1. Shard count must be environment-sensitive by default.
2. A caller must be able to override shard count explicitly for CI or local debugging.
3. The number of registered test entrypoints must remain fixed at compile time because Rust integration tests cannot be generated dynamically.
4. The active shard set must remain deterministic for a given environment and override configuration.
5. Per-shard backend thread count must continue to scale down with the active shard count to avoid oversubscription.
6. Existing `ORACLE_REPLAY_OP` and `ORACLE_REPLAY_CASE_LIMIT` behavior must remain available.

## Approaches Considered

### 1. Environment override only

Use `ORACLE_REPLAY_SHARD_COUNT` and otherwise keep the current fixed default.

Pros: smallest code change.
Cons: no automatic improvement on larger or smaller machines; every environment needs manual tuning.

### 2. Fully automatic shard count

Derive shard count only from `available_parallelism()`.

Pros: zero configuration.
Cons: harder to stabilize across CI and local runs; risks picking too many tiny shards on large machines.

### 3. Automatic default with explicit override

Compute a default active shard count from `available_parallelism()` and bounded local heuristics, but allow `ORACLE_REPLAY_SHARD_COUNT` to override it.

Pros: good default behavior, still reproducible when needed, clean CI escape hatch.
Cons: slightly more logic.

## Chosen Design

Use approach 3.

### Test shape

Keep a fixed compile-time ceiling such as `MAX_ORACLE_REPLAY_SHARDS = 16`. Expose `oracle_replay_shard_0` through `oracle_replay_shard_15` as normal `#[test]` entrypoints. Each entrypoint queries the active shard count at runtime. If its index is outside the active shard count, it returns immediately.

This preserves compatibility with Rust's static test registration while allowing the active shard count to vary by machine or by environment variable.

### Active shard count

Add `oracle_replay_active_shard_count(total_cases: usize) -> usize`.

Resolution order:

1. If `ORACLE_REPLAY_SHARD_COUNT` is set, parse and use it after validation.
2. Otherwise compute a default from `available_parallelism()`.

Default policy:

- start from `available_parallelism()`
- cap at `MAX_ORACLE_REPLAY_SHARDS`
- cap at `total_cases` so we never create empty active shards
- cap again by a minimum target cases-per-shard heuristic so we do not fragment small filtered runs

The default should favor predictable coarse shards over maximal fan-out.

### Shard balancing

Build a lightweight replay manifest from the filtered case set, then assign entries to active shards with deterministic round-robin selection:

- shard `i` executes entries where `entry_index % active_shard_count == i`

This keeps large case files and expensive ops interleaved across shards. A contiguous weighted-range prototype was measured and rejected because the manifest order clusters expensive operations near the tail of the replay corpus; that produced badly imbalanced shards even when case counts looked balanced.

### Thread budgeting

Continue deriving backend thread count from active shard count:

- `threads = max(1, available_parallelism / active_shard_count)`

This keeps the outer test parallelism and inner CPU backend parallelism in balance.

## Data Flow

1. Discover case files.
2. Apply `ORACLE_REPLAY_OP`.
3. Load or count cases needed to build the filtered manifest.
4. Apply `ORACLE_REPLAY_CASE_LIMIT`.
5. Compute active shard count.
6. Assign manifest entries to shards with round-robin selection.
7. Each registered shard test either exits early or runs its assigned entry subset with the derived backend thread budget.

## Error Handling

- Invalid `ORACLE_REPLAY_SHARD_COUNT` should panic with a precise message, matching the existing style used for `ORACLE_REPLAY_CASE_LIMIT`.
- Values less than 1 or greater than `MAX_ORACLE_REPLAY_SHARDS` should be rejected explicitly.
- If filtering leaves zero runnable cases, shard tests should return successfully without work.

## Testing

Add focused tests for:

1. environment override parsing and validation
2. automatic shard count clamping
3. shard assignment coverage: every manifest entry belongs to exactly one active shard
4. shard assignment balance: entry counts across shards stay within a small bound on deterministic fixtures
5. inactive shard entrypoints exiting without replay work

Retain any helper-level partition tests only as analysis support; the execution path should be validated against the round-robin selector that the test runner actually uses.

## Expected Outcome

The workspace `nextest` run should still pass cleanly. Exact timing remains machine-dependent, but the design should remove the need to hard-code shard count for each environment while staying close to or better than the fixed-4 baseline on typical developer machines.
