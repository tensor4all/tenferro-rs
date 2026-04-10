# Oracle Replay Sharding Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split `oracle_replay_all` into a small fixed number of deterministic shards so `nextest` can run oracle replay work in parallel and reduce end-to-end wall time.

**Architecture:** Extract the existing sequential runner in `tenferro/tests/oracle_replay/main.rs` into a shard-aware helper that filters cases by deterministic global case index. Keep the existing summary and replay logic, add focused shard-assignment tests first, then expose multiple `#[test]` entrypoints that `nextest` can schedule concurrently while constraining per-shard backend threads.

**Tech Stack:** Rust integration tests, `cargo test`, `cargo nextest`, existing tenferro CPU backend

---

### Task 1: Add shard-selection tests

**Files:**
- Modify: `tenferro/tests/oracle_replay/main.rs`

- [ ] **Step 1: Write the failing test**

Add tests for a pure shard-selection helper:
- one test that verifies every global case index in a sample range is assigned to exactly one shard
- one test that verifies shard assignment is deterministic and balanced within 1 case

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p tenferro --test oracle_replay --release shard_`
Expected: FAIL because the helper does not exist yet

- [ ] **Step 3: Write minimal implementation**

Add a small pure helper such as `case_in_shard(global_index, shard_index, shard_count) -> bool`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p tenferro --test oracle_replay --release shard_`
Expected: PASS

### Task 2: Refactor the replay runner for shard-aware execution

**Files:**
- Modify: `tenferro/tests/oracle_replay/main.rs`

- [ ] **Step 1: Extract the existing loop into a runner**

Move the body of `oracle_replay_all` into a helper that accepts shard parameters and accumulates `ReplayStats`.

- [ ] **Step 2: Keep existing env-filter behavior**

Preserve `ORACLE_REPLAY_OP` and `ORACLE_REPLAY_CASE_LIMIT`, applying the limit after shard filtering so each shard remains locally debuggable.

- [ ] **Step 3: Bound per-shard backend threads**

Create `Engine` with `CpuBackend::with_threads(...)` when running sharded tests so each shard does not grab the full machine by default.

- [ ] **Step 4: Add shard entrypoint tests**

Replace the single monolithic replay entrypoint with a small fixed set of tests such as `oracle_replay_shard_0` to `oracle_replay_shard_3`, each calling the shared runner.

### Task 3: Verify correctness and wall-clock behavior

**Files:**
- Modify: `tenferro/tests/oracle_replay/main.rs` if follow-up adjustments are needed

- [ ] **Step 1: Run focused oracle replay tests**

Run: `cargo test -p tenferro --test oracle_replay --release -- --nocapture shard_`
Expected: all shard tests pass

- [ ] **Step 2: Run workspace nextest**

Run: `cargo nextest run --workspace --release --no-fail-fast`
Expected: all tests pass, and oracle replay work is no longer represented by one long tail test

- [ ] **Step 3: Record outcome**

Capture the new shard timings and compare them against the baseline `oracle_replay_all` runtime of about 80 seconds.
