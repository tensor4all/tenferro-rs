# Oracle Replay Adaptive Sharding Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make oracle replay shard count adapt to the local environment, while keeping shard runtimes balanced enough that workspace `nextest` wall time improves beyond the current fixed 4-shard implementation.

**Architecture:** Replace the fixed shard count constant with a runtime-selected active shard count and a fixed compile-time ceiling of registered shard tests. Build a filtered replay manifest first, compute the active shard count, then let each registered shard test execute the manifest entries selected by deterministic round-robin assignment or return immediately when it is outside the active set.

**Tech Stack:** Rust integration tests, `cargo test`, `cargo nextest`, existing tenferro CPU backend

---

### Task 1: Add pure tests for adaptive shard helpers

**Files:**
- Modify: `tenferro/tests/oracle_replay/main.rs`

- [ ] **Step 1: Write the failing tests**

Add focused tests for pure helpers:
- override parsing and validation for shard count
- automatic shard count clamping from `(available, total_cases, max_supported, min_cases_per_shard)`
- shard assignment coverage: every manifest entry belongs to exactly one active shard
- shard assignment balance on a deterministic fixture

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p tenferro --test oracle_replay --release adaptive_`
Expected: FAIL because the helper APIs do not exist yet

- [ ] **Step 3: Write minimal helper implementations**

Add pure functions for:
- shard count resolution inputs
- active shard count calculation
- deterministic active-shard entry assignment helpers

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p tenferro --test oracle_replay --release adaptive_`
Expected: PASS

### Task 2: Refactor replay execution around a manifest

**Files:**
- Modify: `tenferro/tests/oracle_replay/main.rs`

- [ ] **Step 1: Add manifest structs**

Introduce small test-local structs for loaded files and manifest entries.

- [ ] **Step 2: Build the filtered manifest first**

Discover files, load case vectors, apply `ORACLE_REPLAY_OP`, then apply `ORACLE_REPLAY_CASE_LIMIT` while building a deterministic manifest.

- [ ] **Step 3: Resolve active shard count**

Read `ORACLE_REPLAY_SHARD_COUNT` when present; otherwise compute a default from available parallelism and total filtered cases.

- [ ] **Step 4: Assign manifest entries to active shards**

Use round-robin assignment across manifest entry indices for the active shard count and skip execution entirely for registered shards outside the active set.

- [ ] **Step 5: Keep thread budgeting aligned**

Derive backend thread count from the active shard count and continue using `CpuBackend::with_threads(...)`.

### Task 3: Replace fixed shard entrypoints with a fixed ceiling

**Files:**
- Modify: `tenferro/tests/oracle_replay/main.rs`

- [ ] **Step 1: Replace fixed 4-shard setup**

Introduce a compile-time `MAX_ORACLE_REPLAY_SHARDS` and register shard tests up to that ceiling, preferably with a macro to keep the file readable.

- [ ] **Step 2: Preserve the ignored sequential test**

Keep `oracle_replay_all` available for baseline comparison and debugging.

- [ ] **Step 3: Update summary output**

Print active shard count and the executed shard selector so `--nocapture` output remains useful.

### Task 4: Verify behavior and timing

**Files:**
- Modify: `tenferro/tests/oracle_replay/main.rs` if follow-up fixes are needed

- [ ] **Step 1: Run focused helper tests**

Run: `cargo test -p tenferro --test oracle_replay --release adaptive_`
Expected: PASS

- [ ] **Step 2: Run focused replay tests**

Run: `cargo test -p tenferro --test oracle_replay --release shard_`
Expected: PASS

- [ ] **Step 3: Run workspace nextest**

Run: `cargo nextest run --workspace --release --no-fail-fast`
Expected: PASS, with no single oracle replay shard regressing toward the old single-test baseline

- [ ] **Step 4: Record outcome**

Capture the new slowest shard timing and compare it against both earlier baselines:
- `oracle_replay_all` about 80 seconds
- fixed 4-shard workspace about 30.5 seconds
