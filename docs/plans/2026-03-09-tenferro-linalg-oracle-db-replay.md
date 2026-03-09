# tenferro-linalg Oracle DB Replay Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a `tenferro-linalg` integration harness that replays the published `tensor-ad-oracles` database against the Rust AD implementation and reports supported vs unsupported coverage.

**Architecture:** Build a small Rust integration-test harness that parses JSONL case files, decodes `float64` tensors, maps processed observables onto `tenferro-linalg` APIs, and checks JVP/VJP consistency against the stored references. Keep the harness self-contained under `tenferro-linalg/tests/oracle_db/` and make the database root configurable via environment variable with a sibling-checkout default.

**Tech Stack:** Rust integration tests, `serde`, `serde_json`, `tenferro-linalg`, `tenferro-tensor`

---

### Task 1: Add a failing end-to-end oracle DB replay test

**Files:**
- Modify: `tenferro-linalg/Cargo.toml`
- Create: `tenferro-linalg/tests/oracle_db/main.rs`

**Step 1: Write the failing test**

Create `tenferro-linalg/tests/oracle_db/main.rs` with a test that:

- locates the DB root
- calls `replay::run_database_replay()`
- expects:
  - `validated_records == 348`
  - `unsupported_case_ids == ["eigh_c128_gauge_ill_defined_001", "svd_c128_gauge_ill_defined_001"]`
  - `failures.is_empty()`

**Step 2: Run test to verify it fails**

Run:

```bash
cargo test -p tenferro-linalg oracle_db_replay_against_tensor_ad_oracles -- --nocapture
```

Expected: FAIL because the replay module and JSON dependencies do not exist yet.

**Step 3: Add minimal dev dependencies**

In `tenferro-linalg/Cargo.toml`, add:

- `serde = { version = "1", features = ["derive"] }`
- `serde_json = "1"`

under `[dev-dependencies]`.

**Step 4: Commit**

```bash
git add tenferro-linalg/Cargo.toml tenferro-linalg/tests/oracle_db/main.rs
git commit -m "test: add failing oracle DB replay entrypoint"
```

### Task 2: Add DB discovery and JSON decoding helpers

**Files:**
- Create: `tenferro-linalg/tests/oracle_db/db.rs`
- Create: `tenferro-linalg/tests/oracle_db/decode.rs`
- Modify: `tenferro-linalg/tests/oracle_db/main.rs`

**Step 1: Write failing helper-focused assertions**

Add tests for:

- resolving the default DB root `../tensor-ad-oracles`
- reading JSONL case counts
- decoding one `float64` tensor object into `Tensor<f64>`

**Step 2: Run targeted test to verify it fails**

Run:

```bash
cargo test -p tenferro-linalg oracle_db:: -- --nocapture
```

Expected: FAIL with missing helper modules/functions.

**Step 3: Write minimal implementation**

Implement:

- DB root resolution
- JSONL record loading
- minimal `serde` structs for success records and probes
- `float64` tensor decode from row-major DB layout into `Tensor<f64>`

**Step 4: Run targeted test to verify it passes**

Run the same command and confirm the helper tests pass.

**Step 5: Commit**

```bash
git add tenferro-linalg/tests/oracle_db/main.rs tenferro-linalg/tests/oracle_db/db.rs tenferro-linalg/tests/oracle_db/decode.rs
git commit -m "feat: add oracle DB loading helpers for tenferro-linalg"
```

### Task 3: Replay direct-identity families

**Files:**
- Create: `tenferro-linalg/tests/oracle_db/replay.rs`
- Modify: `tenferro-linalg/tests/oracle_db/main.rs`

**Step 1: Write a failing test**

Add a test that replays one case each for:

- `solve/identity`
- `cholesky/identity`
- `qr/identity`

and expects no failures.

**Step 2: Run test to verify it fails**

Run:

```bash
cargo test -p tenferro-linalg oracle_db_replay_identity_families -- --nocapture
```

Expected: FAIL because replay logic is not implemented.

**Step 3: Write minimal implementation**

Implement direct-family replay:

- forward call into `solve`, `cholesky`, `qr`
- raw `rrule` / `frule` replay
- observable comparison
- summary reporting

**Step 4: Run test to verify it passes**

Run the same command and confirm the identity-family replay passes.

**Step 5: Commit**

```bash
git add tenferro-linalg/tests/oracle_db/main.rs tenferro-linalg/tests/oracle_db/replay.rs
git commit -m "feat: replay direct oracle DB families in tenferro-linalg"
```

### Task 4: Replay spectral processed-observable families

**Files:**
- Modify: `tenferro-linalg/tests/oracle_db/replay.rs`
- Modify: `tenferro-linalg/tests/oracle_db/main.rs`

**Step 1: Write a failing test**

Add replay coverage for:

- `svd/u_abs`
- `svd/s`
- `svd/vh_abs`
- `svd/uvh_product`
- `eigh/values_vectors_abs`

**Step 2: Run test to verify it fails**

Run:

```bash
cargo test -p tenferro-linalg oracle_db_replay_spectral_families -- --nocapture
```

Expected: FAIL because processed observable mapping is missing.

**Step 3: Write minimal implementation**

Implement:

- observable adapters from raw primal/tangent outputs
- cotangent pullback from observable space to raw `svd_rrule` / `eigen_rrule`
- shape-safe matrix helper routines for `abs`, `u @ vt`, and sign-based pullbacks

**Step 4: Run test to verify it passes**

Run the same command and confirm the spectral replay passes.

**Step 5: Commit**

```bash
git add tenferro-linalg/tests/oracle_db/main.rs tenferro-linalg/tests/oracle_db/replay.rs
git commit -m "feat: replay spectral oracle DB families in tenferro-linalg"
```

### Task 5: Replay `pinv_singular` and full database summary

**Files:**
- Modify: `tenferro-linalg/tests/oracle_db/replay.rs`
- Modify: `tenferro-linalg/tests/oracle_db/main.rs`

**Step 1: Write a failing test**

Add full-database replay expectations:

- `validated_records == 348`
- unsupported IDs match the two complex gauge-error records
- `failures.is_empty()`

**Step 2: Run test to verify it fails**

Run:

```bash
cargo test -p tenferro-linalg oracle_db_replay_against_tensor_ad_oracles -- --nocapture
```

Expected: FAIL because `pinv_singular` and/or unsupported-case accounting are incomplete.

**Step 3: Write minimal implementation**

Implement:

- `pinv_singular` chain rule through `a @ b^T`
- unsupported-case accounting for the two complex error records
- full tree traversal with deterministic summary

**Step 4: Run test to verify it passes**

Run the same command and confirm the full DB replay passes.

**Step 5: Commit**

```bash
git add tenferro-linalg/tests/oracle_db/main.rs tenferro-linalg/tests/oracle_db/replay.rs
git commit -m "feat: replay the published oracle DB against tenferro-linalg"
```

### Task 6: Final verification

**Files:**
- Modify: `tenferro-linalg/tests/oracle_db/main.rs`
- Modify: `tenferro-linalg/tests/oracle_db/*.rs`
- Optionally modify: `tenferro-linalg/Cargo.toml`

**Step 1: Run focused replay verification**

```bash
cargo test -p tenferro-linalg oracle_db -- --nocapture
```

Expected: PASS with validated `348` and unsupported `2`.

**Step 2: Run crate-level verification**

```bash
cargo test -p tenferro-linalg --release
```

Expected: PASS

**Step 3: Run formatting**

```bash
cargo fmt --all
cargo fmt --all --check
```

Expected: PASS

**Step 4: Commit**

```bash
git add tenferro-linalg/Cargo.toml tenferro-linalg/tests/oracle_db/main.rs tenferro-linalg/tests/oracle_db/db.rs tenferro-linalg/tests/oracle_db/decode.rs tenferro-linalg/tests/oracle_db/replay.rs docs/plans/2026-03-09-tenferro-linalg-oracle-db-replay-design.md docs/plans/2026-03-09-tenferro-linalg-oracle-db-replay.md
git commit -m "test: validate tenferro-linalg against tensor-ad-oracles"
```
