# tensor-ad-oracles Replay Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Vendor `tensor-ad-oracles` into `tenferro-rs` and add an always-on Rust replay harness that validates `tenferro-linalg` against the published JSON oracle database.

**Architecture:** Vendor the full upstream repository under `third_party/tensor-ad-oracles/`, then keep all replay code under `tenferro-linalg/tests/oracle_db/`. The harness decodes row-major oracle tensors into tenferro tensors, maps each published observable to a `tenferro-linalg` computation, and checks forward/JVP/VJP consistency using the per-case tolerances stored in the database.

**Tech Stack:** git subtree, Rust integration tests, `serde`, `serde_json`, `tenferro-linalg`, `tenferro-tensor`

---

### Task 1: Vendor the oracle repository snapshot

**Files:**
- Create: `third_party/tensor-ad-oracles/` via git subtree
- Modify: `.gitignore` only if subtree-generated local artifacts need to be ignored

**Step 1: Add the subtree on a dedicated prefix**

Run:

```bash
git subtree add --prefix=third_party/tensor-ad-oracles <remote-url> main --squash
```

Expected: the full oracle repository appears under `third_party/tensor-ad-oracles/`.

**Step 2: Verify the vendored layout is present**

Run:

```bash
find third_party/tensor-ad-oracles -maxdepth 2 -type f | sort | sed -n '1,80p'
```

Expected: `README.md`, `schema/case.schema.json`, and `cases/*/*.jsonl` exist.

**Step 3: Commit**

```bash
git add third_party/tensor-ad-oracles
git commit -m "test: vendor tensor-ad-oracles subtree"
```

### Task 2: Add a failing replay entrypoint test

**Files:**
- Modify: `tenferro-linalg/Cargo.toml`
- Create: `tenferro-linalg/tests/oracle_db/main.rs`

**Step 1: Write the failing entrypoint test**

Create `tenferro-linalg/tests/oracle_db/main.rs` with a test that:

- locates `third_party/tensor-ad-oracles/cases`
- calls `replay::run_database_replay()`
- asserts:
  - all expected success records were validated
  - gauge-ill-defined records were classified as expected failures
  - `failures.is_empty()`

**Step 2: Run the targeted test to verify it fails**

Run:

```bash
cargo test -p tenferro-linalg oracle_db_replay_against_tensor_ad_oracles -- --nocapture
```

Expected: FAIL because the replay modules and JSON parsing do not exist yet.

**Step 3: Add minimal test-only dependencies**

Add under `[dev-dependencies]` in `tenferro-linalg/Cargo.toml`:

```toml
serde = { workspace = true, features = ["derive"] }
serde_json = { workspace = true }
```

If the workspace root does not already expose them, add them there first.

**Step 4: Commit**

```bash
git add tenferro-linalg/Cargo.toml tenferro-linalg/tests/oracle_db/main.rs Cargo.toml
git commit -m "test: add failing oracle replay entrypoint"
```

### Task 3: Add database and decode helpers

**Files:**
- Create: `tenferro-linalg/tests/oracle_db/db.rs`
- Create: `tenferro-linalg/tests/oracle_db/decode.rs`
- Modify: `tenferro-linalg/tests/oracle_db/main.rs`

**Step 1: Write failing helper tests**

Add tests covering:

- vendored DB root discovery
- JSONL file enumeration
- decoding one `float64` tensor from row-major JSON into a tenferro tensor
- rejecting unsupported dtype/order combinations with clear errors

**Step 2: Run helper tests to verify they fail**

Run:

```bash
cargo test -p tenferro-linalg oracle_db:: -- --nocapture
```

Expected: FAIL with missing helper modules or decode functions.

**Step 3: Implement minimal helper structs and decoding**

Implement:

- case-file enumeration from `third_party/tensor-ad-oracles/cases`
- `serde` structs for case records, tensor payloads, probes, and comparisons
- row-major to column-major tensor decode for `float64`
- error-case parsing for gauge-ill-defined records

**Step 4: Run helper tests to verify they pass**

Run the same command and confirm the helper tests pass.

**Step 5: Commit**

```bash
git add tenferro-linalg/tests/oracle_db/main.rs tenferro-linalg/tests/oracle_db/db.rs tenferro-linalg/tests/oracle_db/decode.rs
git commit -m "feat: add oracle DB loading and decode helpers"
```

### Task 4: Replay direct identity families

**Files:**
- Create: `tenferro-linalg/tests/oracle_db/replay.rs`
- Create: `tenferro-linalg/tests/oracle_db/observables.rs`
- Modify: `tenferro-linalg/tests/oracle_db/main.rs`

**Step 1: Write a failing identity-family replay test**

Add a test that replays one case each for:

- `solve/identity`
- `cholesky/identity`
- `qr/identity`

and expects no failures.

**Step 2: Run the targeted test to verify it fails**

Run:

```bash
cargo test -p tenferro-linalg oracle_db_replay_identity_families -- --nocapture
```

Expected: FAIL because replay logic and observable handling are incomplete.

**Step 3: Implement direct observable replay**

Implement:

- `identity` observable mapping
- forward replay for `solve`, `cholesky`, and `qr`
- probe JVP replay using the stored direction tensors
- probe VJP / adjoint-consistency replay using the stored cotangents
- concise failure formatting that includes `case_id`

**Step 4: Re-run the targeted test**

Run the same command and confirm it passes.

**Step 5: Commit**

```bash
git add tenferro-linalg/tests/oracle_db/main.rs tenferro-linalg/tests/oracle_db/replay.rs tenferro-linalg/tests/oracle_db/observables.rs
git commit -m "feat: replay identity oracle families in tenferro-linalg"
```

### Task 5: Replay spectral observables

**Files:**
- Modify: `tenferro-linalg/tests/oracle_db/observables.rs`
- Modify: `tenferro-linalg/tests/oracle_db/replay.rs`
- Modify: `tenferro-linalg/tests/oracle_db/main.rs`

**Step 1: Write failing spectral replay tests**

Add coverage for:

- `svd/u_abs`
- `svd/s`
- `svd/vh_abs`
- `svd/uvh_product`
- `eigh/values_vectors_abs`

**Step 2: Run the targeted test to verify it fails**

Run:

```bash
cargo test -p tenferro-linalg oracle_db_replay_spectral_families -- --nocapture
```

Expected: FAIL because the observable adapters and pullback/JVP mappings are not complete.

**Step 3: Implement minimal spectral observable adapters**

Implement:

- absolute-value projection for singular/eigenvector observables
- singular-value-only observable extraction
- `U @ Vh` observable for SVD
- observable-space cotangent mapping back into raw `svd_rrule` / `eigen_rrule` inputs
- tolerance-aware tensor-map comparisons

**Step 4: Re-run the targeted test**

Run the same command and confirm it passes.

**Step 5: Commit**

```bash
git add tenferro-linalg/tests/oracle_db/main.rs tenferro-linalg/tests/oracle_db/observables.rs tenferro-linalg/tests/oracle_db/replay.rs
git commit -m "feat: replay spectral oracle observables"
```

### Task 6: Handle expected error families and full replay

**Files:**
- Modify: `tenferro-linalg/tests/oracle_db/replay.rs`
- Modify: `tenferro-linalg/tests/oracle_db/main.rs`

**Step 1: Write a failing full-replay summary test**

Add a top-level test that:

- replays the entire vendored case tree
- asserts all `success` cases validated
- asserts the expected gauge-ill-defined records were rejected
- asserts `failures.is_empty()`

**Step 2: Run the full replay test to verify it fails**

Run:

```bash
cargo test -p tenferro-linalg oracle_db_replay_against_tensor_ad_oracles -- --nocapture
```

Expected: FAIL because error-family accounting and/or one or more observable families are still incomplete.

**Step 3: Implement expected-error classification**

Implement:

- error-case dispatch by `reason_code`
- gauge-ill-defined case matching for `svd` and `eigh`
- summary accounting that separates validated, expected-error, and hard-failure records

**Step 4: Re-run the full replay test**

Run the same command and confirm it passes.

**Step 5: Commit**

```bash
git add tenferro-linalg/tests/oracle_db/main.rs tenferro-linalg/tests/oracle_db/replay.rs
git commit -m "feat: validate full tensor-ad-oracles replay in tenferro-linalg"
```

### Task 7: Wire the replay into normal verification

**Files:**
- Modify: `README.md` only if local verification instructions need an oracle note
- Modify: CI workflow files only if the replay needs explicit naming or reporting

**Step 1: Confirm the replay runs under normal workspace tests**

Run:

```bash
cargo test --workspace --release
```

Expected: PASS, including the vendored oracle replay tests.

**Step 2: Run the full repository verification suite**

Run:

```bash
cargo fmt --all --check
cargo test --workspace --release
cargo llvm-cov --workspace --json --output-path coverage.json
python3 scripts/check-coverage.py coverage.json
cargo doc --workspace --no-deps
python3 scripts/check-docs-site.py
```

Expected: all commands PASS.

**Step 3: Update docs only if needed**

If the replay meaningfully changes contributor workflow, add a short note to
`README.md` describing the vendored oracle subtree and the always-on replay
test coverage.

**Step 4: Commit**

```bash
git add README.md .github/workflows scripts coverage-thresholds.json
git commit -m "docs: document vendored oracle replay workflow"
```

Only include files that actually changed.
