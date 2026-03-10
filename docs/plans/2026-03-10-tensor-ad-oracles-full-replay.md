# Tensor AD Oracles Full Replay Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Update tenferro's vendored oracle replay so it understands the latest `tensor-ad-oracles` schema, validates supported HVP-enabled families, and documents unsupported published families in a checked-in coverage report.

**Architecture:** Refresh the vendored oracle subtree, add a support registry plus schema/HVP-aware parser in `tenferro-linalg/tests/oracle_db/`, extend replay for currently supported families, and generate a deterministic Markdown support report from the same registry used by the tests.

**Tech Stack:** Rust integration tests, git subtree, serde/serde_json, existing tenferro linalg AD APIs, workspace docs checks.

---

### Task 1: Refresh the vendored oracle subtree

**Files:**
- Modify: `third_party/tensor-ad-oracles/`

**Step 1: Update the subtree to upstream `origin/main`**

Run:

```bash
git subtree pull --prefix=third_party/tensor-ad-oracles \
  https://github.com/tensor4all/tensor-ad-oracles.git main --squash
```

Expected: subtree merge completes without local conflicts.

**Step 2: Inspect the changed oracle surface**

Run:

```bash
git diff --stat HEAD~1 -- third_party/tensor-ad-oracles
```

Expected: updated schema, README, cases, and generator metadata appear.

**Step 3: Commit the subtree refresh**

```bash
git add third_party/tensor-ad-oracles
git commit -m "chore: refresh tensor-ad-oracles subtree"
```

### Task 2: Add a failing replay test for the new schema/HVP contract

**Files:**
- Modify: `tenferro-linalg/tests/oracle_db/main.rs`

**Step 1: Add a failing assertion for new full-replay accounting**

Add or update a test so it expects:

- no replay failures
- stable unsupported coverage output
- HVP-aware record counting for the refreshed subtree

**Step 2: Run the targeted test to verify it fails for the right reason**

Run:

```bash
cargo test -p tenferro-linalg --test oracle_db oracle_db_replay_against_tensor_ad_oracles -- --nocapture
```

Expected: FAIL because the current parser/replay cannot handle the new schema or HVP fields yet.

**Step 3: Commit the failing test**

```bash
git add tenferro-linalg/tests/oracle_db/main.rs
git commit -m "test: expose full oracle replay schema drift"
```

### Task 3: Update oracle DB parsing for the current schema

**Files:**
- Modify: `tenferro-linalg/tests/oracle_db/db.rs`
- Test: `tenferro-linalg/tests/oracle_db/main.rs`

**Step 1: Extend the local data model**

Add structs for:

- first-order comparison
- second-order comparison
- HVP-optional `pytorch_ref`
- HVP-optional `fd_ref`

Reject half-present HVP payloads at parse/validation time.

**Step 2: Add a focused parser test**

Add a small fixture-style test in `main.rs` or a helper module asserting:

- new comparison fields decode
- HVP fields decode when present
- malformed half-present HVP fails

**Step 3: Run the targeted parser test**

Run:

```bash
cargo test -p tenferro-linalg --test oracle_db oracle_db_parser_handles_current_schema -- --nocapture
```

Expected: PASS.

**Step 4: Commit**

```bash
git add tenferro-linalg/tests/oracle_db/db.rs tenferro-linalg/tests/oracle_db/main.rs
git commit -m "test: parse current tensor-ad-oracles schema"
```

### Task 4: Introduce a support registry for the full published surface

**Files:**
- Create: `tenferro-linalg/tests/oracle_db/support.rs`
- Modify: `tenferro-linalg/tests/oracle_db/main.rs`
- Modify: `tenferro-linalg/tests/oracle_db/replay.rs`

**Step 1: Define explicit support classifications**

Add a registry keyed by:

- `op`
- `family`
- `observable kind`

Classify records as:

- supported
- expected error
- unsupported with reason

**Step 2: Write a failing classification test**

Add a test asserting:

- all published records classify without falling into an implicit default branch

**Step 3: Run the classification test**

Run:

```bash
cargo test -p tenferro-linalg --test oracle_db oracle_db_every_record_is_classified -- --nocapture
```

Expected: FAIL until replay wiring uses the registry everywhere.

**Step 4: Wire replay through the registry**

Update `replay.rs` so support decisions come from the registry instead of hard-coded `matches!`.

**Step 5: Re-run the classification test**

Run the same command. Expected: PASS.

**Step 6: Commit**

```bash
git add tenferro-linalg/tests/oracle_db/support.rs \
  tenferro-linalg/tests/oracle_db/main.rs \
  tenferro-linalg/tests/oracle_db/replay.rs
git commit -m "feat: classify full oracle replay surface"
```

### Task 5: Keep first-order replay working for currently supported families

**Files:**
- Modify: `tenferro-linalg/tests/oracle_db/replay.rs`
- Modify: `tenferro-linalg/tests/oracle_db/decode.rs`

**Step 1: Adapt existing replay code to the new comparison model**

Replace old flat tolerance reads with:

- `comparison.first_order.rtol`
- `comparison.first_order.atol`

in all first-order checks.

**Step 2: Preserve Hermitian-wrapper behavior**

Make sure the `eigh`/`cholesky` paths still apply structured Hermitian mapping
under the refactored schema.

**Step 3: Run the targeted replay test**

Run:

```bash
cargo test -p tenferro-linalg --test oracle_db oracle_db_replay_against_tensor_ad_oracles -- --nocapture
```

Expected: either PASS for first-order supported families or fail only on missing HVP/report handling.

**Step 4: Commit**

```bash
git add tenferro-linalg/tests/oracle_db/replay.rs tenferro-linalg/tests/oracle_db/decode.rs
git commit -m "refactor: carry first-order oracle replay to current schema"
```

### Task 6: Add HVP replay for supported families

**Files:**
- Create: `tenferro-linalg/tests/oracle_db/hvp.rs`
- Modify: `tenferro-linalg/tests/oracle_db/replay.rs`
- Modify: `tenferro-linalg/tests/oracle_db/main.rs`

**Step 1: Add a failing HVP regression test**

Add a targeted test for at least one supported HVP-enabled family already in the
replay set, such as `solve` or `svd`, asserting that oracle HVP payloads are
checked.

**Step 2: Run the HVP test to verify it fails**

Run:

```bash
cargo test -p tenferro-linalg --test oracle_db oracle_db_replays_supported_hvp_cases -- --nocapture
```

Expected: FAIL because HVP comparison is not implemented yet.

**Step 3: Implement HVP helpers**

Create helper code that:

- decodes optional HVP maps
- computes tenferro-side scalarized HVPs for supported families
- compares them with `comparison.second_order`

Prefer the most stable existing second-order path per family. Do not broaden
family support in this task.

**Step 4: Re-run the HVP test**

Run the same command. Expected: PASS.

**Step 5: Commit**

```bash
git add tenferro-linalg/tests/oracle_db/hvp.rs \
  tenferro-linalg/tests/oracle_db/replay.rs \
  tenferro-linalg/tests/oracle_db/main.rs
git commit -m "feat: validate oracle HVP payloads for supported families"
```

### Task 7: Generate and check in the unsupported coverage report

**Files:**
- Create: `tenferro-linalg/tests/oracle_db/report.rs`
- Create: `docs/generated/tensor-ad-oracles-support.md`
- Modify: `tenferro-linalg/tests/oracle_db/main.rs`

**Step 1: Add deterministic report rendering**

Render a Markdown report from the support registry plus the vendored case tree.

Include:

- supported families
- expected error families
- unsupported families with reason
- record counts

**Step 2: Add a failing golden-style test**

Assert the checked-in report exactly matches regenerated output.

**Step 3: Run the report test to verify it fails**

Run:

```bash
cargo test -p tenferro-linalg --test oracle_db oracle_db_support_report_matches_checked_in_markdown -- --nocapture
```

Expected: FAIL until the generated Markdown is written.

**Step 4: Write the generated report**

Update `docs/generated/tensor-ad-oracles-support.md` from the renderer output.

**Step 5: Re-run the report test**

Run the same command. Expected: PASS.

**Step 6: Commit**

```bash
git add tenferro-linalg/tests/oracle_db/report.rs \
  tenferro-linalg/tests/oracle_db/main.rs \
  docs/generated/tensor-ad-oracles-support.md
git commit -m "docs: publish tensor-ad-oracles support coverage"
```

### Task 8: Link the support report from the README

**Files:**
- Modify: `README.md`

**Step 1: Add a short oracle coverage note**

Add a concise section linking to `docs/generated/tensor-ad-oracles-support.md`
and explain that supported cases are replay-validated while unsupported cases
are tracked explicitly.

**Step 2: Run docs checks that exercise the README build path**

Run:

```bash
cargo doc --workspace --no-deps
python3 scripts/check-docs-site.py
```

Expected: PASS.

**Step 3: Commit**

```bash
git add README.md
git commit -m "docs: link oracle replay support coverage"
```

### Task 9: Run targeted replay verification

**Files:**
- Modify: none

**Step 1: Run the oracle integration test suite**

Run:

```bash
cargo test -p tenferro-linalg --test oracle_db -- --nocapture
```

Expected: PASS with zero replay failures.

**Step 2: Sanity-check unsupported coverage**

Run:

```bash
rg -n \"## Unsupported|sample count|reason\" docs/generated/tensor-ad-oracles-support.md
```

Expected: report contains unsupported families and reasons.

**Step 3: Commit if any generated artifact changed during verification**

```bash
git add docs/generated/tensor-ad-oracles-support.md
git commit -m "test: refresh oracle support report after verification"
```

Only commit if verification changed the checked-in report.

### Task 10: Run full repository verification

**Files:**
- Modify: none

**Step 1: Format check**

Run:

```bash
cargo fmt --all --check
```

Expected: PASS. If it fails, run `cargo fmt --all`, then rerun the check.

**Step 2: Workspace tests**

Run:

```bash
cargo test --workspace --release
```

Expected: PASS.

**Step 3: Coverage gate**

Run:

```bash
cargo llvm-cov --workspace --json --output-path coverage.json
python3 scripts/check-coverage.py coverage.json
```

Expected: PASS.

**Step 4: Docs gate**

Run:

```bash
cargo doc --workspace --no-deps
python3 scripts/check-docs-site.py
```

Expected: PASS.

**Step 5: Commit**

```bash
git add -A
git commit -m "feat: expand tensor-ad-oracles replay coverage"
```

### Task 11: Create and monitor the PR

**Files:**
- Modify: none

**Step 1: Push the branch**

```bash
git push -u origin feat/tensor-ad-oracles-full-replay
```

**Step 2: Create the PR**

```bash
gh pr create --base main --head feat/tensor-ad-oracles-full-replay \
  --title "feat: expand tensor-ad-oracles replay coverage" \
  --body "$(cat <<'EOF'\n## Summary\n- refresh the vendored tensor-ad-oracles subtree and parser for the current schema\n- validate HVP payloads for supported replay families\n- publish explicit unsupported oracle coverage and link it from the README\n\nGenerated with [Claude Code](https://claude.com/claude-code)\nEOF\n)"
```

**Step 3: Enable auto-merge**

```bash
gh pr merge --auto --squash --delete-branch
```

**Step 4: Monitor checks**

Run:

```bash
bash scripts/monitor-pr-checks.sh <pr-number-or-url> --interval 30
```

Expected: all required checks pass.
