# Scalarized HVP Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add scalarized second-order HVP oracle data to `tensor-ad-oracles` for upstream PyTorch families that support forward-over-backward AD.

**Architecture:** Extend the existing paired-probe model so each eligible success probe stores first-order data plus scalarized HVP data computed from the same `(direction, cotangent)` pair. Use PyTorch `grad`+`jvp` as one oracle and finite-difference on `grad(phi)` as the second oracle, with separate second-order tolerances and replay checks.

**Tech Stack:** Python 3.12, `uv`, pinned `torch==2.10.0`, `jsonschema`, `unittest`, GitHub Actions.

---

### Task 1: Add schema tests for second-order probe data

**Files:**
- Modify: `tests/test_schema_contract.py`
- Modify: `schema/case.schema.json`

**Step 1: Write the failing test**

Extend schema tests to require:

- `comparison.first_order`
- `comparison.second_order`
- optional `pytorch_ref.hvp`
- optional `fd_ref.hvp`

for HVP-enabled success probes.

**Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_schema_contract -v`

Expected: FAIL because the schema only knows first-order comparison and has no `hvp`.

**Step 3: Write minimal implementation**

Update the schema to:

- split comparison blocks into first- and second-order sections
- allow `pytorch_ref.hvp` and `fd_ref.hvp`
- keep backward compatibility for existing error cases

**Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.test_schema_contract -v`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_schema_contract.py schema/case.schema.json
git commit -m "feat: add schema support for scalarized hvp"
```

### Task 2: Add scalarized HVP helper tests

**Files:**
- Create: `tests/test_hvp_helpers.py`
- Modify: `generators/runtime.py`
- Modify: `generators/tolerance_audit.py`

**Step 1: Write the failing test**

Add helper tests that require:

- building `phi(x) = <cotangent, observable(x)>`
- computing PyTorch HVP from `grad(phi)` and `jvp`
- computing FD HVP from central differences of `grad(phi)`
- reusing the real-part scalarization for complex outputs

**Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_hvp_helpers -v`

Expected: FAIL because no HVP helper path exists.

**Step 3: Write minimal implementation**

Add runtime helpers for:

- scalarized observable inner-product closures
- PyTorch HVP generation
- FD HVP generation
- max abs/rel HVP residual measurement

**Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.test_hvp_helpers -v`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_hvp_helpers.py generators/runtime.py generators/tolerance_audit.py
git commit -m "feat: add scalarized hvp helper primitives"
```

### Task 3: Add upstream HVP eligibility inventory

**Files:**
- Modify: `generators/upstream_inventory.py`
- Modify: `tests/test_upstream_inventory.py`
- Modify: `tests/test_family_mapping.py`

**Step 1: Write the failing test**

Require the upstream inventory to expose:

- `supports_fwgrad_bwgrad`
- any explicit upstream `fwgrad_bwgrad` xfail classification needed for exclusion

**Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_upstream_inventory tests.test_family_mapping -v`

Expected: FAIL because the inventory does not expose enough second-order eligibility metadata.

**Step 3: Write minimal implementation**

Extend the normalized inventory record and mapping layer so each case family can be classified as:

- HVP-enabled
- explicitly unsupported for HVP

**Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.test_upstream_inventory tests.test_family_mapping -v`

Expected: PASS

**Step 5: Commit**

```bash
git add generators/upstream_inventory.py tests/test_upstream_inventory.py tests/test_family_mapping.py
git commit -m "feat: track upstream hvp eligibility"
```

### Task 4: Add HVP fields to probe construction

**Files:**
- Modify: `generators/probes.py`
- Modify: `tests/test_probes.py`
- Modify: `tests/test_materialize.py`

**Step 1: Write the failing test**

Require probe assembly to accept and emit:

- `pytorch_ref.hvp`
- `fd_ref.hvp`

for HVP-enabled probes.

**Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_probes tests.test_materialize -v`

Expected: FAIL because current probe records only carry JVP/VJP/FD-JVP.

**Step 3: Write minimal implementation**

Update probe construction to:

- keep first-order fields untouched
- optionally include HVP fields
- preserve current encoding layout

**Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.test_probes tests.test_materialize -v`

Expected: PASS

**Step 5: Commit**

```bash
git add generators/probes.py tests/test_probes.py tests/test_materialize.py
git commit -m "feat: extend probes with hvp payloads"
```

### Task 5: Materialize HVP for one stable family

**Files:**
- Modify: `generators/pytorch_v1.py`
- Modify: `tests/test_pytorch_v1.py`
- Modify: `tests/test_solve_generation.py`

**Step 1: Write the failing test**

Start with `solve/identity` and require:

- HVP-enabled probes when upstream says `supports_fwgrad_bwgrad=True`
- generated `pytorch_ref.hvp`
- generated `fd_ref.hvp`
- separate second-order tolerance block

**Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_pytorch_v1 tests.test_solve_generation -v`

Expected: FAIL because generation is still first-order only.

**Step 3: Write minimal implementation**

Add to the generator:

- per-family HVP eligibility
- scalarized `phi`
- PyTorch HVP generation
- FD HVP generation
- second-order tolerance derivation from observed HVP residuals

Implement only enough to make `solve/identity` work first.

**Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.test_pytorch_v1 tests.test_solve_generation -v`

Expected: PASS

**Step 5: Commit**

```bash
git add generators/pytorch_v1.py tests/test_pytorch_v1.py tests/test_solve_generation.py
git commit -m "feat: materialize scalarized hvp for solve"
```

### Task 6: Extend replay validation to second order

**Files:**
- Modify: `validators/replay.py`
- Modify: `tests/test_db_replay.py`
- Modify: `scripts/check_replay.py`

**Step 1: Write the failing test**

Require replay validation to check:

- stored/replayed `pytorch_ref.hvp`
- stored/replayed `fd_ref.hvp`
- live `pytorch_ref.hvp ~= fd_ref.hvp` under second-order tolerance

**Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_db_replay -v`

Expected: FAIL because replay ignores HVP fields.

**Step 3: Write minimal implementation**

Extend replay to:

- reconstruct scalarized `phi`
- recompute both HVP oracles
- validate second-order tolerance separately from first-order checks

**Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.test_db_replay -v`

Expected: PASS

**Step 5: Commit**

```bash
git add validators/replay.py tests/test_db_replay.py scripts/check_replay.py
git commit -m "feat: replay scalarized hvp references"
```

### Task 7: Extend tolerance audit to second order

**Files:**
- Modify: `generators/tolerance_audit.py`
- Modify: `scripts/check_tolerances.py`
- Modify: `tests/test_tolerance_audit.py`
- Modify: `tests/test_tolerance_audit_script.py`

**Step 1: Write the failing test**

Require the audit to:

- compute first-order and second-order maxima separately
- propose second-order tolerances from HVP residuals
- fail when second-order tolerance is more than ten orders too loose

**Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_tolerance_audit tests.test_tolerance_audit_script -v`

Expected: FAIL because the audit only knows JVP/VJP residuals.

**Step 3: Write minimal implementation**

Extend audit records and CLI output to include second-order statistics and proposed updates.

**Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.test_tolerance_audit tests.test_tolerance_audit_script -v`

Expected: PASS

**Step 5: Commit**

```bash
git add generators/tolerance_audit.py scripts/check_tolerances.py tests/test_tolerance_audit.py tests/test_tolerance_audit_script.py
git commit -m "feat: audit scalarized hvp tolerances"
```

### Task 8: Expand HVP coverage across eligible families

**Files:**
- Modify: `generators/pytorch_v1.py`
- Modify: `cases/**/*.jsonl`
- Reference: `generators/upstream_inventory.py`

**Step 1: Add/extend one family at a time**

Expand from `solve` to additional stable HVP-enabled families, prioritizing:

- `cholesky`
- `inv`
- `det`
- `slogdet`
- `qr`
- `svd`
- `eigh`
- `eig`

Skip upstream-explicit second-order xfail families until explicitly addressed.

**Step 2: Regenerate the database**

Run: `uv run python -m generators.pytorch_v1 --materialize-all --limit 1`

Then:

Run: `uv run python - <<'PY'\nfrom generators.pytorch_v1 import materialize_all_case_families\nmaterialize_all_case_families(limit=None)\nPY`

Expected: all supported HVP-enabled families regenerate.

**Step 3: Verify case tree**

Run:

- `uv run python scripts/validate_schema.py`
- `uv run python scripts/verify_cases.py`
- `uv run python scripts/check_replay.py`
- `uv run python scripts/check_regeneration.py`
- `uv run python scripts/check_tolerances.py`

Expected: PASS

**Step 4: Commit**

```bash
git add generators/pytorch_v1.py cases
git commit -m "feat: add scalarized hvp oracle coverage"
```

### Task 9: Update docs and CI contract

**Files:**
- Modify: `README.md`
- Modify: `.github/workflows/oracle-integrity.yml`
- Modify: `.github/workflows/oracle-regeneration.yml` if needed
- Modify: `tests/test_repo_config.py`

**Step 1: Write the failing test**

Require docs/config tests to mention:

- scalarized HVP support
- split first- and second-order tolerances
- HVP replay/tolerance audit coverage in CI

**Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_repo_config -v`

Expected: FAIL until docs and workflow text are updated.

**Step 3: Write minimal implementation**

Update README and workflow expectations to match the second-order contract.

**Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.test_repo_config -v`

Expected: PASS

**Step 5: Commit**

```bash
git add README.md .github/workflows/oracle-integrity.yml .github/workflows/oracle-regeneration.yml tests/test_repo_config.py
git commit -m "docs: document scalarized hvp oracle contract"
```

### Task 10: Final verification

**Files:**
- No new files

**Step 1: Run the full verification suite**

Run:

```bash
uv run python -m unittest discover -s tests -v
uv run python scripts/validate_schema.py
uv run python scripts/verify_cases.py
uv run python scripts/check_replay.py
uv run python scripts/check_regeneration.py
uv run python scripts/check_tolerances.py
python3 -m py_compile generators/*.py scripts/*.py tests/*.py validators/*.py
```

Expected: PASS

**Step 2: Commit any remaining fixes**

```bash
git status --short
git add -A
git commit -m "test: finalize scalarized hvp support"
```

If nothing remains, skip this commit.
