# PyTorch Full AD Case Extraction Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Expand `tensor-ad-oracles` to cover the full set of PyTorch linalg AD-relevant case families and tighten family tolerances using measured `torch` vs finite-difference residuals.

**Architecture:** Add an upstream inventory layer that extracts AD-relevant `OpInfo` metadata from the pinned PyTorch checkout, map that inventory to DB families and observables, then regenerate the JSON database from the expanded registry. Separately, add a tolerance-audit pass that measures live residuals across each family and proposes tighter tolerances when current thresholds are more than ten orders of magnitude looser than the observed residuals.

**Tech Stack:** Python 3.12, `uv`, pinned `torch==2.10.0`, `jsonschema`, `unittest`, GitHub Actions.

---

### Task 1: Add upstream inventory tests

**Files:**
- Create: `tests/test_upstream_inventory.py`
- Reference: `generators/pytorch_v1.py`
- Reference: `/sharehome/shinaoka/projects/tensor4all/pytorch/torch/testing/_internal/opinfo/definitions/linalg.py`

**Step 1: Write the failing test**

Add tests that require an inventory module to:
- list all `OpInfo` entries with `supports_forward_ad` or `supports_fwgrad_bwgrad`
- preserve `name`, `variant_test_name`, `sample_inputs_func`, `gradcheck_wrapper`, `output_process_fn_grad`, and `gradcheck_fast_mode`
- restrict to linalg ops

**Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_upstream_inventory -v`

Expected: FAIL because the upstream inventory module does not exist yet.

**Step 3: Write minimal implementation**

Create `generators/upstream_inventory.py` with:
- a record type for extracted upstream metadata
- a function that imports pinned PyTorch OpInfo data
- filters to AD-relevant linalg entries
- returns normalized inventory rows

**Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.test_upstream_inventory -v`

Expected: PASS

### Task 2: Add family mapping coverage tests

**Files:**
- Create: `tests/test_family_mapping.py`
- Modify: `generators/pytorch_v1.py`
- Modify: `generators/observables.py`
- Reference: `generators/upstream_inventory.py`

**Step 1: Write the failing test**

Add tests that:
- compare the upstream inventory against the local case registry
- fail if an AD-relevant upstream entry has no family mapping
- verify processed observables are mapped explicitly for spectral ops

**Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_family_mapping -v`

Expected: FAIL because the current registry only covers the small v1 subset.

**Step 3: Write minimal implementation**

Extend `generators/pytorch_v1.py` with:
- a mapping layer from upstream inventory rows to DB family specs
- explicit handling for variants such as `pinv`, `pinv.hermitian`, `eig`, `eigvals`, `svdvals`, `solve_triangular`, `lu_solve`, `inv`, `det`, `slogdet`, `lstsq.grad_oriented`, and any other AD-relevant linalg op exposed by upstream
- explicit classification for unsupported-but-known families

**Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.test_family_mapping -v`

Expected: PASS

### Task 3: Add observable and wrapper coverage tests

**Files:**
- Create: `tests/test_observable_coverage.py`
- Modify: `generators/observables.py`
- Modify: `generators/runtime.py`

**Step 1: Write the failing test**

Add tests that:
- verify every mapped upstream `output_process_fn_grad` resolves to a DB observable
- verify every mapped `gradcheck_wrapper` resolves to a DB-side wrapper
- fail on unknown upstream process/wrapper names

**Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_observable_coverage -v`

Expected: FAIL because the current observable and wrapper set is incomplete.

**Step 3: Write minimal implementation**

Implement:
- missing processed observables
- missing wrapper application logic
- explicit normalization/encoding paths for any new structured outputs

**Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.test_observable_coverage -v`

Expected: PASS

### Task 4: Add tolerance proposal tests

**Files:**
- Create: `tests/test_tolerance_audit.py`
- Create: `generators/tolerance_audit.py`
- Modify: `generators/pytorch_v1.py`

**Step 1: Write the failing test**

Add tests that:
- compute proposed `rtol`/`atol` from synthetic residual maxima
- round proposals up to the next power of ten
- apply a safety factor
- only mark a family as tighten-able if current tolerance is more than `1e10` looser than the observed residuals

**Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_tolerance_audit -v`

Expected: FAIL because the audit module does not exist yet.

**Step 3: Write minimal implementation**

Create `generators/tolerance_audit.py` with:
- residual accumulation helpers
- proposal logic for `rtol` and `atol`
- a family-level audit result record
- helpers that compare observed residuals against current case tolerances

**Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.test_tolerance_audit -v`

Expected: PASS

### Task 5: Add tolerance audit CLI tests

**Files:**
- Create: `tests/test_tolerance_audit_script.py`
- Create: `scripts/check_tolerances.py`
- Modify: `README.md`

**Step 1: Write the failing test**

Add tests that:
- run a tolerance-audit CLI over the current DB
- emit a nonzero exit code when a family is more than `1e10` looser than observed residuals
- print per-family proposed tolerance updates

**Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_tolerance_audit_script -v`

Expected: FAIL because the script does not exist yet.

**Step 3: Write minimal implementation**

Create `scripts/check_tolerances.py` that:
- replays all success cases
- groups residuals by `(op, family, dtype)`
- computes proposed `rtol` and `atol`
- exits nonzero if any family exceeds the `1e10` looseness threshold

Update `README.md` with the new CI contract.

**Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.test_tolerance_audit_script -v`

Expected: PASS

### Task 6: Expand generator coverage incrementally

**Files:**
- Modify: `generators/pytorch_v1.py`
- Modify: `generators/runtime.py`
- Modify: `generators/probes.py`
- Modify: `schema/case.schema.json`
- Test: `tests/test_materialize.py`
- Test: `tests/test_db_replay.py`

**Step 1: Write the failing test**

Add/extend tests for one missing upstream family at a time. Start with the easiest identity-output families, then structured families:
- `inv` / `inv_ex`
- `det` / `slogdet`
- `solve_triangular`
- `lu_factor` / `lu` / `lu_solve`
- `eig` / `eigvals`
- `svdvals`
- `pinv` / `pinv.hermitian`
- `lstsq.grad_oriented`

**Step 2: Run test to verify it fails**

Run targeted tests, for example:
- `uv run python -m unittest tests.test_materialize -v`
- `uv run python -m unittest tests.test_db_replay -v`

Expected: FAIL for each missing family before implementation.

**Step 3: Write minimal implementation**

For each family:
- add the upstream mapping
- implement output normalization and probe handling
- materialize success or error cases
- keep new schema usage minimal and reuse existing wire format

**Step 4: Run test to verify it passes**

Run the targeted test after each family lands.

Expected: PASS

### Task 7: Regenerate the database

**Files:**
- Modify: `cases/**/*.jsonl`
- Modify: `generators/pytorch_v1.py`

**Step 1: Run materialization for the full expanded registry**

Run: `uv run python -m generators.pytorch_v1 --materialize-all`

Expected: all mapped families are regenerated into canonical JSONL paths.

**Step 2: Validate generated output**

Run:
- `uv run python scripts/validate_schema.py`
- `uv run python scripts/verify_cases.py`
- `uv run python scripts/check_replay.py`

Expected: PASS

**Step 3: Run tolerance audit**

Run: `uv run python scripts/check_tolerances.py`

Expected: PASS, or actionable output listing families whose tolerances should be tightened.

### Task 8: Tighten tolerances and lock them in

**Files:**
- Modify: `generators/pytorch_v1.py`
- Modify: `cases/**/*.jsonl`
- Test: `tests/test_tolerance_audit.py`

**Step 1: Apply proposed tolerance reductions**

For every family flagged by the audit:
- update the generator’s canonical `comparison` values
- regenerate the affected JSONL files

**Step 2: Verify the tightened values**

Run:
- `uv run python scripts/check_tolerances.py`
- `uv run python scripts/check_regeneration.py`
- `uv run python scripts/check_replay.py`

Expected: PASS with no remaining families that are more than `1e10` looser than observed residuals.

### Task 9: Add CI enforcement

**Files:**
- Modify: `.github/workflows/oracle-integrity.yml`
- Modify: `.github/workflows/oracle-regeneration.yml`
- Modify: `tests/test_repo_config.py`

**Step 1: Write the failing test**

Add repo-config tests that require:
- the tolerance audit to run on every PR
- the tolerance audit to run on every push to `main`

**Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_repo_config -v`

Expected: FAIL before the workflow updates.

**Step 3: Write minimal implementation**

Update workflows so that:
- integrity CI still checks replay
- regeneration CI still checks semantic regeneration
- one of the workflows also runs `scripts/check_tolerances.py`

**Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.test_repo_config -v`

Expected: PASS

### Task 10: Final verification and documentation

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`

**Step 1: Update docs**

Document:
- what “all PyTorch tests” means in this repo
- how tolerance tightening works
- how to run the inventory, regeneration, replay, and tolerance-audit steps

**Step 2: Run full verification**

Run:
- `uv run python -m unittest discover -s tests -v`
- `uv run python scripts/validate_schema.py`
- `uv run python scripts/verify_cases.py`
- `uv run python scripts/check_replay.py`
- `uv run python scripts/check_regeneration.py`
- `uv run python scripts/check_tolerances.py`
- `python3 -m py_compile generators/*.py scripts/*.py tests/*.py validators/*.py`

Expected: all commands pass.

**Step 3: Commit**

```bash
git -C /sharehome/shinaoka/projects/tensor4all/tensor-ad-oracles add README.md AGENTS.md .github/workflows/oracle-integrity.yml .github/workflows/oracle-regeneration.yml generators scripts tests cases docs/plans
git -C /sharehome/shinaoka/projects/tensor4all/tensor-ad-oracles commit -m "feat: extract full PyTorch AD case inventory and tighten tolerances"
```
