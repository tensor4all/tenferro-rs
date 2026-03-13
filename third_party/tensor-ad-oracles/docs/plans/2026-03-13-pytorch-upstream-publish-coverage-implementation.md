# PyTorch Upstream Publish Coverage Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make PyTorch AD-relevant upstream cases the source of truth for the published DB surface, add optional case-intent comments, and expand generation so every currently publishable upstream family is materialized.

**Architecture:** Extend the existing upstream inventory and family-mapping pipeline so it also measures publish coverage by family and dtype, emits a generated Markdown coverage report, and widens generation where mapped publishable families are still missing. Keep the JSON wire format stable except for an optional `provenance.comment` field.

**Tech Stack:** Python 3.12, `uv`, pinned `torch==2.10.0`, `unittest`, `jsonschema`, Markdown generation, existing `generators/pytorch_v1.py` registry/materialization flow.

---

### Task 1: Add `provenance.comment` schema coverage

**Files:**
- Modify: `schema/case.schema.json`
- Modify: `tests/test_schema_contract.py`
- Modify: `tests/test_materialize.py`
- Modify: `generators/pytorch_v1.py`

**Step 1: Write the failing test**

Add tests that require:

- `provenance.comment` to be accepted as an optional string on both success and
  error cases
- `build_provenance()` to preserve the comment when provided
- materialized cases to carry the comment field through unchanged

**Step 2: Run test to verify it fails**

Run:

```bash
uv run python -m unittest tests.test_schema_contract tests.test_materialize -v
```

Expected: FAIL because `comment` is not yet part of the schema or provenance
builder.

**Step 3: Write minimal implementation**

Update:

- `schema/case.schema.json` to add optional `provenance.comment`
- `generators/pytorch_v1.py::build_provenance()` to accept `comment: str | None`
- the relevant materialize helpers to preserve the field

**Step 4: Run test to verify it passes**

Run:

```bash
uv run python -m unittest tests.test_schema_contract tests.test_materialize -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add schema/case.schema.json tests/test_schema_contract.py tests/test_materialize.py generators/pytorch_v1.py
git commit -m "feat: add optional provenance comments to case records"
```

### Task 2: Add publish-coverage inventory tests

**Files:**
- Modify: `tests/test_family_mapping.py`
- Modify: `tests/test_pytorch_v1.py`
- Modify: `generators/pytorch_v1.py`
- Modify: `generators/upstream_inventory.py`
- Modify: `generators/upstream_scalar_inventory.py`

**Step 1: Write the failing test**

Add tests that require:

- every upstream AD-relevant row to be classified as publishable or outside the
  current publish scope
- dtype coverage to be tracked per mapped family
- `linalg.svd` publish coverage to include complex success support when upstream
  supports it

Use one explicit assertion for the motivating gap:

```python
self.assertIn("complex128", publish_index[("svd", "u_abs")].supported_dtypes)
```

**Step 2: Run test to verify it fails**

Run:

```bash
uv run python -m unittest tests.test_family_mapping tests.test_pytorch_v1 -v
```

Expected: FAIL because the current mapping layer does not expose publish-coverage
metadata strongly enough to prove complex SVD success coverage.

**Step 3: Write minimal implementation**

Extend the inventory/mapping model so the generator can answer, for each mapped
family:

- which upstream row it came from
- which DB family it maps to
- which DB dtypes are publishable
- whether it is `success` or `error`

Keep the existing mapping API stable where possible, and add a new publish
coverage index if that produces less churn.

**Step 4: Run test to verify it passes**

Run:

```bash
uv run python -m unittest tests.test_family_mapping tests.test_pytorch_v1 -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_family_mapping.py tests/test_pytorch_v1.py generators/pytorch_v1.py generators/upstream_inventory.py generators/upstream_scalar_inventory.py
git commit -m "test: lock upstream publish coverage metadata"
```

### Task 3: Add generated publish-coverage report tests

**Files:**
- Create: `tests/test_publish_coverage_report.py`
- Create: `scripts/report_upstream_publish_coverage.py`
- Create: `docs/generated/pytorch-upstream-publish-coverage.md`
- Modify: `README.md`

**Step 1: Write the failing test**

Add tests that require a report generator to:

- emit a Markdown report under `docs/generated/`
- include sections for upstream inventory, mapped publishable families, and
  missing publishable families
- show dtype coverage for representative families such as `svd/u_abs`

**Step 2: Run test to verify it fails**

Run:

```bash
uv run python -m unittest tests.test_publish_coverage_report -v
```

Expected: FAIL because no report script or checked-in report exists yet.

**Step 3: Write minimal implementation**

Create `scripts/report_upstream_publish_coverage.py` that:

- loads the upstream inventory
- loads the mapped publishable surface
- inspects the checked-in `cases/` tree
- emits `docs/generated/pytorch-upstream-publish-coverage.md`

Document the report in `README.md`.

**Step 4: Run test to verify it passes**

Run:

```bash
uv run python -m unittest tests.test_publish_coverage_report -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_publish_coverage_report.py scripts/report_upstream_publish_coverage.py docs/generated/pytorch-upstream-publish-coverage.md README.md
git commit -m "docs: add upstream publish coverage report"
```

### Task 4: Add targeted generator regression tests for missing publishable coverage

**Files:**
- Modify: `tests/test_pytorch_v1.py`
- Modify: `tests/test_materialize.py`
- Modify: `generators/pytorch_v1.py`
- Modify: `generators/runtime.py`

**Step 1: Write the failing test**

Add targeted tests that require:

- `--materialize svd --family u_abs` to be able to emit complex success records
- representative complex success records to preserve the expected observable
  kind and provenance comment
- family materialization to skip only genuinely nonfinite/unusable samples, not
  whole publishable dtypes by accident

Prefer a temporary cases root and assert on the written JSONL rows.

**Step 2: Run test to verify it fails**

Run:

```bash
uv run python -m unittest tests.test_pytorch_v1 tests.test_materialize -v
```

Expected: FAIL because complex SVD success records are not currently published.

**Step 3: Write minimal implementation**

Update the materialization path in `generators/pytorch_v1.py` and any supporting
runtime helpers so mapped publishable dtypes are actually emitted. Keep the
existing observable model (`u_abs`, `s`, `vh_abs`, `uvh_product`) and do not add
new spectral observables in this task.

For representative new cases, attach a provenance comment such as:

- `from PyTorch OpInfo complex SVD success coverage`

**Step 4: Run test to verify it passes**

Run:

```bash
uv run python -m unittest tests.test_pytorch_v1 tests.test_materialize -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_pytorch_v1.py tests/test_materialize.py generators/pytorch_v1.py generators/runtime.py
git commit -m "feat: publish missing upstream complex success coverage"
```

### Task 5: Expand publishable upstream coverage incrementally

**Files:**
- Modify: `generators/pytorch_v1.py`
- Modify: `generators/runtime.py`
- Modify: `generators/observables.py`
- Modify: `tests/test_family_mapping.py`
- Modify: `tests/test_pytorch_v1.py`
- Modify: `tests/test_materialize.py`

**Step 1: Write the failing test**

For each missing publishable upstream family, add or extend one targeted test
that proves it is still absent. Prioritize:

- complex spectral success families first
- then remaining publishable linalg families that upstream already exposes and
  the current schema can encode

Keep one family per red-green cycle.

**Step 2: Run test to verify it fails**

Run only the targeted test for the current family, for example:

```bash
uv run python -m unittest tests.test_pytorch_v1.PytorchV1RegistryTests.test_main_materialize_<family> -v
```

Expected: FAIL before implementation.

**Step 3: Write minimal implementation**

For each family:

- add the upstream mapping if missing
- ensure the observable is supported
- materialize the family into canonical JSONL output
- add a provenance comment only where the case intent is worth preserving

Do not introduce placeholder families that are not yet publishable.

**Step 4: Run test to verify it passes**

Run the targeted test again and keep moving family-by-family.

Expected: PASS

**Step 5: Commit**

```bash
git add generators/pytorch_v1.py generators/runtime.py generators/observables.py tests/test_family_mapping.py tests/test_pytorch_v1.py tests/test_materialize.py
git commit -m "feat: widen upstream publish coverage for <family>"
```

### Task 6: Regenerate the database and coverage report

**Files:**
- Modify: `cases/**/*.jsonl`
- Modify: `docs/generated/pytorch-upstream-publish-coverage.md`

**Step 1: Regenerate the coverage report**

Run:

```bash
uv run python scripts/report_upstream_publish_coverage.py
```

Expected: the generated Markdown report is refreshed from the current mapping
and case tree.

**Step 2: Materialize the expanded surface**

Run:

```bash
uv run python -m generators.pytorch_v1 --materialize-all
```

Expected: canonical JSONL files are regenerated for all mapped publishable
families.

**Step 3: Run targeted regeneration checks**

Run:

```bash
uv run python scripts/validate_schema.py
uv run python scripts/verify_cases.py
uv run python scripts/check_replay.py
uv run python scripts/check_regeneration.py
```

Expected: PASS

**Step 4: Commit**

```bash
git add cases docs/generated/pytorch-upstream-publish-coverage.md
git commit -m "data: regenerate upstream publish coverage cases"
```

### Task 7: Final verification

**Files:**
- Verify only

**Step 1: Run focused unit and integration coverage**

Run:

```bash
uv run python -m unittest tests.test_schema_contract tests.test_materialize tests.test_family_mapping tests.test_pytorch_v1 tests.test_publish_coverage_report -v
```

Expected: PASS

**Step 2: Run repo integrity checks**

Run:

```bash
uv run python scripts/validate_schema.py
uv run python scripts/verify_cases.py
uv run python scripts/check_replay.py
uv run python scripts/check_regeneration.py
```

Expected: PASS

**Step 3: Commit final documentation touch-ups if needed**

```bash
git add README.md docs/generated/pytorch-upstream-publish-coverage.md
git commit -m "docs: finalize upstream publish coverage documentation"
```
