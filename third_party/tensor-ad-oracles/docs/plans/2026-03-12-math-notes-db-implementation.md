# Math Notes And Oracle DB Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a first-class mathematical note corpus to `tensor-ad-oracles`, connect published DB families to those notes through a central registry, and validate the linkage in tests and CI without changing the JSON case schema.

**Architecture:** Keep the human math notes under `docs/math/`, keep the published oracle DB under `cases/`, and connect them through `docs/math/registry.json`. Put reusable validation logic in `validators/math_registry.py`, expose it through `scripts/check_math_registry.py`, and keep the public DB contract unchanged. In the first migration pass, keep scalar-family note coverage in `docs/math/scalar_ops.md` with stable per-op anchors so the registry can cover the current scalar DB without forcing a 100+ file split immediately.

**Tech Stack:** Python 3.12, `uv`, pinned `torch==2.10.0`, `unittest`, Markdown docs, GitHub Actions.

---

### Task 1: Add the math-note scaffold and public repo contract

**Files:**
- Create: `docs/math/index.md`
- Create: `docs/math/registry.json`
- Modify: `README.md`
- Modify: `tests/test_repo_config.py`

**Step 1: Write the failing test**

Extend `tests/test_repo_config.py` with checks that require:

- `docs/math/index.md` to exist
- `docs/math/registry.json` to exist
- `README.md` to describe the repository as both math notes and oracle DB
- `README.md` to mention the central math-note registry

Example assertion block:

```python
def test_readme_documents_math_notes_and_registry(self) -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    self.assertIn("mathematical AD notes", readme)
    self.assertIn("oracle database", readme)
    self.assertIn("docs/math/registry.json", readme)
    self.assertTrue((REPO_ROOT / "docs" / "math" / "index.md").exists())
    self.assertTrue((REPO_ROOT / "docs" / "math" / "registry.json").exists())
```

**Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_repo_config -v`

Expected: FAIL because `docs/math/` does not exist and the README does not yet document the math-note artifact.

**Step 3: Write minimal implementation**

Add:

- `docs/math/index.md` with a short overview of:
  - note purpose
  - one-note-per-op intent for standalone ops
  - registry role
- `docs/math/registry.json` with:

```json
{
  "version": 1,
  "entries": []
}
```

Update `README.md` to add a short section that states:

- the repo now has two first-class artifacts
- math notes live under `docs/math/`
- DB-to-note linkage lives in `docs/math/registry.json`

**Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.test_repo_config -v`

Expected: PASS

**Step 5: Commit**

```bash
git add docs/math/index.md docs/math/registry.json README.md tests/test_repo_config.py
git commit -m "docs: add math note scaffold and repo contract"
```

### Task 2: Add registry validation helpers

**Files:**
- Create: `validators/math_registry.py`
- Create: `tests/test_math_registry.py`
- Modify: `validators/__init__.py`

**Step 1: Write the failing test**

Create `tests/test_math_registry.py` with unit tests that require the validator to detect:

- duplicate `(op, family)` entries
- missing `note_path`
- missing anchor in the target note
- missing registry coverage for a materialized `cases/<op>/<family>.jsonl`

Example test:

```python
def test_validate_registry_rejects_missing_anchor(self) -> None:
    note = root / "docs" / "math" / "svd.md"
    note.write_text("# SVD\n\n## DB Families\n", encoding="utf-8")
    registry.write_text(
        json.dumps(
            {
                "version": 1,
                "entries": [
                    {
                        "op": "svd",
                        "family": "u_abs",
                        "note_path": "docs/math/svd.md",
                        "anchor": "family-u-abs",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with self.assertRaisesRegex(ValueError, "missing anchor"):
        math_registry.validate_registry(root)
```

**Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_math_registry -v`

Expected: FAIL because no math-registry validator exists yet.

**Step 3: Write minimal implementation**

Implement `validators/math_registry.py` with focused helpers:

- `load_registry(root: Path) -> dict`
- `materialized_case_families(cases_root: Path) -> set[tuple[str, str]]`
- `extract_markdown_anchors(text: str) -> set[str]`
- `validate_registry(root: Path) -> None`

Validation rules:

- exactly one entry per `(op, family)`
- `note_path` resolves inside the repo
- `anchor` exists in the note
- every materialized case family has a matching registry entry

Export the module in `validators/__init__.py`.

**Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.test_math_registry -v`

Expected: PASS

**Step 5: Commit**

```bash
git add validators/math_registry.py validators/__init__.py tests/test_math_registry.py
git commit -m "feat: add math registry validation helpers"
```

### Task 3: Add the math-registry check script

**Files:**
- Create: `scripts/check_math_registry.py`
- Modify: `scripts/__init__.py`
- Modify: `tests/test_scripts.py`

**Step 1: Write the failing test**

Extend `tests/test_scripts.py` with script-level tests that require:

- `check_math_registry.main()` to return `0` for a valid temporary repo tree
- `check_math_registry.main()` to raise `SystemExit` with a useful error for an invalid registry

Example test:

```python
with patch.object(check_math_registry, "REPO_ROOT", root):
    self.assertEqual(check_math_registry.main(), 0)
```

**Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_scripts -v`

Expected: FAIL because the script is not present or not exported.

**Step 3: Write minimal implementation**

Create `scripts/check_math_registry.py` following the existing script style:

```python
from validators.math_registry import validate_registry

def main() -> int:
    validate_registry(REPO_ROOT)
    print("math_registry_ok=1")
    return 0
```

Export it from `scripts/__init__.py`.

**Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.test_scripts -v`

Expected: PASS

**Step 5: Commit**

```bash
git add scripts/check_math_registry.py scripts/__init__.py tests/test_scripts.py
git commit -m "feat: add math registry check script"
```

### Task 4: Migrate the DB-backed linalg notes with fixed family anchors

**Files:**
- Create: `docs/math/svd.md`
- Create: `docs/math/solve.md`
- Create: `docs/math/qr.md`
- Create: `docs/math/lu.md`
- Create: `docs/math/cholesky.md`
- Create: `docs/math/inv.md`
- Create: `docs/math/det.md`
- Create: `docs/math/eig.md`
- Create: `docs/math/eigen.md`
- Create: `docs/math/pinv.md`
- Create: `docs/math/lstsq.md`
- Create: `docs/math/norm.md`
- Modify: `docs/math/index.md`
- Modify: `tests/test_math_registry.py`

**Step 1: Write the failing test**

Add tests that require:

- each listed note file to exist
- each listed note to contain a `## DB Families` section
- `docs/math/svd.md` to expose:
  - `family-u-abs`
  - `family-s`
  - `family-vh-abs`
  - `family-uvh-product`
- `docs/math/eig.md` and `docs/math/eigen.md` to distinguish general and Hermitian eigen rules

Example check:

```python
svd_text = (REPO_ROOT / "docs" / "math" / "svd.md").read_text(encoding="utf-8")
self.assertIn("## DB Families", svd_text)
self.assertIn("{#family-u-abs}", svd_text)
self.assertIn("{#family-s}", svd_text)
```

**Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_math_registry -v`

Expected: FAIL because the note corpus has not been migrated yet.

**Step 3: Write minimal implementation**

Port and adapt the existing `../tenferro-rs/docs/AD/*.md` content into the new
files. Keep the raw math first, then add a `## DB Families` section with fixed
anchors that explain the DB observable mapping.

For `docs/math/svd.md`, add anchors like:

```md
## DB Families

### `u_abs` {#family-u-abs}

The DB publishes `U.abs()` rather than raw `U` to remove sign/phase gauge ambiguity.

### `s` {#family-s}

The DB publishes the singular values directly.
```

Update `docs/math/index.md` to link every migrated note.

**Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.test_math_registry -v`

Expected: PASS

**Step 5: Commit**

```bash
git add docs/math/index.md docs/math/svd.md docs/math/solve.md docs/math/qr.md docs/math/lu.md docs/math/cholesky.md docs/math/inv.md docs/math/det.md docs/math/eig.md docs/math/eigen.md docs/math/pinv.md docs/math/lstsq.md docs/math/norm.md tests/test_math_registry.py
git commit -m "docs: migrate linalg math notes"
```

### Task 5: Add the remaining known-rule notes outside the current DB core

**Files:**
- Create: `docs/math/matrix_exp.md`
- Create: `docs/math/scalar_ops.md`
- Create: `docs/math/dyadtensor_reverse.md`
- Modify: `docs/math/index.md`
- Modify: `tests/test_math_registry.py`

**Step 1: Write the failing test**

Extend `tests/test_math_registry.py` with checks that require:

- the remaining note files to exist
- `docs/math/scalar_ops.md` to expose per-op anchors for representative scalar families:
  - `op-abs`
  - `op-add`
  - `op-sum`
  - `op-var`
- `docs/math/matrix_exp.md` to state that the note exists even though there is no current DB family

Example assertion:

```python
scalar_text = (REPO_ROOT / "docs" / "math" / "scalar_ops.md").read_text(encoding="utf-8")
self.assertIn("{#op-abs}", scalar_text)
self.assertIn("{#op-add}", scalar_text)
self.assertIn("not yet materialized", (REPO_ROOT / "docs" / "math" / "matrix_exp.md").read_text(encoding="utf-8"))
```

**Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_math_registry -v`

Expected: FAIL because the remaining notes are still missing.

**Step 3: Write minimal implementation**

Migrate the remaining note files from `../tenferro-rs/docs/AD/` and adapt them
to the new structure.

For `docs/math/scalar_ops.md`:

- preserve the shared basis formulas
- add stable per-op anchors for the currently published scalar families
- add a short note that this file is the initial scalar landing zone for many
  DB families

For `docs/math/matrix_exp.md`:

- keep the raw rule
- add a `DB status` note explaining that the operation is documented but not yet
  materialized in `cases/`

Update `docs/math/index.md` to link the new files.

**Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.test_math_registry -v`

Expected: PASS

**Step 5: Commit**

```bash
git add docs/math/index.md docs/math/matrix_exp.md docs/math/scalar_ops.md docs/math/dyadtensor_reverse.md tests/test_math_registry.py
git commit -m "docs: add remaining math note corpus"
```

### Task 6: Populate the registry for all published DB families

**Files:**
- Modify: `docs/math/registry.json`
- Modify: `tests/test_math_registry.py`

**Step 1: Write the failing test**

Add a repository-level coverage test that scans `cases/` and requires every
materialized `(op, family)` pair to appear in `docs/math/registry.json`.

Also add representative assertions such as:

- `svd/u_abs -> docs/math/svd.md#family-u-abs`
- `eig/values_vectors_abs -> docs/math/eig.md#family-values-vectors-abs`
- `solve/identity -> docs/math/solve.md#family-identity`
- `abs/identity -> docs/math/scalar_ops.md#op-abs`
- `sum/identity -> docs/math/scalar_ops.md#op-sum`

Example test:

```python
entries = math_registry.load_registry(REPO_ROOT)["entries"]
index = {(row["op"], row["family"]): row for row in entries}
self.assertEqual(index[("svd", "u_abs")]["note_path"], "docs/math/svd.md")
self.assertEqual(index[("svd", "u_abs")]["anchor"], "family-u-abs")
math_registry.validate_registry(REPO_ROOT)
```

**Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_math_registry -v`

Expected: FAIL because the registry is still empty or incomplete.

**Step 3: Write minimal implementation**

Populate `docs/math/registry.json` with entries for every current
`cases/<op>/<family>.jsonl`.

Use these mapping rules:

- linalg families point to their dedicated note files
- scalar identity families point to `docs/math/scalar_ops.md` with stable
  `op-<name>` anchors
- error families point to the same op note with dedicated family anchors

Keep the file sorted for review stability:

- sort by `op`
- then by `family`

**Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.test_math_registry -v`

Expected: PASS

**Step 5: Commit**

```bash
git add docs/math/registry.json tests/test_math_registry.py
git commit -m "docs: map published db families to math notes"
```

### Task 7: Wire the math-registry check into README and CI

**Files:**
- Modify: `README.md`
- Modify: `.github/workflows/oracle-integrity.yml`
- Modify: `tests/test_repo_config.py`

**Step 1: Write the failing test**

Extend `tests/test_repo_config.py` with checks that require:

- `README.md` to include `uv run python scripts/check_math_registry.py`
- `.github/workflows/oracle-integrity.yml` to run the same command

Example assertions:

```python
self.assertIn("uv run python scripts/check_math_registry.py", readme)
self.assertIn("uv run python scripts/check_math_registry.py", workflow)
```

**Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_repo_config -v`

Expected: FAIL because the new check is not yet documented or wired into CI.

**Step 3: Write minimal implementation**

Update `README.md` to document the math-note validation command alongside the
existing schema and replay checks.

Update `.github/workflows/oracle-integrity.yml` to run:

```bash
uv run python scripts/check_math_registry.py
```

Place it before schema/replay checks so broken note linkage fails early.

**Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.test_repo_config -v`

Expected: PASS

**Step 5: Commit**

```bash
git add README.md .github/workflows/oracle-integrity.yml tests/test_repo_config.py
git commit -m "ci: validate math note registry"
```

### Task 8: Run full repository verification for the new note contract

**Files:**
- Verify: `docs/math/index.md`
- Verify: `docs/math/registry.json`
- Verify: `validators/math_registry.py`
- Verify: `scripts/check_math_registry.py`
- Verify: `README.md`
- Verify: `.github/workflows/oracle-integrity.yml`
- Verify: `tests/test_math_registry.py`
- Verify: `tests/test_scripts.py`
- Verify: `tests/test_repo_config.py`

**Step 1: Run focused unit and script coverage**

Run:

```bash
uv run python -m unittest tests.test_math_registry tests.test_scripts tests.test_repo_config -v
```

Expected: PASS

**Step 2: Run the new standalone math-registry check**

Run:

```bash
uv run python scripts/check_math_registry.py
```

Expected: PASS and print `math_registry_ok=1`

**Step 3: Run the existing repository contract checks**

Run:

```bash
uv run python scripts/validate_schema.py
uv run python scripts/verify_cases.py
uv run python scripts/check_replay.py
uv run python scripts/check_regeneration.py
```

Expected: PASS

**Step 4: Commit the final integrated state**

```bash
git add docs/math README.md .github/workflows/oracle-integrity.yml scripts validators tests
git commit -m "feat: add math notes and db registry linkage"
```
